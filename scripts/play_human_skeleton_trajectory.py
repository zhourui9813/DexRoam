#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Play every frame of a recorded whole-body trajectory in Open3D.

This is the dynamic counterpart of
``visualize_human_skeleton_distance_frames_open3d.py``.  It uses the same
colors, point/line-to-mesh size conversion, head frustum, ground grid, selected
trajectory, white background, and default camera, but plays the selected
frame range instead of sampling several poses into a static figure.

Example (PowerShell):
    python scripts/play_human_skeleton_trajectory_open3d.py `
        E:/path/to/episode0004.h5 --loop

Controls:
    Space       play / pause
    N / Right   next frame
    P / Left    previous frame
    R           restart at the selected first frame
    T           show / hide the selected trajectory
    Y           switch between chest and head trajectory
    F           cycle wire / image / both frustum modes
    G           show / hide the ground grid
    L           enable / disable looping
    Q / Esc     quit
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import h5py
import numpy as np
import open3d as o3d
from tqdm import tqdm


# ---------------- Reference display parameters ----------------
# Copied from the referenced static Open3D visualization and its two helper
# modules.  Keep visual tuning in this block so the dynamic and static views
# remain easy to compare.

BACKGROUND_COLOR = np.array([1.0, 1.0, 1.0], dtype=np.float64)
WINDOW_BACKGROUND_COLOR = np.array([1.0, 1.0, 1.0], dtype=np.float64)

BODY_POINT_COLOR = "#524646"
BODY_LINE_COLOR = "#A8A492"
LEFT_HAND_COLOR = "#F67280"
RIGHT_HAND_COLOR = "#F67280"
HEAD_FRUSTUM_COLOR = "#A8CD89"
CHEST_TRAJECTORY_COLOR = "#FF9137"
GROUND_GRID_COLOR = "#9BB4C0"

# The static distance-frame overlay uses these lighter multi-frame sizes.
BODY_POINT_SIZE = 3
HAND_POINT_SIZE = 1
HEAD_POINT_SIZE = 12
BODY_LINE_WIDTH = 2.0
HAND_LINE_WIDTH = 0.9
FRUSTUM_LINE_WIDTH = 1.0

CHEST_TRAJECTORY_LINE_WIDTH = 1.2
CHEST_SAMPLE_SIZE = 5

GROUND_GRID_SIZE = 0.1
GROUND_GRID_PADDING = 10.0
GROUND_GRID_ALPHA = 0.5
GROUND_GRID_FALLBACK_Z = 0.0
GROUND_GRID_LINESET_PIXEL_WIDTH = 1.0

POINT_AREA_TO_RADIUS = 0.0080
LINEWIDTH_TO_RADIUS = 0.0040
MIN_MESH_RADIUS = 0.0012

HEAD_FRUSTUM_DEPTH = 0.18
HEAD_FRUSTUM_VERTICAL_FOV_DEG = 45.0
HEAD_FRUSTUM_ASPECT = 16.0 / 9.0
DEFAULT_VIEW_ELEV = 18.0
DEFAULT_VIEW_AZIM = -62.0

DEFAULT_FRAME_ALPHA = 0.95
CHEST_TRAJECTORY_ALPHA = 0.55
CHEST_MARKER_ALPHA = 0.95
DEFAULT_FRUSTUM_IMAGE_WIDTH = 96
DEFAULT_FRUSTUM_IMAGE_ALPHA = 0.85
FRUSTUM_MODES = ("wire", "image", "both")
TRAJECTORY_TARGETS = ("chest", "head")
IMAGE_EYES = ("auto", "left", "right", "full")
IMAGE_COLOR_ORDERS = ("auto", "rgb", "bgr")
IMAGE_DATASET_CANDIDATES = (
    "zed/frame_jpeg_bytes",
    "zed/frame_rgb_u8",
    "images_dict/head/rgb",
    "images_dict/head_rgbd/color",
    "images_dict/torso_rgbd/color",
    "images_dict/head_stereo/rgb",
    "images_dict/head_stereo/color",
    "images_dict/stereo_left/rgb",
    "images_dict/stereo_right/rgb",
)
DEFAULT_PLAYBACK_FPS = 30.0
PLAYBACK_MODES = ("fps", "timestamp")
LATE_FRAME_POLICIES = ("slow", "skip")
TIMESTAMP_SOURCES = ("auto", "zed", "image", "body", "head")
TIMESTAMP_DATASETS = {
    "zed": "zed/capture_monotonic_ns",
    "image": "zed/image_ts_ns",
    "body": "body/sample_monotonic_ns",
    "head": "head/sample_monotonic_ns",
}
DEFAULT_WINDOW_WIDTH = 1080
DEFAULT_WINDOW_HEIGHT = 1080
MESH_RESOLUTION = 12
TRAJECTORY_MESH_RESOLUTION = 8
# ----------------------------------------------------------------


BODY_BONES_BY_NAME = (
    ("hips", "spine-lower"),
    ("spine-lower", "spine-middle"),
    ("spine-middle", "spine-upper"),
    ("spine-upper", "chest"),
    ("chest", "neck"),
    ("neck", "head"),
    ("chest", "left-shoulder"),
    ("left-shoulder", "left-scapula"),
    ("left-scapula", "left-arm-upper"),
    ("left-arm-upper", "left-arm-lower"),
    ("left-arm-lower", "left-hand-wrist-twist"),
    ("left-hand-wrist-twist", "left-hand-palm"),
    ("chest", "right-shoulder"),
    ("right-shoulder", "right-scapula"),
    ("right-scapula", "right-arm-upper"),
    ("right-arm-upper", "right-arm-lower"),
    ("right-arm-lower", "right-hand-wrist-twist"),
    ("right-hand-wrist-twist", "right-hand-palm"),
    ("hips", "left-upper-leg"),
    ("left-upper-leg", "left-lower-leg"),
    ("left-lower-leg", "left-foot-ankle-twist"),
    ("left-foot-ankle-twist", "left-foot-ankle"),
    ("left-foot-ankle", "left-foot-subtalar"),
    ("left-foot-subtalar", "left-foot-transverse"),
    ("left-foot-transverse", "left-foot-ball"),
    ("hips", "right-upper-leg"),
    ("right-upper-leg", "right-lower-leg"),
    ("right-lower-leg", "right-foot-ankle-twist"),
    ("right-foot-ankle-twist", "right-foot-ankle"),
    ("right-foot-ankle", "right-foot-subtalar"),
    ("right-foot-subtalar", "right-foot-transverse"),
    ("right-foot-transverse", "right-foot-ball"),
)

HAND_BONES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8), (8, 9),
    (0, 10), (10, 11), (11, 12), (12, 13), (13, 14),
    (0, 15), (15, 16), (16, 17), (17, 18), (18, 19),
    (0, 20), (20, 21), (21, 22), (22, 23), (23, 24),
)

FRUSTUM_EDGES = (
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 1),
)

FOOT_JOINT_NAMES = (
    "left-foot-ankle-twist",
    "left-foot-ankle",
    "left-foot-subtalar",
    "left-foot-transverse",
    "left-foot-ball",
    "right-foot-ankle-twist",
    "right-foot-ankle",
    "right-foot-subtalar",
    "right-foot-transverse",
    "right-foot-ball",
)


@dataclass(frozen=True)
class MotionSequence:
    path: Path
    playback_fps: float
    joint_names: list[str]
    body: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray
    head: np.ndarray
    timestamps_ns: dict[str, np.ndarray]

    @property
    def frame_count(self) -> int:
        return int(self.body.shape[0])


@dataclass(frozen=True)
class MeshPart:
    vertices: np.ndarray
    triangles: np.ndarray
    colors: np.ndarray


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value!r}.")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"Expected a positive number, got {value!r}.")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError(f"Expected a non-negative number, got {value!r}.")
    return parsed


def unit_interval(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError(f"Expected a value in [0, 1], got {value!r}.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Play all frames of a whole-body HDF5 trajectory with the same "
            "Open3D visual parameters as the static distance-frame figure."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("h5_path", type=Path, help="Path to an episode .h5/.hdf5 file.")
    parser.add_argument("--start-frame", type=int, default=0, help="First frame to play.")
    parser.add_argument(
        "--end-frame",
        type=int,
        default=-1,
        help="Last frame to play, inclusive; -1 selects the final frame.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help=(
            "Fixed-mode playback FPS; 0 uses /meta.attrs['fps'] and falls back to 30. "
            "Timestamp mode uses recorded intervals instead."
        ),
    )
    parser.add_argument(
        "--playback-mode",
        choices=PLAYBACK_MODES,
        default="fps",
        help="Use a fixed FPS clock or reproduce the recorded per-frame timestamp intervals.",
    )
    parser.add_argument(
        "--playback-speed",
        type=positive_float,
        default=1.0,
        help="Playback speed multiplier applied to either timing mode.",
    )
    parser.add_argument(
        "--timestamp-source",
        choices=TIMESTAMP_SOURCES,
        default="auto",
        help=(
            "Timestamp stream for timestamp mode. Auto prefers ZED capture time, then "
            "body/head sample time and camera image time."
        ),
    )
    parser.add_argument(
        "--late-frame-policy",
        choices=LATE_FRAME_POLICIES,
        default="slow",
        help=(
            "When rendering falls behind, slow shows every frame and lets playback run late; "
            "skip keeps wall-clock synchronization by dropping intermediate display frames."
        ),
    )
    parser.add_argument(
        "--fps-report-interval",
        type=nonnegative_float,
        default=2.0,
        help="Seconds between measured display-FPS reports; 0 disables reporting.",
    )
    parser.add_argument("--loop", action="store_true", help="Loop when playback reaches the final selected frame.")
    parser.add_argument("--paused", action="store_true", help="Open on the first selected frame without autoplaying.")
    parser.add_argument(
        "--frame-alpha",
        type=unit_interval,
        default=DEFAULT_FRAME_ALPHA,
        help="Opacity of the moving skeleton after compositing onto white.",
    )
    parser.add_argument(
        "--trajectory-target",
        choices=TRAJECTORY_TARGETS,
        default="chest",
        help="Tracked point whose complete trajectory is visualized.",
    )
    parser.add_argument(
        "--no-trajectory",
        "--no-chest-trajectory",
        dest="no_trajectory",
        action="store_true",
        help="Hide the selected orange trajectory and its moving marker.",
    )
    parser.add_argument("--no-ground-grid", action="store_true", help="Hide the ground grid.")
    parser.add_argument(
        "--frustum-mode",
        choices=FRUSTUM_MODES,
        default="wire",
        help="Render the head frustum as a wireframe, camera image plane, or both.",
    )
    parser.add_argument(
        "--image-dataset",
        default=None,
        help="HDF5 image dataset used by image/both mode; auto-detected when omitted.",
    )
    parser.add_argument(
        "--image-eye",
        choices=IMAGE_EYES,
        default="auto",
        help="Select a stereo eye, the full image, or automatic selection.",
    )
    parser.add_argument(
        "--image-color-order",
        choices=IMAGE_COLOR_ORDERS,
        default="auto",
        help="Interpret image channels automatically or force RGB/BGR input.",
    )
    parser.add_argument(
        "--frustum-image-width",
        type=positive_int,
        default=DEFAULT_FRUSTUM_IMAGE_WIDTH,
        help="Width of the image mesh in vertices/pixels after resizing.",
    )
    parser.add_argument(
        "--frustum-image-alpha",
        type=unit_interval,
        default=DEFAULT_FRUSTUM_IMAGE_ALPHA,
        help="Opacity of the camera image on the frustum plane.",
    )
    parser.add_argument(
        "--preload-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decode and resize selected frustum images before playback; use --no-preload-images for on-demand decoding.",
    )
    parser.add_argument(
        "--preload-workers",
        type=positive_int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Worker threads used to decode/resize images during preload.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional directory for persistent preprocessed .npy image caches.",
    )
    parser.add_argument("--frustum-length", type=positive_float, default=HEAD_FRUSTUM_DEPTH)
    parser.add_argument("--frustum-fov-deg", type=positive_float, default=HEAD_FRUSTUM_VERTICAL_FOV_DEG)
    parser.add_argument("--frustum-aspect", type=positive_float, default=HEAD_FRUSTUM_ASPECT)
    parser.add_argument("--elev", type=float, default=DEFAULT_VIEW_ELEV, help="Initial camera elevation in degrees.")
    parser.add_argument("--azim", type=float, default=DEFAULT_VIEW_AZIM, help="Initial camera azimuth in degrees.")
    parser.add_argument("--window-width", type=positive_int, default=DEFAULT_WINDOW_WIDTH)
    parser.add_argument("--window-height", type=positive_int, default=DEFAULT_WINDOW_HEIGHT)
    return parser.parse_args()


def decode_names(values: Iterable[object]) -> list[str]:
    names: list[str] = []
    for value in values:
        names.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
    return names


def load_sequence(path: Path) -> MotionSequence:
    with h5py.File(path, "r") as h5_file:
        required = (
            "body/joint_names",
            "body/matrices_f32",
            "body/left_hand_key_points_world_f32",
            "body/right_hand_key_points_world_f32",
            "head/matrix_f32",
        )
        missing = [name for name in required if name not in h5_file]
        if missing:
            raise ValueError(f"Missing required HDF5 datasets: {missing}")

        joint_names = decode_names(h5_file["body/joint_names"][...])
        body = h5_file["body/matrices_f32"][:, :, :3, 3].astype(np.float64)
        left_hand = h5_file["body/left_hand_key_points_world_f32"][:].astype(np.float64)
        right_hand = h5_file["body/right_hand_key_points_world_f32"][:].astype(np.float64)
        head = h5_file["head/matrix_f32"][:].astype(np.float64)

        lengths = [len(body), len(left_hand), len(right_hand), len(head)]
        frame_count = min(lengths)
        if frame_count <= 0:
            raise ValueError(f"No synchronized skeleton frames found in {path}.")
        if len(joint_names) != body.shape[1]:
            raise ValueError(
                f"body/joint_names has {len(joint_names)} entries but body/matrices_f32 "
                f"contains {body.shape[1]} joints."
            )

        playback_fps = DEFAULT_PLAYBACK_FPS
        if "meta" in h5_file:
            playback_fps = float(h5_file["meta"].attrs.get("fps", DEFAULT_PLAYBACK_FPS))
        if not np.isfinite(playback_fps) or playback_fps <= 0.0:
            playback_fps = DEFAULT_PLAYBACK_FPS
        timestamps_ns = {
            source: np.asarray(h5_file[dataset_name], dtype=np.int64)
            for source, dataset_name in TIMESTAMP_DATASETS.items()
            if dataset_name in h5_file
        }

    return MotionSequence(
        path=path,
        playback_fps=playback_fps,
        joint_names=joint_names,
        body=body[:frame_count],
        left_hand=left_hand[:frame_count],
        right_hand=right_hand[:frame_count],
        head=head[:frame_count],
        timestamps_ns=timestamps_ns,
    )


def normalize_frame_range(frame_count: int, start_frame: int, end_frame: int) -> tuple[int, int]:
    if start_frame < 0:
        start_frame += frame_count
    if end_frame < 0:
        end_frame += frame_count
    if start_frame < 0 or start_frame >= frame_count:
        raise IndexError(f"start-frame {start_frame} is outside [0, {frame_count - 1}].")
    if end_frame < 0 or end_frame >= frame_count:
        raise IndexError(f"end-frame {end_frame} is outside [0, {frame_count - 1}].")
    if start_frame > end_frame:
        raise ValueError(f"start-frame {start_frame} must not be after end-frame {end_frame}.")
    return start_frame, end_frame


def resolve_timestamp_stream(
    sequence: MotionSequence,
    source: str,
    start_frame: int,
    end_frame: int,
) -> tuple[str, np.ndarray]:
    candidates = ("zed", "body", "head", "image") if source == "auto" else (source,)
    rejection_reasons: list[str] = []
    for candidate in candidates:
        values = sequence.timestamps_ns.get(candidate)
        if values is None:
            rejection_reasons.append(f"{candidate}: missing {TIMESTAMP_DATASETS[candidate]}")
            continue
        if values.ndim != 1 or len(values) <= end_frame:
            rejection_reasons.append(
                f"{candidate}: only {len(values)} timestamp entries for end-frame {end_frame}"
            )
            continue
        selected = values[start_frame:end_frame + 1]
        if np.any(selected < 0):
            rejection_reasons.append(f"{candidate}: contains negative timestamps")
            continue
        differences = np.diff(selected)
        if np.any(differences < 0):
            rejection_reasons.append(f"{candidate}: timestamps are not monotonic")
            continue
        if len(selected) > 1 and not np.any(differences > 0):
            rejection_reasons.append(f"{candidate}: timestamps do not advance")
            continue
        return candidate, values

    details = "; ".join(rejection_reasons) or "no candidate timestamp streams"
    raise ValueError(f"No valid timestamp stream is available ({details}).")


def encoded_image_bytes(frame: object) -> Optional[bytes]:
    if isinstance(frame, bytes):
        return frame
    if isinstance(frame, str):
        return frame.encode("latin1")
    if isinstance(frame, np.void):
        return bytes(frame)
    array = np.asarray(frame)
    if array.ndim == 1 and array.dtype.kind in ("u", "i"):
        return array.astype(np.uint8, copy=False).tobytes()
    if array.shape == ():
        return encoded_image_bytes(array.item())
    return None


def should_swap_red_blue(dataset_name: str, color_order: str) -> bool:
    if color_order == "rgb":
        return False
    if color_order == "bgr":
        return True
    lowered = dataset_name.lower()
    return (
        "frame_jpeg_bytes" in lowered
        or lowered.endswith("_jpeg_bytes")
        or (lowered.startswith("images_dict/") and lowered.endswith("/rgb"))
    )


class FrustumImageLoader:
    """Decode/resize HDF5 camera frames for preload or on-demand playback."""

    def __init__(
        self,
        h5_path: Path,
        dataset_name: Optional[str],
        eye: str,
        color_order: str,
        target_width: int,
        required_end_frame: int,
    ) -> None:
        self.h5_path = h5_path
        self.h5_file = h5py.File(h5_path, "r")
        try:
            self.dataset_name, self.dataset = self._resolve_dataset(dataset_name)
            if required_end_frame >= len(self.dataset):
                raise ValueError(
                    f"Image dataset '{self.dataset_name}' has {len(self.dataset)} frames, "
                    f"but selected end-frame is {required_end_frame}."
                )
        except Exception:
            self.h5_file.close()
            raise

        self.eye = eye
        self.color_order = color_order
        self.target_width = int(target_width)
        self.used_eye: Optional[str] = None
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.cache_size = 16
        self.preloaded_frames: Optional[np.ndarray] = None
        self.preload_start_frame = 0
        print(f"[INFO] frustum image dataset={self.dataset_name}")

    def _resolve_dataset(self, requested: Optional[str]) -> tuple[str, h5py.Dataset]:
        if requested is not None:
            name = requested.strip("/")
            if name not in self.h5_file or not isinstance(self.h5_file[name], h5py.Dataset):
                raise KeyError(f"HDF5 image dataset not found: {requested!r}")
            return name, self.h5_file[name]

        for name in IMAGE_DATASET_CANDIDATES:
            if name in self.h5_file and isinstance(self.h5_file[name], h5py.Dataset):
                return name, self.h5_file[name]
        raise KeyError(
            "Could not auto-detect a camera image dataset. Pass --image-dataset explicitly. "
            f"Tried: {IMAGE_DATASET_CANDIDATES}"
        )

    def close(self) -> None:
        self.h5_file.close()

    def _decode_rgb_value(self, frame: object, frame_index: int) -> np.ndarray:
        encoded = encoded_image_bytes(frame)
        if encoded is not None:
            buffer = np.frombuffer(encoded, dtype=np.uint8)
            decoded_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if decoded_bgr is None:
                raise RuntimeError(
                    f"Failed to decode frame {frame_index} from '{self.dataset_name}'."
                )
            rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
        else:
            rgb = np.asarray(frame)
            if rgb.ndim == 2:
                rgb = np.repeat(rgb[:, :, None], 3, axis=2)
            elif rgb.ndim == 3 and rgb.shape[0] in (1, 3, 4) and rgb.shape[-1] not in (1, 3, 4):
                rgb = np.moveaxis(rgb, 0, -1)
            if rgb.ndim != 3 or rgb.shape[2] not in (1, 3, 4):
                raise ValueError(
                    f"Unsupported image frame shape {rgb.shape} in '{self.dataset_name}'."
                )
            if rgb.shape[2] == 1:
                rgb = np.repeat(rgb, 3, axis=2)
            elif rgb.shape[2] == 4:
                rgb = rgb[:, :, :3]
            rgb = rgb.astype(np.uint8, copy=False)

        if should_swap_red_blue(self.dataset_name, self.color_order):
            rgb = rgb[:, :, ::-1]
        return np.ascontiguousarray(rgb)

    def _decode_rgb(self, frame_index: int) -> np.ndarray:
        return self._decode_rgb_value(self.dataset[frame_index], frame_index)

    def _resolve_eye(self, image: np.ndarray) -> str:
        if self.eye != "auto":
            return self.eye
        height, width = image.shape[:2]
        lowered = self.dataset_name.lower()
        looks_stereo = (
            "head_stereo" in lowered
            or "frame_jpeg_bytes" in lowered
            or lowered.endswith("/stereo/rgb")
            or width / max(height, 1) > 2.2
        )
        return "left" if looks_stereo else "full"

    def _crop_eye(self, image: np.ndarray) -> np.ndarray:
        resolved_eye = self._resolve_eye(image)
        self.used_eye = self.used_eye or resolved_eye
        if resolved_eye == "full":
            return image
        half_width = image.shape[1] // 2
        if half_width <= 0:
            raise ValueError(f"Cannot crop a stereo eye from image shape {image.shape}.")
        if resolved_eye == "left":
            return image[:, :half_width]
        return image[:, image.shape[1] - half_width:]

    def _prepare_value(self, frame: object, frame_index: int) -> np.ndarray:
        image = self._crop_eye(self._decode_rgb_value(frame, frame_index))
        height, width = image.shape[:2]
        target_height = max(2, int(round(height * self.target_width / max(width, 1))))
        interpolation = cv2.INTER_AREA if self.target_width < width else cv2.INTER_LINEAR
        image = cv2.resize(image, (self.target_width, target_height), interpolation=interpolation)
        return np.ascontiguousarray(image, dtype=np.uint8)

    def _cache_path(self, cache_dir: Path, start_frame: int, end_frame: int) -> Path:
        stat = self.h5_path.stat()
        signature = "|".join(
            (
                str(self.h5_path),
                str(stat.st_size),
                str(stat.st_mtime_ns),
                self.dataset_name,
                self.eye,
                self.color_order,
                str(self.target_width),
                str(start_frame),
                str(end_frame),
            )
        )
        digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
        return cache_dir / f"{self.h5_path.stem}_frustum_{digest}.npy"

    def preload(
        self,
        start_frame: int,
        end_frame: int,
        workers: int,
        cache_dir: Optional[Path],
    ) -> None:
        frame_count = end_frame - start_frame + 1
        cache_path: Optional[Path] = None
        if cache_dir is not None:
            cache_dir = cache_dir.expanduser().resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self._cache_path(cache_dir, start_frame, end_frame)
            if cache_path.is_file():
                try:
                    cached = np.load(cache_path, mmap_mode="r", allow_pickle=False)
                    if (
                        cached.dtype != np.uint8
                        or cached.ndim != 4
                        or len(cached) != frame_count
                        or cached.shape[-1] != 3
                    ):
                        raise ValueError(
                            f"expected uint8 [{frame_count}, H, W, C], got {cached.dtype} {cached.shape}"
                        )
                    # A cache file only accelerates startup.  Copy it eagerly so
                    # playback reads ordinary RAM rather than faulting mmap pages.
                    self.preloaded_frames = np.array(cached, dtype=np.uint8, copy=True)
                    del cached
                    self.preload_start_frame = start_frame
                    self.used_eye = self.eye if self.eye != "auto" else "auto (cached)"
                    cache_mib = self.preloaded_frames.nbytes / (1024.0 * 1024.0)
                    print(
                        f"[INFO] loaded frustum image cache into RAM: {cache_path} "
                        f"({cache_mib:.1f} MiB)"
                    )
                    return
                except Exception as exc:
                    print(f"[WARN] ignoring invalid frustum image cache {cache_path}: {exc}")

        first_value = np.asarray(self.dataset[start_frame]).copy()
        first = self._prepare_value(first_value, start_frame)
        preloaded = np.empty((frame_count,) + first.shape, dtype=np.uint8)
        preloaded[0] = first

        progress = tqdm(
            total=frame_count,
            initial=1,
            desc="Preloading frustum images",
            unit="frame",
            dynamic_ncols=True,
            ascii=True,
        )
        pending = deque()
        max_pending = max(1, int(workers) * 2)

        def store_oldest() -> None:
            relative_index, future = pending.popleft()
            image = future.result()
            if image.shape != first.shape:
                raise ValueError(
                    f"Frustum image shape changed at frame {start_frame + relative_index}: "
                    f"expected {first.shape}, got {image.shape}."
                )
            preloaded[relative_index] = image
            progress.update(1)

        try:
            with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
                for relative_index, frame_index in enumerate(
                    range(start_frame + 1, end_frame + 1),
                    start=1,
                ):
                    frame_value = np.asarray(self.dataset[frame_index]).copy()
                    pending.append(
                        (
                            relative_index,
                            executor.submit(self._prepare_value, frame_value, frame_index),
                        )
                    )
                    if len(pending) >= max_pending:
                        store_oldest()
                while pending:
                    store_oldest()
        finally:
            progress.close()

        if cache_path is not None:
            temporary_path = cache_path.with_name(f"{cache_path.stem}.tmp{cache_path.suffix}")
            try:
                with temporary_path.open("wb") as cache_file:
                    np.save(cache_file, preloaded, allow_pickle=False)
                temporary_path.replace(cache_path)
                print(f"[INFO] saved frustum image cache: {cache_path}")
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()
        self.preloaded_frames = preloaded
        self.preload_start_frame = start_frame

    def frame(self, frame_index: int) -> np.ndarray:
        frame_index = int(frame_index)
        if self.preloaded_frames is not None:
            relative_index = frame_index - self.preload_start_frame
            if 0 <= relative_index < len(self.preloaded_frames):
                return self.preloaded_frames[relative_index]

        cached = self.cache.pop(frame_index, None)
        if cached is not None:
            self.cache[frame_index] = cached
            return cached

        result = self._prepare_value(np.asarray(self.dataset[frame_index]).copy(), frame_index)
        self.cache[frame_index] = result
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return result


def hex_to_rgb(color: str) -> tuple[float, float, float]:
    value = color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected #RRGGBB color, got {color!r}.")
    return tuple(int(value[index:index + 2], 16) / 255.0 for index in (0, 2, 4))


def alpha_composite_color(
    color: str | tuple[float, float, float],
    alpha: float,
) -> np.ndarray:
    rgb = np.asarray(hex_to_rgb(color) if isinstance(color, str) else color, dtype=np.float64)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return np.clip(rgb * alpha + BACKGROUND_COLOR * (1.0 - alpha), 0.0, 1.0)


def point_radius(point_size: float) -> float:
    return max(MIN_MESH_RADIUS, math.sqrt(max(float(point_size), 0.0)) * POINT_AREA_TO_RADIUS)


def line_radius(line_width: float) -> float:
    return max(MIN_MESH_RADIUS, float(line_width) * LINEWIDTH_TO_RADIUS)


def normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return vector if norm < eps else vector / norm


def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = normalize(a)
    b = normalize(b)
    cross = np.cross(a, b)
    dot = float(np.dot(a, b))

    if dot > 1.0 - 1e-8:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-8:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = normalize(np.cross(a, axis))
        skew = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
            dtype=np.float64,
        )
        return np.eye(3, dtype=np.float64) + 2.0 * skew @ skew

    skew = np.array(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]],
        dtype=np.float64,
    )
    cross_norm = float(np.linalg.norm(cross))
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * ((1.0 - dot) / (cross_norm * cross_norm))


def transform_points(matrix: np.ndarray, local_points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([local_points, np.ones(len(local_points), dtype=np.float64)])
    return (np.asarray(matrix, dtype=np.float64) @ homogeneous.T).T[:, :3]


def local_frustum_points(length: float, fov_deg: float, aspect: float) -> np.ndarray:
    half_height = length * np.tan(np.deg2rad(fov_deg) * 0.5)
    half_width = half_height * aspect
    corners = np.array(
        [
            [length, half_width, half_height],
            [length, -half_width, half_height],
            [length, -half_width, -half_height],
            [length, half_width, -half_height],
        ],
        dtype=np.float64,
    )
    return np.vstack([np.zeros(3, dtype=np.float64), corners])


def body_bones_from_names(joint_names: list[str]) -> list[tuple[int, int]]:
    indices = {name: index for index, name in enumerate(joint_names)}
    return [
        (indices[start], indices[end])
        for start, end in BODY_BONES_BY_NAME
        if start in indices and end in indices
    ]


class PrimitiveTemplates:
    def __init__(self, resolution: int) -> None:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=1.0, height=1.0, resolution=resolution)
        self.sphere_vertices = np.asarray(sphere.vertices).copy()
        self.sphere_triangles = np.asarray(sphere.triangles).astype(np.int32, copy=True)
        self.cylinder_vertices = np.asarray(cylinder.vertices).copy()
        self.cylinder_triangles = np.asarray(cylinder.triangles).astype(np.int32, copy=True)

    def spheres(self, points: np.ndarray, radius: float, color: np.ndarray) -> Optional[MeshPart]:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) == 0:
            return None

        vertices_per_sphere = len(self.sphere_vertices)
        vertices = (
            points[:, None, :] + self.sphere_vertices[None, :, :] * float(radius)
        ).reshape(-1, 3)
        offsets = np.arange(len(points), dtype=np.int32)[:, None, None] * vertices_per_sphere
        triangles = (self.sphere_triangles[None, :, :] + offsets).reshape(-1, 3)
        colors = np.repeat(np.asarray(color, dtype=np.float64)[None, :], len(vertices), axis=0)
        return MeshPart(vertices, triangles, colors)

    def cylinders(
        self,
        points: np.ndarray,
        bones: Iterable[tuple[int, int]],
        radius: float,
        color: np.ndarray,
    ) -> Optional[MeshPart]:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        segment_indices = np.asarray(
            [
                (start, end)
                for start, end in bones
                if 0 <= start < len(points) and 0 <= end < len(points)
            ],
            dtype=np.int32,
        ).reshape(-1, 2)
        if len(segment_indices) == 0:
            return None

        p0 = points[segment_indices[:, 0]]
        p1 = points[segment_indices[:, 1]]
        vectors = p1 - p0
        lengths = np.linalg.norm(vectors, axis=1)
        valid = (
            np.isfinite(p0).all(axis=1)
            & np.isfinite(p1).all(axis=1)
            & (lengths >= 1e-8)
        )
        if not np.any(valid):
            return None

        p0 = p0[valid]
        p1 = p1[valid]
        vectors = vectors[valid]
        lengths = lengths[valid]
        directions = vectors / lengths[:, None]

        # Build one orthonormal cylinder frame per bone in a batch.  Rotation
        # around a cylinder's own axis is visually irrelevant, so this avoids
        # dozens of Python loops and per-bone rotation-matrix allocations.
        references = np.zeros_like(directions)
        references[:, 2] = 1.0
        near_parallel = np.abs(directions[:, 2]) > 0.9
        references[near_parallel] = np.array([0.0, 1.0, 0.0])
        basis_x = np.cross(references, directions)
        basis_x /= np.linalg.norm(basis_x, axis=1)[:, None]
        basis_y = np.cross(directions, basis_x)

        local = self.cylinder_vertices
        vertices = (
            (p0 + p1)[:, None, :] * 0.5
            + local[None, :, 0, None] * float(radius) * basis_x[:, None, :]
            + local[None, :, 1, None] * float(radius) * basis_y[:, None, :]
            + local[None, :, 2, None] * lengths[:, None, None] * directions[:, None, :]
        ).reshape(-1, 3)

        vertices_per_cylinder = len(self.cylinder_vertices)
        valid_segment_count = len(lengths)
        offsets = np.arange(valid_segment_count, dtype=np.int32)[:, None, None] * vertices_per_cylinder
        triangles = (self.cylinder_triangles[None, :, :] + offsets).reshape(-1, 3)
        colors = np.repeat(np.asarray(color, dtype=np.float64)[None, :], len(vertices), axis=0)
        return MeshPart(vertices, triangles, colors)


def update_mesh(mesh: o3d.geometry.TriangleMesh, parts: Iterable[Optional[MeshPart]]) -> None:
    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    vertex_offset = 0
    for part in parts:
        if part is None or len(part.vertices) == 0:
            continue
        vertices.append(part.vertices)
        triangles.append(part.triangles + vertex_offset)
        colors.append(part.colors)
        vertex_offset += len(part.vertices)

    if not vertices:
        mesh.clear()
        return

    mesh.vertices = o3d.utility.Vector3dVector(np.vstack(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.vstack(triangles))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.vstack(colors))


def build_image_plane_triangles(row_count: int, col_count: int) -> np.ndarray:
    triangles: list[list[int]] = []
    for row in range(row_count - 1):
        for col in range(col_count - 1):
            top_left = row * col_count + col
            top_right = top_left + 1
            bottom_left = top_left + col_count
            bottom_right = bottom_left + 1
            triangles.append([top_left, bottom_left, top_right])
            triangles.append([top_right, bottom_left, bottom_right])
    return np.asarray(triangles, dtype=np.int32)


class FrustumImageActor:
    def __init__(self, frustum_local: np.ndarray, alpha: float) -> None:
        self.frustum_local = np.asarray(frustum_local, dtype=np.float64)
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.mesh = o3d.geometry.TriangleMesh()
        self.image_shape: Optional[tuple[int, int]] = None
        self.local_points = np.empty((0, 3), dtype=np.float64)

    def _set_image_shape(self, height: int, width: int) -> None:
        if height < 2 or width < 2:
            raise ValueError(f"Frustum image must be at least 2x2, got {height}x{width}.")
        corners = self.frustum_local[1:]
        y_values = np.linspace(float(corners[0, 1]), float(corners[1, 1]), width)
        z_values = np.linspace(float(corners[0, 2]), float(corners[3, 2]), height)
        yy, zz = np.meshgrid(y_values, z_values)
        self.local_points = np.column_stack(
            [
                np.full(yy.size, float(corners[0, 0]), dtype=np.float64),
                yy.reshape(-1),
                zz.reshape(-1),
            ]
        )
        self.mesh.triangles = o3d.utility.Vector3iVector(
            build_image_plane_triangles(height, width)
        )
        self.image_shape = (height, width)

    def set_frame(self, head_matrix: np.ndarray, image: np.ndarray) -> None:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Frustum image must have shape [H, W, 3], got {image.shape}.")
        height, width = image.shape[:2]
        if self.image_shape != (height, width):
            self._set_image_shape(height, width)

        world_points = transform_points(head_matrix, self.local_points)
        image_rgb = image.astype(np.float64, copy=False)
        if np.issubdtype(image.dtype, np.integer):
            image_rgb = image_rgb / 255.0
        colors = (
            np.clip(image_rgb.reshape(-1, 3), 0.0, 1.0) * self.alpha
            + BACKGROUND_COLOR[None, :] * (1.0 - self.alpha)
        )
        self.mesh.vertices = o3d.utility.Vector3dVector(world_points)
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(colors)


class SkeletonActor:
    def __init__(
        self,
        sequence: MotionSequence,
        frustum_local: np.ndarray,
        body_bones: list[tuple[int, int]],
        frame_alpha: float,
        trajectory_points: Optional[np.ndarray],
        show_trajectory_marker: bool,
        show_frustum_wire: bool,
    ) -> None:
        self.sequence = sequence
        self.frustum_local = frustum_local
        self.body_bones = body_bones
        self.trajectory_points = trajectory_points
        self.show_trajectory_marker = show_trajectory_marker
        self.show_frustum_wire = show_frustum_wire
        self.templates = PrimitiveTemplates(MESH_RESOLUTION)
        self.mesh = o3d.geometry.TriangleMesh()

        self.body_point_color = alpha_composite_color(BODY_POINT_COLOR, frame_alpha)
        self.body_line_color = alpha_composite_color(BODY_LINE_COLOR, frame_alpha)
        self.left_hand_color = alpha_composite_color(LEFT_HAND_COLOR, frame_alpha)
        self.right_hand_color = alpha_composite_color(RIGHT_HAND_COLOR, frame_alpha)
        self.frustum_color = alpha_composite_color(HEAD_FRUSTUM_COLOR, frame_alpha)
        self.chest_marker_color = alpha_composite_color(CHEST_TRAJECTORY_COLOR, CHEST_MARKER_ALPHA)

    def set_frame(self, frame_index: int) -> None:
        body = self.sequence.body[frame_index]
        left_hand = self.sequence.left_hand[frame_index]
        right_hand = self.sequence.right_hand[frame_index]
        frustum = transform_points(self.sequence.head[frame_index], self.frustum_local)

        parts: list[Optional[MeshPart]] = [
            self.templates.cylinders(body, self.body_bones, line_radius(BODY_LINE_WIDTH), self.body_line_color),
            self.templates.cylinders(left_hand, HAND_BONES, line_radius(HAND_LINE_WIDTH), self.left_hand_color),
            self.templates.cylinders(right_hand, HAND_BONES, line_radius(HAND_LINE_WIDTH), self.right_hand_color),
            (
                self.templates.cylinders(
                    frustum,
                    FRUSTUM_EDGES,
                    line_radius(FRUSTUM_LINE_WIDTH),
                    self.frustum_color,
                )
                if self.show_frustum_wire
                else None
            ),
            self.templates.spheres(body, point_radius(BODY_POINT_SIZE), self.body_point_color),
            self.templates.spheres(left_hand, point_radius(HAND_POINT_SIZE), self.left_hand_color),
            self.templates.spheres(right_hand, point_radius(HAND_POINT_SIZE), self.right_hand_color),
            self.templates.spheres(frustum[:1], point_radius(HEAD_POINT_SIZE), self.frustum_color),
        ]
        if self.show_trajectory_marker and self.trajectory_points is not None:
            parts.append(
                self.templates.spheres(
                    self.trajectory_points[frame_index:frame_index + 1],
                    point_radius(CHEST_SAMPLE_SIZE),
                    self.chest_marker_color,
                )
            )
        update_mesh(self.mesh, parts)


def selected_frustum_points(
    head: np.ndarray,
    start_frame: int,
    end_frame: int,
    frustum_local: np.ndarray,
) -> np.ndarray:
    matrices = head[start_frame:end_frame + 1]
    homogeneous = np.column_stack([frustum_local, np.ones(len(frustum_local), dtype=np.float64)])
    return np.einsum("tij,pj->tpi", matrices, homogeneous)[:, :, :3]


def finite_min_max(arrays: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    found = False
    for array in arrays:
        points = np.asarray(array, dtype=np.float64).reshape(-1, 3)
        finite = points[np.isfinite(points).all(axis=1)]
        if len(finite) == 0:
            continue
        minimum = np.minimum(minimum, finite.min(axis=0))
        maximum = np.maximum(maximum, finite.max(axis=0))
        found = True
    if not found:
        raise ValueError("No finite 3D points are available in the selected frame range.")
    return minimum, maximum


def scene_center_and_extent(
    sequence: MotionSequence,
    start_frame: int,
    end_frame: int,
    frustum_local: np.ndarray,
) -> tuple[np.ndarray, float]:
    frame_slice = slice(start_frame, end_frame + 1)
    minimum, maximum = finite_min_max(
        (
            sequence.body[frame_slice],
            sequence.left_hand[frame_slice],
            sequence.right_hand[frame_slice],
            selected_frustum_points(sequence.head, start_frame, end_frame, frustum_local),
        )
    )
    center = (minimum + maximum) * 0.5
    radius = max(float(np.max(maximum - minimum)) * 0.55, 0.25)
    return center, radius * 2.0


def first_frame_foot_ground_z(sequence: MotionSequence, frame_index: int) -> float:
    index_by_name = {name: index for index, name in enumerate(sequence.joint_names)}
    foot_indices = [index_by_name[name] for name in FOOT_JOINT_NAMES if name in index_by_name]
    points = sequence.body[frame_index, foot_indices] if foot_indices else sequence.body[frame_index]
    finite_z = points[np.isfinite(points).all(axis=1), 2]
    return float(np.min(finite_z)) if finite_z.size else GROUND_GRID_FALLBACK_Z


def create_ground_grid(
    sequence: MotionSequence,
    start_frame: int,
    end_frame: int,
    grid_z: float,
) -> o3d.geometry.LineSet:
    frame_slice = slice(start_frame, end_frame + 1)
    xy_sources = (
        sequence.body[frame_slice],
        sequence.left_hand[frame_slice],
        sequence.right_hand[frame_slice],
        sequence.head[frame_slice, :3, 3],
    )
    minimum, maximum = finite_min_max(xy_sources)
    xy_min = minimum[:2] - GROUND_GRID_PADDING
    xy_max = maximum[:2] + GROUND_GRID_PADDING
    x_min, y_min = np.floor(xy_min / GROUND_GRID_SIZE) * GROUND_GRID_SIZE
    x_max, y_max = np.ceil(xy_max / GROUND_GRID_SIZE) * GROUND_GRID_SIZE
    xs = np.arange(x_min, x_max + GROUND_GRID_SIZE * 0.5, GROUND_GRID_SIZE)
    ys = np.arange(y_min, y_max + GROUND_GRID_SIZE * 0.5, GROUND_GRID_SIZE)

    points: list[list[float]] = []
    lines: list[list[int]] = []
    for x_value in xs:
        index = len(points)
        points.extend([[float(x_value), float(y_min), grid_z], [float(x_value), float(y_max), grid_z]])
        lines.append([index, index + 1])
    for y_value in ys:
        index = len(points)
        points.extend([[float(x_min), float(y_value), grid_z], [float(x_max), float(y_value), grid_z]])
        lines.append([index, index + 1])

    color = alpha_composite_color(GROUND_GRID_COLOR, GROUND_GRID_ALPHA)
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    grid.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    grid.colors = o3d.utility.Vector3dVector(np.repeat(color[None, :], len(lines), axis=0))
    return grid


def trajectory_points(
    sequence: MotionSequence,
    target: str,
) -> Optional[np.ndarray]:
    if target == "head":
        return sequence.head[:, :3, 3].astype(np.float64, copy=False)
    if target == "chest":
        joint_indices = {name: index for index, name in enumerate(sequence.joint_names)}
        chest_index = joint_indices.get("chest")
        if chest_index is None:
            return None
        return sequence.body[:, chest_index].astype(np.float64, copy=False)
    raise ValueError(f"Unsupported trajectory target: {target!r}.")


class ProgressiveTrajectoryActor:
    """Precompute the trajectory once and reveal only its elapsed prefix."""

    def __init__(self, points: np.ndarray, start_frame: int, end_frame: int) -> None:
        self.start_frame = start_frame
        selected = np.asarray(points[start_frame:end_frame + 1], dtype=np.float64)
        segment_count = max(0, len(selected) - 1)
        bones = tuple((index, index + 1) for index in range(segment_count))
        templates = PrimitiveTemplates(TRAJECTORY_MESH_RESOLUTION)
        part = templates.cylinders(
            selected,
            bones,
            line_radius(CHEST_TRAJECTORY_LINE_WIDTH),
            alpha_composite_color(CHEST_TRAJECTORY_COLOR, CHEST_TRAJECTORY_ALPHA),
        )

        if segment_count:
            vectors = selected[1:] - selected[:-1]
            valid_segments = (
                np.isfinite(selected[:-1]).all(axis=1)
                & np.isfinite(selected[1:]).all(axis=1)
                & (np.linalg.norm(vectors, axis=1) >= 1e-8)
            )
            self.valid_prefix = np.concatenate(
                [np.zeros(1, dtype=np.int32), np.cumsum(valid_segments, dtype=np.int32)]
            )
        else:
            self.valid_prefix = np.zeros(1, dtype=np.int32)

        self.vertices_per_segment = len(templates.cylinder_vertices)
        self.triangles_per_segment = len(templates.cylinder_triangles)
        self.full_vertices = (
            part.vertices if part is not None else np.empty((0, 3), dtype=np.float64)
        )
        self.full_triangles = (
            part.triangles if part is not None else np.empty((0, 3), dtype=np.int32)
        )
        self.full_colors = (
            part.colors if part is not None else np.empty((0, 3), dtype=np.float64)
        )
        self.mesh = o3d.geometry.TriangleMesh()
        self.visible_segment_count = -1
        self.set_frame(start_frame)

    def set_frame(self, frame_index: int) -> None:
        elapsed_segments = int(
            np.clip(frame_index - self.start_frame, 0, len(self.valid_prefix) - 1)
        )
        valid_segment_count = int(self.valid_prefix[elapsed_segments])
        if valid_segment_count == self.visible_segment_count:
            return
        self.visible_segment_count = valid_segment_count
        if valid_segment_count == 0:
            self.mesh.clear()
            return

        vertex_count = valid_segment_count * self.vertices_per_segment
        triangle_count = valid_segment_count * self.triangles_per_segment
        self.mesh.vertices = o3d.utility.Vector3dVector(self.full_vertices[:vertex_count])
        self.mesh.triangles = o3d.utility.Vector3iVector(self.full_triangles[:triangle_count])
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(self.full_colors[:vertex_count])


def create_axis_anchors(center: np.ndarray, extent: float) -> o3d.geometry.TriangleMesh:
    half_extent = extent * 0.5
    corners = np.array(
        [
            [x, y, z]
            for x in (center[0] - half_extent, center[0] + half_extent)
            for y in (center[1] - half_extent, center[1] + half_extent)
            for z in (center[2] - half_extent, center[2] + half_extent)
        ],
        dtype=np.float64,
    )
    templates = PrimitiveTemplates(6)
    mesh = o3d.geometry.TriangleMesh()
    update_mesh(mesh, [templates.spheres(corners, MIN_MESH_RADIUS, WINDOW_BACKGROUND_COLOR)])
    return mesh


def view_vectors(elev: float, azim: float) -> tuple[np.ndarray, np.ndarray]:
    elev_rad = np.deg2rad(elev)
    azim_rad = np.deg2rad(azim)
    camera_from_target = np.array(
        [
            np.cos(elev_rad) * np.cos(azim_rad),
            np.cos(elev_rad) * np.sin(azim_rad),
            np.sin(elev_rad),
        ],
        dtype=np.float64,
    )
    front = normalize(-camera_from_target)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = normalize(np.cross(front, world_up))
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64) if np.linalg.norm(right) < 1e-8 else normalize(np.cross(right, front))
    return front, up


def configure_visualizer(
    vis: o3d.visualization.Visualizer,
    center: np.ndarray,
    extent: float,
    elev: float,
    azim: float,
) -> None:
    render_option = vis.get_render_option()
    render_option.background_color = WINDOW_BACKGROUND_COLOR.copy()
    render_option.mesh_show_back_face = True
    render_option.light_on = False
    try:
        render_option.line_width = GROUND_GRID_LINESET_PIXEL_WIDTH
    except Exception:
        pass

    view_control = vis.get_view_control()
    front, up = view_vectors(elev, azim)
    view_control.set_lookat(center)
    view_control.set_front(front)
    view_control.set_up(up)
    view_control.set_zoom(0.72)
    try:
        view_control.set_constant_z_near(0.001)
        view_control.set_constant_z_far(max(10.0, float(extent) * 8.0))
    except Exception:
        pass


class TrajectoryPlayer:
    def __init__(
        self,
        sequence: MotionSequence,
        start_frame: int,
        end_frame: int,
        playback_fps: float,
        playback_mode: str,
        playback_speed: float,
        timestamp_source: str,
        late_frame_policy: str,
        fps_report_interval: float,
        loop: bool,
        paused: bool,
        frame_alpha: float,
        show_trajectory: bool,
        trajectory_target: str,
        show_ground_grid: bool,
        frustum_mode: str,
        frustum_local: np.ndarray,
        image_dataset: Optional[str],
        image_eye: str,
        image_color_order: str,
        frustum_image_width: int,
        frustum_image_alpha: float,
        preload_images: bool,
        preload_workers: int,
        cache_dir: Optional[Path],
        elev: float,
        azim: float,
        window_width: int,
        window_height: int,
    ) -> None:
        self.sequence = sequence
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.frame_index = start_frame
        self.playback_fps = playback_fps
        self.playback_mode = playback_mode
        self.playback_speed = playback_speed
        self.late_frame_policy = late_frame_policy
        self.fps_report_interval = fps_report_interval
        self.loop = loop
        self.paused = paused
        self.show_trajectory = show_trajectory
        self.trajectory_target = trajectory_target
        self.show_ground_grid = show_ground_grid
        self.frustum_mode = frustum_mode
        self.frustum_local = frustum_local
        self.image_dataset = image_dataset
        self.image_eye = image_eye
        self.image_color_order = image_color_order
        self.frustum_image_width = frustum_image_width
        self.frustum_image_alpha = frustum_image_alpha
        self.preload_images = preload_images
        self.preload_workers = preload_workers
        self.cache_dir = cache_dir
        self.elev = elev
        self.azim = azim
        self.window_width = window_width
        self.window_height = window_height
        self.should_close = False
        self.last_tick_time = time.perf_counter()
        self.last_fps_report_time = self.last_tick_time
        self.presented_frame_times: deque[float] = deque()
        self.frame_update_pending = False
        self.render_requested = True
        self.frame_accumulator = 0.0
        self.timestamp_source: Optional[str] = None
        self.playback_timestamps_ns: Optional[np.ndarray] = None
        self.timestamp_tail_interval_ns = int(round(1e9 / self.playback_fps))
        self.timestamp_anchor_wall_time = self.last_tick_time
        self.timestamp_anchor_ns = 0

        if playback_mode == "timestamp":
            self.timestamp_source, self.playback_timestamps_ns = resolve_timestamp_stream(
                sequence,
                timestamp_source,
                start_frame,
                end_frame,
            )
            selected_timestamps = self.playback_timestamps_ns[start_frame:end_frame + 1]
            positive_intervals = np.diff(selected_timestamps)
            positive_intervals = positive_intervals[positive_intervals > 0]
            if len(positive_intervals) > 0:
                self.timestamp_tail_interval_ns = max(
                    1,
                    int(round(float(np.median(positive_intervals)))),
                )
            self.timestamp_anchor_ns = int(self.playback_timestamps_ns[self.frame_index])

        self.current_trajectory_points = trajectory_points(sequence, trajectory_target)
        if self.current_trajectory_points is None:
            print(
                f"[WARN] Trajectory target '{trajectory_target}' is unavailable; "
                "trajectory rendering has been disabled."
            )
            self.show_trajectory = False

        self.center, self.extent = scene_center_and_extent(
            sequence, start_frame, end_frame, frustum_local
        )
        self.skeleton = SkeletonActor(
            sequence=sequence,
            frustum_local=frustum_local,
            body_bones=body_bones_from_names(sequence.joint_names),
            frame_alpha=frame_alpha,
            trajectory_points=self.current_trajectory_points,
            show_trajectory_marker=self.show_trajectory,
            show_frustum_wire=frustum_mode in {"wire", "both"},
        )
        self.frustum_image_actor = FrustumImageActor(frustum_local, frustum_image_alpha)
        self.image_loader: Optional[FrustumImageLoader] = None
        self.grid = create_ground_grid(
            sequence,
            start_frame,
            end_frame,
            first_frame_foot_ground_z(sequence, start_frame),
        )
        self.trajectory_actor = (
            ProgressiveTrajectoryActor(self.current_trajectory_points, start_frame, end_frame)
            if self.current_trajectory_points is not None
            else None
        )
        self.trajectory = (
            self.trajectory_actor.mesh
            if self.trajectory_actor is not None
            else o3d.geometry.TriangleMesh()
        )
        self.anchors = create_axis_anchors(self.center, self.extent)
        self.grid_added = False
        self.trajectory_added = False
        self.frustum_image_added = False

    def ensure_image_loader(self) -> FrustumImageLoader:
        if self.image_loader is None:
            self.image_loader = FrustumImageLoader(
                h5_path=self.sequence.path,
                dataset_name=self.image_dataset,
                eye=self.image_eye,
                color_order=self.image_color_order,
                target_width=self.frustum_image_width,
                required_end_frame=self.end_frame,
            )
        return self.image_loader

    def update_frustum_image(self, frame_index: int) -> None:
        loader = self.ensure_image_loader()
        image = loader.frame(frame_index)
        self.frustum_image_actor.set_frame(self.sequence.head[frame_index], image)

    def preload_frustum_images(self) -> None:
        if not self.preload_images or self.frustum_mode not in {"image", "both"}:
            return
        loader = self.ensure_image_loader()
        if (
            loader.preloaded_frames is not None
            and loader.preload_start_frame <= self.start_frame
            and loader.preload_start_frame + len(loader.preloaded_frames) > self.end_frame
        ):
            return
        loader.preload(
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            workers=self.preload_workers,
            cache_dir=self.cache_dir,
        )

    def set_frame(self, frame_index: int, vis: o3d.visualization.Visualizer, announce: bool = False) -> None:
        previous_frame = self.frame_index
        self.frame_index = int(np.clip(frame_index, self.start_frame, self.end_frame))
        self.skeleton.set_frame(self.frame_index)
        vis.update_geometry(self.skeleton.mesh)
        if self.frustum_mode in {"image", "both"}:
            self.update_frustum_image(self.frame_index)
            if self.frustum_image_added:
                vis.update_geometry(self.frustum_image_actor.mesh)
        if self.trajectory_actor is not None:
            self.trajectory_actor.set_frame(self.frame_index)
            if self.show_trajectory and len(self.trajectory.vertices) > 0:
                if self.trajectory_added:
                    vis.update_geometry(self.trajectory)
                else:
                    vis.add_geometry(self.trajectory, reset_bounding_box=False)
                    self.trajectory_added = True
            elif self.trajectory_added:
                vis.remove_geometry(self.trajectory, reset_bounding_box=False)
                self.trajectory_added = False
        if self.frame_index != previous_frame:
            self.frame_update_pending = True
        self.render_requested = True
        if announce:
            selected_index = self.frame_index - self.start_frame + 1
            selected_count = self.end_frame - self.start_frame + 1
            print(
                f"[INFO] frame={self.frame_index} "
                f"(selected {selected_index}/{selected_count})"
            )

    def toggle_play(self) -> None:
        self.paused = not self.paused
        self.reset_playback_clock()
        print(f"[INFO] playback={'paused' if self.paused else 'running'}")

    def reset_playback_clock(self, now: Optional[float] = None) -> None:
        now = time.perf_counter() if now is None else now
        self.last_tick_time = now
        self.frame_accumulator = 0.0
        self.timestamp_anchor_wall_time = now
        self.last_fps_report_time = now
        self.presented_frame_times.clear()
        self.frame_update_pending = False
        if self.playback_timestamps_ns is not None:
            self.timestamp_anchor_ns = int(self.playback_timestamps_ns[self.frame_index])

    def record_and_report_fps(self, now: Optional[float] = None) -> None:
        now = time.perf_counter() if now is None else now
        if self.frame_update_pending:
            self.presented_frame_times.append(now)
            self.frame_update_pending = False

        if self.fps_report_interval <= 0.0:
            return
        cutoff = now - self.fps_report_interval
        while self.presented_frame_times and self.presented_frame_times[0] < cutoff:
            self.presented_frame_times.popleft()
        if now - self.last_fps_report_time < self.fps_report_interval:
            return

        times = np.asarray(self.presented_frame_times, dtype=np.float64)
        if len(times) >= 2:
            intervals_ms = np.diff(times) * 1000.0
            measured_fps = (len(times) - 1) / max(times[-1] - times[0], 1e-9)
            print(
                f"[PERF] displayed={measured_fps:.2f} FPS, "
                f"interval p95={np.percentile(intervals_ms, 95):.1f} ms, "
                f"max={intervals_ms.max():.1f} ms"
            )
        elif not self.paused:
            print("[PERF] fewer than 2 frames were displayed in the reporting window")
        self.last_fps_report_time = now

    def step(self, delta: int, vis: o3d.visualization.Visualizer) -> None:
        self.paused = True
        self.set_frame(self.frame_index + delta, vis, announce=True)
        self.reset_playback_clock()

    def restart(self, vis: o3d.visualization.Visualizer) -> None:
        self.paused = True
        self.set_frame(self.start_frame, vis, announce=True)
        self.reset_playback_clock()

    def toggle_trajectory(self, vis: o3d.visualization.Visualizer) -> None:
        if self.current_trajectory_points is None:
            print(f"[INFO] {self.trajectory_target} trajectory=unavailable")
            return
        self.show_trajectory = not self.show_trajectory
        self.skeleton.show_trajectory_marker = self.show_trajectory
        self.set_frame(self.frame_index, vis)
        print(
            f"[INFO] {self.trajectory_target} trajectory="
            f"{'on' if self.show_trajectory else 'off'}"
        )

    def switch_trajectory_target(self, vis: o3d.visualization.Visualizer) -> None:
        next_target = "head" if self.trajectory_target == "chest" else "chest"
        next_points = trajectory_points(self.sequence, next_target)
        if next_points is None:
            print(f"[INFO] {next_target} trajectory=unavailable")
            return

        if self.trajectory_added:
            vis.remove_geometry(self.trajectory, reset_bounding_box=False)
            self.trajectory_added = False
        self.trajectory_target = next_target
        self.current_trajectory_points = next_points
        self.trajectory_actor = ProgressiveTrajectoryActor(
            next_points,
            self.start_frame,
            self.end_frame,
        )
        self.trajectory_actor.set_frame(self.frame_index)
        self.trajectory = self.trajectory_actor.mesh
        self.skeleton.trajectory_points = next_points
        self.set_frame(self.frame_index, vis)
        print(f"[INFO] trajectory target={self.trajectory_target}")

    def cycle_frustum_mode(self, vis: o3d.visualization.Visualizer) -> None:
        next_index = (FRUSTUM_MODES.index(self.frustum_mode) + 1) % len(FRUSTUM_MODES)
        next_mode = FRUSTUM_MODES[next_index]
        if next_mode in {"image", "both"}:
            try:
                if self.preload_images:
                    loader = self.ensure_image_loader()
                    if loader.preloaded_frames is None:
                        loader.preload(
                            start_frame=self.start_frame,
                            end_frame=self.end_frame,
                            workers=self.preload_workers,
                            cache_dir=self.cache_dir,
                        )
                self.update_frustum_image(self.frame_index)
            except Exception as exc:
                print(f"[ERROR] Cannot enable frustum image mode: {exc}")
                return

        self.frustum_mode = next_mode
        self.skeleton.show_frustum_wire = next_mode in {"wire", "both"}
        if next_mode in {"image", "both"} and not self.frustum_image_added:
            vis.add_geometry(self.frustum_image_actor.mesh, reset_bounding_box=False)
            self.frustum_image_added = True
        elif next_mode == "wire" and self.frustum_image_added:
            vis.remove_geometry(self.frustum_image_actor.mesh, reset_bounding_box=False)
            self.frustum_image_added = False
        self.set_frame(self.frame_index, vis)
        print(f"[INFO] frustum mode={self.frustum_mode}")

    def toggle_grid(self, vis: o3d.visualization.Visualizer) -> None:
        self.show_ground_grid = not self.show_ground_grid
        if self.show_ground_grid and not self.grid_added:
            vis.add_geometry(self.grid, reset_bounding_box=False)
            self.grid_added = True
        elif not self.show_ground_grid and self.grid_added:
            vis.remove_geometry(self.grid, reset_bounding_box=False)
            self.grid_added = False
        self.render_requested = True
        print(f"[INFO] ground grid={'on' if self.show_ground_grid else 'off'}")

    def tick_fixed_fps(self, vis: o3d.visualization.Visualizer, now: float) -> None:
        elapsed = now - self.last_tick_time
        self.last_tick_time = now
        self.frame_accumulator += elapsed * self.playback_fps * self.playback_speed
        due_steps = int(self.frame_accumulator)
        if due_steps <= 0:
            return
        if self.late_frame_policy == "slow":
            steps = 1
            # Preserve sub-frame timing error so a render cycle just below the
            # target interval does not force us to wait through a second render.
            # Cap old backlog while still displaying every visual frame.
            self.frame_accumulator = min(self.frame_accumulator - 1.0, 1.0)
        else:
            steps = due_steps
            self.frame_accumulator -= steps

        next_frame = self.frame_index + steps
        if next_frame > self.end_frame:
            if self.loop:
                selected_count = self.end_frame - self.start_frame + 1
                next_frame = self.start_frame + (next_frame - self.start_frame) % selected_count
            else:
                next_frame = self.end_frame
                self.paused = True
                print("[INFO] playback reached the selected final frame")
        self.set_frame(next_frame, vis)

    def tick_timestamp(self, vis: o3d.visualization.Visualizer, now: float) -> None:
        assert self.playback_timestamps_ns is not None
        elapsed_ns = int(
            round((now - self.timestamp_anchor_wall_time) * 1e9 * self.playback_speed)
        )
        target_timestamp_ns = self.timestamp_anchor_ns + elapsed_ns
        first_timestamp_ns = int(self.playback_timestamps_ns[self.start_frame])
        end_deadline_ns = (
            int(self.playback_timestamps_ns[self.end_frame])
            + self.timestamp_tail_interval_ns
        )

        if target_timestamp_ns >= end_deadline_ns:
            if self.late_frame_policy == "slow" and self.frame_index < self.end_frame:
                next_frame = self.frame_index + 1
                self.set_frame(next_frame, vis)
                self.timestamp_anchor_wall_time = now
                self.timestamp_anchor_ns = int(self.playback_timestamps_ns[next_frame])
                return
            if self.loop:
                self.timestamp_anchor_wall_time = now
                if self.late_frame_policy == "slow":
                    target_timestamp_ns = first_timestamp_ns
                    self.timestamp_anchor_ns = first_timestamp_ns
                else:
                    period_ns = max(1, end_deadline_ns - first_timestamp_ns)
                    target_timestamp_ns = (
                        first_timestamp_ns
                        + (target_timestamp_ns - first_timestamp_ns) % period_ns
                    )
                    self.timestamp_anchor_ns = target_timestamp_ns
            else:
                self.set_frame(self.end_frame, vis)
                self.paused = True
                print("[INFO] playback reached the selected final timestamp")
                return

        selected = self.playback_timestamps_ns[self.start_frame:self.end_frame + 1]
        relative_index = int(np.searchsorted(selected, target_timestamp_ns, side="right") - 1)
        target_frame = self.start_frame + int(np.clip(relative_index, 0, len(selected) - 1))
        clock_was_more_than_one_frame_late = (
            self.late_frame_policy == "slow"
            and target_frame > self.frame_index + 1
        )
        if clock_was_more_than_one_frame_late:
            target_frame = self.frame_index + 1
        if target_frame != self.frame_index:
            self.set_frame(target_frame, vis)
            # Keep the original clock during normal playback so a few
            # milliseconds of event-loop overshoot do not accumulate once per
            # frame.  Re-anchor only after genuine multi-frame lateness.
            if clock_was_more_than_one_frame_late:
                self.timestamp_anchor_wall_time = now
                self.timestamp_anchor_ns = int(self.playback_timestamps_ns[target_frame])

    def tick(self, vis: o3d.visualization.Visualizer) -> None:
        now = time.perf_counter()
        if self.paused or self.start_frame == self.end_frame:
            self.last_tick_time = now
            return
        if self.playback_mode == "timestamp":
            self.tick_timestamp(vis, now)
        else:
            self.tick_fixed_fps(vis, now)

    def register_callbacks(self, vis: o3d.visualization.VisualizerWithKeyCallback) -> None:
        def on_space(_vis):
            self.toggle_play()
            return False

        def on_next(_vis):
            self.step(1, _vis)
            return False

        def on_previous(_vis):
            self.step(-1, _vis)
            return False

        def on_restart(_vis):
            self.restart(_vis)
            return False

        def on_trajectory(_vis):
            self.toggle_trajectory(_vis)
            return False

        def on_trajectory_target(_vis):
            self.switch_trajectory_target(_vis)
            return False

        def on_frustum_mode(_vis):
            self.cycle_frustum_mode(_vis)
            return False

        def on_grid(_vis):
            self.toggle_grid(_vis)
            return False

        def on_loop(_vis):
            self.loop = not self.loop
            print(f"[INFO] loop={'on' if self.loop else 'off'}")
            return False

        def on_quit(_vis):
            self.should_close = True
            return False

        vis.register_key_callback(32, on_space)
        vis.register_key_callback(ord("N"), on_next)
        vis.register_key_callback(262, on_next)       # GLFW_KEY_RIGHT
        vis.register_key_callback(ord("P"), on_previous)
        vis.register_key_callback(263, on_previous)   # GLFW_KEY_LEFT
        vis.register_key_callback(ord("R"), on_restart)
        vis.register_key_callback(ord("T"), on_trajectory)
        vis.register_key_callback(ord("Y"), on_trajectory_target)
        vis.register_key_callback(ord("F"), on_frustum_mode)
        vis.register_key_callback(ord("G"), on_grid)
        vis.register_key_callback(ord("L"), on_loop)
        vis.register_key_callback(ord("Q"), on_quit)
        vis.register_key_callback(256, on_quit)       # GLFW_KEY_ESCAPE

    def run(self) -> None:
        self.preload_frustum_images()
        vis = o3d.visualization.VisualizerWithKeyCallback()
        created = vis.create_window(
            window_name=f"Human skeleton trajectory - {self.sequence.path.name}",
            width=self.window_width,
            height=self.window_height,
        )
        if not created:
            raise RuntimeError("Open3D could not create a visualization window.")

        try:
            self.skeleton.set_frame(self.frame_index)
            if self.frustum_mode in {"image", "both"}:
                self.update_frustum_image(self.frame_index)
            vis.add_geometry(self.anchors, reset_bounding_box=True)
            vis.add_geometry(self.skeleton.mesh, reset_bounding_box=False)
            if self.frustum_mode in {"image", "both"}:
                vis.add_geometry(self.frustum_image_actor.mesh, reset_bounding_box=False)
                self.frustum_image_added = True
            if self.show_ground_grid:
                vis.add_geometry(self.grid, reset_bounding_box=False)
                self.grid_added = True
            if self.show_trajectory and len(self.trajectory.vertices) > 0:
                vis.add_geometry(self.trajectory, reset_bounding_box=False)
                self.trajectory_added = True

            vis.reset_view_point(True)
            configure_visualizer(vis, self.center, self.extent, self.elev, self.azim)
            self.register_callbacks(vis)

            print("")
            print(f"Loaded: {self.sequence.path}")
            print(
                f"Frames: {self.start_frame}..{self.end_frame} "
                f"({self.end_frame - self.start_frame + 1} total)"
            )
            if self.playback_mode == "timestamp":
                assert self.playback_timestamps_ns is not None
                recorded_duration = (
                    int(self.playback_timestamps_ns[self.end_frame])
                    - int(self.playback_timestamps_ns[self.start_frame])
                ) / 1e9
                selected_timestamps = self.playback_timestamps_ns[
                    self.start_frame:self.end_frame + 1
                ]
                intervals_ms = np.diff(selected_timestamps).astype(np.float64) / 1e6
                recorded_fps = (
                    (len(selected_timestamps) - 1) / recorded_duration
                    if recorded_duration > 0.0 and len(selected_timestamps) > 1
                    else 0.0
                )
                print(
                    f"Playback: timestamp, source={self.timestamp_source} "
                    f"({TIMESTAMP_DATASETS[self.timestamp_source]}), "
                    f"recorded duration={recorded_duration:.3f}s, "
                    f"recorded average={recorded_fps:.3f} FPS, "
                    f"speed={self.playback_speed:g}x"
                )
                if len(intervals_ms) > 0:
                    print(
                        f"Recorded intervals: median={np.median(intervals_ms):.1f} ms, "
                        f"p95={np.percentile(intervals_ms, 95):.1f} ms, "
                        f"max={intervals_ms.max():.1f} ms"
                    )
            else:
                print(
                    f"Playback: fixed FPS={self.playback_fps:.3f}, "
                    f"effective FPS={self.playback_fps * self.playback_speed:.3f}, "
                    f"speed={self.playback_speed:g}x"
                )
            print(f"Late-frame policy: {self.late_frame_policy}")
            print(
                f"Trajectory: {self.trajectory_target}, frustum: {self.frustum_mode}"
            )
            if self.image_loader is not None:
                image_source = (
                    "preloaded"
                    if self.image_loader.preloaded_frames is not None
                    else "on demand"
                )
                print(
                    f"Frustum image eye: {self.image_loader.used_eye}, "
                    f"color order: {self.image_color_order}, source: {image_source}"
                )
            print("Controls:")
            print("  Space       play / pause")
            print("  N / Right   next frame")
            print("  P / Left    previous frame")
            print("  R           restart")
            print("  T           toggle selected trajectory")
            print("  Y           switch chest / head trajectory")
            print("  F           cycle wire / image / both frustum")
            print("  G           toggle ground grid")
            print("  L           toggle loop")
            print("  Q / Esc     quit")
            print("")

            self.reset_playback_clock()
            self.render_requested = True
            while not self.should_close and vis.poll_events():
                self.tick(vis)
                if self.render_requested:
                    vis.update_renderer()
                    self.render_requested = False
                self.record_and_report_fps()
                time.sleep(0.001)
        finally:
            if self.image_loader is not None:
                self.image_loader.close()
            vis.destroy_window()


def main() -> None:
    args = parse_args()
    h5_path = args.h5_path.expanduser().resolve()
    if not h5_path.is_file():
        raise SystemExit(f"HDF5 file not found: {h5_path}")
    if args.fps < 0.0 or not np.isfinite(args.fps):
        raise SystemExit(f"--fps must be 0 or a positive finite value, got {args.fps!r}.")

    sequence = load_sequence(h5_path)
    start_frame, end_frame = normalize_frame_range(
        sequence.frame_count,
        args.start_frame,
        args.end_frame,
    )
    playback_fps = float(args.fps) if args.fps > 0.0 else sequence.playback_fps
    frustum_local = local_frustum_points(
        args.frustum_length,
        args.frustum_fov_deg,
        args.frustum_aspect,
    )

    player = TrajectoryPlayer(
        sequence=sequence,
        start_frame=start_frame,
        end_frame=end_frame,
        playback_fps=playback_fps,
        playback_mode=args.playback_mode,
        playback_speed=args.playback_speed,
        timestamp_source=args.timestamp_source,
        late_frame_policy=args.late_frame_policy,
        fps_report_interval=args.fps_report_interval,
        loop=args.loop,
        paused=args.paused,
        frame_alpha=args.frame_alpha,
        show_trajectory=not args.no_trajectory,
        trajectory_target=args.trajectory_target,
        show_ground_grid=not args.no_ground_grid,
        frustum_mode=args.frustum_mode,
        frustum_local=frustum_local,
        image_dataset=args.image_dataset,
        image_eye=args.image_eye,
        image_color_order=args.image_color_order,
        frustum_image_width=args.frustum_image_width,
        frustum_image_alpha=args.frustum_image_alpha,
        preload_images=args.preload_images,
        preload_workers=args.preload_workers,
        cache_dir=args.cache_dir,
        elev=args.elev,
        azim=args.azim,
        window_width=args.window_width,
        window_height=args.window_height,
    )
    player.run()


if __name__ == "__main__":
    main()
