"""HDF5 input/output helpers for the offline retargeting commands."""

from __future__ import annotations

from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from .constants import HAND_POINT_COUNT
from .constants import HAND_POSE_DIM
from .constants import HAND_SIDES
from .geometry import transforms_to_pose7


UPPER_BODY_TARGETS = (
    ("astribot_arm_left", "astribot_arm_left"),
    ("astribot_arm_right", "astribot_arm_right"),
    ("astribot_torso", "astribot_torso"),
)
HEAD_STEREO_RGB_KEY = "images_dict/head_stereo/rgb"
STEREO_LEFT_RGB_KEY = "images_dict/stereo_left/rgb"
STEREO_RIGHT_RGB_KEY = "images_dict/stereo_right/rgb"


def read_joint_index(h5_file: h5py.File) -> dict[str, int]:
    return {
        (name.decode("utf-8") if isinstance(name, bytes) else str(name)): idx
        for idx, name in enumerate(h5_file["body/joint_names"][:])
    }


def build_retarget_frames(h5_file: h5py.File) -> list[dict]:
    body_matrices = h5_file["body/matrices_f32"][:]
    head_matrices = h5_file["head/matrix_f32"][:]
    hand_matrices = {
        side: h5_file[f"body/{side}_hand_matrices_f32"][:]
        for side in HAND_SIDES
    }
    frame_times = h5_file["body/sample_wall_ns"][:] * 1e-9

    return [
        {
            "body_matrices": body_matrices[idx],
            "head_matrix": head_matrices[idx],
            "hand_matrices_by_side": {
                side: hand_matrices[side][idx]
                for side in HAND_SIDES
            },
            "frame_index": idx,
            "frame_time": float(frame_times[idx]),
        }
        for idx in range(body_matrices.shape[0])
    ]


def decode_jpeg_rgb(jpeg_data) -> np.ndarray:
    if isinstance(jpeg_data, (bytes, bytearray)):
        encoded = np.frombuffer(jpeg_data, dtype=np.uint8)
    else:
        encoded = np.asarray(jpeg_data, dtype=np.uint8).reshape(-1)

    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Failed to decode JPEG frame with OpenCV.")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def center_crop(frame: np.ndarray, crop: tuple[int, int]) -> np.ndarray:
    target_height, target_width = crop
    height, width, _ = frame.shape
    if target_height > height or target_width > width:
        raise ValueError(
            f"Crop size {target_height}x{target_width} is larger than "
            f"frame size {height}x{width}."
        )

    y0 = (height - target_height) // 2
    x0 = (width - target_width) // 2
    return frame[y0:y0 + target_height, x0:x0 + target_width, :]


def split_and_resize_head_stereo_frame(
    frame: np.ndarray,
    crop: tuple[int, int] | None,
    resize: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, total_width, channels = frame.shape
    if channels != 3:
        raise ValueError(
            f"Expected head stereo frame shape (H, W, 3), got {frame.shape}."
        )
    if total_width % 2 != 0:
        raise ValueError(
            f"Head stereo frame width {total_width} is not divisible by 2."
        )

    half_width = total_width // 2
    left = frame[:, :half_width, :]
    right = frame[:, half_width:, :]
    if crop is not None:
        left = center_crop(left, crop)
        right = center_crop(right, crop)
    if resize is not None:
        target_height, target_width = resize
        left = cv2.resize(
            left,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        right = cv2.resize(
            right,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        frame = np.concatenate((left, right), axis=1)
    elif crop is not None:
        frame = np.concatenate((left, right), axis=1)
    return frame, left, right


def write_head_stereo_rgb(
    h5_file: h5py.File,
    source_h5_file: h5py.File,
    *,
    crop: tuple[int, int] | None = None,
    resize: tuple[int, int] | None = None,
) -> None:
    jpeg_frames = source_h5_file["zed/frame_jpeg_bytes"]
    with tqdm(
        total=len(jpeg_frames),
        desc="Write images",
        unit="frame",
        leave=False,
    ) as progress:
        first_frame = decode_jpeg_rgb(jpeg_frames[0])
        first_head_stereo, first_left, _ = split_and_resize_head_stereo_frame(
            first_frame,
            crop,
            resize,
        )
        height, total_width, channels = first_head_stereo.shape
        _, half_width, _ = first_left.shape

        images_group = h5_file.create_group("images_dict")
        head_stereo_dataset = images_group.create_group(
            "head_stereo"
        ).create_dataset(
            "rgb",
            shape=(len(jpeg_frames), height, total_width, channels),
            dtype=np.uint8,
        )
        left_dataset = images_group.create_group("stereo_left").create_dataset(
            "rgb",
            shape=(len(jpeg_frames), height, half_width, channels),
            dtype=np.uint8,
        )
        right_dataset = images_group.create_group("stereo_right").create_dataset(
            "rgb",
            shape=(len(jpeg_frames), height, half_width, channels),
            dtype=np.uint8,
        )

        for idx in range(len(jpeg_frames)):
            raw_frame = first_frame if idx == 0 else decode_jpeg_rgb(jpeg_frames[idx])
            frame, left, right = split_and_resize_head_stereo_frame(
                raw_frame,
                crop,
                resize,
            )
            if frame.shape != first_head_stereo.shape:
                raise ValueError(
                    f"Frame {idx} shape {frame.shape} does not match "
                    f"first output frame shape {first_head_stereo.shape}."
                )
            head_stereo_dataset[idx] = frame
            left_dataset[idx] = left
            right_dataset[idx] = right
            progress.update()


def write_human_hand_dict(h5_file: h5py.File, retargeted: dict) -> None:
    human_hands = retargeted.get("human_hand_pose7_by_side") or {}
    if not human_hands:
        return

    human_hand_group = h5_file.create_group("human_hand_dict")
    for side in tqdm(
        HAND_SIDES,
        desc="Write human hands",
        unit="side",
        leave=False,
    ):
        if side not in human_hands:
            continue
        poses = np.asarray(human_hands[side], dtype=np.float64)
        if poses.ndim != 3 or poses.shape[1:] != (
            HAND_POINT_COUNT,
            HAND_POSE_DIM,
        ):
            raise ValueError(
                f"Invalid human hand pose shape for {side}: expected "
                f"(T, {HAND_POINT_COUNT}, {HAND_POSE_DIM}), got {poses.shape}."
            )
        human_hand_group.create_group(side).create_dataset(
            "raw_data",
            data=poses.reshape(poses.shape[0], -1),
        )


def write_output(
    path: Path,
    retargeted: dict,
    source_h5_file: h5py.File,
    *,
    write_images: bool = True,
    crop: tuple[int, int] | None = None,
    resize: tuple[int, int] | None = None,
) -> None:
    transforms = retargeted["transforms"]
    commands = retargeted["xhand_commands"]

    with h5py.File(path, "w") as h5_file:
        h5_file.create_dataset("time", data=retargeted["frame_times"])

        fk_poses = h5_file.create_group("fk_poses_dict_from_state")
        for dataset_name, target_name in tqdm(
            UPPER_BODY_TARGETS,
            desc="Write fk poses",
            unit="dataset",
            leave=False,
        ):
            fk_poses.create_dataset(
                dataset_name,
                data=transforms_to_pose7(transforms[target_name]),
            )

        dex_hand = h5_file.create_group("dex_hand_dict")
        for side in tqdm(
            HAND_SIDES,
            desc="Write dex hands",
            unit="side",
            leave=False,
        ):
            dex_hand.create_group(side).create_dataset(
                "state",
                data=np.asarray(
                    [command[side] for command in commands],
                    dtype=np.float64,
                ),
            )

        write_human_hand_dict(h5_file, retargeted)

        joints = h5_file.create_group("joints_dict")
        joints.create_dataset(
            "joints_position_command",
            data=np.asarray(retargeted["head_joint_command"], dtype=np.float64),
        )

        mobile = h5_file.create_group("mobile_dict")
        mobile.create_dataset(
            "absolute_position",
            data=np.asarray(retargeted["chassis_xyyaw"], dtype=np.float64),
        )

        if write_images:
            write_head_stereo_rgb(
                h5_file,
                source_h5_file,
                crop=crop,
                resize=resize,
            )


__all__ = [
    "HEAD_STEREO_RGB_KEY",
    "STEREO_LEFT_RGB_KEY",
    "STEREO_RIGHT_RGB_KEY",
    "UPPER_BODY_TARGETS",
    "build_retarget_frames",
    "center_crop",
    "decode_jpeg_rgb",
    "read_joint_index",
    "split_and_resize_head_stereo_frame",
    "write_head_stereo_rgb",
    "write_human_hand_dict",
    "write_output",
]
