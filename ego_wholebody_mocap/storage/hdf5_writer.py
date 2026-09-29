from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from ego_wholebody_mocap.config import JPEG_QUALITY
from ego_wholebody_mocap.storage.schema import (
    WriterFinalResult,
    _schema_for_zed_storage_mode,
    _validate_zed_storage_mode,
)

try:
    import h5py
except ImportError:
    h5py = None


def require_h5py() -> None:
    if h5py is None:
        raise RuntimeError("h5py is not available in this environment.")


class EpisodeH5StreamWriter:
    def __init__(
        self,
        output_path: Path,
        frame_shape: tuple[int, int, int],
        joint_names: tuple[str, ...],
        zed_res: str,
        fps: int,
        stream_mode: str,
        zed_storage_mode: str,
    ) -> None:
        if h5py is None:
            raise RuntimeError("h5py is required to write whole-body HDF5 files.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self.frame_shape = frame_shape
        self.joint_names = joint_names
        self.zed_storage_mode = _validate_zed_storage_mode(zed_storage_mode)
        self._file = h5py.File(output_path, "w")

        meta = self._file.create_group("meta")
        meta.attrs["schema"] = _schema_for_zed_storage_mode(self.zed_storage_mode)
        meta.attrs["created_at"] = datetime.now().isoformat()
        meta.attrs["zed_res"] = zed_res
        meta.attrs["fps"] = int(fps)
        meta.attrs["stream_mode"] = stream_mode
        meta.attrs["frame_height"] = int(frame_shape[0])
        meta.attrs["frame_width"] = int(frame_shape[1])
        meta.attrs["zed_encoding"] = self.zed_storage_mode
        if self.zed_storage_mode == "jpeg":
            meta.attrs["jpeg_quality"] = int(JPEG_QUALITY)
        meta.attrs["capture_status"] = "recording"

        string_dtype = h5py.string_dtype(encoding="utf-8")

        head_group = self._file.create_group("head")
        self._head_sample_monotonic_ns = head_group.create_dataset(
            "sample_monotonic_ns",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=True,
        )
        self._head_sample_wall_ns = head_group.create_dataset(
            "sample_wall_ns",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=True,
        )
        self._head_event_monotonic_ns = head_group.create_dataset(
            "event_monotonic_ns",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=True,
        )
        self._head_event_wall_ns = head_group.create_dataset(
            "event_wall_ns",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=True,
        )
        self._head_matrix_f32 = head_group.create_dataset(
            "matrix_f32",
            shape=(0, 4, 4),
            maxshape=(None, 4, 4),
            dtype=np.float32,
            chunks=(1, 4, 4),
        )

        body_group = self._file.create_group("body")
        body_group.create_dataset(
            "joint_names",
            data=np.asarray(joint_names, dtype=string_dtype),
        )
        self._body_sample_monotonic_ns = body_group.create_dataset(
            "sample_monotonic_ns",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=True,
        )
        self._body_sample_wall_ns = body_group.create_dataset(
            "sample_wall_ns",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=True,
        )
        self._body_event_monotonic_ns = body_group.create_dataset(
            "event_monotonic_ns",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=True,
        )
        self._body_event_wall_ns = body_group.create_dataset(
            "event_wall_ns",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=True,
        )
        self._body_matrices_f32 = body_group.create_dataset(
            "matrices_f32",
            shape=(0, len(joint_names), 4, 4),
            maxshape=(None, len(joint_names), 4, 4),
            dtype=np.float32,
            chunks=(1, len(joint_names), 4, 4),
        )
        self._body_left_hand = body_group.create_dataset(
            "left_hand_key_points_world_f32",
            shape=(0, 25, 3),
            maxshape=(None, 25, 3),
            dtype=np.float32,
            chunks=(1, 25, 3),
        )
        self._body_right_hand = body_group.create_dataset(
            "right_hand_key_points_world_f32",
            shape=(0, 25, 3),
            maxshape=(None, 25, 3),
            dtype=np.float32,
            chunks=(1, 25, 3),
        )
        self._body_left_hand_matrices = body_group.create_dataset(
            "left_hand_matrices_f32",
            shape=(0, 25, 4, 4),
            maxshape=(None, 25, 4, 4),
            dtype=np.float32,
            chunks=(1, 25, 4, 4),
        )
        self._body_right_hand_matrices = body_group.create_dataset(
            "right_hand_matrices_f32",
            shape=(0, 25, 4, 4),
            maxshape=(None, 25, 4, 4),
            dtype=np.float32,
            chunks=(1, 25, 4, 4),
        )

        zed_group = self._file.create_group("zed")
        self._zed_capture_monotonic_ns = zed_group.create_dataset(
            "capture_monotonic_ns",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=True,
        )
        self._zed_capture_wall_ns = zed_group.create_dataset(
            "capture_wall_ns",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=True,
        )
        self._zed_image_ts_ns = zed_group.create_dataset(
            "image_ts_ns",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=True,
        )
        self._zed_frame_jpeg_bytes = None
        self._zed_frame_rgb_u8 = None
        if self.zed_storage_mode == "jpeg":
            self._zed_frame_jpeg_bytes = zed_group.create_dataset(
                "frame_jpeg_bytes",
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.vlen_dtype(np.dtype(np.uint8)),
                chunks=True,
            )
        else:
            self._zed_frame_rgb_u8 = zed_group.create_dataset(
                "frame_rgb_u8",
                shape=(0,) + frame_shape,
                maxshape=(None,) + frame_shape,
                dtype=np.uint8,
                chunks=(1,) + frame_shape,
            )

        self._head_count = 0
        self._body_count = 0
        self._zed_count = 0

    def append_head(
        self,
        sample_monotonic_ns: int,
        sample_wall_ns: int,
        event_monotonic_ns: int,
        event_wall_ns: int,
        matrix_f32: np.ndarray,
    ) -> None:
        idx = self._head_count
        self._head_sample_monotonic_ns.resize((idx + 1,))
        self._head_sample_wall_ns.resize((idx + 1,))
        self._head_event_monotonic_ns.resize((idx + 1,))
        self._head_event_wall_ns.resize((idx + 1,))
        self._head_matrix_f32.resize((idx + 1, 4, 4))

        self._head_sample_monotonic_ns[idx] = int(sample_monotonic_ns)
        self._head_sample_wall_ns[idx] = int(sample_wall_ns)
        self._head_event_monotonic_ns[idx] = int(event_monotonic_ns)
        self._head_event_wall_ns[idx] = int(event_wall_ns)
        self._head_matrix_f32[idx] = np.asarray(matrix_f32, dtype=np.float32)
        self._head_count += 1

    def append_body(
        self,
        sample_monotonic_ns: int,
        sample_wall_ns: int,
        event_monotonic_ns: int,
        event_wall_ns: int,
        matrices_f32: np.ndarray,
        left_hand_f32: np.ndarray,
        right_hand_f32: np.ndarray,
        left_hand_matrices_f32: np.ndarray,
        right_hand_matrices_f32: np.ndarray,
    ) -> None:
        idx = self._body_count
        left_hand_matrices = np.asarray(left_hand_matrices_f32, dtype=np.float32).reshape(25, 4, 4)
        right_hand_matrices = np.asarray(right_hand_matrices_f32, dtype=np.float32).reshape(25, 4, 4)

        self._body_sample_monotonic_ns.resize((idx + 1,))
        self._body_sample_wall_ns.resize((idx + 1,))
        self._body_event_monotonic_ns.resize((idx + 1,))
        self._body_event_wall_ns.resize((idx + 1,))
        self._body_matrices_f32.resize((idx + 1, len(self.joint_names), 4, 4))
        self._body_left_hand.resize((idx + 1, 25, 3))
        self._body_right_hand.resize((idx + 1, 25, 3))
        self._body_left_hand_matrices.resize((idx + 1, 25, 4, 4))
        self._body_right_hand_matrices.resize((idx + 1, 25, 4, 4))

        self._body_sample_monotonic_ns[idx] = int(sample_monotonic_ns)
        self._body_sample_wall_ns[idx] = int(sample_wall_ns)
        self._body_event_monotonic_ns[idx] = int(event_monotonic_ns)
        self._body_event_wall_ns[idx] = int(event_wall_ns)
        self._body_matrices_f32[idx] = np.asarray(matrices_f32, dtype=np.float32)
        self._body_left_hand[idx] = left_hand_matrices[:, :3, 3]
        self._body_right_hand[idx] = right_hand_matrices[:, :3, 3]
        self._body_left_hand_matrices[idx] = left_hand_matrices
        self._body_right_hand_matrices[idx] = right_hand_matrices
        self._body_count += 1

    def append_zed(
        self,
        capture_monotonic_ns: int,
        capture_wall_ns: int,
        image_ts_ns: int,
        *,
        jpeg_bytes: Optional[np.ndarray] = None,
        frame_rgb_u8: Optional[np.ndarray] = None,
    ) -> None:
        idx = self._zed_count
        self._zed_capture_monotonic_ns.resize((idx + 1,))
        self._zed_capture_wall_ns.resize((idx + 1,))
        self._zed_image_ts_ns.resize((idx + 1,))

        self._zed_capture_monotonic_ns[idx] = int(capture_monotonic_ns)
        self._zed_capture_wall_ns[idx] = int(capture_wall_ns)
        self._zed_image_ts_ns[idx] = int(image_ts_ns)
        if self.zed_storage_mode == "jpeg":
            if jpeg_bytes is None:
                raise ValueError("jpeg_bytes is required when zed_storage_mode='jpeg'.")
            assert self._zed_frame_jpeg_bytes is not None
            self._zed_frame_jpeg_bytes.resize((idx + 1,))
            self._zed_frame_jpeg_bytes[idx] = np.asarray(jpeg_bytes, dtype=np.uint8).reshape(-1)
        else:
            if frame_rgb_u8 is None:
                raise ValueError("frame_rgb_u8 is required when zed_storage_mode='raw_uint8'.")
            assert self._zed_frame_rgb_u8 is not None
            self._zed_frame_rgb_u8.resize((idx + 1,) + self.frame_shape)
            self._zed_frame_rgb_u8[idx] = np.asarray(frame_rgb_u8, dtype=np.uint8)
        self._zed_count += 1

    def finalize(
        self,
        capture_status: str,
        backlog_peak_bytes: int,
        overflow_reason: str = "",
        error_message: str = "",
    ) -> WriterFinalResult:
        self._file["meta"].attrs["capture_status"] = capture_status
        stats = self._file.create_group("stats")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        stats.create_dataset("head_count", data=np.int64(self._head_count))
        stats.create_dataset("body_count", data=np.int64(self._body_count))
        stats.create_dataset("zed_count", data=np.int64(self._zed_count))
        stats.create_dataset("backlog_peak_bytes", data=np.int64(backlog_peak_bytes))
        stats.create_dataset(
            "overflow_reason",
            data=np.asarray(overflow_reason or "", dtype=string_dtype),
        )
        self._file.flush()
        self._file.close()
        return WriterFinalResult(
            capture_status=capture_status,
            overflow_reason=overflow_reason,
            head_count=self._head_count,
            body_count=self._body_count,
            zed_count=self._zed_count,
            backlog_peak_bytes=int(backlog_peak_bytes),
            error_message=error_message,
        )
