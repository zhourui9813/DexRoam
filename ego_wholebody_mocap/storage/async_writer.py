from __future__ import annotations

import multiprocessing as mp
from collections import deque
from multiprocessing import shared_memory
from pathlib import Path
from queue import Empty
from typing import Optional

import cv2
import numpy as np

from ego_wholebody_mocap._compat import _resource_tracker_unregister
from ego_wholebody_mocap.config import (
    ABORT_NONE,
    ABORT_OVERFLOW,
    FRAME_SLOTS_PER_SLAB,
    INITIAL_FRAME_SLABS,
    JPEG_QUALITY,
)
from ego_wholebody_mocap.storage.hdf5_writer import EpisodeH5StreamWriter, h5py
from ego_wholebody_mocap.storage.schema import (
    WriterFinalResult,
    _validate_zed_storage_mode,
)


def _body_matrices_array(
    body_matrices: dict[str, np.ndarray],
    joint_names: tuple[str, ...],
) -> np.ndarray:
    matrices = np.full((len(joint_names), 4, 4), np.nan, dtype=np.float32)
    for idx, joint_name in enumerate(joint_names):
        matrix = body_matrices.get(joint_name)
        if matrix is not None:
            matrices[idx] = np.asarray(matrix, dtype=np.float32)
    return matrices


def _register_slab(
    *,
    message: dict,
    frame_shape: tuple[int, int, int],
    slab_handles: dict[int, shared_memory.SharedMemory],
    slab_arrays: dict[int, np.ndarray],
) -> None:
    slab_index = int(message["slab_index"])
    slot_count = int(message["slot_count"])
    shm = shared_memory.SharedMemory(name=message["shm_name"])
    if _resource_tracker_unregister is not None:
        try:
            _resource_tracker_unregister(shm._name, "shared_memory")
        except Exception:
            pass
    slab_handles[slab_index] = shm
    slab_arrays[slab_index] = np.ndarray(
        (slot_count,) + frame_shape,
        dtype=np.uint8,
        buffer=shm.buf,
    )


def _append_head_message(writer: EpisodeH5StreamWriter, message: dict) -> None:
    writer.append_head(
        sample_monotonic_ns=message["sample_monotonic_ns"],
        sample_wall_ns=message["sample_wall_ns"],
        event_monotonic_ns=message["event_monotonic_ns"],
        event_wall_ns=message["event_wall_ns"],
        matrix_f32=message["matrix_f32"], #head pose (T,4,4)
    )


def _append_body_message(writer: EpisodeH5StreamWriter, message: dict) -> None:
    writer.append_body(
        sample_monotonic_ns=message["sample_monotonic_ns"],
        sample_wall_ns=message["sample_wall_ns"],
        event_monotonic_ns=message["event_monotonic_ns"],
        event_wall_ns=message["event_wall_ns"],
        matrices_f32=message["matrices_f32"], #body joints(T,33,4,4)
        left_hand_f32=message["left_hand_key_points_world_f32"],
        right_hand_f32=message["right_hand_key_points_world_f32"],
        left_hand_matrices_f32=message["left_hand_matrices_f32"], # left hand pose(T,25,4,4)
        right_hand_matrices_f32=message["right_hand_matrices_f32"], # right hand pose(T,25,4,4)
    )


def _append_zed_message(
    *,
    writer: EpisodeH5StreamWriter,
    message: dict,
    slab_arrays: dict[int, np.ndarray],
    free_slot_queue: mp.Queue,
    slots_per_slab: int,
    zed_storage_mode: str,
) -> None:
    slot_id = int(message["slot_id"])
    slab_index = slot_id // slots_per_slab
    slot_offset = slot_id % slots_per_slab
    frame_rgb_u8 = slab_arrays[slab_index][slot_offset]
    try:
        if zed_storage_mode == "jpeg":
            ok, encoded = cv2.imencode(
                ".jpg",
                frame_rgb_u8,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)],
            )
            if not ok:
                raise RuntimeError("OpenCV failed to encode a ZED frame as JPEG.")
            writer.append_zed(
                capture_monotonic_ns=message["capture_monotonic_ns"],
                capture_wall_ns=message["capture_wall_ns"],
                image_ts_ns=message["image_ts_ns"],
                jpeg_bytes=encoded,
            )
        else:
            writer.append_zed(
                capture_monotonic_ns=message["capture_monotonic_ns"],
                capture_wall_ns=message["capture_wall_ns"],
                image_ts_ns=message["image_ts_ns"],
                frame_rgb_u8=frame_rgb_u8,
            )
    finally:
        free_slot_queue.put(slot_id)


def _async_episode_writer_main(
    output_path: str,
    frame_shape: tuple[int, int, int],
    joint_names: tuple[str, ...],
    zed_res: str,
    fps: int,
    stream_mode: str,
    zed_storage_mode: str,
    command_queue: mp.Queue,
    free_slot_queue: mp.Queue,
    status_queue: mp.Queue,
    abort_event: mp.Event,
    abort_code: mp.Value,
    slots_per_slab: int,
    backlog_peak_bytes: mp.Value,
) -> None:
    slab_handles: dict[int, shared_memory.SharedMemory] = {}
    slab_arrays: dict[int, np.ndarray] = {}
    writer: Optional[EpisodeH5StreamWriter] = None
    capture_status = "writer_error"
    overflow_reason = ""
    error_message = ""

    try:
        writer = EpisodeH5StreamWriter(
            output_path=Path(output_path),
            frame_shape=frame_shape,
            joint_names=joint_names,
            zed_res=zed_res,
            fps=fps,
            stream_mode=stream_mode,
            zed_storage_mode=zed_storage_mode,
        )

        while True:
            if abort_event.is_set():
                code = int(abort_code.value)
                if code == ABORT_OVERFLOW:
                    capture_status = "overflow_error"
                    overflow_reason = "backlog_budget_exceeded"
                else:
                    capture_status = "discarded"
                break

            try:
                message = command_queue.get(timeout=0.05)
            except Empty:
                continue

            message_type = message["type"]
            if message_type == "register_slab":
                _register_slab(
                    message=message,
                    frame_shape=frame_shape,
                    slab_handles=slab_handles,
                    slab_arrays=slab_arrays,
                )
            elif message_type == "head":
                _append_head_message(writer, message)
            elif message_type == "body":
                _append_body_message(writer, message)
            elif message_type == "zed":
                _append_zed_message(
                    writer=writer,
                    message=message,
                    slab_arrays=slab_arrays,
                    free_slot_queue=free_slot_queue,
                    slots_per_slab=slots_per_slab,
                    zed_storage_mode=zed_storage_mode,
                )
            elif message_type == "stop":
                capture_status = "stopped"
                break
    except Exception as exc:
        capture_status = "writer_error"
        error_message = str(exc)
    finally:
        try:
            if writer is not None:
                result = writer.finalize(
                    capture_status=capture_status,
                    backlog_peak_bytes=int(backlog_peak_bytes.value),
                    overflow_reason=overflow_reason,
                    error_message=error_message,
                )
            else:
                result = WriterFinalResult(
                    capture_status=capture_status,
                    overflow_reason=overflow_reason,
                    head_count=0,
                    body_count=0,
                    zed_count=0,
                    backlog_peak_bytes=int(backlog_peak_bytes.value),
                    error_message=error_message,
                )
            status_queue.put(result)
        finally:
            for shm in slab_handles.values():
                try:
                    shm.close()
                except FileNotFoundError:
                    pass


class AsyncEpisodeWriteBroker:
    def __init__(
        self,
        pending_path: Path,
        final_path: Path,
        error_path: Path,
        frame_shape: tuple[int, int, int],
        joint_names: tuple[str, ...],
        zed_res: str,
        fps: int,
        stream_mode: str,
        zed_storage_mode: str,
        backlog_budget_bytes: int,
        slots_per_slab: int = FRAME_SLOTS_PER_SLAB,
        initial_slabs: int = INITIAL_FRAME_SLABS,
    ) -> None:
        if h5py is None:
            raise RuntimeError("h5py is required to write whole-body HDF5 files.")

        self.pending_path = pending_path
        self.final_path = final_path
        self.error_path = error_path
        self.frame_shape = frame_shape
        self.joint_names = joint_names
        self.zed_res = zed_res
        self.fps = int(fps)
        self.stream_mode = stream_mode
        self.zed_storage_mode = _validate_zed_storage_mode(zed_storage_mode)
        self.backlog_budget_bytes = int(backlog_budget_bytes)
        self.slots_per_slab = int(slots_per_slab)
        self.initial_slabs = int(initial_slabs)
        self.frame_bytes = int(np.prod(frame_shape)) * np.dtype(np.uint8).itemsize
        self.max_slots = max(1, self.backlog_budget_bytes // self.frame_bytes)
        self.command_queue: mp.Queue = mp.Queue()
        self.free_slot_queue: mp.Queue = mp.Queue()
        self.status_queue: mp.Queue = mp.Queue()
        self.abort_event: mp.Event = mp.Event()
        self.abort_code: mp.Value = mp.Value("i", ABORT_NONE)
        self.backlog_peak_bytes: mp.Value = mp.Value("q", 0)
        self._slabs: list[shared_memory.SharedMemory] = []
        self._slab_arrays: list[np.ndarray] = []
        self._free_slots: deque[int] = deque()
        self._next_slot_id = 0
        self._occupied_slot_count = 0
        self._final_result: Optional[WriterFinalResult] = None

        self.process = mp.Process(
            target=_async_episode_writer_main,
            args=(
                str(self.pending_path),
                self.frame_shape,
                self.joint_names,
                self.zed_res,
                self.fps,
                self.stream_mode,
                self.zed_storage_mode,
                self.command_queue,
                self.free_slot_queue,
                self.status_queue,
                self.abort_event,
                self.abort_code,
                self.slots_per_slab,
                self.backlog_peak_bytes,
            ),
        )
        self.process.daemon = True
        self.process.start()

        for _ in range(self.initial_slabs):
            if not self._allocate_slab():
                break

    def _allocate_slab(self) -> bool:
        if (len(self._slabs) + 1) * self.slots_per_slab > self.max_slots:
            return False

        slab_index = len(self._slabs)
        shm = shared_memory.SharedMemory(create=True, size=self.slots_per_slab * self.frame_bytes)
        if _resource_tracker_unregister is not None:
            try:
                _resource_tracker_unregister(shm._name, "shared_memory")
            except Exception:
                pass
        slab_array = np.ndarray(
            (self.slots_per_slab,) + self.frame_shape,
            dtype=np.uint8,
            buffer=shm.buf,
        )
        self._slabs.append(shm)
        self._slab_arrays.append(slab_array)
        self.command_queue.put(
            {
                "type": "register_slab",
                "slab_index": slab_index,
                "slot_count": self.slots_per_slab,
                "shm_name": shm.name,
            }
        )
        for slot_offset in range(self.slots_per_slab):
            self._free_slots.append(self._next_slot_id + slot_offset)
        self._next_slot_id += self.slots_per_slab
        return True

    def _drain_released_slots(self) -> None:
        while True:
            try:
                slot_id = int(self.free_slot_queue.get_nowait())
            except Empty:
                break
            self._free_slots.append(slot_id)
            self._occupied_slot_count = max(0, self._occupied_slot_count - 1)

    def _update_backlog_peak(self) -> None:
        current_bytes = self._occupied_slot_count * self.frame_bytes
        if current_bytes > int(self.backlog_peak_bytes.value):
            self.backlog_peak_bytes.value = int(current_bytes)

    def submit_head(
        self,
        sample_monotonic_ns: int,
        sample_wall_ns: int,
        head_event_monotonic_ns: int,
        head_event_wall_ns: int,
        head_mat: np.ndarray,
    ) -> None:
        self.command_queue.put(
            {
                "type": "head",
                "sample_monotonic_ns": int(sample_monotonic_ns),
                "sample_wall_ns": int(sample_wall_ns),
                "event_monotonic_ns": int(head_event_monotonic_ns),
                "event_wall_ns": int(head_event_wall_ns),
                "matrix_f32": np.asarray(head_mat, dtype=np.float32),
            }
        )

    def submit_body(
        self,
        sample_monotonic_ns: int,
        sample_wall_ns: int,
        body_event_monotonic_ns: int,
        body_event_wall_ns: int,
        body_matrices: dict[str, np.ndarray],
        left_hand_key_points_world: np.ndarray,
        right_hand_key_points_world: np.ndarray,
        left_hand_matrices_world: np.ndarray,
        right_hand_matrices_world: np.ndarray,
    ) -> None:
        self.command_queue.put(
            {
                "type": "body",
                "sample_monotonic_ns": int(sample_monotonic_ns),
                "sample_wall_ns": int(sample_wall_ns),
                "event_monotonic_ns": int(body_event_monotonic_ns),
                "event_wall_ns": int(body_event_wall_ns),
                "matrices_f32": _body_matrices_array(body_matrices, self.joint_names),
                "left_hand_key_points_world_f32": np.asarray(left_hand_key_points_world, dtype=np.float32),
                "right_hand_key_points_world_f32": np.asarray(right_hand_key_points_world, dtype=np.float32),
                "left_hand_matrices_f32": np.asarray(left_hand_matrices_world, dtype=np.float32),
                "right_hand_matrices_f32": np.asarray(right_hand_matrices_world, dtype=np.float32),
            }
        )

    def submit_zed_frame(
        self,
        frame_rgb_u8: np.ndarray,
        capture_monotonic_ns: int,
        capture_wall_ns: int,
        image_ts_ns: int,
    ) -> bool:
        self._drain_released_slots()
        if not self._free_slots and not self._allocate_slab():
            return False
        if not self._free_slots:
            return False

        slot_id = self._free_slots.popleft()
        slab_index = slot_id // self.slots_per_slab
        slot_offset = slot_id % self.slots_per_slab
        np.copyto(self._slab_arrays[slab_index][slot_offset], np.asarray(frame_rgb_u8, dtype=np.uint8))
        self._occupied_slot_count += 1
        self._update_backlog_peak()
        self.command_queue.put(
            {
                "type": "zed",
                "slot_id": int(slot_id),
                "capture_monotonic_ns": int(capture_monotonic_ns),
                "capture_wall_ns": int(capture_wall_ns),
                "image_ts_ns": int(image_ts_ns),
            }
        )
        return True

    def poll_final_result(self) -> Optional[WriterFinalResult]:
        if self._final_result is not None:
            return self._final_result
        try:
            result = self.status_queue.get_nowait()
        except Empty:
            return None
        self._final_result = result
        return result

    def stop(self) -> WriterFinalResult:
        self.command_queue.put({"type": "stop"})
        return self._join_process()

    def abort(self, abort_code: int) -> WriterFinalResult:
        self.abort_code.value = int(abort_code)
        self.abort_event.set()
        return self._join_process()

    def _join_process(self) -> WriterFinalResult:
        self.process.join()
        result = self.poll_final_result()
        if result is None:
            result = WriterFinalResult(
                capture_status="writer_error",
                overflow_reason="",
                head_count=0,
                body_count=0,
                zed_count=0,
                backlog_peak_bytes=int(self.backlog_peak_bytes.value),
                error_message="writer exited without final status",
            )
        self._cleanup_shared_memory()
        for queue_obj in (self.command_queue, self.free_slot_queue, self.status_queue):
            try:
                queue_obj.close()
                queue_obj.join_thread()
            except Exception:
                pass
        return result

    def _cleanup_shared_memory(self) -> None:
        while self._slabs:
            shm = self._slabs.pop()
            try:
                shm.close()
            finally:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass
        self._slab_arrays.clear()
