from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ego_wholebody_mocap.config import (
    ABORT_DISCARD,
    ABORT_OVERFLOW,
    FRAME_SLOTS_PER_SLAB,
    INITIAL_FRAME_SLABS,
)
from ego_wholebody_mocap.paths import _episode_path
from ego_wholebody_mocap.storage.async_writer import AsyncEpisodeWriteBroker
from ego_wholebody_mocap.storage.schema import (
    WriterFinalResult,
    _validate_zed_storage_mode,
)


@dataclass
class ActiveEpisode:
    idx: int
    broker: AsyncEpisodeWriteBroker
    started_monotonic: float


@dataclass
class PendingEpisode:
    idx: int
    pending_path: Path
    final_path: Path
    result: WriterFinalResult
    duration_s: float


@dataclass
class EpisodeCaptureManager:
    output_dir: Path
    output_prefix: Optional[str]
    backlog_budget_bytes: int
    zed_storage_mode: str = "jpeg"
    slots_per_slab: int = FRAME_SLOTS_PER_SLAB
    initial_slabs: int = INITIAL_FRAME_SLABS
    saved_episode_counter: int = 1
    active: Optional[ActiveEpisode] = None
    pending: Optional[PendingEpisode] = None

    def has_pending_episode(self) -> bool:
        return self.pending is not None

    def has_active_episode(self) -> bool:
        return self.active is not None

    def start_recording(
        self,
        frame_shape: tuple[int, int, int],
        body_joint_names: tuple[str, ...],
        zed_res: str,
        fps: int,
        stream_mode: str,
        zed_storage_mode: str,
    ) -> Optional[int]:
        if self.has_active_episode() or self.has_pending_episode():
            return None

        episode_idx = self.saved_episode_counter
        started_monotonic = time.monotonic()
        self.zed_storage_mode = _validate_zed_storage_mode(zed_storage_mode)
        pending_path = _episode_path(self.output_dir, self.output_prefix, episode_idx, ".pending.h5")
        final_path = _episode_path(self.output_dir, self.output_prefix, episode_idx, ".h5")
        error_path = _episode_path(self.output_dir, self.output_prefix, episode_idx, ".error.h5")
        broker = AsyncEpisodeWriteBroker(
            pending_path=pending_path,
            final_path=final_path,
            error_path=error_path,
            frame_shape=frame_shape,
            joint_names=body_joint_names,
            zed_res=zed_res,
            fps=fps,
            stream_mode=stream_mode,
            zed_storage_mode=self.zed_storage_mode,
            backlog_budget_bytes=self.backlog_budget_bytes,
            slots_per_slab=self.slots_per_slab,
            initial_slabs=self.initial_slabs,
        )
        self.active = ActiveEpisode(
            idx=episode_idx,
            broker=broker,
            started_monotonic=started_monotonic,
        )
        return episode_idx

    def stop_recording(self) -> tuple[Optional[int], Optional[WriterFinalResult]]:
        if self.active is None:
            return None, None

        active = self.active
        result = active.broker.stop()
        self.active = None
        duration_seconds = max(0.0, time.monotonic() - active.started_monotonic)

        if result.capture_status == "stopped":
            self.pending = PendingEpisode(
                idx=active.idx,
                pending_path=_episode_path(self.output_dir, self.output_prefix, active.idx, ".pending.h5"),
                final_path=_episode_path(self.output_dir, self.output_prefix, active.idx, ".h5"),
                result=result,
                duration_s=duration_seconds,
            )
            return active.idx, result

        self._move_to_error(active.idx)
        return active.idx, result

    def abort_active(self, abort_code: int) -> tuple[Optional[int], Optional[WriterFinalResult], Optional[Path]]:
        if self.active is None:
            return None, None, None

        active = self.active
        result = active.broker.abort(abort_code)
        self.active = None

        if abort_code == ABORT_OVERFLOW or result.capture_status == "writer_error":
            error_path = self._move_to_error(active.idx)
            return active.idx, result, error_path

        pending_path = _episode_path(self.output_dir, self.output_prefix, active.idx, ".pending.h5")
        pending_path.unlink(missing_ok=True)
        return active.idx, result, None

    def poll_active_failure(self) -> Optional[WriterFinalResult]:
        if self.active is None:
            return None
        result = self.active.broker.poll_final_result()
        if result is not None and result.capture_status != "stopped":
            return result
        return None

    def _move_to_error(self, episode_idx: int) -> Path:
        pending_path = _episode_path(self.output_dir, self.output_prefix, episode_idx, ".pending.h5")
        error_path = _episode_path(self.output_dir, self.output_prefix, episode_idx, ".error.h5")
        error_path.parent.mkdir(parents=True, exist_ok=True)
        if pending_path.exists():
            pending_path.replace(error_path)
        return error_path

    def save_pending(self) -> tuple[Optional[int], Optional[Path], Optional[WriterFinalResult]]:
        if self.pending is None:
            return None, None, None

        pending = self.pending
        pending.final_path.parent.mkdir(parents=True, exist_ok=True)
        pending.pending_path.replace(pending.final_path)
        self.pending = None
        self.saved_episode_counter += 1
        return pending.idx, pending.final_path, pending.result

    def discard_pending(self) -> tuple[Optional[int], Optional[Path]]:
        if self.pending is None:
            return None, None

        pending = self.pending
        pending_path = pending.pending_path
        pending_path.unlink(missing_ok=True)
        self.pending = None
        return pending.idx, pending_path

    def discard_unsaved(self) -> list[str]:
        messages: list[str] = []

        if self.has_active_episode():
            episode_idx, result, _ = self.abort_active(ABORT_DISCARD)
            if episode_idx is not None:
                messages.append(
                    f"[exit] Discarded active episode {episode_idx} before save "
                    f"(head={result.head_count if result else 0}, "
                    f"body={result.body_count if result else 0}, "
                    f"zed={result.zed_count if result else 0})."
                )

        if self.has_pending_episode():
            episode_idx, pending_path = self.discard_pending()
            if episode_idx is not None:
                messages.append(f"[exit] Discarded pending episode {episode_idx} ({pending_path}).")

        return messages

    def display_episode_text(self) -> str:
        if self.active is not None:
            return f"{self.active.idx:04d}"
        if self.pending is not None:
            return f"{self.pending.idx:04d}"
        return f"{self.saved_episode_counter:04d}"

    def display_episode_duration_text(self, now_monotonic: float) -> str:
        duration_seconds = 0.0
        if self.active is not None:
            duration_seconds = max(0.0, now_monotonic - self.active.started_monotonic)
        elif self.pending is not None:
            duration_seconds = max(0.0, self.pending.duration_s)

        total_seconds = int(duration_seconds)
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"
