from __future__ import annotations

from dataclasses import dataclass

from ego_wholebody_mocap.config import STATUS_FEEDBACK_SECONDS


@dataclass
class RuntimeDisplayState:
    last_status_name: str | None = None
    feedback_text: str | None = None
    feedback_deadline: float = 0.0
    last_loop_monotonic: float | None = None
    loop_fps: float = 0.0


def _update_loop_fps(display_state: RuntimeDisplayState, now_monotonic: float) -> None:
    if display_state.last_loop_monotonic is not None:
        loop_dt = now_monotonic - display_state.last_loop_monotonic
        if loop_dt > 0.0:
            instant_fps = 1.0 / loop_dt
            display_state.loop_fps = (
                instant_fps
                if display_state.loop_fps <= 0.0
                else (0.9 * display_state.loop_fps + 0.1 * instant_fps)
            )
    display_state.last_loop_monotonic = now_monotonic


def _set_status_feedback(
    display_state: RuntimeDisplayState,
    text: str,
    now_monotonic: float,
) -> None:
    display_state.feedback_text = text
    display_state.feedback_deadline = now_monotonic + STATUS_FEEDBACK_SECONDS
