from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ego_wholebody_mocap.gestures.capture_state_machine import CaptureState


def _status_name(update_state: CaptureState, has_pending_episode: bool) -> str:
    if update_state == CaptureState.AWAIT_DECISION:
        return "AwaitDecision"
    if update_state == CaptureState.SAVING:
        return "Saving"
    if has_pending_episode:
        return "AwaitDecision"
    if update_state == CaptureState.IDLE:
        return "Ready"
    if update_state == CaptureState.START_DELAY:
        return "Arming"
    if update_state == CaptureState.RECORDING:
        return "Recording"
    return "Ready"


def draw_status_on_tele(
    tele_img: np.ndarray,
    per_eye_w: int,
    status_text: str,
    episode_text: str,
    duration_text: str,
    loop_fps_text: str,
    push_fps_text: str,
) -> None:
    height = tele_img.shape[0]
    font_scale = max(0.6, height / 720.0 * 0.9)
    thickness = 2
    pad_x = max(20, per_eye_w // 24)
    pad_y = max(16, height // 32)
    line_gap = max(14, height // 30)
    lines = (
        (f"Status: {status_text}", (255, 255, 0)),
        (f"Episode: {episode_text}", (0, 255, 255)),
        (f"Duration: {duration_text}", (255, 220, 120)),
        (f"Loop FPS: {loop_fps_text}", (0, 220, 120)),
        (f"Push FPS: {push_fps_text}", (255, 180, 80)),
    )

    for x0 in (0, per_eye_w):
        roi = tele_img[:, x0:x0 + per_eye_w]
        try:
            metrics = [
                cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                for text, _ in lines
            ]
            panel_w = max(size[0][0] for size in metrics) + 2 * pad_x
            panel_h = (
                sum(size[0][1] + size[1] for size in metrics)
                + line_gap * (len(lines) - 1)
                + 2 * pad_y
            )
            panel_x0 = max(0, (per_eye_w - panel_w) // 2)
            panel_y0 = max(0, (height - panel_h) // 2)
            panel_x1 = min(per_eye_w, panel_x0 + panel_w)
            panel_y1 = min(height, panel_y0 + panel_h)

            panel = roi.copy()
            cv2.rectangle(panel, (panel_x0, panel_y0), (panel_x1, panel_y1), (0, 0, 0), -1)
            cv2.addWeighted(panel, 0.42, roi, 0.58, 0.0, dst=roi)

            line_y = panel_y0 + pad_y
            for (text, color), ((text_w, text_h), text_base) in zip(lines, metrics):
                text_x = panel_x0 + (panel_w - text_w) // 2
                line_y += text_h
                cv2.putText(
                    roi,
                    text,
                    (text_x, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
                line_y += text_base + line_gap
        except cv2.error:
            pass


def _display_status(
    gesture_state: CaptureState,
    gesture_status_text: str,
    has_pending_episode: bool,
    feedback_text: Optional[str],
    now_monotonic: float,
    feedback_deadline: float,
) -> str:
    if feedback_text is not None and now_monotonic < feedback_deadline:
        return feedback_text

    if gesture_state == CaptureState.START_DELAY:
        return gesture_status_text
    if gesture_state == CaptureState.RECORDING:
        return "Recording"
    if gesture_state == CaptureState.SAVING:
        return "Saving"
    if has_pending_episode or gesture_state == CaptureState.AWAIT_DECISION:
        return "AwaitDecision"
    return "Ready"
