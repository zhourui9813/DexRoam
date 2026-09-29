from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

from ego_wholebody_mocap.gestures.recognizer import GestureRecognizer, Pinch


class CaptureState(Enum):
    IDLE = auto()
    START_DELAY = auto()
    RECORDING = auto()
    AWAIT_DECISION = auto()
    SAVING = auto()


class CaptureAction(Enum):
    START_RECORDING = auto()
    STOP_RECORDING = auto()
    SAVE_EPISODE = auto()
    DISCARD_EPISODE = auto()
    CANCEL_START = auto()


@dataclass(frozen=True)
class CaptureConfig:
    debounce_s: float = 0.2
    start_delay_s: float = 1.0
    start_gesture: Pinch = Pinch.THUMB_INDEX
    stop_gesture: Pinch = Pinch.THUMB_MIDDLE
    save_gesture: Pinch = Pinch.THUMB_RING
    discard_gesture: Pinch = Pinch.THUMB_MIDDLE


@dataclass(frozen=True)
class CaptureUpdate:
    state: CaptureState
    action: Optional[CaptureAction]
    countdown_remaining_s: float
    status_text: str


@dataclass
class _DebouncedActivation:
    debounce_s: float
    stable_active: bool = False
    _pressed_since: Optional[float] = None

    def update(self, now: float, raw_active: bool) -> bool:
        if not raw_active:
            self.stable_active = False
            self._pressed_since = None
            return False

        if self.stable_active:
            return False

        if self._pressed_since is None:
            self._pressed_since = now
            return False

        if (now - self._pressed_since) >= self.debounce_s:
            self.stable_active = True
            return True

        return False


class GestureCaptureController:
    def __init__(
        self,
        config: Optional[CaptureConfig] = None,
        recognizer: Optional[GestureRecognizer] = None,
    ) -> None:
        self.config = config or CaptureConfig()
        self.recognizer = recognizer or GestureRecognizer()
        self.state = CaptureState.IDLE
        self._start_deadline: Optional[float] = None

        self._start_signal = _DebouncedActivation(self.config.debounce_s)
        self._stop_signal = _DebouncedActivation(self.config.debounce_s)
        self._save_signal = _DebouncedActivation(self.config.debounce_s)
        self._discard_signal = _DebouncedActivation(self.config.debounce_s)

    def set_state(self, state: CaptureState) -> None:
        self.state = state
        if state != CaptureState.START_DELAY:
            self._start_deadline = None

    def _reset_gesture_latches(self) -> None:
        for signal in (
            self._start_signal,
            self._stop_signal,
            self._save_signal,
            self._discard_signal,
        ):
            signal.stable_active = False
            signal._pressed_since = None

    def update(
        self,
        now_monotonic: float,
        left_hand_points: np.ndarray,
        right_hand_points: np.ndarray,
        saving_in_progress: bool = False,
    ) -> CaptureUpdate:
        action: Optional[CaptureAction] = None

        if saving_in_progress:
            self.state = CaptureState.SAVING
            self._start_deadline = None
            self._reset_gesture_latches()
            return self._build_update(action, now_monotonic)

        if self.state == CaptureState.SAVING:
            self.state = CaptureState.IDLE
            self._reset_gesture_latches()

        left_gestures = self.recognizer.detect(left_hand_points)
        right_gestures = self.recognizer.detect(right_hand_points)

        start_edge = self._start_signal.update(
            now_monotonic,
            left_gestures[self.config.start_gesture],
        )
        stop_edge = self._stop_signal.update(
            now_monotonic,
            left_gestures[self.config.stop_gesture],
        )
        save_edge = self._save_signal.update(
            now_monotonic,
            right_gestures[self.config.save_gesture],
        )
        discard_edge = self._discard_signal.update(
            now_monotonic,
            right_gestures[self.config.discard_gesture],
        )

        if self.state == CaptureState.IDLE:
            if start_edge:
                self.state = CaptureState.START_DELAY
                self._start_deadline = now_monotonic + self.config.start_delay_s

        elif self.state == CaptureState.START_DELAY:
            if stop_edge:
                self.state = CaptureState.IDLE
                self._start_deadline = None
                action = CaptureAction.CANCEL_START
            elif self._start_deadline is not None and now_monotonic >= self._start_deadline:
                self.state = CaptureState.RECORDING
                self._start_deadline = None
                action = CaptureAction.START_RECORDING

        elif self.state == CaptureState.RECORDING:
            if stop_edge:
                self.state = CaptureState.AWAIT_DECISION
                action = CaptureAction.STOP_RECORDING

        elif self.state == CaptureState.AWAIT_DECISION:
            if save_edge:
                self.state = CaptureState.SAVING
                action = CaptureAction.SAVE_EPISODE
            elif discard_edge:
                self.state = CaptureState.IDLE
                action = CaptureAction.DISCARD_EPISODE

        return self._build_update(action, now_monotonic)

    def _build_update(
        self,
        action: Optional[CaptureAction],
        now_monotonic: float,
    ) -> CaptureUpdate:
        countdown = 0.0
        if self.state == CaptureState.START_DELAY and self._start_deadline is not None:
            countdown = max(0.0, self._start_deadline - now_monotonic)

        return CaptureUpdate(
            state=self.state,
            action=action,
            countdown_remaining_s=countdown,
            status_text=self._status_text(countdown),
        )

    def _status_text(self, countdown_remaining_s: float) -> str:
        if self.state == CaptureState.IDLE:
            return "Ready"
        if self.state == CaptureState.START_DELAY:
            return f"Arming ({countdown_remaining_s:.1f}s)"
        if self.state == CaptureState.RECORDING:
            return "Recording"
        if self.state == CaptureState.AWAIT_DECISION:
            return "Right ring=save / right middle=discard"
        return "Waiting: Saving Data to Disk..."
