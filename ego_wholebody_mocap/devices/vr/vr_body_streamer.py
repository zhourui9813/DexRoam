from __future__ import annotations

import asyncio
import time
from multiprocessing import Array, Process, Value, shared_memory
from typing import Dict, Sequence

import numpy as np

from ego_wholebody_mocap.config import (
    VR_IMAGE_HEIGHT,
    VR_IMAGE_POSITION,
)
from ego_wholebody_mocap.devices.vr.constants import (
    BODY_JOINT_NAMES,
    NUM_BODY_JOINTS,
    NUM_HAND_LANDMARKS,
)

try:
    from vuer import Vuer
    from vuer.schemas import Body, DefaultScene, Head, ImageBackground
except ImportError:
    Vuer = None
    Body = DefaultScene = Head = ImageBackground = None


def _coerce_matrix(matrix_raw: object) -> np.ndarray | None:
    matrix = np.asarray(matrix_raw, dtype=np.float64)
    if matrix.size != 16:
        return None
    if not np.isfinite(matrix).all():
        return None
    return matrix.reshape(16)


class VrBodyStreamer:
    def __init__(
        self,
        img_shape,
        shm_name,
        image_push_sleep_s: float = 0.03,
        image_aspect: float | None = None,
        vr_image_height: float | None = None,
        vr_image_position: Sequence[float] | None = None,
    ):
        if Vuer is None:
            raise RuntimeError("vuer is required to start VrBodyStreamer.")

        self.img_shape = (img_shape[0], 2 * img_shape[1], 3)
        self.img_height, self.img_width = img_shape[:2]
        self.image_aspect = float(image_aspect if image_aspect is not None else img_shape[1] / img_shape[0])
        self.vr_image_height = float(
            VR_IMAGE_HEIGHT if vr_image_height is None else vr_image_height
        )
        self.vr_image_position = [
            float(value)
            for value in (
                VR_IMAGE_POSITION if vr_image_position is None else vr_image_position
            )
        ]

        self.app = Vuer(host="0.0.0.0", queries=dict(grid=False), queue_len=3)

        self.app.add_handler("BODY_MOVE")(self.on_body_move)
        self.app.add_handler("HEAD_MOVE")(self.on_head_move)

        existing_shm = shared_memory.SharedMemory(name=shm_name)
        self.img_array = np.ndarray(
            (self.img_shape[0], self.img_shape[1], 3),
            dtype=np.uint8,
            buffer=existing_shm.buf,
        )
        self.app.spawn(start=False)(self.main_image)

        self.left_hand_key_points_world_shared = Array("d", NUM_HAND_LANDMARKS * 3, lock=True)
        self.right_hand_key_points_world_shared = Array("d", NUM_HAND_LANDMARKS * 3, lock=True)
        self.left_hand_matrices_shared = Array("d", NUM_HAND_LANDMARKS * 16, lock=True)
        self.right_hand_matrices_shared = Array("d", NUM_HAND_LANDMARKS * 16, lock=True)

        self.head_matrix_shared = Array("d", 16, lock=True)
        self.head_event_monotonic_ns_shared = Value("q", -1, lock=True)
        self.head_event_wall_ns_shared = Value("q", -1, lock=True)

        self.body_matrices_shared = Array("d", NUM_BODY_JOINTS * 16, lock=True)
        self.body_valid_shared = Array("i", NUM_BODY_JOINTS, lock=True)
        self.body_event_monotonic_ns_shared = Value("q", -1, lock=True)
        self.body_event_wall_ns_shared = Value("q", -1, lock=True)
        self.push_fps_shared = Value("d", 0.0, lock=True)
        self.image_push_sleep_s = max(0.0, float(image_push_sleep_s))

        self.process = Process(target=self.run)
        self.process.daemon = True
        self.process.start()

    def run(self):
        self.app.run()

    async def on_head_move(self, event, session):
        try:
            payload = event.value or {}
            matrix = _coerce_matrix(payload.get("matrix"))
            if matrix is not None:
                self.head_matrix_shared[:] = matrix
                self.head_event_monotonic_ns_shared.value = time.monotonic_ns()
                self.head_event_wall_ns_shared.value = time.time_ns()
        except Exception:
            pass

    async def on_body_move(self, event, session):
        try:
            payload = event.value or {}

            body = np.asarray(payload["body"], dtype=np.float64).reshape(NUM_BODY_JOINTS, 16)
            left = np.asarray(payload["leftHand"], dtype=np.float64).reshape(NUM_HAND_LANDMARKS, 16)
            right = np.asarray(payload["rightHand"], dtype=np.float64).reshape(NUM_HAND_LANDMARKS, 16)

            self.body_matrices_shared[:] = body.reshape(-1).tolist()
            self.body_valid_shared[:] = np.ones(NUM_BODY_JOINTS, dtype=np.int32).tolist()

            self.left_hand_matrices_shared[:] = left.reshape(-1).tolist()
            self.right_hand_matrices_shared[:] = right.reshape(-1).tolist()
            self.left_hand_key_points_world_shared[:] = left[:, 12:15].reshape(-1).tolist()
            self.right_hand_key_points_world_shared[:] = right[:, 12:15].reshape(-1).tolist()
            self.body_event_monotonic_ns_shared.value = time.monotonic_ns()
            self.body_event_wall_ns_shared.value = time.time_ns()
        except Exception as e:
            print(f"on_body_move error: {e}", flush=True)

    async def main_image(self, session, fps=60):
        session.set @ DefaultScene(frameloop="always")
        session.remove @ "hands"
        session.upsert(
            Body(
                key="body_tracking",
                stream=True,
                leftHand=True,
                rightHand=True,
                fps=fps,
                showBody=False,
                showFrame=False,
                frameScale=0.02,
            ),
            to="children",
        )
        session.upsert(
            Head(key="head_tracking", stream=True, fps=fps, show=False),
            to="children",
        )
        last_push_monotonic = None
        push_fps = 0.0
        while True:
            display_image = self.img_array
            session.upsert(
                [
                    ImageBackground(
                        display_image[::2, :self.img_width],
                        format="jpeg",
                        quality=80,
                        key="left-image",
                        interpolate=True,
                        aspect=self.image_aspect,
                        height=self.vr_image_height,
                        position=self.vr_image_position,
                        layers=1,
                    ),
                    ImageBackground(
                        display_image[::2, self.img_width:],
                        format="jpeg",
                        quality=80,
                        key="right-image",
                        interpolate=True,
                        aspect=self.image_aspect,
                        height=self.vr_image_height,
                        position=self.vr_image_position,
                        layers=2,
                    ),
                ],
                to="bgChildren",
            )
            now_monotonic = time.monotonic()
            if last_push_monotonic is not None:
                push_dt = now_monotonic - last_push_monotonic
                if push_dt > 0.0:
                    instant_fps = 1.0 / push_dt
                    push_fps = instant_fps if push_fps <= 0.0 else (0.9 * push_fps + 0.1 * instant_fps)
                    self.push_fps_shared.value = float(push_fps)
            last_push_monotonic = now_monotonic
            await asyncio.sleep(self.image_push_sleep_s)

    @property
    def left_hand_key_points_world(self):
        return np.array(self.left_hand_key_points_world_shared[:]).reshape(NUM_HAND_LANDMARKS, 3)

    @property
    def right_hand_key_points_world(self):
        return np.array(self.right_hand_key_points_world_shared[:]).reshape(NUM_HAND_LANDMARKS, 3)

    @property
    def left_hand_matrices(self):
        raw = np.array(self.left_hand_matrices_shared[:]).reshape(NUM_HAND_LANDMARKS, 16)
        return np.asarray([matrix.reshape(4, 4, order="F") for matrix in raw])

    @property
    def right_hand_matrices(self):
        raw = np.array(self.right_hand_matrices_shared[:]).reshape(NUM_HAND_LANDMARKS, 16)
        return np.asarray([matrix.reshape(4, 4, order="F") for matrix in raw])

    @property
    def head_matrix(self):
        return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")

    @property
    def head_event_monotonic_ns(self):
        return int(self.head_event_monotonic_ns_shared.value)

    @property
    def head_event_wall_ns(self):
        return int(self.head_event_wall_ns_shared.value)

    @property
    def body_event_monotonic_ns(self):
        return int(self.body_event_monotonic_ns_shared.value)

    @property
    def body_event_wall_ns(self):
        return int(self.body_event_wall_ns_shared.value)

    @property
    def push_fps(self):
        return float(self.push_fps_shared.value)

    @property
    def body_joint_names(self):
        return BODY_JOINT_NAMES

    @property
    def body_matrices(self) -> Dict[str, np.ndarray]:
        raw = np.array(self.body_matrices_shared[:])
        valid = np.array(self.body_valid_shared[:])
        result: Dict[str, np.ndarray] = {}
        for idx, name in enumerate(BODY_JOINT_NAMES):
            if valid[idx]:
                result[name] = raw[idx * 16: (idx + 1) * 16].reshape(4, 4, order="F")
        return result
