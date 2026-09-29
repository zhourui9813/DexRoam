from __future__ import annotations

import argparse
import time
from multiprocessing import shared_memory
from typing import Sequence

import numpy as np

from ego_wholebody_mocap.config import (
    VR_IMAGE_HEIGHT,
    VR_IMAGE_POSITION,
)
from ego_wholebody_mocap.devices.vr.preprocessor import VuerPreprocessor
from ego_wholebody_mocap.devices.vr.vr_body_streamer import VrBodyStreamer


class VuerRecorder:
    def __init__(
        self,
        resolution=(720, 1280),
        stream_mode="image",
        cert_file=None,
        key_file=None,
        ngrok=False,
        image_push_sleep_s: float = 0.03,
        image_aspect: float | None = None,
        vr_image_height: float | None = None,
        vr_image_position: Sequence[float] | None = None,
    ):
        self.resolution = resolution
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (
            self.resolution[0] - self.crop_size_h,
            self.resolution[1] - 2 * self.crop_size_w,
        )

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]
        self.vr_image_height = float(
            VR_IMAGE_HEIGHT if vr_image_height is None else vr_image_height
        )
        self.vr_image_position = tuple(
            float(value)
            for value in (
                VR_IMAGE_POSITION if vr_image_position is None else vr_image_position
            )
        )

        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray(
            (self.img_shape[0], self.img_shape[1], 3),
            dtype=np.uint8,
            buffer=self.shm.buf,
        )
        self.vr_streamer = VrBodyStreamer(
            self.resolution_cropped,
            self.shm.name,
            image_push_sleep_s=image_push_sleep_s,
            image_aspect=image_aspect,
            vr_image_height=self.vr_image_height,
            vr_image_position=self.vr_image_position,
        )
        self.processor = VuerPreprocessor()

    def step(self):
        return self.processor.process(self.vr_streamer)

    @property
    def body_matrices(self):
        return self.processor.body_matrices

    def close(self):
        for attr in ("process", "webrtc_process"):
            proc = getattr(self.vr_streamer, attr, None)
            if proc is not None and proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)

        try:
            self.shm.close()
        finally:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass


VuerTeleop = VuerRecorder


def _draw_stream_test_frame(img_array: np.ndarray, frame_idx: int) -> None:
    height, total_width, _ = img_array.shape
    half_width = total_width // 2

    x = np.arange(half_width, dtype=np.uint16)[None, :]
    y = np.arange(height, dtype=np.uint16)[:, None]

    left = img_array[:, :half_width]
    right = img_array[:, half_width:]

    left[..., 0] = (x + frame_idx) % 256
    left[..., 1] = (y * 2 + frame_idx) % 256
    left[..., 2] = ((x // 2 + y // 3) + frame_idx * 3) % 256

    right[..., 0] = (255 - left[..., 1]) % 256
    right[..., 1] = (255 - left[..., 2]) % 256
    right[..., 2] = (255 - left[..., 0]) % 256

    bar_width = max(4, half_width // 40)
    left_bar = frame_idx % max(1, half_width - bar_width)
    right_bar = half_width - bar_width - left_bar

    left[:, left_bar:left_bar + bar_width] = (255, 255, 255)
    right[:, right_bar:right_bar + bar_width] = (255, 255, 255)

    center_band = max(2, height // 80)
    center_y = height // 2
    left[center_y - center_band:center_y + center_band] = (255, 32, 32)
    right[center_y - center_band:center_y + center_band] = (32, 160, 255)


def main():
    parser = argparse.ArgumentParser(description="Minimal image-streaming test for VuerTeleop.")
    parser.add_argument("--height", type=int, default=720, help="Per-eye image height.")
    parser.add_argument("--width", type=int, default=1280, help="Per-eye image width.")
    parser.add_argument("--fps", type=float, default=30.0, help="Synthetic test pattern FPS.")
    parser.add_argument("--ngrok", action="store_true", help="Start Vuer without local cert/key.")
    parser.add_argument("--cert-file", default=None, help="Optional TLS cert for local HTTPS.")
    parser.add_argument("--key-file", default=None, help="Optional TLS key for local HTTPS.")
    args = parser.parse_args()

    teleop = VuerRecorder(
        resolution=(args.height, args.width),
        stream_mode="image",
        cert_file=args.cert_file,
        key_file=args.key_file,
        ngrok=args.ngrok,
    )

    frame_idx = 0
    frame_period = 1.0 / max(args.fps, 1e-6)

    print(
        f"Streaming synthetic frames at {args.fps:.1f} FPS with resolution "
        f"{args.height}x{args.width} per eye. Press Ctrl+C to stop.",
        flush=True,
    )
    print(
        f"VR image plane: height={teleop.vr_streamer.vr_image_height:.3f}, "
        f"position={tuple(teleop.vr_streamer.vr_image_position)}",
        flush=True,
    )

    try:
        while True:
            _draw_stream_test_frame(teleop.img_array, frame_idx)
            frame_idx += 1
            time.sleep(frame_period)
    except KeyboardInterrupt:
        pass
    finally:
        teleop.close()


if __name__ == "__main__":
    main()
