from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

try:
    import pyzed.sl as sl
except ImportError:
    sl = None


def require_zed() -> None:
    if sl is None:
        raise RuntimeError("pyzed.sl is not available in this environment.")


def zed_res_to_enum(name: str):
    require_zed()

    name = name.upper()
    return {
        "VGA": sl.RESOLUTION.VGA,
        "HD720": sl.RESOLUTION.HD720,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD2K": sl.RESOLUTION.HD2K,
    }[name]


def _zed_image_timestamp_ns(zed: "sl.Camera") -> int:
    if sl is None:
        return -1
    try:
        return int(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
    except Exception:
        return -1


def apply_zed_stream_frame(
    sbs_bgra: np.ndarray,
    tele_img: np.ndarray,
    per_eye_width: int,
    stream_mode: str,
) -> None:
    if stream_mode == "both":
        cv2.cvtColor(sbs_bgra, cv2.COLOR_BGRA2RGB, dst=tele_img)
        return

    if stream_mode == "left":
        eye_bgra = sbs_bgra[:, :per_eye_width, :]
    elif stream_mode == "right":
        eye_bgra = sbs_bgra[:, per_eye_width:2 * per_eye_width, :]
    else:
        raise ValueError(f"Unsupported stream mode: {stream_mode}")

    if not eye_bgra.flags["C_CONTIGUOUS"]:
        eye_bgra = np.ascontiguousarray(eye_bgra)

    eye_rgb = cv2.cvtColor(eye_bgra, cv2.COLOR_BGRA2RGB)
    tele_img[:, :per_eye_width, :] = eye_rgb
    tele_img[:, per_eye_width:2 * per_eye_width, :] = eye_rgb


class ZedCameraSession:
    def __init__(self, *, zed_res: str, fps: int) -> None:
        require_zed()
        self.camera: Optional["sl.Camera"] = sl.Camera()
        self.runtime: Optional["sl.RuntimeParameters"] = None
        self.mat_sbs: Optional["sl.Mat"] = None
        self.image_height = 0
        self.per_eye_width = 0

        init_params = sl.InitParameters()
        init_params.camera_resolution = zed_res_to_enum(zed_res)
        init_params.camera_fps = fps
        init_params.depth_mode = sl.DEPTH_MODE.NONE
        init_params.enable_image_enhancement = False

        err = self.camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            self.close()
            raise RuntimeError(f"ZED open failed: {err}")

        self.runtime = sl.RuntimeParameters()
        info = self.camera.get_camera_information()
        res = info.camera_configuration.resolution
        self.image_height, self.per_eye_width = res.height, res.width
        self.mat_sbs = sl.Mat(
            self.image_height,
            2 * self.per_eye_width,
            sl.MAT_TYPE.U8_C4,
            sl.MEM.CPU,
        )

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        return self.image_height, 2 * self.per_eye_width, 3

    def grab_side_by_side(self) -> Optional[np.ndarray]:
        assert self.camera is not None
        assert self.runtime is not None
        assert self.mat_sbs is not None
        if self.camera.grab(self.runtime) != sl.ERROR_CODE.SUCCESS:
            return None
        self.camera.retrieve_image(self.mat_sbs, sl.VIEW.SIDE_BY_SIDE)
        return self.mat_sbs.get_data()

    def image_timestamp_ns(self) -> int:
        if self.camera is None:
            return -1
        return _zed_image_timestamp_ns(self.camera)

    def close(self) -> None:
        if self.mat_sbs is not None:
            self.mat_sbs.free()
            self.mat_sbs = None
        if self.camera is not None:
            self.camera.close()
            self.camera = None
