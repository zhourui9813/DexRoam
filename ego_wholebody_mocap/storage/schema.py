from __future__ import annotations

from dataclasses import dataclass

WHOLEBODY_ASYNC_JPEG_SCHEMA = "wholebody-async-jpeg-v1"
WHOLEBODY_ASYNC_RAW_SCHEMA = "wholebody-async-raw-v1"
_ZED_STORAGE_JPEG = "jpeg"
_ZED_STORAGE_MODES = (_ZED_STORAGE_JPEG, "raw_uint8")


@dataclass
class WriterFinalResult:
    capture_status: str
    overflow_reason: str
    head_count: int
    body_count: int
    zed_count: int
    backlog_peak_bytes: int
    error_message: str = ""


def _validate_zed_storage_mode(zed_storage_mode: str) -> str:
    if zed_storage_mode not in _ZED_STORAGE_MODES:
        raise ValueError(
            f"Unsupported zed storage mode: {zed_storage_mode!r}. "
            f"Expected one of {_ZED_STORAGE_MODES!r}."
        )
    return zed_storage_mode


def _schema_for_zed_storage_mode(zed_storage_mode: str) -> str:
    zed_storage_mode = _validate_zed_storage_mode(zed_storage_mode)
    if zed_storage_mode == _ZED_STORAGE_JPEG:
        return WHOLEBODY_ASYNC_JPEG_SCHEMA
    return WHOLEBODY_ASYNC_RAW_SCHEMA
