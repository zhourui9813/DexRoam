"""HDF5 episode length estimation and frame-aligned resampling."""

from __future__ import annotations

import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm


HDF5_SUFFIXES = (".h5", ".hdf5")
ROBOT_LENGTH_CANDIDATES = (
    "time",
    "joints_dict/joints_position_state",
    "joints_dict/joints_position_command",
    "mobile_dict/relative_position",
    "mobile_dict/absolute_position",
    "dex_hand_dict/left/state",
    "dex_hand_dict/right/state",
    "fk_poses_dict_from_state/astribot_torso",
)
HUMAN_LENGTH_CANDIDATES = (
    "time",
    "fk_poses_dict_from_state/astribot_torso",
    "dex_hand_dict/left/state",
    "dex_hand_dict/right/state",
    "human_hand_dict/left/raw_data",
    "human_hand_dict/right/raw_data",
    "joints_dict/joints_position_command",
    "mobile_dict/absolute_position",
    "mobile_dict/relative_position",
)
RAW_ROBOT_TIMESTAMP_GROUPS = (
    "images_dict/head_rgbd/color",
    "images_dict/head_stereo/color",
    "images_dict/head/color",
)
TIMESTAMP_NAMES = {
    "time",
    "timestamp",
    "state_timestamp",
    "command_timestamp",
    "recv_timestamp",
    "rgb_timestamp",
    "frame_times",
}
ACTION_GROUP_NAMES = {"fk_poses_delta_action", "fk_poses_relative_action"}


class ResampleError(ValueError):
    """Raised when an HDF5 episode cannot be resampled safely."""


def natural_sort_key(value: Any) -> list[Any]:
    text = str(value)
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def is_path_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def collect_hdf5_files(
    input_path: Path,
    *,
    recursive: bool,
) -> tuple[Path, list[Path]]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in HDF5_SUFFIXES:
            raise ValueError(f"Input file is not an HDF5 file: {input_path}")
        return input_path.parent, [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    paths = [
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in HDF5_SUFFIXES
    ]
    paths.sort(key=lambda path: natural_sort_key(path.relative_to(input_path)))
    if not paths:
        raise FileNotFoundError(f"No HDF5 episodes found under: {input_path}")
    return input_path, paths


def _dataset_at(root: h5py.File | h5py.Group, dataset_path: str) -> h5py.Dataset | None:
    obj: h5py.File | h5py.Group | h5py.Dataset = root
    for part in dataset_path.split("/"):
        if not isinstance(obj, (h5py.File, h5py.Group)) or part not in obj:
            return None
        obj = obj[part]
    return obj if isinstance(obj, h5py.Dataset) else None


def detect_episode_length(
    root: h5py.File,
    candidates: tuple[str, ...],
    *,
    allow_raw_robot_fallback: bool,
) -> tuple[int, str]:
    for dataset_path in candidates:
        dataset = _dataset_at(root, dataset_path)
        if dataset is not None and dataset.ndim > 0 and dataset.shape[0] > 0:
            return int(dataset.shape[0]), dataset_path

    if allow_raw_robot_fallback:
        for group_path in RAW_ROBOT_TIMESTAMP_GROUPS:
            if group_path not in root:
                continue
            group_obj = root[group_path]
            if not isinstance(group_obj, h5py.Group):
                continue
            chunks = [
                item.shape[0]
                for name, item in group_obj.items()
                if re.fullmatch(r"timestamp_\d+", name)
                and isinstance(item, h5py.Dataset)
                and item.ndim > 0
            ]
            if chunks and sum(chunks) > 0:
                return sum(chunks), f"{group_path}/timestamp_*"

    raise ResampleError("Unable to detect episode step length")


def compute_robot_target_length(robot_files: list[Path]) -> tuple[int, dict[str, Any]]:
    lengths: list[int] = []
    failures: list[dict[str, str]] = []
    for path in robot_files:
        try:
            with h5py.File(path, "r") as root:
                length, source = detect_episode_length(
                    root,
                    ROBOT_LENGTH_CANDIDATES,
                    allow_raw_robot_fallback=True,
                )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        lengths.append(length)

    if not lengths:
        raise ResampleError("No valid robot episode lengths were detected.")
    values = np.asarray(lengths, dtype=np.float64)
    target_length = max(2, int(math.floor(float(values.mean()) + 0.5)))
    stats = {
        "files_total": len(robot_files),
        "files_valid": len(lengths),
        "files_failed": len(failures),
        "mean": float(values.mean()),
        "target_length": target_length,
        "median": float(np.median(values)),
        "min": int(values.min()),
        "max": int(values.max()),
        "failures": failures,
    }
    return target_length, stats


def validate_speed_factor(speed_factor: float) -> float:
    speed_factor = float(speed_factor)
    if not math.isfinite(speed_factor) or speed_factor <= 0.0:
        raise ValueError(
            f"speed_factor must be finite and greater than 0, got {speed_factor}"
        )
    return speed_factor


def compute_speed_target_length(source_length: int, speed_factor: float) -> int:
    if source_length <= 0:
        raise ValueError(f"source_length must be positive, got {source_length}")
    speed_factor = validate_speed_factor(speed_factor)
    scaled_length = float(source_length) / speed_factor
    if not math.isfinite(scaled_length):
        raise ValueError(
            f"speed_factor {speed_factor} produces a non-finite target length"
        )
    return max(2, int(math.floor(scaled_length + 0.5)))


def _validate_resample_mode(
    target_length: int | None,
    speed_factor: float | None,
) -> None:
    if (target_length is None) == (speed_factor is None):
        raise ValueError("Specify exactly one of target_length or speed_factor.")
    if target_length is not None and target_length < 2:
        raise ValueError(f"target_length must be at least 2, got {target_length}")
    if speed_factor is not None:
        validate_speed_factor(speed_factor)


def _source_axis(length: int) -> np.ndarray:
    if length <= 0:
        raise ResampleError("Cannot resample an empty episode")
    return np.linspace(0.0, 1.0, length, dtype=np.float64)


def _nearest_indices(source_length: int, target_length: int) -> np.ndarray:
    return np.rint(
        np.linspace(0.0, source_length - 1, target_length, dtype=np.float64)
    ).astype(np.int64)


def _linear_float(data: np.ndarray, target_length: int) -> np.ndarray:
    data = np.asarray(data)
    source_length = int(data.shape[0])
    if source_length == target_length:
        return data.copy()
    if source_length == 1:
        return np.repeat(data, target_length, axis=0)

    source_axis = _source_axis(source_length)
    target_axis = _source_axis(target_length)
    flat = data.reshape(source_length, -1).astype(np.float64, copy=False)
    output = np.empty((target_length, flat.shape[1]), dtype=np.float64)
    for column in range(flat.shape[1]):
        output[:, column] = np.interp(target_axis, source_axis, flat[:, column])
    return output.reshape((target_length,) + data.shape[1:])


def _timestamps(data: np.ndarray, target_length: int) -> np.ndarray:
    data = np.asarray(data)
    if data.shape[0] == target_length:
        return data.copy()
    if data.shape[0] == 1:
        return np.repeat(data, target_length, axis=0)
    return np.linspace(
        float(data[0]),
        float(data[-1]),
        target_length,
        dtype=np.float64,
    )


def _pose7(data: np.ndarray, target_length: int, dataset_path: str) -> np.ndarray:
    data = np.asarray(data)
    source_length = int(data.shape[0])
    if data.shape[-1] == 7:
        block_shape = data.shape[1:-1]
        blocks = data.reshape(source_length, -1, 7)
        output_shape = (target_length,) + block_shape + (7,)
    elif data.ndim == 2 and data.shape[1] % 7 == 0:
        blocks = data.reshape(source_length, -1, 7)
        output_shape = (target_length, data.shape[1])
    else:
        raise ResampleError(
            f"Dataset is not pose7-shaped: {dataset_path} {data.shape}"
        )

    if source_length == target_length:
        return data.copy()
    if source_length == 1:
        return np.repeat(blocks, target_length, axis=0).reshape(output_shape)

    source_axis = _source_axis(source_length)
    target_axis = _source_axis(target_length)
    output = np.empty((target_length, blocks.shape[1], 7), dtype=np.float64)
    for block_index in range(blocks.shape[1]):
        block = blocks[:, block_index, :].astype(np.float64, copy=False)
        for dimension in range(3):
            output[:, block_index, dimension] = np.interp(
                target_axis,
                source_axis,
                block[:, dimension],
            )
        quaternions = block[:, 3:7]
        norms = np.linalg.norm(quaternions, axis=1)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
            raise ResampleError(f"Invalid quaternion values in {dataset_path}")
        quaternions = quaternions / norms[:, None]
        for index in range(1, quaternions.shape[0]):
            if float(np.dot(quaternions[index - 1], quaternions[index])) < 0.0:
                quaternions[index] *= -1.0
        output[:, block_index, 3:7] = Slerp(
            source_axis,
            Rotation.from_quat(quaternions),
        )(target_axis).as_quat()
    return output.reshape(output_shape)


def _chassis_absolute(data: np.ndarray, target_length: int) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    if data.shape[0] == target_length:
        return data.copy()
    if data.shape[0] == 1:
        return np.repeat(data, target_length, axis=0)
    source_axis = _source_axis(data.shape[0])
    target_axis = _source_axis(target_length)
    output = np.empty((target_length, 3), dtype=np.float64)
    output[:, :2] = np.column_stack(
        [
            np.interp(target_axis, source_axis, data[:, dimension])
            for dimension in range(2)
        ]
    )
    output[:, 2] = np.interp(target_axis, source_axis, np.unwrap(data[:, 2]))
    return output


def _copy_attrs(source: h5py.Group | h5py.Dataset, target: h5py.Group | h5py.Dataset) -> None:
    for key, value in source.attrs.items():
        target.attrs[key] = value


def _dataset_kwargs(source: h5py.Dataset, shape: tuple[int, ...]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if source.chunks is not None and shape and all(dimension > 0 for dimension in shape):
        kwargs["chunks"] = tuple(
            min(chunk, dimension)
            for chunk, dimension in zip(source.chunks, shape)
        )
    if source.compression is not None:
        kwargs["compression"] = source.compression
        if source.compression_opts is not None:
            kwargs["compression_opts"] = source.compression_opts
    if source.shuffle:
        kwargs["shuffle"] = True
    if source.fletcher32:
        kwargs["fletcher32"] = True
    if source.scaleoffset is not None:
        kwargs["scaleoffset"] = source.scaleoffset
    return kwargs


def _write_array(
    target_group: h5py.Group,
    name: str,
    data: np.ndarray,
    source: h5py.Dataset,
) -> None:
    data = np.asarray(data, dtype=source.dtype)
    target = target_group.create_dataset(
        name,
        data=data,
        dtype=source.dtype,
        **_dataset_kwargs(source, data.shape),
    )
    _copy_attrs(source, target)


def _write_nearest(
    target_group: h5py.Group,
    name: str,
    source: h5py.Dataset,
    target_length: int,
) -> None:
    target_shape = (target_length,) + tuple(source.shape[1:])
    target = target_group.create_dataset(
        name,
        shape=target_shape,
        dtype=source.dtype,
        **_dataset_kwargs(source, target_shape),
    )
    _copy_attrs(source, target)
    for target_index, source_index in enumerate(
        _nearest_indices(int(source.shape[0]), target_length)
    ):
        target[target_index] = source[int(source_index)]


def _is_timestamp(dataset_path: str) -> bool:
    name = dataset_path.rsplit("/", maxsplit=1)[-1]
    return name in TIMESTAMP_NAMES or name.endswith("_timestamp")


def _is_image(dataset_path: str) -> bool:
    return dataset_path == "images_dict" or dataset_path.startswith("images_dict/")


def _is_action(dataset_path: str) -> bool:
    return dataset_path in ACTION_GROUP_NAMES or dataset_path.startswith(
        tuple(f"{name}/" for name in ACTION_GROUP_NAMES)
    )


def _is_pose7(dataset_path: str, source: h5py.Dataset) -> bool:
    if source.dtype.kind != "f" or source.ndim < 2 or _is_timestamp(dataset_path):
        return False
    if dataset_path.startswith("human_hand_dict/"):
        return source.ndim == 2 and source.shape[1] % 7 == 0
    prefixes = (
        "fk_poses_dict_from_state/",
        "poses_dict/",
        "command_poses_dict/",
    )
    if not dataset_path.startswith(prefixes):
        return False
    return source.shape[-1] == 7 or (
        source.ndim == 2 and source.shape[1] % 7 == 0
    )


def _copy_or_resample_dataset(
    source: h5py.Dataset,
    target_group: h5py.Group,
    name: str,
    dataset_path: str,
    source_length: int,
    target_length: int,
) -> None:
    frame_aligned = source.ndim > 0 and int(source.shape[0]) == source_length
    if not frame_aligned:
        _write_array(target_group, name, source[()], source)
    elif dataset_path == "mobile_dict/absolute_position":
        _write_array(target_group, name, _chassis_absolute(source[()], target_length), source)
    elif _is_image(dataset_path) or source.dtype.kind not in ("f", "c"):
        _write_nearest(target_group, name, source, target_length)
    elif _is_pose7(dataset_path, source):
        _write_array(target_group, name, _pose7(source[()], target_length, dataset_path), source)
    elif _is_timestamp(dataset_path):
        _write_array(target_group, name, _timestamps(source[()], target_length), source)
    else:
        _write_array(target_group, name, _linear_float(source[()], target_length), source)


def _copy_or_resample_group(
    source_group: h5py.Group,
    target_group: h5py.Group,
    source_length: int,
    target_length: int,
    *,
    skip_images: bool,
    group_path: str = "",
) -> None:
    _copy_attrs(source_group, target_group)
    for name, item in source_group.items():
        item_path = f"{group_path}/{name}" if group_path else name
        if _is_action(item_path) or (skip_images and _is_image(item_path)):
            continue
        if isinstance(item, h5py.Group):
            child = target_group.create_group(name)
            _copy_or_resample_group(
                item,
                child,
                source_length,
                target_length,
                skip_images=skip_images,
                group_path=item_path,
            )
        else:
            _copy_or_resample_dataset(
                item,
                target_group,
                name,
                item_path,
                source_length,
                target_length,
            )


def resample_hdf5_file(
    input_path: Path,
    output_path: Path,
    target_length: int | None = None,
    *,
    speed_factor: float | None = None,
    overwrite: bool = False,
    skip_images: bool = False,
) -> str:
    _validate_resample_mode(target_length, speed_factor)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Resampling input and output paths must be different.")
    if output_path.exists() and not overwrite:
        return "skipped"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(
        tempfile.mkstemp(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
        )[1]
    )
    try:
        with h5py.File(input_path, "r") as source_root:
            source_length, source_name = detect_episode_length(
                source_root,
                HUMAN_LENGTH_CANDIDATES,
                allow_raw_robot_fallback=False,
            )
            if target_length is not None:
                resolved_target_length = target_length
            else:
                assert speed_factor is not None
                resolved_target_length = compute_speed_target_length(
                    source_length,
                    speed_factor,
                )
            with h5py.File(temporary_path, "w") as target_root:
                _copy_or_resample_group(
                    source_root,
                    target_root,
                    source_length,
                    resolved_target_length,
                    skip_images=skip_images,
                )
                target_root.attrs["resampled_to_robot_avg_episode_length"] = int(
                    resolved_target_length
                )
                target_root.attrs["resample_target_episode_length"] = int(
                    resolved_target_length
                )
                target_root.attrs["resample_source_episode_length"] = int(source_length)
                target_root.attrs["resample_source_length_dataset"] = source_name
                if speed_factor is not None:
                    target_root.attrs["resample_speed_factor"] = float(speed_factor)
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    return "processed"


def resample_directory(
    input_dir: Path,
    output_dir: Path,
    target_length: int | None = None,
    *,
    speed_factor: float | None = None,
    recursive: bool = False,
    overwrite: bool = False,
    skip_images: bool = False,
) -> int:
    _validate_resample_mode(target_length, speed_factor)
    input_base, input_files = collect_hdf5_files(input_dir, recursive=recursive)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped = 0
    failed = 0
    for input_path in tqdm(input_files, desc="Resample", unit="file"):
        relative_path = input_path.relative_to(input_base)
        output_path = output_dir / relative_path
        try:
            status = resample_hdf5_file(
                input_path,
                output_path,
                target_length,
                speed_factor=speed_factor,
                overwrite=overwrite,
                skip_images=skip_images,
            )
        except Exception as exc:  # noqa: BLE001
            tqdm.write(f"[FAIL] {relative_path}: {type(exc).__name__}: {exc}")
            failed += 1
            continue
        if status == "skipped":
            tqdm.write(f"[SKIP] {relative_path}")
            skipped += 1
        else:
            processed += 1
    mode_summary = (
        f"target_length={target_length}"
        if target_length is not None
        else f"speed_factor={speed_factor}"
    )
    print(
        "Resample summary: "
        f"processed={processed}, skipped={skipped}, failed={failed}, "
        f"{mode_summary}"
    )
    return 1 if failed else 0


__all__ = [
    "ResampleError",
    "collect_hdf5_files",
    "compute_robot_target_length",
    "compute_speed_target_length",
    "detect_episode_length",
    "is_path_relative_to",
    "resample_directory",
    "resample_hdf5_file",
    "validate_speed_factor",
]
