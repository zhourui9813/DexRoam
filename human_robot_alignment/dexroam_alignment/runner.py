"""Execution layer for single-file and directory retargeting tasks."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
from tqdm import tqdm

from .hands import build_xhand_retargeting_by_side
from .io import build_retarget_frames
from .io import read_joint_index
from .io import write_output
from .pipeline import retarget_sequence
from .resampling import collect_hdf5_files
from .resampling import compute_robot_target_length
from .resampling import is_path_relative_to
from .resampling import resample_directory
from .resampling import resample_hdf5_file
from .resampling import validate_speed_factor


OUTPUT_SUFFIX = "_astribot_retargeted"


def default_output_path(input_path: Path) -> Path:
    if input_path.is_dir():
        return input_path.with_name(f"{input_path.name}{OUTPUT_SUFFIX}")
    return input_path.with_name(f"{input_path.stem}{OUTPUT_SUFFIX}.hdf5")


def iter_input_files(input_dir: Path, recursive: bool = False) -> list[Path]:
    candidates = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(
        path
        for path in candidates
        if path.is_file()
        and path.suffix.lower() in (".h5", ".hdf5")
        and not path.stem.endswith(OUTPUT_SUFFIX)
    )


def output_path_for(
    input_path: Path,
    input_dir: Path,
    output_dir: Path,
) -> Path:
    relative_path = input_path.relative_to(input_dir)
    return output_dir / relative_path.parent / f"{input_path.stem}{OUTPUT_SUFFIX}.hdf5"


def retarget_file(
    input_path: Path,
    output_path: Path,
    xhand_retargeting_by_side,
    *,
    overwrite: bool = False,
    write_images: bool = True,
    crop: tuple[int, int] | None = None,
    resize: tuple[int, int] | None = None,
) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output HDF5 paths must be different.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_path, "r") as h5_file:
        frames = build_retarget_frames(h5_file)
        retargeted = retarget_sequence(
            tqdm(frames, desc="Retarget frames", unit="frame", leave=False),
            joint_index=read_joint_index(h5_file),
            xhand_retargeting_by_side=xhand_retargeting_by_side,
        )
        write_output(
            output_path,
            retargeted,
            h5_file,
            write_images=write_images,
            crop=crop,
            resize=resize,
        )


def retarget_directory(
    input_dir: Path,
    output_dir: Path,
    xhand_retargeting_by_side,
    *,
    recursive: bool = False,
    overwrite: bool = False,
    write_images: bool = True,
    crop: tuple[int, int] | None = None,
    resize: tuple[int, int] | None = None,
) -> int:
    input_files = iter_input_files(input_dir, recursive=recursive)
    if not input_files:
        raise FileNotFoundError(f"No HDF5 episodes found under: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped = 0
    failed = []
    for input_path in tqdm(input_files, desc="Retarget", unit="file"):
        output_path = output_path_for(input_path, input_dir, output_dir)
        if output_path.exists() and not overwrite:
            tqdm.write(f"[SKIP] {input_path.relative_to(input_dir)}")
            skipped += 1
            continue
        try:
            retarget_file(
                input_path,
                output_path,
                xhand_retargeting_by_side,
                overwrite=overwrite,
                write_images=write_images,
                crop=crop,
                resize=resize,
            )
        except Exception as exc:  # noqa: BLE001
            tqdm.write(
                f"[FAIL] {input_path.relative_to(input_dir)}: "
                f"{type(exc).__name__}: {exc}"
            )
            failed.append((input_path, exc))
            continue
        processed += 1

    print(f"Summary: processed={processed}, skipped={skipped}, failed={len(failed)}")
    return 1 if failed else 0


def retarget_path(
    input_path: Path,
    output_path: Path | None = None,
    *,
    recursive: bool = False,
    overwrite: bool = False,
    write_images: bool = True,
    crop: tuple[int, int] | None = None,
    resize: tuple[int, int] | None = None,
    config_dir: Path | None = None,
    robot_dir: Path | None = None,
    target_length: int | None = None,
    speed_factor: float | None = None,
) -> int:
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_path = (
        output_path.expanduser().resolve()
        if output_path is not None
        else default_output_path(input_path)
    )
    resample_option_count = sum(
        option is not None for option in (robot_dir, target_length, speed_factor)
    )
    if resample_option_count > 1:
        raise ValueError(
            "Specify only one of robot_dir, target_length, or speed_factor."
        )
    if target_length is not None and target_length < 2:
        raise ValueError(f"target_length must be at least 2, got {target_length}")
    if speed_factor is not None:
        speed_factor = validate_speed_factor(speed_factor)

    resample_target = target_length
    if robot_dir is not None:
        robot_base, robot_files = collect_hdf5_files(
            robot_dir,
            recursive=recursive,
        )
        resample_target, robot_stats = compute_robot_target_length(robot_files)
        print(
            "Robot episode lengths: "
            f"valid={robot_stats['files_valid']}/{robot_stats['files_total']} "
            f"mean={robot_stats['mean']:.3f} "
            f"target={resample_target}"
        )
        if robot_stats["files_failed"]:
            print(
                "[WARN] Failed to detect lengths for "
                f"{robot_stats['files_failed']} robot file(s)."
            )
        if not robot_base.is_dir():
            raise ValueError(f"Robot path must resolve to a directory: {robot_base}")

    resampling_requested = resample_target is not None or speed_factor is not None

    if input_path.is_dir() and resampling_requested:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if is_path_relative_to(output_path, input_path):
            raise ValueError(
                f"Output directory must be outside input directory: {output_path}"
            )
        if robot_dir is not None and is_path_relative_to(output_path, robot_base):
            raise ValueError(
                f"Output directory must be outside robot directory: {output_path}"
            )

    retargeting = build_xhand_retargeting_by_side(config_dir=config_dir)

    if input_path.is_dir():
        if resampling_requested:
            with TemporaryDirectory(
                prefix=".dexroam-retarget-",
                dir=output_path.parent,
            ) as temporary_dir:
                temporary_path = Path(temporary_dir)
                retarget_result = retarget_directory(
                    input_path,
                    temporary_path,
                    retargeting,
                    recursive=recursive,
                    overwrite=True,
                    write_images=write_images,
                    crop=crop,
                    resize=resize,
                )
                if retarget_result:
                    return retarget_result
                return resample_directory(
                    temporary_path,
                    output_path,
                    resample_target,
                    speed_factor=speed_factor,
                    recursive=recursive,
                    overwrite=overwrite,
                    skip_images=not write_images,
                )
        return retarget_directory(
            input_path,
            output_path,
            retargeting,
            recursive=recursive,
            overwrite=overwrite,
            write_images=write_images,
            crop=crop,
            resize=resize,
        )

    if not resampling_requested:
        retarget_file(
            input_path,
            output_path,
            retargeting,
            overwrite=overwrite,
            write_images=write_images,
            crop=crop,
            resize=resize,
        )
    else:
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"{output_path} already exists. Use --overwrite.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            prefix=".dexroam-retarget-",
            dir=output_path.parent,
        ) as temporary_dir:
            intermediate_path = Path(temporary_dir) / "retargeted.hdf5"
            retarget_file(
                input_path,
                intermediate_path,
                retargeting,
                overwrite=True,
                write_images=write_images,
                crop=crop,
                resize=resize,
            )
            resample_hdf5_file(
                intermediate_path,
                output_path,
                resample_target,
                speed_factor=speed_factor,
                overwrite=overwrite,
                skip_images=not write_images,
            )
    print(f"Wrote {output_path}")
    return 0


__all__ = [
    "OUTPUT_SUFFIX",
    "default_output_path",
    "iter_input_files",
    "output_path_for",
    "retarget_directory",
    "retarget_file",
    "retarget_path",
]
