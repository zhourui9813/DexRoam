"""Command-line entry point for offline Astribot retargeting."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from .runner import retarget_path


def parse_image_size(value: str) -> tuple[int, int]:
    try:
        height_text, width_text = value.lower().split("x", maxsplit=1)
        height = int(height_text)
        width = int(width_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"image size must be formatted as HEIGHTxWIDTH, got {value!r}"
        ) from exc

    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError(
            f"image dimensions must be positive, got {value!r}"
        )
    return height, width


def parse_speed_factor(value: str) -> float:
    try:
        speed_factor = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"speed factor must be a number, got {value!r}"
        ) from exc

    if not math.isfinite(speed_factor) or speed_factor <= 0.0:
        raise argparse.ArgumentTypeError(
            f"speed factor must be finite and greater than 0, got {value!r}"
        )
    return speed_factor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input HDF5 file or directory.")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input directories recursively.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-write-images",
        dest="write_images",
        action="store_false",
        help="Omit stereo RGB datasets from output HDF5 files.",
    )
    parser.add_argument("--crop", type=parse_image_size, metavar="HEIGHTxWIDTH")
    parser.add_argument("--resize", type=parse_image_size, metavar="HEIGHTxWIDTH")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Directory containing xhand_left.yml and xhand_right.yml.",
    )
    resample_group = parser.add_mutually_exclusive_group()
    resample_group.add_argument(
        "--robot-dir",
        type=Path,
        default=None,
        help=(
            "Robot HDF5 directory used to compute the average episode length "
            "for resampling."
        ),
    )
    resample_group.add_argument(
        "--target-length",
        type=int,
        default=None,
        help="Resample to this episode length instead of using --robot-dir.",
    )
    resample_group.add_argument(
        "--speed-factor",
        type=parse_speed_factor,
        default=None,
        help=(
            "Resample each episode by playback speed. For example, 2.0 "
            "produces about half as many frames and 0.5 produces about twice as many."
        ),
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return retarget_path(
        args.input,
        args.output,
        recursive=args.recursive,
        overwrite=args.overwrite,
        write_images=args.write_images,
        crop=args.crop,
        resize=args.resize,
        config_dir=args.config_dir,
        robot_dir=args.robot_dir,
        target_length=args.target_length,
        speed_factor=args.speed_factor,
    )


if __name__ == "__main__":
    raise SystemExit(main())
