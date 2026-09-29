from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "data" / "test_data" / f"vuer_teleop_jpeg_{stamp}.h5"


def _resolve_output_target(output_arg: Path) -> tuple[Path, Optional[str]]:
    output_path = output_arg.expanduser().resolve()
    if output_path.suffix.lower() in {".h5", ".hdf5"}:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path.parent, output_path.stem

    output_path.mkdir(parents=True, exist_ok=True)
    return output_path, None


def _episode_path(output_dir: Path, prefix: Optional[str], episode_idx: int, suffix: str) -> Path:
    if prefix is None:
        filename = f"episode{episode_idx:04d}{suffix}"
    else:
        filename = f"{prefix}_episode{episode_idx:04d}{suffix}"
    return output_dir / filename
