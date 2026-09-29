"""Human-hand conversion and XHand joint retargeting."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from .constants import ASTRIBOT_SDK_PINOCCHIO_PATH_MARKER
from .constants import DEFAULT_DEX_RETARGETING_CONFIG_DIR
from .constants import DEXHAND_COMMAND_DIM
from .constants import DEX_RETARGETING_SRC
from .constants import HAND_POINT_COUNT
from .constants import HAND_POSE_DIM
from .constants import HAND_SIDES
from .constants import VR_TO_MANUS_ROT_MATS
from .constants import XHAND_JOINT_SUFFIXES
from .constants import XHAND_URDF_DIR


def ensure_local_dex_retargeting() -> None:
    """Prefer the repository's pinned dex-retargeting implementation."""
    sys.path[:] = [
        path
        for path in sys.path
        if ASTRIBOT_SDK_PINOCCHIO_PATH_MARKER not in path
    ]
    src = str(DEX_RETARGETING_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


def xhand_command_joint_names(side: str) -> list[str]:
    return [f"{side}_hand_{suffix}" for suffix in XHAND_JOINT_SUFFIXES]


def map_xhand_command_order(retargeting: Any, side: str) -> np.ndarray:
    retarget_joint_names = list(retargeting.joint_names)
    command_joint_names = xhand_command_joint_names(side)
    missing = [
        name
        for name in command_joint_names
        if name not in retarget_joint_names
    ]
    if missing:
        raise ValueError(
            f"Retargeting output for {side} is missing command joint(s): "
            f"{missing}. Available joints: {retarget_joint_names}"
        )
    return np.asarray(
        [retarget_joint_names.index(name) for name in command_joint_names],
        dtype=np.int64,
    )


def build_xhand_retargeting_by_side(
    config_dir: Path | str | None = None,
) -> dict[str, tuple[Any, np.ndarray]]:
    config_root = (
        DEFAULT_DEX_RETARGETING_CONFIG_DIR
        if config_dir is None
        else Path(config_dir).expanduser().resolve()
    )
    ensure_local_dex_retargeting()

    from dex_retargeting.retargeting_config import RetargetingConfig

    RetargetingConfig.set_default_urdf_dir(XHAND_URDF_DIR)
    retargeting_by_side = {}
    for side in HAND_SIDES:
        config_path = config_root / f"xhand_{side}.yml"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing retargeting config: {config_path}")
        retargeting = RetargetingConfig.load_from_file(config_path).build()
        retargeting_by_side[side] = (
            retargeting,
            map_xhand_command_order(retargeting, side),
        )
    return retargeting_by_side


def make_xhand_retarget_ref_value(retargeting: Any, pose: np.ndarray) -> np.ndarray:
    indices = np.asarray(
        retargeting.optimizer.target_link_human_indices,
        dtype=np.int64,
    )
    retargeting_type = str(retargeting.optimizer.retargeting_type).upper()
    if retargeting_type == "POSITION":
        if indices.ndim != 1:
            raise ValueError(
                "POSITION retargeting expects 1D target_link_human_indices, "
                f"got {indices.shape}."
            )
        ref_value = pose[indices, :3]
    elif retargeting_type in ("VECTOR", "DEXPILOT"):
        if indices.ndim != 2 or indices.shape[0] != 2:
            raise ValueError(
                f"{retargeting_type} retargeting expects target_link_human_indices "
                f"with shape (2, N), got {indices.shape}."
            )
        origin_indices, task_indices = indices
        ref_value = pose[task_indices, :3] - pose[origin_indices, :3]
    else:
        raise ValueError(f"Unsupported XHand retargeting type: {retargeting_type!r}.")
    return np.ascontiguousarray(ref_value.astype(np.float32, copy=False))


def retarget_xhand_pose7_by_side(
    hand_pose7_by_side: Mapping[str, np.ndarray],
    xhand_retargeting_by_side: Mapping[str, tuple[Any, np.ndarray]],
) -> dict[str, list[float]]:
    commands = {}
    for side, pose7 in hand_pose7_by_side.items():
        pose = np.asarray(pose7, dtype=np.float64)
        if pose.shape == (HAND_POINT_COUNT * HAND_POSE_DIM,):
            pose = pose.reshape(HAND_POINT_COUNT, HAND_POSE_DIM)
        if pose.shape != (HAND_POINT_COUNT, HAND_POSE_DIM):
            raise ValueError(
                f"Invalid hand pose for {side}: expected "
                f"({HAND_POINT_COUNT}, {HAND_POSE_DIM}) or "
                f"({HAND_POINT_COUNT * HAND_POSE_DIM},), got {pose.shape}."
            )
        if side not in xhand_retargeting_by_side:
            raise ValueError(f"Unsupported XHand side: {side!r}.")
        retargeting, command_indices = xhand_retargeting_by_side[side]
        ref_value = make_xhand_retarget_ref_value(retargeting, pose)
        qpos = np.asarray(retargeting.retarget(ref_value)).reshape(-1)
        if qpos.shape[0] <= int(np.max(command_indices)):
            raise ValueError(
                f"Invalid XHand qpos for {side}: expected at least "
                f"{int(np.max(command_indices)) + 1} values, got {qpos.shape}."
            )
        command = np.rad2deg(qpos[command_indices]).reshape(-1)
        if command.shape != (DEXHAND_COMMAND_DIM,):
            raise ValueError(
                f"Invalid XHand command for {side}: expected "
                f"{DEXHAND_COMMAND_DIM} values, got {command.shape}."
            )
        commands[side] = command.tolist()
    return commands


def vr_matrices_to_manus(matrices: np.ndarray, side: str) -> np.ndarray:
    """Convert local VR hand matrices to the Manus coordinate convention."""
    rot = VR_TO_MANUS_ROT_MATS[side]
    converted = np.asarray(matrices, dtype=np.float64).copy()
    converted[:, :3, :3] = np.matmul(
        np.matmul(rot, converted[:, :3, :3]),
        rot.T,
    )
    converted[:, :3, 3] = converted[:, :3, 3] @ rot.T
    return converted


def smooth_hand_quaternion_frame(
    quaternions: np.ndarray,
    previous_quaternions: np.ndarray | None,
    side: str,
) -> np.ndarray:
    """Choose quaternion signs consistently with the previous frame."""
    smoothed = np.asarray(quaternions, dtype=np.float64).copy()
    if previous_quaternions is not None:
        dots = np.sum(
            np.asarray(previous_quaternions, dtype=np.float64) * smoothed,
            axis=-1,
        )
        smoothed[dots < 0.0] *= -1.0
    return smoothed


def convert_hand_frame_to_manus(
    hand_matrices: np.ndarray,
    side: str,
    previous_quaternions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one frame of world-space hand matrices into Manus pose7."""
    matrices = np.asarray(hand_matrices, dtype=np.float64)
    root_inv = np.linalg.inv(matrices[0])
    local_matrices = vr_matrices_to_manus(np.matmul(root_inv, matrices), side)
    rotations = local_matrices[:, :3, :3]
    translations = local_matrices[:, :3, 3]

    quaternions = Rotation.from_matrix(rotations).as_quat()
    quaternions = quaternions / np.linalg.norm(quaternions, axis=-1)[:, None]
    quaternions = smooth_hand_quaternion_frame(
        quaternions,
        previous_quaternions,
        side,
    )
    return np.concatenate([translations, quaternions], axis=-1), quaternions.copy()


_smooth_hand_quaternion_frame = smooth_hand_quaternion_frame


__all__ = [
    "build_xhand_retargeting_by_side",
    "convert_hand_frame_to_manus",
    "ensure_local_dex_retargeting",
    "make_xhand_retarget_ref_value",
    "map_xhand_command_order",
    "retarget_xhand_pose7_by_side",
    "smooth_hand_quaternion_frame",
    "vr_matrices_to_manus",
    "xhand_command_joint_names",
]
