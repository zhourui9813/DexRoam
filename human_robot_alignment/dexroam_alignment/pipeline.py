"""Frame and sequence orchestration for Astribot retargeting."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .constants import DEFAULT_FRONT
from .constants import HAND_SIDES
from .constants import TARGET_NAMES
from .geometry import compute_chassis_frame
from .geometry import compute_head_rpy
from .geometry import compute_human_root_frame
from .geometry import retarget_upper_body_frame
from .hands import convert_hand_frame_to_manus
from .hands import retarget_xhand_pose7_by_side


def make_retarget_state(
    joint_index: Mapping[str, int],
    previous_front: np.ndarray | None = None,
    previous_root_transform: np.ndarray | None = None,
    chassis_xyyaw: np.ndarray | None = None,
    previous_hand_quaternions_by_side: Mapping[str, np.ndarray] | None = None,
    ground_z: float = 0.0,
    xhand_retargeting_by_side: Mapping[str, tuple[Any, np.ndarray]] | None = None,
) -> dict:
    """Create the mutable state carried between retargeted frames."""
    return {
        "joint_index": joint_index,
        "previous_front": (
            DEFAULT_FRONT.copy()
            if previous_front is None
            else previous_front
        ),
        "previous_root_transform": previous_root_transform,
        "chassis_xyyaw": (
            np.zeros(3, dtype=np.float64)
            if chassis_xyyaw is None
            else chassis_xyyaw
        ),
        "previous_hand_quaternions_by_side": (
            previous_hand_quaternions_by_side or {}
        ),
        "ground_z": ground_z,
        "xhand_retargeting_by_side": xhand_retargeting_by_side,
    }


def copy_hand_quaternions(
    previous_hand_quaternions_by_side: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Copy the previous-frame quaternion state for both hands."""
    return {
        side: np.asarray(quaternions, dtype=np.float64).copy()
        for side, quaternions in previous_hand_quaternions_by_side.items()
    }


def prepare_hand_pose7_by_side(
    frame: Mapping[str, Any],
    state: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray] | None, dict[str, np.ndarray]]:
    """Convert a frame's hand inputs and return the next quaternion state."""
    next_hand_quaternions = copy_hand_quaternions(
        state["previous_hand_quaternions_by_side"]
    )
    hand_pose7_by_side = {}

    if frame["hand_matrices_by_side"] is not None:
        for side, hand_matrices in frame["hand_matrices_by_side"].items():
            pose7, quaternions = convert_hand_frame_to_manus(
                hand_matrices,
                side,
                previous_quaternions=next_hand_quaternions.get(side),
            )
            hand_pose7_by_side[side] = pose7
            next_hand_quaternions[side] = quaternions

    if not hand_pose7_by_side:
        return None, next_hand_quaternions
    return hand_pose7_by_side, next_hand_quaternions


def build_sequence_result(frames: list[dict]) -> dict:
    """Stack per-frame results into the historical sequence result format."""
    frame_count = len(frames)
    frame_indices = np.asarray(
        [
            frame["frame_index"]
            if frame["frame_index"] is not None
            else frame_idx
            for frame_idx, frame in enumerate(frames)
        ],
        dtype=np.int64,
    )
    if any(frame["frame_time"] is None for frame in frames):
        frame_times = np.arange(frame_count, dtype=np.float64)
    else:
        frame_times = np.asarray([float(frame["frame_time"]) for frame in frames])

    return {
        "frames": frames,
        "frame_count": frame_count,
        "frame_indices": frame_indices,
        "frame_times": frame_times,
        "duration": (
            float(frame_times[-1] - frame_times[0])
            if frame_count > 1
            else 0.0
        ),
        "transforms": {
            name: np.stack(
                [frame["transforms"][name] for frame in frames],
                axis=0,
            )
            for name in TARGET_NAMES
        },
        "chassis_xyyaw": np.stack(
            [frame["chassis_xyyaw"] for frame in frames],
            axis=0,
        ),
        "head_joint_command": (
            np.stack(
                [frame["head_joint_command"] for frame in frames],
                axis=0,
            )
            if all(frame["head_joint_command"] is not None for frame in frames)
            else None
        ),
        "xhand_commands": [frame["xhand_command"] for frame in frames],
        "human_hand_pose7_by_side": {
            side: np.stack(
                [
                    frame["human_hand_pose7_by_side"][side]
                    for frame in frames
                ],
                axis=0,
            )
            for side in HAND_SIDES
            if all(
                frame["human_hand_pose7_by_side"] is not None
                and side in frame["human_hand_pose7_by_side"]
                for frame in frames
            )
        },
    }


def retarget_one_frame(
    frame: Mapping[str, Any],
    state: Mapping[str, Any],
) -> tuple[dict, dict]:
    """Retarget one human frame and return its result and next state."""
    root_transform, next_front = compute_human_root_frame(
        frame["body_matrices"],
        joint_index=state["joint_index"],
        ground_z=state["ground_z"],
        previous_front=state["previous_front"],
    )
    chassis_xyyaw, chassis_transform = compute_chassis_frame(
        root_transform=root_transform,
        previous_root_transform=state["previous_root_transform"],
        previous_chassis_xyyaw=state["chassis_xyyaw"],
    )
    upper_body_transforms = retarget_upper_body_frame(
        frame["body_matrices"],
        joint_index=state["joint_index"],
        root_transform=root_transform,
    )
    transforms = {
        "astribot_chassis": chassis_transform,
        **upper_body_transforms,
    }

    head_joint_command = None
    head_matrix = frame.get("head_matrix")
    if head_matrix is not None:
        head_joint_command = compute_head_rpy(head_matrix, root_transform)

    hand_pose7_by_side, next_hand_quaternions = prepare_hand_pose7_by_side(
        frame,
        state,
    )
    xhand_command = None
    if (
        hand_pose7_by_side is not None
        and state["xhand_retargeting_by_side"] is not None
    ):
        xhand_command = retarget_xhand_pose7_by_side(
            hand_pose7_by_side,
            state["xhand_retargeting_by_side"],
        )

    result = {
        "frame_index": frame["frame_index"],
        "frame_time": frame["frame_time"],
        "transforms": transforms,
        "human_root_transform": root_transform,
        "chassis_xyyaw": chassis_xyyaw,
        "head_joint_command": head_joint_command,
        "human_hand_pose7_by_side": hand_pose7_by_side,
        "xhand_command": xhand_command,
    }
    next_state = make_retarget_state(
        joint_index=state["joint_index"],
        previous_front=next_front,
        previous_root_transform=root_transform.copy(),
        chassis_xyyaw=chassis_xyyaw.copy(),
        previous_hand_quaternions_by_side=next_hand_quaternions,
        ground_z=state["ground_z"],
        xhand_retargeting_by_side=state["xhand_retargeting_by_side"],
    )
    return result, next_state


def retarget_sequence(
    frames: Sequence[Mapping[str, Any]],
    joint_index: Mapping[str, int],
    ground_z: float = 0.0,
    xhand_retargeting_by_side: Mapping[str, tuple[Any, np.ndarray]] | None = None,
) -> dict:
    """Retarget a sequence of human frames."""
    state = make_retarget_state(
        joint_index=joint_index,
        ground_z=ground_z,
        xhand_retargeting_by_side=xhand_retargeting_by_side,
    )
    results = []
    for frame in frames:
        result, state = retarget_one_frame(frame, state)
        results.append(result)
    if not results:
        raise ValueError("retarget_sequence requires at least one frame.")
    return build_sequence_result(results)


# Private aliases keep compatibility with the original single-file module.
_make_retarget_state = make_retarget_state
_copy_hand_quaternions = copy_hand_quaternions
_retarget_hand_pose7_by_side = prepare_hand_pose7_by_side
_build_loop_result = build_sequence_result


__all__ = [
    "build_sequence_result",
    "copy_hand_quaternions",
    "make_retarget_state",
    "prepare_hand_pose7_by_side",
    "retarget_one_frame",
    "retarget_sequence",
]
