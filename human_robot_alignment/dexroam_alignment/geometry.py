"""Body-frame and coordinate-transform operations for retargeting."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from .constants import ASTRIBOT_BASE_FROM_HUMAN_ROOT
from .constants import DEFAULT_FRONT
from .constants import UPPER_BODY_TARGET_SPECS
from .constants import WORLD_UP
from .constants import WRIST_TARGET_LOCAL_Y_CORRECTIONS


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Return a normalized three-dimensional vector."""
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-9:
        raise ValueError("Normalize ERROR!")
    return vector / norm


def horizontal_projection(vector: np.ndarray) -> np.ndarray | None:
    """Return a vector's normalized projection onto the horizontal plane."""
    projected = np.asarray(vector, dtype=np.float64).reshape(3).copy()
    projected[2] = 0.0
    norm = float(np.linalg.norm(projected))
    if not np.isfinite(norm) or norm <= 1e-9:
        return None
    return projected / norm


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a 4x4 transform from a rotation matrix and translation."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def transforms_to_pose7(transforms: np.ndarray) -> np.ndarray:
    """Convert 4x4 transforms to continuous ``[x, y, z, qx, qy, qz, qw]`` poses."""
    transforms = np.asarray(transforms, dtype=np.float64)
    pose7 = np.empty((transforms.shape[0], 7), dtype=np.float64)
    pose7[:, :3] = transforms[:, :3, 3]
    pose7[:, 3:] = Rotation.from_matrix(transforms[:, :3, :3]).as_quat()
    for idx in range(1, pose7.shape[0]):
        if float(np.dot(pose7[idx - 1, 3:], pose7[idx, 3:])) < 0.0:
            pose7[idx, 3:] *= -1.0
    return pose7


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def signed_heading_delta(
    previous_front: np.ndarray,
    current_front: np.ndarray,
) -> float:
    """Return the signed horizontal angle between two forward vectors."""
    cross = np.cross(previous_front, current_front)
    sin_angle = float(np.dot(WORLD_UP, cross))
    cos_angle = float(np.dot(previous_front, current_front))
    return wrap_to_pi(float(np.arctan2(sin_angle, cos_angle)))


def chassis_transform_from_xyyaw(xyyaw: np.ndarray) -> np.ndarray:
    """Convert a chassis x/y/yaw pose into a 4x4 transform."""
    x, y, theta = np.asarray(xyyaw, dtype=np.float64).reshape(3)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation = np.asarray(
        [
            [cos_theta, -sin_theta, 0.0],
            [sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return make_transform(rotation, np.array([x, y, 0.0], dtype=np.float64))


def compute_human_root_frame(
    frame_matrices: np.ndarray,
    joint_index: Mapping[str, int],
    ground_z: float = 0.0,
    previous_front: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the current human root frame from hips and chest transforms."""
    matrices = np.asarray(frame_matrices, dtype=np.float64)
    hips_matrix = matrices[int(joint_index["hips"])]

    front = horizontal_projection(hips_matrix[:3, 2])
    if front is None and "chest" in joint_index:
        front = horizontal_projection(matrices[int(joint_index["chest"]), :3, 2])
    if front is None and previous_front is not None:
        front = np.asarray(previous_front, dtype=np.float64).reshape(3)
    if front is None:
        raise ValueError("hips matrix error!")

    left = normalize_vector(np.cross(WORLD_UP, front))
    front = normalize_vector(np.cross(left, WORLD_UP))

    origin = hips_matrix[:3, 3].copy()
    origin[2] = float(ground_z)
    root_transform = make_transform(
        rotation=np.column_stack([left, WORLD_UP, front]),
        translation=origin,
    )
    return root_transform, front.copy()


def compute_chassis_frame(
    root_transform: np.ndarray,
    previous_root_transform: np.ndarray | None,
    previous_chassis_xyyaw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the chassis pose by integrating human-root motion."""
    root_transform = np.asarray(root_transform, dtype=np.float64).reshape(4, 4)
    previous_chassis_xyyaw = np.asarray(
        previous_chassis_xyyaw,
        dtype=np.float64,
    ).reshape(3)

    if previous_root_transform is None:
        xyyaw = np.zeros(3, dtype=np.float64)
        return xyyaw, chassis_transform_from_xyyaw(xyyaw)

    previous_root_transform = np.asarray(
        previous_root_transform,
        dtype=np.float64,
    ).reshape(4, 4)
    delta_world = root_transform[:3, 3] - previous_root_transform[:3, 3]
    delta_human_root = previous_root_transform[:3, :3].T @ delta_world
    delta_astribot_base = ASTRIBOT_BASE_FROM_HUMAN_ROOT @ delta_human_root

    dtheta = signed_heading_delta(
        previous_root_transform[:3, 2],
        root_transform[:3, 2],
    )

    previous_theta = previous_chassis_xyyaw[2]
    cos_theta = np.cos(previous_theta)
    sin_theta = np.sin(previous_theta)
    dx_local = float(delta_astribot_base[0])
    dy_local = float(delta_astribot_base[1])
    xyyaw = previous_chassis_xyyaw.copy()
    xyyaw[0] += cos_theta * dx_local - sin_theta * dy_local
    xyyaw[1] += sin_theta * dx_local + cos_theta * dy_local
    xyyaw[2] += dtheta
    return xyyaw, chassis_transform_from_xyyaw(xyyaw)


def retarget_pose_transform(
    root_transform: np.ndarray,
    joint_transform: np.ndarray,
    target_from_human_joint: np.ndarray,
) -> np.ndarray:
    """Convert a human-joint transform into the Astribot base frame."""
    root_transform = np.asarray(root_transform, dtype=np.float64).reshape(4, 4)
    joint_transform = np.asarray(joint_transform, dtype=np.float64).reshape(4, 4)
    target_from_human_joint = np.asarray(
        target_from_human_joint,
        dtype=np.float64,
    ).reshape(3, 3)

    astribot_base_from_human_root = np.eye(4, dtype=np.float64)
    astribot_base_from_human_root[:3, :3] = ASTRIBOT_BASE_FROM_HUMAN_ROOT

    human_joint_from_target = np.eye(4, dtype=np.float64)
    human_joint_from_target[:3, :3] = target_from_human_joint

    human_root_from_human_joint = np.linalg.inv(root_transform) @ joint_transform
    return (
        astribot_base_from_human_root
        @ human_root_from_human_joint
        @ human_joint_from_target
    )


def retarget_upper_body_frame(
    frame_matrices: np.ndarray,
    joint_index: Mapping[str, int],
    root_transform: np.ndarray,
) -> dict[str, np.ndarray]:
    """Retarget the torso and both wrist effectors for one frame."""
    matrices = np.asarray(frame_matrices, dtype=np.float64)
    transforms = {}
    for target_name, joint_name, target_from_human_joint in UPPER_BODY_TARGET_SPECS:
        transform = retarget_pose_transform(
            root_transform=root_transform,
            joint_transform=matrices[int(joint_index[joint_name])],
            target_from_human_joint=target_from_human_joint,
        )
        local_y_correction = WRIST_TARGET_LOCAL_Y_CORRECTIONS.get(target_name)
        if local_y_correction is not None:
            transform[:3, :3] = transform[:3, :3] @ local_y_correction
        transforms[target_name] = transform
    return transforms


def compute_head_rpy(
    head_matrix: np.ndarray,
    root_transform: np.ndarray,
) -> np.ndarray:
    """Convert human head orientation into Astribot pitch/yaw commands."""
    head_matrix = np.asarray(head_matrix, dtype=np.float64).reshape(4, 4)
    root_transform = np.asarray(root_transform, dtype=np.float64).reshape(4, 4)

    rotation_human_root = root_transform[:3, :3].T @ head_matrix[:3, :3]
    rotation_astribot_base = ASTRIBOT_BASE_FROM_HUMAN_ROOT @ rotation_human_root
    euler = Rotation.from_matrix(rotation_astribot_base).as_euler("xyz", degrees=False)

    robot_pitch = euler[1]
    robot_yaw = euler[2]
    return np.asarray(
        [wrap_to_pi(robot_yaw), wrap_to_pi(robot_pitch)],
        dtype=np.float64,
    )


# Private aliases keep compatibility with the original single-file module.
_normalize_vector = normalize_vector
_horizontal_projection = horizontal_projection
_make_transform = make_transform
_signed_heading_delta = signed_heading_delta


__all__ = [
    "DEFAULT_FRONT",
    "chassis_transform_from_xyyaw",
    "compute_chassis_frame",
    "compute_head_rpy",
    "compute_human_root_frame",
    "horizontal_projection",
    "make_transform",
    "normalize_vector",
    "retarget_pose_transform",
    "retarget_upper_body_frame",
    "signed_heading_delta",
    "transforms_to_pose7",
    "wrap_to_pi",
]
