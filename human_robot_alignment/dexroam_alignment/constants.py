"""Constants shared by the Astribot retargeting pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np


REQUIRED_JOINT_NAMES = (
    "hips",
    "chest",
    "left-hand-wrist-twist",
    "right-hand-wrist-twist",
)

UPPER_BODY_TARGET_NAMES = (
    "astribot_torso",
    "astribot_arm_left",
    "astribot_arm_right",
)
TARGET_NAMES = ("astribot_chassis", *UPPER_BODY_TARGET_NAMES)

ASTRIBOT_BASE_FROM_HUMAN_ROOT = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

TORSO_TARGET_FROM_HUMAN_CHEST = np.asarray(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
LEFT_EFFECTOR_TARGET_FROM_HUMAN_WRIST = np.asarray(
    [
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
RIGHT_EFFECTOR_TARGET_FROM_HUMAN_WRIST = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
WRIST_TARGET_LOCAL_Y_CLOCKWISE_90 = np.asarray(
    [
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
WRIST_TARGET_LOCAL_Y_COUNTERCLOCKWISE_90 = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
WRIST_TARGET_LOCAL_Y_CORRECTIONS = {
    "astribot_arm_left": WRIST_TARGET_LOCAL_Y_CLOCKWISE_90,
    "astribot_arm_right": WRIST_TARGET_LOCAL_Y_COUNTERCLOCKWISE_90,
}
UPPER_BODY_TARGET_SPECS = (
    ("astribot_torso", "chest", TORSO_TARGET_FROM_HUMAN_CHEST),
    ("astribot_arm_left", "left-hand-wrist-twist", LEFT_EFFECTOR_TARGET_FROM_HUMAN_WRIST),
    ("astribot_arm_right", "right-hand-wrist-twist", RIGHT_EFFECTOR_TARGET_FROM_HUMAN_WRIST),
)

WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
DEFAULT_FRONT = np.array([1.0, 0.0, 0.0], dtype=np.float64)

HAND_SIDES = ("left", "right")
VR_TO_MANUS_ROT_MATS = {
    "left": np.asarray(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    ),
    "right": np.asarray(
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    ),
}
HAND_POINT_COUNT = 25
HAND_POSE_DIM = 7
DEXHAND_COMMAND_DIM = 12
XHAND_TIP_INDICES = np.asarray([4, 9, 14, 19, 24], dtype=np.int64)
XHAND_JOINT_SUFFIXES = (
    "thumb_bend_joint",
    "thumb_rota_joint1",
    "thumb_rota_joint2",
    "index_bend_joint",
    "index_joint1",
    "index_joint2",
    "mid_joint1",
    "mid_joint2",
    "ring_joint1",
    "ring_joint2",
    "pinky_joint1",
    "pinky_joint2",
)

RETARGET_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RETARGET_ROOT
DEX_RETARGETING_SRC = RETARGET_ROOT / "third_party" / "dex-retargeting" / "src"
DEFAULT_DEX_RETARGETING_CONFIG_DIR = RETARGET_ROOT / "configs"
XHAND_URDF_DIR = RETARGET_ROOT / "assets" / "xhand"
ASTRIBOT_SDK_PINOCCHIO_PATH_MARKER = (
    "astribot_sdk_ros2/third_party/third_pkg/pinocchio"
)
