"""Public API for Astribot human-motion retargeting.

The package is split by responsibility:

- :mod:`constants` owns coordinate and joint conventions.
- :mod:`geometry` owns body-frame transforms.
- :mod:`hands` owns Manus conversion and XHand retargeting.
- :mod:`pipeline` owns frame and sequence orchestration.
- :mod:`io` owns HDF5 and image data boundaries.
- :mod:`runner` connects I/O and the pipeline for offline tasks.
- :mod:`resampling` aligns retargeted episodes to robot episode lengths.
"""

from .constants import ASTRIBOT_BASE_FROM_HUMAN_ROOT
from .constants import ASTRIBOT_SDK_PINOCCHIO_PATH_MARKER
from .constants import DEFAULT_DEX_RETARGETING_CONFIG_DIR
from .constants import DEFAULT_FRONT
from .constants import DEXHAND_COMMAND_DIM
from .constants import DEX_RETARGETING_SRC
from .constants import HAND_POINT_COUNT
from .constants import HAND_POSE_DIM
from .constants import HAND_SIDES
from .constants import LEFT_EFFECTOR_TARGET_FROM_HUMAN_WRIST
from .constants import PROJECT_ROOT
from .constants import REQUIRED_JOINT_NAMES
from .constants import RIGHT_EFFECTOR_TARGET_FROM_HUMAN_WRIST
from .constants import TARGET_NAMES
from .constants import TORSO_TARGET_FROM_HUMAN_CHEST
from .constants import UPPER_BODY_TARGET_NAMES
from .constants import UPPER_BODY_TARGET_SPECS
from .constants import VR_TO_MANUS_ROT_MATS
from .constants import WORLD_UP
from .constants import WRIST_TARGET_LOCAL_Y_CLOCKWISE_90
from .constants import WRIST_TARGET_LOCAL_Y_CORRECTIONS
from .constants import WRIST_TARGET_LOCAL_Y_COUNTERCLOCKWISE_90
from .constants import XHAND_JOINT_SUFFIXES
from .constants import XHAND_TIP_INDICES
from .constants import XHAND_URDF_DIR
from .geometry import chassis_transform_from_xyyaw
from .geometry import compute_chassis_frame
from .geometry import compute_head_rpy
from .geometry import compute_human_root_frame
from .geometry import horizontal_projection
from .geometry import make_transform
from .geometry import normalize_vector
from .geometry import retarget_pose_transform
from .geometry import retarget_upper_body_frame
from .geometry import signed_heading_delta
from .geometry import transforms_to_pose7
from .geometry import wrap_to_pi
from .hands import build_xhand_retargeting_by_side
from .hands import convert_hand_frame_to_manus
from .hands import ensure_local_dex_retargeting
from .hands import make_xhand_retarget_ref_value
from .hands import map_xhand_command_order
from .hands import retarget_xhand_pose7_by_side
from .hands import smooth_hand_quaternion_frame
from .hands import vr_matrices_to_manus
from .hands import xhand_command_joint_names
from .pipeline import build_sequence_result
from .pipeline import copy_hand_quaternions
from .pipeline import make_retarget_state
from .pipeline import prepare_hand_pose7_by_side
from .pipeline import retarget_one_frame
from .pipeline import retarget_sequence
from .resampling import compute_robot_target_length
from .resampling import compute_speed_target_length
from .resampling import resample_directory
from .resampling import resample_hdf5_file
from .resampling import validate_speed_factor


__all__ = [
    "ASTRIBOT_BASE_FROM_HUMAN_ROOT",
    "ASTRIBOT_SDK_PINOCCHIO_PATH_MARKER",
    "DEFAULT_DEX_RETARGETING_CONFIG_DIR",
    "DEFAULT_FRONT",
    "DEXHAND_COMMAND_DIM",
    "DEX_RETARGETING_SRC",
    "HAND_POINT_COUNT",
    "HAND_POSE_DIM",
    "HAND_SIDES",
    "LEFT_EFFECTOR_TARGET_FROM_HUMAN_WRIST",
    "PROJECT_ROOT",
    "REQUIRED_JOINT_NAMES",
    "RIGHT_EFFECTOR_TARGET_FROM_HUMAN_WRIST",
    "TARGET_NAMES",
    "TORSO_TARGET_FROM_HUMAN_CHEST",
    "UPPER_BODY_TARGET_NAMES",
    "UPPER_BODY_TARGET_SPECS",
    "VR_TO_MANUS_ROT_MATS",
    "WORLD_UP",
    "WRIST_TARGET_LOCAL_Y_CLOCKWISE_90",
    "WRIST_TARGET_LOCAL_Y_CORRECTIONS",
    "WRIST_TARGET_LOCAL_Y_COUNTERCLOCKWISE_90",
    "XHAND_JOINT_SUFFIXES",
    "XHAND_TIP_INDICES",
    "XHAND_URDF_DIR",
    "build_sequence_result",
    "build_xhand_retargeting_by_side",
    "chassis_transform_from_xyyaw",
    "compute_chassis_frame",
    "compute_head_rpy",
    "compute_human_root_frame",
    "convert_hand_frame_to_manus",
    "copy_hand_quaternions",
    "ensure_local_dex_retargeting",
    "horizontal_projection",
    "make_retarget_state",
    "make_transform",
    "make_xhand_retarget_ref_value",
    "map_xhand_command_order",
    "normalize_vector",
    "prepare_hand_pose7_by_side",
    "retarget_one_frame",
    "retarget_pose_transform",
    "retarget_sequence",
    "retarget_upper_body_frame",
    "retarget_xhand_pose7_by_side",
    "compute_robot_target_length",
    "compute_speed_target_length",
    "resample_directory",
    "resample_hdf5_file",
    "validate_speed_factor",
    "signed_heading_delta",
    "smooth_hand_quaternion_frame",
    "transforms_to_pose7",
    "vr_matrices_to_manus",
    "wrap_to_pi",
    "xhand_command_joint_names",
]
