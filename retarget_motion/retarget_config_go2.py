import numpy as np

from motion_imitation.utilities import asset_utils


URDF_FILENAME = asset_utils.resolve_repo_asset_path("assets/robots/go2/urdf/go2.urdf")

REF_POS_SCALE = 0.825
INIT_POS = np.array([0.0, 0.0, 0.35], dtype=np.float64)
INIT_ROT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
ADAPT_ROOT_HEIGHT_OFFSET = 0.02

# Toe/hip lookup for retargeting follows the source motion convention:
#   FR, RR, FL, RL
SIM_TOE_JOINT_IDS = [
    13,  # front right foot
    25,  # rear right foot
    7,  # front left foot
    19,  # rear left foot
]
SIM_HIP_JOINT_IDS = [
    8,  # front right hip
    20,  # rear right hip
    2,  # front left hip
    14,  # rear left hip
]
SIM_ROOT_OFFSET = np.array([0.0, 0.0, -0.04], dtype=np.float64)
SIM_TOE_OFFSET_LOCAL = [
    np.array([0.0, -0.0465, 0.0], dtype=np.float64),
    np.array([0.0, -0.0465, 0.0], dtype=np.float64),
    np.array([0.0, 0.0465, 0.0], dtype=np.float64),
    np.array([0.0, 0.0465, 0.0], dtype=np.float64),
]

# Joint pose arrays produced by PyBullet follow the robot q-order:
#   FL, FR, RL, RR
# File/export order must match LeggedLab Go2 AMP:
#   FR, FL, RR, RL
SIM_LEG_ORDER = ("FL", "FR", "RL", "RR")
OUTPUT_LEG_ORDER = ("FR", "FL", "RR", "RL")

DEFAULT_JOINT_POSE = np.array([
    0.0, 0.8, -1.5,
    0.0, 0.8, -1.5,
    0.0, 0.8, -1.5,
    0.0, 0.8, -1.5,
], dtype=np.float64)
JOINT_DAMPING = [
    0.1, 0.05, 0.01,
    0.1, 0.05, 0.01,
    0.1, 0.05, 0.01,
    0.1, 0.05, 0.01,
]

FORWARD_DIR_OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float64)
