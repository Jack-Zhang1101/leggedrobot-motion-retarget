import numpy as np

from motion_imitation.utilities import asset_utils


URDF_FILENAME = asset_utils.resolve_repo_asset_path("assets/robots/sizu/urdf/sizu.urdf")

REF_POS_SCALE = 0.825
INIT_POS = np.array([0.0, 0.0, 0.3], dtype=np.float64)
INIT_ROT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
ADAPT_ROOT_HEIGHT_OFFSET = 0.3
ADAPT_FOOT_CLEARANCE_TARGET = 0.05
ADAPT_FOOT_BOTTOM_OFFSET = 0.04
ADAPT_FOOT_CLEARANCE_STAT = "percentile"
ADAPT_FOOT_CLEARANCE_PERCENTILE = 5.0
USE_ANALYTIC_ADAPT_IK = False

# Toe/hip lookup for retargeting follows the source motion convention:
#   FR, RR, FL, RL
SIM_TOE_JOINT_IDS = [
    7,   # front right foot
    15,  # rear right foot
    3,   # front left foot
    11,  # rear left foot
]
SIM_HIP_JOINT_IDS = [
    4,   # front right hip
    12,  # rear right hip
    0,   # front left hip
    8,   # rear left hip
]
SIM_ROOT_OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float64)
SIM_TOE_OFFSET_LOCAL = [
    np.array([0.0, -0.12, 0.0], dtype=np.float64),
    np.array([0.0, -0.12, 0.0], dtype=np.float64),
    np.array([0.0, 0.12, 0.0], dtype=np.float64),
    np.array([0.0, 0.12, 0.0], dtype=np.float64),
]

# Sizu controller joint order:
#   FL, FR, RL, RR
SIM_LEG_ORDER = ("FL", "FR", "RL", "RR")
OUTPUT_LEG_ORDER = ("FL", "FR", "RL", "RR")

DEFAULT_JOINT_POSE = np.array([
    0.0, 0.8, -1.5,
    0.0, 0.8, -1.5,
    0.0, 0.8, -1.5,
    0.0, 0.8, -1.5,
], dtype=np.float64)
ADAPT_SCALE_JOINT_POSE = np.zeros(12, dtype=np.float64)
JOINT_DAMPING = [
    0.1, 0.05, 0.01,
    0.1, 0.05, 0.01,
    0.1, 0.05, 0.01,
    0.1, 0.05, 0.01,
]

FORWARD_DIR_OFFSET = np.array([0.0, 0.0, 0.05], dtype=np.float64)
