import numpy as np
from motion_imitation.utilities import asset_utils

URDF_FILENAME = asset_utils.resolve_repo_asset_path("assets/robots/a1/urdf/a1.urdf")

REF_POS_SCALE = 0.825
INIT_POS = np.array([0, 0, 0.32])
INIT_ROT = np.array([0, 0, 0, 1.0])

SIM_TOE_JOINT_IDS = [
    6,  # front right foot
    16,  # rear right foot
    11,  # front left foot
    21,  # rear left foot
]
SIM_HIP_JOINT_IDS = [2, 12, 7, 17]
SIM_ROOT_OFFSET = np.array([0, 0, -0.06])
SIM_TOE_OFFSET_LOCAL = [
    np.array([0, -0.05, 0.0]),
    np.array([0, -0.05, 0.01]),
    np.array([0, 0.05, 0.0]),
    np.array([0, 0.05, 0.01])
]

DEFAULT_JOINT_POSE = np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])
JOINT_DAMPING = [0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01]

FORWARD_DIR_OFFSET = np.array([0, 0, 0])
