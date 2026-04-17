"""Source table and simple transforms for the A1 motion library."""

from pathlib import Path

import numpy as np


RAW = "raw"
DERIVED = "derived"
ADAPTED = "adapted"


REPO_ROOT = Path(__file__).resolve().parent.parent
MOTIONS_DIR = REPO_ROOT / "motion_imitation" / "data" / "motions"
RAW_DIR = REPO_ROOT / "retarget_motion" / "data"


MOTION_SPECS = {
    "dog_pace": {
        "mode": RAW,
        "joint_pos_path": RAW_DIR / "dog_pace_joint_pos.txt",
        "metadata_motion_path": MOTIONS_DIR / "dog_pace.txt",
    },
    "dog_trot": {
        "mode": RAW,
        "joint_pos_path": RAW_DIR / "dog_trot_joint_pos.txt",
        "metadata_motion_path": MOTIONS_DIR / "dog_trot.txt",
    },
    "dog_backwards_pace": {
        "mode": DERIVED,
        "source_motion": "dog_pace",
        "metadata_motion_path": MOTIONS_DIR / "dog_backwards_pace.txt",
    },
    "dog_backwards_trot": {
        "mode": DERIVED,
        "source_motion": "dog_trot",
        "metadata_motion_path": MOTIONS_DIR / "dog_backwards_trot.txt",
    },
    "dog_spin": {
        "mode": ADAPTED,
        "source_motion_path": MOTIONS_DIR / "dog_spin.txt",
        "metadata_motion_path": MOTIONS_DIR / "dog_spin.txt",
    },
    "hopturn": {
        "mode": ADAPTED,
        "source_motion_path": MOTIONS_DIR / "hopturn.txt",
        "metadata_motion_path": MOTIONS_DIR / "hopturn.txt",
    },
    "inplace_steps": {
        "mode": ADAPTED,
        "source_motion_path": MOTIONS_DIR / "inplace_steps.txt",
        "metadata_motion_path": MOTIONS_DIR / "inplace_steps.txt",
    },
    "runningman": {
        "mode": ADAPTED,
        "source_motion_path": MOTIONS_DIR / "runningman.txt",
        "metadata_motion_path": MOTIONS_DIR / "runningman.txt",
    },
    "sidesteps": {
        "mode": ADAPTED,
        "source_motion_path": MOTIONS_DIR / "sidesteps.txt",
        "metadata_motion_path": MOTIONS_DIR / "sidesteps.txt",
    },
    "turn": {
        "mode": ADAPTED,
        "source_motion_path": MOTIONS_DIR / "turn.txt",
        "metadata_motion_path": MOTIONS_DIR / "turn.txt",
    },
}


def ordered_motion_names():
  return list(MOTION_SPECS.keys())


def normalize_root_xy_origin(frames):
  normalized = np.array(frames, dtype=np.float64, copy=True)
  normalized[:, 0] -= normalized[0, 0]
  normalized[:, 1] -= normalized[0, 1]
  return normalized


def reverse_motion_frames(frames):
  reversed_frames = np.asarray(frames, dtype=np.float64)[::-1]
  return normalize_root_xy_origin(reversed_frames)
