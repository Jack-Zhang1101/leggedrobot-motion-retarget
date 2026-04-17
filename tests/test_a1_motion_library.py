import importlib
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from retarget_motion import retarget_core


def _has_pybullet():
  try:
    importlib.import_module("pybullet")
    return True
  except ImportError:
    return False


class RetargetCoreTest(unittest.TestCase):

  def setUp(self):
    self.fixture_path = (
        Path(__file__).resolve().parent.parent /
        "retarget_motion" / "data" / "dog_pace_joint_pos.txt")

  def test_retarget_motion_module_imports(self):
    module = importlib.import_module("retarget_motion.retarget_motion")
    self.assertIs(module.retarget_core, retarget_core)

  def test_load_robot_config_a1(self):
    config = retarget_core.load_robot_config("a1")
    self.assertEqual(config.URDF_FILENAME, "a1/a1.urdf")

  def test_retarget_joint_data_requires_pybullet(self):
    joint_pos_data = retarget_core.load_ref_data(str(self.fixture_path), 0, 2)
    with mock.patch.object(retarget_core, "_load_pybullet", return_value=(None, None)):
      with self.assertRaises(ImportError):
        retarget_core.retarget_joint_data("a1", joint_pos_data, gui=False)

  def test_retarget_motion_frames_requires_pybullet(self):
    config = retarget_core.load_robot_config("a1")
    joint_pos_data = retarget_core.load_ref_data(str(self.fixture_path), 0, 2)
    with mock.patch.object(retarget_core, "_load_pybullet", return_value=(None, None)):
      with self.assertRaises(ImportError):
        retarget_core.retarget_motion_frames(
            robot=object(), config=config, joint_pos_data=joint_pos_data)

  @unittest.skipUnless(_has_pybullet(), "pybullet is not available in this interpreter")
  def test_retarget_joint_data_returns_non_default_19d_pose_for_a1(self):
    config = retarget_core.load_robot_config("a1")
    joint_pos_data = retarget_core.load_ref_data(
        str(self.fixture_path), 0, 2)
    frames = retarget_core.retarget_joint_data("a1", joint_pos_data, gui=False)
    self.assertEqual(frames.shape, (2, 19))
    default_joint_pose = np.tile(config.DEFAULT_JOINT_POSE, (frames.shape[0], 1))
    self.assertFalse(np.allclose(frames[:, 7:], default_joint_pose))
