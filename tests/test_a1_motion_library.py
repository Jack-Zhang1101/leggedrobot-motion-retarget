import importlib
import unittest

from retarget_motion import retarget_core


class RetargetCoreTest(unittest.TestCase):

  def test_retarget_motion_module_imports(self):
    module = importlib.import_module("retarget_motion.retarget_motion")
    self.assertIs(module.retarget_core, retarget_core)

  def test_load_robot_config_a1(self):
    config = retarget_core.load_robot_config("a1")
    self.assertEqual(config.URDF_FILENAME, "a1/a1.urdf")

  def test_retarget_joint_data_returns_19d_pose_for_a1(self):
    joint_pos_data = retarget_core.load_ref_data(
        "retarget_motion/data/dog_pace_joint_pos.txt", 0, 2)
    frames = retarget_core.retarget_joint_data("a1", joint_pos_data, gui=False)
    self.assertEqual(frames.shape, (2, 19))
