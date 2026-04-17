"""Adapt existing quadruped motion files into A1-compatible reference motions."""

from pathlib import Path

import numpy as np

from motion_imitation.utilities import motion_data
from retarget_motion import retarget_core


IDENTITY_ROTATION = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def _require_pybullet():
  pybullet, pd = retarget_core._load_pybullet()
  if pybullet is None or pd is None:
    raise ImportError("pybullet is required for motion adaptation")
  return pybullet, pd


def _invert_base_transform(pybullet, base_pos, base_rot):
  return pybullet.invertTransform(base_pos, base_rot)


def _world_positions_to_base_frame(pybullet, base_pos, base_rot, world_positions):
  inv_pos, inv_rot = _invert_base_transform(pybullet, base_pos, base_rot)
  local_positions = []
  for world_pos in world_positions:
    local_pos, _ = pybullet.multiplyTransforms(
        inv_pos, inv_rot, world_pos, IDENTITY_ROTATION)
    local_positions.append(np.array(local_pos, dtype=np.float64))
  return local_positions


def _local_positions_to_world(pybullet, base_pos, base_rot, local_positions):
  world_positions = []
  for local_pos in local_positions:
    world_pos, _ = pybullet.multiplyTransforms(
        base_pos, base_rot, local_pos, IDENTITY_ROTATION)
    world_positions.append(np.array(world_pos, dtype=np.float64))
  return world_positions


def _get_link_world_positions(pybullet, robot, link_ids):
  positions = []
  for link_id in link_ids:
    link_state = pybullet.getLinkState(robot, link_id, computeForwardKinematics=True)
    positions.append(np.array(link_state[4], dtype=np.float64))
  return positions


def _solve_target_joint_pose(pybullet, robot, config, target_world_toe_positions):
  joint_low, joint_high = retarget_core.get_joint_limits(pybullet, robot)
  joint_pose = pybullet.calculateInverseKinematics2(
      robot,
      config.SIM_TOE_JOINT_IDS,
      target_world_toe_positions,
      jointDamping=config.JOINT_DAMPING,
      lowerLimits=joint_low,
      upperLimits=joint_high,
      restPoses=config.DEFAULT_JOINT_POSE)
  return np.array(joint_pose, dtype=np.float64)


def adapt_motion_to_a1(source_motion):
  if not isinstance(source_motion, motion_data.MotionData):
    raise TypeError("source_motion must be a MotionData instance")

  pybullet, pybullet_data = _require_pybullet()
  source_config = retarget_core.load_robot_config("laikago")
  target_config = retarget_core.load_robot_config("a1")

  client = pybullet.connect(pybullet.DIRECT)
  try:
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    pybullet.resetSimulation()
    source_robot = pybullet.loadURDF(
        source_config.URDF_FILENAME, source_config.INIT_POS, source_config.INIT_ROT)
    target_robot = pybullet.loadURDF(
        target_config.URDF_FILENAME, target_config.INIT_POS, target_config.INIT_ROT)

    first_source_frame = np.array(source_motion.get_frame(0), dtype=np.float64, copy=True)
    source_root_z0 = first_source_frame[2]
    target_root_z0 = float(target_config.INIT_POS[2])

    frames = []
    for frame_id in range(source_motion.get_num_frames()):
      source_frame = np.array(source_motion.get_frame(frame_id), dtype=np.float64, copy=True)
      source_root_pos = source_frame[:3]
      source_root_rot = source_frame[3:7]

      retarget_core.set_pose(pybullet, source_robot, source_frame)
      source_world_toes = _get_link_world_positions(
          pybullet, source_robot, source_config.SIM_TOE_JOINT_IDS)
      source_local_toes = _world_positions_to_base_frame(
          pybullet, source_root_pos, source_root_rot, source_world_toes)

      target_root_pos = source_root_pos.copy()
      target_root_pos[2] = source_root_pos[2] - source_root_z0 + target_root_z0
      target_root_rot = source_root_rot.copy()

      pybullet.resetBasePositionAndOrientation(target_robot, target_root_pos, target_root_rot)
      target_world_toes = _local_positions_to_world(
          pybullet, target_root_pos, target_root_rot, source_local_toes)
      target_joint_pose = _solve_target_joint_pose(
          pybullet, target_robot, target_config, target_world_toes)
      target_frame = np.concatenate([target_root_pos, target_root_rot, target_joint_pose])
      frames.append(target_frame)

    return np.asarray(frames, dtype=np.float64)
  finally:
    pybullet.disconnect(client)


def load_and_adapt_motion_file(source_motion_path):
  motion = motion_data.MotionData(str(Path(source_motion_path)))
  return adapt_motion_to_a1(motion)
