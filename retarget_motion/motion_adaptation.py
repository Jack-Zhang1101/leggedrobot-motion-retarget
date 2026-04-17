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


def _infer_leg_labels_from_positions(link_positions):
  link_positions = np.asarray(link_positions, dtype=np.float64)
  if link_positions.shape[0] != 4:
    raise ValueError("expected exactly 4 leg positions, got {}".format(link_positions.shape[0]))

  x_mid = np.median(link_positions[:, 0])
  y_mid = np.median(link_positions[:, 1])
  labels = []
  for pos in link_positions:
    front_rear = "F" if pos[0] >= x_mid else "R"
    left_right = "L" if pos[1] >= y_mid else "R"
    labels.append(front_rear + left_right)

  if len(set(labels)) != len(labels):
    raise ValueError("could not infer unique leg labels from positions: {}".format(labels))
  return labels


def _reorder_local_toes_for_target(source_local_toes, remap):
  return [source_local_toes[idx] for idx in remap["target_from_source_indices"]]


def _hip_toe_deltas(toe_positions, hip_positions):
  return [
      np.asarray(toe_pos, dtype=np.float64) - np.asarray(hip_pos, dtype=np.float64)
      for toe_pos, hip_pos in zip(toe_positions, hip_positions)
  ]


def _apply_hip_toe_deltas(target_hip_positions, hip_toe_deltas):
  return [
      np.asarray(hip_pos, dtype=np.float64) + np.asarray(delta, dtype=np.float64)
      for hip_pos, delta in zip(target_hip_positions, hip_toe_deltas)
  ]


def _convert_root_rotation_between_robots(source_root_rot, source_init_rot, target_init_rot):
  source_root_rot = np.asarray(source_root_rot, dtype=np.float64)
  source_init_rot = np.asarray(source_init_rot, dtype=np.float64)
  target_init_rot = np.asarray(target_init_rot, dtype=np.float64)
  canonical_root_rot = retarget_core._quat_multiply(  # pylint: disable=protected-access
      source_root_rot,
      retarget_core._quat_inverse(source_init_rot))  # pylint: disable=protected-access
  target_root_rot = retarget_core._quat_multiply(  # pylint: disable=protected-access
      canonical_root_rot, target_init_rot)
  return retarget_core._quat_normalize(target_root_rot)  # pylint: disable=protected-access


def _compute_leg_remap(source_robot_name, target_robot_name):
  pybullet, pybullet_data = _require_pybullet()
  source_config = retarget_core.load_robot_config(source_robot_name)
  target_config = retarget_core.load_robot_config(target_robot_name)

  client = pybullet.connect(pybullet.DIRECT)
  try:
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    pybullet.resetSimulation()
    source_robot = pybullet.loadURDF(
        source_config.URDF_FILENAME, source_config.INIT_POS, source_config.INIT_ROT)
    target_robot = pybullet.loadURDF(
        target_config.URDF_FILENAME, target_config.INIT_POS, target_config.INIT_ROT)

    source_positions = _get_link_world_positions(
        pybullet, source_robot, source_config.SIM_TOE_JOINT_IDS)
    target_positions = _get_link_world_positions(
        pybullet, target_robot, target_config.SIM_TOE_JOINT_IDS)
    source_labels = _infer_leg_labels_from_positions(source_positions)
    target_labels = _infer_leg_labels_from_positions(target_positions)
    source_index_by_label = {label: idx for idx, label in enumerate(source_labels)}
    target_from_source_indices = [source_index_by_label[label] for label in target_labels]
    return {
        "source_labels": source_labels,
        "target_labels": target_labels,
        "target_from_source_indices": target_from_source_indices,
    }
  finally:
    pybullet.disconnect(client)


def _compute_leg_length_scales(source_robot_name, target_robot_name, remap=None):
  pybullet, pybullet_data = _require_pybullet()
  source_config = retarget_core.load_robot_config(source_robot_name)
  target_config = retarget_core.load_robot_config(target_robot_name)
  remap = remap or _compute_leg_remap(source_robot_name, target_robot_name)

  client = pybullet.connect(pybullet.DIRECT)
  try:
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    pybullet.resetSimulation()
    source_robot = pybullet.loadURDF(
        source_config.URDF_FILENAME, source_config.INIT_POS, source_config.INIT_ROT)
    target_robot = pybullet.loadURDF(
        target_config.URDF_FILENAME, target_config.INIT_POS, target_config.INIT_ROT)

    retarget_core.set_pose(pybullet, source_robot, np.concatenate([
        source_config.INIT_POS, source_config.INIT_ROT, source_config.DEFAULT_JOINT_POSE]))
    retarget_core.set_pose(pybullet, target_robot, np.concatenate([
        target_config.INIT_POS, target_config.INIT_ROT, target_config.DEFAULT_JOINT_POSE]))

    source_toes = _get_link_world_positions(
        pybullet, source_robot, source_config.SIM_TOE_JOINT_IDS)
    source_hips = _get_link_world_positions(
        pybullet, source_robot, source_config.SIM_HIP_JOINT_IDS)
    source_toes = _reorder_local_toes_for_target(source_toes, remap)
    source_hips = _reorder_local_toes_for_target(source_hips, remap)

    target_toes = _get_link_world_positions(
        pybullet, target_robot, target_config.SIM_TOE_JOINT_IDS)
    target_hips = _get_link_world_positions(
        pybullet, target_robot, target_config.SIM_HIP_JOINT_IDS)

    source_lengths = np.linalg.norm(
        np.asarray(source_toes, dtype=np.float64) -
        np.asarray(source_hips, dtype=np.float64),
        axis=1)
    target_lengths = np.linalg.norm(
        np.asarray(target_toes, dtype=np.float64) -
        np.asarray(target_hips, dtype=np.float64),
        axis=1)
    return target_lengths / source_lengths
  finally:
    pybullet.disconnect(client)


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
  leg_remap = _compute_leg_remap("laikago", "a1")
  leg_length_scales = _compute_leg_length_scales("laikago", "a1", remap=leg_remap)

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
      source_world_hips = _get_link_world_positions(
          pybullet, source_robot, source_config.SIM_HIP_JOINT_IDS)
      source_world_toes = _reorder_local_toes_for_target(source_world_toes, leg_remap)
      source_world_hips = _reorder_local_toes_for_target(source_world_hips, leg_remap)
      source_hip_toe_deltas = _hip_toe_deltas(source_world_toes, source_world_hips)
      source_hip_toe_deltas = [
          float(scale) * np.asarray(delta, dtype=np.float64)
          for scale, delta in zip(leg_length_scales, source_hip_toe_deltas)
      ]

      target_root_pos = source_root_pos.copy()
      target_root_pos[2] = source_root_pos[2] - source_root_z0 + target_root_z0
      target_root_rot = _convert_root_rotation_between_robots(
          source_root_rot, source_config.INIT_ROT, target_config.INIT_ROT)

      pybullet.resetBasePositionAndOrientation(target_robot, target_root_pos, target_root_rot)
      target_world_hips = _get_link_world_positions(
          pybullet, target_robot, target_config.SIM_HIP_JOINT_IDS)
      target_world_toes = _apply_hip_toe_deltas(target_world_hips, source_hip_toe_deltas)
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
