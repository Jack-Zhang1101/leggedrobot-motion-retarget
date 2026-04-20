"""Adapt existing quadruped motion files into target-robot reference motions."""

from pathlib import Path

import numpy as np

from motion_imitation.utilities import motion_data
from retarget_motion import retarget_core


IDENTITY_ROTATION = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
class AnalyticLegSpec(object):

  def __init__(self, upper_leg_length, lower_leg_length, hip_link_length):
    self.upper_leg_length = upper_leg_length
    self.lower_leg_length = lower_leg_length
    self.hip_link_length = hip_link_length


ANALYTIC_LEG_SPECS = {
    "a1": AnalyticLegSpec(
        upper_leg_length=0.2,
        lower_leg_length=0.2,
        hip_link_length=0.08505),
    "go2": AnalyticLegSpec(
        upper_leg_length=0.213,
        lower_leg_length=0.213,
        hip_link_length=0.0955),
    "sizu": AnalyticLegSpec(
        upper_leg_length=0.295833,
        lower_leg_length=0.280714,
        hip_link_length=0.0105),
}


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

    source_scale_joint_pose = np.asarray(
        getattr(source_config, "ADAPT_SCALE_JOINT_POSE", source_config.DEFAULT_JOINT_POSE),
        dtype=np.float64)
    target_scale_joint_pose = np.asarray(
        getattr(target_config, "ADAPT_SCALE_JOINT_POSE", target_config.DEFAULT_JOINT_POSE),
        dtype=np.float64)

    retarget_core.set_pose(pybullet, source_robot, np.concatenate([
        source_config.INIT_POS, source_config.INIT_ROT, source_scale_joint_pose]))
    retarget_core.set_pose(pybullet, target_robot, np.concatenate([
        target_config.INIT_POS, target_config.INIT_ROT, target_scale_joint_pose]))

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


def _smooth_series(values, window):
  values = np.asarray(values, dtype=np.float64)
  if window is None or window <= 1 or values.shape[0] <= 2:
    return values.copy()

  window = int(window)
  if window % 2 == 0:
    window += 1

  half = window // 2
  weights = np.arange(1, half + 2, dtype=np.float64)
  weights = np.concatenate([weights, weights[-2::-1]])
  weights /= np.sum(weights)

  padded = np.pad(values, ((half, half), (0, 0)), mode="edge")
  smoothed = np.empty_like(values, dtype=np.float64)
  for col in range(values.shape[1]):
    smoothed[:, col] = np.convolve(padded[:, col], weights, mode="valid")
  return smoothed


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


def _get_joint_leg_reorder(pybullet, robot, sim_hip_joint_ids):
  q_indices = []
  for leg_idx, hip_joint_id in enumerate(sim_hip_joint_ids):
    joint_info = pybullet.getJointInfo(robot, hip_joint_id)
    q_indices.append((joint_info[3], leg_idx))
  q_indices.sort()
  return [leg_idx for _, leg_idx in q_indices]


def _project_foot_position_to_knee_limits(foot_position_in_hip,
                                          knee_low,
                                          knee_high,
                                          upper_leg_length,
                                          lower_leg_length,
                                          hip_link_length):
  foot_position_in_hip = np.asarray(foot_position_in_hip, dtype=np.float64)
  radius_sq = float(np.dot(foot_position_in_hip, foot_position_in_hip))
  if radius_sq <= 0.0:
    return foot_position_in_hip

  min_radius_sq = (
      hip_link_length**2 + upper_leg_length**2 + lower_leg_length**2 +
      2.0 * upper_leg_length * lower_leg_length * np.cos(knee_low))
  max_radius_sq = (
      hip_link_length**2 + upper_leg_length**2 + lower_leg_length**2 +
      2.0 * upper_leg_length * lower_leg_length * np.cos(knee_high))
  clipped_radius_sq = float(np.clip(radius_sq, min_radius_sq, max_radius_sq))
  if np.isclose(clipped_radius_sq, radius_sq):
    return foot_position_in_hip

  scale = np.sqrt(clipped_radius_sq / radius_sq)
  return foot_position_in_hip * scale


def _solve_analytic_leg_joint_pose(foot_position_in_hip,
                                   l_hip_sign,
                                   joint_low,
                                   joint_high,
                                   leg_spec):
  foot_position_in_hip = _project_foot_position_to_knee_limits(
      foot_position_in_hip,
      knee_low=joint_low[2],
      knee_high=joint_high[2],
      upper_leg_length=leg_spec.upper_leg_length,
      lower_leg_length=leg_spec.lower_leg_length,
      hip_link_length=leg_spec.hip_link_length)

  l_up = leg_spec.upper_leg_length
  l_low = leg_spec.lower_leg_length
  l_hip = leg_spec.hip_link_length * l_hip_sign
  x, y, z = foot_position_in_hip

  cos_knee = (
      x**2 + y**2 + z**2 - l_hip**2 - l_low**2 - l_up**2) / (2.0 * l_low * l_up)
  cos_knee = float(np.clip(cos_knee, -1.0, 1.0))
  theta_knee = -np.arccos(cos_knee)

  leg_length = np.sqrt(l_up**2 + l_low**2 + 2.0 * l_up * l_low * np.cos(theta_knee))
  sin_hip = float(np.clip(-x / leg_length, -1.0, 1.0))
  theta_hip = np.arcsin(sin_hip) - theta_knee / 2.0
  c1 = l_hip * y - leg_length * np.cos(theta_hip + theta_knee / 2.0) * z
  s1 = leg_length * np.cos(theta_hip + theta_knee / 2.0) * y + l_hip * z
  theta_ab = np.arctan2(s1, c1)

  return np.clip(
      np.array([theta_ab, theta_hip, theta_knee], dtype=np.float64),
      np.asarray(joint_low, dtype=np.float64),
      np.asarray(joint_high, dtype=np.float64))


def _solve_analytic_joint_pose(base_pos,
                               base_rot,
                               target_world_toe_positions,
                               target_world_hip_positions,
                               joint_low,
                               joint_high,
                               pybullet,
                               target_robot,
                               target_config,
                               leg_spec):
  target_local_toes = _world_positions_to_base_frame(
      pybullet, base_pos, base_rot, target_world_toe_positions)
  target_local_hips = _world_positions_to_base_frame(
      pybullet, base_pos, base_rot, target_world_hip_positions)
  joint_leg_order = _get_joint_leg_reorder(
      pybullet, target_robot, target_config.SIM_HIP_JOINT_IDS)
  target_local_toes = [
      np.asarray(target_local_toes[idx], dtype=np.float64)
      for idx in joint_leg_order
  ]
  target_local_hips = [
      np.asarray(target_local_hips[idx], dtype=np.float64)
      for idx in joint_leg_order
  ]

  joint_pose = []
  for leg_idx, (target_local_toe, target_local_hip) in enumerate(
      zip(target_local_toes, target_local_hips)):
    hip_sign = 1.0 if target_local_hip[1] >= 0.0 else -1.0
    foot_position_in_hip = target_local_toe - target_local_hip
    leg_joint_pose = _solve_analytic_leg_joint_pose(
        foot_position_in_hip,
        l_hip_sign=hip_sign,
        joint_low=joint_low[leg_idx],
        joint_high=joint_high[leg_idx],
        leg_spec=leg_spec)
    joint_pose.append(np.asarray(leg_joint_pose, dtype=np.float64))

  return np.concatenate(joint_pose, axis=0)


def _solve_target_robot_joint_pose(pybullet,
                                   target_robot,
                                   target_config,
                                   target_robot_name,
                                   target_root_pos,
                                   target_root_rot,
                                   target_world_hip_positions,
                                   target_world_toe_positions,
                                   target_joint_low,
                                   target_joint_high):
  use_analytic = getattr(target_config, "USE_ANALYTIC_ADAPT_IK", True)
  analytic_leg_spec = ANALYTIC_LEG_SPECS.get(target_robot_name)
  if use_analytic and analytic_leg_spec is not None:
    return _solve_analytic_joint_pose(
        target_root_pos,
        target_root_rot,
        target_world_toe_positions,
        target_world_hip_positions,
        target_joint_low,
        target_joint_high,
        pybullet,
        target_robot,
        target_config,
        analytic_leg_spec)
  return _solve_target_joint_pose(
      pybullet, target_robot, target_config, target_world_toe_positions)


def _smooth_joint_pose_frames(frames, window):
  frames = np.asarray(frames, dtype=np.float64)
  if window is None or window <= 1:
    return frames.copy()

  smoothed = np.array(frames, dtype=np.float64, copy=True)
  smoothed[:, 7:19] = _smooth_series(smoothed[:, 7:19], window)
  return smoothed


def _adjust_frames_root_height_for_contact(pybullet, target_robot, target_config, frames):
  target_clearance = getattr(target_config, "ADAPT_FOOT_CLEARANCE_TARGET", None)
  if target_clearance is None:
    return np.asarray(frames, dtype=np.float64)

  frames = np.array(frames, dtype=np.float64, copy=True)
  foot_bottom_offset = float(getattr(target_config, "ADAPT_FOOT_BOTTOM_OFFSET", 0.0))
  foot_bottom_heights = []

  for frame in frames:
    retarget_core.set_pose(pybullet, target_robot, frame)
    for toe_id in target_config.SIM_TOE_JOINT_IDS:
      toe_center_z = pybullet.getLinkState(
          target_robot, toe_id, computeForwardKinematics=True)[4][2]
      foot_bottom_heights.append(toe_center_z - foot_bottom_offset)

  foot_bottom_heights = np.asarray(foot_bottom_heights, dtype=np.float64)
  clearance_stat = str(
      getattr(target_config, "ADAPT_FOOT_CLEARANCE_STAT", "min")).lower()
  if clearance_stat == "percentile":
    percentile = float(getattr(target_config, "ADAPT_FOOT_CLEARANCE_PERCENTILE", 5.0))
    anchor_height = np.percentile(foot_bottom_heights, percentile)
  else:
    anchor_height = np.min(foot_bottom_heights)

  root_height_delta = float(target_clearance) - float(anchor_height)
  frames[:, 2] += root_height_delta
  return frames


def adapt_motion_to_robot(source_motion,
                          target_robot_name="a1",
                          source_robot_name="laikago",
                          target_root_height_offset=None,
                          leg_scale_multiplier=1.0,
                          joint_smoothing_window=None):
  if not isinstance(source_motion, motion_data.MotionData):
    raise TypeError("source_motion must be a MotionData instance")

  pybullet, pybullet_data = _require_pybullet()
  source_config = retarget_core.load_robot_config(source_robot_name)
  target_config = retarget_core.load_robot_config(target_robot_name)
  leg_remap = _compute_leg_remap(source_robot_name, target_robot_name)
  leg_length_scales = _compute_leg_length_scales(
      source_robot_name, target_robot_name, remap=leg_remap)

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
    target_joint_low, target_joint_high = retarget_core.get_joint_limits(pybullet, target_robot)
    target_joint_low = np.asarray(target_joint_low, dtype=np.float64).reshape(4, 3)
    target_joint_high = np.asarray(target_joint_high, dtype=np.float64).reshape(4, 3)

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
          float(scale) * float(leg_scale_multiplier) * np.asarray(delta, dtype=np.float64)
          for scale, delta in zip(leg_length_scales, source_hip_toe_deltas)
      ]

      target_root_pos = source_root_pos.copy()
      target_root_pos[2] = source_root_pos[2] - source_root_z0 + target_root_z0
      if target_root_height_offset is None:
        target_root_pos[2] += float(getattr(target_config, "ADAPT_ROOT_HEIGHT_OFFSET", 0.0))
      else:
        target_root_pos[2] += float(target_root_height_offset)
      target_root_rot = _convert_root_rotation_between_robots(
          source_root_rot, source_config.INIT_ROT, target_config.INIT_ROT)

      pybullet.resetBasePositionAndOrientation(target_robot, target_root_pos, target_root_rot)
      target_world_hips = _get_link_world_positions(
          pybullet, target_robot, target_config.SIM_HIP_JOINT_IDS)
      target_world_toes = _apply_hip_toe_deltas(target_world_hips, source_hip_toe_deltas)
      target_joint_pose = _solve_target_robot_joint_pose(
          pybullet,
          target_robot,
          target_config,
          target_robot_name,
          target_root_pos,
          target_root_rot,
          target_world_hips,
          target_world_toes,
          target_joint_low,
          target_joint_high)
      target_frame_sim = np.concatenate([target_root_pos, target_root_rot, target_joint_pose])
      frames.append(target_frame_sim)

    frames = _smooth_joint_pose_frames(frames, joint_smoothing_window)
    frames = _adjust_frames_root_height_for_contact(
        pybullet, target_robot, target_config, frames)
    return np.asarray([
        retarget_core.sim_pose_to_output(frame, target_config) for frame in frames
    ], dtype=np.float64)
  finally:
    pybullet.disconnect(client)


def adapt_motion_to_a1(source_motion):
  return adapt_motion_to_robot(source_motion, target_robot_name="a1", source_robot_name="laikago")


def load_and_adapt_motion_file(source_motion_path,
                               target_robot_name="a1",
                               source_robot_name="laikago",
                               target_root_height_offset=None,
                               leg_scale_multiplier=1.0,
                               joint_smoothing_window=None):
  motion = motion_data.MotionData(str(Path(source_motion_path)))
  return adapt_motion_to_robot(
      motion,
      target_robot_name=target_robot_name,
      source_robot_name=source_robot_name,
      target_root_height_offset=target_root_height_offset,
      leg_scale_multiplier=leg_scale_multiplier,
      joint_smoothing_window=joint_smoothing_window)
