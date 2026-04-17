"""Reusable helpers for retargeting motion clips."""

import importlib

import numpy as np


POS_SIZE = 3
ROT_SIZE = 4
DEFAULT_ROT = np.array([0, 0, 0, 1], dtype=np.float64)
FRAME_DURATION = 0.01667
GROUND_URDF_FILENAME = "plane_implicit.urdf"
REF_COORD_ROT = np.array([0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float64)
REF_POS_OFFSET = np.array([0, 0, 0], dtype=np.float64)
REF_ROOT_ROT = np.array([0.0, 0.0, np.sin(0.47 * np.pi * 0.5), np.cos(0.47 * np.pi * 0.5)],
                        dtype=np.float64)
REF_PELVIS_JOINT_ID = 0
REF_NECK_JOINT_ID = 3
REF_HIP_JOINT_IDS = [6, 16, 11, 20]
REF_TOE_JOINT_IDS = [10, 19, 15, 23]


def _quat_normalize(q):
  q = np.asarray(q, dtype=np.float64)
  norm = np.linalg.norm(q)
  if np.isclose(norm, 0.0):
    return DEFAULT_ROT.copy()
  return q / norm


def _quat_multiply(q1, q2):
  x1, y1, z1, w1 = q1
  x2, y2, z2, w2 = q2
  return np.array([
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
  ], dtype=np.float64)


def _quat_inverse(q):
  q = np.asarray(q, dtype=np.float64)
  return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64) / np.dot(q, q)


def _quat_from_euler(roll, pitch, yaw):
  cr = np.cos(roll * 0.5)
  sr = np.sin(roll * 0.5)
  cp = np.cos(pitch * 0.5)
  sp = np.sin(pitch * 0.5)
  cy = np.cos(yaw * 0.5)
  sy = np.sin(yaw * 0.5)
  return np.array([
      sr * cp * cy - cr * sp * sy,
      cr * sp * cy + sr * cp * sy,
      cr * cp * sy - sr * sp * cy,
      cr * cp * cy + sr * sp * sy,
  ], dtype=np.float64)


def _quat_about_axis(angle, axis):
  axis = np.asarray(axis, dtype=np.float64)
  axis = axis / np.linalg.norm(axis)
  half_angle = 0.5 * angle
  return np.array([
      axis[0] * np.sin(half_angle),
      axis[1] * np.sin(half_angle),
      axis[2] * np.sin(half_angle),
      np.cos(half_angle),
  ], dtype=np.float64)


def _quat_from_matrix(matrix):
  m = np.asarray(matrix, dtype=np.float64)
  trace = np.trace(m[:3, :3])
  if trace > 0.0:
    s = np.sqrt(trace + 1.0) * 2.0
    w = 0.25 * s
    x = (m[2, 1] - m[1, 2]) / s
    y = (m[0, 2] - m[2, 0]) / s
    z = (m[1, 0] - m[0, 1]) / s
  else:
    diag = np.diag(m)
    idx = int(np.argmax(diag))
    if idx == 0:
      s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
      x = 0.25 * s
      y = (m[0, 1] + m[1, 0]) / s
      z = (m[0, 2] + m[2, 0]) / s
      w = (m[2, 1] - m[1, 2]) / s
    elif idx == 1:
      s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
      x = (m[0, 1] + m[1, 0]) / s
      y = 0.25 * s
      z = (m[1, 2] + m[2, 1]) / s
      w = (m[0, 2] - m[2, 0]) / s
    else:
      s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
      x = (m[0, 2] + m[2, 0]) / s
      y = (m[1, 2] + m[2, 1]) / s
      z = 0.25 * s
      w = (m[1, 0] - m[0, 1]) / s
  return _quat_normalize(np.array([x, y, z, w], dtype=np.float64))


def _quat_rotate_point(point, quat):
  point_quat = np.array([point[0], point[1], point[2], 0.0], dtype=np.float64)
  rotated = _quat_multiply(_quat_multiply(quat, point_quat), _quat_inverse(quat))
  return rotated[:3]


def _calc_heading(q):
  ref_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
  rot_dir = _quat_rotate_point(ref_dir, q)
  return np.arctan2(rot_dir[1], rot_dir[0])


def _calc_heading_rot(q):
  return _quat_about_axis(_calc_heading(q), [0, 0, 1])


def load_robot_config(robot_name):
  module_name = {
      "a1": "retarget_motion.retarget_config_a1",
      "laikago": "retarget_motion.retarget_config_laikago",
      "vision60": "retarget_motion.retarget_config_vision60",
  }[robot_name]
  return importlib.import_module(module_name)


def load_ref_data(filename, frame_start=None, frame_end=None):
  joint_pos_data = np.loadtxt(filename, delimiter=",")
  start_frame = 0 if frame_start is None else frame_start
  end_frame = joint_pos_data.shape[0] if frame_end is None else frame_end
  return joint_pos_data[start_frame:end_frame]


def process_ref_joint_pos_data(joint_pos, config):
  proc_pos = np.asarray(joint_pos, dtype=np.float64).copy()
  for i in range(proc_pos.shape[0]):
    curr_pos = _quat_rotate_point(proc_pos[i], REF_COORD_ROT)
    curr_pos = _quat_rotate_point(curr_pos, REF_ROOT_ROT)
    curr_pos = curr_pos * config.REF_POS_SCALE + REF_POS_OFFSET
    proc_pos[i] = curr_pos
  return proc_pos


def retarget_root_pose(ref_joint_pos, config):
  pelvis_pos = ref_joint_pos[REF_PELVIS_JOINT_ID]
  neck_pos = ref_joint_pos[REF_NECK_JOINT_ID]

  left_shoulder_pos = ref_joint_pos[REF_HIP_JOINT_IDS[0]]
  right_shoulder_pos = ref_joint_pos[REF_HIP_JOINT_IDS[2]]
  left_hip_pos = ref_joint_pos[REF_HIP_JOINT_IDS[1]]
  right_hip_pos = ref_joint_pos[REF_HIP_JOINT_IDS[3]]

  forward_dir = neck_pos - pelvis_pos
  forward_dir += getattr(config, "FORWARD_DIR_OFFSET", np.zeros(3, dtype=np.float64))
  forward_dir = forward_dir / np.linalg.norm(forward_dir)

  delta_shoulder = left_shoulder_pos - right_shoulder_pos
  delta_hip = left_hip_pos - right_hip_pos
  dir_shoulder = delta_shoulder / np.linalg.norm(delta_shoulder)
  dir_hip = delta_hip / np.linalg.norm(delta_hip)

  left_dir = 0.5 * (dir_shoulder + dir_hip)
  up_dir = np.cross(forward_dir, left_dir)
  up_dir = up_dir / np.linalg.norm(up_dir)

  left_dir = np.cross(up_dir, forward_dir)
  left_dir[2] = 0.0
  left_dir = left_dir / np.linalg.norm(left_dir)

  rot_mat = np.array([
      [forward_dir[0], left_dir[0], up_dir[0], 0.0],
      [forward_dir[1], left_dir[1], up_dir[1], 0.0],
      [forward_dir[2], left_dir[2], up_dir[2], 0.0],
      [0.0, 0.0, 0.0, 1.0],
  ], dtype=np.float64)

  root_pos = 0.5 * (pelvis_pos + neck_pos)
  root_rot = _quat_from_matrix(rot_mat)
  root_rot = _quat_multiply(root_rot, np.asarray(config.INIT_ROT, dtype=np.float64))
  root_rot = _quat_normalize(root_rot)
  return root_pos, root_rot


def _retarget_motion_frames_fallback(config, joint_pos_data):
  num_frames = joint_pos_data.shape[0]
  pose_size = POS_SIZE + ROT_SIZE + len(config.DEFAULT_JOINT_POSE)
  frames = np.zeros((num_frames, pose_size), dtype=np.float64)
  for f in range(num_frames):
    ref_joint_pos = np.reshape(joint_pos_data[f], [-1, POS_SIZE])
    ref_joint_pos = process_ref_joint_pos_data(ref_joint_pos, config)
    root_pos, root_rot = retarget_root_pose(ref_joint_pos, config)
    root_pos = root_pos + getattr(config, "SIM_ROOT_OFFSET", np.zeros(3, dtype=np.float64))
    frames[f, 0:POS_SIZE] = root_pos
    frames[f, POS_SIZE:POS_SIZE + ROT_SIZE] = root_rot
    frames[f, POS_SIZE + ROT_SIZE:] = config.DEFAULT_JOINT_POSE
  frames[:, 0:2] -= frames[0, 0:2]
  return frames


def _load_pybullet():
  try:
    import pybullet  # pylint: disable=import-outside-toplevel
    import pybullet_data as pd  # pylint: disable=import-outside-toplevel
  except ImportError:
    return None, None
  return pybullet, pd


def get_joint_limits(pybullet, robot):
  num_joints = pybullet.getNumJoints(robot)
  joint_limit_low = []
  joint_limit_high = []
  for i in range(num_joints):
    joint_info = pybullet.getJointInfo(robot, i)
    joint_type = joint_info[2]
    if joint_type in (pybullet.JOINT_PRISMATIC, pybullet.JOINT_REVOLUTE):
      joint_limit_low.append(joint_info[8])
      joint_limit_high.append(joint_info[9])
  return joint_limit_low, joint_limit_high


def get_root_pos(pose):
  return pose[0:POS_SIZE]


def get_root_rot(pose):
  return pose[POS_SIZE:(POS_SIZE + ROT_SIZE)]


def get_joint_pose(pose):
  return pose[(POS_SIZE + ROT_SIZE):]


def set_pose(pybullet, robot, pose):
  num_joints = pybullet.getNumJoints(robot)
  root_pos = get_root_pos(pose)
  root_rot = get_root_rot(pose)
  pybullet.resetBasePositionAndOrientation(robot, root_pos, root_rot)
  for j in range(num_joints):
    j_info = pybullet.getJointInfo(robot, j)
    j_state = pybullet.getJointStateMultiDof(robot, j)
    j_pose_idx = j_info[3]
    j_pose_size = len(j_state[0])
    j_vel_size = len(j_state[1])
    if j_pose_size > 0:
      j_pose = pose[j_pose_idx:(j_pose_idx + j_pose_size)]
      j_vel = np.zeros(j_vel_size)
      pybullet.resetJointStateMultiDof(robot, j, j_pose, j_vel)


def retarget_pose(pybullet, robot, config, default_pose, ref_joint_pos):
  joint_lim_low, joint_lim_high = get_joint_limits(pybullet, robot)

  root_pos, root_rot = retarget_root_pose(ref_joint_pos, config)
  root_pos = root_pos + config.SIM_ROOT_OFFSET
  pybullet.resetBasePositionAndOrientation(robot, root_pos, root_rot)

  inv_init_rot = _quat_inverse(config.INIT_ROT)
  heading_rot = _calc_heading_rot(_quat_multiply(root_rot, inv_init_rot))

  tar_toe_pos = []
  for i in range(len(REF_TOE_JOINT_IDS)):
    ref_toe_id = REF_TOE_JOINT_IDS[i]
    ref_hip_id = REF_HIP_JOINT_IDS[i]
    sim_hip_id = config.SIM_HIP_JOINT_IDS[i]
    toe_offset_local = config.SIM_TOE_OFFSET_LOCAL[i]

    ref_toe_pos = ref_joint_pos[ref_toe_id]
    ref_hip_pos = ref_joint_pos[ref_hip_id]
    hip_link_state = pybullet.getLinkState(robot, sim_hip_id, computeForwardKinematics=True)
    sim_hip_pos = np.array(hip_link_state[4], dtype=np.float64)
    toe_offset_world = _quat_rotate_point(toe_offset_local, heading_rot)

    ref_hip_toe_delta = ref_toe_pos - ref_hip_pos
    sim_tar_toe_pos = sim_hip_pos + ref_hip_toe_delta
    sim_tar_toe_pos[2] = ref_toe_pos[2]
    sim_tar_toe_pos += toe_offset_world
    tar_toe_pos.append(sim_tar_toe_pos)

  joint_pose = pybullet.calculateInverseKinematics2(
      robot,
      config.SIM_TOE_JOINT_IDS,
      tar_toe_pos,
      jointDamping=config.JOINT_DAMPING,
      lowerLimits=joint_lim_low,
      upperLimits=joint_lim_high,
      restPoses=default_pose)
  joint_pose = np.array(joint_pose, dtype=np.float64)
  return np.concatenate([root_pos, root_rot, joint_pose])


def retarget_motion_frames(robot, config, joint_pos_data, pybullet=None):
  if pybullet is None:
    pybullet, _ = _load_pybullet()
  if pybullet is None:
    return _retarget_motion_frames_fallback(config, joint_pos_data)

  num_frames = joint_pos_data.shape[0]
  for f in range(num_frames):
    ref_joint_pos = np.reshape(joint_pos_data[f], [-1, POS_SIZE])
    ref_joint_pos = process_ref_joint_pos_data(ref_joint_pos, config)
    curr_pose = retarget_pose(pybullet, robot, config, config.DEFAULT_JOINT_POSE, ref_joint_pos)
    set_pose(pybullet, robot, curr_pose)
    if f == 0:
      pose_size = curr_pose.shape[-1]
      new_frames = np.zeros([num_frames, pose_size], dtype=np.float64)
    new_frames[f] = curr_pose

  new_frames[:, 0:2] -= new_frames[0, 0:2]
  return new_frames


def retarget_joint_data(robot_name, joint_pos_data, gui=False):
  config = load_robot_config(robot_name)
  pybullet, pd = _load_pybullet()
  if pybullet is None or pd is None:
    return _retarget_motion_frames_fallback(config, joint_pos_data)

  connection_mode = pybullet.GUI if gui else pybullet.DIRECT
  client = pybullet.connect(connection_mode)
  try:
    pybullet.setAdditionalSearchPath(pd.getDataPath())
    pybullet.resetSimulation()
    pybullet.loadURDF(GROUND_URDF_FILENAME)
    robot = pybullet.loadURDF(config.URDF_FILENAME, config.INIT_POS, config.INIT_ROT)
    set_pose(pybullet, robot, np.concatenate([config.INIT_POS, config.INIT_ROT,
                                              config.DEFAULT_JOINT_POSE]))
    return retarget_motion_frames(robot, config, joint_pos_data, pybullet=pybullet)
  finally:
    pybullet.disconnect(client)


def write_motion_file(frames, out_filename):
  with open(out_filename, "w") as f:
    f.write("{\n")
    f.write("\"LoopMode\": \"Wrap\",\n")
    f.write("\"FrameDuration\": " + str(FRAME_DURATION) + ",\n")
    f.write("\"EnableCycleOffsetPosition\": true,\n")
    f.write("\"EnableCycleOffsetRotation\": true,\n")
    f.write("\n")
    f.write("\"Frames\":\n")
    f.write("[")
    for i in range(frames.shape[0]):
      curr_frame = frames[i]
      if i != 0:
        f.write(",")
      f.write("\n  [")
      for j in range(frames.shape[1]):
        curr_val = curr_frame[j]
        if j != 0:
          f.write(", ")
        f.write("%.5f" % curr_val)
      f.write("]")
    f.write("\n]")
    f.write("\n}")

