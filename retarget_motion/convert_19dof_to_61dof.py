"""Convert 19-value quadruped motion frames into Isaac-style 61-value frames.

The source motion format used in this repo stores:
  [root_pos(3), root_rot(4), joint_pose(12)] = 19 values per frame

The target format stored by IsaacgymLoco-style datasets stores:
  [pose(19), toe_local_pos(12), root_lin_vel(3), root_ang_vel(3),
   joint_vel(12), toe_local_vel(12)] = 61 values per frame
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if __package__ in (None, ""):
  if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
  from retarget_motion import retarget_core
else:
  from . import retarget_core


POS_SIZE = 3
ROT_SIZE = 4
JOINT_POS_SIZE = 12
TOE_LOCAL_POS_SIZE = 12
LINEAR_VEL_SIZE = 3
ANGULAR_VEL_SIZE = 3
JOINT_VEL_SIZE = 12
TOE_LOCAL_VEL_SIZE = 12
SOURCE_FRAME_SIZE = POS_SIZE + ROT_SIZE + JOINT_POS_SIZE
TARGET_FRAME_SIZE = (SOURCE_FRAME_SIZE + TOE_LOCAL_POS_SIZE + LINEAR_VEL_SIZE +
                     ANGULAR_VEL_SIZE + JOINT_VEL_SIZE + TOE_LOCAL_VEL_SIZE)

IDENTITY_ROTATION = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
ROBOT_CHOICES = ("a1", "go2", "sizu", "laikago", "vision60")
DEFAULT_INPUT_DIR = REPO_ROOT / "motion_imitation" / "data" / "motions_a1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "motion_imitation" / "data" / "motions_a1_61dof"


def _load_pybullet():
  return retarget_core._load_pybullet()


def _require_pybullet():
  pybullet, pybullet_data = _load_pybullet()
  if pybullet is None or pybullet_data is None:
    raise ImportError("pybullet is required for 19dof to 61dof motion conversion")
  return pybullet, pybullet_data


def _quat_inverse(quat):
  quat = np.asarray(quat, dtype=np.float64)
  return np.array([-quat[0], -quat[1], -quat[2], quat[3]], dtype=np.float64) / np.dot(quat, quat)


def _load_motion_payload(input_path):
  with open(input_path, "r") as f:
    return json.load(f)


def _extract_metadata(payload):
  return {
      "LoopMode": payload.get("LoopMode", "Wrap"),
      "FrameDuration": float(payload.get("FrameDuration", retarget_core.FRAME_DURATION)),
      "EnableCycleOffsetPosition": bool(payload.get("EnableCycleOffsetPosition", True)),
      "EnableCycleOffsetRotation": bool(payload.get("EnableCycleOffsetRotation", True)),
  }


def _validate_source_frames(frames):
  frames = np.asarray(frames, dtype=np.float64)
  if frames.ndim != 2:
    raise ValueError("source motion frames must be a 2D array")
  if frames.shape[0] < 2:
    raise ValueError("source motion must contain at least 2 frames")
  if frames.shape[1] != SOURCE_FRAME_SIZE:
    raise ValueError(
        "expected {} values per source frame, got {}".format(
            SOURCE_FRAME_SIZE, frames.shape[1]))
  return frames


def _world_positions_to_base_frame(pybullet, base_pos, base_rot, world_positions):
  inv_pos, inv_rot = pybullet.invertTransform(base_pos, base_rot)
  local_positions = []
  for world_pos in world_positions:
    local_pos, _ = pybullet.multiplyTransforms(
        inv_pos, inv_rot, world_pos.tolist(), IDENTITY_ROTATION.tolist())
    local_positions.append(np.array(local_pos, dtype=np.float64))
  return np.concatenate(local_positions, axis=0)


def _get_toe_world_positions(pybullet, robot, toe_link_ids):
  world_positions = []
  for link_id in toe_link_ids:
    link_state = pybullet.getLinkState(robot, link_id, computeForwardKinematics=True)
    world_positions.append(np.array(link_state[4], dtype=np.float64))
  return world_positions


def _get_toe_local_positions(pybullet, robot, pose, toe_link_ids):
  retarget_core.set_pose(pybullet, robot, pose)
  base_pos = np.asarray(retarget_core.get_root_pos(pose), dtype=np.float64)
  base_rot = np.asarray(retarget_core.get_root_rot(pose), dtype=np.float64)
  world_positions = _get_toe_world_positions(pybullet, robot, toe_link_ids)
  return _world_positions_to_base_frame(pybullet, base_pos, base_rot, world_positions)


def _calc_root_linear_velocity(pybullet, curr_pose, next_pose, frame_duration):
  del_linear_vel = (
      np.asarray(retarget_core.get_root_pos(next_pose), dtype=np.float64) -
      np.asarray(retarget_core.get_root_pos(curr_pose), dtype=np.float64)
  ) / frame_duration
  rot_mat = np.array(
      pybullet.getMatrixFromQuaternion(retarget_core.get_root_rot(curr_pose)),
      dtype=np.float64).reshape(3, 3)
  return np.matmul(del_linear_vel, rot_mat)


def _calc_root_angular_velocity(pybullet, curr_pose, next_pose, frame_duration, init_rot):
  curr_rot = np.asarray(retarget_core.get_root_rot(curr_pose), dtype=np.float64)
  next_rot = np.asarray(retarget_core.get_root_rot(next_pose), dtype=np.float64)

  del_angular_vel = pybullet.getDifferenceQuaternion(curr_rot.tolist(), next_rot.tolist())
  axis, angle = pybullet.getAxisAngleFromQuaternion(del_angular_vel)
  del_angular_vel = np.array(axis, dtype=np.float64) * angle / frame_duration

  inv_init_rot = _quat_inverse(init_rot)
  _, base_orientation_quat_from_init = pybullet.multiplyTransforms(
      (0, 0, 0), inv_init_rot.tolist(), (0, 0, 0), curr_rot.tolist())
  _, inverse_base_orientation = pybullet.invertTransform(
      (0, 0, 0), base_orientation_quat_from_init)
  del_angular_vel, _ = pybullet.multiplyTransforms(
      (0, 0, 0),
      inverse_base_orientation,
      del_angular_vel.tolist(),
      IDENTITY_ROTATION.tolist())
  return np.array(del_angular_vel, dtype=np.float64)


def convert_motion_frames(source_frames, robot_name, frame_duration):
  source_frames = _validate_source_frames(source_frames)
  config = retarget_core.load_robot_config(robot_name)
  pybullet, pybullet_data = _require_pybullet()

  client = pybullet.connect(pybullet.DIRECT)
  try:
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    pybullet.resetSimulation()
    robot = pybullet.loadURDF(config.URDF_FILENAME, config.INIT_POS, config.INIT_ROT)

    num_source_frames = source_frames.shape[0]
    converted_frames = np.zeros((num_source_frames - 1, TARGET_FRAME_SIZE), dtype=np.float64)

    for frame_idx in range(num_source_frames - 1):
      curr_pose = retarget_core.output_pose_to_sim(source_frames[frame_idx], config)
      next_pose = retarget_core.output_pose_to_sim(source_frames[frame_idx + 1], config)

      curr_toe_local = _get_toe_local_positions(
          pybullet, robot, curr_pose, config.SIM_TOE_JOINT_IDS)
      next_toe_local = _get_toe_local_positions(
          pybullet, robot, next_pose, config.SIM_TOE_JOINT_IDS)
      curr_toe_local = retarget_core.reorder_leg_blocks(
          curr_toe_local,
          getattr(config, "SIM_LEG_ORDER", ()),
          getattr(config, "OUTPUT_LEG_ORDER", getattr(config, "SIM_LEG_ORDER", ())),
          block_size=3) if getattr(config, "SIM_LEG_ORDER", ()) else curr_toe_local
      next_toe_local = retarget_core.reorder_leg_blocks(
          next_toe_local,
          getattr(config, "SIM_LEG_ORDER", ()),
          getattr(config, "OUTPUT_LEG_ORDER", getattr(config, "SIM_LEG_ORDER", ())),
          block_size=3) if getattr(config, "SIM_LEG_ORDER", ()) else next_toe_local

      root_linear_vel = _calc_root_linear_velocity(
          pybullet, curr_pose, next_pose, frame_duration)
      root_angular_vel = _calc_root_angular_velocity(
          pybullet, curr_pose, next_pose, frame_duration, config.INIT_ROT)
      joint_vel = (
          np.asarray(retarget_core.get_joint_pose(next_pose), dtype=np.float64) -
          np.asarray(retarget_core.get_joint_pose(curr_pose), dtype=np.float64)
      ) / frame_duration
      joint_vel = retarget_core.reorder_leg_blocks(
          joint_vel,
          getattr(config, "SIM_LEG_ORDER", ()),
          getattr(config, "OUTPUT_LEG_ORDER", getattr(config, "SIM_LEG_ORDER", ())),
          block_size=3) if getattr(config, "SIM_LEG_ORDER", ()) else joint_vel
      toe_local_vel = (next_toe_local - curr_toe_local) / frame_duration

      converted_frames[frame_idx] = np.concatenate([
          retarget_core.sim_pose_to_output(curr_pose, config),
          curr_toe_local,
          root_linear_vel,
          root_angular_vel,
          joint_vel,
          toe_local_vel,
      ])

    converted_frames[:, 0:2] -= converted_frames[0, 0:2]
    return converted_frames
  finally:
    pybullet.disconnect(client)


def _write_motion_file(output_path, metadata, motion_weight, frames):
  payload = dict(metadata)
  payload["MotionWeight"] = float(motion_weight)
  payload["Frames"] = np.asarray(frames, dtype=np.float64).tolist()
  with open(output_path, "w") as f:
    json.dump(payload, f, indent=2)


def convert_motion_file(input_path, output_path, robot_name="a1", motion_weight=1.0):
  input_path = Path(input_path)
  output_path = Path(output_path)

  payload = _load_motion_payload(input_path)
  metadata = _extract_metadata(payload)
  source_frames = _validate_source_frames(payload["Frames"])
  converted_frames = convert_motion_frames(
      source_frames=source_frames,
      robot_name=robot_name,
      frame_duration=metadata["FrameDuration"])

  output_path.parent.mkdir(parents=True, exist_ok=True)
  _write_motion_file(
      output_path=output_path,
      metadata=metadata,
      motion_weight=motion_weight,
      frames=converted_frames)

  return {
      "input_path": str(input_path),
      "output_path": str(output_path),
      "num_frames": int(converted_frames.shape[0]),
      "frame_size": int(converted_frames.shape[1]),
  }


def _resolve_motion_paths(input_dir, motion_names=None):
  input_dir = Path(input_dir)
  if motion_names:
    return [input_dir / "{}.txt".format(name) for name in motion_names]
  return sorted(
      path for path in input_dir.glob("*.txt")
      if path.is_file() and path.name != "provenance.json")


def convert_motion_directory(input_dir=DEFAULT_INPUT_DIR,
                             output_dir=DEFAULT_OUTPUT_DIR,
                             robot_name="a1",
                             motion_names=None,
                             motion_weight=1.0):
  input_dir = Path(input_dir)
  output_dir = Path(output_dir)

  results = []
  for input_path in _resolve_motion_paths(input_dir, motion_names=motion_names):
    output_path = output_dir / input_path.name
    result = convert_motion_file(
        input_path=input_path,
        output_path=output_path,
        robot_name=robot_name,
        motion_weight=motion_weight)
    results.append(result)
  return results


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--input_dir",
      default=str(DEFAULT_INPUT_DIR),
      help="Directory containing 19-value motion JSON files.")
  parser.add_argument(
      "--output_dir",
      default=str(DEFAULT_OUTPUT_DIR),
      help="Directory to write 61-value motion JSON files.")
  parser.add_argument(
      "--robot",
      choices=ROBOT_CHOICES,
      default="a1",
      help="Robot model used to reconstruct toe positions.")
  parser.add_argument(
      "--motion",
      action="append",
      dest="motions",
      help="Convert only the named motion clip. Can be repeated.")
  parser.add_argument(
      "--motion_weight",
      type=float,
      default=1.0,
      help="MotionWeight value written into the target JSON.")
  args = parser.parse_args()

  results = convert_motion_directory(
      input_dir=args.input_dir,
      output_dir=args.output_dir,
      robot_name=args.robot,
      motion_names=args.motions,
      motion_weight=args.motion_weight)

  print("converted", len(results), "motions into", args.output_dir)
  for result in results:
    print(
        Path(result["output_path"]).name,
        result["num_frames"],
        result["frame_size"])


if __name__ == "__main__":
  main()
