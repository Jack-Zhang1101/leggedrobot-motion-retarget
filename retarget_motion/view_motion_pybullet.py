"""Visualize quadruped motion files in PyBullet.

This viewer is designed around the motion files used in
`/data-ssd/zhang/LeggedRobot/IsaacgymLoco/datasets`, where each frame is
typically either:

* 19 values: root pose + 12 joint angles
* 61 values: 19 pose values + toe-local targets + velocity terms
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_ISAACGYMLOCO_ROOT = REPO_ROOT.parent / "IsaacgymLoco"

if __package__ in (None, ""):
  if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
  from retarget_motion import retarget_core
else:
  from . import retarget_core


POS_SIZE = 3
ROT_SIZE = 4
JOINT_SIZE = 12
POSE_SIZE = POS_SIZE + ROT_SIZE + JOINT_SIZE
TOE_LOCAL_POS_SIZE = 12
LINEAR_VEL_SIZE = 3
ANGULAR_VEL_SIZE = 3
JOINT_VEL_SIZE = 12
TOE_LOCAL_VEL_SIZE = 12
FRAME_SIZE_19 = POSE_SIZE
FRAME_SIZE_61 = (POSE_SIZE + TOE_LOCAL_POS_SIZE + LINEAR_VEL_SIZE +
                 ANGULAR_VEL_SIZE + JOINT_VEL_SIZE + TOE_LOCAL_VEL_SIZE)
IDENTITY_ROTATION = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


@dataclass(frozen=True)
class RobotSpec:
  name: str
  urdf_path: str
  init_pos: np.ndarray
  init_rot: np.ndarray
  toe_link_ids: tuple


@dataclass(frozen=True)
class FrameView:
  pose: np.ndarray
  toe_local_pos: np.ndarray = None
  root_linear_vel: np.ndarray = None
  root_angular_vel: np.ndarray = None
  joint_vel: np.ndarray = None
  toe_local_vel: np.ndarray = None


def build_robot_specs(isaacgymloco_root):
  root = Path(isaacgymloco_root)
  return {
      "a1": RobotSpec(
          name="a1",
          urdf_path=str(root / "legged_gym" / "resources" / "robots" / "a1" / "urdf" / "a1.urdf"),
          init_pos=np.array([0.0, 0.0, 0.32], dtype=np.float64),
          init_rot=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
          toe_link_ids=(6, 11, 16, 21)),
      "go2": RobotSpec(
          name="go2",
          urdf_path=str(root / "legged_gym" / "resources" / "robots" / "go2" / "urdf" / "go2.urdf"),
          init_pos=np.array([0.0, 0.0, 0.35], dtype=np.float64),
          init_rot=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
          toe_link_ids=(7, 13, 19, 25)),
      "sizu": RobotSpec(
          name="sizu",
          urdf_path=str(root / "legged_gym" / "resources" / "robots" / "sizu" / "urdf" / "sizu.urdf"),
          init_pos=np.array([0.0, 0.0, 0.3], dtype=np.float64),
          init_rot=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
          toe_link_ids=(12, 14, 13, 15)),
      "aliengo": RobotSpec(
          name="aliengo",
          urdf_path=str(root / "legged_gym" / "resources" / "robots" / "aliengo" / "urdf" / "aliengo.urdf"),
          init_pos=np.array([0.0, 0.0, 0.5], dtype=np.float64),
          init_rot=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
          toe_link_ids=(8, 15, 22, 29)),
  }


def load_motion_payload(motion_file):
  with open(motion_file, "r") as f:
    return json.load(f)


def infer_frame_size(payload):
  frames = payload.get("Frames")
  if not frames:
    raise ValueError("motion payload has no frames")
  frame_size = len(frames[0])
  if frame_size not in (FRAME_SIZE_19, FRAME_SIZE_61):
    raise ValueError("unsupported frame size: {}".format(frame_size))
  return frame_size


def has_toe_and_velocity_data(payload):
  return infer_frame_size(payload) == FRAME_SIZE_61


def build_frame_view(frame):
  frame = np.asarray(frame, dtype=np.float64)
  if frame.shape[-1] == FRAME_SIZE_19:
    return FrameView(pose=frame[:POSE_SIZE])
  if frame.shape[-1] != FRAME_SIZE_61:
    raise ValueError("unsupported frame size: {}".format(frame.shape[-1]))

  return FrameView(
      pose=frame[:POSE_SIZE],
      toe_local_pos=frame[POSE_SIZE:(POSE_SIZE + TOE_LOCAL_POS_SIZE)],
      root_linear_vel=frame[(POSE_SIZE + TOE_LOCAL_POS_SIZE):
                            (POSE_SIZE + TOE_LOCAL_POS_SIZE + LINEAR_VEL_SIZE)],
      root_angular_vel=frame[(POSE_SIZE + TOE_LOCAL_POS_SIZE + LINEAR_VEL_SIZE):
                             (POSE_SIZE + TOE_LOCAL_POS_SIZE + LINEAR_VEL_SIZE +
                              ANGULAR_VEL_SIZE)],
      joint_vel=frame[(POSE_SIZE + TOE_LOCAL_POS_SIZE + LINEAR_VEL_SIZE +
                       ANGULAR_VEL_SIZE):
                      (POSE_SIZE + TOE_LOCAL_POS_SIZE + LINEAR_VEL_SIZE +
                       ANGULAR_VEL_SIZE + JOINT_VEL_SIZE)],
      toe_local_vel=frame[(POSE_SIZE + TOE_LOCAL_POS_SIZE + LINEAR_VEL_SIZE +
                           ANGULAR_VEL_SIZE + JOINT_VEL_SIZE):])


def _load_pybullet():
  try:
    import pybullet  # pylint: disable=import-outside-toplevel
    import pybullet_data  # pylint: disable=import-outside-toplevel
  except ImportError as exc:
    raise ImportError("pybullet is required for motion visualization") from exc
  return pybullet, pybullet_data


def _build_markers(pybullet, rgba_colors, radius):
  markers = []
  for rgba in rgba_colors:
    visual_shape = pybullet.createVisualShape(
        shapeType=pybullet.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
    marker_id = pybullet.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual_shape,
        basePosition=[0, 0, 0],
        useMaximalCoordinates=True)
    markers.append(marker_id)
  return markers


def _update_markers(pybullet, marker_ids, positions):
  for marker_id, position in zip(marker_ids, positions):
    pybullet.resetBasePositionAndOrientation(
        marker_id, position.tolist(), IDENTITY_ROTATION.tolist())


def _local_positions_to_world(pybullet, base_pos, base_rot, local_positions):
  world_positions = []
  for local_pos in local_positions:
    world_pos, _ = pybullet.multiplyTransforms(
        base_pos.tolist(),
        base_rot.tolist(),
        np.asarray(local_pos, dtype=np.float64).tolist(),
        IDENTITY_ROTATION.tolist())
    world_positions.append(np.array(world_pos, dtype=np.float64))
  return np.asarray(world_positions, dtype=np.float64)


def _toe_local_flat_to_world(pybullet, pose, toe_local_pos):
  local_positions = np.asarray(toe_local_pos, dtype=np.float64).reshape(4, 3)
  base_pos = np.asarray(retarget_core.get_root_pos(pose), dtype=np.float64)
  base_rot = np.asarray(retarget_core.get_root_rot(pose), dtype=np.float64)
  return _local_positions_to_world(pybullet, base_pos, base_rot, local_positions)


def _get_link_world_positions(pybullet, robot, link_ids):
  positions = []
  for link_id in link_ids:
    link_state = pybullet.getLinkState(robot, link_id, computeForwardKinematics=True)
    positions.append(np.array(link_state[4], dtype=np.float64))
  return np.asarray(positions, dtype=np.float64)


def _set_vector_line(pybullet,
                     vector,
                     robot_id,
                     start_local,
                     color,
                     scale,
                     unique_id=None):
  if vector is None:
    return unique_id
  start_local = np.asarray(start_local, dtype=np.float64)
  end_local = start_local + np.asarray(vector, dtype=np.float64) * scale
  kwargs = dict(
      lineFromXYZ=start_local.tolist(),
      lineToXYZ=end_local.tolist(),
      lineColorRGB=list(color),
      lineWidth=4,
      parentObjectUniqueId=robot_id)
  if unique_id is not None:
    kwargs["replaceItemUniqueId"] = unique_id
  return pybullet.addUserDebugLine(**kwargs)


def _set_toe_error_lines(pybullet, toe_world_positions, toe_link_positions, unique_ids=None):
  if unique_ids is None:
    unique_ids = [None] * len(toe_world_positions)
  new_ids = []
  for toe_world, link_world, unique_id in zip(toe_world_positions, toe_link_positions, unique_ids):
    kwargs = dict(
        lineFromXYZ=link_world.tolist(),
        lineToXYZ=toe_world.tolist(),
        lineColorRGB=[1, 0, 1],
        lineWidth=2)
    if unique_id is not None:
      kwargs["replaceItemUniqueId"] = unique_id
    new_ids.append(pybullet.addUserDebugLine(**kwargs))
  return new_ids


def _update_camera(pybullet, robot_id):
  base_pos = np.array(pybullet.getBasePositionAndOrientation(robot_id)[0], dtype=np.float64)
  yaw, pitch, dist = pybullet.getDebugVisualizerCamera()[8:11]
  pybullet.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos.tolist())


def visualize_motion(motion_file,
                     robot,
                     isaacgymloco_root=DEFAULT_ISAACGYMLOCO_ROOT,
                     playback_speed=1.0,
                     loops=0,
                     show_toe_markers=True,
                     show_velocity=True,
                     follow_camera=True,
                     show_ground=True):
  if playback_speed <= 0:
    raise ValueError("playback_speed must be positive")

  specs = build_robot_specs(isaacgymloco_root)
  if robot not in specs:
    raise ValueError("unsupported robot: {}".format(robot))
  spec = specs[robot]
  if not Path(spec.urdf_path).exists():
    raise FileNotFoundError("URDF not found: {}".format(spec.urdf_path))

  payload = load_motion_payload(motion_file)
  frame_size = infer_frame_size(payload)
  frames = np.asarray(payload["Frames"], dtype=np.float64)
  frame_duration = float(payload["FrameDuration"])

  pybullet, pybullet_data = _load_pybullet()
  client = pybullet.connect(pybullet.GUI)
  try:
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    pybullet.resetSimulation()
    pybullet.setGravity(0, 0, 0)
    if show_ground:
      pybullet.loadURDF(retarget_core.GROUND_URDF_FILENAME)

    flags = getattr(pybullet, "URDF_MAINTAIN_LINK_ORDER", 0)
    robot_id = pybullet.loadURDF(
        spec.urdf_path,
        spec.init_pos.tolist(),
        spec.init_rot.tolist(),
        flags=flags)

    toe_marker_ids = None
    toe_error_line_ids = None
    if frame_size == FRAME_SIZE_61 and show_toe_markers:
      toe_marker_ids = _build_markers(
          pybullet,
          rgba_colors=([
              [1, 0, 0, 1],
              [1, 0.4, 0, 1],
              [0, 0.7, 1, 1],
              [0.7, 0, 1, 1],
          ]),
          radius=0.025)
      toe_error_line_ids = [None] * 4

    linear_vel_line = None
    angular_vel_line = None
    loops_done = 0
    frame_index = 0

    while True:
      time_start = time.time()
      view = build_frame_view(frames[frame_index])
      retarget_core.set_pose(pybullet, robot_id, view.pose)

      if toe_marker_ids is not None and view.toe_local_pos is not None:
        toe_world_positions = _toe_local_flat_to_world(pybullet, view.pose, view.toe_local_pos)
        toe_link_positions = _get_link_world_positions(pybullet, robot_id, spec.toe_link_ids)
        _update_markers(pybullet, toe_marker_ids, toe_world_positions)
        toe_error_line_ids = _set_toe_error_lines(
            pybullet, toe_world_positions, toe_link_positions, unique_ids=toe_error_line_ids)

      if show_velocity and frame_size == FRAME_SIZE_61:
        linear_vel_line = _set_vector_line(
            pybullet,
            vector=view.root_linear_vel,
            robot_id=robot_id,
            start_local=[0.0, 0.0, 0.25],
            color=[1, 0, 0],
            scale=0.33,
            unique_id=linear_vel_line)
        angular_vel_line = _set_vector_line(
            pybullet,
            vector=view.root_angular_vel,
            robot_id=robot_id,
            start_local=[0.0, 0.0, 0.0],
            color=[0, 1, 0],
            scale=1.0,
            unique_id=angular_vel_line)

      if follow_camera:
        _update_camera(pybullet, robot_id)
      pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

      frame_index += 1
      if frame_index >= frames.shape[0]:
        frame_index = 0
        loops_done += 1
        if loops > 0 and loops_done >= loops:
          break

      sleep_duration = frame_duration / playback_speed - (time.time() - time_start)
      time.sleep(max(0.0, sleep_duration))
  finally:
    pybullet.disconnect(client)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--motion_file", required=True, help="Path to the motion JSON file.")
  parser.add_argument(
      "--robot",
      required=True,
      choices=sorted(build_robot_specs(DEFAULT_ISAACGYMLOCO_ROOT).keys()),
      help="Robot type used by the motion file.")
  parser.add_argument(
      "--isaacgymloco_root",
      default=str(DEFAULT_ISAACGYMLOCO_ROOT),
      help="Root directory of the IsaacgymLoco repository.")
  parser.add_argument(
      "--playback_speed",
      type=float,
      default=1.0,
      help="Playback speed multiplier.")
  parser.add_argument(
      "--loops",
      type=int,
      default=0,
      help="How many times to replay the clip. 0 means infinite.")
  parser.add_argument(
      "--hide_toe_markers",
      action="store_true",
      help="Disable toe target markers and toe error lines for 61dof motions.")
  parser.add_argument(
      "--hide_velocity",
      action="store_true",
      help="Disable linear/angular velocity debug lines for 61dof motions.")
  parser.add_argument(
      "--no_follow_camera",
      action="store_true",
      help="Disable automatic camera following.")
  parser.add_argument(
      "--no_ground",
      action="store_true",
      help="Do not load the ground plane.")
  args = parser.parse_args()

  visualize_motion(
      motion_file=args.motion_file,
      robot=args.robot,
      isaacgymloco_root=args.isaacgymloco_root,
      playback_speed=args.playback_speed,
      loops=args.loops,
      show_toe_markers=not args.hide_toe_markers,
      show_velocity=not args.hide_velocity,
      follow_camera=not args.no_follow_camera,
      show_ground=not args.no_ground)


if __name__ == "__main__":
  main()
