"""Standalone visualization script backed by the shared retarget core."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if __package__ in (None, ""):
  parentdir = os.path.dirname(SCRIPT_DIR)
  if parentdir not in sys.path:
    sys.path.insert(0, parentdir)
  from retarget_motion import retarget_core
else:
  from . import retarget_core


DEFAULT_ROBOT_NAME = "laikago"
MOCAP_MOTIONS = [
    ["pace", "data/dog_walk00_joint_pos.txt", 162, 201],
    ["trot", "data/dog_walk03_joint_pos.txt", 448, 481],
    ["trot2", "data/dog_run04_joint_pos.txt", 630, 663],
    ["canter", "data/dog_run00_joint_pos.txt", 430, 459],
    ["left turn0", "data/dog_walk09_joint_pos.txt", 1085, 1124],
    ["right turn0", "data/dog_walk09_joint_pos.txt", 2404, 2450],
]


def _load_pybullet_modules():
  import pybullet  # pylint: disable=import-outside-toplevel
  import pybullet_data as pybullet_data  # pylint: disable=import-outside-toplevel
  return pybullet, pybullet_data


def _motion_data_path(filename):
  return os.path.join(SCRIPT_DIR, filename)


def build_markers(pybullet, num_markers):
  marker_radius = 0.02
  markers = []
  for i in range(num_markers):
    if (i == retarget_core.REF_NECK_JOINT_ID or
        i == retarget_core.REF_PELVIS_JOINT_ID or
        i in retarget_core.REF_HIP_JOINT_IDS):
      color = [0, 0, 1, 1]
    elif i in retarget_core.REF_TOE_JOINT_IDS:
      color = [1, 0, 0, 1]
    else:
      color = [0, 1, 0, 1]

    visual_shape = pybullet.createVisualShape(
        shapeType=pybullet.GEOM_SPHERE,
        radius=marker_radius,
        rgbaColor=color)
    marker_id = pybullet.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual_shape,
        basePosition=[0, 0, 0],
        useMaximalCoordinates=True)
    markers.append(marker_id)
  return markers


def set_marker_pos(pybullet, marker_pos, marker_ids):
  assert len(marker_ids) == marker_pos.shape[0]
  for marker_id, curr_pos in zip(marker_ids, marker_pos):
    pybullet.resetBasePositionAndOrientation(
        marker_id, curr_pos, retarget_core.DEFAULT_ROT)


def update_camera(pybullet, robot):
  base_pos = np.array(pybullet.getBasePositionAndOrientation(robot)[0])
  yaw, pitch, dist = pybullet.getDebugVisualizerCamera()[8:11]
  pybullet.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)


def main(argv=None):
  del argv

  pybullet, pybullet_data = _load_pybullet_modules()
  config = retarget_core.load_robot_config(DEFAULT_ROBOT_NAME)
  client = pybullet.connect(
      pybullet.GUI,
      options="--width=1920 --height=1080 --mp4=\"test.mp4\" --mp4fps=60")

  try:
    pybullet.configureDebugVisualizer(
        pybullet.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

    while True:
      for motion_name, filename, frame_start, frame_end in MOCAP_MOTIONS:
        pybullet.resetSimulation()
        pybullet.setGravity(0, 0, 0)
        pybullet.loadURDF(retarget_core.GROUND_URDF_FILENAME)
        robot = pybullet.loadURDF(
            config.URDF_FILENAME, config.INIT_POS, config.INIT_ROT)
        retarget_core.set_pose(
            pybullet,
            robot,
            np.concatenate(
                [config.INIT_POS, config.INIT_ROT, config.DEFAULT_JOINT_POSE]))

        pybullet.removeAllUserDebugItems()
        print("mocap_name=", motion_name)
        joint_pos_data = retarget_core.load_ref_data(
            _motion_data_path(filename), frame_start, frame_end)
        marker_ids = build_markers(
            pybullet, joint_pos_data.shape[-1] // retarget_core.POS_SIZE)

        retarget_frames = retarget_core.retarget_motion_frames(
            robot, config, joint_pos_data, pybullet=pybullet)
        retarget_core.write_motion_file(retarget_frames, "{}.txt".format(motion_name))

        num_frames = joint_pos_data.shape[0]
        for frame_idx in range(5 * num_frames):
          time_start = time.time()

          pose_idx = frame_idx % num_frames
          print("Frame {:d}".format(pose_idx))

          ref_joint_pos = np.reshape(
              joint_pos_data[pose_idx], [-1, retarget_core.POS_SIZE])
          ref_joint_pos = retarget_core.process_ref_joint_pos_data(
              ref_joint_pos, config)

          retarget_core.set_pose(pybullet, robot, retarget_frames[pose_idx])
          set_marker_pos(pybullet, ref_joint_pos, marker_ids)
          update_camera(pybullet, robot)
          pybullet.configureDebugVisualizer(
              pybullet.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

          sleep_dur = retarget_core.FRAME_DURATION - (time.time() - time_start)
          time.sleep(max(0, sleep_dur))

        for marker_id in marker_ids:
          pybullet.removeBody(marker_id)
  finally:
    pybullet.disconnect(client)


if __name__ == "__main__":
  main()
