"""Convert external 49-value AMP frames into Isaac-style 61-value frames.

The external Go2 agile motion format stores:
  [pose(19), toe_local_pos(12), root_lin_vel(3), root_ang_vel(3), joint_vel(12)]

The target format used in this repo stores:
  [pose(19), toe_local_pos(12), root_lin_vel(3), root_ang_vel(3),
   joint_vel(12), toe_local_vel(12)]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from retarget_motion import retarget_core


POSE_SIZE = retarget_core.POSE_SIZE_19
TOE_LOCAL_POS_SIZE = retarget_core.TOE_LOCAL_POS_SIZE
LINEAR_VEL_SIZE = retarget_core.LINEAR_VEL_SIZE
ANGULAR_VEL_SIZE = retarget_core.ANGULAR_VEL_SIZE
JOINT_VEL_SIZE = retarget_core.JOINT_VEL_SIZE
TOE_LOCAL_VEL_SIZE = retarget_core.TOE_LOCAL_VEL_SIZE
SOURCE_FRAME_SIZE = (POSE_SIZE + TOE_LOCAL_POS_SIZE + LINEAR_VEL_SIZE +
                     ANGULAR_VEL_SIZE + JOINT_VEL_SIZE)
TARGET_FRAME_SIZE = retarget_core.FRAME_SIZE_61

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "motion_imitation" / "data" / "motions_go2_amp49"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "motion_imitation" / "data" / "motions_go2_61dof"


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


def convert_motion_frames(source_frames, robot_name, frame_duration):
  source_frames = _validate_source_frames(source_frames)
  config = retarget_core.load_robot_config(robot_name)

  converted_frames = np.zeros((source_frames.shape[0] - 1, TARGET_FRAME_SIZE), dtype=np.float64)

  for frame_idx in range(source_frames.shape[0] - 1):
    curr_frame = source_frames[frame_idx]
    next_frame = source_frames[frame_idx + 1]

    curr_toe_local = curr_frame[POSE_SIZE:(POSE_SIZE + TOE_LOCAL_POS_SIZE)]
    next_toe_local = next_frame[POSE_SIZE:(POSE_SIZE + TOE_LOCAL_POS_SIZE)]
    toe_local_vel = (next_toe_local - curr_toe_local) / frame_duration

    frame_61_sim = np.concatenate([
        curr_frame[:SOURCE_FRAME_SIZE],
        toe_local_vel,
    ])
    converted_frames[frame_idx] = retarget_core.sim_frame_61_to_output(frame_61_sim, config)

  converted_frames[:, 0:2] -= converted_frames[0, 0:2]
  return converted_frames


def _write_motion_file(output_path, metadata, motion_weight, frames):
  payload = dict(metadata)
  payload["MotionWeight"] = float(motion_weight)
  payload["Frames"] = np.asarray(frames, dtype=np.float64).tolist()
  with open(output_path, "w") as f:
    json.dump(payload, f, indent=2)


def convert_motion_file(input_path, output_path, robot_name="go2", motion_weight=None):
  input_path = Path(input_path)
  output_path = Path(output_path)

  payload = _load_motion_payload(input_path)
  metadata = _extract_metadata(payload)
  source_frames = _validate_source_frames(payload["Frames"])
  converted_frames = convert_motion_frames(
      source_frames=source_frames,
      robot_name=robot_name,
      frame_duration=metadata["FrameDuration"])

  resolved_motion_weight = payload.get("MotionWeight", 1.0) if motion_weight is None else motion_weight

  output_path.parent.mkdir(parents=True, exist_ok=True)
  _write_motion_file(
      output_path=output_path,
      metadata=metadata,
      motion_weight=resolved_motion_weight,
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
                             robot_name="go2",
                             motion_names=None,
                             motion_weight=None):
  input_dir = Path(input_dir)
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  results = []
  for input_path in _resolve_motion_paths(input_dir, motion_names=motion_names):
    output_path = output_dir / input_path.name
    results.append(convert_motion_file(
        input_path=input_path,
        output_path=output_path,
        robot_name=robot_name,
        motion_weight=motion_weight))
  return results


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--input_dir",
      default=str(DEFAULT_INPUT_DIR),
      help="Directory containing external 49dof AMP motions.")
  parser.add_argument(
      "--output_dir",
      default=str(DEFAULT_OUTPUT_DIR),
      help="Directory for converted 61dof motions.")
  parser.add_argument(
      "--robot",
      default="go2",
      choices=("go2",),
      help="Robot model used to map leg order into repo output format.")
  parser.add_argument(
      "--motion",
      action="append",
      dest="motions",
      help="Convert only the named motion. Can be repeated.")
  parser.add_argument(
      "--motion_weight",
      type=float,
      default=None,
      help="Override MotionWeight in the output file.")
  args = parser.parse_args()

  results = convert_motion_directory(
      input_dir=args.input_dir,
      output_dir=args.output_dir,
      robot_name=args.robot,
      motion_names=args.motions,
      motion_weight=args.motion_weight)
  print("converted", len(results), "motions into", args.output_dir)
  for result in results:
    print(result["output_path"], result["num_frames"], result["frame_size"])


if __name__ == "__main__":
  main()
