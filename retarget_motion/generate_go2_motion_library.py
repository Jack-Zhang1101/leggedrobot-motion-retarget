"""Generate a Go2-compatible motion library."""

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
  from motion_imitation.utilities import motion_data
  from retarget_motion import go2_motion_sources
  from retarget_motion import motion_adaptation
  from retarget_motion import retarget_core
else:
  from motion_imitation.utilities import motion_data
  from retarget_motion import go2_motion_sources
  from retarget_motion import motion_adaptation
  from retarget_motion import retarget_core

DEFAULT_OUTPUT_DIR = REPO_ROOT / "motion_imitation" / "data" / "motions_go2"


def _load_motion_template(template_path):
  if template_path is None:
    return None
  template_path = Path(template_path)
  if not template_path.exists():
    return None
  with open(template_path, "r") as f:
    return json.load(f)


def _write_motion_file_with_template(frames, output_path, template=None):
  metadata = {
      "LoopMode": "Wrap",
      "FrameDuration": retarget_core.FRAME_DURATION,
      "EnableCycleOffsetPosition": True,
      "EnableCycleOffsetRotation": True,
  }
  if template is not None:
    metadata["LoopMode"] = template.get("LoopMode", metadata["LoopMode"])
    metadata["FrameDuration"] = template.get("FrameDuration", metadata["FrameDuration"])
    metadata["EnableCycleOffsetPosition"] = template.get(
        "EnableCycleOffsetPosition", metadata["EnableCycleOffsetPosition"])
    metadata["EnableCycleOffsetRotation"] = template.get(
        "EnableCycleOffsetRotation", metadata["EnableCycleOffsetRotation"])

  payload = dict(metadata)
  payload["Frames"] = np.asarray(frames, dtype=np.float64).tolist()
  with open(output_path, "w") as f:
    json.dump(payload, f, indent=2)


def _generate_raw_motion(spec):
  joint_pos_data = retarget_core.load_ref_data(str(spec["joint_pos_path"]))
  return retarget_core.retarget_joint_data("go2", joint_pos_data, gui=False)


def _generate_adapted_motion(spec):
  return motion_adaptation.load_and_adapt_motion_file(
      spec["source_motion_path"],
      target_robot_name="go2",
      target_root_height_offset=spec.get("adapt_root_height_offset"))


def _generate_motion(name, generated_cache):
  if name in generated_cache:
    return generated_cache[name]

  spec = go2_motion_sources.MOTION_SPECS[name]
  mode = spec["mode"]
  if mode == go2_motion_sources.RAW:
    frames = _generate_raw_motion(spec)
  elif mode == go2_motion_sources.DERIVED:
    base_frames = _generate_motion(spec["source_motion"], generated_cache)
    frames = go2_motion_sources.reverse_motion_frames(base_frames)
  elif mode == go2_motion_sources.ADAPTED:
    frames = _generate_adapted_motion(spec)
  else:
    raise ValueError("Unsupported motion mode: {}".format(mode))

  generated_cache[name] = np.asarray(frames, dtype=np.float64)
  return generated_cache[name]


def generate_library(output_dir=DEFAULT_OUTPUT_DIR, motion_names=None):
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  selected = motion_names or go2_motion_sources.ordered_motion_names()
  generated_cache = {}
  provenance = {}

  for name in selected:
    spec = go2_motion_sources.MOTION_SPECS[name]
    frames = _generate_motion(name, generated_cache)
    template = _load_motion_template(spec.get("metadata_motion_path"))
    output_path = output_dir / "{}.txt".format(name)
    _write_motion_file_with_template(frames, output_path, template=template)
    provenance[name] = {
        "mode": spec["mode"],
        "joint_pos_path": str(spec["joint_pos_path"]) if "joint_pos_path" in spec else None,
        "source_motion": spec.get("source_motion"),
        "source_motion_path": (
            str(spec["source_motion_path"]) if "source_motion_path" in spec else None),
        "adapt_root_height_offset": spec.get("adapt_root_height_offset"),
        "metadata_motion_path": (
            str(spec["metadata_motion_path"]) if "metadata_motion_path" in spec else None),
        "output_path": str(output_path),
    }

  provenance_path = output_dir / "provenance.json"
  with open(provenance_path, "w") as f:
    json.dump(provenance, f, indent=2, sort_keys=True)
  return provenance


def validate_library(output_dir=DEFAULT_OUTPUT_DIR, motion_names=None):
  output_dir = Path(output_dir)
  selected = motion_names or go2_motion_sources.ordered_motion_names()
  results = []
  for name in selected:
    motion_path = output_dir / "{}.txt".format(name)
    motion = motion_data.MotionData(str(motion_path))
    results.append({
        "motion": name,
        "path": str(motion_path),
        "num_frames": motion.get_num_frames(),
        "frame_size": motion.get_frame_size(),
    })
  return results


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--output_dir",
      default=str(DEFAULT_OUTPUT_DIR),
      help="Output directory for generated Go2 motions.")
  parser.add_argument(
      "--motion",
      action="append",
      dest="motions",
      help="Generate only the named motion. Can be repeated.")
  args = parser.parse_args()

  output_dir = Path(args.output_dir)
  provenance = generate_library(output_dir=output_dir, motion_names=args.motions)
  validation = validate_library(output_dir=output_dir, motion_names=args.motions)

  print("generated", len(provenance), "motions into", output_dir)
  for result in validation:
    print(result["motion"], result["num_frames"], result["frame_size"])


if __name__ == "__main__":
  main()
