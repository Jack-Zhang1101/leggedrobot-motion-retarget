"""Import external Go2 agile AMP motions into the local 61dof motion library."""

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if __package__ in (None, ""):
  if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
  from motion_imitation.utilities import motion_data
  from retarget_motion import convert_amp_49dof_to_61dof
  from retarget_motion import go2_motion_sources
else:
  from motion_imitation.utilities import motion_data
  from retarget_motion import convert_amp_49dof_to_61dof
  from retarget_motion import go2_motion_sources

DEFAULT_OUTPUT_DIR = REPO_ROOT / "motion_imitation" / "data" / "motions_go2_61dof"


def _load_existing_provenance(provenance_path):
  if not provenance_path.exists():
    return {}
  with open(provenance_path, "r") as f:
    return json.load(f)


def generate_library(output_dir=DEFAULT_OUTPUT_DIR, motion_names=None):
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  selected = motion_names or go2_motion_sources.ordered_agile_motion_names()
  provenance_path = output_dir / "provenance.json"
  provenance = _load_existing_provenance(provenance_path)

  for name in selected:
    spec = go2_motion_sources.MOTION_SPECS[name]
    if spec["mode"] != go2_motion_sources.EXTERNAL_AMP49:
      raise ValueError("motion {} is not an external_amp49 spec".format(name))

    output_path = output_dir / "{}.txt".format(name)
    convert_amp_49dof_to_61dof.convert_motion_file(
        input_path=spec["source_motion_path"],
        output_path=output_path,
        robot_name="go2",
        motion_weight=spec.get("motion_weight"))
    provenance[name] = {
        "mode": spec["mode"],
        "source_motion_path": str(spec["source_motion_path"]),
        "metadata_motion_path": (
            str(spec["metadata_motion_path"]) if "metadata_motion_path" in spec else None),
        "output_path": str(output_path),
    }

  with open(provenance_path, "w") as f:
    json.dump(provenance, f, indent=2, sort_keys=True)
  return {name: provenance[name] for name in selected}


def validate_library(output_dir=DEFAULT_OUTPUT_DIR, motion_names=None):
  output_dir = Path(output_dir)
  selected = motion_names or go2_motion_sources.ordered_agile_motion_names()
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
      help="Output directory for imported Go2 agile 61dof motions.")
  parser.add_argument(
      "--motion",
      action="append",
      dest="motions",
      help="Import only the named motion. Can be repeated.")
  args = parser.parse_args()

  output_dir = Path(args.output_dir)
  provenance = generate_library(output_dir=output_dir, motion_names=args.motions)
  validation = validate_library(output_dir=output_dir, motion_names=args.motions)

  print("imported", len(provenance), "motions into", output_dir)
  for result in validation:
    print(result["motion"], result["num_frames"], result["frame_size"])


if __name__ == "__main__":
  main()
