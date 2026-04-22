"""Source table and simple transforms for the Go2 motion library."""

from retarget_motion import a1_motion_sources


RAW = a1_motion_sources.RAW
DERIVED = a1_motion_sources.DERIVED
ADAPTED = a1_motion_sources.ADAPTED
EXTERNAL_AMP49 = "external_amp49"

MOTION_SPECS = dict(a1_motion_sources.MOTION_SPECS)
MOTION_SPECS["turn"] = dict(
    MOTION_SPECS["turn"],
    adapt_root_height_offset=-0.02)
AGILE_AMP49_DIR = a1_motion_sources.REPO_ROOT / "motion_imitation" / "data" / "motions_go2_amp49"
MOTION_SPECS.update({
    "quad_backflip": {
        "mode": EXTERNAL_AMP49,
        "source_motion_path": AGILE_AMP49_DIR / "quad_backflip.txt",
        "metadata_motion_path": AGILE_AMP49_DIR / "quad_backflip.meta.json",
    },
    "quad_sideflip": {
        "mode": EXTERNAL_AMP49,
        "source_motion_path": AGILE_AMP49_DIR / "quad_sideflip.txt",
        "metadata_motion_path": AGILE_AMP49_DIR / "quad_sideflip.meta.json",
    },
    "quad_jump_forward_1m": {
        "mode": EXTERNAL_AMP49,
        "source_motion_path": AGILE_AMP49_DIR / "quad_jump_forward_1m.txt",
        "metadata_motion_path": AGILE_AMP49_DIR / "quad_jump_forward_1m.meta.json",
    },
    "spin_inplace": {
        "mode": EXTERNAL_AMP49,
        "source_motion_path": AGILE_AMP49_DIR / "spin_inplace.txt",
        "metadata_motion_path": AGILE_AMP49_DIR / "spin_inplace.meta.json",
    },
    "spin_inplace_ccw": {
        "mode": EXTERNAL_AMP49,
        "source_motion_path": AGILE_AMP49_DIR / "spin_inplace_ccw.txt",
        "metadata_motion_path": AGILE_AMP49_DIR / "spin_inplace_ccw.meta.json",
    },
})


def ordered_motion_names():
  return list(a1_motion_sources.ordered_motion_names())


def ordered_agile_motion_names():
  return [
      "quad_backflip",
      "quad_sideflip",
      "quad_jump_forward_1m",
      "spin_inplace",
      "spin_inplace_ccw",
  ]


def normalize_root_xy_origin(frames):
  return a1_motion_sources.normalize_root_xy_origin(frames)


def reverse_motion_frames(frames):
  return a1_motion_sources.reverse_motion_frames(frames)
