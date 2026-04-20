"""Source table and simple transforms for the Sizu motion library."""

from retarget_motion import a1_motion_sources


RAW = a1_motion_sources.RAW
DERIVED = a1_motion_sources.DERIVED
ADAPTED = a1_motion_sources.ADAPTED

MOTION_SPECS = dict(a1_motion_sources.MOTION_SPECS)
for motion_name in (
    "hopturn",
    "inplace_steps",
    "runningman",
    "sidesteps",
    "turn",
):
  MOTION_SPECS[motion_name] = dict(
      MOTION_SPECS[motion_name],
      adapt_joint_smoothing_window=5)

MOTION_SPECS["hopturn"] = dict(
    MOTION_SPECS["hopturn"],
    adapt_leg_scale_multiplier=0.8)
MOTION_SPECS["inplace_steps"] = dict(
    MOTION_SPECS["inplace_steps"],
    adapt_leg_scale_multiplier=0.8)
MOTION_SPECS["runningman"] = dict(
    MOTION_SPECS["runningman"],
    adapt_leg_scale_multiplier=0.8,
    adapt_joint_smoothing_window=7)
MOTION_SPECS["sidesteps"] = dict(
    MOTION_SPECS["sidesteps"],
    adapt_leg_scale_multiplier=0.8)
MOTION_SPECS["turn"] = dict(
    MOTION_SPECS["turn"],
    adapt_root_height_offset=0.35,
    adapt_leg_scale_multiplier=0.8)


def ordered_motion_names():
  return list(a1_motion_sources.ordered_motion_names())


def normalize_root_xy_origin(frames):
  return a1_motion_sources.normalize_root_xy_origin(frames)


def reverse_motion_frames(frames):
  return a1_motion_sources.reverse_motion_frames(frames)
