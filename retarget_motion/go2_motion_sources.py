"""Source table and simple transforms for the Go2 motion library."""

from retarget_motion import a1_motion_sources


RAW = a1_motion_sources.RAW
DERIVED = a1_motion_sources.DERIVED
ADAPTED = a1_motion_sources.ADAPTED

MOTION_SPECS = dict(a1_motion_sources.MOTION_SPECS)


def ordered_motion_names():
  return list(a1_motion_sources.ordered_motion_names())


def normalize_root_xy_origin(frames):
  return a1_motion_sources.normalize_root_xy_origin(frames)


def reverse_motion_frames(frames):
  return a1_motion_sources.reverse_motion_frames(frames)
