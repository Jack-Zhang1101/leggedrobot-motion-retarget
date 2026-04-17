# A1 Motion Library Design

Date: 2026-04-17

## Goal

Generate an A1 version of the current motion library in
`motion_imitation/data/motions` without overwriting the existing Laikago-oriented
reference motions.

The deliverable is a new motion set under
`motion_imitation/data/motions_a1` plus a provenance manifest that explains how
each A1 motion was produced.

## Scope

This work covers:

- generation of A1-compatible motion files for the 10 existing motions in
  `motion_imitation/data/motions`
- clear provenance labels for each output motion
- validation that the generated files load correctly and are dimensionally
  compatible with the A1 imitation pipeline

This work does not cover:

- retraining policies
- changing the default robot in `run.py`
- reconstructing missing upstream mocap sources that are not present in the repo

## Motion Categories

The implementation will classify source motions into three categories:

1. Raw retarget available:
   motions that have usable upstream `retarget_motion/data/*_joint_pos.txt`
   sources and can be regenerated for A1 from raw marker trajectories.
2. Derived motion:
   motions that can be deterministically generated from other motions already in
   the library, such as backwards variants produced by reversing time.
3. Adapted from existing motion:
   motions that do not have recoverable raw upstream data in this repo and must
   be converted from the existing motion file into an A1-compatible version.

## Output Layout

- `motion_imitation/data/motions_a1/<name>.txt`
- `motion_imitation/data/motions_a1/provenance.json`

The output filenames will mirror the existing motion names so they can be used
with the existing CLI by swapping the `--motion_file` path.

## Generation Strategy

### Raw-retarget path

For motions with available raw upstream data, the generator will reuse the
existing retarget logic but switch robot configuration to A1. This path is the
preferred source because it preserves the animal-to-robot retarget process.

### Derived-motion path

For motions that are clearly derived, the generator will create the A1 motion
from the corresponding A1 base motion using deterministic transforms. For
example, backwards motions will be produced by reversing the frame order and
renormalizing the root translation origin.

### Existing-motion adaptation path

For motions without raw upstream data, the generator will load the existing
motion file, interpret each frame as a robot pose trajectory, and convert it
into an A1-compatible pose sequence. The implementation should prefer kinematic
mapping through root pose and foot targets over ad hoc joint copying so the A1
output better matches A1 geometry and limits.

## Components

1. A motion classification table that maps each source motion to one of the
   three generation paths.
2. A reusable generator script that can emit one or all A1 motions.
3. A provenance writer that records, for each output motion:
   source motion name, generation path, and any upstream raw file or derived
   dependency used.

## Error Handling

- If a raw upstream file expected by the classification table is missing, fail
  that motion with a clear error and continue processing other motions only when
  explicitly requested by the driver script.
- If an adapted motion cannot be mapped into valid A1 joint targets, report the
  failing frame and motion name.
- Do not silently overwrite the original motion library.

## Verification

The implementation is complete only if it demonstrates:

1. every generated `motions_a1/*.txt` file loads through `MotionData`
2. every output frame size remains compatible with a 12-motor quadruped pose
3. the provenance manifest is complete for all generated motions
4. an A1-focused validation pass can load or replay the generated motions
   without immediate pose-size or state-application failures

## Success Criteria

After implementation, a user should be able to point the existing imitation
pipeline at files under `motion_imitation/data/motions_a1` and know, for each
motion, whether it came from raw A1 retarget, deterministic derivation, or
existing-motion adaptation.
