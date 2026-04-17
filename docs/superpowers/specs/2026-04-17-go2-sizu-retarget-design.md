# Go2 And Sizu Retarget Design

**Goal**

Migrate the `go2` and `sizu` keypoint-retarget configuration and generation
pipeline from `IsaacgymLoco/datasets` into this repository, using only
repo-local robot assets and the current retarget/visualization stack. The
resulting pipeline must produce both 19 DoF motion files and 61 DoF motion
files compatible with the existing viewer.

**Context**

This repository has already been reduced to a retarget-and-visualization-only
scope. It already contains:

- repo-local robot assets under `assets/robots/`
- a reusable retarget core in `retarget_motion/retarget_core.py`
- a 19 DoF to 61 DoF converter in
  `retarget_motion/convert_19dof_to_61dof.py`
- a PyBullet viewer in `retarget_motion/view_motion_pybullet.py`

The missing piece is a repo-local keypoint-retarget pipeline for `go2` and
`sizu`. The user explicitly does not want a blind motion-data migration. The
important part is the retarget configuration and pipeline, with 61 DoF outputs
available at the end.

## Architecture

The repo will keep one retarget mainline:

`keypoint joint positions -> 19 DoF retarget motion -> 61 DoF motion`

`go2`, `sizu`, `a1`, `laikago`, and `vision60` should all share the same
retarget core entry points instead of maintaining separate implementations.

The `go2` and `sizu` migration will therefore:

- add robot-specific config modules under `retarget_motion/`
- extend `retarget_motion/retarget_core.py` to load those configs
- add a repo-local keypoint-retarget CLI under `retarget_motion/`
- reuse the existing 19 DoF to 61 DoF converter
- reuse the existing PyBullet motion viewer

## Robot Assets

All robot asset references must be repo-local.

- `go2` uses `assets/robots/go2/urdf/go2.urdf`
- `sizu` uses `assets/robots/sizu/urdf/sizu.urdf`

No runtime dependency on external `legged_gym` asset directories is allowed for
the migrated pipeline.

## Configuration Migration

The migrated `go2` and `sizu` config modules should preserve the meaning of the
upstream parameters while replacing external path assumptions with repo-local
paths.

The important fields are:

- `URDF_FILENAME`
- `OUTPUT_DIR`
- `REF_POS_SCALE`
- `INIT_POS`
- `INIT_ROT`
- `SIM_TOE_JOINT_IDS`
- `SIM_HIP_JOINT_IDS`
- `SIM_ROOT_OFFSET`
- `SIM_TOE_OFFSET_LOCAL`
- `TOE_HEIGHT_OFFSET`
- `DEFAULT_JOINT_POSE`
- `JOINT_DAMPING`
- `FORWARD_DIR_OFFSET`
- foot-name mappings
- `MOCAP_MOTIONS`

### Sizu Leg Order Requirement

`sizu` must keep the leg order `FL, FR, RL, RR`.

The upstream note is critical:

- `right1` is right-front
- `right2` is right-rear

Those legs must not be swapped. This constraint should be encoded directly in
the config comments and reflected in toe and hip link index assignments.

## File Layout

The migration should stay concentrated in `retarget_motion/`.

### New Files

- `retarget_motion/retarget_config_go2.py`
- `retarget_motion/retarget_config_sizu.py`
- `retarget_motion/retarget_kp_motions.py`

### Modified Files

- `retarget_motion/retarget_core.py`
- `retarget_motion/convert_19dof_to_61dof.py`
- `README.md`

## Inputs

Keypoint source data should be read from `retarget_motion/data/` wherever
possible.

If the migrated `go2` or `sizu` motion lists reference keypoint files that are
missing from this repository, only the minimum required files should be added.
There is no requirement to vendor the whole `IsaacgymLoco/datasets` tree.

## Outputs

Recommended output directories:

- 19 DoF:
  - `motion_imitation/data/motions_go2`
  - `motion_imitation/data/motions_sizu`
- 61 DoF:
  - `motion_imitation/data/motions_go2_61dof`
  - `motion_imitation/data/motions_sizu_61dof`

The 61 DoF files must be compatible with
`retarget_motion/view_motion_pybullet.py`.

## CLI Behavior

The new keypoint-retarget script should support:

- robot selection: `--robot go2|sizu`
- optional motion filtering by name
- explicit output directory override
- optional conversion to 61 DoF during the same command

Representative usage:

```bash
python retarget_motion/retarget_kp_motions.py --robot go2
python retarget_motion/retarget_kp_motions.py --robot sizu --motion pace0 trot0
python retarget_motion/retarget_kp_motions.py --robot go2 --to_61dof
```

Default behavior should generate 19 DoF motion files first. `--to_61dof`
should extend the pipeline by invoking the repo-local converter.

## Compatibility

The generated motion payloads must remain JSON-compatible with the current
motion loaders. The 19 DoF files should match the existing motion format:

- root position: 3
- root rotation quaternion: 4
- joint angles: 12

The 61 DoF files should match the existing IsaacgymLoco-style frame layout:

- pose: 19
- toe-local positions: 12
- root linear velocity: 3
- root angular velocity: 3
- joint velocity: 12
- toe-local velocity: 12

## Verification

Verification should happen in three layers.

### 1. Static Integration Checks

- `retarget_core` can load the new `go2` and `sizu` config modules
- the new CLI can enumerate and process configured motions
- output directories are created correctly

### 2. Smoke Generation Checks

- generate at least one `go2` motion
- generate at least one `sizu` motion
- confirm 19 DoF outputs have frame width 19
- convert one clip for each robot to 61 DoF
- confirm 61 DoF outputs have frame width 61

### 3. Viewer Compatibility

- the existing PyBullet viewer can load the generated motion files
- manual visual inspection focuses on:
  - base orientation stability
  - correct leg order
  - absence of obvious lower-leg inversion jumps
  - successful load for both 19 DoF and 61 DoF payloads

## Non-Goals

- migrating all upstream motion outputs
- migrating the `sizu_train` special training pipeline
- rebuilding controller or training code removed from this repository
- introducing new external asset dependencies

## Success Criteria

The migration is successful when:

1. `go2` and `sizu` can be retargeted from keypoint data inside this repo.
2. The pipeline uses only repo-local robot assets.
3. The generated outputs include 19 DoF and 61 DoF variants.
4. The existing viewer can visualize the results.
5. The migrated `sizu` setup preserves the correct leg ordering.
