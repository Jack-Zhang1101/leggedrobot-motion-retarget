# Motion Retarget And Visualization

This repository is now a focused toolset for quadruped motion retargeting and
PyBullet visualization.

The training stack, MPC controller code, and third-party solver/vendor trees
from the original upstream project have been removed. What remains is the
minimal code needed to:

- load source motion clips
- retarget Laikago-style dog motions to A1
- convert 19dof clips to the 61dof format used by downstream datasets
- visualize motions in PyBullet with local robot assets only

## Repository Layout

- `retarget_motion/`
  - retarget configs
  - A1 motion library generation
  - 19dof -> 61dof conversion
  - PyBullet motion viewer
- `motion_imitation/`
  - minimal robot and utility modules still required by retargeting
- `assets/robots/`
  - self-contained robot assets for `a1`, `go2`, `sizu`, `laikago`
- `motion_imitation/data/motions/`
  - source motion clips
- `motion_imitation/data/motions_a1/`
  - retargeted A1 19dof clips
- `motion_imitation/data/motions_a1_61dof/`
  - converted A1 61dof clips

## Environment

The scripts have been verified in the user's conda environment:

```bash
source /home/shibo/anaconda3/etc/profile.d/conda.sh
conda activate unitree-rl
```

Minimal Python dependencies for the remaining code are listed in
`requirements.txt`.

## Generate A1 Motions

Regenerate the A1 19dof motion library from the source clips:

```bash
python retarget_motion/generate_a1_motion_library.py \
  --output_dir motion_imitation/data/motions_a1
```

This writes:

- one `.txt` file per retargeted motion
- `provenance.json` describing how each clip was produced

## Convert 19dof To 61dof

Convert the generated A1 motions into the 61dof format:

```bash
python retarget_motion/convert_19dof_to_61dof.py \
  --input_dir motion_imitation/data/motions_a1 \
  --output_dir motion_imitation/data/motions_a1_61dof \
  --robot a1 \
  --motion dog_pace \
  --motion dog_trot \
  --motion dog_backwards_pace \
  --motion dog_backwards_trot \
  --motion dog_spin \
  --motion hopturn \
  --motion inplace_steps \
  --motion runningman \
  --motion sidesteps \
  --motion turn \
  --motion_weight 1.0
```

## Visualize Motions In PyBullet

Visualize either a single motion file or a whole directory:

```bash
python retarget_motion/view_motion_pybullet.py \
  --motion_file motion_imitation/data/motions_a1 \
  --robot a1
```

Supported robot assets in the viewer:

- `a1`
- `go2`
- `sizu`
- `laikago`

The viewer loads assets only from this repository. It no longer depends on any
external `legged_gym` or IsaacGymLoco robot path.

## Motion Formats

The supported frame layouts are:

- `19dof`: root position (3) + root rotation quaternion (4) + joint angles (12)
- `61dof`: 19dof pose + toe local targets + root/joint/toe velocities

For visualization:

- 19dof is enough to draw the robot pose itself
- 61dof is only needed if you also want toe target markers and velocity overlays

## Notes

- The vendored A1 URDF emits PyBullet warnings about missing inertial data on
  some fixed links. These are warnings, not retarget blockers.
- `motion_imitation/robots/a1.py` still contains a legacy import-time warm-up
  path that can print a `RuntimeWarning` from `arccos`; the generated motion
  outputs used here were still verified separately.

## License

This repository still contains code and assets derived from the original
project. See `LICENSE.txt` and any per-asset license files under `assets/`.
