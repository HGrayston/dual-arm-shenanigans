# so101-dual-arm

Dual-arm [SO-101](https://github.com/TheRobotStudio/SO-ARM100) simulation in MuJoCo, built as a stepping stone to
cooperative manipulation (object handover, then bimanual tasks) and, later, sim-to-real on the physical
leader/follower arms.

> Status: **step 0, scene and tooling**. Two arms, a table and a cube load, hold pose and pass tests.
> Showcase demos of handover, cooperative lifting and stereo vision are in place (see below). Current state and next
> steps: `HANDOFF.md`.
> Rules for anyone (human or AI) editing the repo: `AGENTS.md`.

## Quick start

```bash
git clone https://github.com/HGrayston/dual-arm-shenanigans
cd dual-arm-shenanigans
uv sync --extra dev                     # creates .venv, installs mujoco, numpy, pytest, imageio

uv run python scripts/view_scene.py            # interactive viewer
uv run python scripts/view_scene.py --demo     # slow jig
uv run python scripts/first_motion.py          # scratch script: viewer loop, no arm motion yet
uv run pytest                                  # 8 tests
```

`uv run` uses the project's `.venv` without activating it, so the same commands work on Windows, Linux and macOS.
(macOS: the interactive viewer may need `mjpython` instead of `python`, UNVERIFIED with uv.)

## What is in the scene (AI)

| Item | Detail |
|---|---|
| Arms | Two SO-101 (6 DOF incl. gripper), bases 0.50 m apart, facing each other. Prefixes `left_` / `right_` |
| Model | MuJoCo Menagerie `robotstudio_so101`, position actuators with STS3215-style gains and force limits |
| Table | Top at z = 0.40 m |
| Object | 3 cm cube, 20 g, at the centre |
| Start pose | Folded `READY_Q`, collision-free, 15 cm between gripper tips |
| Cameras | `overview`, `top`, `left_wrist_cam`, `right_wrist_cam` |
| Scene file | `assets/scene.xml` (table, floor, cube, cameras); `src/so101_dual/scene.py` fills its `{placeholders}` and attaches the arms |

The all-zeros pose is **not** a safe start: both arms then reach into each other. That is why a `ready` keyframe
exists (Backspace in the viewer resets to it).

Arm placement uses MuJoCo's `MjSpec.attach`, so the upstream XML is untouched and `arm_separation` is a parameter:

```python
from so101_dual import load, arm_ids, ARMS
model, data = load(arm_separation=0.50)
left = arm_ids(model, "left")   # actuator / qpos / dof indices, ctrlrange, gripper site
data.ctrl[left.actuator] = [...]  # 6 joint targets in rad: pan, lift, elbow, wrist_flex, wrist_roll, gripper
```

## Showcase demos

Three runnable demos of how the arms and cameras interact in code. Each opens the viewer, with the two stereo camera
views in the bottom corners, and prints what it's doing and why.

```bash
uv run --extra dev python scripts/demo_handover.py   # left arm picks up a ball and hands it to the right arm
uv run --extra dev python scripts/demo_beam.py       # both arms lift a beam, kept level by the cameras
uv run --extra dev python scripts/demo_beam.py --no-feedback   # same fault, cameras not used to fix it
uv run --extra dev python scripts/demo_stereo.py     # live 3D tracking of a bouncing ball
uv run --extra dev python scripts/demo_stereo.py --report out/stereo   # depth map, 3D point cloud, accuracy numbers
```

`--trials N` (handover, beam) runs N randomised attempts without a window and reports the results. `--video FILE`
(all three) records to a file instead. `.gif` works as is; `.mp4` also needs `--with imageio-ffmpeg` after `uv run`.

The building blocks are `src/so101_dual/control.py` (IK, smooth Cartesian moves, force-feedback closing) and
`src/so101_dual/vision.py` (colour detection, circle fitting, triangulation). The arms get object positions from the
rendered camera images, not from the simulator. Limits: colours are easy to pick out here, and simulated cameras have no
noise, distortion or calibration error, so real-world accuracy will be worse.

## Roadmap

Each step ends in something demonstrable, with a GIF in this README and a success-rate number.

1. **Single-arm pick and place.** Differential IK ([mink](https://github.com/kevinzakka/mink)) to the `gripperframe`
   site, state machine, cube moved between two table positions.
2. **Handover.** Arm A picks, moves to a meeting point, arm B closes, A releases. Measure success rate over N
   randomised cube poses, plus cycle time and failure modes.
3. **Learned policy baseline.** Collect demonstrations from the scripted controller, train ACT or diffusion policy
   (LeRobot), compare against the classical baseline.
4. **Cooperative carry** of a long object with both grippers (closed kinematic chain).
5. **Real hardware.** Same pipeline on the physical arms, sim and real side by side, with measured sim-to-real gap.
6. Stretch: throw and catch.

## Known limits of the model

- Servo behaviour is a simplified position-actuator model. Backlash is defined upstream but not applied, and there
  is no communication latency or thermal/torque-limit behaviour beyond the force range.
- Gripper convention differs from LeRobot (there it is 0 to 100, here a joint angle in rad). Needs a mapping when
  moving to hardware.
- `READY_Q` has the wrist near its joint limit (1.6 of 1.658 rad). Fine for now, revisit if IK starts hitting limits.

## Credits and licences

- Robot model: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) `robotstudio_so101`
  (Apache-2.0), derived from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (Apache-2.0).
  Copied unmodified into `assets/so101/` together with its licence and upstream README.
  Menagerie commit `f3b6013629c7b94f8ee93fd977b161fa8350f9bc`.
- Code in `src/`, `scripts/`, `tests/`: MIT, see `LICENSE`.
