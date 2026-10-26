# so101-dual-arm

Dual-arm [SO-101](https://github.com/TheRobotStudio/SO-ARM100) simulation in MuJoCo, built as a stepping stone to
cooperative manipulation (object handover, then bimanual tasks) and, later, sim-to-real on the physical
leader/follower arms.

> Status: **step 0, scene and tooling**. Two arms, a table and a cube load, hold pose and pass tests.
> No manipulation behaviour yet. See the roadmap below.

## Quick start

```bash
git clone dual-arm-shenanigans
uv sync
.venv\scripts\activate.sh
code .

python scripts/view_scene.py            # interactive viewer
python scripts/view_scene.py --demo     # slow jig
pytest                                  # 7 tests
```

## What is in the scene (AI)

| Item | Detail |
|---|---|
| Arms | Two SO-101 (6 DOF incl. gripper), bases 0.50 m apart, facing each other. Prefixes `left_` / `right_` |
| Model | MuJoCo Menagerie `robotstudio_so101`, position actuators with STS3215-style gains and force limits |
| Table | Top at z = 0.40 m |
| Object | 3 cm cube, 20 g, at the centre |
| Start pose | Folded `READY_Q`, collision-free, 15 cm between gripper tips |
| Cameras | `overview`, `top`, plus each arm's `wrist_cam` |

The all-zeros pose is **not** a safe start: both arms then reach into each other. That is why a `ready` keyframe
exists (Backspace in the viewer resets to it).

Arm placement uses MuJoCo's `MjSpec.attach`, so the upstream XML is untouched and `separation` is a parameter:

```python
from so101_dual import load, arm_ids, ARMS
model, data = load(arm_separation=0.50)
left = arm_ids(model, "left")   # actuator / qpos / dof indices, ctrlrange, gripper site
data.ctrl[left.actuator] = [...]  # 6 joint targets in rad: pan, lift, elbow, wrist_flex, wrist_roll, gripper
```

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
