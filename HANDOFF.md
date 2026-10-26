# HANDOFF

Written 2026-10-26 against commit `9ad2aa6` (plus the doc changes in the commit that added this file).
Project rules and commands live in `AGENTS.md`; if the two disagree, `AGENTS.md` wins.

## 1. PROJECT
- Dual-arm simulation of the open-source SO-101 robot arm (6 DOF incl. gripper) in MuJoCo.
  Repo `HGrayston/dual-arm-shenanigans`, Python package `so101_dual`, project name `so101-dual-arm`.
- Goal: a portfolio project for the owner (master's in cybernetics and robotics, working as an IT consultant, pivoting
  to hardware-near robotics): handover of an object between two arms, later bimanual tasks, later sim-to-real when the
  physical leader/follower arms arrive (owner is waiting for the parts; arrival date UNVERIFIED).
- Language: Python >= 3.10. Dependencies: `mujoco>=3.2` (locked 3.17.0), `numpy`; extras `dev` (pytest, imageio) and
  `ik` (mink, locked 1.3.0, not used yet). Package manager: `uv` (`uv.lock` committed).
- Owner preference: simple, laser-focused, not production-grade. He wants to write the behaviours himself.

## 2. CURRENT STATUS
Done and working (verified by running on a fresh clone):
- `uv lock --check` ok; `uv run --extra dev pytest` gives 8 passed.
- `load()` compiles the scene: 18 bodies, 12 actuators, keyframe `ready`, 0 contacts at start, cameras `overview`,
  `top`, `left_wrist_cam`, `right_wrist_cam`. `load(with_cube=False)` gives `nq == 12`.
- Position actuators track targets (test: left `shoulder_pan` reaches 0.5 rad within 0.03).
- `demo_ctrl` in `scripts/view_scene.py` over 20 s: 0 arm-to-arm contacts, state stays finite.
- The interactive viewer opens (owner ran it). A headless `--render` is UNVERIFIED: no GL available where I tested.

Half-finished:
- `scripts/first_motion.py` (committed version) only runs `mj_step` + `v.sync()` in a viewer loop. No `data.ctrl`
  writes yet. Line 8 `# q = READY_Q.copy()` is commented out. The owner had just learned that `q` is a 6-vector of
  joint targets in radians (order `JOINTS`) and was about to write it into `data.ctrl[left.actuator]`.
  Uncommitted local edits on the owner's machine are UNVERIFIED. He is writing this himself; do not finish it unasked.

Known issues:
- `src/so101_dual/scene.py:98-110`: `load(ready=False)` leaves `data.ctrl` at zero. Verified: 148 contacts at start,
  94 after 3 s as the arms drive toward the all-zeros pose and intersect.
- `src/so101_dual/scene.py:39`: `str.format` over `assets/scene.xml`; any literal `{`/`}` there raises an error.
- `src/so101_dual/scene.py:47-54` and `assets/scene.xml:40,46`: cube removal relies on the exact `OPTIONAL_CUBE_*`
  marker comments; edit them and `load(with_cube=False)` raises `ValueError` (the test at `tests/test_scene.py` catches it).
- `src/so101_dual/scene.py:19`: `REPO_ROOT = parents[2]` only works from a clone, not an installed wheel.
- `src/so101_dual/scene.py:86`: `READY_Q` wrist_flex is 1.6 of a 1.658 rad limit; may bite once IK is added.
- `scripts/first_motion.py:3`: unused imports `arm_ids`, `READY_Q` (expected, until the script is finished).

## 3. NEXT STEPS (in order)
1. Owner finishes `scripts/first_motion.py`: `left = arm_ids(model, "left")`, then inside the loop write a target
   vector to `data.ctrl[left.actuator]` (e.g. `READY_Q` with index 0 set to 0.5) and print `data.qpos[left.qpos]`
   about once per second. Done when the left arm visibly pans to about 0.5 rad. Offer help, do not do it for him.
2. After his OK, fix the `ready=False` footgun: in `load()`, set `data.ctrl` from the keyframe regardless of `ready`
   (or document it). Add one test. Small, `scene.py` only.
3. Ask before building anything bigger. Roadmap step 1 (README) is single-arm pick-and-place:
   a. `move_to(q_goal, ids, seconds)` helper that ramps `data.ctrl` linearly (about 10 lines, new `control.py` or in script);
   b. find the gripper value that holds the 3 cm cube (cube starts at `(0, 0, 0.415)`);
   c. differential IK to `<arm>_gripperframe` with mink (`uv sync --extra dev --extra ik`);
   d. state machine: above cube, descend, close, lift, place. Report success rate over N randomised cube poses.
4. Roadmap step 2: handover (A picks, moves to the middle, B closes, A releases), measured the same way.
Later (README roadmap): learned policy baseline, cooperative carry, real hardware, throw/catch as stretch only.

## 4. DECISIONS AND RATIONALE
- MuJoCo, not Gazebo/Isaac/Blender: best contact physics for grasping, plain Python, same ecosystem as LeRobot.
  Isaac needs an NVIDIA GPU. Blender is for rendering only. Gazebo suits ROS but is weaker for manipulation.
- Robot model: MuJoCo Menagerie `robotstudio_so101` (commit `f3b6013629c7b94f8ee93fd977b161fa8350f9bc`), derived from
  TheRobotStudio/SO-ARM100, vendored unmodified in `assets/so101/`. Arms are placed with `MjSpec.attach` and prefixes,
  so the upstream XML is never duplicated or edited.
- Layout: bases 0.50 m apart, table top at z = 0.40 m, so both arms (reach about 0.39 m) can meet in the middle.
- `READY_Q = [0, -1.3, 0.7, 1.6, 0, 0.9]`: found by grid search; collision-free for both arms with 15 cm between
  gripper tips. The all-zeros pose interpenetrates (4 cm deep).
- Scene XML lives in `assets/scene.xml` (owner's change, commit `9ad2aa6`), not in a Python string.
- Classical baseline (IK + state machine) first, learned policy (LeRobot ACT/diffusion) later for comparison.
  LeRobot is not a dependency yet.
- Python for simulation now. Owner knows C/C++ and considers it more valuable; a ROS 2 C++ port of the controller is a
  possible later step, not a plan. NTNU's Vortex repo (`vortexntnu/vortex-auv`) uses ROS 2 Humble (checked on GitHub).
Rejected, do not propose again:
- Docker, CI, pre-commit, ruff/clang-format, coverage: owner wants simple and said production tooling kills ownership.
- Throw-and-catch as the main goal: hardware (STS3215 servos, 5 arm DOF) is too imprecise; stretch goal only.
- Building IK/control helpers ahead of the owner: he asked to slow down and learn each part first.

## 5. MCP SERVERS AND TOOLS
- No MCP server is configured or required. Verified: no `.mcp.json` in the repo; the user-level client config in the
  environment I worked in (`~/.claude.json`) had no `mcpServers` entries. The owner's own machine config is UNVERIFIED.
- Used outside MCP: web search/fetch to find model sources and read the Vortex repo; `git`, `uv`, `pytest`.
  The host also offered a "attach GitHub repo" tool (client-specific, not an MCP server definition) to get push access.
- Optional, not used: a Hugging Face Hub MCP server (HTTP) could search models/datasets for LeRobot later. Its URL
  and auth variable name are UNVERIFIED; check Hugging Face docs before adding it.

## 6. OTHER DOCUMENTATION IN THE REPO
| Path | What | State | Reading |
|---|---|---|---|
| `AGENTS.md` | rules, commands, gotchas | new, current | REQUIRED |
| `README.md` | overview, quick start, scene, roadmap, limits | fixed in this commit (clone URL, `uv` flow, 8 tests, camera names, `arm_separation`) | REQUIRED |
| `assets/so101/UPSTREAM_README.md` | upstream Menagerie README | upstream text, accurate for the model | OPTIONAL |
| `assets/so101/CHANGELOG.md` | upstream model changelog | upstream | OPTIONAL |
| `assets/so101/LICENSE`, `LICENSE` | Apache-2.0 (model), MIT (own code) | current | keep |
| `pyproject.toml`, `uv.lock` | dependencies | consistent (`uv lock --check`) | OPTIONAL |
| `scripts/view_scene.py` | docstring documents its flags | current | OPTIONAL |
- Not present and not needed yet: CONTRIBUTING, CHANGELOG, `docs/`, `.env.example`, CI config.
- Will be needed once arms arrive: hardware notes (USB ports, motor IDs, calibration, gripper 0-100 vs radians mapping).
- No `CLAUDE.md` exists, so nothing needed reducing to a pointer.

## 7. INSTRUCTION FILES
- `AGENTS.md` is the single source of truth. No other instruction file exists.

## 8. WORKFLOW
- Setup/test/run: see `AGENTS.md` (`uv sync --extra dev`, `uv run pytest`, `uv run python scripts/view_scene.py`).
- Branch: only `main`; one earlier commit. Commits are short imperative sentences. Push access to GitHub was granted
  for this handoff; whether it persists for the next client is UNVERIFIED.
- Approval needed for: `assets/so101/`, `LICENSE`, public names in `scene.py`, new tooling, and starting roadmap step 1.
  This list is my reading of the owner's wishes, not a rule he wrote down (see section 10).
- Safety: simulation only, no hardware code. Hardware rules are proposed in `AGENTS.md`, not yet agreed.

## 9. ENVIRONMENT AND SECRETS
- No environment variables or secret files are required. Optional: `MUJOCO_GL` (`egl` or `osmesa`) for headless rendering.
- `.env` and `.env.*` are now in `.gitignore`. Scanned all tracked files and git history for tokens, keys and
  passwords: nothing found. Git credentials come from the client, not the repo.
- A display is needed for the viewer. Meshes (about 18 MB) are committed in `assets/so101/assets/`.

## 10. OPEN QUESTIONS (for the owner)
1. Which OS does he use? The old README had a Windows-style path (`.venv\scripts\activate.sh`); macOS needs `mjpython`.
2. Add a two-line `CLAUDE.md` pointing to `AGENTS.md`, for tools that only read that name? Not created: none existed.
3. Is the "approval needed" list in section 8 right, and may the `ready=False` fix go in?
4. When should roadmap step 1 (IK + pick-and-place) start, and should `move_to` come before or after he writes his own?
5. Rename the package or repo to match (`so101-dual-arm` vs `dual-arm-shenanigans`)? Cosmetic.
6. When do the physical arms arrive, and which LeRobot version will be used?
