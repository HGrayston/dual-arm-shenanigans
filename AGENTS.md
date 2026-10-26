# AGENTS.md

Single source of truth for working in this repo. Current status and next steps are in `HANDOFF.md`.

## What this is
Dual-arm SO-101 simulation in MuJoCo (Python). Two arms, a table and a cube. Goal: cooperative manipulation
(handover first), later sim-to-real on physical SO-101 arms. The owner is learning by doing and wants to write the
behaviour code himself.

## Commands (run from the repo root)
- Install: `uv sync --extra dev` (Python >= 3.10, `uv.lock` is committed; add the IK extra later with `--extra ik`)
- Test: `uv run pytest` (expected: 8 passed)
- View: `uv run python scripts/view_scene.py [--demo] [--separation 0.5] [--no-cube]` (needs a display)
- Still image: `uv run python scripts/view_scene.py --render out/overview.png` (needs EGL or OSMesa, may fail headless)
- Scratch motion script (owner's): `uv run python scripts/first_motion.py`

## Structure
- `src/so101_dual/scene.py`: `load()`, `build_spec()`, `arm_ids()`, `READY_Q`, `ARMS`, `JOINTS`. All scene logic.
- `assets/scene.xml`: table, floor, cube, cameras. Read by `scene.py` and filled with `str.format`.
- `assets/so101/`: vendored robot model (MuJoCo Menagerie `robotstudio_so101`, Apache-2.0).
- `scripts/`: runnable scripts. `tests/test_scene.py`: pytest. `README.md`: overview and roadmap.

## Conventions
- Joint order everywhere is `JOINTS`: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper. Radians.
- `data.ctrl` holds position targets, `data.qpos` the actual angles. Arm elements are prefixed `left_` / `right_`.
- Style: type hints, `from __future__ import annotations`, short docstrings on public functions, plain numpy.
  No formatter or linter is configured; match the surrounding code and do not add one unasked.
- Commits: short imperative sentence, direct to `main` (only branch). No other convention exists yet.
- Reply in the language the owner writes (Norwegian or English).

## Gotchas
- `assets/scene.xml` is passed through `str.format`: a literal `{` or `}` (even in a comment) breaks loading.
  The cube is removed by matching the two `OPTIONAL_CUBE_START/END` comment lines; keep them exactly.
- All-zeros joint pose makes the arms intersect. Start from `READY_Q` (the `ready` keyframe).
- `load(ready=False)` leaves `data.ctrl` at zero, so the arms drive into each other. Set `ctrl` yourself.
- The package finds `assets/` relative to the repo, so it works from a clone or `uv sync`, not from a built wheel.
- On macOS the interactive viewer may need `mjpython`.

## Do not change without the owner's approval
- Anything under `assets/so101/` (vendored upstream, must stay unmodified with its `LICENSE`).
- `LICENSE`, `uv.lock` by hand (use `uv lock`), and the public names `load`, `arm_ids`, `READY_Q`, `ARMS`, `JOINTS`.
- Scope: the owner wants it simple and laser-focused. Do not add Docker, CI, pre-commit, linters, coverage or
  other production tooling, and do not implement the next roadmap step unasked.

## Safety
- Simulation only; no hardware code exists. For future hardware work (not yet agreed, UNVERIFIED): stay within the
  `ctrlrange` limits in `assets/so101/so101.xml`, start with torque off or slow moves, and never run motion on real
  arms without the owner present.
- Never commit secrets. None are needed today; `.env` and `.env.*` are git-ignored.
