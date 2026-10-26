"""Open the dual-arm SO-101 scene.

    python scripts/view_scene.py              # interactive viewer, arms hold ready pose
    python scripts/view_scene.py --demo       # arms wiggle, proves control works
    python scripts/view_scene.py --render out/overview.png   # headless still (needs EGL/OSMesa)

In the viewer, Backspace resets to the ``ready`` keyframe.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco as mj
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from so101_dual import ARMS, READY_Q, arm_ids, load  # noqa: E402


def demo_ctrl(t: float, sign: float) -> np.ndarray:
    """Small, collision-safe motion around the ready pose."""
    q = READY_Q.copy()
    q[0] += sign * 0.35 * np.sin(1.2 * t)  # shoulder_pan, mirrored between arms
    q[1] += 0.15 * np.sin(0.8 * t)  # shoulder_lift
    q[4] += 0.8 * np.sin(1.5 * t)  # wrist_roll
    q[5] = 0.9 + 0.7 * np.sin(2.0 * t)  # gripper open/close
    return q


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--separation", type=float, default=0.50, help="distance between arm bases (m)")
    ap.add_argument("--no-cube", action="store_true")
    ap.add_argument("--demo", action="store_true", help="drive the joints with a sine pattern")
    ap.add_argument("--render", metavar="PNG", help="save a still image and exit")
    ap.add_argument("--camera", default="overview", help="camera for --render (overview|top)")
    args = ap.parse_args()

    model, data = load(args.separation, with_cube=not args.no_cube)
    ids = {a: arm_ids(model, a) for a in ARMS}

    if args.render:
        import imageio.v3 as iio

        out = Path(args.render)
        out.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(200):  # let everything settle
            mj.mj_step(model, data)
        with mj.Renderer(model, 720, 1280) as r:
            r.update_scene(data, camera=args.camera)
            iio.imwrite(out, r.render())
        print(f"wrote {out}")
        return

    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        while viewer.is_running():
            step_start = time.time()
            if args.demo:
                t = data.time
                for arm, sign in zip(ARMS, (+1.0, -1.0)):
                    q = demo_ctrl(t, sign)
                    ctrl = np.clip(q, ids[arm].ctrlrange[:, 0], ids[arm].ctrlrange[:, 1])
                    data.ctrl[ids[arm].actuator] = ctrl
            mj.mj_step(model, data)
            viewer.sync()
            sleep = model.opt.timestep - (time.time() - step_start)
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    main()
