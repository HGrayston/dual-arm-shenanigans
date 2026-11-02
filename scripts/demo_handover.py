"""Demo: the left arm picks up a ball and hands it to the right arm, guided by stereo vision.

    uv run --extra dev python scripts/demo_handover.py                  # watch it in the viewer
    uv run --extra dev python scripts/demo_handover.py --trials 20      # no window, measure success rate
    uv run --extra dev python scripts/demo_handover.py --video out/handover.gif

What drives what:
  * Vision -> motion: the arms never read the ball's true position. They get it from the
    stereo cameras, both on the table and when it's held in mid-air.
  * Vision -> decisions: every stage is checked by looking. Did the ball rise with the gripper?
    Did it stay put when the left arm let go? If not, the demo stops and says why.
  * Touch -> motion: the jaws close until the servo feels the ball, not to a preset angle.
"""

from __future__ import annotations

import argparse

import mujoco as mj
import numpy as np

from so101_dual import READY_Q, TABLE_HEIGHT
from so101_dual.control import FIXED_JAW_X, GRIPPER_OPEN, Sim, build_demo, grasp_rotation, open_video
from so101_dual.vision import Display, StereoRig, add_stereo_cameras

BALL_RADIUS = 0.025
MEET = np.array([0.0, 0.0, TABLE_HEIGHT + 0.28])  # where the handover takes place
PLACE = np.array([0.05, 0.0, TABLE_HEIGHT + BALL_RADIUS])  # where the right arm puts the ball down
# Centre the ball between the jaws: the fixed jaw doesn't move, so the ball must start touching-close to it.
OFFSET = FIXED_JAW_X + BALL_RADIUS + 0.002
DEPTH = 0.015  # m, how far behind the fingertips the ball's centre sits

DOWN_LEFT = grasp_rotation([0, 0, -1], [0, 1, 0])
DOWN_RIGHT = grasp_rotation([0, 0, -1], [0, -1, 0])
ACROSS_LEFT = grasp_rotation([1, 0, 0], [0, 1, 0])  # left jaws close sideways...
ACROSS_RIGHT = grasp_rotation([-1, 0, 0], [0, 0, 1])  # ...right jaws close up-and-down, so they don't clash


def add_ball(spec: mj.MjSpec) -> None:
    body = spec.worldbody.add_body(name="ball", pos=[-0.03, 0.0, TABLE_HEIGHT + BALL_RADIUS])
    body.add_freejoint(name="ball_free")
    body.add_geom(
        name="ball",
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=[BALL_RADIUS, 0, 0],
        mass=0.01,
        rgba=[0.9, 0.1, 0.1, 1],
        condim=6,
        friction=[2.0, 0.05, 0.01],  # rubbery; the rolling friction keeps it from wandering off
        priority=2,  # use this friction, not the gripper's, wherever it touches something
    )
    add_stereo_cameras(spec)


def grasp_target(ball: np.ndarray, approach: np.ndarray) -> np.ndarray:
    """Where to put the grasp point: a little past the ball's centre, so the fingers wrap around it."""
    return ball + DEPTH * approach


class Failure(RuntimeError):
    """A vision check failed. The message says which one."""


def run(sim: Sim, rig: StereoRig, ball_start: np.ndarray, log=print) -> dict:
    """Run one handover. Returns measurements, or raises ``Failure``."""
    model, data = sim.model, sim.data
    ball_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "ball")
    qadr = model.jnt_qposadr[model.body_jntadr[ball_id]]
    data.qpos[qadr : qadr + 3] = ball_start
    mj.mj_forward(model, data)
    sim.wait(0.5)

    def see(what: str) -> np.ndarray:
        det = rig.locate(data, "red", ball=True)
        for _ in range(5):  # a moving arm can block the view for a moment, so give it a few tries
            if det is not None:
                break
            sim.wait(0.2)
            det = rig.locate(data, "red", ball=True)
        if det is None:
            raise Failure(f"{what}: the cameras can't see the ball")
        truth = data.xpos[ball_id]
        log(
            f"  [vision] {what}: ball at {np.round(det.position, 3)} "
            f"(true error {1000 * np.linalg.norm(det.position - truth):.1f} mm)"
        )
        return det.position

    down, across = np.array([0, 0, -1.0]), np.array([1.0, 0, 0])
    results = {}

    log("1. Find the ball on the table")
    ball = see("on table")
    results["table_error_mm"] = 1000 * float(np.linalg.norm(ball - data.xpos[ball_id]))

    log("2. Left arm: reach over, open, descend, close until it feels the ball")
    grasp = grasp_target(ball, down)
    sim.grip({"left": GRIPPER_OPEN, "right": GRIPPER_OPEN}, 0.6)
    sim.move({"left": (grasp + [0, 0, 0.09], DOWN_LEFT, OFFSET)}, 2.0)
    sim.move({"left": (grasp, DOWN_LEFT)}, 1.2)
    angle = sim.close_until("left")
    log(f"  [touch] jaws stopped at {angle:.2f} rad, servo feels the ball")

    log("3. Lift, and check by looking that the ball came along")
    sim.move({"left": (grasp + [0, 0, 0.09], DOWN_LEFT)}, 1.2)
    lifted = see("after lift")
    rise = lifted[2] - ball[2]
    if rise < 0.05:
        raise Failure(f"ball only rose {100 * rise:.1f} cm, the grasp failed")
    log(f"  [check] ball rose {100 * rise:.1f} cm with the gripper: grasp confirmed")

    log("4. Carry it to the meeting point")
    sim.move({"left": (MEET - DEPTH * across, ACROSS_LEFT)}, 2.5)
    sim.wait(0.3)
    held = see("held at meeting point")
    results["held_error_mm"] = 1000 * float(np.linalg.norm(held - data.xpos[ball_id]))

    log("5. Right arm: aim for where the cameras see the ball, not where the left arm thinks it is")
    toward = np.array([-1.0, 0, 0])  # the right arm approaches along -x
    sim.move({"right": (held - 0.09 * toward, ACROSS_RIGHT, OFFSET)}, 2.5)
    # Look again before committing: a held ball can sag, and the cameras will notice.
    held = see("re-check before closing in")
    sim.move({"right": (held - 0.03 * toward, ACROSS_RIGHT)}, 1.0)
    held = see("re-check at arm's length")
    sim.move({"right": (grasp_target(held, toward), ACROSS_RIGHT)}, 1.0)
    sim.close_until("right")

    log("6. Left arm lets go and backs away")
    sim.grip({"left": GRIPPER_OPEN}, 0.8)
    sim.wait(0.3)
    after = see("after release")
    if np.linalg.norm(after - held) > 0.03:
        raise Failure("the ball moved when the left arm let go, so the right arm didn't have it")
    log("  [check] ball stayed put: the right arm has it")
    left_point = sim.arms["left"].point
    sim.move({"left": (left_point - [0.09, 0, 0], ACROSS_LEFT)}, 1.2)
    sim.joints({"left": READY_Q}, 1.5)

    log("7. Right arm carries the ball over the target and uses the cameras to line it up")
    hover = PLACE + np.array([0, 0, 0.05]) + DEPTH * down
    sim.move({"right": (hover, DOWN_RIGHT)}, 2.5)
    sim.wait(0.3)
    for _ in range(2):  # visual servoing: measure the error in the image, correct the arm, repeat
        ball_now = see("over target")
        error = PLACE[:2] - ball_now[:2]
        log(f"  [servo] ball is {1000 * np.linalg.norm(error):.1f} mm off target, correcting")
        point = sim.arms["right"].point + np.array([error[0], error[1], 0.0])
        sim.move({"right": (point, DOWN_RIGHT)}, 0.8)
        sim.wait(0.2)
    # Lower by the height the cameras say is left, stopping just above the table.
    drop = see("before lowering")[2] - PLACE[2] - 0.003
    sim.move({"right": (sim.arms["right"].point - [0, 0, max(drop, 0.0)], DOWN_RIGHT)}, 1.2)
    sim.grip({"right": GRIPPER_OPEN}, 0.8)
    right_point = sim.arms["right"].point
    sim.move({"right": (right_point + [0, 0, 0.09], DOWN_RIGHT)}, 1.2)
    sim.joints({"right": READY_Q}, 1.5)
    sim.wait(0.5)
    final = see("placed")
    miss = float(np.linalg.norm((final - PLACE)[:2]))
    results["placement_error_mm"] = 1000 * miss
    if miss > 0.03 or final[2] > TABLE_HEIGHT + 0.04:
        raise Failure(f"the ball ended up {100 * miss:.1f} cm from the target")
    log(f"  [check] ball is {1000 * miss:.1f} mm from the target. Done!")
    return results


def random_start(rng: np.random.Generator) -> np.ndarray:
    return np.array([rng.uniform(-0.07, -0.01), rng.uniform(-0.05, 0.05), TABLE_HEIGHT + BALL_RADIUS])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--trials", type=int, help="run N randomised handovers without a window and report the success rate")
    ap.add_argument("--video", help="write a video instead of opening a window (.gif, or .mp4 with imageio-ffmpeg)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    if args.trials:
        wins, stats = 0, []
        model, data = build_demo(add_ball)
        rig = StereoRig(model)
        for i in range(args.trials):
            mj.mj_resetDataKeyframe(model, data, model.key("ready").id)
            try:
                stats.append(run(Sim(model, data), rig, random_start(rng), log=lambda *_: None))
                wins += 1
                print(f"trial {i + 1}: ok, placed within {stats[-1]['placement_error_mm']:.1f} mm")
            except Failure as err:
                print(f"trial {i + 1}: FAILED, {err}")
        print(f"\nsuccess: {wins}/{args.trials}")
        if stats:
            for key in stats[0]:
                print(f"{key}: mean {np.mean([s[key] for s in stats]):.1f}, max {np.max([s[key] for s in stats]):.1f}")
        return

    model, data = build_demo(add_ball)
    rig = StereoRig(model)
    if args.video:
        display = Display(model, rig)
        display.colours = ["red"]
        display.writer = open_video(args.video)
        sim = Sim(model, data)
        sim.frame_hook = lambda: display.record(data)
        try:
            run(sim, rig, random_start(rng))
        except Failure as err:
            print(f"FAILED: {err}")
        display.writer.close()
        print(f"wrote {args.video}")
        return

    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        try:
            sim = Sim(model, data, realtime=True, viewer=viewer)
            display = Display(model, rig)
            display.colours = ["red"]
            sim.video_fps = 5  # refresh the camera panels 5 times per simulated second
            sim.frame_hook = lambda: display.to_viewer(viewer, data)
            run(sim, rig, random_start(rng))
        except Failure as err:
            print(f"FAILED: {err}")
        print("Close the window to exit.")
        while viewer.is_running():
            viewer.sync()


if __name__ == "__main__":
    main()
