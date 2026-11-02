"""Demo: both arms lift a beam together, one on each end, kept level by stereo vision.

    uv run --extra dev python scripts/demo_beam.py                  # watch it in the viewer
    uv run --extra dev python scripts/demo_beam.py --no-feedback    # same fault, cameras not used to fix it
    uv run --extra dev python scripts/demo_beam.py --trials 20      # no window, measure success rate
    uv run --extra dev python scripts/demo_beam.py --video out/beam.gif

The right arm has a fake fault: it only rises 80% as far as it's told, like a worn or badly
calibrated joint. The arm can't sense this, because its own joint readings look fine. The cameras
can: they track two coloured markers on the beam, work out how tilted it is, and correct
both arms while they move. That's closed-loop control with vision as the sensor.
"""

from __future__ import annotations

import argparse
import math

import mujoco as mj
import numpy as np

from so101_dual import READY_Q, TABLE_HEIGHT
from so101_dual.control import FIXED_JAW_X, GRIPPER_OPEN, Sim, build_demo, grasp_rotation, open_video
from so101_dual.vision import Display, StereoRig, add_stereo_cameras

SEPARATION = 0.70  # m between the arm bases, wider than the default to leave room for a long beam
HALF_LENGTH, HALF_WIDTH, HALF_HEIGHT = 0.21, 0.0125, 0.02
GRASP_FROM_CENTRE = 0.15  # m, each arm grips this far from the middle
MARKER_FROM_CENTRE = 0.11  # m, inside the grippers, where the arms can't hide them from the cameras
MARKER_RADIUS = 0.012
MARKER_ABOVE_CENTRE = HALF_HEIGHT + MARKER_RADIUS - 0.002
LEAN = math.radians(35)  # grippers lean inward: the arms can't lift this high pointing straight down
LIFT = 0.13  # m
FAULT = 0.20  # the right arm falls short by this fraction of the height
OFFSET = FIXED_JAW_X + HALF_WIDTH + 0.002
DEPTH = 0.012  # m past the beam's centre line, so the fingers wrap well around it


def add_beam(spec: mj.MjSpec) -> None:
    body = spec.worldbody.add_body(name="beam", pos=[0, 0, TABLE_HEIGHT + HALF_HEIGHT])
    body.add_freejoint(name="beam_free")
    body.add_geom(
        name="beam",
        type=mj.mjtGeom.mjGEOM_BOX,
        size=[HALF_LENGTH, HALF_WIDTH, HALF_HEIGHT],
        mass=0.06,
        rgba=[0.85, 0.85, 0.88, 1],
        condim=6,
        friction=[2.0, 0.05, 0.01],
        priority=2,
    )
    for name, x, rgba in (("left", -1, [0.9, 0.1, 0.8, 1]), ("right", 1, [0.1, 0.8, 0.2, 1])):
        body.add_geom(
            name=f"marker_{name}",
            type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[MARKER_RADIUS, 0, 0],
            pos=[x * MARKER_FROM_CENTRE, 0, MARKER_ABOVE_CENTRE],
            rgba=rgba,
            mass=0.001,
            contype=0,
            conaffinity=0,
        )
    add_stereo_cameras(spec)


class Failure(RuntimeError):
    """A vision check failed. The message says which one."""


class BeamView:
    """What the cameras can tell us about the beam."""

    def __init__(self, rig: StereoRig, data: mj.MjData):
        self.rig, self.data = rig, data

    def look(self):
        """Return ``(centre, yaw, tilt)`` in metres and radians, or raise ``Failure``."""
        images = self.rig.capture(self.data)
        a = self.rig.locate(self.data, "magenta", images, ball=True)
        b = self.rig.locate(self.data, "green", images, ball=True)
        if a is None or b is None:
            raise Failure("the cameras can't see both beam markers")
        d = b.position - a.position
        centre = (a.position + b.position) / 2 - [0, 0, MARKER_ABOVE_CENTRE]
        yaw = math.atan2(d[1], d[0])
        tilt = math.atan2(d[2], np.hypot(d[0], d[1]))
        return centre, yaw, tilt


def grasps(centre: np.ndarray, yaw: float) -> dict:
    """Grasp points and gripper orientations for a beam at this position and yaw."""
    along = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    pinch = np.array([-math.sin(yaw), math.cos(yaw), 0.0])  # across the beam
    out = {}
    for arm, sign in (("left", -1.0), ("right", 1.0)):
        approach = -sign * math.sin(LEAN) * along + np.array([0, 0, -math.cos(LEAN)])
        point = centre + sign * GRASP_FROM_CENTRE * along + DEPTH * approach
        out[arm] = (point, grasp_rotation(approach, pinch), approach)
    return out


def run(sim: Sim, rig: StereoRig, start: tuple[float, float, float], feedback: bool = True, log=print) -> dict:
    """Run one lift. Returns measurements, or raises ``Failure``."""
    model, data = sim.model, sim.data
    beam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "beam")
    qadr = model.jnt_qposadr[model.body_jntadr[beam_id]]
    data.qpos[qadr : qadr + 3] = [start[0], start[1], TABLE_HEIGHT + HALF_HEIGHT]
    half = start[2] / 2
    data.qpos[qadr + 3 : qadr + 7] = [math.cos(half), 0, 0, math.sin(half)]
    mj.mj_forward(model, data)
    sim.wait(0.5)
    view = BeamView(rig, data)

    def true_tilt() -> float:
        axis = data.xmat[beam_id].reshape(3, 3)[:, 0]
        return math.degrees(math.asin(axis[2]))

    log("1. Find the beam from its two markers")
    centre, yaw, tilt = view.look()
    true_centre = data.xpos[beam_id]
    log(
        f"  [vision] centre {np.round(centre, 3)}, turned {math.degrees(yaw):.1f} deg "
        f"(true errors {1000 * np.linalg.norm(centre - true_centre):.1f} mm, "
        f"{abs(math.degrees(yaw) - math.degrees(start[2])):.2f} deg)"
    )
    plan = grasps(centre, yaw)

    log("2. Both arms reach for their end of the beam at the same time")
    sim.grip({"left": GRIPPER_OPEN, "right": GRIPPER_OPEN}, 0.6)
    above = {a: plan[a][0] + [0, 0, 0.08] for a in plan}
    sim.move({a: (above[a], plan[a][1], OFFSET) for a in plan}, 2.5)
    sim.move({a: (plan[a][0], plan[a][1]) for a in plan}, 1.5)

    log("3. Close both grippers until each servo feels the beam")
    sim.close_until("left")
    sim.close_until("right")
    start_z = view.look()[0][2]

    log(f"4. Lift together. Fault: the right arm only rises {100 * (1 - FAULT):.0f}% as far as commanded")
    correction, last_look, worst = [0.0], [-1.0], [0.0]
    rise_of = {a: plan[a][0][2] for a in plan}

    def correct(arm: str, point: np.ndarray) -> np.ndarray:
        # Vision feedback at 10 Hz: if the right end is low, raise it and lower the left, half each.
        if feedback and data.time - last_look[0] > 0.1:
            last_look[0] = data.time
            correction[0] += 0.5 * (-view.look()[2]) * (2 * GRASP_FROM_CENTRE)
        worst[0] = max(worst[0], abs(true_tilt()))
        target = point.copy()
        if arm == "right":
            target[2] -= FAULT * (point[2] - rise_of[arm])  # the fake fault
        target[2] += (1 if arm == "right" else -1) * correction[0] / 2
        return target

    up = {a: (plan[a][0] + [0, 0, LIFT], plan[a][1]) for a in plan}
    sim.move(up, 4.0, correct)
    sim.move(up, 1.0, correct)  # hold still and let the feedback finish its job

    centre_up, _, tilt_up = view.look()
    risen = centre_up[2] - start_z
    results = {
        "lift_cm": 100 * float(risen),
        "tilt_deg_vision": math.degrees(tilt_up),
        "tilt_deg_true": true_tilt(),
        "worst_tilt_deg": worst[0],
    }
    log(
        f"  [vision] beam rose {100 * risen:.1f} cm, tilt {results['tilt_deg_vision']:.2f} deg "
        f"(true tilt {results['tilt_deg_true']:.2f} deg, worst during lift {worst[0]:.2f} deg)"
    )
    if risen < 0.5 * LIFT:
        raise Failure(f"the beam only rose {100 * risen:.1f} cm, so at least one grip slipped")

    log("5. Carry it sideways together, so one arm can't move without the other")
    side = np.array([0, 0.04, 0.0])
    over = {a: (up[a][0] + side, plan[a][1]) for a in plan}
    sim.move(over, 2.0, correct)
    centre_side, _, _ = view.look()
    moved = centre_side[1] - centre_up[1]
    log(f"  [vision] beam moved {100 * moved:.1f} cm sideways (commanded {100 * side[1]:.1f} cm)")
    results["side_move_cm"] = 100 * float(moved)

    log("6. Put it back down, let go and step away")
    back = {a: (plan[a][0] + side + [0, 0, 0.005], plan[a][1]) for a in plan}
    sim.move(back, 3.0, correct)
    sim.grip({"left": GRIPPER_OPEN, "right": GRIPPER_OPEN}, 0.8)
    clear = {a: (plan[a][0] + side + [0, 0, 0.09], plan[a][1]) for a in plan}
    sim.move(clear, 1.5)
    sim.joints({"left": READY_Q, "right": READY_Q}, 2.0)
    sim.wait(0.5)
    final, _, final_tilt = view.look()
    results["final_tilt_deg"] = math.degrees(final_tilt)
    if final[2] > TABLE_HEIGHT + HALF_HEIGHT + 0.015:
        raise Failure("the beam isn't resting on the table")
    log(f"  [check] beam is back on the table, {100 * (final[1] - centre[1]):.1f} cm to the side of where it started. Done!")
    return results


def random_start(rng: np.random.Generator) -> tuple[float, float, float]:
    return rng.uniform(-0.02, 0.02), rng.uniform(-0.03, 0.03), math.radians(rng.uniform(-8, 8))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--no-feedback", action="store_true", help="leave the fault uncorrected, to compare")
    ap.add_argument("--trials", type=int, help="run N randomised lifts without a window and report the results")
    ap.add_argument("--video", help="write a video instead of opening a window (.gif, or .mp4 with imageio-ffmpeg)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    feedback = not args.no_feedback

    def scene():
        model, data = build_demo(add_beam, SEPARATION)
        return model, data, StereoRig(model)

    if args.trials:
        wins, stats = 0, []
        model, data, rig = scene()
        for i in range(args.trials):
            mj.mj_resetDataKeyframe(model, data, model.key("ready").id)
            try:
                stats.append(run(Sim(model, data), rig, random_start(rng), feedback, log=lambda *_: None))
                wins += 1
                print(f"trial {i + 1}: ok, level to within {abs(stats[-1]['tilt_deg_true']):.2f} deg")
            except Failure as err:
                print(f"trial {i + 1}: FAILED, {err}")
        print(f"\nsuccess: {wins}/{args.trials}")
        if stats:
            for key in stats[0]:
                vals = np.abs([s[key] for s in stats])
                print(f"{key}: mean {vals.mean():.2f}, max {vals.max():.2f} (absolute)")
        return

    model, data, rig = scene()
    if args.video:
        display = Display(model, rig)
        display.colours = ["magenta", "green"]
        display.writer = open_video(args.video)
        sim = Sim(model, data)
        sim.frame_hook = lambda: display.record(data)
        try:
            run(sim, rig, random_start(rng), feedback)
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
            display.colours = ["magenta", "green"]
            sim.video_fps = 5  # refresh the camera panels 5 times per simulated second
            sim.frame_hook = lambda: display.to_viewer(viewer, data)
            run(sim, rig, random_start(rng), feedback)
        except Failure as err:
            print(f"FAILED: {err}")
        print("Close the window to exit.")
        while viewer.is_running():
            viewer.sync()


if __name__ == "__main__":
    main()
