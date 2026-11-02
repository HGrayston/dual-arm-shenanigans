"""Demo: how two cameras turn into a 3D sensor. Tracks a bouncing ball and builds a 3D model of a scene.

    uv run --extra dev python scripts/demo_stereo.py                # live: a green dot shows the 3D estimate
    uv run --extra dev python scripts/demo_stereo.py --report out/stereo   # no window: depth map, 3D model, numbers

The idea: a nearby object appears further apart in the two images than a distant one. That
shift is called disparity. With focal length f (pixels) and camera spacing b (metres):

    depth = f * b / disparity

The report covers:
  * tracking: the ball's 3D path measured by the cameras, against the true path
  * spacing: how the camera baseline changes accuracy
  * dense 3D: matching every image patch to get a depth map and a point cloud you can open in
    any 3D viewer (for example Blender or MeshLab), checked against the simulator's true depth.
    Note the plain blue block: blank surfaces are stereo's classic blind spot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco as mj
import numpy as np

from so101_dual import TABLE_HEIGHT
from so101_dual.control import Sim, build_demo, open_video
from so101_dual.vision import Display, StereoRig, add_stereo_cameras, draw_markers

BALL_RADIUS = 0.025
BASELINES = (0.05, 0.10, 0.20, 0.40)


def add_objects(spec: mj.MjSpec) -> None:
    """A ball to track, some shapes to model, and a speckled table to give the matcher texture."""
    spec.add_texture(
        name="speckle",
        type=mj.mjtTexture.mjTEXTURE_2D,
        builtin=mj.mjtBuiltin.mjBUILTIN_FLAT,
        mark=mj.mjtMark.mjMARK_RANDOM,
        random=0.25,
        width=512,
        height=512,
        rgb1=[0.72, 0.60, 0.45],
        markrgb=[0.15, 0.12, 0.08],
    )
    material = spec.material("table")
    material.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = "speckle"
    material.texrepeat = [3, 3]
    material.texuniform = True
    material.rgba = [1, 1, 1, 1]

    ball = spec.worldbody.add_body(name="ball", pos=[-0.04, -0.08, TABLE_HEIGHT + BALL_RADIUS])
    ball.add_freejoint(name="ball_free")
    ball.add_geom(
        name="ball", type=mj.mjtGeom.mjGEOM_SPHERE, size=[BALL_RADIUS, 0, 0], mass=0.02, rgba=[0.9, 0.1, 0.1, 1],
        solref=[-3000, -6],  # springy, so it bounces
        condim=6, friction=[1.0, 0.01, 0.003],  # rolling friction, so it settles instead of rolling away
    )
    # The first two blocks are speckled. The third is plain on purpose: stereo matching can't
    # find depth on a blank surface, because every patch of it looks the same.
    block_material = spec.add_material(name="speckled_block", rgba=[1, 1, 1, 1], texrepeat=[6, 6], texuniform=True)
    block_material.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = "speckle"
    for i, (x, half, material_name, rgb) in enumerate(
        [(-0.16, 0.03, "speckled_block", None), (0.0, 0.05, "speckled_block", None), (0.17, 0.04, None, [0.3, 0.3, 0.8, 1])]
    ):
        geom = spec.worldbody.add_geom(
            name=f"block_{i}", type=mj.mjtGeom.mjGEOM_BOX, size=[half, half, half], pos=[x, -0.27, TABLE_HEIGHT + half]
        )
        if material_name:
            geom.material = material_name
        else:
            geom.rgba = rgb
    add_stereo_cameras(spec)


def reset_ball(model: mj.MjModel, data: mj.MjData, rng: np.random.Generator) -> None:
    """Throw the ball onto the table from a new spot."""
    joint = model.joint("ball_free")
    adr, dof = joint.qposadr[0], joint.dofadr[0]
    data.qpos[adr : adr + 3] = [rng.uniform(-0.1, 0.0), rng.uniform(-0.15, -0.11), TABLE_HEIGHT + 0.30]
    data.qvel[dof : dof + 6] = 0
    data.qvel[dof : dof + 3] = [rng.uniform(0.05, 0.2), rng.uniform(-0.03, 0.03), 0.0]


def track(sim: Sim, rig: StereoRig, seconds: float = 1.5, rate: float = 20.0):
    """Follow the ball. Returns ``(estimates, truths, misses)`` as arrays of 3D points."""
    model, data = sim.model, sim.data
    ball = model.body("ball").id
    estimates, truths, misses = [], [], 0
    for _ in range(int(seconds * rate)):
        sim.step(1.0 / rate)
        det = rig.locate(data, "red", ball=True)
        if det is None:
            misses += 1
            continue
        estimates.append(det.position)
        truths.append(data.xpos[ball].copy())
    return np.array(estimates), np.array(truths), misses


def box_sum(image: np.ndarray, radius: int) -> np.ndarray:
    """Sum over a square window around each pixel, in constant time per pixel."""
    pad = np.pad(image, ((radius + 1, radius), (radius + 1, radius)), mode="edge")
    table = pad.cumsum(0).cumsum(1)
    size = 2 * radius + 1
    return table[size:, size:] - table[:-size, size:] - table[size:, :-size] + table[:-size, :-size]


def disparity_map(left: np.ndarray, right: np.ndarray, low: int, high: int, radius: int = 4) -> np.ndarray:
    """Block matching: for each patch in the left image, find the best sideways shift in the right.

    Returns disparity in pixels, ``nan`` where the match isn't trustworthy (no texture, or
    the two best matches are nearly equally good).
    """
    grey_l = left.astype(float).mean(axis=2)
    grey_r = right.astype(float).mean(axis=2)
    costs = np.stack([box_sum(np.abs(grey_l - np.roll(grey_r, d, axis=1)), radius) for d in range(low, high)])
    best = costs.argmin(axis=0)
    rows, cols = np.indices(best.shape)
    inner = np.clip(best, 1, len(costs) - 2)
    c0, c1, c2 = costs[inner - 1, rows, cols], costs[inner, rows, cols], costs[inner + 1, rows, cols]
    # A parabola through the three best costs gives a shift finer than one pixel.
    bend = c0 - 2 * c1 + c2
    fine = np.where(bend > 1e-6, 0.5 * (c0 - c2) / np.maximum(bend, 1e-6), 0.0)
    disparity = low + inner + fine
    texture = box_sum(np.abs(np.diff(grey_l, axis=1, prepend=grey_l[:, :1])), radius)
    disparity[texture < 30.0 * (2 * radius + 1)] = np.nan
    disparity[(best == 0) | (best == len(costs) - 1)] = np.nan
    disparity[:, : high + radius] = np.nan
    # Uniqueness test: the best match must clearly beat every other shift (ignoring its neighbours).
    others = costs.copy()
    for step in range(-2, 3):
        others[np.clip(best + step, 0, len(costs) - 1), rows, cols] = np.inf
    disparity[costs[best, rows, cols] > 0.85 * others.min(axis=0)] = np.nan
    return disparity



def true_depth(rig: StereoRig, data: mj.MjData) -> np.ndarray:
    """The simulator's actual distance along the viewing axis for each left-camera pixel."""
    rig.renderer.enable_depth_rendering()
    rig.renderer.update_scene(data, camera=rig.names[0])
    depth = rig.renderer.render().copy()
    rig.renderer.disable_depth_rendering()
    return depth


def to_points(rig: StereoRig, data: mj.MjData, depth: np.ndarray) -> np.ndarray:
    """Turn a depth image into world-frame 3D points, one per valid pixel (others are ``nan``)."""
    vs, us = np.indices(depth.shape)
    x = (us + 0.5 - rig.width / 2) / rig.focal * depth
    y = (rig.height / 2 - vs - 0.5) / rig.focal * depth
    cam = np.stack([x, y, -depth], axis=-1)
    rot = data.cam_xmat[rig.cam_ids[0]].reshape(3, 3)
    return cam @ rot.T + data.cam_xpos[rig.cam_ids[0]]


def colourise(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Near = bright, far = dark, invalid = deep blue."""
    level = (255 * np.clip((high - values) / (high - low), 0, 1))
    level = np.nan_to_num(level)
    image = np.stack([level, level, level], axis=-1).astype(np.uint8)
    image[~np.isfinite(values)] = (10, 20, 70)
    return image


def write_ply(path: Path, points: np.ndarray, colours: np.ndarray) -> None:
    """Save a coloured point cloud in plain-text PLY format."""
    with open(path, "w", encoding="ascii") as f:
        f.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        for p, c in zip(points, colours):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")


def dense_report(model: mj.MjModel, data: mj.MjData, rig: StereoRig, out: Path) -> dict:
    """Build the depth map and 3D model of a still scene, and check them against the truth."""
    import imageio.v2 as iio

    mj.mj_resetDataKeyframe(model, data, model.key("ready").id)
    Sim(model, data).wait(1.5)  # let the ball come to rest
    left, right = rig.capture(data)

    # Only search the shifts the workspace can produce: from 1.8 m away to 0.8 m away.
    low = int(rig.focal * rig.baseline / 1.8)
    high = int(rig.focal * rig.baseline / 0.8)
    disparity = disparity_map(left, right, low, high)
    depth = rig.focal * rig.baseline / disparity
    truth = true_depth(rig, data)

    # Score only the workspace: pixels that truly show the table or something on it.
    actual = to_points(rig, data, truth)
    workspace = (actual[..., 2] > TABLE_HEIGHT - 0.01) & (np.abs(actual[..., 0]) < 0.6) & (np.abs(actual[..., 1]) < 0.35)
    # MuJoCo's depth render measures along the camera axis, which is what the formula gives too.
    valid = np.isfinite(depth) & workspace
    error = np.abs(depth - truth)[valid]

    top = np.hstack([left, right])
    near, far = 0.7, 1.9
    shown = np.where(workspace, depth, np.nan)  # floor and sky aren't part of the model
    bottom = np.hstack([colourise(shown, near, far), colourise(np.where(workspace, truth, np.nan), near, far)])
    iio.imwrite(out.with_name(out.name + "_depth.png"), np.vstack([top, bottom]))

    # Save the workspace as a point cloud, dropping the few points that matched wrongly and land below the table.
    points = to_points(rig, data, np.nan_to_num(depth, nan=1.0))
    keep = valid & (points[..., 2] > TABLE_HEIGHT - 0.03)
    write_ply(out.with_name(out.name + "_model.ply"), points[keep], left[keep])

    return {
        "focal_px": rig.focal,
        "valid_pct": 100 * float(valid.sum() / workspace.sum()),
        "median_error_mm": 1000 * float(np.median(error)),
        "p90_error_mm": 1000 * float(np.percentile(error, 90)),
        "points": int(keep.sum()),
    }


def baseline_report(model: mj.MjModel, data: mj.MjData, rig: StereoRig, trials: int = 12, noise_px: float = 0.5) -> list[dict]:
    """Measure 3D error for several camera spacings, with realistic pixel noise added.

    Simulated images are unnaturally clean, so each detection is shifted by ``noise_px``
    of random error to show what real sensors would face.
    """
    rows = []
    for baseline in BASELINES:
        # Move the two cameras apart in the model, the same as sliding them along a real rig bar.
        model.cam_pos[rig.cam_ids[0], 0] = -baseline / 2
        model.cam_pos[rig.cam_ids[1], 0] = baseline / 2
        rng = np.random.default_rng(1)
        ball = model.body("ball").id
        joint = model.joint("ball_free")
        errors_depth, errors_all, missed = [], [], 0
        for _ in range(trials):
            data.qpos[joint.qposadr[0] : joint.qposadr[0] + 3] = [
                rng.uniform(-0.2, 0.2), rng.uniform(-0.1, 0.15), TABLE_HEIGHT + rng.uniform(0.03, 0.3),
            ]
            data.qvel[joint.dofadr[0] : joint.dofadr[0] + 6] = 0
            mj.mj_forward(model, data)
            det = rig.locate(data, "red", ball=True)
            if det is None:
                missed += 1
                continue
            for _ in range(20):
                a = np.array(det.pixels[0]) + rng.normal(0, noise_px, 2)
                b = np.array(det.pixels[1]) + rng.normal(0, noise_px, 2)
                estimate, _ = rig.triangulate(data, a, b)
                wrong = estimate - data.xpos[ball]
                forward = -data.cam_xmat[rig.cam_ids[0]].reshape(3, 3)[:, 2]
                errors_depth.append(abs(wrong @ forward))
                errors_all.append(np.linalg.norm(wrong))
        rows.append(
            {
                "baseline_m": baseline,
                "measured_depth_mm": 1000 * float(np.mean(errors_depth)),
                "measured_3d_mm": 1000 * float(np.mean(errors_all)),
                "missed": missed,
            }
        )
    model.cam_pos[rig.cam_ids[0], 0] = -BASELINES[2] / 2  # back to the default rig
    model.cam_pos[rig.cam_ids[1], 0] = BASELINES[2] / 2
    return rows


def run_report(path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Tracking a thrown ball (clean images):")
    model, data = build_demo(add_objects)
    rig = StereoRig(model)
    sim = Sim(model, data)
    rng = np.random.default_rng(0)
    errors = []
    for _ in range(3):
        reset_ball(model, data, rng)
        est, truth, misses = track(sim, rig)
        errors.append(np.linalg.norm(est - truth, axis=1))
        print(f"  {len(est)} frames, {misses} missed, mean error {1000 * errors[-1].mean():.1f} mm")
    print(f"  overall: mean {1000 * np.concatenate(errors).mean():.1f} mm, worst {1000 * np.concatenate(errors).max():.1f} mm")

    print("\nCamera spacing, with 0.5 px of pixel noise added:")
    print("  baseline   depth error   3D error   ball out of view")
    for row in baseline_report(model, data, rig):
        print(
            f"  {row['baseline_m']:.2f} m     {row['measured_depth_mm']:6.1f} mm   "
            f"{row['measured_3d_mm']:6.1f} mm   {row['missed']}"
        )

    print("\nDense 3D model of the scene:")
    stats = dense_report(model, data, rig, out)
    print(f"  workspace pixels with a depth: {stats['valid_pct']:.0f}%")
    print(f"  depth error against the truth: median {stats['median_error_mm']:.1f} mm, 90% within {stats['p90_error_mm']:.1f} mm")
    print(f"  wrote {out.name}_depth.png (top: left and right camera; bottom: measured and true depth)")
    print(f"  wrote {out.name}_model.ply ({stats['points']} points)")


def run_live(model: mj.MjModel, data: mj.MjData, rig: StereoRig) -> None:
    import mujoco.viewer

    rng = np.random.default_rng()
    print("Green dot = where the cameras think the ball is. Drag the ball with Ctrl + right mouse button.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim = Sim(model, data, realtime=True, viewer=viewer)
        display = Display(model, rig)
        display.colours = ["red"]
        ball = model.body("ball").id
        while viewer.is_running():
            reset_ball(model, data, rng)
            for i in range(60):
                sim.step(0.05)
                det = rig.locate(data, "red", ball=True)
                viewer.user_scn.ngeom = 0
                display.to_viewer(viewer, data)
                if det is not None:
                    draw_markers(viewer.user_scn, [(det.position, (0.1, 1.0, 0.2, 0.9), 0.008)])
                    if i % 10 == 0:
                        error = 1000 * np.linalg.norm(det.position - data.xpos[ball])
                        print(f"ball at {np.round(det.position, 3)}, error {error:.1f} mm")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--report", metavar="PREFIX", help="measure everything and write files, with no window")
    ap.add_argument("--video", help="write a video of the tracking instead of opening a window")
    args = ap.parse_args()
    if args.report:
        run_report(args.report)
        return

    model, data = build_demo(add_objects)
    rig = StereoRig(model)
    if args.video:
        display = Display(model, rig)
        display.colours = ["red"]
        display.writer = open_video(args.video)
        sim = Sim(model, data)
        sim.frame_hook = lambda: display.record(data)
        rng = np.random.default_rng(0)
        for _ in range(3):
            reset_ball(model, data, rng)
            sim.step(2.0)
        display.writer.close()
        print(f"wrote {args.video}")
        return
    run_live(model, data, rig)


if __name__ == "__main__":
    main()
