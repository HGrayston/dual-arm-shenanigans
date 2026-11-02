"""Stereo vision from simulated cameras: find a coloured blob in two images, then triangulate it.

Only the rendered pixels and the camera poses are used here, never the simulator's
knowledge of where an object is. That makes this the same pipeline a real rig would run,
apart from calibration and lens distortion, which simulated cameras do not have.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco as mj
import numpy as np

# Each rule says which colour channel must dominate, and by how much.
COLOURS = {
    "red": lambda r, g, b: (r > 90) & (r > 2.0 * g) & (r > 2.0 * b),
    "green": lambda r, g, b: (g > 90) & (g > 1.8 * r) & (g > 1.8 * b),
    "magenta": lambda r, g, b: (r > 90) & (b > 90) & (r > 2.0 * g) & (b > 2.0 * g),
}


def add_stereo_cameras(spec: mj.MjSpec, baseline: float = 0.20, fovy: float = 38.0) -> None:
    """Add a parallel stereo pair above and in front of the table, ``baseline`` metres apart.

    Both cameras point the same way and are offset only sideways, as in a textbook rig. A
    point then appears at the same image row in both views, and depth = focal * baseline / disparity.
    """
    forward = np.array([0.0, 0.78, -0.62])
    right = np.array([1.0, 0.0, 0.0])
    up = np.cross(right, forward)
    mat = np.column_stack([right, up / np.linalg.norm(up), -forward / np.linalg.norm(forward)])
    quat = np.zeros(4)
    mj.mju_mat2Quat(quat, mat.flatten())
    for name, x in (("stereo_left", -baseline / 2), ("stereo_right", baseline / 2)):
        spec.worldbody.add_camera(name=name, pos=[x, -0.75, 1.10], quat=quat, fovy=fovy)
    # Not part of the rig: a closer view of the action, used for the videos.
    view = np.array([0.0, 0.84, -0.54])
    up_view = np.cross(right, view)
    mat = np.column_stack([right, up_view / np.linalg.norm(up_view), -view / np.linalg.norm(view)])
    mj.mju_mat2Quat(quat, mat.flatten())
    spec.worldbody.add_camera(name="demo_view", pos=[0.0, -0.62, 0.92], quat=quat, fovy=48)


@dataclass
class Detection:
    position: np.ndarray  # metres, world frame
    pixels: tuple[tuple[float, float], tuple[float, float]]  # (u, v) in the left and right image
    area: tuple[int, int]  # blob size in pixels, a rough measure of how much is visible
    gap: float  # m, how far apart the two lines of sight pass; large means a bad match


def find_blob(image: np.ndarray, colour: str, min_pixels: int = 12) -> tuple[tuple[float, float], int] | None:
    """Return the centre ``(u, v)`` and pixel count of a colour in an image, or ``None``."""
    r, g, b = (image[..., i].astype(float) for i in range(3))
    vs, us = np.nonzero(COLOURS[colour](r, g, b))
    if len(us) < min_pixels:
        return None
    # Use the median to reject stray pixels, then average what's close to it for subpixel accuracy.
    cu, cv = np.median(us), np.median(vs)
    keep = np.hypot(us - cu, vs - cv) < 3 * max(us.std(), vs.std(), 1.0)
    return (float(us[keep].mean()), float(vs[keep].mean())), int(keep.sum())


def find_circle(image: np.ndarray, colour: str, min_pixels: int = 12, tries: int = 200):
    """Find a partly hidden ball: fit a circle to the edge of its coloured region.

    A blob's centre shifts when a gripper hides part of the ball, but the ball's visible edge is
    still part of the same circle. RANSAC: try circles through 3 random edge pixels, keep the one
    most edge pixels agree with, so edges made by whatever is blocking the view get ignored.
    Returns the centre ``(u, v)`` and the pixel count, or ``None``.
    """
    r, g, b = (image[..., i].astype(float) for i in range(3))
    mask = COLOURS[colour](r, g, b)
    if mask.sum() < min_pixels:
        return None
    inner = mask.copy()
    inner[1:-1, 1:-1] &= mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
    vs, us = np.nonzero(mask & ~inner)
    pts = np.column_stack([us, vs]).astype(float)
    if len(pts) < 8:
        return find_blob(image, colour, min_pixels)
    rng = np.random.default_rng(0)  # fixed, so the same image always gives the same answer
    best, best_count = None, 0
    for _ in range(tries):
        fit = _circle_fit(pts[rng.choice(len(pts), 3, replace=False)])
        if fit is None or not 4.0 < fit[2] < 120.0:
            continue
        count = int((np.abs(np.hypot(*(pts - fit[:2]).T) - fit[2]) < 1.2).sum())
        if count > best_count:
            best, best_count = fit, count
    if best is None or best_count < 8:
        return find_blob(image, colour, min_pixels)
    keep = np.abs(np.hypot(*(pts - best[:2]).T) - best[2]) < 1.2
    refined = _circle_fit(pts[keep])
    centre = refined if refined is not None else best
    return (float(centre[0]), float(centre[1])), int(mask.sum())


def _circle_fit(pts: np.ndarray):
    """Least-squares circle through points. Returns ``(u, v, radius)``, or ``None`` if they're in a line."""
    a = np.column_stack([2 * pts, np.ones(len(pts))])
    sol, _, rank, _ = np.linalg.lstsq(a, (pts**2).sum(axis=1), rcond=None)
    if rank < 3:
        return None
    return np.array([sol[0], sol[1], math.sqrt(max(sol[2] + sol[0] ** 2 + sol[1] ** 2, 0.0))])


class StereoRig:
    """A pair of cameras in the scene, with what's needed to go from pixels to 3D."""

    def __init__(self, model: mj.MjModel, width: int = 640, height: int = 480, names=("stereo_left", "stereo_right")):
        self.model, self.width, self.height = model, width, height
        self.names = names
        self.renderer = mj.Renderer(model, height, width)
        ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, n) for n in names]
        self.cam_ids = ids
        self.focal = (height / 2) / math.tan(math.radians(model.cam_fovy[ids[0]]) / 2)  # pixels

    @property
    def baseline(self) -> float:
        return float(np.linalg.norm(self.model.cam_pos[self.cam_ids[0]] - self.model.cam_pos[self.cam_ids[1]]))

    def capture(self, data: mj.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Render what the two cameras see right now."""
        images = []
        for name in self.names:
            self.renderer.update_scene(data, camera=name)
            images.append(self.renderer.render().copy())
        return images[0], images[1]

    def ray(self, data: mj.MjData, cam: int, pixel: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        """World-frame origin and direction of the line of sight through a pixel."""
        u, v = pixel
        # MuJoCo cameras look along their -z axis, with +x right and +y up.
        d_cam = np.array([u + 0.5 - self.width / 2, self.height / 2 - v - 0.5, -self.focal])
        rot = data.cam_xmat[self.cam_ids[cam]].reshape(3, 3)
        d = rot @ d_cam
        return data.cam_xpos[self.cam_ids[cam]].copy(), d / np.linalg.norm(d)

    def triangulate(self, data: mj.MjData, left_px, right_px) -> tuple[np.ndarray, float]:
        """Point closest to both lines of sight, and how far apart those lines pass."""
        o1, d1 = self.ray(data, 0, left_px)
        o2, d2 = self.ray(data, 1, right_px)
        # Solve for the distances along each ray where they come closest.
        a = np.array([[d1 @ d1, -d1 @ d2], [d1 @ d2, -d2 @ d2]])
        t = np.linalg.solve(a, np.array([(o2 - o1) @ d1, (o2 - o1) @ d2]))
        p1, p2 = o1 + t[0] * d1, o2 + t[1] * d2
        return (p1 + p2) / 2, float(np.linalg.norm(p1 - p2))

    def locate(self, data: mj.MjData, colour: str, images=None, ball: bool = False) -> Detection | None:
        """Find a coloured object in both views. Returns ``None`` if either camera can't see it.

        ``ball=True`` fits a circle, which stays accurate when something hides part of the ball.
        """
        left, right = images if images is not None else self.capture(data)
        finder = find_circle if ball else find_blob
        a, b = finder(left, colour), finder(right, colour)
        if a is None or b is None:
            return None
        # A sanity check stereo gives you for free: with this rig, a real match lies on the same
        # image row in both views. If it doesn't, one camera is seeing something else.
        if abs(a[0][1] - b[0][1]) > 2.5:
            return None
        position, gap = self.triangulate(data, a[0], b[0])
        return Detection(position, (a[0], b[0]), (a[1], b[1]), gap)


def mark(image: np.ndarray, pixel: tuple[float, float], colour=(255, 255, 255), size: int = 9) -> np.ndarray:
    """Draw a crosshair on an image copy."""
    out = image.copy()
    h, w = out.shape[:2]
    u, v = int(round(pixel[0])), int(round(pixel[1]))
    out[max(v - 1, 0) : v + 1, max(u - size, 0) : min(u + size, w)] = colour
    out[max(v - size, 0) : min(v + size, h), max(u - 1, 0) : u + 1] = colour
    return out


class Display:
    """Shows the stereo views: as a composite video frame, or inside the interactive viewer."""

    def __init__(self, model: mj.MjModel, rig: StereoRig, camera: str = "demo_view"):
        self.model, self.rig, self.camera = model, rig, camera
        self.renderer = mj.Renderer(model, 480, 640)
        self.colours: list[str] = []
        self.writer = None

    def views(self, data: mj.MjData) -> tuple[np.ndarray, np.ndarray]:
        left, right = self.rig.capture(data)
        marked_l, marked_r = left, right
        for colour in self.colours:
            det = self.rig.locate(data, colour, (left, right), ball=colour == "red")
            if det is not None:
                marked_l = mark(marked_l, det.pixels[0])
                marked_r = mark(marked_r, det.pixels[1])
        return marked_l, marked_r

    def frame(self, data: mj.MjData) -> np.ndarray:
        """Overview on top, with the left and right stereo image below it."""
        self.renderer.update_scene(data, camera=self.camera)
        main = self.renderer.render()
        left, right = self.views(data)
        return np.vstack([main[:, :], np.hstack([left[::2, ::2], right[::2, ::2]])])

    def to_viewer(self, viewer, data: mj.MjData) -> None:
        """Show what the two cameras see, with detections marked, in the viewer's bottom corners."""
        left, right = self.views(data)
        left, right = left[::3, ::3], right[::3, ::3]
        h, w = left.shape[:2]
        width = viewer.viewport.width if viewer.viewport is not None else 2 * w
        viewer.set_images([(mj.MjrRect(0, 0, w, h), left), (mj.MjrRect(max(width - w, w), 0, w, h), right)])

    def record(self, data: mj.MjData) -> None:
        if self.writer is not None:
            self.writer.append_data(self.frame(data))


def draw_markers(scene: mj.MjvScene, points: list[tuple[np.ndarray, tuple[float, float, float, float], float]]) -> None:
    """Add spheres to a viewer or renderer scene: ``(position, rgba, radius)``."""
    for pos, rgba, radius in points:
        if scene.ngeom >= scene.maxgeom:
            return
        mj.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mj.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0, 0]),
            np.asarray(pos, float),
            np.eye(3).flatten(),
            np.asarray(rgba, np.float32),
        )
        scene.ngeom += 1
