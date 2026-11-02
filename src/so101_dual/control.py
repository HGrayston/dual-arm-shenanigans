"""Small toolkit for the showcase demos: extra objects, IK and smooth Cartesian moves.

The arm has 5 joints plus a gripper, so it cannot reach every position *and* every
orientation. ``solve_ik`` therefore gets the position exactly right first and then gets
as close to the requested orientation as the remaining freedom allows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco as mj
import numpy as np

from .scene import ARMS, READY_Q, ArmIds, arm_ids, build_spec

GRIPPER_OPEN = 1.2  # rad, roomy enough for the ball
FIXED_JAW_X = -0.0205  # m, inner surface of the fixed jaw, measured from the site along the pinch axis


def build_demo(add_objects: Callable[[mj.MjSpec], None], arm_separation: float = 0.50):
    """Compile the two arms and table plus whatever ``add_objects`` adds. Returns ``(model, data)``."""
    spec = build_spec(arm_separation, with_cube=False)
    add_objects(spec)
    model = spec.compile()
    # Friction in MuJoCo is slightly soft, so a held object slowly creeps. A high impratio
    # makes slip much harder, at some cost in speed. This is what keeps objects in the jaws.
    model.opt.impratio = 100
    model.opt.iterations = 50
    data = mj.MjData(model)
    key = model.key("ready")
    key.qpos[:] = model.qpos0
    for arm in ARMS:
        ids = arm_ids(model, arm)
        key.qpos[ids.qpos] = READY_Q
        key.ctrl[ids.actuator] = READY_Q
    mj.mj_resetDataKeyframe(model, data, key.id)
    mj.mj_forward(model, data)
    return model, data


def grasp_rotation(approach, pinch) -> np.ndarray:
    """Gripper orientation from where the fingers point and the axis the jaws close along."""
    x = np.asarray(approach, float)
    x = x / np.linalg.norm(x)
    z = np.asarray(pinch, float)
    z = z - z.dot(x) * x  # make it exactly perpendicular to the approach
    z = z / np.linalg.norm(z)
    return np.column_stack([x, np.cross(z, x), z])


def _rot_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Rotation vector (world frame) that turns ``current`` into ``target``."""
    q_t, q_c, q_neg, q_err = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
    mj.mju_mat2Quat(q_t, target.flatten())
    mj.mju_mat2Quat(q_c, current.flatten())
    mj.mju_negQuat(q_neg, q_c)
    mj.mju_mulQuat(q_err, q_t, q_neg)
    if q_err[0] < 0:
        q_err = -q_err
    err = np.zeros(3)
    mj.mju_quat2Vel(err, q_err, 1.0)
    return err


def solve_ik(
    model: mj.MjModel,
    scratch: mj.MjData,
    ids: ArmIds,
    point: np.ndarray,
    rotation: np.ndarray,
    q_start: np.ndarray,
    offset: float = 0.0,
    iters: int = 30,
) -> tuple[np.ndarray, float, float]:
    """Find 5 arm joint angles that put the grasp point at ``point`` with ``rotation``.

    ``offset`` moves the controlled point along the pinch axis, away from the site.
    Returns ``(q, position_error_m, rotation_error_rad)``. ``scratch`` is a spare
    ``MjData``, so the running simulation is never touched.
    """
    q = np.array(q_start[:5], float)
    site = ids.gripper_site
    body = model.site_bodyid[site]
    dof = ids.dof[:5]
    low, high = ids.ctrlrange[:5, 0], ids.ctrlrange[:5, 1]
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    damping = 1e-5
    pos_err, rot_err = np.zeros(3), np.zeros(3)
    for _ in range(iters):
        scratch.qpos[ids.qpos[:5]] = q
        mj.mj_kinematics(model, scratch)
        mj.mj_comPos(model, scratch)
        rot = scratch.site_xmat[site].reshape(3, 3)
        here = scratch.site_xpos[site] + offset * rot[:, 2]
        pos_err = point - here
        rot_err = _rot_error(rotation, rot)
        mj.mj_jac(model, scratch, jacp, jacr, here, body)
        # Position first; orientation gets whatever freedom is left (null-space projection).
        jp, jr = jacp[:, dof], jacr[:, dof]
        jp_pinv = jp.T @ np.linalg.inv(jp @ jp.T + damping * np.eye(3))
        dq = jp_pinv @ pos_err
        null = np.eye(5) - jp_pinv @ jp
        jr_null = jr @ null
        dq = dq + jr_null.T @ np.linalg.solve(jr_null @ jr_null.T + 1e-3 * np.eye(3), rot_err - jr @ dq)
        q = np.clip(q + np.clip(dq, -0.25, 0.25), low, high)
    return q, float(np.linalg.norm(pos_err)), float(np.linalg.norm(rot_err))


@dataclass
class ArmState:
    ids: ArmIds
    q: np.ndarray  # commanded arm joints (5)
    gripper: float
    point: np.ndarray | None = None
    rotation: np.ndarray | None = None
    offset: float = 0.0


def smooth(s: float) -> float:
    """Minimum-jerk ramp from 0 to 1: starts and stops gently."""
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


class Sim:
    """Runs the physics and moves the arms, with or without a viewer or a video."""

    def __init__(self, model: mj.MjModel, data: mj.MjData, realtime: bool = False, viewer=None):
        self.model, self.data = model, data
        self.realtime, self.viewer = realtime, viewer
        self.scratch = mj.MjData(model)
        self.arms = {
            a: ArmState(arm_ids(model, a), READY_Q[:5].copy(), float(READY_Q[5])) for a in ARMS
        }
        self.frame_hook: Callable[[], None] | None = None
        self.video_fps = 15
        self._last_hook = -1.0
        self._apply()

    def _apply(self) -> None:
        for arm in self.arms.values():
            self.data.ctrl[arm.ids.actuator[:5]] = arm.q
            self.data.ctrl[arm.ids.actuator[5]] = arm.gripper

    def step(self, seconds: float = 0.0, on_tick: Callable[[], None] | None = None) -> None:
        """Advance the physics, keeping any viewer and video up to date."""
        dt = self.model.opt.timestep
        for _ in range(max(1, int(round(seconds / dt)))):
            start = time.time()
            if on_tick is not None:
                on_tick()
            self._apply()
            if self.viewer is not None:
                if not self.viewer.is_running():
                    raise SystemExit(0)
                with self.viewer.lock():
                    mj.mj_step(self.model, self.data)
                self.viewer.sync()
            else:
                mj.mj_step(self.model, self.data)
            if self.frame_hook is not None and self.data.time - self._last_hook >= 1 / self.video_fps:
                self._last_hook = self.data.time
                self.frame_hook()
            if self.realtime:
                time.sleep(max(0.0, dt - (time.time() - start)))

    def wait(self, seconds: float) -> None:
        self.step(seconds)

    def set_pose(self, arm: str, point, rotation, offset: float = 0.0) -> None:
        """Declare where an arm's grasp point is now, without moving it (used before ``move``)."""
        state = self.arms[arm]
        state.point, state.rotation, state.offset = np.array(point, float), rotation, offset

    def joints(self, goals: dict[str, np.ndarray], seconds: float, gripper: dict[str, float] | None = None) -> None:
        """Move arms in joint space to the given 5- or 6-vectors, all finishing together."""
        gripper = gripper or {}
        start = {a: (s.q.copy(), s.gripper) for a, s in self.arms.items()}
        t0 = self.data.time

        def tick() -> None:
            s = smooth((self.data.time - t0) / seconds)
            for a, goal in goals.items():
                self.arms[a].q = start[a][0] + s * (np.asarray(goal[:5]) - start[a][0])
                self.arms[a].point = None
            for a, g in gripper.items():
                self.arms[a].gripper = start[a][1] + s * (g - start[a][1])

        self.step(seconds, tick)

    def move(
        self,
        goals: dict[str, tuple],
        seconds: float,
        correct: Callable[[str, np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """Straight-line moves of the grasp points, all finishing together.

        ``goals`` maps an arm to ``(point, rotation)`` or ``(point, rotation, offset)``.
        ``correct(arm, point)`` may shift the target at every control tick, which is how
        a sensor closes the loop while the motion is running.
        """
        plan = {}
        for a, goal in goals.items():
            state = self.arms[a]
            point, rotation = np.array(goal[0], float), goal[1]
            offset = goal[2] if len(goal) > 2 else state.offset
            if state.point is None:
                self._locate(a, offset)
            state.offset = offset
            plan[a] = (state.point.copy(), point, state.rotation.copy(), rotation)
        t0, last = self.data.time, [-1.0]

        def tick() -> None:
            if self.data.time - last[0] < 0.02:  # 50 Hz control
                return
            last[0] = self.data.time
            s = smooth((self.data.time - t0) / seconds)
            for a, (p0, p1, r0, r1) in plan.items():
                state = self.arms[a]
                state.point = p0 + s * (p1 - p0)
                state.rotation = _slerp(r0, r1, s)
                target = correct(a, state.point) if correct else state.point
                state.q, _, _ = solve_ik(
                    self.model, self.scratch, state.ids, target, state.rotation, state.q, state.offset, iters=6
                )

        self.step(seconds, tick)

    def _locate(self, arm: str, offset: float) -> None:
        state = self.arms[arm]
        self.scratch.qpos[state.ids.qpos[:5]] = state.q
        mj.mj_kinematics(self.model, self.scratch)
        rot = self.scratch.site_xmat[state.ids.gripper_site].reshape(3, 3).copy()
        state.point = self.scratch.site_xpos[state.ids.gripper_site] + offset * rot[:, 2]
        state.rotation = rot

    def close_until(self, arm: str, force: float = 1.5, speed: float = 0.5, limit: float = 4.0) -> float:
        """Close the jaws slowly until the gripper servo feels ``force`` (N m), then hold.

        This is force feedback: the arm finds the object's width by touching it, so it does
        not need to know the size in advance. Returns the gripper angle where it stopped.
        """
        state = self.arms[arm]
        actuator, dof = state.ids.actuator[5], state.ids.dof[5]
        t0, touching, done = self.data.time, [0], [False]

        def tick() -> None:
            if done[0]:
                return
            state.gripper = max(state.gripper - speed * self.model.opt.timestep, -0.17)
            # Touching means the jaw is pressing shut but has nearly stopped moving.
            stalled = abs(self.data.qvel[dof]) < 0.3 * speed
            touching[0] = touching[0] + 1 if stalled and self.data.actuator_force[actuator] < -force else 0
            done[0] = touching[0] >= 3

        while not done[0] and self.data.time - t0 < limit:
            self.step(0.02, tick)
        self.step(0.3)  # let the grip settle
        return state.gripper

    def grip(self, goals: dict[str, float], seconds: float = 0.6) -> None:
        """Ramp gripper targets (rad) while the arms hold still."""
        start = {a: self.arms[a].gripper for a in goals}
        t0 = self.data.time

        def tick() -> None:
            s = smooth((self.data.time - t0) / seconds)
            for a, g in goals.items():
                self.arms[a].gripper = start[a] + s * (g - start[a])

        self.step(seconds, tick)


def _slerp(r0: np.ndarray, r1: np.ndarray, s: float) -> np.ndarray:
    """Interpolate between two rotation matrices."""
    err = _rot_error(r1, r0) * s
    angle = np.linalg.norm(err)
    if angle < 1e-9:
        return r0.copy()
    q, out = np.zeros(4), np.zeros(9)
    mj.mju_axisAngle2Quat(q, err / angle, angle)
    mj.mju_quat2Mat(out, q)
    return out.reshape(3, 3) @ r0


def open_video(path: str | Path, fps: int = 15):
    """Return an imageio writer; ``.mp4`` needs the imageio-ffmpeg package, ``.gif`` does not."""
    import imageio.v2 as iio

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".gif"):
        return iio.get_writer(path, duration=1000 / fps, loop=0)
    return iio.get_writer(path, fps=fps, macro_block_size=None)
