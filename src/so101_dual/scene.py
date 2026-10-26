"""Two-arm SO-101 MuJoCo scene.

The robot model itself is the MuJoCo Menagerie ``robotstudio_so101`` MJCF
(Apache-2.0, see ``assets/so101/``). This module places two copies of it on a
table facing each other, using MuJoCo's ``MjSpec`` attach API so the upstream
XML is never edited or duplicated. Every element of an arm gets a prefix, e.g.
``left_shoulder_pan`` / ``right_gripperframe``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco as mj
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SO101_DIR = REPO_ROOT / "assets" / "so101"
SO101_XML = SO101_DIR / "so101.xml"

ARMS = ("left", "right")
JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

TABLE_HEIGHT = 0.40  # m, top surface of the table
CUBE_HALF = 0.015  # m, 3 cm cube

_SCENE_XML = """
<mujoco model="so101_dual_scene">
  <option integrator="implicitfast" timestep="0.005" cone="elliptic"
          iterations="10" ls_iterations="20" impratio="10"/>
  <compiler angle="radian" autolimits="true"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="90" elevation="-25" offwidth="1280" offheight="720"/>
    <scale contactwidth="0.1" contactheight="0.05" forcewidth="0.02"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.2"/>
    <material name="table" rgba="0.72 0.60 0.45 1"/>
    <material name="cube" rgba="0.85 0.15 0.15 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="groundplane"/>

    <!-- table: top surface at z = {table_h} -->
    <geom name="table_top" type="box" size="0.60 0.35 0.02"
          pos="0 0 {table_top_z}" material="table" friction="1 0.005 0.0001"/>
    <geom name="table_leg_1" type="box" size="0.02 0.02 {leg_half}" pos=" 0.56  0.31 {leg_z}" material="table"/>
    <geom name="table_leg_2" type="box" size="0.02 0.02 {leg_half}" pos="-0.56  0.31 {leg_z}" material="table"/>
    <geom name="table_leg_3" type="box" size="0.02 0.02 {leg_half}" pos=" 0.56 -0.31 {leg_z}" material="table"/>
    <geom name="table_leg_4" type="box" size="0.02 0.02 {leg_half}" pos="-0.56 -0.31 {leg_z}" material="table"/>

    <camera name="overview" pos="0 -1.05 1.05" xyaxes="1 0 0 0 0.62 0.78"/>
    <camera name="top" pos="0 0 1.6" xyaxes="1 0 0 0 1 0"/>
  </worldbody>
</mujoco>
"""


def _scene_xml(with_cube: bool) -> str:
    xml = _SCENE_XML.format(
        table_h=TABLE_HEIGHT,
        table_top_z=TABLE_HEIGHT - 0.02,
        leg_half=(TABLE_HEIGHT - 0.04) / 2,
        leg_z=(TABLE_HEIGHT - 0.04) / 2,
    )
    if with_cube:
        cube = f"""
    <body name="cube" pos="0 0 {TABLE_HEIGHT + CUBE_HALF}">
      <freejoint name="cube_free"/>
      <geom name="cube" type="box" size="{CUBE_HALF} {CUBE_HALF} {CUBE_HALF}"
            mass="0.02" material="cube" condim="4" friction="1 0.005 0.0001"/>
    </body>
  </worldbody>"""
        xml = xml.replace("</worldbody>", cube, 1)
    return xml


def build_spec(arm_separation: float = 0.50, with_cube: bool = True) -> mj.MjSpec:
    """Return an ``MjSpec`` with two SO-101 arms facing each other.

    Args:
        arm_separation: distance between the two arm bases along x (m).
        with_cube: add a free 3 cm cube at the centre of the table.
    """
    spec = mj.MjSpec.from_string(_scene_xml(with_cube))
    spec.meshdir = str(SO101_DIR / "assets")

    placements = {
        "left": (-arm_separation / 2, 0.0),  # faces +x
        "right": (+arm_separation / 2, math.pi),  # rotated to face -x
    }
    for name, (x, yaw) in placements.items():
        arm = mj.MjSpec.from_file(str(SO101_XML))
        frame = spec.worldbody.add_frame(
            pos=[x, 0.0, TABLE_HEIGHT], euler=[0.0, 0.0, yaw]
        )
        spec.attach(arm, frame=frame, prefix=f"{name}_")
    # Placeholder keyframe; its values are filled in by load() once the joint
    # layout is known.
    spec.add_key(name="ready")
    return spec


# Folded "ready" pose, joint order as JOINTS (rad). With both arms in this pose
# the scene is collision-free; at all-zeros the arms reach into each other.
READY_Q = np.array([0.0, -1.3, 0.7, 1.6, 0.0, 0.9])


def load(
    arm_separation: float = 0.50, with_cube: bool = True, ready: bool = True
):
    """Compile the scene. Returns ``(model, data)``.

    With ``ready=True`` (default) the arms start in :data:`READY_Q` and the
    ``ready`` keyframe is stored in the model, so pressing backspace in the
    viewer resets to it.
    """
    model = build_spec(arm_separation, with_cube).compile()
    data = mj.MjData(model)

    key = model.key("ready")
    key.qpos[:] = model.qpos0
    for arm in ARMS:
        ids = arm_ids(model, arm)
        key.qpos[ids.qpos] = READY_Q
        key.ctrl[ids.actuator] = READY_Q

    if ready:
        mj.mj_resetDataKeyframe(model, data, key.id)
    mj.mj_forward(model, data)
    return model, data


@dataclass(frozen=True)
class ArmIds:
    """Index handles for one arm, ordered as :data:`JOINTS`."""

    name: str
    actuator: np.ndarray  # ctrl indices
    qpos: np.ndarray  # qpos addresses
    dof: np.ndarray  # qvel / dof addresses
    ctrlrange: np.ndarray  # (6, 2)
    gripper_site: int  # tool-centre-point site id


def arm_ids(model: mj.MjModel, arm: str) -> ArmIds:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    act = np.array(
        [mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, f"{arm}_{j}") for j in JOINTS]
    )
    jnt = np.array(
        [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f"{arm}_{j}") for j in JOINTS]
    )
    site = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, f"{arm}_gripperframe")
    if (act < 0).any() or (jnt < 0).any() or site < 0:
        raise RuntimeError(f"could not resolve names for arm {arm!r}")
    return ArmIds(
        name=arm,
        actuator=act,
        qpos=model.jnt_qposadr[jnt].copy(),
        dof=model.jnt_dofadr[jnt].copy(),
        ctrlrange=model.actuator_ctrlrange[act].copy(),
        gripper_site=site,
    )
