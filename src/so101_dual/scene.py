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
SCENE_XML = REPO_ROOT / "assets" / "scene.xml"

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


def _scene_xml(with_cube: bool) -> str:
    xml = SCENE_XML.read_text(encoding="utf-8").format(
        table_h=TABLE_HEIGHT,
        table_top_z=TABLE_HEIGHT - 0.02,
        leg_half=(TABLE_HEIGHT - 0.04) / 2,
        leg_z=(TABLE_HEIGHT - 0.04) / 2,
        cube_z=TABLE_HEIGHT + CUBE_HALF,
        cube_half=CUBE_HALF,
    )
    start_marker = "    <!-- OPTIONAL_CUBE_START -->\n"
    end_marker = "    <!-- OPTIONAL_CUBE_END -->\n"
    if with_cube:
        xml = xml.replace(start_marker, "").replace(end_marker, "")
    else:
        start = xml.index(start_marker)
        end = xml.index(end_marker, start) + len(end_marker)
        xml = xml[:start] + xml[end:]
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
