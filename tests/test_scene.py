import mujoco as mj
import numpy as np
import pytest

from so101_dual import ARMS, JOINTS, READY_Q, TABLE_HEIGHT, arm_ids, load
from so101_dual.scene import SO101_DIR


@pytest.fixture()
def sim():
    return load()


def test_scene_has_two_six_dof_arms(sim):
    model, _ = sim
    assert model.nu == 12
    for arm in ARMS:
        ids = arm_ids(model, arm)
        assert len(ids.actuator) == len(JOINTS) == 6
        assert ids.gripper_site >= 0


def test_scene_can_be_built_without_cube():
    model, _ = load(with_cube=False)
    assert mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "cube") == -1


def test_unknown_arm_rejected(sim):
    model, _ = sim
    with pytest.raises(ValueError):
        arm_ids(model, "middle")


def test_ready_pose_is_collision_free(sim):
    _, data = sim
    assert data.ncon == 0


def test_arms_face_each_other_with_clearance(sim):
    model, data = sim
    left = data.site_xpos[arm_ids(model, "left").gripper_site]
    right = data.site_xpos[arm_ids(model, "right").gripper_site]
    assert left[0] < right[0]
    assert right[0] - left[0] > 0.10  # room for an object between the grippers
    assert left[2] > TABLE_HEIGHT and right[2] > TABLE_HEIGHT


def test_position_actuators_track_targets(sim):
    model, data = sim
    left = arm_ids(model, "left")
    target = READY_Q.copy()
    target[0] = 0.5  # shoulder_pan
    data.ctrl[left.actuator] = target
    for _ in range(int(2.0 / model.opt.timestep)):
        mj.mj_step(model, data)
    assert abs(data.qpos[left.qpos[0]] - 0.5) < 0.03


def test_stable_and_cube_rests_on_table(sim):
    model, data = sim
    for _ in range(int(5.0 / model.opt.timestep)):
        mj.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    cube_z = data.xpos[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "cube")][2]
    assert cube_z == pytest.approx(TABLE_HEIGHT + 0.015, abs=2e-3)


def test_upstream_license_ships_with_model():
    assert (SO101_DIR / "LICENSE").exists()
