"""Checks for the showcase demos' building blocks. The last test renders, so it needs a working GL."""

import mujoco as mj
import numpy as np
import pytest

from so101_dual import READY_Q, TABLE_HEIGHT, arm_ids
from so101_dual.control import build_demo, grasp_rotation, solve_ik
from so101_dual.vision import COLOURS, StereoRig, add_stereo_cameras, find_blob, find_circle


def add_red_ball(spec: mj.MjSpec) -> None:
    body = spec.worldbody.add_body(name="ball", pos=[0.05, -0.05, TABLE_HEIGHT + 0.1])
    body.add_geom(type=mj.mjtGeom.mjGEOM_SPHERE, size=[0.025, 0, 0], rgba=[0.9, 0.1, 0.1, 1], contype=0, conaffinity=0)
    add_stereo_cameras(spec)


def test_ik_reaches_a_straight_down_grasp():
    model, _ = build_demo(lambda spec: None)
    ids = arm_ids(model, "left")
    target = np.array([-0.05, 0.03, TABLE_HEIGHT + 0.03])
    rotation = grasp_rotation([0, 0, -1], [0, 1, 0])
    _, pos_err, rot_err = solve_ik(model, mj.MjData(model), ids, target, rotation, READY_Q, iters=100)
    assert pos_err < 1e-3
    assert rot_err < 0.05


def test_circle_fit_survives_a_blocked_view():
    image = np.zeros((200, 200, 3), np.uint8)
    vs, us = np.indices((200, 200))
    image[np.hypot(us - 100, vs - 90) < 40] = (230, 25, 25)
    image[:, 115:] = 0  # something hides the ball's right side
    centre, _ = find_circle(image, "red")
    assert np.hypot(centre[0] - 100, centre[1] - 90) < 1.5
    blob, _ = find_blob(image, "red")
    assert blob[0] < 95  # the plain blob centre gets pulled well off to one side


def test_colour_rules_ignore_the_arm_and_table():
    for rgb in ((255, 209, 31), (184, 153, 115)):  # arm yellow, table brown
        r, g, b = (np.array(float(v)) for v in rgb)
        for rule in COLOURS.values():
            assert not rule(r, g, b)


def test_stereo_locates_a_ball_from_pixels_alone():
    model, data = build_demo(add_red_ball)
    try:
        rig = StereoRig(model)
    except Exception as err:
        pytest.skip(f"this machine can't render: {err}")
    det = rig.locate(data, "red", ball=True)
    assert det is not None
    assert np.linalg.norm(det.position - model.body_pos[model.body("ball").id]) < 0.005
