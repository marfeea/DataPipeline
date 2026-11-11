"""Behavioral checks for the isolated reward proposal; no Isaac runtime needed."""

from dataclasses import replace

import numpy as np
import pytest

from scripts.pick_place_reward_candidate import (
    PickPlaceReward,
    RewardConfig,
    RewardObservation,
    RewardTask,
    rotation_error,
)

TASK = RewardTask((0, 0, 0), (0, 0, 0), (1, 0, 0, 1, 0, 0, 0))
INITIAL = RewardObservation(0, (1, 0, 0), (0, 0, 0, 1, 0, 0, 0), (0,) * 6, (0, 0), "reach")


def observation(time, phase="reach", **kwargs):
    return replace(INITIAL, time_s=time, phase=phase, **kwargs)


def test_stationary_waiting_is_negative_and_rate_invariant():
    def total(hz):
        model = PickPlaceReward(TASK, INITIAL)
        return sum(model.step(observation(i / hz))["reward"] for i in range(1, hz + 1))

    assert total(4) == pytest.approx(-0.1)
    assert total(60) == pytest.approx(total(4))


def test_reach_oscillation_cannot_farm_frontier_credit():
    model = PickPlaceReward(TASK, INITIAL)
    first = model.step(observation(1, tcp_position_w=(0, 0, 0)))
    model.step(observation(2))
    repeat = model.step(observation(3, tcp_position_w=(0, 0, 0)))
    assert first["terms"]["reach"] == 2
    assert repeat["terms"]["reach"] == 0
    assert repeat["reward"] < 0


def test_horizontal_exit_is_not_vertical_lift():
    model = PickPlaceReward(replace(TASK, lift_target_m=0.015), INITIAL)
    result = model.step(observation(1, "lift", object_pose_w=(0, 0.18, 0, 1, 0, 0, 0)))
    assert result["terms"]["lift"] == 0
    assert result["vertical_lift_m"] == 0


def test_live_ungrasped_motion_does_not_receive_lift_or_place_credit():
    initial = replace(INITIAL, physical_dynamics=True, evidence_valid=True)
    model = PickPlaceReward(TASK, initial)
    raised = model.step(
        observation(1, "lift", object_pose_w=(0, 0, 0.05, 1, 0, 0, 0), physical_dynamics=True, evidence_valid=True)
    )
    placed = model.step(
        observation(2, "place", object_pose_w=TASK.goal_pose_w, physical_dynamics=True, evidence_valid=True)
    )
    assert raised["terms"]["lift"] == 0
    assert placed["terms"]["position"] == 0


def test_leaving_goal_takes_back_position_and_orientation_credit():
    model = PickPlaceReward(TASK, INITIAL)
    model.step(observation(1, "lift", object_pose_w=(0, 0, 0.05, 1, 0, 0, 0)))
    placed = model.step(observation(2, "place", object_pose_w=TASK.goal_pose_w))
    departed = model.step(observation(3, "place"))
    assert placed["terms"]["position"] == 4
    assert placed["terms"]["orientation"] == 2
    assert departed["terms"]["position"] == -4
    assert departed["terms"]["orientation"] == -2


def prepared_live_model():
    model = PickPlaceReward(TASK, replace(INITIAL, physical_dynamics=True, evidence_valid=True))
    model.step(observation(0.1, "close", grasp_verified=True, physical_dynamics=True, evidence_valid=True))
    model.step(
        observation(0.2, "lift", object_pose_w=(0, 0, 0.05, 1, 0, 0, 0), physical_dynamics=True, evidence_valid=True)
    )
    return model


def release(time, **kwargs):
    values = dict(
        object_pose_w=TASK.goal_pose_w,
        gripper_position=(0.0115, -0.0115),
        support_verified=True,
        physical_dynamics=True,
        evidence_valid=True,
        object_linear_speed=0.0,
        object_angular_speed=0.0,
    )
    values.update(kwargs)
    return observation(time, "release", **values)


def test_success_needs_post_release_settle_and_is_paid_once():
    model = prepared_live_model()
    assert model.step(release(0.3))["terms"]["success"] == 0
    assert model.step(release(0.5))["terms"]["success"] == 0
    final = model.step(release(0.55))
    assert final["terms"]["success"] == 10
    assert final["reward_valid"] is True
    with pytest.raises(ValueError, match="terminated"):
        model.step(release(0.6))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"support_verified": False},
        {"physical_dynamics": False},
        {"evidence_valid": False},
        {"object_linear_speed": None},
        {"object_angular_speed": None},
        {"gripper_position": (0.0115, -0.005)},
        {"object_pose_w": (1, 0, 0, np.cos(np.pi / 12), 0, 0, np.sin(np.pi / 12))},
    ],
)
def test_unsupported_release_does_not_get_success(kwargs):
    model = prepared_live_model()
    model.step(release(0.3, **kwargs))
    assert model.step(release(0.7, **kwargs))["terms"]["success"] == 0


def test_safety_overrides_success_and_timeout():
    model = prepared_live_model()
    model.step(release(0.3))
    result = model.step(release(0.6, illegal_contact=True, timeout=True))
    assert result["terms"]["success"] == 0
    assert result["failure"] == "illegal_contact"
    assert result["reward"] < -25


def test_success_overrides_simultaneous_timeout():
    model = prepared_live_model()
    model.step(release(0.3))
    result = model.step(release(0.6, timeout=True))
    assert result["terms"]["success"] == 10
    assert result["failure"] is None


def test_overlap_charged_once_and_offline_mask_false():
    model = PickPlaceReward(TASK, INITIAL)
    assert model.step(observation(1, geometry_overlap=True))["terms"]["overlap"] == -5
    second = model.step(observation(2, geometry_overlap=True))
    assert second["terms"]["overlap"] == 0
    assert second["reward_valid"] is False


def test_quaternion_sign_and_nonunit_scale_invariance():
    assert rotation_error((1, 0, 0, 0), (-2, 0, 0, 0)) == pytest.approx(0)
    assert rotation_error((1, 0, 0, 0), (0, 0, 0, 1)) == pytest.approx(np.pi)


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        observation(float("nan"))
    with pytest.raises(ValueError):
        RewardConfig(time_cost_per_s=-1)
    with pytest.raises(ValueError):
        rotation_error((0, 0, 0, 0), (1, 0, 0, 0))
    with pytest.raises(ValueError):
        PickPlaceReward(TASK, INITIAL).step(INITIAL)
