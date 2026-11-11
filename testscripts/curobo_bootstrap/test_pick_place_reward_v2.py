"""IK event accounting and compatibility of the explicitly versioned model."""

from dataclasses import replace

import pytest

from scripts.pick_place_reward_candidate import PickPlaceReward, RewardConfig, RewardObservation, RewardTask
from scripts.pick_place_reward_candidate_v2 import PickPlaceRewardV2, RewardConfigV2

TASK = RewardTask((0, 0, 0), (0, 0, 0), (1, 0, 0, 1, 0, 0, 0))
INITIAL = RewardObservation(0, (1, 0, 0), (0, 0, 0, 1, 0, 0, 0), (0,) * 6, (0, 0), "reach")


def test_ik_abort_charged_once_without_rewriting_previous_progress_or_time():
    model = PickPlaceRewardV2(TASK, INITIAL)
    obs = replace(INITIAL, time_s=1, tcp_position_w=(0, 0, 0))
    step = model.step(obs)
    event = model.abort_ik_failure("legacy_lift")
    assert step["terms"]["reach"] == 2
    assert event["reward"] == -15
    assert model.previous is obs
    assert model.done
    with pytest.raises(ValueError, match="already terminated"):
        model.abort_ik_failure("legacy_lift")
    with pytest.raises(ValueError, match="already terminated"):
        model.step(replace(obs, time_s=2))


def test_unchanged_transition_shaping_without_ik_event_even_at_high_speed():
    old, new = PickPlaceReward(TASK, INITIAL), PickPlaceRewardV2(TASK, INITIAL)
    for time in (0.1, 0.2, 0.3):
        obs = replace(INITIAL, time_s=time, joint_velocity=(100,) * 6)
        assert new.step(obs) == old.step(obs)


def test_existing_safety_failure_prevents_second_ik_terminal_cost():
    model = PickPlaceRewardV2(TASK, INITIAL)
    assert model.step(replace(INITIAL, time_s=1, dropped=True))["failure"] == "dropped"
    with pytest.raises(ValueError, match="already terminated"):
        model.abort_ik_failure("legacy_lift")


def test_ik_configuration_and_event_stage_validation():
    with pytest.raises(ValueError, match="finite and positive"):
        RewardConfigV2(ik_failure_cost=-1)
    with pytest.raises(ValueError, match="RewardConfigV2"):
        PickPlaceRewardV2(TASK, INITIAL, RewardConfig())
    model = PickPlaceRewardV2(TASK, INITIAL)
    with pytest.raises(ValueError, match="unsupported IK"):
        model.abort_ik_failure("planning_world_gate")
    assert model.done is False
