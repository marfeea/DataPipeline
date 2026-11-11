"""Temporal counterexamples for physical grasp/drop evidence and rewards."""

import numpy as np
import pytest

from scripts.pick_place_reward_candidate import RewardTask
from scripts.pipelines.curobo_bootstrap.physical_failure import (
    GraspEvidence,
    make_reward_model,
    reward_observation,
    write_artifacts,
)
from scripts.pipelines.curobo_bootstrap.pick_place_artifact import FIELDS, load_hybrid_trajectory


def sample(time, *, z=0.03, tcp_z=0.03, force=(1, 1, 0), vz=0, available=True, release=False):
    return dict(
        time_s=time,
        tcp_pose_w=[0, 0, tcp_z, 1, 0, 0, 0],
        target_pose_w=[0, 0, z, 1, 0, 0, 0],
        target_velocity_w=[0, 0, vz, 0, 0, 0],
        contact_force_n=list(force),
        contact_available=available,
        normal_supported_release=release,
        q=[0] * 6,
        dq=[0] * 6,
        gripper_position=[0, 0],
        gripper_velocity=[0, 0],
        phase="lift",
        stage="legacy_lift",
        stage_id="MOVE_TO_PLACE",
        event_code="NONE",
        commanded_joint_target=[0] * 6,
        executed_gripper_target=[0, 0],
    )


def verified():
    first = sample(0, z=0, tcp_z=0, force=(0, 0, 1))
    detector = GraspEvidence(first)
    detector.step(first)
    detector.step(sample(0.1))
    result = detector.step(sample(0.36))
    assert result["grasp_verified"] and result["evidence_valid"]
    return detector


def test_never_grasped_falling_is_not_dropped():
    first = sample(0, force=(0, 0, 1), z=0)
    detector = GraspEvidence(first)
    detector.step(first)
    result = detector.step(sample(1, force=(0, 0, 0), z=-0.2, vz=-1))
    assert not result["dropped"] and not result["grasp_verified"]
    assert not result["evidence_valid"]


def test_grasp_requires_dwell_both_fingers_clearance_and_stable_transform():
    first = sample(0, z=0)
    detector = GraspEvidence(first)
    for t, force in ((0.1, (1, 0, 0)), (0.4, (1, 1, 1)), (0.7, (1, 1, 0))):
        assert not detector.step(sample(t, force=force))["grasp_verified"]
    assert not detector.step(sample(0.99, z=0.04))["grasp_verified"]  # relative drift restarts dwell
    assert detector.step(sample(1.25, z=0.04))["grasp_verified"]


def test_confirmed_drop_and_reward_charged_once():
    detector = verified()
    held = sample(0.37)
    model = make_reward_model(
        RewardTask((0, 0, 0), (0, 0, 0), (1, 0, 0, 1, 0, 0, 0)), reward_observation(held, detector.step(held))
    )
    lost = sample(0.4, z=0.018, vz=-0.1, force=(0, 0, 0))
    assert not detector.step(lost)["dropped"]
    fallen = sample(0.46, z=0.01, vz=-0.1, force=(0, 0, 0))
    result = model.step(reward_observation(fallen, detector.step(fallen)))
    assert result["failure"] == "dropped" and result["terms"]["failure"] == -10
    assert result["reward_valid"] and result["terminated"]
    from scripts.pipelines.curobo_bootstrap.physical_reward_validity import gate_reward_validity

    score = result["reward"]
    result["transition_valid"] = True
    gate_reward_validity([result], {"contact_positive_controls": [True, True, True]})
    assert result["candidate_reward_valid"] and not result["reward_valid"]
    assert result["reward"] == score  # observed drop remains scored; missing arm contacts mask full reward
    with pytest.raises(ValueError, match="terminated"):
        model.step(reward_observation(sample(0.5), detector.step(sample(0.5))))


@pytest.mark.parametrize("mode", ["missing", "supported_release", "active_descent", "one_frame_noise"])
def test_counterexamples_do_not_drop(mode):
    detector = verified()
    for t in (0.4, 0.46):
        s = sample(t, z=0.01, vz=-0.1, force=(0, 0, 0))
        if mode == "missing":
            s["contact_available"] = False
        elif mode == "supported_release":
            s["normal_supported_release"] = True
        elif mode == "active_descent":
            s["tcp_pose_w"][2] = 0.01
        elif t == 0.46:
            s = sample(t)
        assert not detector.step(s)["dropped"]


def test_npz_contract_actions_and_terminal_tail(tmp_path):
    detector = verified()
    records = [sample(0.4), sample(0.5), sample(0.6)]
    for r in records:
        r["evidence"] = detector.step(r)
    records[1]["executed_gripper_target"] = [0.0115, -0.0115]
    rows = [
        dict(reward=-10, reward_valid=True, terminated=True, transition_valid=True),
        dict(reward=0, reward_valid=False, terminated=False, transition_valid=False),
    ]
    report = dict(joint_names=["j"] * 6, reference_snapshot="reference.json")
    task = RewardTask((0, 0, 0), (0, 0, 0), (1, 0, 0, 1, 0, 0, 0))
    write_artifacts(tmp_path, records, rows, {}, task, report)
    arrays = load_hybrid_trajectory(tmp_path / "trajectory.npz")
    assert set(arrays) == FIELDS and not arrays["object_attached"].any()
    with np.load(tmp_path / "rewards.npz", allow_pickle=False) as data:
        assert len(data["reward"]) == len(arrays["time_s"]) - 1
        np.testing.assert_equal(data["action_gripper_target"][0], [0.0115, -0.0115])
        np.testing.assert_equal(data["transition_valid"], [True, False])
    assert report["reward_version"] == "pick-place-candidate/2"


def test_supported_release_remains_normal_during_settling_contact_noise():
    detector = verified()
    settling = sample(0.4, z=0.018, vz=-0.1, force=(0, 0, 0.1), release=True)
    assert not detector.step(settling)["dropped"]
    flicker = sample(0.46, z=0.01, vz=-0.1, force=(0, 0, 0), release=False)
    assert not detector.step(flicker)["dropped"]
