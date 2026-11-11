from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from configs.task import TARGET_STATES
from scripts.pipelines.curobo_bootstrap.contracts import config_hash
from scripts.pipelines.curobo_bootstrap.replay import matrix_pose_wxyz
from scripts.pipelines.curobo_bootstrap.robot_adapter import pose_matrix
from scripts.pipelines.curobo_bootstrap.route_identity import task_identity
from scripts.pipelines.curobo_bootstrap.stable_placement_execution import (
    execute_stable_placement,
    placement_geometry,
    placement_profile,
)
from scripts.pipelines.curobo_bootstrap.task_targets import PickPlaceV2Config, task_snapshot

EXPECTED = (
    (1.539, 0.203, 0.9365), (0.91167, 0.1753, 0.97089), (0.91167, 0.03036, 0.96976),
    (0.91435, -0.18557, 0.99391), (0.90264, -0.50461, 1.0945),
)


def fixture(index):
    task = task_snapshot(
        config=PickPlaceV2Config(source_index=1 if index == 0 else 0, destination_index=index),
        environment_origin_w=(0.2, -0.1, 0.0),
        base_pose_w=(1.287, -0.034, 0.816, 0.7071067811865476, 0, 0, -0.7071067811865476),
    )
    grip = pose_matrix((-0.003, 0.049, -0.017), (np.cos(0.03), np.sin(0.03), 0, 0))
    tcp = pose_matrix((1.1, 0.1, 1.5), (1, 0, 0, 0))
    initial = dict(
        time_s=0.0, q=np.zeros(6).tolist(), tcp_pose_w=list(matrix_pose_wxyz(tcp)),
        target_pose_w=list(matrix_pose_wxyz(tcp @ grip)), evidence=dict(grasp_verified=True),
    )
    return task, initial, grip


@pytest.mark.parametrize("index", range(5))
def test_object_release_position_is_converted_through_measured_grasp_and_environment(index):
    task, initial, grip = fixture(index)
    before = deepcopy(task)
    profile = placement_profile(task)
    measured, desired, points = placement_geometry(task, profile, initial)
    np.testing.assert_allclose(profile["pre_place_tcp_position_e"], EXPECTED[index], atol=1e-12)
    np.testing.assert_allclose(measured, grip, atol=1e-12)
    np.testing.assert_allclose(points["place"] @ grip, desired, atol=1e-12)
    np.testing.assert_allclose(points["place"][:3, 3], np.asarray(EXPECTED[index]) + (0.2, -0.1, 0), atol=1e-12)
    assert task == before
    assert task["destination"]["position_e"] == asdict(TARGET_STATES[index])["position_e"]
    if index == 4:
        assert points["place_entry"][2, 3] - points["place"][2, 3] == pytest.approx(0.005)
        np.testing.assert_allclose(points["place_entry"][:2, 3], points["place"][:2, 3], atol=1e-12)
        assert np.linalg.norm(points["place_descent"][:2, 3] - points["place_entry"][:2, 3]) == pytest.approx(0.24)


def test_historical_reference_keeps_identity_and_uses_current_placement_recipe():
    task, _, _ = fixture(2)
    for role in ("source", "destination"):
        event_fields = {"pre_grasp_tcp_position_e", "pre_place_tcp_position_e"}
        task[role] = {k: v for k, v in task[role].items()
                      if not k.startswith("place_") and k not in event_fields}
    task["task_hash"] = config_hash({k: v for k, v in task.items() if k != "task_hash"})
    task_identity(task)
    np.testing.assert_equal(placement_profile(task)["pre_place_tcp_position_e"], EXPECTED[2])
    task["destination"]["position_e"] = (9, 9, 9)
    task["task_hash"] = config_hash({k: v for k, v in task.items() if k != "task_hash"})
    with pytest.raises(ValueError, match="destination"):
        task_identity(task)


@pytest.mark.parametrize("index", [0, 2, 4])
def test_controller_orders_descent_open_and_lateral_withdrawal_without_state_writes(index):
    task, initial, grip = fixture(index)
    profile = placement_profile(task)
    goal = np.asarray(task["frames"]["T_WE"]) @ pose_matrix(
        task["destination"]["position_e"], task["destination"]["rotation_wxyz"]
    )
    records = []

    def step(q, fingers, stage, phase, requested):
        tcp = pose_matrix(requested[:3], requested[3:])
        released = min(abs(fingers)) > 0.001
        sample = dict(
            time_s=(len(records) + 1) / 60, q=q.tolist(), tcp_pose_w=requested,
            target_pose_w=list(matrix_pose_wxyz(goal if released else tcp @ grip)),
            target_velocity_w=np.zeros(6).tolist(), contact_available=True,
            contact_force_n=[0, 0, 0.1] if released else [0.1, 0.1, 0],
            joint_limit=False, gripper_position=fingers.tolist(), evidence=dict(grasp_verified=not released),
            stage=stage, phase=phase,
        )
        records.append(sample)
        return sample

    report = execute_stable_placement(
        task, profile, initial, step, lambda: False,
        solver=lambda *args, **kwargs: SimpleNamespace(success=True, q=np.zeros(6)),
    )
    assert report["passed"], report
    stages = [r["stage"] for r in records]
    assert stages.index("stable_place") < stages.index("stable_open_gripper") < stages.index("stable_withdraw")
    descent = np.array([r["tcp_pose_w"] for r in records if r["stage"] == "stable_place"])
    assert np.ptp(descent[:, :2], axis=0).max() < 1e-12
    assert np.ptp(descent[:, 3:], axis=0).max() < 1e-12
    opened = [r["gripper_position"][0] for r in records if r["stage"] == "stable_open_gripper"]
    assert len(opened) == 18 and np.all(np.diff(opened) > 0)
    assert opened[-1] == pytest.approx(0.0115)
    if index == 4:
        assert stages.index("stable_place_descent") < stages.index("stable_place_entry") < stages.index("stable_place")


def test_unverified_grasp_never_starts_placement_or_opens_gripper():
    task, initial, _ = fixture(2)
    initial["evidence"]["grasp_verified"] = False
    report = execute_stable_placement(task, placement_profile(task), initial,
                                     lambda *args: pytest.fail("physical step must not execute"), lambda: False)
    assert not report["passed"] and "verified" in report["failure_reason"]


@pytest.mark.parametrize("fault", ["unverified_entry", "grasp_loss", "premature_support", "missed_tcp"])
def test_collection_finishes_descent_release_and_withdraw_but_preserves_failed_checks(fault):
    task, initial, grip = fixture(2)
    initial["evidence"]["grasp_verified"] = fault != "unverified_entry"
    goal = np.asarray(task["frames"]["T_WE"]) @ pose_matrix(
        task["destination"]["position_e"], task["destination"]["rotation_wxyz"]
    )
    rows = []

    def step(q, fingers, stage, phase, requested):
        tcp = pose_matrix(requested[:3], requested[3:])
        if fault == "missed_tcp":
            tcp[0, 3] += 0.004
        released = phase == "release"
        forces = [0, 0, 0.1] if released else [0.1, 0.1, 0]
        if fault == "grasp_loss" and not released:
            forces = [0, 0, 0]
        elif fault == "premature_support" and stage == "stable_place_high_hold":
            forces = [0.1, 0.1, 0.1]
        row = dict(time_s=(len(rows) + 1) / 60, q=q.tolist(), tcp_pose_w=list(matrix_pose_wxyz(tcp)),
                   target_pose_w=list(matrix_pose_wxyz(goal if released else tcp @ grip)),
                   target_velocity_w=np.zeros(6).tolist(), contact_available=True, contact_force_n=forces,
                   joint_limit=False, gripper_position=fingers.tolist(), stage=stage,
                   evidence=dict(grasp_verified=not released and fault not in {"unverified_entry", "grasp_loss"}))
        rows.append(row)
        return row

    report = execute_stable_placement(
        task, placement_profile(task), initial, step, lambda: False, continue_on_failure=True,
        solver=lambda *args, **kwargs: SimpleNamespace(success=True, q=np.zeros(6)),
    )
    assert report["execution_completed"] and not report["passed"]
    assert report["validation_failures"] and report["failure_reason"]
    assert report["stable_after_withdraw_s"] >= 0.5  # Late stability cannot erase the failed earlier check.
    stages = [r["stage"] for r in rows]
    assert stages.index("stable_place") < stages.index("stable_open_gripper") < stages.index("stable_withdraw")
    assert stages[-1] == "stable_after_withdraw"


def test_direct_release_tcp_is_independent_of_measured_grasp_offset():
    task, initial, _ = fixture(1)
    profile = placement_profile(task)
    first = placement_geometry(task, profile, initial)[2]["place"]
    changed = deepcopy(initial)
    changed["target_pose_w"][0] += 0.02
    changed["target_pose_w"][2] -= 0.01
    second = placement_geometry(task, profile, changed)[2]["place"]
    np.testing.assert_allclose(first, second, atol=1e-12, rtol=0)
