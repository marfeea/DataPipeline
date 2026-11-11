import numpy as np
import pytest

from scripts.pipelines.curobo_bootstrap.placement_search import (
    PlacementCandidate,
    carrying_metrics,
    continuous_stable_s,
    placement_metrics,
    tcp_release_pose,
)
from scripts.pipelines.curobo_bootstrap.robot_adapter import pose_matrix


def sample():
    return dict(
        object_pose_w=[0, 0, 1, 1, 0, 0, 0],
        object_velocity_w=[0] * 6,
        finger_position=[0.0115, -0.0115],
        contact_force_n=[0, 0, 0.0981],
        contact_available=True,
    )


def test_measured_grasp_corrects_tcp_without_changing_object_goal():
    goal = [0.9, 0.03, 0.967, 0.70710678, 0, 0, -0.70710678]
    relative = pose_matrix([-0.05, 0.0005, -0.017], [1, 0, 0, 0])
    tcp, desired = tcp_release_pose(goal, relative, PlacementCandidate())
    np.testing.assert_allclose(tcp @ relative, desired, atol=1e-12)
    np.testing.assert_allclose(desired[:3, 3], goal[:3])
    assert np.linalg.norm(tcp[:3, 3] - desired[:3, 3]) > 0.05


@pytest.mark.parametrize(
    "field,value",
    [
        ("object_velocity_w", [0, 0, 0, 0.3, 0, 0]),
        ("object_pose_w", [0.03, 0, 1, 1, 0, 0, 0]),
        ("finger_position", [0, 0]),
        ("contact_force_n", [0, 0, 0]),
        ("contact_force_n", [0.1, 0.1, 0.0981]),
        ("contact_available", False),
        ("illegal_contact", True),
    ],
)
def test_gate_rejects_unstable_or_still_held_object(field, value):
    row = sample()
    row[field] = value
    assert not placement_metrics(row, [0, 0, 1, 1, 0, 0, 0], grasp_verified=True)["stable"]


def test_gate_requires_verified_grasp_and_passes_detached_support():
    assert not placement_metrics(sample(), [0, 0, 1, 1, 0, 0, 0], grasp_verified=False)["stable"]
    assert placement_metrics(sample(), [0, 0, 1, 1, 0, 0, 0], grasp_verified=True)["stable"]


def test_continuous_window_resets_on_one_failure():
    rows = [
        dict(time_s=t, metrics=dict(stable=s))
        for t, s in [(0, True), (0.3, True), (0.4, False), (0.5, True), (0.6, True)]
    ]
    assert continuous_stable_s(rows) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        continuous_stable_s(rows + [rows[-1]])


def test_carry_gate_rejects_roof_contact_above_goal():
    row = sample()
    row["tcp_pose_w"] = [0, 0, 1, 1, 0, 0, 0]
    result = carrying_metrics(row, [0, 0, 0.7, 1, 0, 0, 0], np.eye(4))
    assert result["premature_support"]


def test_carry_gate_accepts_goal_support_but_rejects_midair_loss():
    row = sample()
    row["tcp_pose_w"] = [0, 0, 1, 1, 0, 0, 0]
    goal = [0, 0, 1, 1, 0, 0, 0]
    assert not carrying_metrics(row, goal, np.eye(4))["premature_support"]
    assert not carrying_metrics(row, goal, np.eye(4))["unsupported_loss"]
    row["contact_force_n"] = [0, 0, 0]
    assert carrying_metrics(row, goal, np.eye(4))["unsupported_loss"]
