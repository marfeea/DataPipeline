import numpy as np
import pytest

from DataPipe.pipelines.curobo_bootstrap.hybrid_route import (
    HYBRID_ROUTE,
    RouteSource,
    curobo_goal_specs,
    validate_hybrid_route,
)


def test_only_noncontact_motion_uses_curobo():
    assert validate_hybrid_route(HYBRID_ROUTE) == HYBRID_ROUTE
    assert [(step.start, step.goal) for step in HYBRID_ROUTE if step.source is RouteSource.CUROBO] == [
        ("q_start", "pre_pick"),
        ("lift", "place_high"),
    ]
    assert [(step.start, step.goal) for step in HYBRID_ROUTE if step.source is RouteSource.LEGACY_IK] == [
        ("pre_pick", "pick"),
        ("pick", "lift"),
        ("place_high", "place"),
    ]
    assert [
        (step.name, step.start, step.goal) for step in HYBRID_ROUTE if step.source is RouteSource.GRIPPER_EVENT
    ] == [
        ("close_gripper", "pick", "pick"),
        ("open_gripper", "place", "place"),
    ]


def test_curobo_goals_keep_measured_start_contract_and_omit_contact_goals():
    targets = {
        name: np.full(6, index, dtype=np.float64)
        for index, name in enumerate(("q_start", "pre_pick", "pick", "lift", "place_high", "place"))
    }
    queries = curobo_goal_specs(targets)
    assert [name for name, _, _ in queries] == ["free_space_to_pre_pick", "free_space_transfer"]
    assert [(start_label, goal[0]) for _, start_label, goal in queries] == [("q_start", 1), ("lift", 4)]
    queries[0][2][0] = 100
    assert targets["pre_pick"][0] == 1


def test_curobo_goals_reject_missing_or_nonfinite_targets():
    targets = {name: np.zeros(6) for name in ("q_start", "pre_pick", "pick", "lift", "place_high", "place")}
    with pytest.raises(ValueError, match="fields differ"):
        curobo_goal_specs({key: value for key, value in targets.items() if key != "pick"})
    targets["pick"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        curobo_goal_specs(targets)


def test_hybrid_route_rejects_reassignment_of_contact_motion():
    with pytest.raises(ValueError, match="differs"):
        validate_hybrid_route(HYBRID_ROUTE[:1] + HYBRID_ROUTE[2:])
