"""Height policy, identity isolation and frozen experiment regression tests."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from DataPipe.pipelines.curobo_bootstrap.route_experiment import freeze_experiment, require_height, route_config
from DataPipe.pipelines.curobo_bootstrap.route_identity import require_report_identity, task_identity
from DataPipe.pipelines.curobo_bootstrap.task_targets import (
    PickPlaceV2Config,
    solve_legacy_joint_targets,
    task_snapshot,
)


def test_state4_height_policy_is_only_applied_to_its_eight_routes():
    configs = {(s, d): route_config(s, d, state4_lift_clearance_m=0.3) for s in range(5) for d in range(5) if s != d}
    assert sum(config.lift_clearance_m == 0.3 for config in configs.values()) == 8
    assert all(config.lift_clearance_m == (0.3 if 4 in pair else 0.4) for pair, config in configs.items())
    assert PickPlaceV2Config(source_index=0, destination_index=4).lift_clearance_m == 0.4


def test_height_changes_hash_and_rejects_cross_height_report_without_moving_endpoints():
    common = dict(environment_origin_w=(0, 0, 0), base_pose_w=(0, 0, 0.816, 1, 0, 0, 0))
    high = task_snapshot(config=route_config(0, 4), **common)
    low = task_snapshot(config=route_config(0, 4, state4_lift_clearance_m=0.3), **common)
    assert high["task_hash"] != low["task_hash"]
    for name in ("pre_pick", "pick", "place"):
        assert high["waypoint_position_b"][name] == low["waypoint_position_b"][name]
    for name in ("lift", "place_high"):
        np.testing.assert_allclose(
            np.array(high["waypoint_position_b"][name]) - low["waypoint_position_b"][name], [0, 0, 0.1], atol=1e-12
        )
    require_height(low, 0.3)
    with pytest.raises(ValueError, match="height differs"):
        require_height(low, 0.4)
    with pytest.raises(ValueError, match="task_hash differs"):
        require_report_identity({"passed": True, **task_identity(high)}, low, label="contact")
    wrong = deepcopy(low)
    wrong["waypoint_position_b"]["lift"][2] += 0.1
    with pytest.raises(ValueError, match="geometry"):
        require_height(wrong, 0.3)


@pytest.mark.parametrize(
    "field,value",
    [("route_lift_clearance_m", {"0_to_4": 0.4}), ("collision_model_sha256", "other"), ("mode", "headless")],
)
def test_resume_refuses_mixed_height_model_or_gui_mode(tmp_path, field, value):
    experiment = {"route_lift_clearance_m": {"0_to_4": 0.3}, "collision_model_sha256": "frozen", "mode": "gui"}
    freeze_experiment(tmp_path, experiment)
    assert freeze_experiment(tmp_path, experiment) == experiment
    wrong = dict(experiment, **{field: value})
    with pytest.raises(ValueError, match="different frozen experiment"):
        freeze_experiment(tmp_path, wrong)
    assert json.loads((tmp_path / "experiment_manifest.json").read_text()) == experiment


def test_unfrozen_existing_routes_cannot_be_silently_adopted(tmp_path):
    (tmp_path / "0_to_4_evidence").mkdir()
    with pytest.raises(ValueError, match="lacks a frozen experiment"):
        freeze_experiment(tmp_path, {"mode": "gui"})


@pytest.mark.parametrize("source,destination", [(0, 4), (4, 0), (1, 4), (4, 1), (2, 4), (4, 2), (3, 4), (4, 3)])
def test_saved_state4_geometry_all_five_targets_solve_at_030(source, destination):
    project = Path(__file__).resolve().parents[2]
    file = (
        project
        / "data/curobo_pick_place_graph_run003_20261003_r2/evidence"
        / f"{source}_to_{destination}_evidence/targets.json"
    )
    original = json.loads(file.read_text(encoding="utf-8"))
    measured = original["measured"]
    task = task_snapshot(
        config=route_config(source, destination, state4_lift_clearance_m=0.3),
        environment_origin_w=measured["environment_origin_w"],
        base_pose_w=measured["base_pose_w"],
    )
    result = solve_legacy_joint_targets(task, original["legacy_ik"]["q_start"])
    assert result["passed"] and len(result["targets"]) == 5
    assert task["task_hash"] != original["task_hash"]
    require_height(task, 0.3)
    for name in ("pre_pick", "pick", "place"):
        assert task["waypoint_position_b"][name] == original["task_snapshot"]["waypoint_position_b"][name]
