import copy
import json

import numpy as np
import pytest

from configs.task import TARGET_STATES
from DataPipe.pipelines.curobo_bootstrap.task_targets import (
    PickPlaceV2Config,
    environment_position_to_base,
    pick_place_positions_b,
    solve_legacy_joint_targets,
    task_snapshot,
)

ORIGIN_W = (0.0, 0.0, 0.0)
BASE_POSE_W = (1.287, -0.034, 0.816, 0.7071067811865476, 0.0, 0.0, -0.7071067811865476)


def test_v2_config_rejects_bad_edges_and_values():
    for kwargs in (
        {"source_index": -1},
        {"destination_index": len(TARGET_STATES)},
        {"source_index": 1, "destination_index": 1},
        {"control_hz": 0.0},
        {"seed": -1},
        {"lift_clearance_m": float("nan")},
        {"scene_id": ""},
    ):
        with pytest.raises(ValueError):
            PickPlaceV2Config(**kwargs)


def test_environment_to_base_matches_known_production_transform():
    actual = environment_position_to_base(
        TARGET_STATES[0].position_e,
        environment_origin_w=ORIGIN_W,
        base_pose_w=BASE_POSE_W,
    )
    assert np.allclose(actual, (-0.237, 0.25, 0.124), atol=1e-12)


def test_waypoints_are_derived_only_from_target_states_and_clearance():
    positions = pick_place_positions_b(
        environment_origin_w=ORIGIN_W,
        base_pose_w=BASE_POSE_W,
        source_index=0,
        destination_index=1,
        lift_clearance_m=0.40,
    )
    assert tuple(positions) == ("pre_pick", "pick", "lift", "place_high", "place")
    assert np.allclose(positions["pre_pick"], (-0.117, 0.25, 0.141), atol=1e-12)
    assert np.allclose(positions["pick"], (-0.192, 0.25, 0.141), atol=1e-12)
    assert np.allclose(positions["place"], (-0.2093, -0.33033, 0.16889), atol=1e-8)
    assert positions["lift"][2] == pytest.approx(0.56889)
    assert positions["place_high"][2] == pytest.approx(0.56889)


def test_task_snapshot_is_repeatable_and_detects_supplied_geometry_drift():
    config = PickPlaceV2Config()
    first = task_snapshot(config=config, environment_origin_w=ORIGIN_W, base_pose_w=BASE_POSE_W)
    second = task_snapshot(config=config, environment_origin_w=ORIGIN_W, base_pose_w=BASE_POSE_W)
    assert first == second
    assert first["source"] == vars(TARGET_STATES[0])
    assert first["destination"] == vars(TARGET_STATES[1])
    assert len(first["task_hash"]) == 64

    broken = copy.deepcopy(first["waypoint_position_b"])
    broken["pick"][0] += 1e-6
    with pytest.raises(ValueError, match="TARGET_STATES"):
        task_snapshot(
            config=config,
            environment_origin_w=ORIGIN_W,
            base_pose_w=BASE_POSE_W,
            positions_b=broken,
        )


def test_real_legacy_ik_solves_all_five_targets_in_order():
    snapshot = task_snapshot(config=PickPlaceV2Config(), environment_origin_w=ORIGIN_W, base_pose_w=BASE_POSE_W)
    result = solve_legacy_joint_targets(
        snapshot,
        (0.0, -0.5235987901687622, 1.2217304706573486, 0.7853981852531433, 1.5707963705062866, 0.0),
    )
    assert result["passed"]
    assert tuple(result["targets"]) == ("pre_pick", "pick", "lift", "place_high", "place")
    assert all(item["passed"] for item in result["targets"].values())


def test_saved_3_to_0_recovers_without_changing_geometry_or_tolerances():
    snapshot = task_snapshot(
        config=PickPlaceV2Config(source_index=3, destination_index=0),
        environment_origin_w=ORIGIN_W,
        base_pose_w=(1.2869995832443237, -0.03400019183754921, 0.8159999847412109, *BASE_POSE_W[3:]),
    )
    original = copy.deepcopy(snapshot)
    result = solve_legacy_joint_targets(
        snapshot,
        (0.0, -0.5235987901687622, 1.2217304706573486, 0.7853981852531433, 1.5707963705062866, 0.0),
    )
    assert snapshot == original
    assert result["passed"]
    high = result["targets"]["place_high"]
    assert not high["attempts"][0]["passed"]
    assert high["attempts"][0]["reason"] == "maximum iterations reached"
    assert high["selected_attempt_index"] > 0
    assert high["fk_position_error_m"] <= 1e-5
    assert high["fk_orientation_error_rad"] <= 1e-4
    assert all(item["within_joint_limits"] for item in result["targets"].values())


def test_exhausted_seed_search_is_bounded_repeatable_and_stops_at_failed_target(monkeypatch):
    from DataPipe.runtime import ensure_curriculumrl_available

    ensure_curriculumrl_available()
    from DataPipe.native import aubo_ik

    calls = []

    def fail(position, quaternion, *, seed):
        calls.append(seed.copy())
        return aubo_ik.IKResult(False, None, "maximum iterations reached", 200, 1.0, 1.0)

    monkeypatch.setattr(aubo_ik, "solve_tcp_ik", fail)
    snapshot = task_snapshot(config=PickPlaceV2Config(), environment_origin_w=ORIGIN_W, base_pose_w=BASE_POSE_W)
    first = solve_legacy_joint_targets(snapshot, np.zeros(6), random_seed_count=4)
    second = solve_legacy_joint_targets(snapshot, np.zeros(6), random_seed_count=4)
    assert first == second
    assert not first["passed"]
    assert tuple(first["targets"]) == ("pre_pick",)
    target = first["targets"]["pre_pick"]
    assert target["selected_attempt_index"] is None
    assert len(target["attempts"]) == 5  # reset/previous/zero deduplicated, plus four random seeds
    assert len(calls) == 10
    assert all(np.array_equal(a, b) for a, b in zip(calls[:5], calls[5:]))


@pytest.mark.parametrize("q", [np.full(6, np.nan), np.zeros(3), np.full(6, 4.0), np.zeros(6)])
def test_solver_success_never_bypasses_solution_audit(monkeypatch, q):
    from DataPipe.runtime import ensure_curriculumrl_available

    ensure_curriculumrl_available()
    from DataPipe.native import aubo_ik

    monkeypatch.setattr(
        aubo_ik,
        "solve_tcp_ik",
        lambda *args, **kwargs: aubo_ik.IKResult(True, q, iterations=1, position_error_m=0, orientation_error_rad=0),
    )
    snapshot = task_snapshot(config=PickPlaceV2Config(), environment_origin_w=ORIGIN_W, base_pose_w=BASE_POSE_W)
    result = solve_legacy_joint_targets(snapshot, np.zeros(6), random_seed_count=0)
    assert not result["passed"]
    assert result["targets"]["pre_pick"]["selected_attempt_index"] is None
    assert "audit" in result["targets"]["pre_pick"]["reason"]
    json.dumps(result, allow_nan=False)
