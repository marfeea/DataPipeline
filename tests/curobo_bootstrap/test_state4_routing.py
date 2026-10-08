import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from DataPipe.pipelines.curobo_bootstrap.route_experiment import freeze_experiment, require_height, route_config
from DataPipe.pipelines.curobo_bootstrap.task_targets import (
    PickPlaceV2Config,
    solve_legacy_joint_targets,
    task_snapshot,
)
from DataPipe.sequence import DatasetSequence, SequenceConfig, SequenceObservation, SequenceStage, SequenceState
from DataPipe.state4_routing import State4RoutingConfig, routing_payload


def test_route_policy_only_changes_state4_and_resume_refuses_different_standoff(tmp_path):
    policy = State4RoutingConfig()
    assert route_config(4, 0, state4_routing=policy).state4_routing == policy
    assert route_config(3, 0, state4_routing=policy).state4_routing is None
    manifest = dict(state4_routing=vars(policy), mode="gui")
    freeze_experiment(tmp_path, manifest)
    changed = dict(manifest, state4_routing=vars(State4RoutingConfig(place_standoff_m=0.22)))
    with pytest.raises(ValueError, match="different frozen experiment"):
        freeze_experiment(tmp_path, changed)


def sequence(source, destination, stage, policy=None):
    return DatasetSequence(
        preposition_b=(0.1, 0.42, 0.2),
        grasp_tcp_position_b=(0.1, 0.3, 0.2),
        lift_position_b=(0.1, 0.3, 0.8),
        place_high_position_b=(0.2, 0.4, 0.8),
        place_position_b=(0.2, 0.4, 0.1),
        place_preposition_b=(0.2, 0.52, 0.1),
        place_tcp_direction_w=(1, 0, 0),
        tcp_heading_axis=(1, 0, 0),
        config=SequenceConfig(normal_dwell_steps=2),
        source_state_index=source,
        place_state_index=destination,
        state4_routing=policy or State4RoutingConfig(),
        initial_state=SequenceState(stage=stage),
    )


def step(controller, position):
    return controller.command(
        SequenceObservation(position, (1, 0, 0, 0), (0, 0)),
        active_waypoint_b=(0, 0, 0),
        active_waypoint_label="target_settle",
        path_complete=True,
        closed_gripper_position=(0, 0),
        open_gripper_position=(0.01, -0.01),
    )


def test_state4_lift_holds_low_height_before_horizontal_withdrawal_and_preserves_state0_detour():
    controller = sequence(4, 0, SequenceStage.LIFT_OBJECT)
    low, exit_point = controller.lift_waypoints
    assert low == pytest.approx((0.1, 0.3, 0.215))
    assert exit_point == pytest.approx((0.1, 0.48, 0.215))
    assert step(controller, (0.1, 0.3, 0.2)).target_position_b == pytest.approx(low)
    step(controller, low)
    assert step(controller, low).target_position_b == pytest.approx(exit_point)
    assert controller.state.stage == SequenceStage.LIFT_OBJECT
    assert controller.place_high_waypoints[0][0] == pytest.approx(-0.08)
    controller.reset()
    assert controller.state.lift_waypoint_index == 0


def test_state4_descent_stays_outside_roof_before_constant_height_insertion():
    controller = sequence(0, 4, SequenceStage.PLACE_FINAL_Z)
    original = (0.2, 0.4, 0.1)
    descent = (0.2, 0.58, 0.1)
    assert controller.place_high_position_b == pytest.approx((0.2, 0.58, 0.8))
    assert step(controller, original).stage == SequenceStage.PLACE_FINAL_Z
    assert step(controller, original).target_position_b == pytest.approx(descent)
    step(controller, descent)
    command = step(controller, descent)
    assert command.stage == SequenceStage.PLACE_FINAL_XY
    assert command.target_position_b == pytest.approx(original)
    assert command.arm_action[2] == 0
    assert not command.complete


def test_raised_entry_is_reached_before_original_final_pose_and_release():
    policy = State4RoutingConfig(place_entry_lift_m=0.008)
    controller = sequence(0, 4, SequenceStage.PLACE_FINAL_Z, policy)
    descent = (0.2, 0.58, 0.108)
    entry = (0.2, 0.4, 0.108)
    original = (0.2, 0.4, 0.1)
    assert controller.place_descent_position_b == pytest.approx(descent)
    np.testing.assert_allclose(controller.place_final_waypoints, (entry, original), atol=1e-12)
    step(controller, descent)
    command = step(controller, descent)
    assert command.target_position_b == pytest.approx(entry) and command.arm_action[2] == pytest.approx(0, abs=1e-12)
    step(controller, entry)
    command = step(controller, entry)
    assert controller.state.place_final_waypoint_index == 1
    assert command.target_position_b == pytest.approx(original) and command.gripper_command == -1
    assert controller.state.dwell_steps == 0
    assert step(controller, (0.2, 0.4, 0.13)).stage == SequenceStage.PLACE_FINAL_XY
    step(controller, original)
    assert step(controller, original).stage == SequenceStage.RELEASE_GRIPPER
    controller.reset()
    assert controller.state.place_final_waypoint_index == 0


@pytest.mark.parametrize("lift", (-0.001, float("inf"), float("nan")))
def test_entry_lift_rejects_invalid_values(lift):
    with pytest.raises(ValueError, match="place_entry_lift_m"):
        State4RoutingConfig(place_entry_lift_m=lift)


def test_zero_entry_lift_preserves_old_policy_payload():
    assert routing_payload(State4RoutingConfig()) == dict(
        pick_lift_m=0.015, pick_exit_m=0.18, place_standoff_m=0.18
    )


def test_entry_lift_changes_identity_and_checks_extra_geometry_and_ik():
    project = Path(__file__).resolve().parents[2]
    saved = json.loads(
        (project / "data/curobo_pick_place_graph_run003_20261003_r2/evidence/0_to_4_evidence/targets.json").read_text()
    )
    common = dict(
        environment_origin_w=saved["measured"]["environment_origin_w"], base_pose_w=saved["measured"]["base_pose_w"]
    )
    original = task_snapshot(
        config=PickPlaceV2Config(source_index=0, destination_index=4, lift_clearance_m=0.3,
                                state4_routing=State4RoutingConfig()), **common
    )
    raised = task_snapshot(
        config=PickPlaceV2Config(source_index=0, destination_index=4, lift_clearance_m=0.3,
                                state4_routing=State4RoutingConfig(place_entry_lift_m=0.008)), **common
    )
    assert raised["task_hash"] != original["task_hash"]
    assert raised["waypoint_position_b"] == original["waypoint_position_b"]
    for name in ("place_descent", "place_entry"):
        single = dict(raised, waypoint_position_b={"place": raised["contact_waypoint_position_b"][name]})
        assert solve_legacy_joint_targets(single, saved["legacy_ik"]["q_start"])["passed"]
    require_height(raised, 0.3)
    wrong = deepcopy(raised)
    wrong["contact_waypoint_position_b"]["place_entry"][2] += 0.001
    with pytest.raises(ValueError, match="routing geometry"):
        require_height(wrong, 0.3)


@pytest.mark.parametrize("source,destination", [(4, 0), (0, 4)])
def test_real_state4_routing_geometry_hash_and_five_targets(source, destination):
    project = Path(__file__).resolve().parents[2]
    saved = json.loads(
        (
            project
            / "data/curobo_pick_place_graph_run003_20261003_r2/evidence"
            / f"{source}_to_{destination}_evidence/targets.json"
        ).read_text()
    )
    common = saved["measured"]
    original = task_snapshot(
        config=PickPlaceV2Config(source_index=source, destination_index=destination, lift_clearance_m=0.3),
        environment_origin_w=common["environment_origin_w"],
        base_pose_w=common["base_pose_w"],
    )
    routed = task_snapshot(
        config=PickPlaceV2Config(
            source_index=source,
            destination_index=destination,
            lift_clearance_m=0.3,
            state4_routing=State4RoutingConfig(),
        ),
        environment_origin_w=common["environment_origin_w"],
        base_pose_w=common["base_pose_w"],
    )
    assert original["task_hash"] != routed["task_hash"]
    for name in ("pre_pick", "pick", "place"):
        assert original["waypoint_position_b"][name] == routed["waypoint_position_b"][name]
    require_height(routed, 0.3)
    assert solve_legacy_joint_targets(routed, saved["legacy_ik"]["q_start"])["passed"]
    wrong = deepcopy(routed)
    extra = next(iter(wrong["contact_waypoint_position_b"]))
    wrong["contact_waypoint_position_b"][extra][2] += 0.01
    with pytest.raises(ValueError, match="routing geometry"):
        require_height(wrong, 0.3)
    if source == 4:
        np.testing.assert_allclose(
            routed["waypoint_position_b"]["lift"][2], routed["waypoint_position_b"]["pick"][2] + 0.015
        )
