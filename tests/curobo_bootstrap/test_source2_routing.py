import argparse
import math

import numpy as np
import pytest

from DataPipe.pipelines.curobo_bootstrap.route_experiment import require_height
from DataPipe.pipelines.curobo_bootstrap.task_targets import PickPlaceV2Config, task_snapshot
from DataPipe.sequence import DatasetSequence, SequenceConfig, SequenceObservation, SequenceStage, SequenceState
from DataPipe.source2_routing import (
    Source2RoutingConfig,
    add_source2_routing_args,
    source2_advance_waypoints,
    source2_pre_pick,
    source2_routing_arguments,
    source2_routing_from_args,
    source2_routing_payload,
)


def test_side_offset_reproduces_collision_audited_source2_point():
    pre = (-0.0643601918, -0.2553295832, 0.1677600153)
    pick = (-0.0643601918, -0.3303295832, 0.1677600153)
    routed = source2_pre_pick(source_index=2, pre_pick=pre, pick=pick, config=Source2RoutingConfig())
    assert routed == pytest.approx((-0.1243601918, -0.2553295832, 0.1677600153))
    assert pre == (-0.0643601918, -0.2553295832, 0.1677600153)
    assert pick == (-0.0643601918, -0.3303295832, 0.1677600153)


def test_horizontal_rotation_preserves_side_semantics():
    angle = 0.7
    rotation = np.array(((math.cos(angle), -math.sin(angle), 0), (math.sin(angle), math.cos(angle), 0), (0, 0, 1)))
    pre, pick = np.array((0.1, 0.2, 0.3)), np.array((0.1, 0.125, 0.3))
    config = Source2RoutingConfig()
    expected = rotation @ source2_pre_pick(source_index=2, pre_pick=pre, pick=pick, config=config)
    actual = source2_pre_pick(source_index=2, pre_pick=rotation @ pre, pick=rotation @ pick, config=config)
    assert actual == pytest.approx(expected)


def test_default_cli_is_disabled_and_enabled_cli_round_trips():
    parser = argparse.ArgumentParser()
    add_source2_routing_args(parser)
    assert source2_routing_from_args(parser.parse_args([])) is None
    assert source2_routing_arguments(None) == []
    config = Source2RoutingConfig()
    assert source2_routing_from_args(parser.parse_args(source2_routing_arguments(config))) == config
    assert source2_pre_pick(source_index=0, pre_pick=(1, 2, 3), pick=(0, 2, 3), config=None) == (1, 2, 3)


@pytest.mark.parametrize("side", [float("nan"), float("inf"), 0])
def test_invalid_offsets_rejected(side):
    with pytest.raises(ValueError):
        Source2RoutingConfig(side)


def test_wrong_route_and_vertical_approach_rejected():
    config = Source2RoutingConfig()
    with pytest.raises(ValueError, match="source_index"):
        source2_pre_pick(source_index=1, pre_pick=(0, 1, 0), pick=(0, 0, 0), config=config)
    with pytest.raises(ValueError, match="horizontal"):
        source2_pre_pick(source_index=2, pre_pick=(0, 0, 1), pick=(0, 0, 0), config=config)


def test_source2_snapshot_and_sequence_have_same_pre_pick_and_unchanged_grasp():
    base = (1.287, -0.034, 0.816, math.sqrt(0.5), 0, 0, -math.sqrt(0.5))
    policy = Source2RoutingConfig()
    common = dict(environment_origin_w=(0, 0, 0), base_pose_w=base)
    old = task_snapshot(config=PickPlaceV2Config(source_index=2, destination_index=4, lift_clearance_m=0.3), **common)
    routed = task_snapshot(
        config=PickPlaceV2Config(source_index=2, destination_index=4, lift_clearance_m=0.3, source2_routing=policy),
        **common,
    )
    assert old["task_hash"] != routed["task_hash"]
    assert "source2_routing" not in old
    assert old["source"] == routed["source"] and old["destination"] == routed["destination"]
    points = old["waypoint_position_b"]
    for name in ("pick", "lift", "place_high", "place"):
        assert routed["waypoint_position_b"][name] == points[name]
    sequence = DatasetSequence(
        preposition_b=points["pre_pick"],
        grasp_tcp_position_b=points["pick"],
        lift_position_b=points["lift"],
        place_high_position_b=points["place_high"],
        place_position_b=points["place"],
        place_tcp_direction_w=(1, 0, 0),
        config=SequenceConfig(advance_distance_m=0.075),
        source_state_index=2,
        place_state_index=4,
        source2_routing=policy,
    )
    assert sequence.preposition_b == pytest.approx(routed["waypoint_position_b"]["pre_pick"])
    assert sequence.grasp_tcp_position_b == pytest.approx(points["pick"])
    sequence.reset()
    assert sequence.preposition_b == pytest.approx(routed["waypoint_position_b"]["pre_pick"])
    require_height(routed, 0.3)
    routed["waypoint_position_b"]["pre_pick"][0] += 0.001
    with pytest.raises(ValueError, match="geometry differs"):
        require_height(routed, 0.3)
    with pytest.raises(ValueError, match="source_index=2"):
        PickPlaceV2Config(source_index=1, destination_index=4, source2_routing=policy)


def test_raised_approach_waits_for_each_waypoint_before_closing_and_resets():
    policy = Source2RoutingConfig(0.06, 0.04)
    pick = (0.1, 0.2, 0.3)
    sequence = DatasetSequence(
        preposition_b=(0.1, 0.275, 0.3), grasp_tcp_position_b=pick,
        lift_position_b=(0.1, 0.2, 0.6), place_high_position_b=(0.4, 0.4, 0.6),
        place_position_b=(0.4, 0.4, 0.3), place_tcp_direction_w=(1, 0, 0),
        source_state_index=2, place_state_index=4, source2_routing=policy,
        config=SequenceConfig(advance_distance_m=0.075, normal_dwell_steps=3),
        initial_state=SequenceState(SequenceStage.ADVANCE_NORMAL),
    )
    assert sequence.advance_waypoints[0] == pytest.approx((0.1, 0.2, 0.34))
    assert sequence.advance_waypoints[1] == pick
    assert sequence.preposition_b == pytest.approx((0.04, 0.275, 0.34))

    def observe(position):
        return SequenceObservation(position, (1, 0, 0, 0), (0.04, 0.04))

    def advance(position):
        return sequence.command(
            observe(position), active_waypoint_b=sequence.preposition_b, active_waypoint_label="pre_pick",
            path_complete=True, closed_gripper_position=(0, 0), open_gripper_position=(0.04, 0.04),
        )

    # Being at the final grasp point cannot skip the raised entry.
    advance(pick)
    assert sequence.state.stage is SequenceStage.ADVANCE_NORMAL
    assert sequence.state.advance_waypoint_index == 0
    for _ in range(3):
        advance(sequence.advance_waypoints[0])
    assert sequence.state.advance_waypoint_index == 1
    assert sequence.state.stage is SequenceStage.ADVANCE_NORMAL
    for _ in range(2):
        advance(pick)
        assert sequence.state.stage is SequenceStage.ADVANCE_NORMAL
    advance(pick)
    assert sequence.state.stage is SequenceStage.CLOSE_GRIPPER
    sequence.reset()
    assert sequence.state.advance_waypoint_index == 0


def test_lift_cli_snapshot_contact_geometry_and_historical_hash_compatibility():
    parser = argparse.ArgumentParser()
    add_source2_routing_args(parser)
    config = Source2RoutingConfig(0.06, 0.04)
    assert source2_routing_from_args(parser.parse_args(source2_routing_arguments(config))) == config
    assert source2_routing_payload(Source2RoutingConfig()) == {"pre_pick_side_m": 0.06}
    assert source2_advance_waypoints(pick=(1, 2, 3), config=None) == ((1, 2, 3),)
    with pytest.raises(ValueError, match="side offset"):
        source2_routing_from_args(parser.parse_args(["--state2-pre-pick-lift-m", "0.04"]))
    common = dict(environment_origin_w=(0, 0, 0), base_pose_w=(0, 0, 0, 1, 0, 0, 0))
    old = task_snapshot(config=PickPlaceV2Config(source_index=2, destination_index=4), **common)
    raised = task_snapshot(
        config=PickPlaceV2Config(source_index=2, destination_index=4, source2_routing=config), **common
    )
    pick = np.asarray(old["waypoint_position_b"]["pick"])
    assert raised["waypoint_position_b"]["pick"] == pick.tolist()
    assert raised["contact_waypoint_position_b"]["pick_entry"] == pytest.approx(pick + (0, 0, 0.04))
    assert old["destination"] == raised["destination"]
    require_height(raised, raised["lift_clearance_m"])
    raised["contact_waypoint_position_b"]["pick_entry"][2] += 0.001
    with pytest.raises(ValueError, match="geometry differs"):
        require_height(raised, raised["lift_clearance_m"])


@pytest.mark.parametrize("lift", [float("nan"), float("inf"), -0.001])
def test_invalid_lift_is_rejected(lift):
    with pytest.raises(ValueError, match="nonnegative"):
        Source2RoutingConfig(0.06, lift)
