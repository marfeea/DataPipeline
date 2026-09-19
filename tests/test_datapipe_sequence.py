from __future__ import annotations

from math import atan2, cos, pi, sin, sqrt

import pytest

from DataPipe.sequence import (
    DatasetSequence,
    SequenceConfig,
    SequenceObservation,
    SequenceStage,
    SequenceState,
    approach_direction,
    axis_angle_error_rad,
    minimal_rotation_vector,
    planar_heading_from_quaternion,
    quaternion_rotation_error_vector,
    stage_for_waypoint_label,
    stepped_heading_target,
    wrap_angle_rad,
)


def _observation(
    position: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
    gripper: tuple[float, float] = (0.01, 0.0),
    base_quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> SequenceObservation:
    return SequenceObservation(position, quaternion, gripper, base_quaternion)


def test_normal_geometry_uses_prescribed_point_to_target_direction() -> None:
    desired = approach_direction((0.0, 0.12, 0.0), (0.0, 0.0, 0.0))
    assert desired == pytest.approx((0.0, -1.0, 0.0))
    rotation = minimal_rotation_vector((0.0, 0.0, 1.0), desired)
    assert sqrt(sum(value * value for value in rotation)) == pytest.approx(pi / 2.0)
    assert axis_angle_error_rad(desired, desired) == pytest.approx(0.0)


def test_antiparallel_axis_has_a_finite_half_turn_solution() -> None:
    rotation = minimal_rotation_vector((0.0, 0.0, 1.0), (0.0, 0.0, -1.0))
    assert sqrt(sum(value * value for value in rotation)) == pytest.approx(pi)


def test_quaternion_rotation_error_vector_uses_shortest_wxyz_error() -> None:
    half_turn = sqrt(0.5)
    rotation = quaternion_rotation_error_vector((1.0, 0.0, 0.0, 0.0), (half_turn, 0.0, 0.0, half_turn))
    assert rotation == pytest.approx((0.0, 0.0, pi / 2.0))


def test_interpolated_descent_waypoints_keep_match_z_stage() -> None:
    assert stage_for_waypoint_label("vertical_descent_interp") is SequenceStage.MATCH_Z
    assert stage_for_waypoint_label("target_settle_interp") is SequenceStage.MATCH_Z
    assert stage_for_waypoint_label("high_xy_approach_interp") is SequenceStage.HIGH_XY_APPROACH


def test_heading_step_uses_wrapped_shortest_branch_near_half_turn() -> None:
    current = pi / 2.0 + 0.01
    target = -pi / 2.0
    stepped = stepped_heading_target(current, target, 0.1)
    assert wrap_angle_rad(stepped - current) == pytest.approx(0.1)


def test_place_high_xy_turns_toward_midpoint_heading_while_transferring() -> None:
    source_heading = pi / 2.0 + 0.01
    source_quaternion = (
        cos(0.005),
        0.0,
        0.0,
        sin(0.005),
    )
    sequence = DatasetSequence(
        tcp_heading_axis=(0.0, 1.0, 0.0),  # Explicit synthetic/legacy frame for these stage tests.
        preposition_b=(0.0, 0.0, 0.0),
        grasp_tcp_position_b=(0.0, 0.0, 0.0),
        lift_position_b=(0.0, 0.0, 1.0),
        place_high_position_b=(1.0, 0.0, 1.0),
        place_position_b=(1.0, 0.0, 0.2),
        place_tcp_axis_b=(0.0, -1.0, 0.0),
        config=SequenceConfig(normal_dwell_steps=1, gripper_dwell_steps=1),
        grasp_enabled=True,
        place_enabled=True,
        initial_state=SequenceState(
            stage=SequenceStage.PLACE_HIGH_XY,
            tcp_quaternion_ref_b_wxyz=source_quaternion,
            tcp_heading_ref_rad=source_heading,
        ),
    )

    command = sequence.command(
        _observation((0.0, 0.0, 1.0), source_quaternion, (0.0, 0.0)),
        active_waypoint_b=(0.0, 0.0, 0.0),
        active_waypoint_label="target_settle",
        path_complete=True,
        closed_gripper_position=(0.0, 0.0),
        open_gripper_position=(0.01, -0.01),
    )

    assert command.stage is SequenceStage.PLACE_HIGH_XY
    assert wrap_angle_rad(command.tcp_heading_target_rad - source_heading) == pytest.approx(5.0 * pi / 180.0)
    assert command.target_position_b == pytest.approx((1.0, 0.0, 1.0))


def test_grasp_sequence_is_monotonic_and_measurement_gated() -> None:
    sequence = DatasetSequence(
        tcp_heading_axis=(0.0, 1.0, 0.0),  # Explicit synthetic/legacy frame for these stage tests.
        preposition_b=(0.0, 0.0, 0.0),
        target_b=(1.0, 0.0, 0.0),
        config=SequenceConfig(
            position_tolerance_m=0.005,
            normal_angle_tolerance_rad=0.02,
            normal_dwell_steps=2,
            advance_distance_m=0.1,
            gripper_position_tolerance=1.0e-4,
            gripper_dwell_steps=2,
        ),
        grasp_enabled=True,
    )
    # TCP +Z 必须落在 XOY 平面；绕 B 系 Y 轴 +90° 后指向 +X。
    horizontal = (sqrt(0.5), 0.0, sqrt(0.5), 0.0)

    command = sequence.command(
        _observation((0.1, 0.0, 0.2), horizontal),
        active_waypoint_b=(0.0, 0.0, 0.2),
        active_waypoint_label="high_pose_alignment",
        path_complete=False,
        closed_gripper_position=(0.0, 0.0),
    )
    assert command.stage is SequenceStage.HIGH_XY_APPROACH
    assert command.flange_downward_enabled
    assert command.dataset_action[-1] == 1.0

    command = sequence.command(
        _observation((0.0, 0.0, 0.0), horizontal),
        active_waypoint_b=(0.0, 0.0, 0.0),
        active_waypoint_label="target_settle",
        path_complete=True,
        closed_gripper_position=(0.0, 0.0),
    )
    assert command.stage is SequenceStage.ALIGN_NORMAL
    assert command.flange_downward_enabled
    assert command.arm_action[3:] == (0.0, 0.0, 0.0)

    for _ in range(2):
        command = sequence.command(
            _observation((0.0, 0.0, 0.0), horizontal),
            active_waypoint_b=(0.0, 0.0, 0.0),
            active_waypoint_label="target_settle",
            path_complete=True,
            closed_gripper_position=(0.0, 0.0),
        )
    assert command.stage is SequenceStage.ADVANCE_NORMAL
    assert command.target_position_b == pytest.approx((0.1, 0.0, 0.0))
    assert command.flange_downward_enabled
    assert command.arm_action[3:] == (0.0, 0.0, 0.0)

    command = sequence.command(
        _observation((0.0, 0.0, 0.0), (sqrt(0.5), 0.0, 0.0, sqrt(0.5))),
        active_waypoint_b=(0.0, 0.0, 0.0),
        active_waypoint_label="target_settle",
        path_complete=True,
        closed_gripper_position=(0.0, 0.0),
    )
    assert command.stage is SequenceStage.ADVANCE_NORMAL
    assert command.arm_action[3:] == pytest.approx((-1.0, 1.0, -1.0))

    for _ in range(2):
        command = sequence.command(
            _observation((0.1, 0.0, 0.0), horizontal),
            active_waypoint_b=(0.0, 0.0, 0.0),
            active_waypoint_label="target_settle",
            path_complete=True,
            closed_gripper_position=(0.0, 0.0),
        )
    assert command.stage is SequenceStage.CLOSE_GRIPPER
    assert command.gripper_command == -1.0
    assert command.arm_action == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert command.flange_downward_enabled
    assert command.arm_action[3:] == (0.0, 0.0, 0.0)

    for _ in range(2):
        command = sequence.command(
            _observation((0.1, 0.0, 0.0), horizontal, (0.0, 0.0)),
            active_waypoint_b=(0.0, 0.0, 0.0),
            active_waypoint_label="target_settle",
            path_complete=True,
            closed_gripper_position=(0.0, 0.0),
        )
    assert command.stage is SequenceStage.HOLD_COMPLETE
    assert command.complete
    assert command.flange_downward_enabled
    assert command.arm_action[3:] == (0.0, 0.0, 0.0)


def test_position_only_mode_completes_without_grasp_parameters() -> None:
    sequence = DatasetSequence(
        tcp_heading_axis=(0.0, 1.0, 0.0),  # Explicit synthetic/legacy frame for these stage tests.
        preposition_b=(0.0, 0.0, 0.0),
        target_b=(1.0, 0.0, 0.0),
        config=SequenceConfig(),
        grasp_enabled=False,
    )
    command = sequence.command(
        _observation((0.0, 0.0, 0.0), (sqrt(0.5), 0.0, sqrt(0.5), 0.0)),
        active_waypoint_b=(0.0, 0.0, 0.0),
        active_waypoint_label="target_settle",
        path_complete=True,
        closed_gripper_position=(0.0, 0.0),
    )
    assert command.complete
    assert command.stage is SequenceStage.HOLD_COMPLETE
    assert command.target_position_b == pytest.approx((0.0, 0.0, 0.0))


def test_grasp_mode_requires_an_explicit_grasp_tcp_position() -> None:
    with pytest.raises(ValueError, match="requires grasp_tcp_position_b"):
        DatasetSequence(
            preposition_b=(0.0, 0.0, 0.0),
            config=SequenceConfig(),
            grasp_enabled=True,
        )


def test_pick_place_sequence_aligns_then_corrects_xy_then_descends_and_releases() -> None:
    sequence = DatasetSequence(
        tcp_heading_axis=(0.0, 1.0, 0.0),  # Explicit synthetic/legacy frame for these stage tests.
        preposition_b=(0.0, 0.0, 0.0),
        grasp_tcp_position_b=(0.1, 0.0, 0.0),
        lift_position_b=(0.1, 0.0, 1.0),
        place_high_position_b=(1.0, 0.0, 1.0),
        place_position_b=(1.0, 0.0, 0.2),
        place_tcp_axis_b=(0.0, 1.0, 0.0),
        config=SequenceConfig(normal_dwell_steps=1, gripper_dwell_steps=1),
        grasp_enabled=True,
        place_enabled=True,
    )
    horizontal = (sqrt(0.5), 0.0, sqrt(0.5), 0.0)

    command = sequence.command(
        _observation((0.0, 0.0, 0.0), horizontal),
        active_waypoint_b=(0.0, 0.0, 0.0),
        active_waypoint_label="target_settle",
        path_complete=True,
        closed_gripper_position=(0.0, 0.0),
        open_gripper_position=(0.01, -0.01),
    )
    assert command.stage is SequenceStage.ADVANCE_NORMAL
    assert command.target_position_b == pytest.approx((0.1, 0.0, 0.0))

    command = sequence.command(
        _observation((0.1, 0.0, 0.0), horizontal),
        active_waypoint_b=(0.0, 0.0, 0.0),
        active_waypoint_label="target_settle",
        path_complete=True,
        closed_gripper_position=(0.0, 0.0),
        open_gripper_position=(0.01, -0.01),
    )
    assert command.stage is SequenceStage.CLOSE_GRIPPER
    assert command.gripper_command == -1.0
    assert command.arm_action == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    command = sequence.command(
        _observation((0.1, 0.0, 0.0), horizontal, (0.0, 0.0)),
        active_waypoint_b=(0.0, 0.0, 0.0),
        active_waypoint_label="target_settle",
        path_complete=True,
        closed_gripper_position=(0.0, 0.0),
        open_gripper_position=(0.01, -0.01),
    )
    assert command.stage is SequenceStage.LIFT_OBJECT
    assert command.target_position_b == pytest.approx((0.1, 0.0, 1.0))
    assert command.gripper_command == -1.0

    target_axis_quaternion = horizontal
    command = sequence.command(
        _observation((0.1, 0.0, 1.0), horizontal, (0.0, 0.0)),
        active_waypoint_b=(0.0, 0.0, 0.0),
        active_waypoint_label="target_settle",
        path_complete=True,
        closed_gripper_position=(0.0, 0.0),
        open_gripper_position=(0.01, -0.01),
    )
    assert command.stage is SequenceStage.PLACE_HIGH_XY
    assert command.target_position_b == pytest.approx((1.0, 0.0, 1.0))
    assert command.arm_action[3:] == pytest.approx((0.0, 0.0, 0.0))

    command = sequence.command(
        _observation((1.0, 0.0, 1.0), target_axis_quaternion, (0.0, 0.0)),
        active_waypoint_b=(0.0, 0.0, 0.0),
        active_waypoint_label="target_settle",
        path_complete=True,
        closed_gripper_position=(0.0, 0.0),
        open_gripper_position=(0.01, -0.01),
    )
    assert command.stage is SequenceStage.PLACE_FINAL_Z
    assert command.target_position_b == pytest.approx((1.0, 0.0, 0.2))
    assert command.arm_action[:2] == pytest.approx((0.0, 0.0))

    command = sequence.command(
        _observation((1.0, 0.0, 0.2), target_axis_quaternion, (0.0, 0.0)),
        active_waypoint_b=(0.0, 0.0, 0.0),
        active_waypoint_label="target_settle",
        path_complete=True,
        closed_gripper_position=(0.0, 0.0),
        open_gripper_position=(0.01, -0.01),
    )
    assert command.stage is SequenceStage.PLACE_FINAL_XY
    assert command.arm_action[3:] == pytest.approx((0.0, 0.0, 0.0))

    command = sequence.command(
        _observation((1.0, 0.0, 0.2), target_axis_quaternion, (0.0, 0.0)),
        active_waypoint_b=(0.0, 0.0, 0.0),
        active_waypoint_label="target_settle",
        path_complete=True,
        closed_gripper_position=(0.0, 0.0),
        open_gripper_position=(0.01, -0.01),
    )
    assert command.stage is SequenceStage.RELEASE_GRIPPER
    assert command.gripper_command == 1.0

    command = sequence.command(
        _observation((1.0, 0.0, 0.2), target_axis_quaternion, (0.01, -0.01)),
        active_waypoint_b=(0.0, 0.0, 0.0),
        active_waypoint_label="target_settle",
        path_complete=True,
        closed_gripper_position=(0.0, 0.0),
        open_gripper_position=(0.01, -0.01),
    )
    assert command.stage is SequenceStage.PLACE_COMPLETE
    assert command.complete
    assert command.gripper_command == 1.0


def test_place_direction_alignment_follows_fixed_radius_base_xoy_arc() -> None:
    sequence = DatasetSequence(
        preposition_b=(0.0, 0.0, 0.0),
        grasp_tcp_position_b=(0.0, 0.0, 0.0),
        lift_position_b=(1.0, 0.0, 1.0),
        place_high_position_b=(0.0, 1.0, 1.0),
        place_position_b=(0.0, 1.0, 0.2),
        place_tcp_direction_w=(-1.0, 0.0, 0.0),
        config=SequenceConfig(normal_dwell_steps=1, gripper_dwell_steps=1),
        grasp_enabled=True,
        place_enabled=True,
        initial_state=SequenceState(stage=SequenceStage.ALIGN_PLACE_DIRECTION),
    )

    command = sequence.command(
        _observation((1.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0)),
        active_waypoint_b=(0.0, 0.0, 0.0),
        active_waypoint_label="target_settle",
        path_complete=True,
        closed_gripper_position=(0.0, 0.0),
        open_gripper_position=(0.01, -0.01),
    )
    assert command.stage is SequenceStage.ALIGN_PLACE_DIRECTION
    assert (command.target_position_b[0] ** 2 + command.target_position_b[1] ** 2) ** 0.5 == pytest.approx(1.0)
    assert command.target_position_b[2] == pytest.approx(1.0)
    assert command.arm_action[2] == pytest.approx(0.0)
    assert command.target_position_b[1] > 0.0

    command = sequence.command(
        _observation((0.0, 1.0, 1.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0)),
        active_waypoint_b=(0.0, 0.0, 0.0),
        active_waypoint_label="target_settle",
        path_complete=True,
        closed_gripper_position=(0.0, 0.0),
        open_gripper_position=(0.01, -0.01),
    )
    assert command.stage is SequenceStage.PLACE_HIGH_XY
    assert command.target_position_b == pytest.approx((0.0, 1.0, 1.0))
    assert command.arm_action == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_align_place_direction_compares_tcp_and_target_in_world_frame() -> None:
    half_turn = (0.0, 0.0, 0.0, 1.0)
    base_in_world = (sqrt(0.5), 0.0, 0.0, -sqrt(0.5))
    sequence = DatasetSequence(
        tcp_heading_axis=(0.0, 1.0, 0.0),  # Explicit synthetic/legacy frame for these stage tests.
        preposition_b=(0.0, 0.0, 0.0),
        grasp_tcp_position_b=(0.0, 0.0, 0.0),
        lift_position_b=(0.0, 0.0, 1.0),
        place_high_position_b=(1.0, 0.0, 1.0),
        place_position_b=(1.0, 0.0, 0.2),
        place_tcp_direction_w=(-1.0, 0.0, 0.0),
        config=SequenceConfig(normal_dwell_steps=1, gripper_dwell_steps=1),
        grasp_enabled=True,
        place_enabled=True,
        initial_state=SequenceState(stage=SequenceStage.ALIGN_PLACE_DIRECTION),
    )

    # +Y_T 经 q_BT(绕 Z 180°) 后是 -Y_B，再经 q_WB(绕 Z -90°) 后是 -X_W。
    command = sequence.command(
        _observation((1.0, 0.0, 1.0), half_turn, (0.0, 0.0), base_in_world),
        active_waypoint_b=(0.0, 0.0, 0.0),
        active_waypoint_label="target_settle",
        path_complete=True,
        closed_gripper_position=(0.0, 0.0),
        open_gripper_position=(0.01, -0.01),
    )

    assert command.stage is SequenceStage.PLACE_HIGH_XY

@pytest.mark.parametrize("base_yaw", [0.0, pi / 2])
def test_grasp_aligns_before_docking(base_yaw):
    sequence = DatasetSequence(
        preposition_b=(0.0, 0.0, 0.0), grasp_tcp_position_b=(0.1, 0.0, 0.0),
        grasp_tcp_direction_w=(cos(base_yaw), sin(base_yaw), 0.0),
        tcp_heading_axis=(1.0, 0.0, 0.0),
        config=SequenceConfig(normal_dwell_steps=2), grasp_enabled=True,
        initial_state=SequenceState(stage=SequenceStage.ALIGN_NORMAL),
    )
    def step(yaw, position=(0.0, 0.0, 0.0)):
        return sequence.command(
            _observation(position, (cos(yaw / 2), 0.0, 0.0, sin(yaw / 2)),
                         base_quaternion=(cos(base_yaw / 2), 0.0, 0.0, sin(base_yaw / 2))),
            active_waypoint_b=(0.0, 0.0, 0.0), active_waypoint_label="target_settle",
            path_complete=True, closed_gripper_position=(0.0, 0.0),
        )
    for _ in range(4):
        command = step(pi / 2)
        assert command.stage is SequenceStage.ALIGN_NORMAL
        assert command.target_position_b == (0.0, 0.0, 0.0)
        assert command.arm_action[:3] == (0.0, 0.0, 0.0)
        assert command.tcp_heading_target_rad == pytest.approx(pi / 2 - sequence.config.heading_step_rad)
        assert command.flange_downward_enabled
    assert step(0.0).stage is SequenceStage.ALIGN_NORMAL
    assert step(pi / 2).stage is SequenceStage.ALIGN_NORMAL
    assert step(0.0, (0.0, 0.0, 0.1)).stage is SequenceStage.ALIGN_NORMAL
    assert step(0.0).stage is SequenceStage.ALIGN_NORMAL
    command = step(0.0)
    assert command.stage is SequenceStage.ADVANCE_NORMAL
    assert command.target_position_b == (0.1, 0.0, 0.0)
    assert command.tcp_heading_target_rad == pytest.approx(0.0)
    assert step(0.2).tcp_heading_target_rad == pytest.approx(0.0)


@pytest.mark.parametrize("index", [None, 0, 1, 2, 3, 4])
def test_place_detour_index_arrival_and_reset(index):
    initial = SequenceState(stage=SequenceStage.LIFT_OBJECT, tcp_heading_ref_rad=0.0)
    sequence = DatasetSequence(
        preposition_b=(0, 0, 0), grasp_tcp_position_b=(0, 0, 0),
        lift_position_b=(-0.2, -0.3, 0.6), place_high_position_b=(-0.1, 0.3, 0.6),
        place_position_b=(-0.1, 0.3, 0.1), place_tcp_direction_w=(1, 0, 0),
        tcp_heading_axis=(1, 0, 0), config=SequenceConfig(normal_dwell_steps=2),
        grasp_enabled=True, place_enabled=True, place_state_index=index, initial_state=initial,
    )

    def step(position):
        return sequence.command(
            _observation(position, (1, 0, 0, 0), (0, 0)),
            active_waypoint_b=(0, 0, 0), active_waypoint_label="target_settle",
            path_complete=True, closed_gripper_position=(0, 0), open_gripper_position=(0.01, -0.01),
        )

    lift, high = (-0.2, -0.3, 0.6), (-0.1, 0.3, 0.6)
    target_angle = atan2(high[1], high[0])
    radius = sqrt(lift[0] ** 2 + lift[1] ** 2)
    arc_end = (radius * cos(target_angle), radius * sin(target_angle), lift[2])
    step(lift)
    command = step(lift)
    assert command.stage is SequenceStage.ALIGN_PLACE_DIRECTION
    assert (command.target_position_b[0] ** 2 + command.target_position_b[1] ** 2) ** 0.5 == pytest.approx(radius)
    assert command.target_position_b[2] == pytest.approx(lift[2])
    assert step(arc_end).stage is SequenceStage.ALIGN_PLACE_DIRECTION
    command = step(arc_end)
    assert command.stage is SequenceStage.PLACE_HIGH_XY
    expected_high_target = sequence.place_high_waypoints[0]
    assert command.target_position_b == pytest.approx(expected_high_target)
    sequence.reset()
    assert sequence.state == initial
    step(lift)
    assert step(lift).stage is SequenceStage.ALIGN_PLACE_DIRECTION
