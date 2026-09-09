from __future__ import annotations

from math import sqrt

import pytest

from configs.coordinate_frames import (
    CLAW_DIRECTION_AXIS_C,
    FLANGE_TO_CLAW_ROTATION_F_WXYZ,
    FLANGE_NORMAL_AXIS_F,
    FLANGE_TO_TCP_ROTATION_F_WXYZ,
    TARGET_DOCKING_AXIS_G,
    TARGET_TO_TCP_ROTATION_G_WXYZ,
    TCP_DIRECTION_AXIS_T,
    TCP_TOOL_FORWARD_AXIS_T,
    WORLD_DOWN_DIRECTION_W,
    base_to_world_direction,
    base_to_world_position,
    direction_error_rad,
    flange_pose_in_world,
    rotate_direction_wxyz,
    target_direction_in_world,
    tcp_direction_in_world,
    tcp_pose_in_world_from_base_flange,
    world_to_base_direction,
    world_to_base_position,
)
from configs.task import TARGET_STATES


def test_current_direction_axis_contract_is_explicit() -> None:
    assert WORLD_DOWN_DIRECTION_W == (0.0, 0.0, -1.0)
    assert FLANGE_NORMAL_AXIS_F == (-1.0, 0.0, 0.0)
    assert CLAW_DIRECTION_AXIS_C == (0.0, 1.0, 0.0)
    assert TCP_DIRECTION_AXIS_T == (-1.0, 0.0, 0.0)
    assert TARGET_DOCKING_AXIS_G == (0.0, -1.0, 0.0)


def test_world_base_direction_round_trip_does_not_apply_translation() -> None:
    q_wb = (sqrt(0.5), 0.0, 0.0, -sqrt(0.5))
    direction_w = (-1.0, 0.0, 0.0)
    direction_b = world_to_base_direction(direction_w, q_wb)

    assert direction_b == pytest.approx((0.0, -1.0, 0.0), abs=1.0e-7)
    assert base_to_world_direction(direction_b, q_wb) == pytest.approx(direction_w, abs=1.0e-7)


def test_world_base_position_round_trip_includes_translation() -> None:
    position_wb_w = (2.0, 3.0, 4.0)
    q_wb = (sqrt(0.5), 0.0, 0.0, -sqrt(0.5))
    position_b = (1.0, 0.0, 0.0)

    position_w = base_to_world_position(position_b, position_wb_w, q_wb)

    assert position_w == pytest.approx((2.0, 2.0, 4.0), abs=1.0e-7)
    assert world_to_base_position(position_w, position_wb_w, q_wb) == pytest.approx(position_b, abs=1.0e-7)


def test_world_base_flange_tcp_pose_chain_is_explicit() -> None:
    position_wf_w, q_wf = flange_pose_in_world(
        position_wb_w=(1.0, 2.0, 3.0),
        q_wb_wxyz=(1.0, 0.0, 0.0, 0.0),
        position_bf_b=(0.5, 0.0, 0.0),
        q_bf_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    position_wt_w, q_wt = tcp_pose_in_world_from_base_flange(
        position_wb_w=(1.0, 2.0, 3.0),
        q_wb_wxyz=(1.0, 0.0, 0.0, 0.0),
        position_bf_b=(0.5, 0.0, 0.0),
        q_bf_wxyz=(1.0, 0.0, 0.0, 0.0),
    )

    assert position_wf_w == pytest.approx((1.5, 2.0, 3.0), abs=1.0e-7)
    assert q_wf == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1.0e-7)
    assert position_wt_w == pytest.approx((1.32914, 2.0, 3.11305), abs=1.0e-7)
    assert q_wt == pytest.approx(FLANGE_TO_TCP_ROTATION_F_WXYZ, abs=1.0e-7)


def test_target_index_one_configuration_produces_world_negative_x() -> None:
    direction_w = target_direction_in_world(TARGET_STATES[1].rotation_wxyz)
    assert direction_w == pytest.approx((-1.0, 0.0, 0.0), abs=1.0e-7)


def test_tcp_direction_is_compared_after_conversion_to_world() -> None:
    q_wb = (sqrt(0.5), 0.0, 0.0, -sqrt(0.5))
    q_bt = (sqrt(0.5), 0.0, 0.0, sqrt(0.5))
    actual_direction_w = tcp_direction_in_world(q_wb, q_bt)
    target_direction_w = target_direction_in_world(TARGET_STATES[1].rotation_wxyz)

    assert actual_direction_w == pytest.approx((-1.0, 0.0, 0.0), abs=1.0e-7)
    assert direction_error_rad(actual_direction_w, target_direction_w) == pytest.approx(0.0, abs=1.0e-7)


def test_flange_to_tcp_rotation_semantics_are_locked() -> None:
    # 控制器 -X_T 和子 prim +Y_C 必须映射到同一法兰方向 +Z_F。
    assert rotate_direction_wxyz(FLANGE_TO_TCP_ROTATION_F_WXYZ, TCP_DIRECTION_AXIS_T) == pytest.approx(
        (0.0, 0.0, 1.0), abs=1.0e-7
    )
    assert rotate_direction_wxyz(FLANGE_TO_CLAW_ROTATION_F_WXYZ, CLAW_DIRECTION_AXIS_C) == pytest.approx(
        (0.0, 0.0, 1.0), abs=1.0e-7
    )


def test_full_pose_tool_axis_and_planar_heading_axis_are_not_mixed() -> None:
    # 旧 q_GT 只说明 +Z_T 对齐 -Y_G，不能用于子 prim 的平面朝向。
    assert rotate_direction_wxyz(TARGET_TO_TCP_ROTATION_G_WXYZ, TCP_TOOL_FORWARD_AXIS_T) == pytest.approx(
        TARGET_DOCKING_AXIS_G, abs=1.0e-7
    )
    assert rotate_direction_wxyz(TARGET_TO_TCP_ROTATION_G_WXYZ, TCP_DIRECTION_AXIS_T) == pytest.approx(
        (-1.0, 0.0, 0.0), abs=1.0e-7
    )
