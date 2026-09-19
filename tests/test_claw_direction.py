"""Regress the physical child-prim axis, not merely a self-consistent heading."""

from math import sqrt

import pytest

from configs.coordinate_frames import (
    CLAW_CHILD_PRIM_RELATIVE_PATH,
    CLAW_DIRECTION_AXIS_C,
    FLANGE_NORMAL_AXIS_F,
    FLANGE_TO_CLAW_ROTATION_F_WXYZ,
    FLANGE_TO_TCP_ROTATION_F_WXYZ,
    TCP_DIRECTION_AXIS_T,
    WORLD_DOWN_DIRECTION_W,
    quaternion_conjugate_wxyz,
    quaternion_multiply_wxyz,
    rotate_direction_wxyz,
    target_direction_in_world,
    world_to_base_direction,
)
from configs.task import TARGET_STATES
from DataPipe.sequence import (
    DatasetSequence, SequenceConfig, SequenceObservation, SequenceStage, SequenceState,
    planar_heading_from_vector,
)


@pytest.mark.parametrize("stage,next_stage", [
    (SequenceStage.ALIGN_PLACE_DIRECTION, SequenceStage.PLACE_HIGH_XY),
    (SequenceStage.PLACE_HIGH_XY, SequenceStage.PLACE_FINAL_Z),
    (SequenceStage.PLACE_FINAL_Z, SequenceStage.PLACE_FINAL_XY),
    (SequenceStage.PLACE_FINAL_XY, SequenceStage.RELEASE_GRIPPER),
])
def test_wrong_physical_axis_cannot_descend_or_release(stage, next_stage):
    sequence = DatasetSequence(
        preposition_b=(0, 0, 0), grasp_tcp_position_b=(0, 0, 0),
        lift_position_b=(0, 0, 1), place_high_position_b=(1, 0, 1),
        place_position_b=(1, 0, 1), place_tcp_direction_w=(-1, 0, 0),
        config=SequenceConfig(normal_dwell_steps=2), grasp_enabled=True, place_enabled=True,
        initial_state=SequenceState(stage=stage),
    )
    def command(q):
        return sequence.command(
            SequenceObservation((1, 0, 1), q, (0, 0), (sqrt(.5), 0, 0, -sqrt(.5))),
            active_waypoint_b=(0, 0, 0), active_waypoint_label="target_settle",
            path_complete=True, closed_gripper_position=(0, 0), open_gripper_position=(.01, -.01),
        )
    # Old TCP +Y is aligned, but the requested child +Y is 90 degrees away.
    wrong = (0, 0, 0, 1)
    correct = (sqrt(.5), 0, 0, sqrt(.5))
    for _ in range(2):
        result = command(wrong)
        assert result.stage is stage
        assert result.gripper_command == -1
        assert result.flange_downward_enabled
    for _ in range(2):
        result = command(correct)
        assert result.gripper_command == -1
        assert result.flange_downward_enabled
    assert result.stage is next_stage


@pytest.mark.parametrize("index", range(len(TARGET_STATES)))
def test_real_heading_solver_aligns_child_y_and_keeps_flange_down(index):
    pytest.importorskip("torch")
    from DataPipe.runtime import ensure_curriculumrl_available
    ensure_curriculumrl_available()
    from CurriculumRL.logic.aubo_ik import (
        flange_down_tcp_target_pose, flange_down_yaw_offset_for_tcp_heading,
    )
    q_wb = (sqrt(.5), 0, 0, -sqrt(.5))
    target = target_direction_in_world(TARGET_STATES[index].rotation_wxyz)
    heading = planar_heading_from_vector(world_to_base_direction(target, q_wb))
    offset = flange_down_yaw_offset_for_tcp_heading(
        (.3, -.3, .8), heading, root_quaternion_wxyz=q_wb,
        tcp_heading_axis_local=TCP_DIRECTION_AXIS_T,
    )
    _, q_bt = flange_down_tcp_target_pose(
        (.3, -.3, .8), root_quaternion_wxyz=q_wb, downward_yaw_offset_rad=offset,
    )
    q_wf = quaternion_multiply_wxyz(
        quaternion_multiply_wxyz(q_wb, q_bt),
        quaternion_conjugate_wxyz(FLANGE_TO_TCP_ROTATION_F_WXYZ),
    )
    q_wc = quaternion_multiply_wxyz(q_wf, FLANGE_TO_CLAW_ROTATION_F_WXYZ)
    assert rotate_direction_wxyz(q_wc, CLAW_DIRECTION_AXIS_C) == pytest.approx(target, abs=1e-6)
    assert rotate_direction_wxyz(q_wf, FLANGE_NORMAL_AXIS_F) == pytest.approx(WORLD_DOWN_DIRECTION_W, abs=1e-6)


def test_calibration_matches_actual_child_prim_in_robot_usd():
    pytest.importorskip("pxr.Usd")
    from pxr import Gf, Usd, UsdGeom
    from configs.assets import resolve_asset_root
    stage = Usd.Stage.Open(str(resolve_asset_root() / "AUBO_E5/AUBO_E5_Withclaw.usd"))
    root = "/Root/AUBO_E5/"
    flange = stage.GetPrimAtPath(root + "Flange")
    child = stage.GetPrimAtPath(root + CLAW_CHILD_PRIM_RELATIVE_PATH)
    assert flange and child
    cache = UsdGeom.XformCache()
    relative = cache.GetLocalToWorldTransform(child) * cache.GetLocalToWorldTransform(flange).GetInverse()
    for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        assert tuple(relative.TransformDir(Gf.Vec3d(*axis)).GetNormalized()) == pytest.approx(
            rotate_direction_wxyz(FLANGE_TO_CLAW_ROTATION_F_WXYZ, axis), abs=1e-7,
        )
