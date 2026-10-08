from copy import deepcopy

import numpy as np
import pytest

from DataPipe.pipelines.curobo_bootstrap.measured_placement import (
    executed_endpoint_poses,
    measured_state4_placement,
    require_measured_calibration,
)
from DataPipe.pipelines.curobo_bootstrap.replay import matrix_pose_wxyz
from DataPipe.pipelines.curobo_bootstrap.robot_adapter import pose_matrix
from DataPipe.sequence import DatasetSequence, SequenceConfig, SequenceObservation, SequenceStage, SequenceState
from DataPipe.state4_routing import State4RoutingConfig


def matrix(pose):
    return pose_matrix(pose[:3], pose[3:])


def fixture(base_pose=(0.477, -0.075, 0.816, 1, 0, 0, 0), grip_translation=(-0.05, 0, -0.02)):
    base = matrix(base_pose)
    lift_tcp_w = matrix((0.8, -0.2, 1.4, 0, 0, 0, 1))
    half_angle = np.deg2rad(4) / 2
    grip = matrix((*grip_translation, np.cos(half_angle), np.sin(half_angle), 0, 0))
    object_lift = lift_tcp_w @ grip
    goal = (0.90264, -0.50461, 1.0915, 1, 0, 0, 0)
    nominal = (np.linalg.inv(base) @ [*goal[:3], 1])[:3] + [0, 0.045, 0.017]
    task = dict(
        destination_index=4,
        state4_routing=dict(place_entry_lift_m=0.004),
        frames=dict(T_WB=base.tolist(), T_WE=np.eye(4).tolist()),
        destination=dict(position_e=list(goal[:3]), rotation_wxyz=list(goal[3:])),
        waypoint_position_b=dict(place=nominal.tolist()),
        contact_waypoint_position_b=dict(
            place_descent=(nominal + [0, 0.24, 0.004]).tolist(),
            place_entry=(nominal + [0, 0, 0.004]).tolist(),
        ),
    )
    args = dict(
        lift_tcp_pose_b=list(matrix_pose_wxyz(np.linalg.inv(base) @ lift_tcp_w)),
        object_pose_w_after_lift=list(matrix_pose_wxyz(object_lift)),
    )
    return task, args, grip


def test_tilted_measured_grasp_reaches_unchanged_full_object_goal_and_raised_entry():
    task, args, grip = fixture()
    calibrated = measured_state4_placement(task, **args)
    base = np.asarray(task["frames"]["T_WB"])
    goal = matrix((*task["destination"]["position_e"], *task["destination"]["rotation_wxyz"]))
    final_object = base @ matrix(calibrated["tcp_waypoint_pose_b"]["place"]) @ grip
    np.testing.assert_allclose(final_object, goal, rtol=0, atol=1e-12)
    for name, point in task["contact_waypoint_position_b"].items():
        expected = goal.copy()
        offset_b = np.asarray(point) - task["waypoint_position_b"]["place"]
        expected[:3, 3] += base[:3, :3] @ offset_b
        actual = base @ matrix(calibrated["tcp_waypoint_pose_b"][name]) @ grip
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(final_object[:3, :3], np.eye(3), rtol=0, atol=1e-12)


def test_final_tcp_calibration_is_independent_of_base_coordinate_frame():
    first_task, first_args, _ = fixture()
    half_angle = np.deg2rad(37) / 2
    second_task, second_args, _ = fixture((0.1, 0.3, 0.5, np.cos(half_angle), 0, 0, np.sin(half_angle)))
    results = []
    for task, args in ((first_task, first_args), (second_task, second_args)):
        calibrated = measured_state4_placement(task, **args)
        results.append(np.asarray(task["frames"]["T_WB"]) @ matrix(calibrated["tcp_waypoint_pose_b"]["place"]))
    np.testing.assert_allclose(results[0], results[1], rtol=0, atol=1e-12)


def test_source_pick_contacts_are_excluded_from_measured_placement():
    task, args, _ = fixture()
    before = measured_state4_placement(task, **args)
    task["contact_waypoint_position_b"]["pick_entry"] = [0.1, 0.2, 0.3]
    task["contact_waypoint_position_b"]["pick_low_lift"] = [0.1, 0.2, 0.32]
    assert measured_state4_placement(task, **args) == before


def test_each_run_uses_its_own_grasp_translation_instead_of_a_fixed_tcp_offset():
    results = []
    for grip_translation in ((-0.05, 0, -0.02), (-0.04, 0.006, -0.015)):
        task, args, grip = fixture(grip_translation=grip_translation)
        calibrated = measured_state4_placement(task, **args)
        tcp = matrix(calibrated["tcp_waypoint_pose_b"]["place"])
        object_pose = np.asarray(task["frames"]["T_WB"]) @ tcp @ grip
        np.testing.assert_allclose(object_pose[:3, 3], task["destination"]["position_e"], rtol=0, atol=1e-12)
        results.append(tcp)
    assert np.linalg.norm(results[0][:3, 3] - results[1][:3, 3]) > 0.01


@pytest.mark.parametrize("bad_pose", [(0, 0, 0, 0, 0, 0, 0), (0, 0, np.nan, 1, 0, 0, 0)])
def test_invalid_measured_pose_is_rejected(bad_pose):
    task, args, _ = fixture()
    args["object_pose_w_after_lift"] = bad_pose
    with pytest.raises(ValueError):
        measured_state4_placement(task, **args)


def test_calibration_does_not_mutate_task_geometry_or_accept_an_unrouted_destination():
    task, args, _ = fixture()
    original = deepcopy(task)
    measured_state4_placement(task, **args)
    assert task == original
    task["destination_index"] = 3
    with pytest.raises(ValueError, match="state4"):
        measured_state4_placement(task, **args)


def measured_sequence(task):
    point = np.asarray(task["waypoint_position_b"]["place"])
    return DatasetSequence(
        preposition_b=(0.1, 0.42, 0.2),
        grasp_tcp_position_b=(0.1, 0.3, 0.2),
        lift_position_b=(0.1, 0.3, 1.4),
        place_high_position_b=(*point[:2], 1.4),
        place_position_b=point,
        place_preposition_b=point + [0, 0.075, 0],
        place_object_pose_b=(*task["destination"]["position_e"], *task["destination"]["rotation_wxyz"]),
        place_tcp_direction_w=(1, 0, 0),
        config=SequenceConfig(normal_dwell_steps=2),
        source_state_index=0,
        place_state_index=4,
        state4_routing=State4RoutingConfig(place_standoff_m=0.24, place_entry_lift_m=0.004, place_measured_grasp=True),
        initial_state=SequenceState(stage=SequenceStage.PLACE_FINAL_Z),
    )


def sequence_step(sequence, tcp_pose, object_pose):
    return sequence.command(
        SequenceObservation(tuple(tcp_pose[:3]), tuple(tcp_pose[3:]), (0, 0), object_pose_b=object_pose),
        active_waypoint_b=(0, 0, 0),
        active_waypoint_label="target_settle",
        path_complete=True,
        closed_gripper_position=(0, 0),
        open_gripper_position=(0.01, -0.01),
    )


def test_measured_sequence_levels_object_before_insertion_and_releases_only_at_original_object_goal():
    task, args, grip = fixture(base_pose=(0, 0, 0, 1, 0, 0, 0))
    sequence = measured_sequence(task)
    first = sequence_step(sequence, args["lift_tcp_pose_b"], args["object_pose_w_after_lift"])
    assert first.flange_downward_enabled is False
    assert np.linalg.norm(first.arm_action[3:]) > 0
    poses = sequence.placement_calibration["tcp_waypoint_pose_b"]
    for name, expected_stage in (
        ("place_descent", SequenceStage.PLACE_FINAL_XY),
        ("place_entry", SequenceStage.PLACE_FINAL_XY),
        ("place", SequenceStage.RELEASE_GRIPPER),
    ):
        tcp = poses[name]
        object_pose = list(matrix_pose_wxyz(matrix(tcp) @ grip))
        command = sequence_step(sequence, tcp, object_pose)
        assert command.gripper_command == -1.0
        command = sequence_step(sequence, tcp, object_pose)
        assert command.stage is expected_stage
        assert command.gripper_command == (1.0 if name == "place" else -1.0)
    final_object = matrix(poses["place"]) @ grip
    np.testing.assert_allclose(final_object[:3, 3], task["destination"]["position_e"], atol=1e-12)
    nominal = sequence.nominal_place_position_b
    sequence.reset()
    assert sequence.placement_calibration is None and sequence.measured_place_quaternion_b is None
    assert sequence.place_position_b == nominal


def test_measured_sequence_rejects_missing_object_observation_and_holds_tilted_entry():
    task, args, grip = fixture(base_pose=(0, 0, 0, 1, 0, 0, 0))
    sequence = measured_sequence(task)
    with pytest.raises(ValueError, match="observed object"):
        sequence_step(sequence, args["lift_tcp_pose_b"], None)
    sequence_step(sequence, args["lift_tcp_pose_b"], args["object_pose_w_after_lift"])
    poses = sequence.placement_calibration["tcp_waypoint_pose_b"]
    descent = poses["place_descent"]
    object_pose = list(matrix_pose_wxyz(matrix(descent) @ grip))
    sequence_step(sequence, descent, object_pose)
    sequence_step(sequence, descent, object_pose)
    entry = poses["place_entry"]
    object_pose = list(matrix_pose_wxyz(matrix(entry) @ grip))
    object_pose[3:] = [np.cos(np.deg2rad(5) / 2), np.sin(np.deg2rad(5) / 2), 0, 0]
    for _ in range(4):
        command = sequence_step(sequence, entry, object_pose)
        assert command.gripper_command == -1.0
        assert sequence.state.place_final_waypoint_index == 0


@pytest.mark.parametrize("tamper", ["tcp_waypoint", "object_goal", "grasp_rotation", "missing_waypoint"])
def test_saved_calibration_rejects_goal_grasp_and_waypoint_tampering(tamper):
    task, args, _ = fixture()
    calibrated = measured_state4_placement(task, **args)
    require_measured_calibration(task, calibrated, **args)
    if tamper == "tcp_waypoint":
        calibrated["tcp_waypoint_pose_b"]["place_entry"][2] += 0.001
    elif tamper == "object_goal":
        calibrated["desired_object_pose_w"][2] += 0.001
    elif tamper == "grasp_rotation":
        calibrated["tcp_from_object"][0][0] += 0.001
    else:
        del calibrated["tcp_waypoint_pose_b"]["place_descent"]
    with pytest.raises(ValueError, match="measured grasp"):
        require_measured_calibration(task, calibrated, **args)


def test_replay_uses_calibrated_place_pose_and_retains_historical_references():
    task, args, _ = fixture()
    task["state4_routing"]["place_measured_grasp"] = True
    calibrated = measured_state4_placement(task, **args)
    nominal = [*task["waypoint_position_b"]["place"], 1, 0, 0, 0]
    report = dict(
        legacy_ik_targets=dict(pre_pick=dict(tcp_pose_b=nominal), place=dict(tcp_pose_b=nominal)),
        measured=dict(
            lift_tcp_pose_b=args["lift_tcp_pose_b"], target_pose_w_after_lift=args["object_pose_w_after_lift"]
        ),
        placement_calibration=calibrated,
        executed_waypoint_tcp_pose_b=dict(place=calibrated["tcp_waypoint_pose_b"]["place"]),
    )
    poses = executed_endpoint_poses(task, report)
    assert poses["pre_pick"] == nominal
    assert poses["place"] == calibrated["tcp_waypoint_pose_b"]["place"]
    assert poses["place"] != nominal
    assert executed_endpoint_poses(dict(task, state4_routing={}), report)["place"] == nominal
    report["executed_waypoint_tcp_pose_b"]["place"] = nominal
    with pytest.raises(ValueError, match="executed endpoint"):
        executed_endpoint_poses(task, report)
