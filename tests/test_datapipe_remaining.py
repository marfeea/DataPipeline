from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from DataPipe import (  # noqa: E402
    CameraCaptureAdapter,
    CameraFrame,
    CameraSensorSpec,
    ControlState,
    DatasetConfig,
    EpisodeBuilder,
    EpisodeOutcome,
    EpisodeSupervisor,
    R2DreamerBatch,
    StateCaptureAdapter,
    TerminationReason,
    dataset_config_with_project_diagnostic_cameras,
    project_diagnostic_camera_sensor_specs,
)
from DataPipe.episode_naming import next_episode_index  # noqa: E402
from DataPipe.trajectory import (  # noqa: E402
    advance_reached_waypoints,
    build_waypoints,
    interpolate_waypoints,
    make_plan,
    physics_steps_for_control,
    resolve_interpolation_substeps,
)
from DataPipe.validator import validate_dataset  # noqa: E402
from DataPipe.writer import TrajectoryWriter  # noqa: E402


def make_episode(episode_id: str, steps: int = 2):
    builder = EpisodeBuilder(episode_id)
    builder.start({"step": 0})
    for step in range(steps):
        from tests.test_datapipe_p0 import make_action

        builder.append(make_action(), float(step), {"step": step + 1})
    return builder.close(TerminationReason.TIMEOUT, {"step": steps})


def test_scene_config_requires_camera_pose_when_requested() -> None:
    camera = CameraSensorSpec("front", "/front", (64, 64), ("rgb",))
    with pytest.raises(ValueError, match="缺少"):
        DatasetConfig(cameras=(camera,)).validate(require_camera_poses=True)
    DatasetConfig(
        cameras=(CameraSensorSpec("front", "/front", (64, 64), ("rgb",), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0)),)
    ).validate(require_camera_poses=True)


def test_datapipe_camera_uses_project_three_view_configuration() -> None:
    from configs.vision_diagnostics import DIAGNOSTIC_CAMERA_POSES

    specs = project_diagnostic_camera_sensor_specs(resolution=(160, 120))
    assert tuple(spec.name for spec in specs) == ("camera_cfg", "camera_cfg_2", "camera_cfg_3")
    assert tuple(spec.position_e for spec in specs) == tuple(pose.position_e for pose in DIAGNOSTIC_CAMERA_POSES)
    assert all(spec.resolution == (160, 120) for spec in specs)
    assert all(spec.pose_convention == "opengl" for spec in specs)
    assert dataset_config_with_project_diagnostic_cameras(min_free_disk_bytes=0).cameras == (
        project_diagnostic_camera_sensor_specs()
    )


def test_trajectory_kernel_matches_fixed_rate_contract() -> None:
    planning = build_waypoints((0.0, 0.0, 1.0), (1.0, 1.0, 0.5), 4, trajectory_mode="staged")
    control = interpolate_waypoints((0.0, 0.0, 1.0), planning, 3)
    assert len(planning) == 4
    assert len(control) == 12
    assert control[-1][1] == (1.0, 1.0, 0.5)
    assert resolve_interpolation_substeps(4.0, 12.0) == 3
    assert physics_steps_for_control(12.0, 1.0 / 120.0) == 10


def test_trajectory_plan_is_single_source_for_planning_and_control() -> None:
    plan = make_plan(
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.5),
        4,
        planning_hz=4.0,
        control_hz=12.0,
        trajectory_mode="direct",
    )
    assert plan.interpolation_substeps == 3
    assert len(plan.planning_waypoints) == 4
    assert len(plan.control_waypoints) == 12
    assert plan.control_waypoints[-1][1] == (1.0, 1.0, 0.5)


def test_waypoint_progression_waits_for_actual_tcp_and_holds_final_target() -> None:
    waypoints = (
        ("vertical_descent", (0.0, 0.0, 0.9)),
        ("vertical_descent", (0.0, 0.0, 0.8)),
        ("vertical_descent", (0.0, 0.0, 0.7)),
    )
    assert advance_reached_waypoints(waypoints, 0, (0.0, 0.0, 0.92), 0.01) == (0, False)
    assert advance_reached_waypoints(waypoints, 0, (0.0, 0.0, 0.895), 0.01) == (1, False)
    assert advance_reached_waypoints(waypoints, 1, (0.0, 0.0, 0.8), 0.01) == (2, False)
    assert advance_reached_waypoints(waypoints, 2, (0.0, 0.0, 0.705), 0.01) == (2, True)


def test_supervisor_consumes_formal_outcome_once() -> None:
    builder = EpisodeBuilder("supervised")
    builder.start("o0")
    builder.append(make_episode("unused").steps[0].action, 0.0, "o1")
    supervisor = EpisodeSupervisor()
    episode = supervisor.submit(builder, EpisodeOutcome(success=False, timeout=True, safety_failure=False), "terminal")
    assert episode.termination_reason is TerminationReason.TIMEOUT
    with pytest.raises(RuntimeError, match="重复"):
        supervisor.submit(builder, EpisodeOutcome(success=False, timeout=True, safety_failure=False), "terminal")


def test_capture_adapter_keeps_one_sim_step_for_all_fields() -> None:
    from tests.test_datapipe_p0 import make_action

    source = make_action()
    state = ControlState(
        12,
        1.0,
        source.raw_action,
        source.action,
        source.tcp_displacement,
        source.ik_output,
        source.joint_position,
        source.joint_velocity,
        "obs",
        tcp_heading_target_rad=source.tcp_heading_target_rad,
    )
    record = StateCaptureAdapter().capture(state)
    assert record.sim_step == 12
    assert record.action.action == source.action


def test_camera_adapter_rejects_stale_or_unsynchronized_frames() -> None:
    spec = CameraSensorSpec("front", "/front", (64, 64), ("rgb",))
    adapter = CameraCaptureAdapter((spec,))
    frame = CameraFrame("front", 1, 0.25, [[0]], (1, 1), "uint8")
    adapter.capture((frame,), expected_sim_step=1)
    with pytest.raises(ValueError, match="陈旧"):
        adapter.capture((frame,), expected_sim_step=1)


def test_camera_adapter_allows_new_frame_sequence_after_episode_reset() -> None:
    spec = CameraSensorSpec("front", "/front", (64, 64), ("rgb",))
    adapter = CameraCaptureAdapter((spec,))
    frame = CameraFrame("front", 7, 1.0, [[0]], (1, 1), "uint8")
    adapter.capture((frame,), expected_sim_step=7)

    adapter.reset_episode()

    adapter.capture((frame,), expected_sim_step=7)


def test_camera_adapter_requires_each_of_three_views_once() -> None:
    specs = tuple(
        CameraSensorSpec(name, f"/{name}", (64, 64), ("rgb",))
        for name in ("camera_cfg", "camera_cfg_2", "camera_cfg_3")
    )
    frames = tuple(CameraFrame(spec.name, 1, 0.25, [[0]], (1, 1), "uint8") for spec in specs)
    adapter = CameraCaptureAdapter(specs)
    adapter.capture(frames, expected_sim_step=1)

    duplicate = (frames[0], frames[0], frames[2])
    with pytest.raises(ValueError, match="重复"):
        CameraCaptureAdapter(specs).capture(duplicate, expected_sim_step=1)
    with pytest.raises(ValueError, match="集合"):
        CameraCaptureAdapter(specs).capture(frames[:2], expected_sim_step=1)


def test_writer_resume_validator_and_batch(tmp_path: Path) -> None:
    config = DatasetConfig(episodes_per_shard=2, min_free_disk_bytes=0)
    writer = TrajectoryWriter(tmp_path, config)
    writer.write(make_episode("e0"))
    writer.write(make_episode("e1"))
    writer.write(make_episode("e2"))
    with pytest.raises(ValueError, match="重复"):
        writer.write(make_episode("e0"))
    resumed = TrajectoryWriter(tmp_path, config)
    episodes = resumed.load_episodes()
    assert len(episodes) == 3
    report = validate_dataset(tmp_path)
    assert report.passed
    batch = R2DreamerBatch.from_episodes(list(episodes))
    batch.validate()
    assert batch.as_mapping()["action"][0][0] == episodes[0].steps[0].action.action


def test_hdf5_loader_filters_episodes_and_observation_fields(tmp_path: Path) -> None:
    from DataPipe.hdf5_writer import HDF5TrajectoryWriter
    from tests.test_datapipe_p0 import make_action

    def build_replay_episode(episode_id: str):
        builder = EpisodeBuilder(episode_id)
        observation = {
            "q": [0.0] * 6,
            "dq": [0.0] * 6,
            "tcp_pose_wxyz": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "target_pose_wxyz": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "gripper_position": [0.0115, -0.0115],
            "gripper_velocity": [0.0, 0.0],
            "semantic_stage_id": 0,
            "waypoint_index": 0,
            "preposition_position_b": [0.0, 0.0, 0.0],
            "object_position_b": [0.1, 0.0, 0.0],
            "desired_approach_axis_b": [1.0, 0.0, 0.0],
            "advance_target_position_b": [0.05, 0.0, 0.0],
            "image_front": [[[0, 0, 0]]],
        }
        builder.start(observation)
        builder.append(make_action(), 0.0, observation)
        return builder.close(TerminationReason.TIMEOUT, observation)

    writer = HDF5TrajectoryWriter(tmp_path, DatasetConfig(min_free_disk_bytes=0))
    writer.write(build_replay_episode("e0"))
    writer.write(build_replay_episode("e1"))

    episodes = writer.load_episodes(
        episode_ids=("e1",),
        observation_keys=(
            "q",
            "dq",
            "tcp_pose",
            "target_pose",
            "gripper_position",
            "gripper_velocity",
            "semantic_stage_id",
            "waypoint_index",
        ),
    )

    assert tuple(episode.episode_id for episode in episodes) == ("e1",)
    assert set(episodes[0].observations[0].value) == {
        "q",
        "dq",
        "tcp_pose_wxyz",
        "target_pose_wxyz",
        "gripper_position",
        "gripper_velocity",
        "semantic_stage_id",
        "waypoint_index",
    }


def test_next_episode_index_allocates_a_full_new_batch() -> None:
    prefix = "p1-run-seed6-ep"
    assert next_episode_index((), prefix) == 0
    assert next_episode_index((f"{prefix}0000", f"{prefix}0004"), prefix) == 5
    assert next_episode_index((f"{prefix}0000", f"{prefix}0002", "other-ep0009"), prefix) == 3
