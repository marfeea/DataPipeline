import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from configs.curobo.world import CuboidConfig, WorldConfig
from DataPipe.pipelines.curobo_bootstrap.planner import GOAL_Q, JOINT_NAMES, SAFE_Q, TrajectoryData, save_trajectory
from DataPipe.pipelines.curobo_bootstrap.replay import (
    ReplayFixture,
    frame_deadlines,
    load_replay_spec,
    pose_b_to_w,
    pose_w_to_b,
    summarize_tracking,
)
from DataPipe.pipelines.curobo_bootstrap.robot_adapter import pose_error, pose_matrix
from DataPipe.pipelines.curobo_bootstrap.world_adapter import WorldSnapshot


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_fixture(tmp_path):
    urdf = tmp_path / "robot.urdf"
    collision = tmp_path / "collision.json"
    trajectory_path = tmp_path / "trajectory.npz"
    report_path = tmp_path / "report.json"
    urdf.write_text("<robot name='fixture'/>", encoding="utf-8")
    collision.write_text(json.dumps({"attachment_parent": "project_tcp"}), encoding="utf-8")
    time_s = np.arange(4, dtype=np.float64) / 60.0
    q = np.linspace(SAFE_Q, GOAL_Q, len(time_s), dtype=np.float64)
    trajectory = TrajectoryData(time_s, q, np.zeros_like(q), np.zeros_like(q), JOINT_NAMES)
    save_trajectory(trajectory_path, trajectory)
    world = {
        "cuboids": [
            {"name": "probe", "pose_b": [-3, -3, -3, 1, 0, 0, 0], "dimensions_m": [0.02] * 3},
            {"name": "target", "pose_b": [3, 3, 3, 1, 0, 0, 0], "dimensions_m": [0.02] * 3},
        ],
        "target_name": "target",
    }
    world_hash = WorldSnapshot(
        WorldConfig(
            tuple(CuboidConfig(row["name"], row["pose_b"], row["dimensions_m"]) for row in world["cuboids"]),
            "target",
        )
    ).world_hash
    report = {
        "passed": True,
        "stage": "complete",
        "trajectory_path": str(trajectory_path.resolve()),
        "start_q": SAFE_Q.tolist(),
        "goal_q": GOAL_Q.tolist(),
        "planner_config": {"control_hz": 60.0},
        "collision_model": str(collision.resolve()),
        "collision_model_sha256": digest(collision),
        "robot_world": {
            "urdf": str(urdf.resolve()),
            "urdf_sha256": digest(urdf),
            "world_hash": world_hash,
            "world": [
                {"name": row["name"], "pose_b": row["pose_b"], "dimensions_m": row["dimensions_m"]}
                for row in world["cuboids"]
            ],
        },
        "limits": {"motion_gen_loaded": {"joint_names": list(JOINT_NAMES)}},
        "acceptance": {"metrics": {"samples": len(time_s)}},
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    fixture = ReplayFixture(
        "test_fixture",
        digest(trajectory_path),
        digest(report_path),
        digest(urdf),
        digest(collision),
        world_hash,
    )
    return trajectory_path, report_path, fixture


def test_load_replay_spec_cross_checks_complete_identity_chain(tmp_path):
    trajectory, report, fixture = make_fixture(tmp_path)
    spec = load_replay_spec(trajectory, report, fixture=fixture)
    assert spec.fixture_name == "test_fixture"
    assert spec.trajectory.joint_position.shape == (4, 6)
    assert spec.world.world_hash == fixture.world_hash


@pytest.mark.parametrize("field", ["trajectory", "plan_report", "urdf", "collision_model"])
def test_any_fixture_hash_mismatch_is_rejected(tmp_path, field):
    trajectory, report, fixture = make_fixture(tmp_path)
    fixture = replace(fixture, **{field + "_sha256": "0" * 64})
    with pytest.raises(ValueError, match="fixture hash mismatch"):
        load_replay_spec(trajectory, report, fixture=fixture)


def test_swapped_report_is_rejected_before_artifact_loading(tmp_path):
    trajectory, report, fixture = make_fixture(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["trajectory_path"] = str(tmp_path / "other.npz")
    report.write_text(json.dumps(payload), encoding="utf-8")
    fixture = replace(fixture, plan_report_sha256=digest(report))
    with pytest.raises(ValueError, match="does not match"):
        load_replay_spec(trajectory, report, fixture=fixture)


def test_modified_world_is_rejected(tmp_path):
    trajectory, report, fixture = make_fixture(tmp_path)
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["robot_world"]["world"][0]["dimensions_m"][0] = 0.03
    report.write_text(json.dumps(payload), encoding="utf-8")
    fixture = replace(fixture, plan_report_sha256=digest(report))
    with pytest.raises(ValueError, match="world hash"):
        load_replay_spec(trajectory, report, fixture=fixture)


def test_wrong_joint_order_is_rejected(tmp_path):
    trajectory, report, fixture = make_fixture(tmp_path)
    with np.load(trajectory, allow_pickle=False) as raw:
        values = {name: np.array(raw[name]) for name in raw.files}
    values["joint_names"] = values["joint_names"][::-1]
    np.savez(trajectory, **values)
    fixture = replace(fixture, trajectory_sha256=digest(trajectory))
    with pytest.raises(ValueError, match="joint names/order"):
        load_replay_spec(trajectory, report, fixture=fixture)


def test_non_60_hz_timeline_is_rejected(tmp_path):
    trajectory, report, fixture = make_fixture(tmp_path)
    with np.load(trajectory, allow_pickle=False) as raw:
        values = {name: np.array(raw[name]) for name in raw.files}
    values["time_s"] *= 2
    np.savez(trajectory, **values)
    fixture = replace(fixture, trajectory_sha256=digest(trajectory))
    with pytest.raises(ValueError, match="60 Hz"):
        load_replay_spec(trajectory, report, fixture=fixture)


def test_deadlines_are_absolute_not_accumulated():
    assert frame_deadlines([0.0, 1 / 60, 2 / 60], 10.0) == (10.0, 10.0 + 1 / 60, 10.0 + 2 / 60)


def test_base_world_round_trip_with_nonidentity_base():
    half = np.sqrt(0.5)
    base = (1.0, 2.0, 3.0, half, 0.0, 0.0, half)
    local = (0.4, -0.2, 0.1, 1.0, 0.0, 0.0, 0.0)
    world_matrix = pose_b_to_w(base, local)
    world_pose = (*world_matrix[:3, 3], half, 0.0, 0.0, half)
    recovered = pose_w_to_b(base, world_pose)
    position, rotation = pose_error(pose_matrix(local[:3], local[3:]), recovered)
    assert position < 1e-12
    assert rotation < 1e-12


def test_tracking_summary_reports_signed_samples_as_absolute_metrics():
    reference_q = np.zeros((3, 2))
    reference_dq = np.zeros((3, 2))
    actual_q = np.array([[0.0, 0.0], [0.1, -0.2], [-0.3, 0.1]])
    actual_dq = np.array([[0.0, 0.0], [0.4, -0.1], [0.2, -0.5]])
    result = summarize_tracking(reference_q, reference_dq, actual_q, actual_dq)
    assert result["frames"] == 3
    assert result["joint_count"] == 2
    assert result["position_error_rad"]["max_abs"] == pytest.approx(0.3)
    assert result["position_error_rad"]["per_joint_max_abs"] == pytest.approx([0.3, 0.2])
    assert result["velocity_error_rad_s"]["max_abs"] == pytest.approx(0.5)
    assert result["final_position_max_abs_rad"] == pytest.approx(0.3)


@pytest.mark.parametrize(
    "actual_q, message",
    [
        (np.zeros((2, 3)), "matching"),
        (np.array([[0.0, 0.0], [np.nan, 0.0], [0.0, 0.0]]), "finite"),
    ],
)
def test_tracking_summary_rejects_invalid_measurements(actual_q, message):
    reference = np.zeros((3, 2))
    with pytest.raises(ValueError, match=message):
        summarize_tracking(reference, reference, actual_q, reference)
