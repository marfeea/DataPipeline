from dataclasses import asdict

import numpy as np
import pytest

from configs.curobo.world import MeshConfig, WorldConfig
from DataPipe.pipelines.curobo_bootstrap.pick_place_artifact import (
    SEGMENTS,
    assemble_hybrid_trajectory,
    load_hybrid_trajectory,
    save_hybrid_trajectory,
    save_json,
    sha256_file,
    validate_hybrid_trajectory,
    verify_hybrid_artifact,
)
from DataPipe.pipelines.curobo_bootstrap.planner import JOINT_NAMES, TrajectoryData
from DataPipe.pipelines.curobo_bootstrap.task_targets import PickPlaceV2Config, task_snapshot
from DataPipe.pipelines.curobo_bootstrap.world_adapter import WorldSnapshot


def _sample(tmp_path):
    urdf = tmp_path / "limits.urdf"
    urdf.write_text(
        "<robot name='test'>"
        + "".join(
            f"<joint name='{name}' type='revolute'><limit lower='-3' upper='3' velocity='4'/></joint>"
            for name in JOINT_NAMES
        )
        + "</robot>",
        encoding="utf-8",
    )
    lengths = [2, 2, 2, 2, 1, 2, 2]
    records = []
    for name, count in zip(SEGMENTS, lengths, strict=True):
        for _ in range(count):
            q = np.zeros(6)
            q[0] = len(records) * 0.001
            records.append({"stage": name, "q": q.tolist(), "dq": [0.0] * 6})
    q = np.asarray([item["q"] for item in records])
    first = TrajectoryData(np.array([0.0, 1 / 60]), q[:2], np.zeros((2, 6)), np.zeros((2, 6)), JOINT_NAMES)
    transfer = TrajectoryData(np.array([0.0, 1 / 60]), q[7:9], np.zeros((2, 6)), np.zeros((2, 6)), JOINT_NAMES)
    return urdf, records, first, transfer


def test_hybrid_artifact_roundtrip_and_events(tmp_path):
    urdf, records, first, transfer = _sample(tmp_path)
    arrays = assemble_hybrid_trajectory(records, first, transfer)
    path = tmp_path / "trajectory.npz"
    save_hybrid_trajectory(path, arrays)
    loaded = load_hybrid_trajectory(path)
    acceptance = validate_hybrid_trajectory(loaded, urdf=urdf)
    assert acceptance["passed"]
    assert acceptance["samples"] == 13
    assert acceptance["pick_frame"] == 5
    assert acceptance["place_frame"] == 12
    assert loaded["segment_source"][8] == "curobo"
    assert loaded["object_attached"].tolist() == [False] * 5 + [True] * 7 + [False]


@pytest.mark.parametrize("mutation", ["time", "source", "event", "attachment", "seam"])
def test_hybrid_artifact_rejects_corruption(tmp_path, mutation):
    urdf, records, first, transfer = _sample(tmp_path)
    arrays = assemble_hybrid_trajectory(records, first, transfer)
    if mutation == "time":
        arrays["time_s"][3] += 0.001
    elif mutation == "source":
        arrays["segment_source"][8] = "legacy_ik"
    elif mutation == "event":
        arrays["event_code"][4] = "PICK"
    elif mutation == "attachment":
        arrays["object_attached"][7] = False
    else:
        arrays["joint_position"][7, 0] += 0.5
    with pytest.raises(ValueError):
        validate_hybrid_trajectory(arrays, urdf=urdf)


def test_assembly_rejects_wrong_transfer_handoff(tmp_path):
    _, records, first, transfer = _sample(tmp_path)
    transfer.joint_position[0, 0] += 0.1
    with pytest.raises(ValueError, match="measured Legacy lift"):
        assemble_hybrid_trajectory(records, first, transfer)


@pytest.mark.parametrize(
    "corruption",
    [
        "trajectory_hash",
        "missing_snapshot",
        "tray_collision",
        "relocated_assets",
        "relocation_hash",
        "relocation_escape",
    ],
)
def test_artifact_readback_rejects_corrupt_or_partial_files(tmp_path, corruption):
    urdf, records, first, transfer = _sample(tmp_path)
    arrays = assemble_hybrid_trajectory(records, first, transfer)
    trajectory = tmp_path / "trajectory.npz"
    save_hybrid_trajectory(trajectory, arrays)
    mesh_path = tmp_path / "target.obj"
    mesh_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    mesh = MeshConfig("target", (0, 0, 0, 1, 0, 0, 0), str(mesh_path), sha256_file(mesh_path), "/World/target")
    world_config = WorldConfig((), "target", (mesh,))
    collision_model = tmp_path / "collision.json"
    collision_model.write_text("{}", encoding="utf-8")
    task = task_snapshot(
        config=PickPlaceV2Config(),
        environment_origin_w=(0.0, 0.0, 0.0),
        base_pose_w=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    )
    snapshot = {
        "schema": "curobo_pick_place_v2_scene_snapshot/1",
        "task": task,
        "world": {"target_name": "target", "meshes": [asdict(mesh)]},
        "world_hash": WorldSnapshot(world_config).world_hash,
        "target_pose_w_at_reset": [*task["source"]["position_e"], *task["source"]["rotation_wxyz"]],
        "collision_model": str(collision_model),
        "collision_model_sha256": sha256_file(collision_model),
        "urdf": str(urdf),
        "urdf_sha256": sha256_file(urdf),
    }
    snapshot_path = tmp_path / "scene_snapshot.json"
    save_json(snapshot_path, snapshot)
    report = {
        "schema": "curobo_pick_place_v2_plan_report/1",
        "passed": True,
        "source_index": task["source_index"],
        "destination_index": task["destination_index"],
        "source_name": task["source"]["name"],
        "destination_name": task["destination"]["name"],
        "task_hash": task["task_hash"],
        "world_hash": snapshot["world_hash"],
        "sha256": {"trajectory.npz": sha256_file(trajectory), "scene_snapshot.json": sha256_file(snapshot_path)},
        "trajectory_acceptance": validate_hybrid_trajectory(arrays, urdf=urdf),
        "measured": {"reset_q": arrays["joint_position"][0].tolist()},
        "motion_gen": {
            key: {"motion_gen": {"success": True}, "acceptance": {"passed": True}}
            for key in ("curobo_first", "curobo_transfer")
        },
        "executed_waypoint_fk_residual": {"pre_pick": {"passed": True}},
        "collision_policy": {"protected_trays": ["support_tray_01"], "tray_collision_hard_gate": True},
        "legacy_collision_diagnostic": {
            "hard_gate": True,
            "protected_trays": ["support_tray_01"],
            "tray_collision_free": True,
            "segments": {
                key: {"tray_max_world_score": 0.0}
                for key in (
                    "legacy_approach_pick",
                    "close_gripper",
                    "legacy_lift",
                    "legacy_place_approach",
                    "open_gripper",
                )
            },
        },
    }
    save_json(tmp_path / "plan_report.json", report)
    assert verify_hybrid_artifact(tmp_path)["acceptance"]["passed"]
    if corruption == "trajectory_hash":
        arrays["joint_position"][0, 0] += 0.01
        save_hybrid_trajectory(trajectory, arrays)
        with pytest.raises(ValueError, match="SHA256"):
            verify_hybrid_artifact(tmp_path)
    elif corruption == "missing_snapshot":
        snapshot_path.unlink()
        with pytest.raises(FileNotFoundError):
            verify_hybrid_artifact(tmp_path)
    elif corruption == "tray_collision":
        report["legacy_collision_diagnostic"]["segments"]["close_gripper"]["tray_max_world_score"] = 0.01
        save_json(tmp_path / "plan_report.json", report)
        with pytest.raises(ValueError, match="support tray collision"):
            verify_hybrid_artifact(tmp_path)
    else:
        assets = tmp_path / "referenced_assets"
        assets.mkdir()
        paths = {}
        for original in (mesh_path, collision_model, urdf):
            copied = assets / original.name
            copied.write_bytes(original.read_bytes())
            paths[str(original)] = str(copied.relative_to(tmp_path))
            original.unlink()
        mapping = dict(schema="curobo_artifact_asset_paths/1", paths=paths)
        save_json(tmp_path / "asset_path_map.json", mapping)
        assert verify_hybrid_artifact(tmp_path)["acceptance"]["passed"]
        if corruption == "relocation_hash":
            (assets / collision_model.name).write_text("changed", encoding="utf-8")
            with pytest.raises(ValueError, match="collision model SHA256"):
                verify_hybrid_artifact(tmp_path)
        elif corruption == "relocation_escape":
            mapping["paths"][str(mesh_path)] = "../outside.obj"
            save_json(tmp_path / "asset_path_map.json", mapping)
            with pytest.raises(ValueError, match="stay inside"):
                verify_hybrid_artifact(tmp_path)
