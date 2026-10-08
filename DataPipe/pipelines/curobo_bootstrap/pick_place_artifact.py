"""Strict, pickle-free V2 hybrid trajectory assembly and readback."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from configs.curobo.world import MeshConfig, WorldConfig

from .measured_placement import require_measured_calibration
from .planner import JOINT_NAMES, read_urdf_joint_limits
from .route_experiment import require_height
from .route_identity import require_report_identity
from .world_adapter import WorldSnapshot

SEGMENTS = (
    "free_space_to_pre_pick",
    "legacy_approach_pick",
    "close_gripper",
    "legacy_lift",
    "free_space_transfer",
    "legacy_place_approach",
    "open_gripper",
)
STAGES = (
    "MOVE_TO_PICK",
    "MOVE_TO_PICK",
    "PICK",
    "MOVE_TO_PLACE",
    "MOVE_TO_PLACE",
    "MOVE_TO_PLACE",
    "PLACE",
)
SOURCES = (
    "curobo",
    "legacy_ik",
    "gripper_event",
    "legacy_ik",
    "curobo",
    "legacy_ik",
    "gripper_event",
)
FIELDS = frozenset(
    {
        "time_s",
        "joint_position",
        "joint_velocity",
        "joint_acceleration",
        "joint_names",
        "stage_id",
        "segment_source",
        "event_code",
        "gripper_target",
        "object_attached",
    }
)
CONTROL_HZ = 60.0
LEGACY_ACCELERATION_LIMIT_RAD_S2 = 50.0
MAX_FRAME_DISPLACEMENT_RAD = 0.1
MAX_HOLD_DRIFT_RAD = 0.03


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _segment_ranges(names):
    names = tuple(names)
    ranges = {}
    cursor = 0
    for name in SEGMENTS:
        end = cursor
        while end < len(names) and names[end] == name:
            end += 1
        if end == cursor:
            raise ValueError(f"missing or misordered V2 segment: {name}")
        ranges[name] = (cursor, end)
        cursor = end
    if cursor != len(names):
        raise ValueError("unexpected or repeated V2 segment")
    return ranges


def assemble_hybrid_trajectory(records, first, transfer):
    """Use Isaac measured q/dq and original MotionGen acceleration by source."""
    if not records:
        raise ValueError("empty Isaac record")
    names = [item["stage"] for item in records]
    ranges = _segment_ranges(names)
    n = len(records)
    q = np.asarray([item["q"] for item in records], dtype=np.float64)
    dq = np.asarray([item["dq"] for item in records], dtype=np.float64)
    if q.shape != (n, 6) or dq.shape != (n, 6) or not np.isfinite((q, dq)).all():
        raise ValueError("Isaac joint records must be finite [N,6]")
    first_start, first_end = ranges[SEGMENTS[0]]
    transfer_start, transfer_end = ranges[SEGMENTS[4]]
    if first_end - first_start != len(first.time_s) or transfer_end - transfer_start != len(transfer.time_s) - 1:
        raise ValueError("CuRobo source frame counts differ from MotionGen trajectories")
    if not np.allclose(q[first_start:first_end], first.joint_position, atol=1e-5, rtol=0):
        raise ValueError("first Isaac replay does not match MotionGen joint samples")
    if not np.allclose(q[transfer_start:transfer_end], transfer.joint_position[1:], atol=1e-5, rtol=0):
        raise ValueError("transfer Isaac replay does not match MotionGen joint samples")
    if not np.allclose(q[transfer_start - 1], transfer.joint_position[0], atol=1e-5, rtol=0):
        raise ValueError("measured Legacy lift does not match transfer start")
    ddq = np.empty_like(dq)
    ddq[0] = first.joint_acceleration[0]
    ddq[1:] = np.diff(dq, axis=0) * CONTROL_HZ
    ddq[first_start:first_end] = first.joint_acceleration
    ddq[transfer_start:transfer_end] = transfer.joint_acceleration[1:]
    stage = np.empty(n, dtype="<U20")
    source = np.empty(n, dtype="<U15")
    gripper = np.empty((n, 2), dtype=np.float64)
    for name, stage_name, owner in zip(SEGMENTS, STAGES, SOURCES, strict=True):
        start, end = ranges[name]
        stage[start:end] = stage_name
        source[start:end] = owner
        gripper[start:end] = (0.0115, -0.0115) if name in SEGMENTS[:2] else (0.0, 0.0)
    open_start, open_end = ranges["open_gripper"]
    gripper[open_start:open_end] = (0.0115, -0.0115)
    event = np.full(n, "NONE", dtype="<U5")
    pick_index = ranges["close_gripper"][1] - 1
    place_index = open_end - 1
    event[pick_index] = "PICK"
    event[place_index] = "PLACE"
    attached = np.zeros(n, dtype=np.bool_)
    attached[pick_index:place_index] = True
    return {
        "time_s": np.arange(n, dtype=np.float64) / CONTROL_HZ,
        "joint_position": q,
        "joint_velocity": dq,
        "joint_acceleration": ddq,
        "joint_names": np.asarray(JOINT_NAMES, dtype=np.str_),
        "stage_id": stage,
        "segment_source": source,
        "event_code": event,
        "gripper_target": gripper,
        "object_attached": attached,
    }


def validate_hybrid_trajectory(arrays, *, urdf):
    if set(arrays) != FIELDS:
        raise ValueError(f"V2 trajectory fields differ: {sorted(set(arrays) ^ FIELDS)}")
    n = len(arrays["time_s"])
    if n < 8:
        raise ValueError("V2 trajectory is too short")
    shapes = {
        "time_s": (n,),
        "joint_position": (n, 6),
        "joint_velocity": (n, 6),
        "joint_acceleration": (n, 6),
        "joint_names": (6,),
        "stage_id": (n,),
        "segment_source": (n,),
        "event_code": (n,),
        "gripper_target": (n, 2),
        "object_attached": (n,),
    }
    if any(np.asarray(arrays[key]).shape != shape for key, shape in shapes.items()):
        raise ValueError("V2 trajectory field shape mismatch")
    if tuple(arrays["joint_names"].tolist()) != JOINT_NAMES:
        raise ValueError("V2 joint order mismatch")
    if arrays["object_attached"].dtype != np.bool_:
        raise ValueError("object_attached must be boolean")
    numeric = ("time_s", "joint_position", "joint_velocity", "joint_acceleration", "gripper_target")
    if not all(np.issubdtype(arrays[key].dtype, np.number) and np.isfinite(arrays[key]).all() for key in numeric):
        raise ValueError("V2 numeric arrays must be finite")
    if not np.isclose(arrays["time_s"][0], 0, atol=1e-12, rtol=0) or not np.allclose(
        np.diff(arrays["time_s"]), 1 / CONTROL_HZ, atol=1e-12, rtol=0
    ):
        raise ValueError("V2 time axis must be exactly 60 Hz from zero")
    # Source changes within MOVE_TO_PICK/MOVE_TO_PLACE, so ranges are recovered
    # from the fixed stage/source pairs instead of stage labels alone.
    keys = list(zip(arrays["stage_id"].tolist(), arrays["segment_source"].tolist(), strict=True))
    pair_ranges = []
    cursor = 0
    while cursor < n:
        end = cursor + 1
        while end < n and keys[end] == keys[cursor]:
            end += 1
        pair_ranges.append((keys[cursor], cursor, end))
        cursor = end
    expected_pairs = list(zip(STAGES, SOURCES, strict=True))
    if [item[0] for item in pair_ranges] != expected_pairs:
        raise ValueError("V2 stage/source sequence differs from fixed hybrid route")
    if not np.array_equal(np.unique(arrays["event_code"], return_counts=True)[0], ["NONE", "PICK", "PLACE"]):
        raise ValueError("V2 event code set mismatch")
    pick = np.flatnonzero(arrays["event_code"] == "PICK")
    place = np.flatnonzero(arrays["event_code"] == "PLACE")
    if len(pick) != 1 or len(place) != 1 or pick[0] != pair_ranges[2][2] - 1 or place[0] != n - 1:
        raise ValueError("V2 event order or frame mismatch")
    expected_attached = np.zeros(n, dtype=np.bool_)
    expected_attached[pick[0] : place[0]] = True
    if not np.array_equal(arrays["object_attached"], expected_attached):
        raise ValueError("V2 logical attachment transition mismatch")
    expected_gripper = np.tile((0.0, 0.0), (n, 1))
    expected_gripper[: pair_ranges[2][1]] = (0.0115, -0.0115)
    expected_gripper[pair_ranges[-1][1] :] = (0.0115, -0.0115)
    if not np.allclose(arrays["gripper_target"], expected_gripper, atol=1e-12, rtol=0):
        raise ValueError("V2 gripper target transition mismatch")
    lower, upper, velocity_limit = read_urdf_joint_limits(urdf)
    q, dq, ddq = (arrays[key] for key in ("joint_position", "joint_velocity", "joint_acceleration"))
    if not (np.all(q >= lower - 1e-6) and np.all(q <= upper + 1e-6)):
        raise ValueError("V2 joint position limit failure")
    if not np.all(np.abs(dq) <= velocity_limit + 1e-5):
        raise ValueError("V2 joint velocity limit failure")
    from .planner import EXPERIMENTAL_ACCELERATION_LIMIT

    curobo = arrays["segment_source"] == "curobo"
    legacy = ~curobo
    if not np.all(np.abs(ddq[curobo]) <= EXPERIMENTAL_ACCELERATION_LIMIT + 1e-5):
        raise ValueError("V2 CuRobo acceleration limit failure")
    if not np.all(np.abs(ddq[legacy]) <= LEGACY_ACCELERATION_LIMIT_RAD_S2):
        raise ValueError("V2 Legacy measured acceleration envelope failure")
    max_step = float(np.max(np.abs(np.diff(q, axis=0))))
    if max_step > MAX_FRAME_DISPLACEMENT_RAD:
        raise ValueError("V2 joint frame discontinuity")
    hold_drift = {}
    for name, index in (("close_gripper", 2), ("open_gripper", 6)):
        _, start, end = pair_ranges[index]
        drift = float(np.max(np.abs(q[start:end] - q[start])))
        hold_drift[name] = drift
        if drift > MAX_HOLD_DRIFT_RAD:
            raise ValueError(f"V2 {name} arm hold drift exceeded")
    return {
        "passed": True,
        "samples": n,
        "duration_s": float(arrays["time_s"][-1]),
        "segment_ranges": {
            name: [int(start), int(end)] for name, (_, start, end) in zip(SEGMENTS, pair_ranges, strict=True)
        },
        "pick_frame": int(pick[0]),
        "place_frame": int(place[0]),
        "max_frame_displacement_rad": max_step,
        "max_velocity_rad_s": float(np.max(np.abs(dq))),
        "max_curobo_acceleration_rad_s2": float(np.max(np.abs(ddq[curobo]))),
        "max_legacy_acceleration_rad_s2": float(np.max(np.abs(ddq[legacy]))),
        "hold_drift_rad": hold_drift,
        "legacy_acceleration_source": "60 Hz difference of measured Isaac joint velocity; 50 rad/s² MVP envelope",
        "curobo_acceleration_source": "original MotionGen interpolated acceleration",
    }


def save_hybrid_trajectory(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_hybrid_trajectory(path):
    with np.load(Path(path), allow_pickle=False) as data:
        if set(data.files) != FIELDS:
            raise ValueError(f"V2 NPZ fields differ: {sorted(set(data.files) ^ FIELDS)}")
        return {key: np.array(data[key]) for key in data.files}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_hybrid_artifact(directory, *, asset_paths=None):
    """Independently reopen all three files and verify hashes and route gates."""
    directory = Path(directory)
    if asset_paths is None:
        asset_paths = {}
        mapping_path = directory / "asset_path_map.json"
        if mapping_path.exists():
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            if mapping.get("schema") != "curobo_artifact_asset_paths/1":
                raise ValueError("unsupported artifact asset path map schema")
            for original, relative in mapping["paths"].items():
                relocated = (directory / relative).resolve()
                if not relocated.is_relative_to(directory.resolve()):
                    raise ValueError("mapped artifact assets must stay inside the artifact directory")
                asset_paths[original] = relocated

    def asset_path(original):
        return Path(asset_paths.get(str(original), original))

    report = json.loads((directory / "plan_report.json").read_text(encoding="utf-8"))
    snapshot_path = directory / "scene_snapshot.json"
    trajectory_path = directory / "trajectory.npz"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not report.get("passed") or report.get("schema") != "curobo_pick_place_v2_plan_report/1":
        raise ValueError("V2 plan report is not a passed official run")
    if snapshot.get("schema") != "curobo_pick_place_v2_scene_snapshot/1":
        raise ValueError("V2 scene snapshot schema mismatch")
    if report["sha256"]["trajectory.npz"] != sha256_file(trajectory_path):
        raise ValueError("V2 trajectory SHA256 mismatch")
    if report["sha256"]["scene_snapshot.json"] != sha256_file(snapshot_path):
        raise ValueError("V2 scene snapshot SHA256 mismatch")
    task = snapshot["task"]
    require_height(task, task["lift_clearance_m"])
    require_report_identity(report, task, label="V2 plan")
    if task.get("state4_routing", {}).get("place_measured_grasp") and task["destination_index"] == 4:
        measured = report["measured"]
        for saved in (snapshot["placement_calibration"], report["placement_calibration"]):
            expected = require_measured_calibration(
                task,
                saved,
                lift_tcp_pose_b=measured["lift_tcp_pose_b"],
                object_pose_w_after_lift=measured["target_pose_w_after_lift"],
            )
        if not np.allclose(
            report["executed_waypoint_tcp_pose_b"]["place"],
            expected["tcp_waypoint_pose_b"]["place"],
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError("executed place TCP target differs from measured-grasp calibration")
    expected_source_w = np.asarray(task["frames"]["T_WE"], dtype=np.float64) @ np.asarray(
        [*task["source"]["position_e"], 1.0], dtype=np.float64
    )
    if np.linalg.norm(np.asarray(snapshot["target_pose_w_at_reset"][:3]) - expected_source_w[:3]) > 1e-4:
        raise ValueError("V2 measured source target pose differs from selected source state")
    if snapshot["world_hash"] != report["world_hash"]:
        raise ValueError("V2 world hash mismatch")
    world_data = snapshot["world"]
    world = WorldSnapshot(
        WorldConfig((), world_data["target_name"], tuple(MeshConfig(**mesh) for mesh in world_data["meshes"]))
    )
    if world.world_hash != snapshot["world_hash"]:
        raise ValueError("V2 world configuration does not reproduce its hash")
    if sha256_file(asset_path(snapshot["collision_model"])) != snapshot["collision_model_sha256"]:
        raise ValueError("V2 collision model SHA256 mismatch")
    if sha256_file(asset_path(snapshot["urdf"])) != snapshot["urdf_sha256"]:
        raise ValueError("V2 URDF SHA256 mismatch")
    for mesh in snapshot["world"]["meshes"]:
        if sha256_file(asset_path(mesh["file_path"])) != mesh["sha256"]:
            raise ValueError(f"V2 production mesh SHA256 mismatch: {mesh['name']}")
    arrays = load_hybrid_trajectory(trajectory_path)
    acceptance = validate_hybrid_trajectory(arrays, urdf=asset_path(snapshot["urdf"]))
    if acceptance != report["trajectory_acceptance"]:
        raise ValueError("V2 trajectory acceptance differs from frozen report")
    if not np.allclose(arrays["joint_position"][0], report["measured"]["reset_q"], atol=1e-5, rtol=0):
        raise ValueError("V2 first frame differs from same-run Isaac reset q")
    if not all(
        report["motion_gen"][key]["motion_gen"]["success"] and report["motion_gen"][key]["acceptance"]["passed"]
        for key in ("curobo_first", "curobo_transfer")
    ):
        raise ValueError("V2 same-run MotionGen source evidence is incomplete")
    if not all(item["passed"] for item in report["executed_waypoint_fk_residual"].values()):
        raise ValueError("V2 executed Legacy waypoint FK residual evidence failed")
    collision_diagnostic = report["legacy_collision_diagnostic"]
    if set(collision_diagnostic["segments"]) != {
        "legacy_approach_pick",
        "close_gripper",
        "legacy_lift",
        "legacy_place_approach",
        "open_gripper",
    }:
        raise ValueError("V2 Legacy contact collision diagnostic is incomplete")
    if collision_diagnostic.get("hard_gate"):
        trays = collision_diagnostic.get("protected_trays")
        policy = report.get("collision_policy", {})
        if (
            not trays
            or trays != policy.get("protected_trays")
            or policy.get("tray_collision_hard_gate") is not True
            or collision_diagnostic.get("tray_collision_free") is not True
            or any(
                item.get("tray_max_world_score", float("inf")) > 0 for item in collision_diagnostic["segments"].values()
            )
        ):
            raise ValueError("V2 support tray collision gate failed")
    return {"report": report, "snapshot": snapshot, "trajectory": arrays, "acceptance": acceptance}
