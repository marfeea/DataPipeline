"""Pure-data contract for scene-consistent CuRobo trajectory replay."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from configs.curobo.world import CuboidConfig, WorldConfig

from .planner import JOINT_NAMES, TrajectoryData, load_trajectory
from .robot_adapter import pose_matrix, rigid_transform
from .world_adapter import WorldSnapshot


@dataclass(frozen=True)
class ReplayContract:
    frame_rate_hz: float = 60.0
    time_tolerance_s: float = 1e-9
    hash_length: int = 64
    target_name: str = "target"
    joint_names: tuple[str, ...] = JOINT_NAMES


@dataclass(frozen=True)
class ReplayFixture:
    name: str
    trajectory_sha256: str
    plan_report_sha256: str
    urdf_sha256: str
    collision_model_sha256: str
    world_hash: str


RUN_006_FIXTURE = ReplayFixture(
    name="curobo_mvp_run_006",
    trajectory_sha256="899eb467978c4d5215b0c8d3d2b7874150c3e3bef3b96f3d42a02bbe98dcbce0",
    plan_report_sha256="260de79fdc27b06224292fd5ce3c941ce0dc1958263039b1633d741af6547fdd",
    urdf_sha256="93bf0b06c284ca42ed689d91977964fc26e146187a084dad06beb752f68a29d8",
    collision_model_sha256="609e9184c7e4f0050172511d66de2a7cdf6de45b96112d45cf4d94fbb475b41f",
    world_hash="13d929d26a31b5e238f641bfc64fbde2dc331ea4d4352e3cfbeb7b443e2e8706",
)


@dataclass(frozen=True)
class ReplaySpec:
    trajectory_path: Path
    plan_report_path: Path
    urdf_path: Path
    collision_model_path: Path
    ee_link: str
    trajectory: TrajectoryData
    world: WorldSnapshot
    plan_report: dict
    hashes: dict[str, str]
    fixture_name: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_path(value, report_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = report_path.parent / path
    return path.resolve()


def _require_finite_json(value, name="report") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_json(item, f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_json(item, f"{name}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} contains a non-finite number")


def _load_world(report: dict, contract: ReplayContract) -> WorldSnapshot:
    robot_world = report.get("robot_world")
    if not isinstance(robot_world, dict) or not isinstance(robot_world.get("world"), list):
        raise ValueError("plan report has no robot_world.world list")
    cuboids = []
    for item in robot_world["world"]:
        if not isinstance(item, dict) or set(item) != {"name", "pose_b", "dimensions_m"}:
            raise ValueError("plan report cuboid fields differ")
        cuboids.append(CuboidConfig(item["name"], item["pose_b"], item["dimensions_m"]))
    return WorldSnapshot(WorldConfig(tuple(cuboids), contract.target_name))


def _validate_npz(path: Path, trajectory: TrajectoryData, report: dict, contract: ReplayContract) -> None:
    with np.load(path, allow_pickle=False) as raw:
        if raw["time_s"].dtype != np.dtype("float64"):
            raise ValueError("time_s must use float64")
        for field in ("joint_position", "joint_velocity", "joint_acceleration"):
            if raw[field].dtype != np.dtype("float64"):
                raise ValueError(f"{field} must use float64")
        if raw["joint_names"].dtype.kind != "U":
            raise ValueError("joint_names must use a non-object Unicode dtype")
    time_s = np.asarray(trajectory.time_s)
    q = np.asarray(trajectory.joint_position)
    dq = np.asarray(trajectory.joint_velocity)
    ddq = np.asarray(trajectory.joint_acceleration)
    count = len(time_s) if time_s.ndim == 1 else 0
    if count < 2 or q.shape != dq.shape or q.shape != ddq.shape or q.shape != (count, 6):
        raise ValueError("trajectory arrays must have matching [N,6] shapes with N >= 2")
    if tuple(trajectory.joint_names) != contract.joint_names:
        raise ValueError("trajectory joint names/order differ from the replay contract")
    if not all(np.isfinite(value).all() for value in (time_s, q, dq, ddq)):
        raise ValueError("trajectory contains non-finite values")
    expected_dt = 1.0 / contract.frame_rate_hz
    if abs(float(time_s[0])) > contract.time_tolerance_s:
        raise ValueError("trajectory time must start at zero")
    if not np.all(np.diff(time_s) > 0) or not np.allclose(
        np.diff(time_s), expected_dt, rtol=0.0, atol=contract.time_tolerance_s
    ):
        raise ValueError("trajectory must use a strictly increasing 60 Hz timeline")
    for key, values in (("start_q", q[0]), ("goal_q", q[-1])):
        expected = np.asarray(report.get(key), dtype=np.float64)
        if expected.shape != (6,) or not np.allclose(values, expected, rtol=0.0, atol=1e-6):
            raise ValueError(f"trajectory {key} differs from the plan report")
    samples = report.get("acceptance", {}).get("metrics", {}).get("samples")
    if samples != count:
        raise ValueError("trajectory frame count differs from the plan report")


def load_replay_spec(
    trajectory_path,
    plan_report_path,
    *,
    fixture: ReplayFixture = RUN_006_FIXTURE,
    contract: ReplayContract = ReplayContract(),
) -> ReplaySpec:
    """Load and cross-check every artifact required before Isaac may start."""
    trajectory_path = Path(trajectory_path).expanduser().resolve()
    report_path = Path(plan_report_path).expanduser().resolve()
    for path, label in ((trajectory_path, "trajectory"), (report_path, "plan report")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require_finite_json(report)
    if report.get("passed") is not True or report.get("stage") != "complete":
        raise ValueError("plan report is not a completed passing run")
    reported_trajectory = _artifact_path(report.get("trajectory_path", ""), report_path)
    if reported_trajectory != trajectory_path:
        raise ValueError("requested trajectory does not match plan report trajectory_path")
    planner_config = report.get("planner_config", {})
    if not math.isclose(float(planner_config.get("control_hz", 0.0)), contract.frame_rate_hz):
        raise ValueError("plan report control rate is not 60 Hz")
    robot_world = report.get("robot_world", {})
    urdf_path = _artifact_path(robot_world.get("urdf", ""), report_path)
    collision_path = _artifact_path(report.get("collision_model", ""), report_path)
    if not urdf_path.is_file() or not collision_path.is_file():
        raise FileNotFoundError("plan report robot assets are missing")
    collision_model = json.loads(collision_path.read_text(encoding="utf-8"))
    ee_link = collision_model.get("attachment_parent")
    if ee_link != "project_tcp":
        raise ValueError(f"collision model has an unexpected planning ee link: {ee_link!r}")
    hashes = {
        "trajectory": sha256_file(trajectory_path),
        "plan_report": sha256_file(report_path),
        "urdf": sha256_file(urdf_path),
        "collision_model": sha256_file(collision_path),
    }
    reported_hashes = {
        "urdf": robot_world.get("urdf_sha256"),
        "collision_model": report.get("collision_model_sha256"),
    }
    for name, expected in reported_hashes.items():
        if hashes[name] != expected:
            raise ValueError(f"{name} content hash differs from the plan report")
    expected_fixture = {
        "trajectory": fixture.trajectory_sha256,
        "plan_report": fixture.plan_report_sha256,
        "urdf": fixture.urdf_sha256,
        "collision_model": fixture.collision_model_sha256,
    }
    mismatches = [name for name, expected in expected_fixture.items() if hashes[name] != expected]
    if mismatches:
        raise ValueError(f"acceptance fixture hash mismatch: {mismatches}")
    trajectory = load_trajectory(trajectory_path)
    _validate_npz(trajectory_path, trajectory, report, contract)
    world = _load_world(report, contract)
    planned_world_hash = robot_world.get("world_hash")
    if world.world_hash != planned_world_hash or world.world_hash != fixture.world_hash:
        raise ValueError("reconstructed world hash differs from plan report or fixture")
    loaded_names = tuple(report.get("limits", {}).get("motion_gen_loaded", {}).get("joint_names", ()))
    if loaded_names != contract.joint_names:
        raise ValueError("MotionGen runtime joint order differs from replay contract")
    return ReplaySpec(
        trajectory_path=trajectory_path,
        plan_report_path=report_path,
        urdf_path=urdf_path,
        collision_model_path=collision_path,
        ee_link=ee_link,
        trajectory=trajectory,
        world=world,
        plan_report=report,
        hashes=hashes,
        fixture_name=fixture.name,
    )


def frame_deadlines(time_s, start_monotonic: float) -> tuple[float, ...]:
    values = np.asarray(time_s, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite one-dimensional frame times required")
    if not math.isfinite(start_monotonic):
        raise ValueError("finite monotonic start required")
    return tuple(float(start_monotonic + value) for value in values)


def summarize_tracking(reference_q, reference_dq, actual_q, actual_dq) -> dict:
    """Return finite per-joint and aggregate tracking errors for physics replay."""
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in (reference_q, reference_dq, actual_q, actual_dq))
    if any(value.ndim != 2 for value in arrays) or len({value.shape for value in arrays}) != 1:
        raise ValueError("tracking arrays must have one matching [N,J] shape")
    if arrays[0].shape[0] < 2 or arrays[0].shape[1] < 1:
        raise ValueError("tracking arrays require at least two frames and one joint")
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("tracking arrays must be finite")
    q_error = arrays[2] - arrays[0]
    dq_error = arrays[3] - arrays[1]

    def metrics(error):
        return {
            "max_abs": float(np.max(np.abs(error))),
            "rms": float(np.sqrt(np.mean(np.square(error)))),
            "per_joint_max_abs": np.max(np.abs(error), axis=0).tolist(),
            "per_joint_rms": np.sqrt(np.mean(np.square(error), axis=0)).tolist(),
        }

    return {
        "frames": int(arrays[0].shape[0]),
        "joint_count": int(arrays[0].shape[1]),
        "position_error_rad": metrics(q_error),
        "velocity_error_rad_s": metrics(dq_error),
        "final_position_error_rad": np.abs(q_error[-1]).tolist(),
        "final_position_max_abs_rad": float(np.max(np.abs(q_error[-1]))),
    }


def pose_b_to_w(base_pose_w, object_pose_b) -> np.ndarray:
    """Compose T_WB and T_BO; poses use xyz+wxyz."""
    return rigid_transform(
        pose_matrix(base_pose_w[:3], base_pose_w[3:]) @ pose_matrix(object_pose_b[:3], object_pose_b[3:])
    )


def pose_w_to_b(base_pose_w, object_pose_w) -> np.ndarray:
    """Recover T_BO from measured T_WB and T_WO."""
    base = pose_matrix(base_pose_w[:3], base_pose_w[3:])
    world = pose_matrix(object_pose_w[:3], object_pose_w[3:])
    return rigid_transform(np.linalg.inv(base) @ world)


def matrix_pose_wxyz(value) -> tuple[float, ...]:
    """Convert a rigid transform to xyz+wxyz with a deterministic quaternion sign."""
    matrix = rigid_transform(value)
    rotation = matrix[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        quaternion = np.array(
            (
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            )
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = 2.0 * math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            quaternion = np.array(
                (
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                )
            )
        elif index == 1:
            scale = 2.0 * math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            quaternion = np.array(
                (
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                )
            )
        else:
            scale = 2.0 * math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            quaternion = np.array(
                (
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                )
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return tuple(float(value) for value in (*matrix[:3, 3], *quaternion))


__all__ = [
    "RUN_006_FIXTURE",
    "ReplayContract",
    "ReplayFixture",
    "ReplaySpec",
    "frame_deadlines",
    "load_replay_spec",
    "matrix_pose_wxyz",
    "pose_b_to_w",
    "pose_w_to_b",
    "sha256_file",
    "summarize_tracking",
]
