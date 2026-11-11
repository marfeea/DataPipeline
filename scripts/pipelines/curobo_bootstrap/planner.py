"""Minimal, version-gated CuRobo MotionGen planning and artifact validation."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np

from configs.curobo.planner import PlannerConfig
from configs.curobo.world import CuboidConfig, WorldConfig

from .collision_model import curobo_robot_dict, load_collision_model
from .curobo_collision import CuroboCollision, pack_attached_spheres
from .world_adapter import AttachmentState, WorldSnapshot, curobo_world_dict

JOINT_NAMES = ("Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Flange")
SAFE_Q = np.array([0.0, -0.52359879, 1.22173047, 0.78539819, 1.57079637, 0.0], dtype=np.float64)
GOAL_Q = SAFE_Q + np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
EXPERIMENTAL_ACCELERATION_LIMIT = np.full(6, 5.0, dtype=np.float64)
EXPERIMENTAL_JERK_LIMIT = np.full(6, 50.0, dtype=np.float64)


@dataclass(frozen=True)
class TrajectoryData:
    time_s: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    joint_acceleration: np.ndarray
    joint_names: tuple[str, ...]


def mvp_world():
    """Return the deterministic, deliberately distant synthetic MVP world."""
    identity = (1.0, 0.0, 0.0, 0.0)
    return WorldSnapshot(
        WorldConfig(
            (
                CuboidConfig("target", (3.0, 3.0, 3.0, *identity), (0.02, 0.02, 0.02)),
                CuboidConfig("probe", (-3.0, -3.0, -3.0, *identity), (0.02, 0.02, 0.02)),
            ),
            "target",
        )
    )


def read_urdf_joint_limits(urdf):
    """Read the six planning-joint position and velocity limits in contract order."""
    root = ET.parse(Path(urdf)).getroot()
    joints = {node.attrib["name"]: node for node in root.findall("joint")}
    lower, upper, velocity = [], [], []
    for name in JOINT_NAMES:
        if name not in joints or joints[name].find("limit") is None:
            raise ValueError(f"missing URDF limit for {name}")
        limit = joints[name].find("limit")
        lower.append(float(limit.attrib["lower"]))
        upper.append(float(limit.attrib["upper"]))
        velocity.append(float(limit.attrib["velocity"]))
    arrays = tuple(np.asarray(values, dtype=np.float64) for values in (lower, upper, velocity))
    if not all(values.shape == (6,) and np.isfinite(values).all() for values in arrays):
        raise ValueError("URDF joint limits must be finite six-axis vectors")
    if np.any(arrays[0] >= arrays[1]) or np.any(arrays[2] <= 0):
        raise ValueError("URDF joint position/velocity limits are invalid")
    return arrays


def planning_robot_dict(collision_model, retract_q=SAFE_Q):
    """Add the explicit CuRobo 0.7.8 cspace fields required by MotionGen."""
    payload = curobo_robot_dict(collision_model)
    retract = np.asarray(retract_q, dtype=np.float64)
    if retract.shape != (6,) or not np.isfinite(retract).all():
        raise ValueError("retract_q must be a finite six-axis vector")
    payload["kinematics"]["cspace"] = {
        "joint_names": list(JOINT_NAMES),
        "retract_config": retract.tolist(),
        "null_space_weight": [1.0] * 6,
        "cspace_distance_weight": [1.0] * 6,
        "max_acceleration": EXPERIMENTAL_ACCELERATION_LIMIT.tolist(),
        "max_jerk": EXPERIMENTAL_JERK_LIMIT.tolist(),
    }
    return payload


def save_trajectory(path, trajectory):
    """Save an artifact that can be loaded with allow_pickle=False."""
    path = Path(path)
    np.savez(
        path,
        time_s=np.asarray(trajectory.time_s, dtype=np.float64),
        joint_position=np.asarray(trajectory.joint_position, dtype=np.float64),
        joint_velocity=np.asarray(trajectory.joint_velocity, dtype=np.float64),
        joint_acceleration=np.asarray(trajectory.joint_acceleration, dtype=np.float64),
        joint_names=np.asarray(trajectory.joint_names, dtype=np.str_),
    )


def load_trajectory(path):
    with np.load(Path(path), allow_pickle=False) as data:
        required = {"time_s", "joint_position", "joint_velocity", "joint_acceleration", "joint_names"}
        if set(data.files) != required:
            raise ValueError(f"trajectory fields differ: {sorted(set(data.files) ^ required)}")
        return TrajectoryData(
            np.array(data["time_s"], dtype=np.float64),
            np.array(data["joint_position"], dtype=np.float64),
            np.array(data["joint_velocity"], dtype=np.float64),
            np.array(data["joint_acceleration"], dtype=np.float64),
            tuple(str(value) for value in data["joint_names"].tolist()),
        )


def validate_trajectory(
    trajectory,
    *,
    start_q,
    goal_q,
    position_lower,
    position_upper,
    velocity_limit,
    acceleration_limit,
    control_hz=60.0,
    collision_query=None,
    numeric_tolerance=1e-5,
):
    """Validate a reloaded trajectory and return auditable per-check results."""
    start_q = np.asarray(start_q, dtype=np.float64)
    goal_q = np.asarray(goal_q, dtype=np.float64)
    lower = np.asarray(position_lower, dtype=np.float64)
    upper = np.asarray(position_upper, dtype=np.float64)
    velocity_limit = np.asarray(velocity_limit, dtype=np.float64)
    acceleration_limit = np.asarray(acceleration_limit, dtype=np.float64)
    q = np.asarray(trajectory.joint_position, dtype=np.float64)
    dq = np.asarray(trajectory.joint_velocity, dtype=np.float64)
    ddq = np.asarray(trajectory.joint_acceleration, dtype=np.float64)
    time_s = np.asarray(trajectory.time_s, dtype=np.float64)
    n = len(time_s) if time_s.ndim == 1 else 0
    checks = {
        "sample_count": n >= 2,
        "field_shapes": q.shape == dq.shape == ddq.shape == (n, 6),
        "joint_order": tuple(trajectory.joint_names) == JOINT_NAMES,
        "finite": all(np.isfinite(values).all() for values in (time_s, q, dq, ddq)),
    }
    shape_ok = checks["sample_count"] and checks["field_shapes"] and checks["finite"]
    expected_dt = 1.0 / float(control_hz)
    checks.update(
        {
            "nonzero_motion": bool(shape_ok and np.max(np.abs(q[-1] - q[0])) >= 0.01),
            "time_starts_zero": bool(n and np.isfinite(time_s[0]) and abs(time_s[0]) <= 1e-12),
            "time_strictly_increasing": bool(n >= 2 and np.all(np.diff(time_s) > 0)),
            "time_step": bool(n >= 2 and np.allclose(np.diff(time_s), expected_dt, rtol=0, atol=1e-9)),
            "start": bool(shape_ok and np.max(np.abs(q[0] - start_q)) <= 1e-4),
            "goal": bool(shape_ok and np.max(np.abs(q[-1] - goal_q)) <= 0.01),
            "position_limits": bool(
                shape_ok and np.all(q >= lower - numeric_tolerance) and np.all(q <= upper + numeric_tolerance)
            ),
            "velocity_limits": bool(shape_ok and np.all(np.abs(dq) <= velocity_limit + numeric_tolerance)),
            "acceleration_limits": bool(shape_ok and np.all(np.abs(ddq) <= acceleration_limit + numeric_tolerance)),
        }
    )
    collision_scores = None
    if collision_query is not None and shape_ok:
        collision_scores = collision_query(q)
        world = np.asarray(collision_scores["world"])
        self_collision = np.asarray(collision_scores["self"])
        collision_shapes_ok = world.shape == self_collision.shape == (n,)
        checks["collision_result_shapes"] = collision_shapes_ok
        checks["world_collision_free"] = bool(collision_shapes_ok and np.isfinite(world).all() and np.max(world) <= 0.0)
        checks["self_collision_free"] = bool(
            collision_shapes_ok and np.isfinite(self_collision).all() and np.max(self_collision) <= 0.0
        )
    else:
        checks["collision_result_shapes"] = False
        checks["world_collision_free"] = False
        checks["self_collision_free"] = False
    metrics = {
        "samples": n,
        "duration_s": float(time_s[-1]) if n and np.isfinite(time_s[-1]) else None,
        "expected_dt_s": expected_dt,
        "max_start_error_rad": float(np.max(np.abs(q[0] - start_q))) if shape_ok else None,
        "max_goal_error_rad": float(np.max(np.abs(q[-1] - goal_q))) if shape_ok else None,
        "max_joint_displacement_rad": float(np.max(np.abs(q[-1] - q[0]))) if shape_ok else None,
        "max_abs_velocity_rad_s": float(np.max(np.abs(dq))) if shape_ok else None,
        "max_abs_acceleration_rad_s2": float(np.max(np.abs(ddq))) if shape_ok else None,
    }
    if collision_scores is not None and checks["collision_result_shapes"]:
        metrics["max_world_collision_score"] = float(np.max(collision_scores["world"]))
        metrics["max_self_collision_score"] = float(np.max(collision_scores["self"]))
    return {"passed": all(checks.values()), "checks": checks, "metrics": metrics}


class CuroboMvpPlanner:
    """Narrow runtime adapter for one fixed joint-space MotionGen query."""

    def __init__(self, collision_model, *, config=None, world=None):
        self.config = config or PlannerConfig(version="0.7.8")
        self.version = version("nvidia_curobo")
        if self.version != self.config.version or self.version != "0.7.8":
            raise ValueError(f"MotionGen requires tested CuRobo 0.7.8, got {self.version}")
        self.asset = load_collision_model(collision_model)
        self.world = world or mvp_world()
        self.position_lower, self.position_upper, self.velocity_limit = read_urdf_joint_limits(self.asset["urdf"])
        self.collision = CuroboCollision(self.asset, self.world, device=self.config.device)
        # The production workstation mesh contains the robot's fixed mounting
        # socket.  Base_Link is immobile and intentionally intersects it.
        # Moving-link world collision remains enabled. The independent checker
        # rechecks exported samples with the complete self-collision model.
        self.world_collision_ignore_links = ("Base_Link",) if self.world.config.meshes else ()

        import torch

        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.geom.types import WorldConfig as CuroboWorldConfig
        from curobo.types.base import TensorDeviceType
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

        self.torch = torch
        self.tensor_args = TensorDeviceType(device=torch.device(self.config.device))
        checker_type = CollisionCheckerType.MESH if self.world.config.meshes else CollisionCheckerType.PRIMITIVE
        motion_config = MotionGenConfig.load_from_robot_config(
            planning_robot_dict(self.asset),
            world_model=CuroboWorldConfig.from_dict(curobo_world_dict(self.world)),
            tensor_args=self.tensor_args,
            collision_checker_type=checker_type,
            collision_cache={
                "obb": max(1, len(self.world.active_cuboids)),
                "mesh": max(1, len(self.world.config.meshes)),
            },
            self_collision_check=True,
            self_collision_opt=True,
            interpolation_dt=1.0 / self.config.control_hz,
            minimum_trajectory_dt=1.0 / self.config.control_hz,
            evaluate_interpolated_trajectory=True,
            use_cuda_graph=self.config.cuda_graph_enabled,
        )
        self.motion_gen = MotionGen(motion_config)
        for link_name in self.world_collision_ignore_links:
            self.motion_gen.kinematics.kinematics_config.disable_link_spheres(link_name)
        runtime = self.motion_gen.kinematics.kinematics_config

        def cpu_array(value):
            return value.detach().cpu().numpy().astype(np.float64)

        self.runtime_limits = {
            "joint_names": list(runtime.joint_limits.joint_names),
            "position_lower_rad": cpu_array(runtime.joint_limits.position[0]).tolist(),
            "position_upper_rad": cpu_array(runtime.joint_limits.position[1]).tolist(),
            "velocity_rad_s": cpu_array(runtime.joint_limits.velocity[1]).tolist(),
            "acceleration_rad_s2": cpu_array(runtime.joint_limits.acceleration[1]).tolist(),
            "jerk_rad_s3": cpu_array(runtime.joint_limits.jerk[1]).tolist(),
        }
        expected = {
            "position_lower_rad": self.position_lower,
            "position_upper_rad": self.position_upper,
            "velocity_rad_s": self.velocity_limit,
            "acceleration_rad_s2": EXPERIMENTAL_ACCELERATION_LIMIT,
            "jerk_rad_s3": EXPERIMENTAL_JERK_LIMIT,
        }
        if self.runtime_limits["joint_names"] != list(JOINT_NAMES):
            raise RuntimeError(f"MotionGen loaded unexpected joint order: {self.runtime_limits['joint_names']}")
        mismatches = [
            name
            for name, values in expected.items()
            if not np.allclose(self.runtime_limits[name], values, rtol=0.0, atol=1e-6)
        ]
        if mismatches:
            raise RuntimeError(f"MotionGen loaded limits differ from requested values: {mismatches}")

    def endpoint_checks(self, start_q, goal_q):
        values = np.stack((start_q, goal_q))
        scores = self.collision.query(values, world_ignore_links=self.world_collision_ignore_links)
        return {
            "within_position_limits": bool(
                np.all(values >= self.position_lower) and np.all(values <= self.position_upper)
            ),
            "world_collision_free": bool(np.max(scores["world"]) <= 0.0),
            "self_collision_free": bool(np.max(scores["self"]) <= 0.0),
            "max_world_collision_score": float(np.max(scores["world"])),
            "max_self_collision_score": float(np.max(scores["self"])),
        }

    def attach(self, world):
        """Switch the planned target from world mesh to TCP-frame spheres."""
        if self.world.target_state is not AttachmentState.WORLD or self.collision.attached:
            raise RuntimeError("planner target is already attached")
        if not isinstance(world, WorldSnapshot) or world.target_state is not AttachmentState.ATTACHED:
            raise ValueError("attached WorldSnapshot required")
        if world.config != self.world.config:
            raise ValueError("attachment must preserve the frozen production world")
        from curobo.geom.types import WorldConfig as CuroboWorldConfig

        packed = pack_attached_spheres(world.attached_spheres, self.asset["attachment_slots"])
        self.motion_gen.attach_spheres_to_robot(
            sphere_tensor=self.tensor_args.to_device(packed), link_name=self.asset["attachment_link"]
        )
        self.motion_gen.update_world(CuroboWorldConfig.from_dict(curobo_world_dict(world)))
        self.collision.attach(world)
        self.world = world

    def warmup(self):
        self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=True, warmup_joint_delta=0.1)

    def plan(self, start_q=SAFE_Q, goal_q=GOAL_Q):
        from curobo.types.state import JointState
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

        start_q = np.asarray(start_q, dtype=np.float64)
        goal_q = np.asarray(goal_q, dtype=np.float64)
        if start_q.shape != (6,) or goal_q.shape != (6,) or not np.isfinite((start_q, goal_q)).all():
            raise ValueError("finite six-axis start and goal are required")
        endpoint = self.endpoint_checks(start_q, goal_q)
        if not all(endpoint[key] for key in ("within_position_limits", "world_collision_free", "self_collision_free")):
            raise ValueError(f"invalid fixed planning endpoints: {endpoint}")
        start = JointState.from_position(
            self.tensor_args.to_device(start_q.astype(np.float32)).unsqueeze(0), joint_names=list(JOINT_NAMES)
        )
        goal = JointState.from_position(
            self.tensor_args.to_device(goal_q.astype(np.float32)).unsqueeze(0), joint_names=list(JOINT_NAMES)
        )
        result = self.motion_gen.plan_single_js(
            start,
            goal,
            MotionGenPlanConfig(
                max_attempts=self.config.max_attempts,
                timeout=self.config.timeout_s,
                enable_graph=self.config.graph_first,
                enable_graph_attempt=1 if self.config.graph_first else 3,
                check_start_validity=True,
            ),
        )
        success = bool(result.success is not None and result.success.item())
        status = str(result.status)
        if not success or result.interpolated_plan is None:
            raise RuntimeError(f"MotionGen failed: status={status}, valid_query={result.valid_query}")
        path = result.get_interpolated_plan()
        arrays = []
        for name in ("position", "velocity", "acceleration"):
            value = getattr(path, name)
            if value is None:
                raise RuntimeError(f"MotionGen returned no interpolated {name}")
            arrays.append(value.detach().cpu().numpy().astype(np.float64))
        q, dq, ddq = arrays
        if q.ndim == 3 and q.shape[0] == 1:
            q, dq, ddq = (value[0] for value in arrays)
        if q.ndim != 2:
            raise RuntimeError(f"unexpected interpolated trajectory shape: {q.shape}")
        dt = float(result.interpolation_dt)
        if not math.isclose(dt, 1.0 / self.config.control_hz, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(f"unexpected MotionGen interpolation dt: {dt}")
        trajectory = TrajectoryData(np.arange(len(q), dtype=np.float64) * dt, q, dq, ddq, JOINT_NAMES)
        metadata = {
            "success": success,
            "status": status,
            "valid_query": bool(result.valid_query),
            "attempts": int(result.attempts),
            "used_graph": bool(result.used_graph),
            "solve_time_s": float(result.solve_time),
            "total_time_s": float(result.total_time),
            "trajopt_time_s": float(result.trajopt_time),
            "finetune_time_s": float(result.finetune_time),
            "interpolation_dt_s": dt,
            "path_buffer_last_tstep": (
                [int(value.item()) if hasattr(value, "item") else int(value) for value in result.path_buffer_last_tstep]
                if result.path_buffer_last_tstep is not None
                else None
            ),
            "endpoint_checks": endpoint,
        }
        return trajectory, metadata
