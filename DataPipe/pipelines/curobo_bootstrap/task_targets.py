"""Shared production pick-place targets for the CuRobo V2 MVP.

This module is deliberately free of Isaac, Torch and CuRobo imports.  The
Legacy IK orientation helpers are loaded lazily only when full TCP poses are
requested.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import atan2, isfinite
from typing import Any

import numpy as np

from configs.task import TARGET_STATES, TargetState
from DataPipe.source2_routing import (
    Source2RoutingConfig,
    source2_advance_waypoints,
    source2_pre_pick,
    source2_routing_payload,
)
from DataPipe.state4_routing import State4RoutingConfig, routing_payload, state4_waypoints

from .contracts import config_hash
from .robot_adapter import pose_matrix, rigid_transform

SOURCE_INDEX = 0
DESTINATION_INDEX = 1
CONTROL_HZ = 60.0
SEED = 7
LIFT_CLEARANCE_M = 0.40
SCENE_ID = "CurriculumRL-TcpDocking-IK-v0:production-reset"


@dataclass(frozen=True)
class PickPlaceV2Config:
    source_index: int = SOURCE_INDEX
    destination_index: int = DESTINATION_INDEX
    control_hz: float = CONTROL_HZ
    seed: int = SEED
    lift_clearance_m: float = LIFT_CLEARANCE_M
    scene_id: str = SCENE_ID
    state4_routing: State4RoutingConfig | None = None
    source2_routing: Source2RoutingConfig | None = None

    def __post_init__(self):
        count = len(TARGET_STATES)
        if type(self.source_index) is not int or not 0 <= self.source_index < count:
            raise ValueError(f"source_index must be in [0, {count - 1}]")
        if type(self.destination_index) is not int or not 0 <= self.destination_index < count:
            raise ValueError(f"destination_index must be in [0, {count - 1}]")
        if self.source_index == self.destination_index:
            raise ValueError("source and destination must differ")
        if self.source2_routing is not None and self.source_index != 2:
            raise ValueError("source2 routing requires source_index=2")
        if not isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be finite and positive")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isfinite(self.lift_clearance_m) or self.lift_clearance_m <= 0:
            raise ValueError("lift_clearance_m must be finite and positive")
        if not isinstance(self.scene_id, str) or not self.scene_id.strip():
            raise ValueError("scene_id must be nonempty")


def _finite_pose(value: Any, name: str) -> tuple[float, ...]:
    pose = tuple(float(item) for item in value)
    if len(pose) != 7 or not all(isfinite(item) for item in pose):
        raise ValueError(f"{name} must be a finite xyz+wxyz pose")
    return pose


def _finite_position(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 3D position")
    return result


def environment_position_to_base(position_e, *, environment_origin_w, base_pose_w) -> np.ndarray:
    """Apply the Legacy ``E -> W -> B`` position transform without Isaac."""
    position_e = _finite_position(position_e, "position_e")
    origin_w = _finite_position(environment_origin_w, "environment_origin_w")
    base_pose = _finite_pose(base_pose_w, "base_pose_w")
    base = pose_matrix(base_pose[:3], base_pose[3:])
    return (np.linalg.inv(base) @ np.append(origin_w + position_e, 1.0))[:3]


def pick_place_positions_b(
    *,
    environment_origin_w,
    base_pose_w,
    source_index: int,
    destination_index: int,
    lift_clearance_m: float,
    state4_routing: State4RoutingConfig | None = None,
    source2_routing: Source2RoutingConfig | None = None,
) -> dict[str, np.ndarray]:
    """Return the five V2 waypoint positions from ``TARGET_STATES`` only."""
    config = PickPlaceV2Config(
        source_index=source_index,
        destination_index=destination_index,
        lift_clearance_m=lift_clearance_m,
        state4_routing=state4_routing,
        source2_routing=source2_routing,
    )
    source = TARGET_STATES[config.source_index]
    destination = TARGET_STATES[config.destination_index]
    convert = lambda value: environment_position_to_base(  # noqa: E731
        value, environment_origin_w=environment_origin_w, base_pose_w=base_pose_w
    )
    pre_pick = convert(source.preposition_e)
    pick = convert(source.grasp_tcp_position_e)
    pre_pick = np.asarray(
        source2_pre_pick(source_index=source_index, pre_pick=pre_pick, pick=pick, config=config.source2_routing)
    )
    place = convert(destination.grasp_tcp_position_e)
    safe_z = max(float(pick[2]), float(place[2])) + config.lift_clearance_m
    lift = pick.copy()
    lift[2] = safe_z
    place_high = place.copy()
    place_high[2] = safe_z
    positions = {
        "pre_pick": pre_pick,
        "pick": pick,
        "lift": lift,
        "place_high": place_high,
        "place": place,
    }
    if config.state4_routing is not None:
        routed, _ = state4_waypoints(
            source_index=source_index,
            destination_index=destination_index,
            pre_pick=pre_pick,
            pick=pick,
            lift=lift,
            place_high=place_high,
            place=place,
            place_preposition=convert(destination.preposition_e),
            config=config.state4_routing,
        )
        positions.update({name: np.asarray(value) for name, value in routed.items()})
    return positions


def _target_state_payload(state: TargetState) -> dict[str, Any]:
    return asdict(state)


def task_snapshot(
    *,
    config: PickPlaceV2Config,
    environment_origin_w,
    base_pose_w,
    positions_b: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the stable task/transform portion of ``scene_snapshot.json``."""
    origin = _finite_position(environment_origin_w, "environment_origin_w")
    base_pose = _finite_pose(base_pose_w, "base_pose_w")
    expected = pick_place_positions_b(
        environment_origin_w=origin,
        base_pose_w=base_pose,
        source_index=config.source_index,
        destination_index=config.destination_index,
        lift_clearance_m=config.lift_clearance_m,
        state4_routing=config.state4_routing,
        source2_routing=config.source2_routing,
    )
    if positions_b is not None:
        if set(positions_b) != set(expected):
            raise ValueError("positions_b fields differ from the five-waypoint contract")
        for name, value in positions_b.items():
            if not np.allclose(_finite_position(value, name), expected[name], rtol=0.0, atol=1e-12):
                raise ValueError(f"{name} differs from TARGET_STATES geometry")
    world_from_environment = pose_matrix(origin, (1.0, 0.0, 0.0, 0.0))
    world_from_base = pose_matrix(base_pose[:3], base_pose[3:])
    payload = {
        "schema": "curobo_pick_place_v2_scene_snapshot/1",
        "scene_id": config.scene_id,
        "source_index": config.source_index,
        "destination_index": config.destination_index,
        "source": _target_state_payload(TARGET_STATES[config.source_index]),
        "destination": _target_state_payload(TARGET_STATES[config.destination_index]),
        "control_hz": config.control_hz,
        "seed": config.seed,
        "lift_clearance_m": config.lift_clearance_m,
        "frames": {
            "pose_convention": "xyz+wxyz",
            "T_WE": world_from_environment.tolist(),
            "T_WB": world_from_base.tolist(),
        },
        "waypoint_position_b": {name: value.tolist() for name, value in expected.items()},
    }
    if config.state4_routing is not None:
        convert = lambda value: environment_position_to_base(  # noqa: E731
            value, environment_origin_w=origin, base_pose_w=base_pose
        )
        original = pick_place_positions_b(
            environment_origin_w=origin,
            base_pose_w=base_pose,
            source_index=config.source_index,
            destination_index=config.destination_index,
            lift_clearance_m=config.lift_clearance_m,
        )
        _, contact = state4_waypoints(
            source_index=config.source_index,
            destination_index=config.destination_index,
            pre_pick=original["pre_pick"],
            pick=original["pick"],
            lift=original["lift"],
            place_high=original["place_high"],
            place=original["place"],
            place_preposition=convert(TARGET_STATES[config.destination_index].preposition_e),
            config=config.state4_routing,
        )
        payload["state4_routing"] = routing_payload(config.state4_routing)
        payload["contact_waypoint_position_b"] = {name: list(value) for name, value in contact.items()}
    if config.source2_routing is not None:
        payload["source2_routing"] = source2_routing_payload(config.source2_routing)
        advance = source2_advance_waypoints(pick=expected["pick"], config=config.source2_routing)
        if len(advance) > 1:
            payload.setdefault("contact_waypoint_position_b", {})["pick_entry"] = list(advance[0])
    payload["task_hash"] = config_hash(payload)
    return payload


def legacy_tcp_poses_b(snapshot: dict[str, Any]) -> dict[str, tuple[float, ...]]:
    """Add the exact Legacy flange-down heading orientation to five positions."""

    from configs.coordinate_frames import TCP_DIRECTION_AXIS_T, target_direction_in_world  # noqa: PLC0415
    from DataPipe.native.aubo_ik import (  # noqa: PLC0415
        flange_down_tcp_target_pose,
        flange_down_yaw_offset_for_tcp_heading,
    )

    transform_wb = np.asarray(snapshot["frames"]["T_WB"], dtype=np.float64)
    base_pose = _finite_pose(
        (*transform_wb[:3, 3], *_rotation_wxyz_from_matrix(transform_wb)),
        "base_pose_w",
    )
    root_rotation = transform_wb[:3, :3]
    source_direction_b = root_rotation.T @ np.asarray(
        target_direction_in_world(snapshot["source"]["rotation_wxyz"]), dtype=np.float64
    )
    destination_direction_b = root_rotation.T @ np.asarray(
        target_direction_in_world(snapshot["destination"]["rotation_wxyz"]), dtype=np.float64
    )
    headings = {
        "pre_pick": atan2(source_direction_b[1], source_direction_b[0]),
        "pick": atan2(source_direction_b[1], source_direction_b[0]),
        "lift": atan2(source_direction_b[1], source_direction_b[0]),
        "place_high": atan2(destination_direction_b[1], destination_direction_b[0]),
        "place": atan2(destination_direction_b[1], destination_direction_b[0]),
    }
    result = {}
    for name, position in snapshot["waypoint_position_b"].items():
        offset = flange_down_yaw_offset_for_tcp_heading(
            position,
            headings[name],
            root_quaternion_wxyz=base_pose[3:],
            tcp_heading_axis_local=TCP_DIRECTION_AXIS_T,
        )
        _, quaternion = flange_down_tcp_target_pose(
            position,
            root_quaternion_wxyz=base_pose[3:],
            downward_yaw_offset_rad=offset,
        )
        result[name] = tuple(float(value) for value in (*position, *quaternion))
    return result


def solve_legacy_joint_targets(snapshot: dict[str, Any], q_start, *, random_seed_count: int = 16) -> dict[str, Any]:
    """Audit waypoint feasibility with bounded retries, not executable connections."""

    from DataPipe.native.aubo_ik import (  # noqa: PLC0415
        JOINT_LIMITS_RAD,
        forward_pose,
        solve_tcp_ik,
    )

    if type(random_seed_count) is not int or random_seed_count < 0:
        raise ValueError("random_seed_count must be a nonnegative integer")
    q_seed = np.asarray(q_start, dtype=np.float64).copy()
    if q_seed.shape != (6,) or not np.isfinite(q_seed).all():
        raise ValueError("q_start must contain six finite joint positions")
    lower, upper = JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1]
    if np.any(q_seed < lower) or np.any(q_seed > upper):
        raise ValueError("q_start violates Legacy IK joint limits")
    q_reset = q_seed.copy()
    poses = legacy_tcp_poses_b(snapshot)

    def finite_number(value):
        number = float(value)
        return number if np.isfinite(number) else None

    def audit_result(result, pose, seed, source):
        record = {
            "seed_source": source,
            "seed_q": seed.tolist(),
            "success": bool(result.success),
            "reason": result.reason,
            "iterations": int(result.iterations),
            "position_error_m": finite_number(result.position_error_m),
            "orientation_error_rad": finite_number(result.orientation_error_rad),
            "joint_position": None,
            "finite": False,
            "within_joint_limits": False,
            "fk_position_error_m": None,
            "fk_orientation_error_rad": None,
        }
        if result.success and result.q is not None:
            q = np.asarray(result.q, dtype=np.float64)
            record["finite"] = bool(q.shape == (6,) and np.isfinite(q).all())
            if record["finite"]:
                record["joint_position"] = q.tolist()
                record["within_joint_limits"] = bool(np.all(q >= lower) and np.all(q <= upper))
                if record["within_joint_limits"]:
                    current_position, current_rotation = forward_pose(q)
                    target = pose_matrix(pose[:3], pose[3:])
                    relative = target[:3, :3].T @ current_rotation
                    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
                    record.update(
                        fk_position_error_m=finite_number(np.linalg.norm(current_position - target[:3, 3])),
                        fk_orientation_error_rad=finite_number(np.arccos(cosine)),
                    )
        record["passed"] = bool(
            record["success"]
            and record["finite"]
            and record["within_joint_limits"]
            and record["fk_position_error_m"] is not None
            and record["fk_orientation_error_rad"] is not None
            and record["fk_position_error_m"] <= 1e-5
            and record["fk_orientation_error_rad"] <= 1e-4
        )
        if record["success"] and not record["passed"]:
            record["reason"] = "IK solution failed finite, joint-limit or FK audit"
        return record

    targets = {}
    for waypoint_index, (name, pose) in enumerate(poses.items()):
        rng = np.random.default_rng(np.random.SeedSequence([snapshot["seed"], waypoint_index]))

        def seeds():
            yield ("reset" if waypoint_index == 0 else "previous_target"), q_seed
            yield "reset", q_reset
            yield "zero", np.zeros(6, dtype=np.float64)
            for index in range(random_seed_count):
                yield f"random_{index:02d}", rng.uniform(lower, upper)

        attempts = []
        tried = []
        selected_attempt_index = None
        for source, seed in seeds():
            if any(np.allclose(seed, prior, rtol=0.0, atol=1e-12) for prior in tried):
                continue
            tried.append(seed.copy())
            result = solve_tcp_ik(pose[:3], pose[3:], seed=seed.copy())
            attempt = audit_result(result, pose, seed, source)
            attempts.append(attempt)
            if attempt["passed"]:
                selected_attempt_index = len(attempts) - 1
                q_seed = np.asarray(attempt["joint_position"], dtype=np.float64)
                break
        record = dict(attempts[-1])
        record.update(
            tcp_pose_b=list(pose),
            attempts=attempts,
            selected_attempt_index=selected_attempt_index,
        )
        targets[name] = record
        if not record["passed"]:
            break
    return {
        "passed": tuple(targets) == tuple(poses) and all(item["passed"] for item in targets.values()),
        "q_start": q_reset.tolist(),
        "search_policy": {
            "deterministic_seed": snapshot["seed"],
            "random_seed_count_per_waypoint": random_seed_count,
            "seed_order": ["previous_target_or_reset", "reset", "zero", "random"],
            "scope": "endpoint feasibility only; execution requires independent branch and path validation",
        },
        "joint_limits_rad": {"lower": lower.tolist(), "upper": upper.tolist()},
        "targets": targets,
    }


def _rotation_wxyz_from_matrix(value) -> tuple[float, float, float, float]:
    """Convert the validated rotation of a rigid matrix to canonical wxyz."""
    rotation = rigid_transform(np.asarray(value, dtype=np.float64))[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
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
            scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            quaternion = np.array(
                (
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                )
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            quaternion = np.array(
                (
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                )
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            quaternion = np.array(
                (
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                )
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0:
        quaternion *= -1
    return tuple(float(item) for item in quaternion)


__all__ = [
    "CONTROL_HZ",
    "DESTINATION_INDEX",
    "LIFT_CLEARANCE_M",
    "PickPlaceV2Config",
    "SCENE_ID",
    "SEED",
    "SOURCE_INDEX",
    "environment_position_to_base",
    "legacy_tcp_poses_b",
    "pick_place_positions_b",
    "solve_legacy_joint_targets",
    "task_snapshot",
]
