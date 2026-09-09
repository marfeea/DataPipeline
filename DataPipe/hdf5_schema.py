"""Formal HDF5 schema for the AUBO IK data pipeline.

The writer stores observations on the ``T+1`` axis and transitions on the
``T`` axis.  This module deliberately contains no Isaac Lab or training-side
dependencies so the file contract can be checked offline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from .config import ACTION_DIMENSION, TCP_HEADING_SCALE_RAD
from .records import Episode
from .sequence import LEGACY_SEQUENCE_STAGE_ORDER_V3, SEQUENCE_STAGE_ORDER, SequenceStage
from .state import validate_explicit_observation

OBSERVATION_GROUP = "obs"
TRANSITION_FIELDS = (
    "actions",
    "rewards",
    "dones",
    "timeouts",
    "is_terminal",
    "phases",
    "waypoint_indices",
)
DIAGNOSTIC_FIELDS = (
    "raw_action",
    "tcp_displacement",
    "ik_output",
    "commanded_joint_target",
    "joint_position",
    "joint_velocity",
    "gripper_command",
    "tcp_heading_target_rad",
    "tcp_heading_actual_rad",
    "tcp_heading_first_five_rad",
    "tcp_heading_sixth_axis_rad",
    "tcp_heading_error_rad",
    "gripper_joint_position",
    "gripper_joint_velocity",
)
QUATERNION_POSE_WIDTH = 7
REQUIRED_OBSERVATION_FIELDS = frozenset(
    {
        "q",
        "dq",
        "tcp_pose_wxyz",
        "target_pose_wxyz",
        "gripper_position",
        "gripper_velocity",
        "semantic_stage_id",
        "waypoint_index",
        "preposition_position_b",
        "object_position_b",
        "desired_approach_axis_b",
        "advance_target_position_b",
    }
)


@dataclass(frozen=True, slots=True)
class HDF5FieldSpec:
    """Static description of one field in the formal file."""

    name: str
    axis: str
    dtype: str
    unit: str
    frame: str


def wxyz_to_xyzw(value: Sequence[float]) -> tuple[float, ...]:
    """Convert ``position + WXYZ`` to the dataset's ``position + XYZW``."""

    result = tuple(float(item) for item in value)
    if len(result) != QUATERNION_POSE_WIDTH:
        raise ValueError("pose must contain XYZ plus four quaternion values")
    if not all(isfinite(item) for item in result):
        raise ValueError("pose cannot contain NaN or Inf")
    return result[:3] + (result[4], result[5], result[6], result[3])


def xyzw_to_wxyz(value: Sequence[float]) -> tuple[float, ...]:
    """Convert ``position + XYZW`` from disk back to internal ``WXYZ``."""

    result = tuple(float(item) for item in value)
    if len(result) != QUATERNION_POSE_WIDTH:
        raise ValueError("pose must contain XYZ plus four quaternion values")
    if not all(isfinite(item) for item in result):
        raise ValueError("pose cannot contain NaN or Inf")
    return result[:3] + (result[6], result[3], result[4], result[5])


def _as_array(value: Any, *, name: str) -> np.ndarray:
    """Convert a field to a finite ndarray without silently object-encoding it."""

    if isinstance(value, str | bytes | bytearray) or value is None:
        raise TypeError(f"observation field {name!r} must be numeric")
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise TypeError(f"observation field {name!r} must be numeric")
    if array.dtype.kind in "uf" and not np.isfinite(array).all():
        raise ValueError(f"observation field {name!r} contains NaN or Inf")
    return array


def _observation_values(episode: Episode, key: str) -> tuple[np.ndarray, ...]:
    values: list[np.ndarray] = []
    for frame in episode.observations:
        if not isinstance(frame.value, Mapping):
            raise TypeError("formal HDF5 export requires mapping observations, not opaque values")
        if key not in frame.value:
            raise KeyError(f"observation field {key!r} is missing at timestep {frame.timestep}")
        value = frame.value[key]
        if key.endswith("_wxyz"):
            value = wxyz_to_xyzw(value)
        values.append(_as_array(value, name=key))
    return tuple(values)


def _stack_observation(episode: Episode, key: str) -> np.ndarray:
    values = _observation_values(episode, key)
    first_shape = values[0].shape
    if any(value.shape != first_shape for value in values):
        raise ValueError(f"observation field {key!r} changes shape within an episode")
    return np.stack(values, axis=0).astype(np.float32 if values[0].dtype.kind == "f" else values[0].dtype)


def observation_keys(episode: Episode) -> tuple[str, ...]:
    """Return stable explicit observation keys shared by every frame."""

    if not episode.observations or not isinstance(episode.observations[0].value, Mapping):
        raise TypeError("formal HDF5 export requires mapping observations, not opaque values")
    for frame in episode.observations:
        if not isinstance(frame.value, Mapping):
            raise TypeError("formal HDF5 export requires mapping observations, not opaque values")
        validate_explicit_observation(frame.value)
    keys = tuple(str(key) for key in episode.observations[0].value)
    missing_required = REQUIRED_OBSERVATION_FIELDS.difference(keys)
    if missing_required:
        raise KeyError(f"required observation fields are missing: {sorted(missing_required)}")
    for frame in episode.observations[1:]:
        if not isinstance(frame.value, Mapping) or tuple(str(key) for key in frame.value) != keys:
            raise ValueError("all observation frames must contain the same ordered fields")
    return keys


def field_specs(episode: Episode) -> tuple[HDF5FieldSpec, ...]:
    """Describe the concrete fields present in an episode."""

    specs: list[HDF5FieldSpec] = []
    for key in observation_keys(episode):
        disk_key = key.removesuffix("_wxyz")
        unit = "unitless"
        frame = "task"
        if disk_key in {"q", "joint_position", "ik_output", "commanded_joint_target"}:
            unit, frame = "rad", "robot joint space"
        elif disk_key == "dq" or disk_key == "joint_velocity":
            unit, frame = "rad/s", "robot joint space"
        elif disk_key.endswith("pose"):
            unit, frame = "xyz m + xyzw", "world"
        specs.append(HDF5FieldSpec(disk_key, "T+1", "float32", unit, frame))
    return tuple(specs)


def _write_dataset(group: Any, name: str, values: np.ndarray, *, compress: bool = False) -> None:
    kwargs: dict[str, Any] = {}
    if compress and values.ndim >= 3:
        kwargs.update(compression="gzip", compression_opts=4, shuffle=True)
    group.create_dataset(name, data=values, **kwargs)


def write_episode_group(group: Any, episode: Episode, *, metadata: Mapping[str, Any] | None = None) -> None:
    """Write one validated episode into an open ``data/demo_*`` group."""

    episode.validate()
    if episode.length <= 0:
        raise ValueError("an HDF5 demo must contain at least one transition")
    observation_group = group.create_group(OBSERVATION_GROUP)
    for spec in field_specs(episode):
        source_key = spec.name + "_wxyz" if spec.name + "_wxyz" in observation_keys(episode) else spec.name
        compress = spec.name.startswith(("image_", "depth_", "foreground_mask"))
        _write_dataset(observation_group, spec.name, _stack_observation(episode, source_key), compress=compress)

    actions = np.asarray([step.action.action for step in episode.steps], dtype=np.float32)
    diagnostics = group.create_group("diagnostics")
    _write_dataset(group, "actions", actions)
    _write_dataset(group, "rewards", np.asarray([step.reward for step in episode.steps], dtype=np.float32))
    dones = np.zeros(episode.length, dtype=np.uint8)
    dones[-1] = 1
    timeouts = np.zeros(episode.length, dtype=np.uint8)
    timeouts[-1] = int(episode.termination_reason.value == "timeout")
    is_terminal = np.zeros(episode.length, dtype=np.uint8)
    is_terminal[-1] = int(episode.termination_reason.is_terminal)
    _write_dataset(group, "dones", dones)
    _write_dataset(group, "timeouts", timeouts)
    _write_dataset(group, "is_terminal", is_terminal)
    phases = np.asarray([step.action.semantic_stage_id for step in episode.steps], dtype=np.uint8)
    waypoint_indices = np.asarray([step.action.waypoint_index for step in episode.steps], dtype=np.uint16)
    _write_dataset(group, "phases", phases)
    _write_dataset(group, "waypoint_indices", waypoint_indices)

    diagnostic_values = {
        "raw_action": [step.action.raw_action for step in episode.steps],
        "tcp_displacement": [step.action.tcp_displacement for step in episode.steps],
        "ik_output": [step.action.ik_output for step in episode.steps],
        "waypoint_tcp_position_b": [step.action.waypoint_tcp_position_b for step in episode.steps],
        "controller_target_tcp_position_b": [step.action.controller_target_tcp_position_b for step in episode.steps],
        "ik_seed_joint_position": [step.action.ik_seed_joint_position for step in episode.steps],
        "ik_solution_joint_position": [step.action.ik_solution_joint_position for step in episode.steps],
        "limited_joint_target": [step.action.limited_joint_target for step in episode.steps],
        "commanded_joint_target": [step.action.commanded_joint_target for step in episode.steps],
        "actual_tcp_position_b": [step.action.actual_tcp_position_b for step in episode.steps],
        "ik_success": [step.action.ik_success for step in episode.steps],
        "delta_limited": [step.action.delta_limited for step in episode.steps],
        "velocity_limited": [step.action.velocity_limited for step in episode.steps],
        "position_limited": [step.action.position_limited for step in episode.steps],
        "joint_position": [step.action.joint_position for step in episode.steps],
        "joint_velocity": [step.action.joint_velocity for step in episode.steps],
        "gripper_command": [step.action.gripper_command for step in episode.steps],
        "tcp_heading_target_rad": [step.action.tcp_heading_target_rad for step in episode.steps],
        "tcp_heading_actual_rad": [step.action.tcp_heading_actual_rad for step in episode.steps],
        "tcp_heading_first_five_rad": [step.action.tcp_heading_first_five_rad for step in episode.steps],
        "tcp_heading_sixth_axis_rad": [step.action.tcp_heading_sixth_axis_rad for step in episode.steps],
        "tcp_heading_error_rad": [step.action.tcp_heading_error_rad for step in episode.steps],
        "gripper_joint_position": [step.action.gripper_joint_position for step in episode.steps],
        "gripper_joint_velocity": [step.action.gripper_joint_velocity for step in episode.steps],
    }
    for name, values in diagnostic_values.items():
        is_flag = name in {"ik_success", "delta_limited", "velocity_limited", "position_limited"}
        dtype = np.uint8 if is_flag else np.float32
        _write_dataset(diagnostics, name, np.asarray(values, dtype=dtype))

    component_names = tuple(sorted({name for step in episode.steps for name in step.reward_components}))
    if component_names:
        reward_components = diagnostics.create_group("reward_components")
        for name in component_names:
            values = [step.reward_components.get(name, 0.0) for step in episode.steps]
            _write_dataset(reward_components, name, np.asarray(values, dtype=np.float32))
        diagnostics.attrs["reward_component_names"] = ",".join(component_names)

    group.attrs["num_samples"] = episode.length
    group.attrs["successful"] = int(episode.termination_reason.value == "success")
    group.attrs["termination_reason"] = episode.termination_reason.value
    group.attrs["episode_id"] = episode.episode_id
    if metadata:
        for key, value in metadata.items():
            if value is not None:
                group.attrs[str(key)] = value


def validate_episode_group(group: Any) -> tuple[str, ...]:
    """Return precise schema errors for one HDF5 demo group."""

    errors: list[str] = []
    required = set(TRANSITION_FIELDS) | {OBSERVATION_GROUP, "diagnostics"}
    missing = sorted(required.difference(group.keys()))
    errors.extend(f"missing {name}" for name in missing)
    if errors:
        return tuple(errors)
    action_count = int(group["actions"].shape[0])
    for name in TRANSITION_FIELDS:
        if int(group[name].shape[0]) != action_count:
            errors.append(f"{name} has T={group[name].shape[0]}, expected {action_count}")
    if group["actions"].shape[1:] != (ACTION_DIMENSION,):
        errors.append(f"actions shape must be [T, {ACTION_DIMENSION}]")
    actions = np.asarray(group["actions"])
    if not np.isfinite(actions).all() or np.any(actions < -1.0) or np.any(actions > 1.0):
        errors.append("actions must be finite and normalized to [-1, 1]")
    phases = np.asarray(group["phases"])
    valid_stage_ids = np.asarray(sorted(stage.value for stage in SequenceStage), dtype=np.int16)
    phase_ids = phases.astype(np.int16)
    if phases.ndim != 1 or not np.isin(phase_ids, valid_stage_ids).all():
        errors.append(f"phases must be one-dimensional valid sequence stage IDs: {valid_stage_ids.tolist()}")
    else:
        # 旧 v3 采集器把 vertical_descent 的插值点标成 0、终点标成 1，
        # 因此路径阶段 0/1 可在进入正式序列前交替。后续阶段必须按显式
        # 语义顺序前进；兼容 ID 7/10/12/8/9 不能按整数大小判断。
        sequence_start = next(
            (index for index, phase in enumerate(phase_ids) if phase not in {0, 1}),
            len(phase_ids),
        )
        ordered_phase_ids = phase_ids[sequence_start:]
        def follows_order(stage_order: Sequence[SequenceStage]) -> bool:
            stage_rank = {stage: rank for rank, stage in enumerate(stage_order)}
            stage_rank[SequenceStage.HOLD_COMPLETE] = len(stage_order) - 1
            ranks = np.asarray(
                [stage_rank[SequenceStage(int(phase))] for phase in ordered_phase_ids],
                dtype=np.int16,
            )
            return not np.any(np.diff(ranks) < 0)

        if not any(follows_order(order) for order in (SEQUENCE_STAGE_ORDER, LEGACY_SEQUENCE_STAGE_ORDER_V3)):
            errors.append("phases must follow the semantic sequence order after path stages 0/1")
    waypoint_indices = np.asarray(group["waypoint_indices"])
    if np.any(np.diff(waypoint_indices.astype(np.int64)) < 0):
        errors.append("waypoint_indices must be monotonic")
    for name in group[OBSERVATION_GROUP]:
        if int(group[OBSERVATION_GROUP][name].shape[0]) != action_count + 1:
            errors.append(f"obs/{name} must have T+1 samples")
    required_disk_observations = {name.removesuffix("_wxyz") for name in REQUIRED_OBSERVATION_FIELDS}
    for name in sorted(required_disk_observations.difference(group[OBSERVATION_GROUP].keys())):
        errors.append(f"missing obs/{name}")
    dones = np.asarray(group["dones"])
    if not np.array_equal(dones, np.r_[np.zeros(max(action_count - 1, 0), dtype=np.uint8), 1]):
        errors.append("dones must be zero before the final transition and one at the final transition")
    timeouts = np.asarray(group["timeouts"])
    terminals = np.asarray(group["is_terminal"])
    if np.any(timeouts[:-1]) or np.any(terminals[:-1]) or not (timeouts[-1] or terminals[-1]):
        errors.append("timeouts and is_terminal may only be set on the final transition")
    if np.any(timeouts & terminals):
        errors.append("timeouts and is_terminal are mutually exclusive")
    if "reward_components" in group["diagnostics"]:
        for name in group["diagnostics"]["reward_components"]:
            if int(group["diagnostics"]["reward_components"][name].shape[0]) != action_count:
                errors.append(f"diagnostics/reward_components/{name} must have T samples")
    diagnostics = group["diagnostics"]
    for name in DIAGNOSTIC_FIELDS:
        if name not in diagnostics:
            errors.append(f"missing diagnostics/{name}")
        elif int(diagnostics[name].shape[0]) != action_count:
            errors.append(f"diagnostics/{name} must have T samples")
    if "gripper_command" in diagnostics and actions.shape[1:] == (ACTION_DIMENSION,):
        if not np.allclose(actions[:, -1], np.asarray(diagnostics["gripper_command"]), atol=1.0e-6):
            errors.append("actions gripper component must match diagnostics/gripper_command")
    if "tcp_heading_target_rad" in diagnostics and actions.shape[1:] == (ACTION_DIMENSION,):
        heading_targets = np.asarray(diagnostics["tcp_heading_target_rad"])
        if not np.allclose(actions[:, -2] * TCP_HEADING_SCALE_RAD, heading_targets, atol=1.0e-6):
            errors.append("actions TCP heading component must match diagnostics/tcp_heading_target_rad")
    return tuple(errors)


__all__ = [
    "DIAGNOSTIC_FIELDS",
    "HDF5FieldSpec",
    "OBSERVATION_GROUP",
    "REQUIRED_OBSERVATION_FIELDS",
    "TRANSITION_FIELDS",
    "field_specs",
    "observation_keys",
    "validate_episode_group",
    "wxyz_to_xyzw",
    "write_episode_group",
    "xyzw_to_wxyz",
]
