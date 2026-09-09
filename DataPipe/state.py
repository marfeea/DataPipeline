"""Explicit AUBO physical observation and action boundary helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite, sqrt
from typing import Any

from .config import (
    ACTION_DIMENSION,
    ARM_ACTION_DIMENSION,
    NORMALIZED_ACTION_HIGH,
    NORMALIZED_ACTION_LOW,
    TCP_HEADING_SCALE_RAD,
    TCP_POSITION_SCALE_M,
    TCP_ROTATION_SCALE_RAD,
)


def _vector(value: Sequence[float], *, name: str, size: int) -> tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if len(result) != size or not all(isfinite(item) for item in result):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return result


def validate_normalized_action(action: Sequence[float]) -> tuple[float, ...]:
    """Validate normalized ``[dx,dy,dz,rx,ry,rz,tcp_heading,gripper]``."""

    result = _vector(action, name="action", size=ACTION_DIMENSION)
    if any(item < NORMALIZED_ACTION_LOW or item > NORMALIZED_ACTION_HIGH for item in result):
        raise ValueError("action must be within [-1, 1]")
    return result


def normalized_action_to_tcp_delta(action: Sequence[float]) -> tuple[float, ...]:
    """Scale arm increments to SI units and preserve the unitless gripper command."""

    result = validate_normalized_action(action)
    scales = (TCP_POSITION_SCALE_M,) * 3 + (TCP_ROTATION_SCALE_RAD,) * 3 + (TCP_HEADING_SCALE_RAD, 1.0)
    return tuple(item * scale for item, scale in zip(result, scales, strict=True))


def tcp_delta_to_normalized_action(delta: Sequence[float]) -> tuple[float, ...]:
    """Map SI TCP increments to the normalized policy action and reject clipping."""

    result = _vector(delta, name="tcp_delta", size=ACTION_DIMENSION)
    scales = (TCP_POSITION_SCALE_M,) * 3 + (TCP_ROTATION_SCALE_RAD,) * 3 + (TCP_HEADING_SCALE_RAD, 1.0)
    normalized = tuple(item / scale for item, scale in zip(result, scales, strict=True))
    return validate_normalized_action(normalized)


def validate_explicit_observation(observation: Mapping[str, Any]) -> None:
    """Reject opaque observations and validate common AUBO state invariants."""

    if not observation:
        raise ValueError("observation must contain explicit physical fields")
    for key in ("q", "dq"):
        if key in observation:
            _vector(observation[key], name=key, size=ARM_ACTION_DIMENSION)
    for key in ("gripper_position", "gripper_velocity"):
        if key in observation:
            _vector(observation[key], name=key, size=2)
    for key, value in observation.items():
        if not key.endswith("pose") and not key.endswith("pose_wxyz"):
            continue
        pose = _vector(value, name=key, size=7)
        quaternion = pose[3:]
        norm = sqrt(sum(item * item for item in quaternion))
        if norm < 1.0e-6:
            raise ValueError(f"{key} quaternion has zero norm")


__all__ = [
    "normalized_action_to_tcp_delta",
    "tcp_delta_to_normalized_action",
    "validate_explicit_observation",
    "validate_normalized_action",
]
