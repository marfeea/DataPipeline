"""Derive contact TCP speed from contiguous CuRobo trajectory intervals."""
import hashlib
from pathlib import Path

import numpy as np

from scripts.native.aubo_ik import forward_kinematics

from .planner import JOINT_NAMES


def curobo_tcp_speed_reference(path):
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as data:
        q = np.asarray(data["joint_position"], dtype=float)
        time = np.asarray(data["time_s"], dtype=float)
        names = tuple(data["joint_names"].tolist())
        source = np.asarray(data["segment_source"]) if "segment_source" in data else np.full(len(time), "curobo")
        stage = np.asarray(data["stage_id"]) if "stage_id" in data else np.full(len(time), "curobo")
    if names != JOINT_NAMES or q.shape != (len(time), 6) or len(time) < 2:
        raise ValueError("speed reference must contain a six-axis trajectory in the project joint order")
    if not np.isfinite(q).all() or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("speed reference must be finite with strictly increasing time")
    if source.shape != time.shape or stage.shape != time.shape:
        raise ValueError("speed reference source/stage shapes differ from time")
    intervals = (source[:-1] == "curobo") & (source[1:] == "curobo") & (stage[:-1] == stage[1:])
    if not intervals.any():
        raise ValueError("speed reference has no contiguous CuRobo intervals")
    position = np.asarray([forward_kinematics(value)[:3, 3] for value in q])
    speeds = (np.linalg.norm(np.diff(position, axis=0), axis=1) / np.diff(time))[intervals]
    median = float(np.median(speeds))
    if not np.isfinite(median) or median <= 0:
        raise ValueError("CuRobo median TCP speed must be finite and positive")
    return dict(trajectory=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                method="median of TCP FK displacement / dt within CuRobo segments; segment seams excluded",
                interval_count=len(speeds), median_m_s=median, max_m_s=float(np.max(speeds)))


def tcp_step_toward(current, goal, *, speed_m_s, control_hz, step_limit_m):
    if any(not np.isfinite(value) or value <= 0 for value in (speed_m_s, control_hz, step_limit_m)):
        raise ValueError("speed, control frequency, and step limit must be finite and positive")
    delta = np.asarray(goal, dtype=float) - np.asarray(current, dtype=float)
    if delta.shape != (3,) or not np.isfinite(delta).all():
        raise ValueError("finite three-axis TCP positions are required")
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-12:
        return np.zeros(3)
    scale = min(1.0, min(speed_m_s / control_hz, step_limit_m) / distance)
    return delta * scale


def tracked_tcp_target(previous, current, goal, *, speed_m_s, control_hz, step_limit_m, max_lead_m=0.02):
    """Advance the command clock independently of servo lag, with bounded tracking lead."""
    if not np.isfinite(max_lead_m) or max_lead_m <= 0:
        raise ValueError("tracking lead must be finite and positive")
    candidate = np.asarray(previous, dtype=float) + tcp_step_toward(
        previous, goal, speed_m_s=speed_m_s, control_hz=control_hz, step_limit_m=step_limit_m
    )
    lead = candidate - np.asarray(current, dtype=float)
    distance = float(np.linalg.norm(lead))
    return np.asarray(current, dtype=float) + lead * min(1.0, max_lead_m / max(distance, 1e-12))
