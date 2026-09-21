"""Numerical replay comparison, independent of Isaac and rendering."""

from __future__ import annotations

import numpy as np


def replay_error(reference, actual):
    """Compare joint/gripper states and TCP pose; input quaternions use WXYZ."""
    joint = np.asarray(actual["q"]) - np.asarray(reference["q"])
    gripper = np.asarray(actual["gripper_position"]) - np.asarray(reference["gripper_position"])
    first = np.asarray(reference["tcp_pose_wxyz"], dtype=float)
    second = np.asarray(actual["tcp_pose_wxyz"], dtype=float)
    if not all(np.isfinite(value).all() for value in (joint, gripper, first, second)):
        raise ValueError("replay states must be finite")
    if first.shape != (7,) or second.shape != (7,):
        raise ValueError("replay TCP poses must have shape [7]")
    norms = [np.linalg.norm(first[3:]), np.linalg.norm(second[3:])]
    if not np.allclose(norms, 1, atol=1e-5):
        raise ValueError("replay TCP quaternions must have unit norm")
    qa, qb = first[3:] / norms[0], second[3:] / norms[1]
    return {
        "joint_max_abs_rad": float(np.max(np.abs(joint))),
        "gripper_max_abs_m": float(np.max(np.abs(gripper))),
        "tcp_position_error_m": float(np.linalg.norm(second[:3] - first[:3])),
        "tcp_rotation_error_rad": float(2 * np.arccos(np.clip(abs(np.dot(qa, qb)), 0, 1))),
    }


def summarize_replay(errors):
    """Summarize per-frame error magnitudes without inventing a pass tolerance."""
    if not errors:
        return {"compared_frames": 0, "metrics": {}}
    return {
        "compared_frames": len(errors),
        "metrics": {
            key: {
                "max": float(np.max([row[key] for row in errors])),
                "rms": float(np.sqrt(np.mean(np.square([row[key] for row in errors])))),
                "final": float(errors[-1][key]),
            }
            for key in errors[0]
        },
    }
