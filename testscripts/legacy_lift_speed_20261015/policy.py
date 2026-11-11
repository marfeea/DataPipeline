"""Experimental fixed-clock, bounded TCP displacement policy."""
import numpy as np


def bounded_lift_delta(current, goal, limit_m=0.05):
    delta = np.asarray(goal, dtype=float) - np.asarray(current, dtype=float)
    if delta.shape != (3,) or not np.isfinite(delta).all() or not np.isfinite(limit_m) or limit_m <= 0:
        raise ValueError("finite 3D positions and a positive displacement limit are required")
    distance = float(np.linalg.norm(delta))
    return delta * min(1.0, limit_m / max(distance, 1e-12))
