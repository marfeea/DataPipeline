"""SI/XYZW geometry for the versioned laboratory trajectory contract.

All transforms are target-from-source. Twists are velocities of the TCP origin,
expressed in base/world axes, not spatial twists about those frames' origins.
"""

from __future__ import annotations

import numpy as np


def pose_matrix(pose):
    """Convert XYZ + unit XYZW to a rigid transform; reject corrupt rotations."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("pose must be finite XYZ + XYZW")
    q = pose[3:]
    if not np.isclose(np.linalg.norm(q), 1.0, atol=1e-5):
        raise ValueError("pose quaternion must have unit norm")
    x, y, z, w = q / np.linalg.norm(q)
    result = np.eye(4)
    result[:3, :3] = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    result[:3, 3] = pose[:3]
    return result


def rigid_transform(value):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError("T_w_b must be a finite 4x4 rigid transform")
    rotation = result[:3, :3]
    if (
        not np.allclose(result[3], (0, 0, 0, 1), atol=1e-6)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        raise ValueError("T_w_b must be a proper rigid transform")
    return result


def matrix_pose(value):
    """Stable conversion including rotations close to pi (XYZW order)."""
    value = rigid_transform(value)
    r = value[:3, :3]
    # Symmetric eigenproblem avoids division by a small scalar at 180 degrees.
    k = (
        np.array(
            [
                [r[0, 0] - r[1, 1] - r[2, 2], r[1, 0] + r[0, 1], r[2, 0] + r[0, 2], r[2, 1] - r[1, 2]],
                [r[1, 0] + r[0, 1], r[1, 1] - r[0, 0] - r[2, 2], r[2, 1] + r[1, 2], r[0, 2] - r[2, 0]],
                [r[2, 0] + r[0, 2], r[2, 1] + r[1, 2], r[2, 2] - r[0, 0] - r[1, 1], r[1, 0] - r[0, 1]],
                [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], np.trace(r)],
            ]
        )
        / 3
    )
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1]
    if q[3] < 0:
        q = -q
    return np.r_[value[:3, 3], q]


def transform_pose(transform, pose):
    return matrix_pose(rigid_transform(transform) @ pose_matrix(pose))


def rotation_vector(rotation):
    transform = np.eye(4)
    transform[:3, :3] = rotation
    q = matrix_pose(transform)[3:]
    sine = np.linalg.norm(q[:3])
    return 2 * q[:3] if sine < 1e-10 else q[:3] * (2 * np.arctan2(sine, q[3]) / sine)


def pose_displacement(before, after):
    """Measured translation and rotation-vector increment, both in base axes."""
    first, last = pose_matrix(before), pose_matrix(after)
    return np.r_[last[:3, 3] - first[:3, 3], rotation_vector(last[:3, :3] @ first[:3, :3].T)]
