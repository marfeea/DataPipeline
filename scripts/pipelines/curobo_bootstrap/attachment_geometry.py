"""Build conservative TCP attachment geometry from measured Isaac poses."""

import numpy as np

from .robot_adapter import pose_matrix
from .world_adapter import attached_sphere_from_local_aabb


def _measured_pose(value):
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("measured pose must be finite xyz/wxyz")
    norm = np.linalg.norm(pose[3:])
    if abs(norm - 1.0) > 1e-3:
        raise ValueError("measured quaternion must be approximately unit length")
    return pose_matrix(pose[:3], pose[3:] / norm)


def attached_sphere_from_measured_target(
    vertices_base_at_reset,
    *,
    base_pose_w,
    object_pose_w_at_reset,
    object_pose_w_current,
    tcp_pose_b_current,
):
    """Move reset Base-frame mesh with observed object pose, then enclose it in TCP."""
    vertices = np.asarray(vertices_base_at_reset, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 2 or not np.isfinite(vertices).all():
        raise ValueError("target mesh requires finite Nx3 vertices")
    world_from_base = _measured_pose(base_pose_w)
    world_from_object_reset = _measured_pose(object_pose_w_at_reset)
    world_from_object_now = _measured_pose(object_pose_w_current)
    base_from_tcp_now = _measured_pose(tcp_pose_b_current)
    tcp_from_base_now = np.linalg.inv(base_from_tcp_now)
    base_from_world = np.linalg.inv(world_from_base)
    transform = (
        tcp_from_base_now
        @ base_from_world
        @ world_from_object_now
        @ np.linalg.inv(world_from_object_reset)
        @ world_from_base
    )
    vertices_tcp = (transform @ np.column_stack((vertices, np.ones(len(vertices)))).T).T[:, :3]
    return attached_sphere_from_local_aabb(vertices_tcp.min(axis=0), vertices_tcp.max(axis=0))


def root_vertices_in_base(vertices_local, *, root_pose_w, base_pose_w):
    """Place rigid-root local geometry using the measured live pose, not USD pose."""
    vertices = np.asarray(vertices_local, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("root mesh vertices must be finite Nx3")
    base_from_root = np.linalg.inv(_measured_pose(base_pose_w)) @ _measured_pose(root_pose_w)
    return (base_from_root @ np.column_stack((vertices, np.ones(len(vertices)))).T).T[:, :3]
