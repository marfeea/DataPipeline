"""Deterministic primitive world and explicit attach/detach transitions."""

from dataclasses import asdict, dataclass, replace
from enum import Enum
from math import isfinite

import numpy as np

from configs.curobo.world import CuboidConfig, WorldConfig

from .contracts import config_hash
from .robot_adapter import pose_matrix, rigid_transform


class AttachmentState(str, Enum):
    WORLD = "world"
    ATTACHED = "attached"


@dataclass(frozen=True)
class AttachedSphere:
    center_tcp: tuple[float, float, float]
    radius_m: float

    def __post_init__(self):
        center = tuple(float(value) for value in self.center_tcp)
        if len(center) != 3 or not all(isfinite(value) for value in center):
            raise ValueError("attached sphere center must be three finite TCP-frame values")
        if not isfinite(self.radius_m) or self.radius_m <= 0:
            raise ValueError("attached sphere radius must be finite and positive")
        object.__setattr__(self, "center_tcp", center)


def _rotation_wxyz(rotation):
    transform = np.eye(4)
    transform[:3, :3] = rotation
    rotation = rigid_transform(transform)[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = 2 * np.sqrt(trace + 1)
        quaternion = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = 2 * np.sqrt(1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            quaternion = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = 2 * np.sqrt(1 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            quaternion = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ]
            )
        else:
            scale = 2 * np.sqrt(1 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            quaternion = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0:
        quaternion *= -1
    return tuple(float(value) for value in quaternion)


def cuboid_from_world_aabb(name, minimum_w, maximum_w, base_pose_w):
    """Conservatively express a world-aligned Isaac bound as an oriented Base-frame box."""
    minimum = np.asarray(minimum_w, dtype=np.float64)
    maximum = np.asarray(maximum_w, dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,) or not np.isfinite((minimum, maximum)).all():
        raise ValueError("finite 3D AABB bounds required")
    if np.any(maximum <= minimum):
        raise ValueError("AABB maximum must exceed minimum")
    base_from_world = np.linalg.inv(rigid_transform(base_pose_w))
    center_w = np.append(0.5 * (minimum + maximum), 1.0)
    center_b = (base_from_world @ center_w)[:3]
    quaternion = _rotation_wxyz(base_from_world[:3, :3])
    return CuboidConfig(name, (*center_b, *quaternion), tuple(maximum - minimum))


def aabb_bounds_in_frame(minimum_w, maximum_w, frame_pose_w):
    """Transform all eight world AABB corners and return their aligned bounds in a frame."""
    minimum = np.asarray(minimum_w, dtype=np.float64)
    maximum = np.asarray(maximum_w, dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,) or not np.isfinite((minimum, maximum)).all():
        raise ValueError("finite 3D AABB bounds required")
    if np.any(maximum <= minimum):
        raise ValueError("AABB maximum must exceed minimum")
    corners_w = np.array(
        [
            [x, y, z, 1.0]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ]
    )
    corners_frame = (np.linalg.inv(rigid_transform(frame_pose_w)) @ corners_w.T).T[:, :3]
    return tuple(corners_frame.min(axis=0)), tuple(corners_frame.max(axis=0))


def attached_sphere_from_local_aabb(minimum, maximum, *, padding_m=0.0):
    """Return one conservative sphere containing an object's complete local AABB."""
    low = np.asarray(minimum, dtype=np.float64)
    high = np.asarray(maximum, dtype=np.float64)
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite((low, high)).all():
        raise ValueError("finite 3D AABB bounds required")
    if np.any(high <= low) or not isfinite(padding_m) or padding_m < 0:
        raise ValueError("positive AABB extents and nonnegative padding required")
    return AttachedSphere(tuple(0.5 * (low + high)), float(0.5 * np.linalg.norm(high - low) + padding_m))


@dataclass(frozen=True)
class WorldSnapshot:
    config: WorldConfig
    target_state: AttachmentState = AttachmentState.WORLD
    attached_spheres: tuple[AttachedSphere, ...] = ()
    target_contact_allowed: bool = False

    def __post_init__(self):
        if not isinstance(self.config, WorldConfig) or not isinstance(self.target_state, AttachmentState):
            raise ValueError("typed world configuration and attachment state required")
        if not isinstance(self.target_contact_allowed, bool):
            raise ValueError("target_contact_allowed must be boolean")
        spheres = tuple(self.attached_spheres)
        if not all(isinstance(sphere, AttachedSphere) for sphere in spheres):
            raise ValueError("attached_spheres requires AttachedSphere values")
        if (self.target_state is AttachmentState.ATTACHED) != bool(spheres):
            raise ValueError("attached state and sphere payload must change atomically")
        object.__setattr__(self, "attached_spheres", spheres)

    @property
    def world_hash(self):
        """Static geometry identity. Attachment state is a separate runtime transition."""
        payload = asdict(self.config)
        if not self.config.meshes:
            payload.pop("meshes", None)
        else:
            for mesh in payload["meshes"]:
                mesh.pop("file_path")
        return config_hash(payload)

    @property
    def active_cuboids(self):
        return tuple(
            cuboid
            for cuboid in self.config.cuboids
            if not ((self.target_state is AttachmentState.ATTACHED or self.target_contact_allowed) and cuboid.name == self.config.target_name)
        )

    @property
    def active_meshes(self):
        return tuple(
            mesh
            for mesh in self.config.meshes
            if not ((self.target_state is AttachmentState.ATTACHED or self.target_contact_allowed) and mesh.name == self.config.target_name)
        )

    def permit_target_contact(self):
        """Exclude only the selected target from robot-world collision queries."""
        if self.target_state is not AttachmentState.WORLD:
            raise RuntimeError("target-contact policy must be selected before attachment")
        if self.config.target_name.startswith("support_tray_"):
            raise ValueError("support trays cannot be exempted from world collision")
        return replace(self, target_contact_allowed=True)

    def attach(self, spheres):
        if self.target_state is not AttachmentState.WORLD:
            raise RuntimeError("target is already attached")
        spheres = tuple(spheres)
        if not spheres:
            raise ValueError("attachment requires a nonempty measured sphere model")
        return replace(self, target_state=AttachmentState.ATTACHED, attached_spheres=spheres)

    def detach(self):
        if self.target_state is not AttachmentState.ATTACHED:
            raise RuntimeError("target is not attached")
        return replace(self, target_state=AttachmentState.WORLD, attached_spheres=())

    def attached_spheres_b(self, tcp_pose_b):
        if self.target_state is not AttachmentState.ATTACHED:
            return np.empty((0, 4), dtype=np.float64)
        transform = rigid_transform(tcp_pose_b)
        centers = np.asarray([(*sphere.center_tcp, 1.0) for sphere in self.attached_spheres]).T
        centers_b = (transform @ centers).T[:, :3]
        radii = np.asarray([sphere.radius_m for sphere in self.attached_spheres])[:, None]
        return np.concatenate((centers_b, radii), axis=1)


def curobo_world_dict(snapshot):
    """Return CuRobo 0.7.8 primitive world input using Base-frame xyz+wxyz poses."""
    if not isinstance(snapshot, WorldSnapshot):
        raise ValueError("WorldSnapshot required")
    result = {
        "cuboid": {
            cuboid.name: {"pose": list(cuboid.pose_b), "dims": list(cuboid.dimensions_m)}
            for cuboid in snapshot.active_cuboids
        }
    }
    if snapshot.config.meshes:
        result["mesh"] = {
            mesh.name: {"pose": list(mesh.pose_b), "file_path": mesh.file_path}
            for mesh in snapshot.active_meshes
        }
    return result


def cuboid_clearance(point_b, cuboid):
    """Exact signed point-to-oriented-box clearance; positive outside, negative inside."""
    transform = pose_matrix(cuboid.pose_b[:3], cuboid.pose_b[3:])
    local = np.linalg.inv(transform) @ np.asarray((*point_b, 1.0), dtype=np.float64)
    delta = np.abs(local[:3]) - 0.5 * np.asarray(cuboid.dimensions_m)
    outside = np.linalg.norm(np.maximum(delta, 0))
    inside = min(max(delta), 0)
    return float(outside + inside)
