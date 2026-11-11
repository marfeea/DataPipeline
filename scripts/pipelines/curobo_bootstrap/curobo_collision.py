"""Version-gated CuRobo GPU collision adapter with explicit attachment state."""

from importlib.metadata import version
from pathlib import Path

import numpy as np

from .collision_model import curobo_robot_dict, load_collision_model
from .world_adapter import AttachedSphere, WorldSnapshot, curobo_world_dict


def pack_attached_spheres(spheres, slots):
    spheres = tuple(spheres)
    if not spheres or not all(isinstance(sphere, AttachedSphere) for sphere in spheres):
        raise ValueError("a nonempty AttachedSphere sequence is required")
    if not isinstance(slots, int) or slots < len(spheres):
        raise ValueError("attachment sphere count exceeds configured slots")
    result = np.zeros((slots, 4), dtype=np.float32)
    result[:, 3] = -100.0
    for index, sphere in enumerate(spheres):
        result[index] = (*sphere.center_tcp, sphere.radius_m)
    return result


class CuroboCollision:
    """Thin runtime boundary; importing this module does not import CuRobo or Torch."""

    def __init__(self, collision_model, world, *, device="cuda:0"):
        self.version = version("nvidia_curobo")
        if self.version != "0.7.8":
            raise ValueError(f"CuRobo collision adapter requires tested 0.7.8, got {self.version}")
        if not isinstance(world, WorldSnapshot):
            raise ValueError("WorldSnapshot required")

        import torch

        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.geom.types import WorldConfig as CuroboWorldConfig
        from curobo.types.base import TensorDeviceType
        from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CuRobo collision checking requires a working CUDA device")
        self.tensor_args = TensorDeviceType(device=self.device)
        self.asset = (
            load_collision_model(collision_model)
            if isinstance(collision_model, str | bytes | Path)
            else collision_model
        )
        initial_world = CuroboWorldConfig.from_dict(curobo_world_dict(world))
        checker_type = CollisionCheckerType.MESH if world.config.meshes else CollisionCheckerType.PRIMITIVE
        config = RobotWorldConfig.load_from_config(
            curobo_robot_dict(self.asset),
            world_model=initial_world,
            tensor_args=self.tensor_args,
            n_cuboids=max(1, len(world.active_cuboids)),
            n_meshes=max(1, len(world.config.meshes)),
            collision_checker_type=checker_type,
            collision_activation_distance=0.0,
            self_collision_activation_distance=0.0,
        )
        self.robot_world = RobotWorld(config)
        self.attachment_link = self.asset["attachment_link"]
        self.attachment_slots = self.asset["attachment_slots"]
        self.world = world
        self.attached = False
        self._sphere_ranges = {}
        offset = 0
        for name, spheres in self.asset["collision_spheres"].items():
            self._sphere_ranges[name] = slice(offset, offset + len(spheres))
            offset += len(spheres)

    @property
    def joint_names(self):
        return tuple(self.robot_world.kinematics.joint_names)

    @property
    def total_spheres(self):
        return self.robot_world.kinematics.total_spheres

    def update_world(self, world):
        if not isinstance(world, WorldSnapshot):
            raise ValueError("WorldSnapshot required")
        from curobo.geom.types import WorldConfig as CuroboWorldConfig

        self.robot_world.update_world(CuroboWorldConfig.from_dict(curobo_world_dict(world)))
        self.world = world

    def attach(self, world):
        if self.attached:
            raise RuntimeError("an object is already attached")
        if not isinstance(world, WorldSnapshot) or not world.attached_spheres:
            raise ValueError("attached WorldSnapshot with sphere payload required")
        packed = pack_attached_spheres(world.attached_spheres, self.attachment_slots)
        tensor = self.tensor_args.to_device(packed)
        self.robot_world.kinematics.kinematics_config.attach_object(
            sphere_tensor=tensor, link_name=self.attachment_link
        )
        self.update_world(world)
        self.attached = True

    def detach(self, world):
        if not self.attached:
            raise RuntimeError("no object is attached")
        if not isinstance(world, WorldSnapshot) or world.attached_spheres:
            raise ValueError("detached WorldSnapshot required")
        self.robot_world.kinematics.kinematics_config.detach_object(self.attachment_link)
        self.update_world(world)
        self.attached = False

    def kinematics(self, q):
        tensor = self.tensor_args.to_device(q)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[1] != len(self.joint_names):
            raise ValueError("q must have shape [batch, dof]")
        if not self.torch.isfinite(tensor).all():
            raise ValueError("q must be finite")
        return self.robot_world.get_kinematics(tensor)

    def query(self, q, *, world_ignore_links=()):
        state = self.kinematics(q)
        spheres = state.link_spheres_tensor.unsqueeze(1)
        ignored = tuple(world_ignore_links)
        unknown = sorted(set(ignored) - set(self._sphere_ranges))
        if unknown:
            raise ValueError(f"unknown world-ignore collision links: {unknown}")
        world_spheres = spheres.clone() if ignored else spheres
        for name in ignored:
            world_spheres[:, :, self._sphere_ranges[name], 3] = -10.0
        world = self.robot_world.get_collision_constraint(world_spheres).squeeze(1)
        self_collision = self.robot_world.get_self_collision_distance(spheres).squeeze(1)
        return {
            "world": world.detach().cpu().numpy(),
            "self": self_collision.detach().cpu().numpy(),
            "spheres": state.link_spheres_tensor.detach().cpu().numpy(),
            "ee_position": state.ee_position.detach().cpu().numpy(),
            "ee_quaternion": state.ee_quaternion.detach().cpu().numpy(),
            "world_ignore_links": ignored,
        }


def self_collision_pairs(model, spheres, *, limit=3):
    """Explain positive sphere overlaps between links that CuRobo does not ignore."""
    values = np.asarray(spheres, dtype=np.float64)
    names = (*model["collision_link_names"], model["attachment_link"])
    ranges = {}
    offset = 0
    for name in names:
        count = (model["attachment_slots"] if name == model["attachment_link"]
                 else len(model["collision_spheres"][name]))
        ranges[name] = slice(offset, offset + count)
        offset += count
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] != offset:
        raise ValueError("sphere array does not match collision model")
    ignored = {frozenset((name, other)) for name, others in model["self_collision_ignore"].items() for other in others}
    pairs = []
    for index, left in enumerate(names):
        a = values[ranges[left]]
        for right in names[index + 1:]:
            if frozenset((left, right)) in ignored:
                continue
            b = values[ranges[right]]
            overlap = a[:, None, 3] + b[None, :, 3] - np.linalg.norm(
                a[:, None, :3] - b[None, :, :3], axis=2
            )
            row, col = np.unravel_index(np.argmax(overlap), overlap.shape)
            if overlap[row, col] > 0:
                pairs.append({
                    "links": [left, right],
                    "sphere_indices": [int(row), int(col)],
                    "overlap_m": float(overlap[row, col]),
                    "centers_base_m": [a[row, :3].tolist(), b[col, :3].tolist()],
                    "radii_m": [float(a[row, 3]), float(b[col, 3])],
                })
    pairs.sort(key=lambda item: item["overlap_m"], reverse=True)
    return pairs[:limit]


