"""Deterministic first-version cuboid world description."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class CuboidConfig:
    name: str
    pose_b: tuple[float, ...]
    dimensions_m: tuple[float, float, float]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("cuboid name is required")
        pose = tuple(float(x) for x in self.pose_b)
        dims = tuple(float(x) for x in self.dimensions_m)
        if len(pose) != 7 or not all(isfinite(x) for x in pose) or abs(sum(x * x for x in pose[3:]) - 1) > 1e-6:
            raise ValueError("pose_b requires xyz and unit wxyz")
        if len(dims) != 3 or not all(isfinite(x) and x > 0 for x in dims):
            raise ValueError("dimensions_m must be three positive finite lengths")
        object.__setattr__(self, "pose_b", pose)
        object.__setattr__(self, "dimensions_m", dims)


@dataclass(frozen=True)
class MeshConfig:
    name: str
    pose_b: tuple[float, ...]
    file_path: str
    sha256: str
    source_prim_path: str

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("mesh name is required")
        pose = tuple(float(x) for x in self.pose_b)
        if len(pose) != 7 or not all(isfinite(x) for x in pose) or abs(sum(x * x for x in pose[3:]) - 1) > 1e-6:
            raise ValueError("pose_b requires xyz and unit wxyz")
        if not Path(self.file_path).is_absolute():
            raise ValueError("mesh file_path must be absolute")
        if len(self.sha256) != 64 or any(value not in "0123456789abcdef" for value in self.sha256):
            raise ValueError("mesh sha256 must be a lowercase digest")
        if not isinstance(self.source_prim_path, str) or not self.source_prim_path.startswith("/"):
            raise ValueError("source_prim_path must be an absolute USD prim path")
        object.__setattr__(self, "pose_b", pose)


@dataclass(frozen=True)
class WorldConfig:
    cuboids: tuple[CuboidConfig, ...]
    target_name: str
    meshes: tuple[MeshConfig, ...] = ()

    def __post_init__(self):
        cuboids = tuple(self.cuboids)
        if not all(isinstance(c, CuboidConfig) for c in cuboids):
            raise ValueError("world requires CuboidConfig objects")
        names = [c.name for c in cuboids]
        if len(names) != len(set(names)):
            raise ValueError("world cuboid names must be unique")
        meshes = tuple(self.meshes)
        if not all(isinstance(mesh, MeshConfig) for mesh in meshes):
            raise ValueError("world meshes require MeshConfig objects")
        mesh_names = [mesh.name for mesh in meshes]
        if len(mesh_names) != len(set(mesh_names)) or set(names) & set(mesh_names):
            raise ValueError("world obstacle names must be unique")
        if self.target_name not in set(names) | set(mesh_names):
            raise ValueError("world requires an explicit target obstacle")
        # Order does not change collision geometry or its content hash.
        object.__setattr__(self, "cuboids", tuple(sorted(cuboids, key=lambda c: c.name)))
        object.__setattr__(self, "meshes", tuple(sorted(meshes, key=lambda item: item.name)))
