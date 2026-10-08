"""Extract the V2 planning world from a live production Isaac scene.

Import this module only after AppLauncher has started Kit.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import trimesh
from CurriculumRL.configs.assets import SCENE_ENTITY_TARGET, TEST_SUPPORT_TRAY_INDICES

from pxr import Usd, UsdGeom

from configs.curobo.world import MeshConfig, WorldConfig

from .robot_adapter import pose_matrix
from .attachment_geometry import root_vertices_in_base
from .world_adapter import WorldSnapshot


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _triangles(mesh):
    counts = [int(value) for value in mesh.GetFaceVertexCountsAttr().Get()]
    indices = [int(value) for value in mesh.GetFaceVertexIndicesAttr().Get()]
    triangles = []
    offset = 0
    for count in counts:
        face = indices[offset : offset + count]
        offset += count
        if count < 3:
            raise ValueError(f"mesh has a face with {count} vertices")
        triangles.extend((face[0], face[index], face[index + 1]) for index in range(1, count - 1))
    if offset != len(indices) or not triangles:
        raise ValueError("mesh topology is incomplete")
    return np.asarray(triangles, dtype=np.int64)


def _export_mesh_b(stage, prim_path, base_pose_w, path, name, *, live_root_pose_w=None, rigid_root_path=None, base_pose_vector=None):
    prim = stage.GetPrimAtPath(prim_path)
    mesh = UsdGeom.Mesh(prim)
    if not prim.IsValid() or not mesh:
        raise ValueError(f"production collision mesh missing: {prim_path}")
    points = mesh.GetPointsAttr().Get()
    transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    vertices_w = np.asarray([transform.Transform(point) for point in points], dtype=np.float64)
    vertices_b = (
        np.linalg.inv(base_pose_w)
        @ np.concatenate((vertices_w, np.ones((len(vertices_w), 1), dtype=np.float64)), axis=1).T
    ).T[:, :3]
    if live_root_pose_w is not None:
        root_prim = stage.GetPrimAtPath(rigid_root_path)
        if not root_prim.IsValid() or not prim.GetPath().HasPrefix(root_prim.GetPath()):
            raise ValueError("target mesh must belong to the measured rigid root")
        root_inverse = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(root_prim).GetInverse()
        vertices_local = np.asarray([root_inverse.Transform(transform.Transform(point)) for point in points], dtype=np.float64)
        vertices_b = root_vertices_in_base(vertices_local, root_pose_w=live_root_pose_w, base_pose_w=base_pose_vector)
    artifact = trimesh.Trimesh(vertices=vertices_b, faces=_triangles(mesh), process=False)
    watertight = bool(artifact.is_watertight)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact.export(path, file_type="obj")
    digest = _sha256(path)
    config = MeshConfig(
        name,
        (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        str(path.resolve()),
        digest,
        prim_path,
    )
    return config, {
        "name": name,
        "source_prim_path": prim_path,
        "vertices": int(len(artifact.vertices)),
        "triangles": int(len(artifact.faces)),
        "watertight": watertight,
        "sha256": digest,
        "file_path": str(path.resolve()),
        "pose_source": "measured rigid root + USD local geometry" if live_root_pose_w is not None else "USD static geometry",
        "rigid_root_path": rigid_root_path,
        "live_root_pose_w": live_root_pose_w,
    }


def extract_production_world(env, output_dir):
    """Freeze the actual production collision meshes in robot Base frame."""
    output_dir = Path(output_dir)
    controller = env.action_manager.get_term("joint_trajectory").controller
    robot = controller.robot
    base_position = robot.data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
    base_quaternion = robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
    norm = float(np.linalg.norm(base_quaternion))
    if not np.isfinite(norm) or abs(norm - 1.0) > 1e-6:
        raise ValueError("production Base quaternion is not unit")
    base_quaternion /= norm
    base_pose_w = pose_matrix(base_position, base_quaternion)
    env_root = str(env.scene.env_prim_paths[0])
    stage = env.scene.stage
    mesh_paths = {
        "glass": f"{env_root}/station/static/workstation/WorkStation/Glass/M_Glass",
        "workstation": f"{env_root}/station/static/workstation/WorkStation/WorkStation/M_WorkStation",
        **{
            f"support_tray_{index:02d}": (
                f"{env_root}/station/static/M_SupportTray_{index:02d}/SupportTray_{index:02d}/"
                f"M_SupportTray_{index:02d}"
            )
            for index in TEST_SUPPORT_TRAY_INDICES
        },
        "target": (
            f"{env_root}/station/interactive/ws_interactive_reagent_01_sample_bottle/"
            "Reagent_01/M_Reagent_01"
        ),
    }
    meshes, mesh_evidence = [], []
    target = env.scene[SCENE_ENTITY_TARGET]
    target_pose_w = [
        *target.data.root_pos_w[0].detach().cpu().tolist(),
        *target.data.root_quat_w[0].detach().cpu().tolist(),
    ]
    target_rigid_root = str(target.root_physx_view.prim_paths[0])
    base_pose_vector = [*base_position.tolist(), *base_quaternion.tolist()]
    for name, prim_path in mesh_paths.items():
        kwargs = dict(live_root_pose_w=target_pose_w, rigid_root_path=target_rigid_root, base_pose_vector=base_pose_vector) if name == "target" else {}
        config, evidence = _export_mesh_b(stage, prim_path, base_pose_w, output_dir / f"{name}.obj", name, **kwargs)
        meshes.append(config)
        mesh_evidence.append(evidence)
    snapshot = WorldSnapshot(WorldConfig((), "target", tuple(meshes)))
    return snapshot, {
        "env_root": env_root,
        "base_pose_w": [*base_position.tolist(), *base_quaternion.tolist()],
        "base_quaternion_raw_norm": norm,
        "mesh_evidence": mesh_evidence,
        "target_pose_w": target_pose_w,
        "world_hash": snapshot.world_hash,
        "scope": "exact production workstation, glass, tray and target collision triangle meshes",
    }


__all__ = ["extract_production_world"]
