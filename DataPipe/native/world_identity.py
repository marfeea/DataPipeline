"""Audit live native geometry before reusing an immutable planning snapshot."""

from pathlib import Path

import numpy as np
import trimesh

from configs.curobo.world import MeshConfig, WorldConfig
from DataPipe.pipelines.curobo_bootstrap.pick_place_artifact import sha256_file
from DataPipe.pipelines.curobo_bootstrap.world_adapter import WorldSnapshot


def reconcile_world(actual, expected_config, *, asset_paths=None, tolerance_m=2e-7):
    """Reject geometry drift; preserve both actual and planning world identities.

    OBJ serialization hashes change for sub-micron PhysX root readback rounding.
    Equal topology and a 0.2 micrometre maximum vertex error are required before
    the original frozen meshes may be used for collision planning.
    """
    if not 0 < tolerance_m <= 2e-7:
        raise ValueError("geometry reconciliation tolerance cannot exceed 0.2 micrometres")
    expected = WorldSnapshot(
        WorldConfig(
            tuple(expected_config["cuboids"]),
            expected_config["target_name"],
            tuple(MeshConfig(**m) for m in expected_config["meshes"]),
        )
    )
    left = {m.name: m for m in actual.config.meshes}
    right = {m.name: m for m in expected.config.meshes}
    if set(left) != set(right) or actual.config.target_name != expected.config.target_name:
        raise ValueError("native production world mesh identities differ")
    residuals = {}
    for name, frozen in right.items():
        path = Path((asset_paths or {}).get(frozen.file_path, frozen.file_path))
        if sha256_file(path) != frozen.sha256:
            raise ValueError(f"frozen planning mesh hash differs: {name}")
        a = trimesh.load_mesh(path, process=False)
        b = trimesh.load_mesh(left[name].file_path, process=False)
        if frozen.source_prim_path != left[name].source_prim_path or not np.array_equal(
            frozen.pose_b, left[name].pose_b
        ):
            raise ValueError(f"native mesh pose/path differs: {name}")
        if a.vertices.shape != b.vertices.shape or not np.array_equal(a.faces, b.faces):
            raise ValueError(f"native mesh topology differs: {name}")
        error = float(np.max(np.abs(a.vertices - b.vertices)))
        residuals[name] = error
        if error > tolerance_m:
            raise ValueError(f"native collision geometry drift: {name} error={error}m")
    return expected, {
        "passed": True,
        "actual_world_hash": actual.world_hash,
        "planning_world_hash": expected.world_hash,
        "tolerance_m": tolerance_m,
        "max_vertex_error_m": max(residuals.values()),
        "mesh_max_vertex_errors_m": residuals,
        "identical_topology": True,
        "scope": "live native mesh vertices versus SHA256-verified frozen planning geometry",
    }
