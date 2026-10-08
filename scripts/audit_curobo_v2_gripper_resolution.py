"""Compare conservative gripper sphere resolutions against a frozen production world."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.curobo.world import MeshConfig, WorldConfig
from DataPipe.pipelines.curobo_bootstrap.collision_model import derive_collision_model, load_collision_model
from DataPipe.pipelines.curobo_bootstrap.curobo_collision import CuroboCollision
from DataPipe.pipelines.curobo_bootstrap.world_adapter import WorldSnapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-report", type=Path, required=True)
    parser.add_argument("--joint-position", type=float, nargs=6, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gripper-cells-m", type=float, nargs="+", default=(0.025, 0.02, 0.015))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = json.loads(args.world_report.read_text(encoding="utf-8"))
    world_data = source["world"]
    if world_data["cuboids"]:
        raise ValueError("this audit requires the mesh-only production world")
    world = WorldSnapshot(
        WorldConfig(
            (),
            world_data["target_name"],
            tuple(MeshConfig(**mesh) for mesh in world_data["meshes"]),
        )
    )
    for mesh in world.config.meshes:
        if hashlib.sha256(Path(mesh.file_path).read_bytes()).hexdigest() != mesh.sha256:
            raise ValueError(f"world mesh evidence no longer matches: {mesh.name}")
    reference = load_collision_model(source["collision_model"])
    q = np.asarray(args.joint_position, dtype=np.float64)
    if not np.isfinite(q).all():
        raise ValueError("joint position must be finite")
    result = {
        "schema": "curobo_v2_gripper_resolution_audit/1",
        "world_report": str(args.world_report.resolve()),
        "world_hash": world.world_hash,
        "joint_position": q.tolist(),
        "world_collision_ignore_links": ["Base_Link"],
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for cell in args.gripper_cells_m:
        model = derive_collision_model(
            reference["urdf"],
            max_cell_m=reference["max_cell_m"],
            padding_m=reference["padding_m"],
            attachment_slots=reference["attachment_slots"],
            gripper_mode="fixed_open_components",
            refined_flange_cell_m=reference["refined_flange_cell_m"],
            gripper_cell_m=cell,
        )
        sphere_count = sum(len(spheres) for spheres in model["collision_spheres"].values())
        if sphere_count + model["attachment_slots"] >= 1024:
            result["results"].append(
                {
                    "gripper_cell_m": cell,
                    "sphere_count": sphere_count,
                    "skipped": "CuRobo self-collision kernel requires fewer than 1024 spheres",
                }
            )
            continue
        checker = CuroboCollision(model, world)
        scores = checker.query(q, world_ignore_links=("Base_Link",))
        result["results"].append(
            {
                "gripper_cell_m": cell,
                "sphere_counts": {name: len(spheres) for name, spheres in model["collision_spheres"].items()},
                "world_collision_score": float(scores["world"][0]),
                "self_collision_score": float(scores["self"][0]),
                "world_collision_free": bool(scores["world"][0] <= 0),
                "self_collision_free": bool(scores["self"][0] <= 0),
                "model": {
                    key: model[key]
                    for key in ("max_cell_m", "padding_m", "gripper_mode", "refined_flange_cell_m", "gripper_cell_m")
                },
            }
        )
        del checker
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "results": result["results"]}, indent=2))


if __name__ == "__main__":
    main()
