"""Try finer conservative Link_01/02 coverage without changing meshes, padding or ignores."""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.curobo.world import MeshConfig, WorldConfig
from DataPipe.pipelines.curobo_bootstrap.collision_model import aabb_cover_spheres, load_collision_model
from DataPipe.pipelines.curobo_bootstrap.curobo_collision import CuroboCollision, self_collision_pairs
from DataPipe.pipelines.curobo_bootstrap.world_adapter import WorldSnapshot


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-report", type=Path, required=True)
    parser.add_argument("--branches", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells", type=float, nargs="+", default=(0.04, 0.025, 0.02))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = read(args.world_report)
    branches = read(args.branches)
    base = load_collision_model(args.base_model)
    world_data = report["world"]
    world = WorldSnapshot(
        WorldConfig((), world_data["target_name"], tuple(MeshConfig(**m) for m in world_data["meshes"]))
    )
    assert world.world_hash == report["world_hash"] == branches["world_hash"]
    collision_world = world.permit_target_contact()
    q = np.asarray([c["joint_position"] for c in branches["candidates"]])
    args.output.mkdir(parents=True)
    results = []
    for cell in args.cells:
        assert 0 < cell < base["max_cell_m"]
        model = copy.deepcopy(base)
        model["refined_link_cells_m"] = {name: cell for name in ("Link_01", "Link_02")}
        model["derived_from_sha256"] = hashlib.sha256(args.base_model.read_bytes()).hexdigest()
        for name in model["refined_link_cells_m"]:
            bounds = model["geometry_bounds"][name]
            spheres = aabb_cover_spheres(
                bounds["minimum"], bounds["maximum"], max_cell_m=cell, padding_m=model["padding_m"]
            )
            model["collision_spheres"][name] = [dict(center=list(s.center), radius=s.radius) for s in spheres]
        assert model["self_collision_ignore"] == base["self_collision_ignore"]
        assert model["padding_m"] == base["padding_m"]
        path = args.output / f"collision_model_links_{round(cell * 1000):03d}mm.json"
        path.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
        sphere_count = sum(len(values) for values in model["collision_spheres"].values()) + model["attachment_slots"]
        if sphere_count >= 1024:
            skipped = dict(cell_m=cell, total_spheres=sphere_count, skipped="CuRobo 0.7.8 requires nspheres < 1024")
            results.append(skipped)
            print(json.dumps(skipped), flush=True)
            continue
        checker = CuroboCollision(path, collision_world)
        scores = checker.query(q, world_ignore_links=("Base_Link",))
        candidates = []
        for index, (self_score, world_score) in enumerate(zip(scores["self"], scores["world"], strict=True)):
            candidates.append(
                dict(
                    candidate=index,
                    self_score=float(self_score),
                    world_score=float(world_score),
                    collision_free=bool(self_score == 0 and world_score == 0),
                    largest_pairs=self_collision_pairs(model, scores["spheres"][index], limit=2),
                )
            )
        result = dict(
            cell_m=cell,
            model=str(path.resolve()),
            model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            total_spheres=checker.total_spheres,
            candidates=candidates,
        )
        results.append(result)
        print(json.dumps(result), flush=True)
        (args.output / "refinement_result.json").write_text(
            json.dumps(
                dict(
                    schema="state2_conservative_sphere_refinement/1",
                    base_model=str(args.base_model.resolve()),
                    world_hash=world.world_hash,
                    branches=str(args.branches.resolve()),
                    results=results,
                    padding_preserved=True,
                    ignore_pairs_preserved=True,
                    coverage=(
                        "Same full local Link_01/02 AABBs covered by smaller circumscribed cells; "
                        "other spheres unchanged"
                    ),
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        del checker


if __name__ == "__main__":
    main()
