"""Attribute state2 approach world scores to individual unchanged production meshes."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.curobo.world import MeshConfig, WorldConfig
from DataPipe.pipelines.curobo_bootstrap.curobo_collision import CuroboCollision
from DataPipe.pipelines.curobo_bootstrap.world_adapter import WorldSnapshot


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approach", type=Path, required=True)
    parser.add_argument("--world-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    approach = read(args.approach)
    report = read(args.world_report)
    branch = approach["results"][0]
    q = np.asarray(branch["q"])
    assert q.shape == (approach["sample_count"], 6) and branch["max_self_score"] == 0
    data = report["world"]
    meshes = tuple(MeshConfig(**m) for m in data["meshes"])
    target = next(m for m in meshes if m.name == data["target_name"])
    world = WorldSnapshot(WorldConfig((), data["target_name"], meshes))
    checker = CuroboCollision(report["collision_model"], world.permit_target_contact())
    original = checker.query(q, world_ignore_links=("Base_Link",))
    np.testing.assert_allclose(original["world"], branch["world_scores"], atol=1e-6, rtol=0)
    checks = {}
    for mesh in meshes:
        if mesh.name == target.name:
            continue
        isolated = WorldSnapshot(WorldConfig((), target.name, (target, mesh))).permit_target_contact()
        checker.update_world(isolated)
        scores = checker.query(q, world_ignore_links=("Base_Link",))["world"]
        positive = np.flatnonzero(scores > 0)
        checks[mesh.name] = dict(
            max_score=float(np.max(scores)),
            first_collision_sample=int(positive[0]) if len(positive) else None,
            scores=scores.tolist(),
            mesh_sha256=mesh.sha256,
            tray_contact_diagnostic=mesh.name.startswith("support_tray_"),
        )
        print(json.dumps(dict(mesh=mesh.name, max_score=checks[mesh.name]["max_score"])), flush=True)
    result = dict(
        schema="state2_approach_world_mesh_attribution/1",
        source_case_index=branch["case_index"],
        source_candidate_index=branch["candidate_index"],
        source_world_hash=world.world_hash,
        model_sha256=hashlib.sha256(Path(report["collision_model"]).read_bytes()).hexdigest(),
        per_mesh=checks,
        only_tray_scores=all(v["max_score"] == 0 or v["tray_contact_diagnostic"] for v in checks.values()),
        input_sha256={
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.approach, args.world_report)
        },
        scope="Sampled collision attribution only; run003 tray contact remains diagnostic; not physical acceptance",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            dict(
                only_tray_scores=result["only_tray_scores"],
                positive_meshes=[name for name, value in checks.items() if value["max_score"] > 0],
            )
        )
    )


if __name__ == "__main__":
    main()
