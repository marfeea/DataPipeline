"""Search an approach point for source2 using the unchanged production collision model."""

import argparse
import hashlib
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.curobo.world import MeshConfig, WorldConfig
from DataPipe.pipelines.curobo_bootstrap.curobo_collision import CuroboCollision
from DataPipe.pipelines.curobo_bootstrap.task_targets import legacy_tcp_poses_b
from DataPipe.pipelines.curobo_bootstrap.world_adapter import WorldSnapshot
from DataPipe.runtime import ensure_curriculumrl_available


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distances", type=float, nargs="+", default=(0.03, 0.05, 0.075))
    parser.add_argument("--heights", type=float, nargs="+", default=(0, 0.04, 0.08))
    parser.add_argument("--side-offsets", type=float, nargs="+", default=(0,))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    ensure_curriculumrl_available()
    from CurriculumRL.logic.aubo_ik import JOINT_LIMITS_RAD, solve_tcp_ik

    targets = read(args.evidence / "targets.json")
    branches = read(args.evidence / "pre_pick_branches.json")
    world_report = read(args.evidence / "world/report.json")
    task = targets["task_snapshot"]
    assert task["source_index"] == 2
    world_data = world_report["world"]
    world = WorldSnapshot(
        WorldConfig((), world_data["target_name"], tuple(MeshConfig(**m) for m in world_data["meshes"]))
    )
    assert world.world_hash == branches["world_hash"]
    checker = CuroboCollision(world_report["collision_model"], world.permit_target_contact())
    pick = np.asarray(task["waypoint_position_b"]["pick"])
    pre = np.asarray(task["waypoint_position_b"]["pre_pick"])
    direction = pre - pick
    direction[2] = 0
    direction /= np.linalg.norm(direction)
    lateral = np.asarray((-direction[1], direction[0], 0.0))
    rng = np.random.default_rng(task["seed"])
    seeds = [np.asarray(targets["legacy_ik"]["q_start"])]
    seeds += [np.asarray(c["joint_position"]) for c in branches["candidates"]]
    seeds += [rng.uniform(JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1]) for _ in range(16)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for z_lift, distance, side in product(args.heights, args.distances, args.side_offsets):
        point = pick + direction * distance + lateral * side + [0, 0, z_lift]
        snapshot = dict(task, waypoint_position_b={"pre_pick": point.tolist()})
        pose = legacy_tcp_poses_b(snapshot)["pre_pick"]
        candidates = []
        for seed in seeds:
            solution = solve_tcp_ik(pose[:3], pose[3:], seed=seed)
            if not solution.success or solution.q is None:
                continue
            if solution.position_error_m > 1e-5 or solution.orientation_error_rad > 1e-4:
                continue
            q = np.asarray(solution.q)
            if not any(np.max(np.abs(q - prior)) < 0.01 for prior in candidates):
                candidates.append(q)
        records = []
        if candidates:
            scores = checker.query(np.asarray(candidates), world_ignore_links=("Base_Link",))
            records = [
                dict(
                    q=q.tolist(),
                    world_score=float(scores["world"][i]),
                    self_score=float(scores["self"][i]),
                    collision_free=bool(scores["world"][i] == 0 and scores["self"][i] == 0),
                )
                for i, q in enumerate(candidates)
            ]
        result = dict(
            normal_distance_m=distance,
            pre_pick_lift_m=z_lift,
            side_offset_m=side,
            pre_pick_b=point.tolist(),
            pre_pick_pose_b=list(pose),
            candidate_count=len(records),
            collision_free_count=sum(r["collision_free"] for r in records),
            candidates=records,
        )
        results.append(result)
        args.output.write_text(
            json.dumps(
                dict(
                    schema="source2_pre_pick_search/1",
                    source_evidence=str(args.evidence.resolve()),
                    collision_model=world_report["collision_model"],
                    collision_model_sha256=hashlib.sha256(
                        Path(world_report["collision_model"]).read_bytes()
                    ).hexdigest(),
                    results=results,
                    scope="Endpoint IK and unchanged collision gate only; not an executed grasp",
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({k: v for k, v in result.items() if k != "candidates"}), flush=True)


if __name__ == "__main__":
    main()
