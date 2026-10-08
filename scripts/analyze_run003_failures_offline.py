"""Reproduce saved run003 IK failures without Isaac or changing source evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DataPipe.runtime import ensure_curriculumrl_available

ensure_curriculumrl_available()
from CurriculumRL.logic import aubo_ik as ik


def read(file):
    return json.loads(file.read_text(encoding="utf-8"))


def pack(result, pose):
    output = dict(success=bool(result.success), iterations=int(result.iterations),
                  position_error_m=float(result.position_error_m),
                  orientation_error_rad=float(result.orientation_error_rad),
                  reason=result.reason, q=None)
    if result.success:
        p, rotation = ik.forward_pose(result.q)
        target_rotation = ik._quaternion_to_rotation_wxyz(pose[3:])
        angle = np.arccos(np.clip((np.trace(target_rotation.T @ rotation) - 1) / 2, -1, 1))
        output.update(q=result.q.tolist(), fk_position_error_m=float(np.linalg.norm(p - pose[:3])),
                      fk_orientation_error_rad=float(angle),
                      within_joint_limits=bool(np.all(result.q >= ik.JOINT_LIMITS_RAD[:, 0])
                                               and np.all(result.q <= ik.JOINT_LIMITS_RAD[:, 1])))
    return output


def search(pose, seeds):
    best = None
    for index, seed in enumerate(seeds):
        result = pack(ik.solve_tcp_ik(pose[:3], pose[3:], seed=seed), pose)
        result.update(seed_index=index, seed_q=np.asarray(seed).tolist())
        if best is None or result["position_error_m"] < best["position_error_m"]:
            best = result
        if result["success"]:
            return dict(tested_seed_count=index + 1, passed=True, result=result)
    return dict(tested_seed_count=len(seeds), passed=False, result=best)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-seeds", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    evidence = args.run / "evidence"
    routes = []
    reports = {}
    hashes = {}
    for file in sorted(evidence.glob("*_evidence/targets.json")):
        reports[file.parent.name.removesuffix("_evidence")] = read(file)
        hashes[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
    pool = [np.zeros(6)]
    for report in reports.values():
        pool.append(np.asarray(report["legacy_ik"]["q_start"]))
        for item in report["legacy_ik"]["targets"].values():
            if item.get("passed"):
                q = np.asarray(item["joint_position"])
                if not any(np.max(np.abs(q - old)) < .01 for old in pool):
                    pool.append(q)
    rng = np.random.default_rng(7)
    random_seeds = list(rng.uniform(ik.JOINT_LIMITS_RAD[:, 0], ik.JOINT_LIMITS_RAD[:, 1],
                                   size=(args.random_seeds, 6)))
    cache = {}
    for name, report in reports.items():
        run_file = evidence / (name + "_evidence") / "run_report.json"
        run = read(run_file)
        hashes[str(run_file)] = hashlib.sha256(run_file.read_bytes()).hexdigest()
        entry = dict(route=name, original_passed=run["passed"], stage=run["stage"])
        if run["stage"] == "targets":
            failed_name, target = next((key, item) for key, item in report["legacy_ik"]["targets"].items()
                                       if not item["passed"])
            pose = np.asarray(target["tcp_pose_b"])
            original_seed = np.asarray(target["seed_q"])
            repeated = pack(ik.solve_tcp_ik(pose[:3], pose[3:], seed=original_seed), pose)
            long = pack(ik.solve_tcp_ik(pose[:3], pose[3:], seed=original_seed, max_iterations=1000), pose)
            key = tuple(np.round(pose, 7))
            if key not in cache:
                cache[key] = search(pose, pool + random_seeds)
            lower = []
            for clearance in (.35, .30, .25, .20, .15, .10):
                changed = pose.copy()
                changed[2] += clearance - report["config"]["lift_clearance_m"]
                trial = search(changed, [original_seed] + pool[:12] + random_seeds[:8])
                lower.append(dict(clearance_m=clearance, tcp_z_b_m=float(changed[2]), **trial))
            entry.update(failed_waypoint=failed_name, target_pose_b=pose.tolist(),
                         original_residuals=dict(position_m=target["position_error_m"],
                                                 orientation_rad=target["orientation_error_rad"]),
                         original_seed_reproduction=repeated, same_seed_1000_iterations=long,
                         alternative_seed_search=cache[key], height_sweep=lower)
        elif run["stage"] == "pre_pick_branches":
            file = evidence / (name + "_evidence") / "pre_pick_branches.json"
            branches = read(file)
            hashes[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
            candidates = branches["candidates"]
            pairs = sorted({tuple(pair["links"]) for item in candidates for pair in item["self_collision_pairs"]})
            entry.update(candidate_count=len(candidates), collision_free_count=branches["collision_free_count"],
                         self_collision_range=[min(item["self_collision_score"] for item in candidates),
                                               max(item["self_collision_score"] for item in candidates)],
                         world_free_count=sum(item["world_collision_score"] <= 0 for item in candidates),
                         collision_link_pairs=pairs,
                         candidates=[dict(index=i, world=item["world_collision_score"],
                                          self=item["self_collision_score"], pairs=item["self_collision_pairs"])
                                     for i, item in enumerate(candidates)])
        routes.append(entry)
        print(json.dumps({"route": name, "stage": entry["stage"],
                          "alternative_seed_passed": entry.get("alternative_seed_search", {}).get("passed"),
                          "successful_clearances": [x["clearance_m"] for x in entry.get("height_sweep", []) if x["passed"]]}), flush=True)
    result = dict(scope="offline full-pose IK and saved collision evidence only; no path, collision, or physics validation for new IK solutions",
                  run=str(args.run.resolve()), random_seed=7, random_seed_count=args.random_seeds,
                  reference_seed_count=len(pool), solver_source=str(Path(ik.__file__).resolve()),
                  solver_sha256=hashlib.sha256(Path(ik.__file__).read_bytes()).hexdigest(),
                  source_sha256=hashes, routes=routes)
    for file, digest in hashes.items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == digest, "source evidence changed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
