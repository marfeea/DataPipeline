"""Check sampled Legacy approaches from state2's free side preposes to its original pick."""

import argparse
import hashlib
import json
import sys
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
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=51)
    parser.add_argument("--entry-lift-m", type=float, default=0.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.sample_count < 2:
        raise ValueError("approach requires at least two samples")
    if not np.isfinite(args.entry_lift_m) or args.entry_lift_m < 0:
        raise ValueError("entry lift must be finite and nonnegative")
    task = read(args.evidence / "targets.json")["task_snapshot"]
    search = read(args.search)
    report = read(args.evidence / "world/report.json")
    assert task["source_index"] == 2
    model = Path(report["collision_model"])
    assert hashlib.sha256(model.read_bytes()).hexdigest() == search["collision_model_sha256"]
    data = report["world"]
    world = WorldSnapshot(WorldConfig((), data["target_name"], tuple(MeshConfig(**m) for m in data["meshes"])))
    checker = CuroboCollision(model, world.permit_target_contact())
    ensure_curriculumrl_available()
    from CurriculumRL.logic.aubo_ik import JOINT_LIMITS_RAD, solve_tcp_ik

    pick = np.asarray(task["waypoint_position_b"]["pick"])
    results = []
    for case_index, case in enumerate(search["results"]):
        if args.entry_lift_m and not np.isclose(case["pre_pick_lift_m"], args.entry_lift_m):
            continue
        start = np.asarray(case["pre_pick_b"])
        for candidate_index, candidate in enumerate(case["candidates"]):
            if not candidate["collision_free"]:
                continue
            q = np.asarray(candidate["q"])
            configurations = []
            failure = None
            entry = pick + (0, 0, args.entry_lift_m)
            segments = [(start, entry), (entry, pick)] if args.entry_lift_m else [(start, pick)]
            points = [
                (1 - alpha) * begin + alpha * end
                for begin, end in segments for alpha in np.linspace(0, 1, args.sample_count)
            ]
            for sample_index, point in enumerate(points):
                alpha = sample_index / (len(points) - 1)
                pose = legacy_tcp_poses_b(dict(task, waypoint_position_b=dict(pick=point.tolist())))["pick"]
                solved = solve_tcp_ik(pose[:3], pose[3:], seed=q)
                if (
                    not solved.success
                    or solved.q is None
                    or solved.position_error_m > 1e-5
                    or solved.orientation_error_rad > 1e-4
                ):
                    failure = dict(sample_index=sample_index, alpha=float(alpha), reason=solved.reason)
                    break
                q = np.asarray(solved.q)
                if not np.all((q >= JOINT_LIMITS_RAD[:, 0]) & (q <= JOINT_LIMITS_RAD[:, 1])):
                    failure = dict(sample_index=sample_index, alpha=float(alpha), reason="joint limit")
                    break
                configurations.append(q.tolist())
            scores = (
                checker.query(np.asarray(configurations), world_ignore_links=("Base_Link",))
                if configurations
                else dict(world=np.array([]), self=np.array([]))
            )
            valid = (scores["world"] == 0) & (scores["self"] == 0)
            first_invalid = int(np.flatnonzero(~valid)[0]) if np.any(~valid) else None
            result = dict(
                case_index=case_index,
                candidate_index=candidate_index,
                side_offset_m=case["side_offset_m"],
                pre_pick_lift_m=case["pre_pick_lift_m"],
                completed_samples=len(configurations),
                ik_failure=failure,
                first_collision_sample=first_invalid,
                first_collision_alpha=None if first_invalid is None else first_invalid / (len(points) - 1),
                max_world_score=float(np.max(scores["world"])) if configurations else None,
                max_self_score=float(np.max(scores["self"])) if configurations else None,
                passed=bool(failure is None and len(configurations) == len(points) and np.all(valid)),
                entry_lift_m=args.entry_lift_m,
                q=configurations,
                world_scores=scores["world"].tolist(),
                self_scores=scores["self"].tolist(),
            )
            results.append(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    dict(
                        schema="state2_lateral_approach_ik_collision/1",
                        passed_approaches=sum(r["passed"] for r in results),
                        results=results,
                        sample_count=len(points),
                        samples_per_segment=args.sample_count,
                        input_sha256={
                            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (
                                args.search,
                                args.evidence / "targets.json",
                                args.evidence / "world/report.json",
                                model,
                            )
                        },
                        scope="Sampled approach IK and unchanged collision model; not physical grasp or complete route",
                    ),
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                json.dumps({k: v for k, v in result.items() if k not in ("q", "world_scores", "self_scores")}),
                flush=True,
            )


if __name__ == "__main__":
    main()
