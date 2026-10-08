"""Save audited five-waypoint IK feasibility for all eight state4 routes."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from DataPipe.pipelines.curobo_bootstrap.route_experiment import require_height, route_config
from DataPipe.pipelines.curobo_bootstrap.task_targets import solve_legacy_joint_targets, task_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=float, default=0.3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    project = Path(__file__).resolve().parents[1]
    original = project / "data/curobo_pick_place_graph_run003_20261003_r2/evidence"
    routes = []
    for source in range(5):
        for destination in range(5):
            if source == destination or 4 not in (source, destination):
                continue
            name = f"{source}_to_{destination}"
            file = original / (name + "_evidence") / "targets.json"
            prior = json.loads(file.read_text(encoding="utf-8"))
            measured = prior["measured"]
            task = task_snapshot(
                config=route_config(source, destination, state4_lift_clearance_m=args.height),
                environment_origin_w=measured["environment_origin_w"],
                base_pose_w=measured["base_pose_w"],
            )
            require_height(task, args.height)
            for waypoint in ("pre_pick", "pick", "place"):
                assert task["waypoint_position_b"][waypoint] == prior["task_snapshot"]["waypoint_position_b"][waypoint]
            assert task["task_hash"] != prior["task_hash"]
            result = solve_legacy_joint_targets(task, prior["legacy_ik"]["q_start"])
            routes.append(
                dict(
                    route=name,
                    task_snapshot=task,
                    legacy_ik=result,
                    source_targets=str(file),
                    source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                    endpoint_geometry_unchanged=True,
                )
            )
    report = dict(
        scope="offline IK at measured production base pose; not full trajectory or physical grasp acceptance",
        lift_clearance_m=args.height,
        all_height_targets_passed=all(r["legacy_ik"]["passed"] for r in routes),
        routes=routes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(all_height_targets_passed=report["all_height_targets_passed"], route_count=len(routes))))
    return 0 if report["all_height_targets_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
