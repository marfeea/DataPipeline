"""Check candidate state4 XY distances using saved production frames; IK only."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from DataPipe.pipelines.curobo_bootstrap.task_targets import (
    PickPlaceV2Config,
    _rotation_wxyz_from_matrix,
    solve_legacy_joint_targets,
    task_snapshot,
)
from DataPipe.state4_routing import State4RoutingConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    saved = json.loads(args.targets.read_text(encoding="utf-8"))
    task = saved["task_snapshot"]
    transform = np.asarray(task["frames"]["T_WB"])
    rows = []
    for distance in (0.12, 0.18, 0.24):
        config = PickPlaceV2Config(
            source_index=0,
            destination_index=4,
            lift_clearance_m=task["lift_clearance_m"],
            state4_routing=State4RoutingConfig(place_standoff_m=distance),
        )
        candidate = task_snapshot(
            config=config,
            environment_origin_w=np.asarray(task["frames"]["T_WE"])[:3, 3],
            base_pose_w=(*transform[:3, 3], *_rotation_wxyz_from_matrix(transform)),
        )
        assert np.allclose(
            candidate["waypoint_position_b"]["place"], task["waypoint_position_b"]["place"], atol=1e-12, rtol=0
        )
        descent = candidate["contact_waypoint_position_b"]["place_descent"]
        ik = solve_legacy_joint_targets(candidate, saved["legacy_ik"]["q_start"])
        contact_ik = solve_legacy_joint_targets(
            dict(candidate, waypoint_position_b={"place": descent}), saved["legacy_ik"]["q_start"]
        )
        rows.append(dict(standoff_m=distance, place_descent_b=descent, five_targets=ik, descent_ik=contact_ik))
    result = dict(scope="saved-frame endpoint IK only; no collision or execution certification", candidates=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            [
                dict(standoff_m=r["standoff_m"], passed=r["five_targets"]["passed"] and r["descent_ik"]["passed"])
                for r in rows
            ]
        )
    )


if __name__ == "__main__":
    main()
