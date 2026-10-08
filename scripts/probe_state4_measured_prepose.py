"""Check measured-grasp state4 entry geometry and full-pose IK before execution."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from DataPipe.pipelines.curobo_bootstrap.measured_placement import measured_state4_placement
from DataPipe.pipelines.curobo_bootstrap.robot_adapter import pose_matrix
from DataPipe.runtime import ensure_curriculumrl_available


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads((args.run / "report.json").read_text(encoding="utf-8"))
    targets = json.loads(args.targets.read_text(encoding="utf-8"))
    records = json.loads((args.run / "trace.json").read_text(encoding="utf-8"))
    calibration = measured_state4_placement(
        targets["task_snapshot"],
        lift_tcp_pose_b=report["lift_tcp_pose_b"],
        object_pose_w_after_lift=report["target_pose_w_after_lift"],
    )
    ensure_curriculumrl_available()
    from CurriculumRL.logic.aubo_ik import JOINT_LIMITS_RAD, forward_pose, solve_tcp_ik

    rng = np.random.default_rng(7)
    q_start = np.asarray([r for r in records if r["stage"] == "free_space_transfer"][-1]["q"])
    seeds = [q_start, np.asarray(targets["legacy_ik"]["q_start"]), np.zeros(6)]
    seeds += [rng.uniform(JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1]) for _ in range(16)]
    checks = {}
    for name, pose in calibration["tcp_waypoint_pose_b"].items():
        solutions = []
        target_matrix = pose_matrix(pose[:3], pose[3:])
        for seed in seeds:
            solved = solve_tcp_ik(pose[:3], pose[3:], seed=seed)
            if not solved.success or solved.q is None:
                continue
            q = np.asarray(solved.q)
            p, rotation = forward_pose(q)
            position_error = float(np.linalg.norm(p - target_matrix[:3, 3]))
            angle = float(np.arccos(np.clip((np.trace(target_matrix[:3, :3].T @ rotation) - 1) / 2, -1, 1)))
            if position_error > 1e-5 or angle > 1e-4:
                continue
            if not np.all((q >= JOINT_LIMITS_RAD[:, 0]) & (q <= JOINT_LIMITS_RAD[:, 1])):
                continue
            if any(np.max(np.abs(q - prior["q"])) < 0.01 for prior in solutions):
                continue
            solutions.append(dict(q=q.tolist(), fk_position_error_m=position_error, fk_orientation_error_rad=angle))
        checks[name] = dict(passed=bool(solutions), solutions=solutions)
    result = dict(
        schema="state4_measured_prepose_ik/1",
        passed=all(value["passed"] for value in checks.values()),
        calibration=calibration,
        full_pose_ik=checks,
        input_sha256={
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.run / "report.json", args.run / "trace.json", args.targets)
        },
        scope="Measured rigid-grasp geometry and endpoint IK only; not collision or physical execution acceptance",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(passed=result["passed"], branches={k: len(v["solutions"]) for k, v in checks.items()})))


if __name__ == "__main__":
    main()
