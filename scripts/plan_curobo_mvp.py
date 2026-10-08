"""Run the fixed CuRobo MotionGen MVP, save it, reload it, and validate it."""

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.curobo.planner import PlannerConfig
from DataPipe.pipelines.curobo_bootstrap.planner import (
    EXPERIMENTAL_ACCELERATION_LIMIT,
    EXPERIMENTAL_JERK_LIMIT,
    GOAL_Q,
    SAFE_Q,
    CuroboMvpPlanner,
    load_trajectory,
    save_trajectory,
    validate_trajectory,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--collision-model",
    type=Path,
    default=Path("configs/curobo/assets/aubo_e5_project/collision_model.json"),
)
args = parser.parse_args()


def write_report(path, report):
    def json_default(value):
        if hasattr(value, "item"):
            return value.item()
        if hasattr(value, "tolist"):
            return value.tolist()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path.write_text(json.dumps(report, indent=2, default=json_default) + "\n", encoding="utf-8")


def main():
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse MVP output directory: {output}")
    output.mkdir(parents=True)
    report_path = output / "report.json"
    trajectory_path = output / "trajectory.npz"
    stage = "initialize"
    started = time.perf_counter()
    report = {
        "passed": False,
        "stage": stage,
        "python": {"executable": sys.executable, "version": platform.python_version()},
        "expected_curobo_version": "0.7.8",
        "device": args.device,
        "start_q": SAFE_Q.tolist(),
        "goal_q": GOAL_Q.tolist(),
        "limits": {
            "acceleration_rad_s2": EXPERIMENTAL_ACCELERATION_LIMIT.tolist(),
            "jerk_rad_s3": EXPERIMENTAL_JERK_LIMIT.tolist(),
            "acceleration_jerk_policy": "explicit experimental planning values; not hardware-certified limits",
            "numeric_tolerance": 1e-5,
        },
        "thresholds": {
            "minimum_motion_rad": 0.01,
            "start_error_rad": 1e-4,
            "goal_error_rad": 0.01,
            "time_step_absolute_tolerance_s": 1e-9,
        },
        "trajectory_path": str(trajectory_path),
    }
    try:
        config = PlannerConfig(version="0.7.8", device=args.device)
        report["planner_config"] = asdict(config)
        collision_model = args.collision_model.resolve()
        report["collision_model"] = str(collision_model)
        report["collision_model_sha256"] = hashlib.sha256(collision_model.read_bytes()).hexdigest()
        initialization_started = time.perf_counter()
        planner = CuroboMvpPlanner(collision_model, config=config)
        report["planner_initialization_time_s"] = time.perf_counter() - initialization_started
        report["curobo_version"] = planner.version
        report["robot_world"] = {
            "urdf": str(Path(planner.asset["urdf"]).resolve()),
            "urdf_sha256": planner.asset["urdf_sha256"],
            "world_hash": planner.world.world_hash,
            "world": [asdict(cuboid) for cuboid in planner.world.config.cuboids],
        }
        report["limits"].update(
            position_lower_rad=planner.position_lower.tolist(),
            position_upper_rad=planner.position_upper.tolist(),
            velocity_rad_s=planner.velocity_limit.tolist(),
            motion_gen_loaded=planner.runtime_limits,
        )

        stage = "warmup"
        warmup_started = time.perf_counter()
        planner.warmup()
        report["initialization_warmup_time_s"] = time.perf_counter() - warmup_started

        stage = "plan"
        plan_started = time.perf_counter()
        trajectory, backend = planner.plan()
        report["planning_wall_time_s"] = time.perf_counter() - plan_started
        report["motion_gen"] = backend
        save_trajectory(trajectory_path, trajectory)

        stage = "reload_validate"
        reloaded = load_trajectory(trajectory_path)
        acceptance = validate_trajectory(
            reloaded,
            start_q=SAFE_Q,
            goal_q=GOAL_Q,
            position_lower=planner.position_lower,
            position_upper=planner.position_upper,
            velocity_limit=planner.velocity_limit,
            acceleration_limit=EXPERIMENTAL_ACCELERATION_LIMIT,
            control_hz=config.control_hz,
            collision_query=planner.collision.query,
        )
        report["acceptance"] = acceptance
        report["passed"] = bool(backend["success"] and acceptance["passed"])
        report["stage"] = "complete" if report["passed"] else stage
        report["total_wall_time_s"] = time.perf_counter() - started
        if not report["passed"]:
            report["failure_reason"] = "reloaded trajectory failed one or more acceptance checks"
        write_report(report_path, report)
        print(json.dumps(report, indent=2), flush=True)
        return 0 if report["passed"] else 1
    except Exception as error:
        report.update(
            stage=stage,
            failure_reason=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            total_wall_time_s=time.perf_counter() - started,
        )
        write_report(report_path, report)
        print(json.dumps(report, indent=2), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
