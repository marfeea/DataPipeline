"""Execute a frozen CuRobo trajectory in its matching Isaac planning scene."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from DataPipe.pipelines.curobo_bootstrap.replay import (
    frame_deadlines,
    load_replay_spec,
    summarize_tracking,
)
from DataPipe.runtime import ensure_curriculumrl_available

PHYSICS_DT_S = 1.0 / 120.0
CONTROL_HZ = 60.0
PHYSICS_STEPS_PER_TARGET = 2
# MVP envelope for this frozen trajectory and the existing AUBO position actuator.
# These are simulation acceptance limits, not hardware accuracy claims.
MAX_POSITION_ERROR_RAD = 0.03
MAX_FINAL_POSITION_ERROR_RAD = 0.015
SCENE_DRIFT_TOLERANCE = 1e-6


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=_json_default, allow_nan=False), encoding="utf-8")


def _package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--plan-report", type=Path)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--no-realtime", action="store_true", help="Run all physics steps without deadline sleeping")
    parser.add_argument("--kit-cache-dir", type=Path)
    ensure_curriculumrl_available()
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    report_path = args.replay_report.resolve()
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite replay report: {report_path}")
    plan_report = args.plan_report or args.trajectory.parent / "report.json"
    report = {
        "passed": False,
        "stage": "preflight",
        "failure_reason": None,
        "mode": "action",
        "scope": "position_target_physics_replay_same_planning_scene",
        "target_frame_rate_hz": CONTROL_HZ,
        "physics_dt_s": PHYSICS_DT_S,
        "physics_steps_per_target": PHYSICS_STEPS_PER_TARGET,
        "realtime_enabled": not args.no_realtime,
        "headless": bool(args.headless),
        "formal_acceptance": not args.headless and not args.no_realtime,
        "python": {"executable": sys.executable, "version": sys.version.split()[0]},
        "device": args.device,
        "thresholds": {
            "max_position_error_rad": MAX_POSITION_ERROR_RAD,
            "max_final_position_error_rad": MAX_FINAL_POSITION_ERROR_RAD,
            "scene_drift": SCENE_DRIFT_TOLERANCE,
        },
    }
    try:
        spec = load_replay_spec(args.trajectory, plan_report)
        report.update(
            acceptance_fixture=spec.fixture_name,
            trajectory_path=str(spec.trajectory_path),
            plan_report_path=str(spec.plan_report_path),
            trajectory_sha256=spec.hashes["trajectory"],
            plan_report_sha256=spec.hashes["plan_report"],
            urdf_sha256=spec.hashes["urdf"],
            collision_model_sha256=spec.hashes["collision_model"],
            planned_world_hash=spec.world.world_hash,
            reconstructed_world_hash=spec.world.world_hash,
            source_frames=len(spec.trajectory.time_s),
            source_duration_s=float(spec.trajectory.time_s[-1]),
            source_time_step_s=float(np.diff(spec.trajectory.time_s)[0]),
            npz_joint_names=list(spec.trajectory.joint_names),
        )
    except Exception as error:
        report["failure_reason"] = f"{type(error).__name__}: {error}"
        _write_report(report_path, report)
        traceback.print_exc()
        return 1

    launcher = None
    scene = None
    try:
        if args.kit_cache_dir is not None:
            cache = args.kit_cache_dir.resolve()
            cache.mkdir(parents=True, exist_ok=True)
            os.environ["OV_CACHE_ROOT"] = str(cache)
        launcher = AppLauncher(args)
        from DataPipe.curobo_replay_scene import CuroboReplayScene

        report["stage"] = "scene_initialization"
        scene = CuroboReplayScene(spec, device=args.device)
        scene.set_camera_view()
        report["stage"] = "scene_validation"
        scene_report = scene.validate_scene()
        report.update(
            isaac_sim_version=_package_version("isaacsim"),
            isaac_lab_version=_package_version("isaaclab"),
            isaac_joint_names=list(spec.trajectory.joint_names),
            scene_consistent=scene_report["passed"],
            scene_consistency=scene_report,
        )
        if not scene_report["passed"]:
            raise RuntimeError("scene consistency gate failed")
        if not np.isclose(scene.sim.get_physics_dt(), PHYSICS_DT_S, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"unexpected physics dt: {scene.sim.get_physics_dt()}")

        reference_q = np.asarray(spec.trajectory.joint_position)
        reference_dq = np.asarray(spec.trajectory.joint_velocity)
        actual_q = []
        actual_dq = []
        render_calls = 0
        executed_targets = 0
        physics_steps = 0
        late_targets = 0
        max_lag_s = 0.0
        max_drift = {
            name: 0.0 for name in ("root_position_m", "root_rotation_rad", "world_position_m", "world_rotation_rad")
        }

        initial_q, initial_dq = scene.initialize_physics_replay(reference_q[0], reference_dq[0])
        actual_q.append(initial_q)
        actual_dq.append(initial_dq)
        scene.sim.render()
        render_calls += 1
        start = time.monotonic()
        deadlines = frame_deadlines(spec.trajectory.time_s, start)
        print(f"REPLAY_START frames={len(reference_q)} mode=action frame_rate={CONTROL_HZ:.1f}", flush=True)
        report["stage"] = "replay"
        for index in range(1, len(reference_q)):
            if not launcher.app.is_running():
                raise RuntimeError(f"SimulationApp stopped before target {index}")
            deadline = deadlines[index]
            if not args.no_realtime:
                remaining = deadline - time.monotonic()
                if remaining > 0.0:
                    time.sleep(remaining)
            lag = max(0.0, time.monotonic() - deadline)
            if lag > 1e-6:
                late_targets += 1
            max_lag_s = max(max_lag_s, lag)
            measured_q, measured_dq = scene.step_position_target(
                reference_q[index], physics_steps=PHYSICS_STEPS_PER_TARGET
            )
            executed_targets += 1
            physics_steps += PHYSICS_STEPS_PER_TARGET
            actual_q.append(measured_q)
            actual_dq.append(measured_dq)
            drift = scene.invariant_drift()
            for name, value in drift.items():
                max_drift[name] = max(max_drift[name], value)
            if any(value > SCENE_DRIFT_TOLERANCE for value in drift.values()):
                raise RuntimeError(f"scene invariant drift at target {index}: {drift}")
            scene.sim.render()
            render_calls += 1

        elapsed = time.monotonic() - start
        tracking = summarize_tracking(reference_q, reference_dq, np.asarray(actual_q), np.asarray(actual_dq))
        source_frames = len(reference_q)
        checks = {
            "scene_consistent": scene_report["passed"],
            "frame_zero_initialized_once": len(actual_q) == source_frames,
            "all_noninitial_targets_executed": executed_targets == source_frames - 1,
            "two_physics_steps_per_target": physics_steps == (source_frames - 1) * PHYSICS_STEPS_PER_TARGET,
            "one_render_per_source_frame": render_calls == source_frames,
            "position_tracking": tracking["position_error_rad"]["max_abs"] <= MAX_POSITION_ERROR_RAD,
            "final_position_tracking": tracking["final_position_max_abs_rad"] <= MAX_FINAL_POSITION_ERROR_RAD,
            "root_static": max_drift["root_position_m"] <= SCENE_DRIFT_TOLERANCE
            and max_drift["root_rotation_rad"] <= SCENE_DRIFT_TOLERANCE,
            "world_static": max_drift["world_position_m"] <= SCENE_DRIFT_TOLERANCE
            and max_drift["world_rotation_rad"] <= SCENE_DRIFT_TOLERANCE,
        }
        report.update(
            stage="complete",
            passed=all(checks.values()),
            checks=checks,
            initialized_frames=1,
            executed_targets=executed_targets,
            physics_steps=physics_steps,
            render_calls=render_calls,
            wall_clock_duration_s=elapsed,
            achieved_control_hz=(source_frames - 1) / elapsed if elapsed > 0.0 else None,
            late_targets=late_targets,
            max_deadline_lag_ms=max_lag_s * 1000.0,
            tracking=tracking,
            root_pose_drift={
                "position_m": max_drift["root_position_m"],
                "rotation_rad": max_drift["root_rotation_rad"],
            },
            world_pose_drift={
                "position_m": max_drift["world_position_m"],
                "rotation_rad": max_drift["world_rotation_rad"],
            },
            formal_acceptance_passed=bool(report["formal_acceptance"] and all(checks.values())),
        )
        if not report["passed"]:
            report["failure_reason"] = "physics replay completion gate failed"
            return 1
        print(
            f"REPLAY_COMPLETE targets={executed_targets}/{source_frames - 1} "
            f"max_q_error={tracking['position_error_rad']['max_abs']:.6g}",
            flush=True,
        )
        return 0
    except Exception as error:
        report["failure_reason"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
        return 1
    finally:
        try:
            if report_path.exists():
                raise FileExistsError(f"Replay report appeared during run: {report_path}")
            _write_report(report_path, report)
        finally:
            if scene is not None:
                print("REPLAY_CLEANUP scene", flush=True)
                scene.close()
                scene = None
            if launcher is not None:
                print("REPLAY_CLEANUP app", flush=True)
                launcher.app.close()
                print("REPLAY_CLEANUP complete", flush=True)


if __name__ == "__main__":
    os._exit(main())
