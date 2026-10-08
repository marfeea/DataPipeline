"""Replay the frozen CuRobo NPZ in an Isaac scene matching its planning world."""

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

from DataPipe.pipelines.curobo_bootstrap.replay import frame_deadlines, load_replay_spec
from DataPipe.runtime import ensure_curriculumrl_available


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
    parser.add_argument("--no-realtime", action="store_true", help="Render every frame without deadline sleeping")
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
        "mode": "state",
        "scope": "state_replay_same_planning_scene",
        "target_frame_rate_hz": 60.0,
        "realtime_enabled": not args.no_realtime,
        "headless": bool(args.headless),
        "formal_acceptance": not args.headless and not args.no_realtime,
        "python": {"executable": sys.executable, "version": sys.version.split()[0]},
        "device": args.device,
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

        source_frames = len(spec.trajectory.time_s)
        written_frames = 0
        render_calls = 0
        late_frames = 0
        max_lag_s = 0.0
        max_q_error = 0.0
        max_dq_error = 0.0
        max_drift = {
            "root_position_m": 0.0,
            "root_rotation_rad": 0.0,
            "world_position_m": 0.0,
            "world_rotation_rad": 0.0,
        }
        start = time.monotonic()
        deadlines = frame_deadlines(spec.trajectory.time_s, start)
        print(f"REPLAY_START frames={source_frames} mode=state frame_rate=60.0", flush=True)
        report["stage"] = "replay"
        for index, deadline in enumerate(deadlines):
            if not launcher.app.is_running():
                raise RuntimeError(f"SimulationApp stopped before frame {index}")
            if not args.no_realtime:
                remaining = deadline - time.monotonic()
                if remaining > 0.0:
                    time.sleep(remaining)
            lag = max(0.0, time.monotonic() - deadline)
            if lag > 1e-6:
                late_frames += 1
            max_lag_s = max(max_lag_s, lag)
            expected_q = spec.trajectory.joint_position[index]
            expected_dq = spec.trajectory.joint_velocity[index]
            actual_q, actual_dq = scene.write_frame(expected_q, expected_dq)
            written_frames += 1
            max_q_error = max(max_q_error, float(np.max(np.abs(actual_q - expected_q))))
            max_dq_error = max(max_dq_error, float(np.max(np.abs(actual_dq - expected_dq))))
            drift = scene.invariant_drift()
            for name, value in drift.items():
                max_drift[name] = max(max_drift[name], value)
            if (
                drift["root_position_m"] > 1e-6
                or drift["root_rotation_rad"] > 1e-6
                or drift["world_position_m"] > 1e-6
                or drift["world_rotation_rad"] > 1e-6
            ):
                raise RuntimeError(f"scene invariant drift at frame {index}: {drift}")
            scene.sim.render()
            render_calls += 1
        elapsed = time.monotonic() - start
        achieved_fps = (source_frames - 1) / elapsed if elapsed > 0.0 and source_frames > 1 else None
        checks = {
            "scene_consistent": scene_report["passed"],
            "all_frames_written": written_frames == source_frames,
            "one_render_per_frame": render_calls == source_frames,
            "q_writeback": max_q_error <= 1e-6,
            "root_static": max_drift["root_position_m"] <= 1e-6 and max_drift["root_rotation_rad"] <= 1e-6,
            "world_static": max_drift["world_position_m"] <= 1e-6 and max_drift["world_rotation_rad"] <= 1e-6,
        }
        report.update(
            stage="complete",
            passed=all(checks.values()),
            checks=checks,
            written_frames=written_frames,
            render_calls=render_calls,
            wall_clock_duration_s=elapsed,
            achieved_fps=achieved_fps,
            late_frames=late_frames,
            max_deadline_lag_ms=max_lag_s * 1000.0,
            max_q_writeback_error_rad=max_q_error,
            max_dq_writeback_error_rad_s=max_dq_error,
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
            report["failure_reason"] = "replay completion gate failed"
            return_code = 1
        else:
            print(
                f"REPLAY_COMPLETE frames={render_calls}/{source_frames} "
                f"scene_consistent={str(scene_report['passed']).lower()}",
                flush=True,
            )
            return_code = 0
        return return_code
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
