"""Replay a selected physical experiment NPZ in the native Isaac Sim scene.

state: display measured robot, finger and object states without physics steps.
action: reset once, then send the recorded actual commands through PhysX.
The NPZ must retain its report.json, scene_snapshot.json and trace.json sidecars.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs.assets import ASSETS, ROBOT_PRIM_CONTRACT, asset_path
from scripts.pick_place_reward_candidate import rotation_error
from scripts.pipelines.curobo_bootstrap.pick_place_artifact import load_hybrid_trajectory, save_json, sha256_file


def load_recording(trajectory):
    trajectory = Path(trajectory).resolve()
    run = trajectory.parent
    paths = [trajectory, *(run / name for name in ("report.json", "scene_snapshot.json", "trace.json"))]
    hashes = {str(p): sha256_file(p) for p in paths}
    report, snapshot, records = [json.loads(p.read_text(encoding="utf-8")) for p in paths[1:]]
    if report.get("schema") != "physical_failure_experiment/1":
        raise ValueError("expected a physical experiment trajectory with matching sidecars")
    if not report.get("passed") or report.get("accepted_for_use") is False:
        raise ValueError("recording was incomplete or withdrawn")
    task = snapshot["task"]
    if task["task_hash"] != report["task_hash"]:
        raise ValueError("scene/task identity mismatch")
    arrays = load_hybrid_trajectory(trajectory)
    n = len(records)
    if n < 2 or tuple(arrays["joint_names"]) != ROBOT_PRIM_CONTRACT.arm_joints:
        raise ValueError("invalid sample count or joint order")
    fields = {
        "q": 6,
        "dq": 6,
        "gripper_position": 2,
        "gripper_velocity": 2,
        "target_pose_w": 7,
        "target_velocity_w": 6,
        "commanded_joint_target": 6,
        "executed_gripper_target": 2,
    }
    states = {key: np.asarray([r[key] for r in records], dtype=float) for key in fields}
    for key, width in fields.items():
        if states[key].shape != (n, width) or not np.isfinite(states[key]).all():
            raise ValueError(f"invalid recorded field: {key}")
    if np.any(np.linalg.norm(states["target_pose_w"][:, 3:], axis=1) < 1e-12):
        raise ValueError("zero object quaternion")
    np.testing.assert_allclose(arrays["time_s"], np.arange(n) / 60, rtol=0, atol=1e-12)
    np.testing.assert_allclose(arrays["time_s"], [r["time_s"] for r in records], rtol=0, atol=1e-12)
    for key, field in (
        ("joint_position", "q"),
        ("joint_velocity", "dq"),
        ("gripper_target", "executed_gripper_target"),
    ):
        np.testing.assert_array_equal(arrays[key], states[field])
    if arrays["object_attached"].any():
        raise ValueError("physical recording must not contain logical attachment")
    hz = report["execution_contract"]["physics_hz"]
    if not isinstance(hz, int) or hz < 60 or hz % 60:
        raise ValueError("invalid recorded physics rate")
    return dict(
        trajectory=trajectory,
        task=task,
        report=report,
        arrays=arrays,
        states=states,
        records=records,
        hashes=hashes,
        frames=n,
    )


def frame_commands(recording, index):
    """Actions are recorded targets, never the measured joint response."""
    states = recording["states"]
    return states["commanded_joint_target"][index], states["executed_gripper_target"][index]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True, help="Physical experiment trajectory.npz")
    parser.add_argument("--mode", choices=("state", "action"), default="state")
    parser.add_argument(
        "--output", type=Path, help="New output directory; defaults to data/physical_replays/<timestamp>"
    )
    parser.add_argument("--check-only", action="store_true", help="Validate inputs/assets without starting Isaac Sim")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--save-images", action="store_true", help="Save three 256x256 camera views every 12 frames")
    parser.add_argument("--max-frames", type=int, help="Limit replay for a smoke check")
    parser.add_argument(
        "--hold-s", type=float, default=5.0, help="Keep the final GUI frame visible without advancing physics"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kit-cache-dir", type=Path, default=ROOT / "data/isaacsim_kit_cache")
    args = parser.parse_args()
    if not np.isfinite(args.hold_s) or args.hold_s < 0 or args.max_frames is not None and args.max_frames < 1:
        parser.error("hold-s must be finite/nonnegative and max-frames must be positive")
    args.backend = "isaacsim"
    return args


def main():
    args = parse_args()
    recording = load_recording(args.trajectory)
    missing = [str(asset_path(s)) for s in ASSETS if s.required and not asset_path(s).is_file()]
    if missing:
        raise FileNotFoundError(f"missing USD assets: {missing}")
    if args.check_only:
        print(
            json.dumps(
                dict(
                    passed=True,
                    trajectory=str(recording["trajectory"]),
                    mode=args.mode,
                    frames=recording["frames"],
                    duration_s=float(recording["arrays"]["time_s"][-1]),
                ),
                indent=2,
            )
        )
        return 0
    output = (args.output or ROOT / "data/physical_replays" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    if output == recording["trajectory"].parent or recording["trajectory"].parent in output.parents:
        raise ValueError("output must be outside the source recording")
    output.mkdir(parents=True, exist_ok=False)
    runtime = Path(sys.executable).parent
    os.environ["PATH"] = os.pathsep.join(
        [str(runtime), str(runtime / "Scripts"), str(runtime / "Library/bin"), os.environ.get("PATH", "")]
    )
    report = dict(
        schema="physical_trajectory_replay/1",
        passed=False,
        stage="startup",
        mode=args.mode,
        source_trajectory=str(recording["trajectory"]),
        input_sha256=recording["hashes"],
        source_frames=recording["frames"],
        replayed_frames=0,
        physics_steps=0,
        reward_evaluation_performed=False,
        logical_attachment=False,
        state_writes_after_initialization=0,
        headless=args.headless,
    )
    env = launcher = recorder = None
    rows = []
    status = 1
    try:
        from scripts.sim_backend import launch

        print("REPLAY_STARTUP SimulationApp", flush=True)
        launcher = launch(args)
        import torch
        from isaacsim.core.utils.viewports import set_camera_view
        from scripts.native.scene import NativeTrajectoryEnv, NativeTrajectoryEnvCfg

        cfg = NativeTrajectoryEnvCfg()
        cfg.sim.device = args.device
        cfg.seed = recording["task"]["seed"]
        cfg.sim.dt = 1 / recording["report"]["execution_contract"]["physics_hz"]
        cfg.decimation = recording["report"]["execution_contract"]["physics_hz"] // 60
        report["stage"] = "scene_initialization"
        save_json(output / "replay_report.json", report)
        env = NativeTrajectoryEnv(cfg)
        env.reset()
        robot, target = env.robot, env.target
        arm_ids, _ = robot.find_joints(ROBOT_PRIM_CONTRACT.arm_joints)
        finger_ids, _ = robot.find_joints(ROBOT_PRIM_CONTRACT.gripper_joints)
        states = recording["states"]

        def tensor(values):
            return torch.as_tensor(values, dtype=robot.data.joint_pos.dtype, device=args.device)

        def write_state(index):
            positions, velocities = robot.data.joint_pos.clone(), robot.data.joint_vel.clone()
            positions[:, arm_ids], velocities[:, arm_ids] = tensor(states["q"][index]), tensor(states["dq"][index])
            positions[:, finger_ids] = tensor(states["gripper_position"][index])
            velocities[:, finger_ids] = tensor(states["gripper_velocity"][index])
            robot.write_joint_state_to_sim(positions, velocities)
            target.write_root_pose_to_sim(tensor(states["target_pose_w"][index])[None])
            target.write_root_velocity_to_sim(tensor(states["target_velocity_w"][index])[None])
            q, g = frame_commands(recording, index)
            robot.set_joint_position_target(tensor(q)[None], joint_ids=arm_ids)
            robot.set_joint_position_target(tensor(g)[None], joint_ids=finger_ids)
            env.sim.forward()
            env.scene.update(cfg.sim.dt)

        write_state(0)
        scale = recording["report"].get("gripper_effort_scale", 1.0)
        robot.view.set_max_efforts(
            robot.view.get_max_efforts(joint_indices=finger_ids) * scale, joint_indices=finger_ids
        )
        # Require the production robot base to match the task world frame.
        from scripts.pipelines.curobo_bootstrap.robot_adapter import pose_matrix

        base = pose_matrix(robot.data.root_pos_w[0].cpu().numpy(), robot.data.root_quat_w[0].cpu().numpy())
        if not np.allclose(base, recording["task"]["frames"]["T_WB"], atol=1e-5, rtol=0):
            raise ValueError("live robot base differs from recording")
        if args.mode == "action":

            def forbidden_write(*unused, **kwargs):
                raise RuntimeError("state writes are forbidden during action replay")

            robot.write_joint_state_to_sim = forbidden_write
            target.write_root_pose_to_sim = forbidden_write
            target.write_root_velocity_to_sim = forbidden_write
        set_camera_view(eye=np.array([2.7, -2.2, 2.2]), target=np.array([1.1, 0.0, 1.0]))
        if args.save_images:
            from scripts.pipelines.curobo_bootstrap.replay_visualization import ReplayFrameRecorder

            env._create_cameras((256, 256))
            recorder = ReplayFrameRecorder(output / "execution_visualization", env.camera_specs)
        total = min(recording["frames"], args.max_frames or recording["frames"])
        start = time.monotonic()
        report["stage"] = "replay"
        for index in range(total):
            if not launcher.app.is_running():
                raise RuntimeError(f"Isaac Sim stopped before frame {index}")
            if not args.no_realtime:
                remaining = start + recording["arrays"]["time_s"][index] - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
            if index and args.mode == "state":
                write_state(index)
                report["state_writes_after_initialization"] += 1
            elif index:
                q, g = frame_commands(recording, index)
                robot.set_joint_position_target(tensor(q)[None], joint_ids=arm_ids)
                robot.set_joint_position_target(tensor(g)[None], joint_ids=finger_ids)
                for _ in range(cfg.decimation):
                    env.scene.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(cfg.sim.dt)
                    report["physics_steps"] += 1
            env.sim.render()
            pose = np.r_[target.data.root_pos_w[0].cpu().numpy(), target.data.root_quat_w[0].cpu().numpy()]
            row = dict(
                index=index,
                time_s=float(recording["arrays"]["time_s"][index]),
                q=robot.data.joint_pos[0, arm_ids].cpu().numpy().tolist(),
                target_pose_w=pose.tolist(),
                gripper_position=robot.data.joint_pos[0, finger_ids].cpu().numpy().tolist(),
                position_error_m=float(np.linalg.norm(pose[:3] - states["target_pose_w"][index, :3])),
                rotation_error_rad=rotation_error(pose[3:], states["target_pose_w"][index, 3:]),
            )
            rows.append(row)
            report["replayed_frames"] = len(rows)
            if recorder:
                recorder.capture(
                    env.cameras,
                    render_frame=index + 1,
                    trajectory_index=index,
                    time_s=row["time_s"],
                    event=str(recording["arrays"]["event_code"][index]),
                )
            if (index + 1) % 120 == 0:
                print(f"REPLAY_PROGRESS {index + 1}/{total}", flush=True)
        if recorder:
            report["visualization"] = recorder.finish(source_frames=total, render_calls=total)
        report["max_position_error_m"] = max(r["position_error_m"] for r in rows)
        report["max_rotation_error_rad"] = max(r["rotation_error_rad"] for r in rows)
        report["max_joint_error_rad"] = float(np.max(np.abs(np.asarray([r["q"] for r in rows]) - states["q"][:total])))
        report["max_gripper_error_m"] = float(
            np.max(np.abs(np.asarray([r["gripper_position"] for r in rows]) - states["gripper_position"][:total]))
        )
        if args.mode == "state" and (
            report["max_position_error_m"] > 1e-5
            or report["max_rotation_error_rad"] > 1e-3
            or report["max_joint_error_rad"] > 1e-5
            or report["max_gripper_error_m"] > 1e-5
        ):
            raise RuntimeError("state replay readback differs from recorded states")
        report.update(passed=True, stage="complete", full_trajectory_replayed=total == recording["frames"])
        status = 0
        print(f"REPLAY_COMPLETE report={output / 'replay_report.json'}", flush=True)
        deadline = time.monotonic() + (0 if args.headless else args.hold_s)
        while launcher.app.is_running() and time.monotonic() < deadline:
            env.sim.render()
    except Exception as error:
        report.update(passed=False, failure_reason=f"{type(error).__name__}: {error}")
        traceback.print_exc()
        status = 1
    finally:
        report["inputs_unchanged"] = all(sha256_file(p) == h for p, h in recording["hashes"].items())
        if not report["inputs_unchanged"]:
            report.update(passed=False, failure_reason="source recording changed during replay")
            status = 1
        save_json(output / "replay_report.json", report)
        save_json(output / "replay_trace.json", rows)
        if env:
            env.close()
        if launcher:
            launcher.app.close()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
