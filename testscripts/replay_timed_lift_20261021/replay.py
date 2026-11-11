"""Replay accepted measured states in a visible native GUI; capture every 12 controls."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_trial(label):
    import numpy as np
    source = PROJECT / "data/test/trajectorys/legacy_timed_lift_20261021_verified/0_to_1" / label
    evidence = PROJECT / "data/test/trajdata/legacy_timed_lift_20261021_verified/0_to_1" / label
    paths = [source / name for name in ("trajectory.npz", "trace.json", "scene_snapshot.json")]
    paths.append(evidence / "report.json")
    hashes = {str(p): sha(p) for p in paths}
    rows, task, report = (read(p) for p in paths[1:])
    assert report["passed"] and task["task_hash"] == report["task_hash"]
    with np.load(paths[0], allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    from configs.assets import ROBOT_PRIM_CONTRACT
    assert tuple(arrays["joint_names"]) == ROBOT_PRIM_CONTRACT.arm_joints
    n = len(rows)
    np.testing.assert_allclose(arrays["time_s"], np.arange(n) / 60, rtol=0, atol=1e-12)
    for key, field in (("joint_position", "q"), ("joint_velocity", "dq")):
        np.testing.assert_array_equal(arrays[key], [r[field] for r in rows])
    for field, width in (("q", 6), ("dq", 6), ("gripper_position", 2), ("target_pose_w", 7)):
        values = np.asarray([r[field] for r in rows], dtype=float)
        assert values.shape == (n, width) and np.isfinite(values).all(), field
    assert report["execution_contract"]["control_hz"] == 60
    return dict(label=label, source=source, rows=rows, task=task, report=report,
                arrays=arrays, hashes=hashes, frames=n)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["isaacsim"], default="isaacsim")
    parser.add_argument("--run-id", default="legacy_timed_lift_replay_20261021")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kit-cache-dir", type=Path, default=PROJECT / "data/isaacsim_kit_cache")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    args.headless = False
    trials = [load_trial(f"trial_{i:02d}") for i in range(1, 4)]
    if args.check_only:
        print(json.dumps([dict(trial=t["label"], frames=t["frames"], images=t["frames"] // 12 * 3)
                          for t in trials], indent=2))
        return
    output = PROJECT / "data/test/trajdata" / args.run_id / "0_to_1"
    trajectory_output = PROJECT / "data/test/trajectorys" / args.run_id / "0_to_1"
    output.mkdir(parents=True, exist_ok=False)
    trajectory_output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "replay_source.py")
    runtime = Path(sys.executable).parent
    os.environ["PATH"] = os.pathsep.join([str(runtime), str(runtime / "Scripts"),
                                         str(runtime / "Library/bin"), os.environ.get("PATH", "")])
    spec = importlib.util.spec_from_file_location("lift_runtime", HERE.parent / "legacy_lift_speed_20261015/run_sweep.py")
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    cache = base.install_shared_runtime(output, camera_warmup_timeout_s=60)
    summary = dict(schema="measured_lift_state_replay/1", passed=False, mode="gui",
                   source_run="legacy_timed_lift_20261021_verified", capture_interval_control_steps=12,
                   control_hz=60, physics_steps_during_replay=0, trials=[])
    try:
        import numpy as np
        from scripts import sim_backend
        launcher = sim_backend.launch(args)
        import torch
        from configs.assets import ROBOT_PRIM_CONTRACT
        from scripts.pipelines.curobo_bootstrap.robot_adapter import pose_matrix
        from scripts.pipelines.curobo_bootstrap.replay_visualization import ReplayFrameRecorder
        from scripts.pipelines.curobo_bootstrap.contact_timing import configure_contact_timing
        from isaacsim.core.utils.viewports import set_camera_view
        Env, Cfg = sim_backend.trajectory_classes()
        cfg = Cfg()
        cfg.sim.device = args.device
        cfg.seed = trials[0]["task"]["seed"]
        configure_contact_timing(cfg, physics_hz=120, control_hz=60, velocity_iterations=4)
        cfg, _ = sim_backend.configure_cameras(cfg, 384, 384)
        cfg.camera_diagnostics_directory = output / "camera_warmup"
        env = Env(cfg)
        env.reset()
        robot, target = env.robot, env.target
        arm_ids, _ = robot.find_joints(ROBOT_PRIM_CONTRACT.arm_joints)
        finger_ids, _ = robot.find_joints(ROBOT_PRIM_CONTRACT.gripper_joints)
        set_camera_view(eye=np.array([2.7, -2.2, 2.2]), target=np.array([1.1, 0., 1.]))

        def tensor(value):
            return torch.as_tensor(value, dtype=robot.data.joint_pos.dtype, device=args.device)

        # State replay must never advance PhysX, even accidentally.
        def forbidden_step(*unused, **kwargs):
            raise RuntimeError("Physics stepping is forbidden in measured-state visual replay")
        env.sim.step = forbidden_step

        for trial in trials:
            label, rows = trial["label"], trial["rows"]
            destination = output / label
            destination.mkdir()
            trdir = trajectory_output / label
            trdir.mkdir()
            write(trdir / "replay_source.json", dict(source_trajectory=str(trial["source"] / "trajectory.npz"),
                  source_sha256=trial["hashes"], mode="measured_state", output=str(destination)))
            report = dict(trial=label, passed=False, source_frames=len(rows), replayed_frames=0,
                          physics_steps=0, mode="measured_state", input_sha256=trial["hashes"],
                          missing_velocity_fields="finger/object velocities were not recorded; set zero for display only",
                          state_replay_is_new_physical_validation=False)
            summary["trials"].append(report)
            write(destination / "replay_report.json", report)
            base_pose = pose_matrix(robot.data.root_pos_w[0].cpu().numpy(), robot.data.root_quat_w[0].cpu().numpy())
            np.testing.assert_allclose(base_pose, trial["task"]["frames"]["T_WB"], rtol=0, atol=1e-5)
            recorder = ReplayFrameRecorder(destination / "execution_visualization", env.camera_specs)
            errors = []
            start = time.monotonic()
            for index, row in enumerate(rows):
                if not launcher.app.is_running():
                    raise RuntimeError("GUI closed before replay completed")
                remaining = start + float(trial["arrays"]["time_s"][index]) - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                q, dq = robot.data.joint_pos.clone(), robot.data.joint_vel.clone()
                q[:, arm_ids], dq[:, arm_ids] = tensor(row["q"]), tensor(row["dq"])
                q[:, finger_ids], dq[:, finger_ids] = tensor(row["gripper_position"]), 0
                robot.write_joint_state_to_sim(q, dq)
                robot.set_joint_position_target(q, joint_ids=list(range(q.shape[-1])))
                target.write_root_pose_to_sim(tensor(row["target_pose_w"])[None])
                target.write_root_velocity_to_sim(tensor([0] * 6)[None])
                env.scene.write_data_to_sim()
                env.sim.forward()
                env.scene.update(cfg.sim.dt)
                env.sim.render()
                actual_q = robot.data.joint_pos[0, arm_ids].cpu().numpy()
                actual_g = robot.data.joint_pos[0, finger_ids].cpu().numpy()
                actual_p = target.data.root_pos_w[0].cpu().numpy()
                actual_r = target.data.root_quat_w[0].cpu().numpy()
                expected_r = np.asarray(row["target_pose_w"][3:])
                cosine = abs(float(np.dot(actual_r / np.linalg.norm(actual_r), expected_r / np.linalg.norm(expected_r))))
                error = dict(control_step=index + 1, trajectory_index=index,
                             time_s=float(trial["arrays"]["time_s"][index]), stage=row["stage"],
                             joint_error_rad=float(np.max(np.abs(actual_q - row["q"]))),
                             finger_error_m=float(np.max(np.abs(actual_g - row["gripper_position"]))),
                             object_error_m=float(np.linalg.norm(actual_p - row["target_pose_w"][:3])),
                             object_rotation_error_rad=float(2 * np.arccos(np.clip(cosine, -1, 1))))
                errors.append(error)
                captured = recorder.capture(env.cameras, render_frame=index + 1, trajectory_index=index,
                                           time_s=error["time_s"], event=row["stage"])
                if captured:
                    for frame in recorder.frames[-captured:]:
                        frame.update(control_step=index + 1, execution_stage=row["stage"])
                report["replayed_frames"] = index + 1
                if (index + 1) % 120 == 0:
                    print(f"REPLAY_PROGRESS {label} {index + 1}/{len(rows)}", flush=True)
            report["visualization"] = recorder.finish(source_frames=len(rows), render_calls=len(rows))
            for key in ("joint_error_rad", "finger_error_m", "object_error_m", "object_rotation_error_rad"):
                report["max_" + key] = max(row[key] for row in errors)
                assert report["max_" + key] < (1e-3 if key.endswith("rotation_error_rad") else 1e-5), key
            report["inputs_unchanged"] = all(sha(p) == h for p, h in trial["hashes"].items())
            assert report["inputs_unchanged"]
            report.update(passed=True, full_trajectory_replayed=True, wall_seconds=time.monotonic() - start)
            write(destination / "replay_report.json", report)
            write(destination / "replay_trace.json", errors)
            write(output / "summary.json", summary)
            print(f"REPLAY_COMPLETE {label} images={report['visualization']['image_count']}", flush=True)
        summary["passed"] = all(t["passed"] for t in summary["trials"])
    except Exception as exc:
        summary["failure_reason"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        summary["all_source_files_unchanged"] = all(sha(p) == h for trial in trials for p, h in trial["hashes"].items())
        summary["passed"] = summary["passed"] and summary["all_source_files_unchanged"]
        write(output / "summary.json", summary)
        print(f"REPLAY_BATCH_RESULT passed={summary['passed']} output={output}", flush=True)
        # Kit may terminate Python at close: persist all evidence first.
        if "real_close" in cache:
            cache["real_close"]()
        if "launcher" in cache:
            cache["launcher"].app.close()


if __name__ == "__main__":
    main()
