"""Visible GUI experiment comparing CuRobo lift targets with physical execution."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
BASE = HERE.parent / "legacy_lift_speed_20261015"
sys.path[:0] = [str(HERE), str(BASE), str(PROJECT)]
spec = importlib.util.spec_from_file_location("lift_sweep_base", BASE / "run_sweep.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def prepare():
    source, manifest = base.prepare()
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError(f"Expected one source match: {old[:100]}")
        source = source.replace(old, new)
    replace("from policy import bounded_lift_delta", "from policy import bounded_lift_delta\nfrom lift_experiment import run_lift")
    replace('parser.add_argument("--experiment-grasp-settle-s", type=float, default=0.0)',
            'parser.add_argument("--experiment-grasp-settle-s", type=float, default=0.0)\n'
            'parser.add_argument("--curobo-lift-mode", choices=("joint", "pose", "line"), default="joint")\n'
            'parser.add_argument("--curobo-time-scale", type=float, default=1.0)')
    replace('if record["stage"] not in ("free_space_to_pre_pick", "free_space_transfer"):',
            'if record["stage"] not in ("free_space_to_pre_pick", "free_space_transfer") and not record.get("curobo_physical"):' )
    replace('        report["lift"] = approach("legacy_lift", lift, -1.0, target_quaternion_b=pick_quaternion)\n'
            '        for _ in range(round(task["control_hz"] * 0.5)):\n'
            '            step_ik(lift, "legacy_lift_hold", -1.0, target_quaternion_b=pick_quaternion)',
            '        report["lift"] = run_lift(env=env, robot=robot, target=target, ik=ik, gripper=gripper,\n'
            '            arm_ids=arm_ids, cfg=cfg, task=task, args=args, records=records, report=report,\n'
            '            lift=lift, pick_quaternion=pick_quaternion, actual_world=actual_world,\n'
            '            frozen_world=frozen_world, capture_sample=capture_sample, contacts=contact_reports)')
    compile(source, str(HERE / "generated_lift_probe.py"), "exec")
    (HERE / "generated_lift_probe.py").write_text(source, encoding="utf-8")
    manifest["lift_probe_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return source, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="curobo_lift_20261015")
    parser.add_argument("--modes", nargs="+", choices=("joint", "pose", "line"), default=["joint", "pose"])
    parser.add_argument("--time-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if any(not 1 <= x <= 4 for x in args.time_scales):
        parser.error("time scales must lie in [1, 4]")
    source, manifest = prepare()
    if args.prepare_only:
        print(json.dumps(manifest, indent=2))
        return
    evidence_root = PROJECT / "data/test/trajdata" / args.run_id / "0_to_1"
    trajectory_root = PROJECT / "data/test/trajectorys" / args.run_id / "0_to_1"
    evidence_root.mkdir(parents=True, exist_ok=False)
    trajectory_root.mkdir(parents=True, exist_ok=False)
    runtime = Path(sys.executable).parent
    os.environ["PATH"] = os.pathsep.join([str(runtime), str(runtime / "Scripts"), str(runtime / "Library/bin"), os.environ.get("PATH", "")])
    reference = PROJECT / "data/0_to_1_1014_tcp/0_to_1_evidence"
    status = dict(mode="gui", scope="CuRobo same endpoint lift, physical target execution", trials=[], manifest=manifest)
    snapshot = evidence_root / "experiment_source"
    snapshot.mkdir()
    for name in ("run_experiment.py", "lift_experiment.py", "README.md", "generated_lift_probe.py", "manifest.json"):
        (snapshot / name).write_bytes((HERE / name).read_bytes())
    (snapshot / "base_run_sweep.py").write_bytes((BASE / "run_sweep.py").read_bytes())
    (snapshot / "policy.py").write_bytes((BASE / "policy.py").read_bytes())
    cache = base.install_shared_runtime(evidence_root, camera_warmup_timeout_s=60)
    try:
        for mode in args.modes:
            for time_scale in args.time_scales:
                label = f"{mode}_time_{time_scale:g}".replace(".", "p")
                evidence, trajectory = evidence_root / label, trajectory_root / label
                command = [str(HERE / "generated_lift_probe.py"), "--backend", "isaacsim", "--device", "cuda:0",
                           "--first-probe", str(reference / "first_probe_01/report.json"),
                           "--targets-report", str(reference / "targets.json"), "--world-report", str(reference / "world/report.json"),
                           "--output", str(evidence), "--trajectory-output", str(trajectory),
                           "--stop-after-lift", "--capture-execution", "--camera-width", "256", "--camera-height", "256",
                           "--physics-hz", "120", "--contact-velocity-iterations", "4", "--contact-policy", "run003",
                           "--pick-lift-step-m", "0.001", "--pick-lift-command-hz", "60", "--pick-lift-speed-mode", "step",
                           "--experiment-grasp-settle-s", "0.35", "--curobo-lift-mode", mode, "--curobo-time-scale", str(time_scale)]
                entry = dict(label=label, planning_mode=mode, time_scale=time_scale, command=[sys.executable, *command],
                             trajectory_directory=str(trajectory), evidence_directory=str(evidence))
                status["trials"].append(entry)
                (evidence_root / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
                print(f"CUROBO_LIFT_START {label} GUI / physical execution", flush=True)
                sys.argv = command
                try:
                    exec(compile(source, command[0], "exec"), {"__name__": "__main__", "__file__": command[0]})
                except SystemExit as error:
                    entry["exit_code"] = error.code
                if (evidence / "report.json").exists():
                    result = base.read(evidence / "report.json")
                    entry.update(execution_passed=result["passed"], failure_reason=result.get("failure_reason"))
                (evidence_root / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
                print("CUROBO_LIFT_FINISH " + json.dumps(entry), flush=True)
                if entry.get("execution_passed") is False and not result.get("record_count"):
                    return
                # Keep completed MotionGen objects from competing with native camera rendering.
                gc.collect()
                import torch
                torch.cuda.empty_cache()
    finally:
        if "real_close" in cache:
            cache["real_close"]()
        if "launcher" in cache:
            cache["launcher"].app.close()


if __name__ == "__main__":
    main()
