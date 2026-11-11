"""Isolated native GUI Legacy IK lift sweep, without changing production files."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(HERE))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reset_robot(owner, initial_q, dt):
    import torch
    q = initial_q.clone()
    owner.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    owner.robot.set_joint_position_target(q, joint_ids=list(range(q.shape[-1])))
    owner.scene.write_data_to_sim()
    owner.sim.forward()
    owner.scene.update(dt)
    owner.sim.render()


def prepare():
    source_path = PROJECT / "scripts/probe_curobo_v2_legacy_contact_isaac.py"
    original = source_path.read_text(encoding="utf-8")
    source = original
    changes = []

    def replace(old, new, count=1):
        nonlocal source
        if source.count(old) != count:
            raise RuntimeError(f"Upstream probe changed; expected {count} copies of {old[:90]!r}")
        source = source.replace(old, new)
        changes.append(dict(old=old, new=new, count=count))

    replace("str(Path(__file__).resolve().parents[1])", "str(Path(__file__).resolve().parents[2])")
    replace("from configs.aubo_ik_control import CONTACT_PROBE_COMMAND_HZ, CONTACT_PROBE_POSITION_STEP_M",
            "from configs.aubo_ik_control import CONTACT_PROBE_COMMAND_HZ, CONTACT_PROBE_POSITION_STEP_M\n"
            "from policy import bounded_lift_delta")
    replace('parser.add_argument("--max-steps-per-stage", type=int, default=600)',
            'parser.add_argument("--max-steps-per-stage", type=int, default=600)\n'
            'parser.add_argument("--experiment-lift-joint-delta-rad", type=float, default=0.08)\n'
            'parser.add_argument("--experiment-lift-step-m", type=float, default=0.05)\n'
            'parser.add_argument("--experiment-lift-command-hz", type=float, default=4.0)\n'
            'parser.add_argument("--experiment-interpolation-fraction", type=float, default=1.0)\n'
            'parser.add_argument("--experiment-grasp-settle-s", type=float, default=0.0)')
    replace('        report["lift_clearance_m"] = task["lift_clearance_m"]',
            '        report["lift_clearance_m"] = task["lift_clearance_m"]\n'
            '        report["lift_experiment"] = dict(command_hz=args.experiment_lift_command_hz, tcp_step_cap_m=args.experiment_lift_step_m,\n'
            '            joint_delta_cap_rad=args.experiment_lift_joint_delta_rad, hold_s=0.5,\n'
            '            interpolation_fraction=args.experiment_interpolation_fraction,\n'
            '            interpolation_physics_steps=max(1, round(args.physics_hz / args.experiment_lift_command_hz * args.experiment_interpolation_fraction)),\n'
            '            grasp_settle_s=args.experiment_grasp_settle_s,\n'
            '            stage_speed_caps_disabled=True, tracking_lead_cap_disabled=True,\n'
            '            source_probe_sha256="' + hashlib.sha256(original.encode()).hexdigest() + '")')
    replace('            command_hz = args.pick_lift_command_hz if pick_lift else task["control_hz"]',
            '            is_lift = args.backend == "isaacsim" and stage in ("legacy_lift", "legacy_lift_hold")\n'
            '            command_hz = args.experiment_lift_command_hz if is_lift else (args.pick_lift_command_hz if pick_lift else task["control_hz"])')
    replace('            if pick_lift and task.get("tcp_geometry"):',
            '            if pick_lift and not is_lift and task.get("tcp_geometry"):', count=2)
    replace('            translation = np.clip((target_b - current) / position_step_m, -1.0, 1.0)',
            '            if is_lift:\n'
            '                position_step_m = args.experiment_lift_step_m\n'
            '            translation = np.clip((target_b - current) / position_step_m, -1.0, 1.0)\n'
            '            if is_lift:\n'
            '                translation = bounded_lift_delta(current, target_b, position_step_m) / position_step_m')
    replace('                    ik.policy_decimation = round(args.physics_hz / command_hz)',
            '                    ik.policy_decimation = max(1, round(args.physics_hz / command_hz *\n'
            '                        (args.experiment_interpolation_fraction if is_lift else 1.0)))\n'
            '                    ik.max_policy_joint_delta_rad.fill_(\n'
            '                        args.experiment_lift_joint_delta_rad if is_lift else 0.08)')
    replace('                "ik_command_hz": command_hz,',
            '                "ik_command_hz": command_hz,\n'
            '                "experiment_joint_delta_cap_rad": args.experiment_lift_joint_delta_rad if is_lift else 0.08,\n'
            '                "interpolation_physics_steps": ik.policy_decimation,\n'
            '                "requested_joint_delta_rad": _array(ik._requested_joint_delta[0]).tolist(),\n'
            '                "applied_joint_delta_rad": _array(ik._applied_joint_delta[0]).tolist(),')
    replace('                stage_budget *= round(task["control_hz"] / args.pick_lift_command_hz)',
            '                stage_budget *= round(task["control_hz"] / (args.experiment_lift_command_hz if label == "legacy_lift" else args.pick_lift_command_hz))')
    replace('        report["lift"] = approach("legacy_lift", lift, -1.0, target_quaternion_b=pick_quaternion)',
            '        report["lift"] = approach("legacy_lift", lift, -1.0, target_quaternion_b=pick_quaternion)\n'
            '        for _ in range(round(task["control_hz"] * 0.5)):\n'
            '            step_ik(lift, "legacy_lift_hold", -1.0, target_quaternion_b=pick_quaternion)')
    replace('        report["target_pose_w_at_pick"] = [',
            '        settle_pose = ik.current_tcp_pose_b()\n'
            '        settle_position = _array(settle_pose[0][0]).copy()\n'
            '        settle_quaternion = _array(settle_pose[1][0]).copy()\n'
            '        for _ in range(round(task["control_hz"] * args.experiment_grasp_settle_s)):\n'
            '            step_ik(settle_position, "grasp_settle", -1.0, target_quaternion_b=settle_quaternion)\n'
            '        report["grasp_settle_gripper_position"] = list(gripper.state()[0])\n'
            '        report["target_pose_w_at_pick"] = [')
    replace('finally:\n    launcher.app.close()', 'finally:\n    pass  # The experimental sweep owns the shared GUI application.')
    compile(source, str(HERE / "generated_probe.py"), "exec")
    (HERE / "generated_probe.py").write_text(source, encoding="utf-8")
    manifest = dict(upstream=str(source_path), upstream_sha256=hashlib.sha256(original.encode()).hexdigest(),
                    generated_sha256=hashlib.sha256(source.encode()).hexdigest(), replacements=changes)
    (HERE / "generated_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return source, manifest


def install_shared_runtime(evidence_root, camera_warmup_timeout_s=300):
    from scripts import sim_backend
    original_launch = sim_backend.launch
    original_classes = sim_backend.trajectory_classes
    cache = {}

    def launch(args):
        if args.headless or args.backend != "isaacsim":
            raise ValueError("This experiment requires a visible native IsaacSim GUI")
        if "launcher" not in cache:
            cache["launcher"] = original_launch(args)
            (evidence_root / "desktop_gui.json").write_text(
                json.dumps(cache["launcher"].desktop_gui, indent=2), encoding="utf-8")
        return cache["launcher"]

    def classes():
        env_class, cfg_class = original_classes()
        if "installed" not in cache:
            from scripts.native import scene
            import numpy as np

            class RGBCheckedSimAdapter(scene.SimAdapter):
                def render(self):
                    started = time.monotonic()
                    retries = 0
                    while True:
                        self.forward()
                        self.owner.world.render()
                        self.owner.scene.update(self.get_physics_dt())
                        if not self.owner.cameras_ready:
                            return
                        missing = []
                        for name, camera in self.owner.cameras.items():
                            pixels = camera.camera.get_rgb()
                            if pixels is None or not np.asarray(pixels).size or np.std(np.asarray(pixels)[..., :3]) <= 5:
                                missing.append(name)
                        if self.owner.background_only_render():
                            missing.append("scene geometry absent")
                        if not missing:
                            return
                        retries += 1
                        if retries % 100 == 0:
                            print(f"EXPERIMENT_RGB_WARMUP retries={retries} missing={missing}", flush=True)
                        if time.monotonic() - started > camera_warmup_timeout_s:
                            raise RuntimeError(f"Native RGB did not become usable: {missing}")

            scene.SimAdapter = RGBCheckedSimAdapter
            cache["installed"] = True

        def shared_env(cfg):
            import torch
            if "env" not in cache:
                cfg.camera_warmup_timeout_s = camera_warmup_timeout_s
                owner = env_class.__new__(env_class)
                cache["env"] = owner
                env_class.__init__(owner, cfg)
                cache["real_close"] = owner.close
                cache["initial_q"] = owner.robot.data.joint_pos.clone()
                owner.close = lambda: None

                def reset():
                    reset_robot(owner, cache["initial_q"], cfg.sim.dt)
                    return {}, {}

                owner.reset = reset
            owner = cache["env"]
            owner.cameras_ready = True
            owner.sim.render()
            return owner

        return shared_env, cfg_class

    sim_backend.launch = launch
    sim_backend.trajectory_classes = classes
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-id", default="legacy_lift_speed_20261015")
    parser.add_argument("--joint-caps", nargs="+", type=float, default=[0.08, 0.16, 0.32, 0.785])
    parser.add_argument("--interpolation-fractions", nargs="+", type=float, default=[1.0])
    parser.add_argument("--grasp-settle-s", type=float, default=0.0)
    parser.add_argument("--lift-step-m", type=float, default=0.05)
    parser.add_argument("--lift-command-hz", type=float, default=4.0)
    args = parser.parse_args()
    if any(not 0 < value <= 3.14 / 4 for value in args.joint_caps):
        parser.error("joint caps must lie in (0, 3.14 / 4] rad")
    if args.lift_command_hz not in (4, 60) or not 0 < args.lift_step_m <= 0.05:
        parser.error("supported lift command frequencies are 4 and 60 Hz; step must lie in (0, 0.05] m")
    if any(not 0 < value <= 1 for value in args.interpolation_fractions) or not 0 <= args.grasp_settle_s <= 2:
        parser.error("interpolation fractions must lie in (0, 1] and grasp settling in [0, 2] seconds")
    source, generated = prepare()
    if args.prepare_only:
        print(json.dumps(generated, indent=2))
        return
    evidence_root = PROJECT / "data/test/trajdata" / args.run_id / "0_to_1"
    trajectory_root = PROJECT / "data/test/trajectorys" / args.run_id / "0_to_1"
    evidence_root.mkdir(parents=True, exist_ok=False)
    trajectory_root.mkdir(parents=True, exist_ok=False)
    reference = PROJECT / "data/0_to_1_1014_tcp/0_to_1_evidence"
    runtime = Path(sys.executable).parent
    os.environ["PATH"] = os.pathsep.join([str(runtime), str(runtime / "Scripts"), str(runtime / "Library/bin"), os.environ.get("PATH", "")])
    status = dict(mode="gui", scope="isolated source-0 lift experiment; no production integration",
                  lift_command_hz=args.lift_command_hz, lift_step_m=args.lift_step_m,
                  reference=str(reference), generated=generated, trials=[])
    snapshot = evidence_root / "experiment_source"
    snapshot.mkdir()
    for name in ("run_sweep.py", "policy.py", "generated_probe.py", "generated_manifest.json", "README.md"):
        (snapshot / name).write_bytes((HERE / name).read_bytes())
    cache = install_shared_runtime(evidence_root)
    try:
        for cap, fraction in itertools.product(args.joint_caps, args.interpolation_fractions):
            label = f"joint_{cap:.3f}_interp_{fraction:.4f}".replace(".", "p")
            evidence, trajectory = evidence_root / label, trajectory_root / label
            command = [str(HERE / "generated_probe.py"), "--backend", "isaacsim", "--device", "cuda:0",
                       "--first-probe", str(reference / "first_probe_01/report.json"),
                       "--targets-report", str(reference / "targets.json"),
                       "--world-report", str(reference / "world/report.json"),
                       "--output", str(evidence), "--trajectory-output", str(trajectory),
                       "--stop-after-lift", "--capture-execution", "--camera-width", "256", "--camera-height", "256",
                       "--physics-hz", "120", "--contact-velocity-iterations", "4", "--contact-policy", "run003",
                       "--pick-lift-step-m", "0.001", "--pick-lift-command-hz", "60", "--pick-lift-speed-mode", "step",
                       "--experiment-lift-joint-delta-rad", str(cap)]
            command.extend(["--experiment-interpolation-fraction", str(fraction),
                            "--experiment-lift-command-hz", str(args.lift_command_hz),
                            "--experiment-lift-step-m", str(args.lift_step_m),
                            "--experiment-grasp-settle-s", str(args.grasp_settle_s)])
            entry = dict(label=label, joint_cap_rad=cap, interpolation_fraction=fraction,
                         grasp_settle_s=args.grasp_settle_s, command=[sys.executable, *command],
                         trajectory_directory=str(trajectory), evidence_directory=str(evidence))
            status["trials"].append(entry)
            (evidence_root / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            print(f"SWEEP_START {label}: lift {args.lift_command_hz:g} Hz / {args.lift_step_m*100:g} cm; GUI / physical 120 Hz / recording 60 Hz", flush=True)
            sys.argv = command
            try:
                exec(compile(source, command[0], "exec"), {"__name__": "__main__", "__file__": command[0]})
            except SystemExit as result:
                entry["exit_code"] = result.code
            except Exception as error:
                entry["runtime_error"] = repr(error)
                raise
            if (evidence / "report.json").exists():
                report = read(evidence / "report.json")
                entry.update(execution_passed=report["passed"], failure_reason=report.get("failure_reason"),
                             lift=report.get("lift"), visualization=report.get("visualization"))
            (evidence_root / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            print("SWEEP_FINISH " + json.dumps(entry), flush=True)
            if entry.get("execution_passed") is False and not report.get("record_count"):
                print("SWEEP_STOP: startup failure; remaining motion trials were not run", flush=True)
                break
    finally:
        if "real_close" in cache:
            cache["real_close"]()
        if "launcher" in cache:
            cache["launcher"].app.close()


if __name__ == "__main__":
    main()
