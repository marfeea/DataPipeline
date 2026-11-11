"""Shared physical pick/place execution for the primary and compatibility entries.

Run with the existing Isaac Python. Frozen hybrid trajectories supply joint
targets; all noninitial observations result from actual PhysX steps.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from time import monotonic

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from scripts.pick_place_reward_candidate import RewardTask, rotation_error
from scripts.pipelines.curobo_bootstrap.contact_timing import (
    configure_contact_timing,
    configure_flange_velocity_limit,
    configure_wrist_velocity_limit,
)
from scripts.pipelines.curobo_bootstrap.physical_failure import (
    GraspEvidence,
    make_reward_model,
    reward_observation,
    write_artifacts,
)
from scripts.pipelines.curobo_bootstrap.physical_reward_validity import finalize_reward_files
from scripts.pipelines.curobo_bootstrap.pick_place_artifact import (
    load_hybrid_trajectory,
    save_json,
    sha256_file,
    verify_hybrid_artifact,
)
from scripts.pipelines.curobo_bootstrap.replay import matrix_pose_wxyz
from scripts.pipelines.curobo_bootstrap.robot_adapter import pose_matrix
from scripts.pipelines.curobo_bootstrap.stable_placement_execution import execute_stable_placement, placement_profile
from scripts.pipelines.curobo_bootstrap.task_targets import require_current_tcp_targets

PHASES = {
    "free_space_to_pre_pick": "reach",
    "legacy_approach_pick": "reach",
    "close_gripper": "close",
    "legacy_lift": "lift",
    "free_space_transfer": "transport",
    "legacy_place_approach": "place",
    "open_gripper": "release",
}
PROTECTED = (
    "scripts/collect_isaacsim.py",
    "scripts/probe_curobo_v2_legacy_contact_isaac.py",
    "scripts/native/scene.py",
    "scripts/pipelines/curobo_bootstrap/pick_place_artifact.py",
    "scripts/pick_place_reward_candidate.py",
    "scripts/pick_place_reward_candidate_v2.py",
)


def array(value):
    return value.detach().cpu().numpy().astype(float) if hasattr(value, "detach") else np.asarray(value, dtype=float)


def add_physical_arguments(parser):
    parser.add_argument("--continue-on-task-failure", action=argparse.BooleanOptionalAction, default=False,
                        help="Record failed task checks and finish the route; collection enables this by default")
    parser.add_argument("--stable-placement", action=argparse.BooleanOptionalAction, default=False,
                        help="Replace frozen placement with the five searched object release positions")
    parser.add_argument("--scenario", choices=("natural", "suppress_closure", "open_after_grasp"), default="natural")
    parser.add_argument("--gripper-effort-scale", type=float, default=1.0)
    parser.add_argument("--lift-hold-s", type=float, default=0.35)
    parser.add_argument("--observation-tail-s", type=float, default=0.75)
    parser.add_argument("--visualization-output", type=Path,
                        help="Separate native RGB directory referenced by report.json")
    parser.add_argument("--kit-cache-dir", type=Path, default=ROOT / "data/isaacsim_kit_cache")
    parser.add_argument("--contact-velocity-iterations", type=int, default=0)
    parser.add_argument("--contact-flange-velocity-limit-rad-s", type=float)
    parser.add_argument("--contact-wrist-velocity-limit-rad-s", type=float)


def validate_physical_arguments(parser, args):
    if args.physics_hz < 60 or args.physics_hz % 60:
        parser.error("physics-hz must be a multiple of 60")
    if not 0 < args.gripper_effort_scale <= 1 or not np.isfinite(args.gripper_effort_scale):
        parser.error("gripper-effort-scale must be in (0,1]")
    if any(not np.isfinite(x) or x < 0.25 for x in (args.lift_hold_s, args.observation_tail_s)):
        parser.error("hold/tail durations must be finite and at least 0.25 s")
    if not 0 <= args.contact_velocity_iterations <= 255:
        parser.error("contact-velocity-iterations must be in [0, 255]")
    for limit in (args.contact_flange_velocity_limit_rad_s, args.contact_wrist_velocity_limit_rad_s):
        if limit is not None and (not np.isfinite(limit) or not 0 < limit <= 3.14):
            parser.error("physical joint velocity limits must be finite and in (0, 3.14]")
    if args.contact_flange_velocity_limit_rad_s is not None and args.contact_wrist_velocity_limit_rad_s is not None:
        parser.error("select either the flange limit or the wrist limit")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    add_physical_arguments(parser)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physics-hz", type=int, default=120)
    args = parser.parse_args(argv)
    validate_physical_arguments(parser, args)
    args.backend = "isaacsim"
    return args


def execute_physical_route(segments, execute, *, lift_hold_s, task_done, should_abort,
                           continue_on_failure=False, place=None):
    """Keep physical execution independent of terminal task/reward evaluation."""
    last_index = 0
    for index in range(1, len(segments)):
        if should_abort():
            raise RuntimeError("SimulationApp stopped during experiment")
        if segments[index] == "free_space_transfer" and segments[index - 1] == "legacy_lift":
            for _ in range(round(lift_hold_s * 60)):
                if should_abort():
                    raise RuntimeError("SimulationApp stopped during experiment")
                execute(index - 1, stage="physical_lift_hold", phase="lift")
                if task_done() and not continue_on_failure:
                    break
        if task_done() and not continue_on_failure:
            break
        if place is not None and segments[index] == "legacy_place_approach":
            place(index)
            return index, True
        execute(index)
        last_index = index
        if task_done() and not continue_on_failure:
            break
    return last_index, False


def run_physical_collection(args):
    """Execute a frozen route with measured dynamics and preserve every outcome."""
    runtime = Path(sys.executable).parent
    os.environ["PATH"] = os.pathsep.join(
        [str(runtime), str(runtime / "Scripts"), str(runtime / "Library/bin"), os.environ.get("PATH", "")]
    )
    reference = args.reference_run.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a new experiment directory: {output}")
    if reference == output or reference in output.parents:
        raise ValueError("experiment output must be outside the reference artifact")
    verify_hybrid_artifact(reference)
    planned = load_hybrid_trajectory(reference / "trajectory.npz")
    snapshot = json.loads((reference / "scene_snapshot.json").read_text(encoding="utf-8"))
    source_plan = json.loads((reference / "plan_report.json").read_text(encoding="utf-8"))
    task = snapshot["task"]
    require_current_tcp_targets(task)
    stable_profile = placement_profile(task) if args.stable_placement else None
    if task["control_hz"] != 60:
        raise ValueError("reference control rate must be 60 Hz")
    segments = np.full(len(planned["time_s"]), "", dtype="<U32")
    for name, (start, end) in source_plan["trajectory_acceptance"]["segment_ranges"].items():
        segments[start:end] = name
    if any(name not in PHASES for name in segments):
        raise ValueError("reference segment ranges do not cover all samples")
    protected = {str(ROOT / p): sha256_file(ROOT / p) for p in PROTECTED}
    experiment_sources = {
        str(ROOT / p): sha256_file(ROOT / p)
        for p in (
            "scripts/collect_physical_failure_isaacsim.py",
            "scripts/pipelines/curobo_bootstrap/physical_execution.py",
            "scripts/pipelines/curobo_bootstrap/contact_timing.py",
            "scripts/audit_physical_failure.py",
            "scripts/pipelines/curobo_bootstrap/physical_failure.py",
            "scripts/pipelines/curobo_bootstrap/physical_reward_validity.py",
            "scripts/pipelines/curobo_bootstrap/stable_placement_execution.py",
            "scripts/pipelines/curobo_bootstrap/placement_search.py",
            "configs/task.py",
            "scripts/pipelines/curobo_bootstrap/replay_visualization.py",
            "scripts/pipelines/curobo_bootstrap/task_targets.py",
            "scripts/pipelines/curobo_bootstrap/route_identity.py",
            "scripts/pipelines/curobo_bootstrap/measured_placement.py",
        )
    }
    inputs = {
        str(reference / p): sha256_file(reference / p)
        for p in ("trajectory.npz", "scene_snapshot.json", "plan_report.json")
    }
    output.mkdir(parents=True)
    source_snapshots = {}
    for source, digest in {**protected, **experiment_sources}.items():
        archived = output / "execution_sources" / Path(source).relative_to(ROOT)
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_bytes(Path(source).read_bytes())
        if sha256_file(archived) != digest:
            raise ValueError("experiment source changed during snapshot")
        source_snapshots[source] = str(archived)
    report = {
        "schema": "physical_failure_experiment/1",
        "passed": False,
        "task_succeeded": False,
        "stage": "startup",
        "backend": "isaacsim",
        "mode": "headless" if args.headless else "gui",
        "reference_snapshot": str(reference / "scene_snapshot.json"),
        "input_sha256": inputs,
        "protected_source_sha256": protected,
        "experiment_source_sha256": experiment_sources,
        "execution_source_snapshot_paths": source_snapshots,
        "scenario": args.scenario,
        "gripper_effort_scale": args.gripper_effort_scale,
        "source_index": task["source_index"],
        "destination_index": task["destination_index"],
        "task_hash": task["task_hash"],
        "joint_names": planned["joint_names"].tolist(),
        "execution_contract": {
            "physics_hz": args.physics_hz,
            "control_hz": 60,
            "target_state_writes_after_reset": 0,
            "arm_state_writes_after_reset": 0,
            "logical_attachment": False,
            "camera_resolution": [256, 256],
            "lift_hold_s": args.lift_hold_s,
            "observation_tail_s": args.observation_tail_s,
            "continue_on_task_failure": args.continue_on_task_failure,
        },
        "outcome": "unknown",
        "injection_applied": False,
        "scope": "independent joint-target physical execution of frozen route; all outcomes retained",
    }
    if stable_profile is not None:
        report["placement_profile"] = stable_profile
        report["execution_contract"]["placement_control"] = "direct_pre_place_tcp"
        report["scope"] = "frozen transport followed by direct pre-place TCP execution; grasp evidence is recorded"
    save_json(output / "report.json", report)
    env = launcher = recorder = contacts = None
    records, rewards = [], []
    reward_task = model = None
    try:
        faulthandler.dump_traceback_later(90, repeat=True)
        # Reuse launch/scene implementation; do not change their default behavior.
        from scripts.sim_backend import launch

        print("PHYSICAL_STARTUP SimulationApp", flush=True)
        launcher = launch(args)
        print("PHYSICAL_STARTUP importing native scene", flush=True)
        import torch

        from isaacsim.core.api.sensors import RigidContactView
        from pxr import PhysxSchema, Usd, UsdPhysics

        from configs.assets import ROBOT_PRIM_CONTRACT, SCENE_ENTITY_AUBO, SCENE_ENTITY_TARGET
        from scripts.native.ik_controller import AuboIKAction
        from scripts.native.scene import ENV_ROOT, NativeTrajectoryEnv, NativeTrajectoryEnvCfg
        from scripts.pipelines.curobo_bootstrap.physics_contact_reports import PhysicsContactReports
        from scripts.pipelines.curobo_bootstrap.replay_visualization import ReplayFrameRecorder
        from scripts.pipelines.curobo_bootstrap.source_target import set_source_target_pose

        cfg = NativeTrajectoryEnvCfg()
        cfg.sim.device, cfg.seed = args.device, task["seed"]
        report["execution_contract"].update(configure_contact_timing(
            cfg, physics_hz=args.physics_hz, velocity_iterations=args.contact_velocity_iterations,
        ))
        flange_drive = configure_flange_velocity_limit(
            cfg, robot_name=SCENE_ENTITY_AUBO, limit_rad_s=args.contact_flange_velocity_limit_rad_s,
        )
        wrist_drive = configure_wrist_velocity_limit(
            cfg, robot_name=SCENE_ENTITY_AUBO, limit_rad_s=args.contact_wrist_velocity_limit_rad_s,
        )
        # Keep the constructed environment available if first-render shader
        # loading exhausts the legacy camera warmup timeout.
        cfg.camera_resolution = None
        report["stage"] = "scene_initialization"
        save_json(output / "report.json", report)
        print("PHYSICAL_STARTUP building scene", flush=True)
        env = NativeTrajectoryEnv(cfg)
        print("PHYSICAL_STARTUP physics scene ready", flush=True)
        env.reset()
        robot, target = env.scene[SCENE_ENTITY_AUBO], env.scene[SCENE_ENTITY_TARGET]
        set_source_target_pose(env, task)
        arm_ids, _ = robot.find_joints(ROBOT_PRIM_CONTRACT.arm_joints)
        finger_ids, _ = robot.find_joints(ROBOT_PRIM_CONTRACT.gripper_joints)
        for drive, names, ids in (
            (flange_drive, ROBOT_PRIM_CONTRACT.arm_joints[-1:], arm_ids[-1:]),
            (wrist_drive, ROBOT_PRIM_CONTRACT.arm_joints[-3:], arm_ids[-3:]),
        ):
            if drive is None:
                continue
            limits = array(robot.root_physx_view.get_dof_max_velocities()[0])
            actual = {name: float(limits[joint]) for name, joint in zip(names, ids, strict=True)}
            if any(not np.isclose(value, drive["velocity_limit_rad_s"], atol=1e-6, rtol=0)
                   for value in actual.values()):
                raise ValueError("physical joint velocity limit was not applied")
            drive["physx_velocity_limits_rad_s"] = actual
            report["execution_contract"]["flange_drive" if drive is flange_drive else "wrist_drive"] = drive
        positions, velocities = robot.data.joint_pos.clone(), torch.zeros_like(robot.data.joint_vel)
        positions[:, arm_ids] = torch.as_tensor(planned["joint_position"][0], dtype=positions.dtype, device=args.device)
        positions[:, finger_ids] = torch.as_tensor(
            planned["gripper_target"][0], dtype=positions.dtype, device=args.device
        )
        robot.write_joint_state_to_sim(positions, velocities)
        robot.set_joint_position_target(positions[:, arm_ids], joint_ids=arm_ids)
        robot.set_joint_position_target(positions[:, finger_ids], joint_ids=finger_ids)
        env.sim.forward()
        env.scene.update(cfg.sim.dt)

        # Enforce, not merely describe, the absence of carrying/state replay.
        def forbidden_target_write(*unused, **kwargs):
            report["execution_contract"]["target_state_writes_after_reset"] += 1
            raise RuntimeError("Target state writes are forbidden after experiment reset")

        def forbidden_arm_write(*unused, **kwargs):
            report["execution_contract"]["arm_state_writes_after_reset"] += 1
            raise RuntimeError("Arm state writes are forbidden after experiment reset")

        target.write_root_pose_to_sim = forbidden_target_write
        target.write_root_velocity_to_sim = forbidden_target_write
        target_body = target.view.prim_paths[0]
        prim = env.scene.stage.GetPrimAtPath(target_body)
        body = UsdPhysics.RigidBodyAPI(prim)
        report["target_physics"] = {
            "body_path": target_body,
            "rigid_body_enabled": body.GetRigidBodyEnabledAttr().Get(),
            "kinematic_enabled": body.GetKinematicEnabledAttr().Get(),
            "gravity_disabled": PhysxSchema.PhysxRigidBodyAPI(prim).GetDisableGravityAttr().Get(),
        }
        if (
            not report["target_physics"]["rigid_body_enabled"]
            or report["target_physics"]["kinematic_enabled"]
            or report["target_physics"]["gravity_disabled"]
        ):
            raise ValueError("target must be dynamic with gravity")
        robot_root = ENV_ROOT + "/AUBObot/AUBO_E5"
        finger_paths = []
        for name in ROBOT_PRIM_CONTRACT.gripper_joints:
            matches = [
                str(p.GetPath())
                for p in Usd.PrimRange(env.scene.stage.GetPrimAtPath(robot_root))
                if p.GetName() == name and p.IsA(UsdPhysics.Joint)
            ]
            if len(matches) != 1:
                raise ValueError(f"cannot identify finger joint: {name}")
            joint = UsdPhysics.Joint(env.scene.stage.GetPrimAtPath(matches[0]))
            bodies = joint.GetBody1Rel().GetTargets()
            if len(bodies) != 1:
                raise ValueError(f"finger joint lacks a unique child body: {name}")
            finger = env.scene.stage.GetPrimAtPath(bodies[0])
            while finger and not finger.HasAPI(UsdPhysics.RigidBodyAPI):
                finger = finger.GetParent()
            if not finger:
                raise ValueError(f"finger joint child has no rigid body: {name}")
            finger_paths.append(str(finger.GetPath()))
        support_paths = [
            str(p.GetPath())
            for p in Usd.PrimRange(
                env.scene.stage.GetPrimAtPath(ENV_ROOT + "/station/static"), Usd.TraverseInstanceProxies()
            )
            if p.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False
        ]
        if not support_paths or len(set(finger_paths)) != 2:
            raise ValueError("contact filters need distinct fingers and explicit support bodies")
        # PhysX tensor filter expressions must each resolve to one body.
        # Aggregate the individual workstation/tray channels only after reading.
        filters = [*finger_paths, *support_paths]
        sensor = RigidContactView(
            target_body,
            filters,
            name="failure_target_contacts",
            prepare_contact_sensors=False,
            disable_stablization=False,
            max_contact_count=512,
        )
        sensor.initialize(env.world.physics_sim_view)
        report["contact_sensor"] = {
            "filters": filters,
            "num_shapes": sensor.num_shapes,
            "num_filters": sensor.num_filters,
            "support_body_paths": support_paths,
            "source": "PhysX filtered contact tensors, sampled each physics step",
        }
        if sensor.num_shapes < 1 or sensor.num_filters != len(filters):
            raise ValueError("PhysX contact view shape/filter initialization failed")
        contacts = PhysicsContactReports()
        baseline_efforts = array(robot.view.get_max_efforts(joint_indices=finger_ids))
        effective_efforts = baseline_efforts * args.gripper_effort_scale
        robot.view.set_max_efforts(
            torch.as_tensor(effective_efforts, dtype=torch.float32, device=args.device), joint_indices=finger_ids
        )
        readback = array(robot.view.get_max_efforts(joint_indices=finger_ids))
        if not np.allclose(readback, effective_efforts):
            raise ValueError("gripper effort scale did not apply")
        report["gripper_effort"] = {"baseline_n": baseline_efforts.tolist(), "effective_n": readback.tolist()}
        ik = AuboIKAction(robot, cfg.sim.dt, cfg.decimation)
        print(f"PHYSICAL_STARTUP contact sensors ready: {finger_paths}", flush=True)
        cfg.camera_resolution = (256, 256)
        cfg.camera_diagnostics_directory = (args.visualization_output or output / "execution_visualization") / "warmup"
        try:
            env._create_cameras(cfg.camera_resolution)
        except RuntimeError as error:
            if "Native camera warmup did not complete" not in str(error):
                raise
            report["camera_warmup_recovered_from"] = str(error)
            print("PHYSICAL_STARTUP recovering camera warmup on loaded scene", flush=True)
            started = monotonic()
            while monotonic() - started < 180:
                env.sim.render()
                try:
                    for camera in env.cameras.values():
                        camera.refresh()
                        camera.render_reference()
                        if np.std(camera.camera.get_rgb()) <= 5:
                            raise RuntimeError("camera still has a placeholder image")
                    if env.background_only_render():
                        raise RuntimeError("camera scene geometry is still loading")
                    env.cameras_ready = True
                    break
                except RuntimeError:
                    continue
            else:
                raise RuntimeError("Independent camera warmup recovery timed out") from error
        print("PHYSICAL_STARTUP cameras ready", flush=True)
        robot.write_joint_state_to_sim = forbidden_arm_write
        report["execution_contract"]["camera_warmup_pose"] = "initial trajectory pose, matching legacy execution"
        recorder = ReplayFrameRecorder(
            args.visualization_output or output / "execution_visualization", env.camera_specs
        )
        cameras = {s.name: env.scene[s.name] for s in env.camera_specs}
        if not args.headless:
            report["execution_contract"]["camera_capture_policy"] = "periodic_plus_stage_and_gripper_transitions"
        goal = matrix_pose_wxyz(
            np.asarray(task["frames"]["T_WE"])
            @ pose_matrix(task["destination"]["position_e"], task["destination"]["rotation_wxyz"])
        )
        grasp = (np.asarray(task["frames"]["T_WE"]) @ np.r_[task["source"]["pre_grasp_tcp_position_e"], 1])[:3]
        reward_task = RewardTask(
            tuple(grasp),
            tuple(array(target.data.root_pos_w[0])),
            tuple(goal),
            (task.get("state4_routing") or {}).get("pick_lift_m", 0.05),
        )
        detector = None
        terminal_index = None
        injection_open = False
        command_q = np.asarray(planned["joint_position"][0])
        command_g = np.asarray(planned["gripper_target"][0])
        requested_g = command_g.copy()
        requested_tcp = None

        def observe(index, stage, phase, stage_id, contact_forces, contact_available, *, timeout=False):
            nonlocal detector, model, terminal_index
            env.sim.render()
            tcp_b, tcp_q_b, _ = ik.current_tcp_pose_b()
            tcp_pose_b = [*array(tcp_b[0]), *array(tcp_q_b[0])]
            tcp_q = np.asarray(tcp_pose_b[3:])
            tcp_pose_w = matrix_pose_wxyz(
                np.asarray(task["frames"]["T_WB"]) @ pose_matrix(tcp_pose_b[:3], tcp_q / np.linalg.norm(tcp_q))
            )
            sample = {
                "time_s": len(records) / 60,
                "stage": stage,
                "phase": phase,
                "stage_id": stage_id,
                "reference_index": index,
                "q": array(robot.data.joint_pos[0, arm_ids]).tolist(),
                "dq": array(robot.data.joint_vel[0, arm_ids]).tolist(),
                "tcp_position_b": tcp_pose_b[:3],
                "tcp_quaternion_b": tcp_pose_b[3:],
                "tcp_pose_w": tcp_pose_w,
                "target_pose_w": [*array(target.data.root_pos_w[0]), *array(target.data.root_quat_w[0])],
                "target_position_w": array(target.data.root_pos_w[0]).tolist(),
                "target_velocity_w": array(target.view.get_velocities()[0]).tolist(),
                "gripper_position": array(robot.data.joint_pos[0, finger_ids]).tolist(),
                "gripper_velocity": array(robot.data.joint_vel[0, finger_ids]).tolist(),
                "commanded_joint_target": command_q.tolist(),
                "executed_gripper_target": command_g.tolist(),
                "requested_gripper_target": requested_g.tolist(),
                "contact_force_n": contact_forces.tolist(),
                "contact_available": bool(contact_available),
                "physics_contacts": contacts.sample(),
                "state_write_after_reset": False,
                "fault_injection_active": injection_open,
                "event_code": "NONE",
            }
            if requested_tcp is not None:
                sample["requested_tcp_pose_w"] = requested_tcp
            limits = array(robot.data.joint_pos_limits[0, arm_ids])
            sample["joint_limit"] = bool(
                np.any(np.asarray(sample["q"]) < limits[:, 0]) or np.any(np.asarray(sample["q"]) > limits[:, 1])
            )
            if records:
                sample["normal_supported_release"] = bool(
                    phase == "release"
                    and not injection_open
                    and (
                        records[-1]["evidence"]["support_verified"] or (contact_available and contact_forces[2] > 0.01)
                    )
                    and np.linalg.norm(np.asarray(sample["target_pose_w"][:3]) - goal[:3]) < 0.02
                    and rotation_error(sample["target_pose_w"][3:], goal[3:]) < np.deg2rad(10)
                )
            if detector is None:
                detector = GraspEvidence(sample)
            sample["evidence"] = detector.step(sample)
            obs = reward_observation(sample, sample["evidence"], timeout=timeout)
            if model is None:
                model = make_reward_model(reward_task, obs)
                report["reward_config"] = asdict(model.config)
            elif not model.done:
                result = model.step(obs)
                result.update(
                    transition_valid=True,
                    from_observation=len(records) - 1,
                    to_observation=len(records),
                    post_terminal_observation=False,
                )
                rewards.append(result)
                if result["terminated"]:
                    terminal_index = len(records)
                    report["terminal_failure"] = result["failure"]
                    report["task_succeeded"] = result["failure"] is None
                    if report["task_succeeded"]:
                        sample["event_code"] = "PLACE"
            else:
                rewards.append(
                    dict(
                        reward=0.0,
                        reward_valid=False,
                        terminated=False,
                        transition_valid=False,
                        terms={},
                        failure=None,
                        from_observation=len(records) - 1,
                        to_observation=len(records),
                        post_terminal_observation=True,
                    )
                )
            if sample["evidence"]["ever_grasp_verified"] and not any(r["event_code"] == "PICK" for r in records):
                sample["event_code"] = "PICK"
            records.append(sample)
            recorder.capture(
                cameras,
                render_frame=len(records),
                trajectory_index=len(records) - 1,
                time_s=sample["time_s"],
                event=sample["event_code"],
                force=not args.headless and (
                    len(records) == 1 or records[-2]["stage"] != stage or phase == "close"
                    or stage in {"open_gripper", "stable_open_gripper"}
                ),
            )
            if len(records) % 120 == 0:
                print(
                    f"PHYSICAL_PROGRESS samples={len(records)} stage={stage} "
                    f"grasp={detector.ever_grasped} dropped={detector.dropped}",
                    flush=True,
                )

        def execute(
            index, *, stage=None, phase=None, timeout=False, q_override=None, g_override=None, tcp_override=None
        ):
            nonlocal command_q, command_g, requested_g, requested_tcp, injection_open
            command_q = np.asarray(planned["joint_position"][index] if q_override is None else q_override)
            requested_g = np.asarray(planned["gripper_target"][index] if g_override is None else g_override).copy()
            command_g = requested_g.copy()
            requested_tcp = tcp_override
            name = stage or str(segments[index])
            current_phase = phase or PHASES[str(segments[index])]
            if args.scenario == "suppress_closure" and current_phase in {"close", "lift", "transport", "place"}:
                injection_open = True
            if args.scenario == "open_after_grasp" and detector.ever_grasped:
                injection_open = True
            if injection_open:
                command_g = np.array([0.0115, -0.0115])
                report["injection_applied"] = True
                report.setdefault("injection_observation", len(records) - 1)
            robot.set_joint_position_target(
                torch.as_tensor(command_q, dtype=torch.float32, device=args.device)[None], joint_ids=arm_ids
            )
            robot.set_joint_position_target(
                torch.as_tensor(command_g, dtype=torch.float32, device=args.device)[None], joint_ids=finger_ids
            )
            contacts.begin_control_step()
            forces, available = [], True
            for _ in range(cfg.decimation):
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(cfg.sim.dt)
                matrix = sensor.get_contact_force_matrix(dt=cfg.sim.dt)
                if matrix is None or sensor.num_filters != len(filters) or sensor.num_shapes < 1:
                    available = False
                    continue
                force = array(matrix)
                if not np.isfinite(force).all():
                    available = False
                    continue
                channels = np.linalg.norm(force, axis=2).sum(axis=0)
                forces.append(np.r_[channels[:2], channels[2:].sum()])
            mean_forces = np.mean(forces, axis=0) if forces else np.zeros(3)
            available = available and len(forces) == cfg.decimation
            observe(
                index, name, current_phase, str(planned["stage_id"][index]), mean_forces, available, timeout=timeout
            )
            return records[-1]

        observe(0, str(segments[0]), PHASES[str(segments[0])], str(planned["stage_id"][0]), np.zeros(3), False)
        report["stage"] = "physical_execution"
        def place(index):
            def placement_step(q, fingers, stage, phase, tcp_pose):
                return execute(index, stage=stage, phase=phase, q_override=q,
                               g_override=fingers, tcp_override=tcp_pose)

            report["stable_placement"] = execute_stable_placement(
                task, stable_profile, records[-1], placement_step,
                lambda: not launcher.app.is_running() or (
                    not args.continue_on_task_failure and model.done and not report["task_succeeded"]
                ),
                continue_on_failure=args.continue_on_task_failure,
            )

        last_index, placement_executed = execute_physical_route(
            segments, execute, lift_hold_s=args.lift_hold_s,
            task_done=lambda: model.done, should_abort=lambda: not launcher.app.is_running(),
            continue_on_failure=args.continue_on_task_failure,
            place=place if stable_profile is not None else None,
        )
        report["trajectory_execution_completed"] = (
            report["stable_placement"].get("execution_completed", False)
            if placement_executed else last_index == len(segments) - 1
        )
        # Keep raw observations after an early terminal for failure visualization;
        # transition_valid=false prevents additional reward or training after done.
        for tail in range(round(args.observation_tail_s * 60)):
            execute(
                last_index,
                stage="physical_observation_tail",
                timeout=tail == round(args.observation_tail_s * 60) - 1 and not model.done,
                phase="release" if placement_executed else None,
                q_override=command_q if placement_executed else None,
                g_override=requested_g if placement_executed else None,
                tcp_override=requested_tcp if placement_executed else None,
            )
        report["visualization"] = recorder.finish(source_frames=len(records), render_calls=len(records))
        report["terminal_observation"] = terminal_index
        if detector.dropped:
            report["outcome"] = "dropped"
        elif report["task_succeeded"]:
            report["outcome"] = "success"
        elif detector.ever_grasped:
            report["outcome"] = "grasp_verified_task_incomplete"
        else:
            report["outcome"] = "grasp_not_verified"
        report["contact_positive_controls"] = detector.seen_channels.tolist()
        report.update(stage="complete", passed=True)
        return_code = 0
    except Exception as error:
        report["failure_reason"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
        return_code = 1
    finally:
        faulthandler.cancel_dump_traceback_later()
        if contacts is not None:
            report["physics_contact_report"] = contacts.finish()
        if records and reward_task is not None:
            write_artifacts(output, records, rewards, task, reward_task, report)
            finalize_reward_files(output, rewards, report)
        if recorder is not None and "visualization" not in report:
            try:
                report["visualization"] = recorder.finish(source_frames=len(records), render_calls=len(records))
            except Exception as error:
                report["visualization_error"] = str(error)
        report["inputs_unchanged"] = all(sha256_file(p) == h for p, h in inputs.items())
        report["original_collection_unchanged"] = all(sha256_file(p) == h for p, h in protected.items())
        report["experiment_sources_unchanged"] = all(sha256_file(p) == h for p, h in experiment_sources.items())
        report["passed"] = report["passed"] and report["inputs_unchanged"] and report["original_collection_unchanged"]
        report["isaaclab_modules_loaded"] = [n for n in sys.modules if n == "isaaclab" or n.startswith("isaaclab.")]
        save_json(output / "report.json", report)
        save_json(output / "plan_report.json", report)
        if env is not None:
            env.close()
        if launcher is not None:
            launcher.app.close()
    print(f"PHYSICAL_RESULT passed={report['passed']} outcome={report['outcome']} output={output}", flush=True)
    return return_code if report["passed"] else 1
