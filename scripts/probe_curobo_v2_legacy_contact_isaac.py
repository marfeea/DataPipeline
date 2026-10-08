"""Exercise Legacy IK pick and lift in the V2 production Isaac scene."""

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from DataPipe.runtime import ensure_curriculumrl_available

ensure_curriculumrl_available()
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--first-probe", type=Path, required=True)
parser.add_argument("--targets-report", type=Path, required=True)
parser.add_argument("--world-report", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--max-steps-per-stage", type=int, default=600)
parser.add_argument("--physics-hz", type=int, default=120)
parser.add_argument("--contact-velocity-iterations", type=int, default=0)
parser.add_argument("--contact-flange-velocity-limit-rad-s", type=float)
parser.add_argument("--contact-wrist-velocity-limit-rad-s", type=float)
parser.add_argument(
    "--capture-execution", action="store_true", help="Record live three-camera RGB during execution, including failures"
)
parser.add_argument("--camera-width", type=int, default=640)
parser.add_argument("--camera-height", type=int, default=480)
parser.add_argument("--contact-policy", choices=("strict", "run003"), default="strict")
parser.add_argument("--place-high-q", type=float, nargs=6)
parser.add_argument(
    "--stop-after-lift", action="store_true", help="Save measured pick/lift evidence for transfer-goal search"
)
parser.add_argument(
    "--integrated", action="store_true", help="Plan and state-replay both CuRobo segments in this Isaac run"
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.physics_hz < 60 or args.physics_hz % 60:
    parser.error("physics-hz must be a positive integer multiple of 60")
if not 0 <= args.contact_velocity_iterations <= 255:
    parser.error("contact-velocity-iterations must be in [0, 255]")
if args.contact_flange_velocity_limit_rad_s is not None and args.contact_wrist_velocity_limit_rad_s is not None:
    parser.error("select either the flange limit or the wrist limit")
if args.camera_width <= 0 or args.camera_height <= 0:
    parser.error("camera dimensions must be positive")
if args.capture_execution:
    args.enable_cameras = True
if args.output.exists():
    raise FileExistsError(args.output)
launcher = AppLauncher(args)

import numpy as np
import torch
import trimesh
from CurriculumRL.configs.assets import SCENE_ENTITY_AUBO, SCENE_ENTITY_TARGET
from CurriculumRL.logic.aubo_ik import flange_down_yaw_offset_for_tcp_heading
from CurriculumRL.tasks.tcp_docking.aubo_ik_action import AuboIKAction
from CurriculumRL.tasks.tcp_docking.trajectory_env_cfg import TcpDockingTrajectoryEnv, TcpDockingTrajectoryEnvCfg

from configs.coordinate_frames import TCP_DIRECTION_AXIS_T
from configs.curobo.planner import PlannerConfig
from configs.curobo.world import WorldConfig
from DataPipe.camera_config import project_diagnostic_camera_sensor_specs
from DataPipe.collection_cameras import configure_collection_cameras
from DataPipe.config import TCP_POSITION_SCALE_M, TCP_ROTATION_SCALE_RAD
from DataPipe.isaac_gripper import DatasetGripperController
from DataPipe.isaac_scene import _camera_cfg
from DataPipe.pipelines.curobo_bootstrap.attachment_geometry import attached_sphere_from_measured_target
from DataPipe.pipelines.curobo_bootstrap.contact_timing import (
    configure_contact_timing,
    configure_flange_velocity_limit,
    configure_wrist_velocity_limit,
)
from DataPipe.pipelines.curobo_bootstrap.curobo_collision import CuroboCollision
from DataPipe.pipelines.curobo_bootstrap.measured_placement import measured_state4_placement
from DataPipe.pipelines.curobo_bootstrap.physics_contact_reports import PhysicsContactReports
from DataPipe.pipelines.curobo_bootstrap.planner import (
    EXPERIMENTAL_ACCELERATION_LIMIT,
    CuroboMvpPlanner,
    load_trajectory,
    save_trajectory,
    validate_trajectory,
)
from DataPipe.pipelines.curobo_bootstrap.production_world import extract_production_world
from DataPipe.pipelines.curobo_bootstrap.replay import matrix_pose_wxyz
from DataPipe.pipelines.curobo_bootstrap.replay_visualization import ReplayFrameRecorder
from DataPipe.pipelines.curobo_bootstrap.robot_adapter import pose_matrix
from DataPipe.pipelines.curobo_bootstrap.route_experiment import require_height
from DataPipe.pipelines.curobo_bootstrap.route_identity import require_report_identity, task_identity
from DataPipe.pipelines.curobo_bootstrap.source_target import set_source_target_pose
from DataPipe.pipelines.curobo_bootstrap.world_adapter import WorldSnapshot
from DataPipe.sequence import planar_heading_from_quaternion, quaternion_rotation_error_vector


def _array(value):
    return value.detach().cpu().numpy().astype(np.float64)


def main():
    args.output.mkdir(parents=True)
    report = {
        "schema": "curobo_v2_legacy_contact_isaac_probe/1",
        "passed": False,
        "scope": "Isaac Legacy pick/lift contact probe after teleporting to pre-pick; not a full V2 run",
        "stage": "startup",
        "mode": "headless" if args.headless else "gui",
    }
    env = None
    records = []
    recorder = None
    contact_reports = None
    try:
        first = json.loads(args.first_probe.read_text(encoding="utf-8"))
        targets = json.loads(args.targets_report.read_text(encoding="utf-8"))
        frozen_world = json.loads(args.world_report.read_text(encoding="utf-8"))
        task = targets["task_snapshot"]
        require_height(task, task["lift_clearance_m"])
        report["lift_clearance_m"] = task["lift_clearance_m"]
        report["state4_routing"] = task.get("state4_routing")
        report.update(task_identity(task))
        require_report_identity(targets, task, label="targets")
        require_report_identity(first, task, label="first probe", world_hash=frozen_world["world_hash"])
        require_report_identity(frozen_world, task, label="world", require_passed=False)
        if not frozen_world.get("world_geometry_valid"):
            raise ValueError("selected world lacks validated geometry evidence")
        if first["segment"] != "free_space_to_pre_pick":
            raise ValueError("first CuRobo probe has the wrong segment")
        if args.integrated and args.place_high_q is None:
            raise ValueError("--integrated requires --place-high-q")
        if args.integrated and args.stop_after_lift:
            raise ValueError("--stop-after-lift cannot be used with --integrated")
        cfg = TcpDockingTrajectoryEnvCfg()
        cfg.scene.num_envs = 1
        cfg.sim.device = args.device
        cfg.seed = targets["config"]["seed"]
        report["execution_contract"] = configure_contact_timing(
            cfg, physics_hz=args.physics_hz, control_hz=task["control_hz"],
            velocity_iterations=args.contact_velocity_iterations,
        )
        flange_drive = configure_flange_velocity_limit(
            cfg, robot_name=SCENE_ENTITY_AUBO, limit_rad_s=args.contact_flange_velocity_limit_rad_s,
        )
        if flange_drive is not None:
            report["execution_contract"]["flange_drive"] = flange_drive
        wrist_drive = configure_wrist_velocity_limit(
            cfg, robot_name=SCENE_ENTITY_AUBO, limit_rad_s=args.contact_wrist_velocity_limit_rad_s,
        )
        if wrist_drive is not None:
            report["execution_contract"]["wrist_drive"] = wrist_drive
        cfg.actions.joint_trajectory.open_position = (0.0115, -0.0115)
        cfg.actions.joint_trajectory.closed_position = (0.0, 0.0)
        for name in (SCENE_ENTITY_AUBO + "_usd", SCENE_ENTITY_TARGET):
            getattr(cfg.scene, name).spawn.activate_contact_sensors = True
        if args.capture_execution:
            camera_specs = project_diagnostic_camera_sensor_specs(
                resolution=(args.camera_width, args.camera_height), data_types=("rgb",)
            )
            for spec in camera_specs[1:]:
                setattr(cfg.scene, spec.name, _camera_cfg(spec))
            cfg, camera_specs = configure_collection_cameras(cfg, camera_specs)
        env = TcpDockingTrajectoryEnv(cfg)
        env.reset()
        contact_reports = PhysicsContactReports()
        if args.capture_execution:
            cameras = {spec.name: env.scene[spec.name] for spec in camera_specs}
            recorder = ReplayFrameRecorder(args.output / "execution_visualization", camera_specs)
        report["source_target_pose_w"] = set_source_target_pose(env, task)
        actual_world, _ = extract_production_world(env, args.output / "world_assets")
        if actual_world.world_hash != frozen_world["world_evidence"]["world_hash"]:
            raise ValueError("production reset world hash changed")
        robot = env.scene[SCENE_ENTITY_AUBO]
        target = env.scene[SCENE_ENTITY_TARGET]
        arm_ids, arm_names = robot.find_joints(
            ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Flange"], preserve_order=True
        )
        if tuple(arm_names) != ("Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Flange"):
            raise ValueError("arm joint order changed")
        if flange_drive is not None:
            actual_limit = float(robot.root_physx_view.get_dof_max_velocities()[0, arm_ids[-1]])
            if not np.isclose(actual_limit, args.contact_flange_velocity_limit_rad_s, rtol=0, atol=1e-6):
                raise ValueError("contact flange physical velocity limit was not applied")
            flange_drive["physx_velocity_limit_rad_s"] = actual_limit
        if wrist_drive is not None:
            actual_limits = {
                name: float(robot.root_physx_view.get_dof_max_velocities()[0, joint_id])
                for name, joint_id in zip(arm_names[-3:], arm_ids[-3:], strict=True)
            }
            if any(
                not np.isclose(value, args.contact_wrist_velocity_limit_rad_s, rtol=0, atol=1e-6)
                for value in actual_limits.values()
            ):
                raise ValueError("contact wrist physical velocity limits were not applied")
            wrist_drive["physx_velocity_limits_rad_s"] = actual_limits
        report["reset_q"] = _array(robot.data.joint_pos[0, arm_ids]).tolist()
        if np.max(np.abs(np.asarray(report["reset_q"]) - np.asarray(first["start_q"]))) > 1e-5:
            raise RuntimeError("same-run Isaac reset q differs from first CuRobo start")
        if np.max(np.abs(np.asarray(report["reset_q"]) - np.asarray(targets["legacy_ik"]["q_start"]))) > 1e-5:
            raise RuntimeError("same-run Isaac reset q differs from target extraction")
        q_pre = np.asarray(first["goal_q"], dtype=np.float64)
        gripper = DatasetGripperController(robot, open_position=(0.0115, -0.0115), closed_position=(0.0, 0.0))
        ik = AuboIKAction(
            robot,
            env.sim.get_physics_dt(),
            cfg.decimation,
            diagnostics_enabled=True,
            flange_downward_enabled=True,
        )
        root_quaternion = _array(robot.data.root_quat_w[0])
        report["base_pose_w"] = [*_array(robot.data.root_pos_w[0]), *root_quaternion]
        report["target_pose_w_at_reset"] = [
            *_array(target.data.root_pos_w[0]),
            *_array(target.data.root_quat_w[0]),
        ]

        def capture_sample():
            record = records[-1]
            record["target_pose_w"] = [*_array(target.data.root_pos_w[0]), *_array(target.data.root_quat_w[0])]
            record["contact_waypoint"] = report.get("contact_waypoint")
            if len(records) % 120 == 0:
                print(
                    f"CONTACT_PROGRESS frames={len(records)} stage={record['stage']} "
                    f"waypoint={record['contact_waypoint']}",
                    flush=True,
                )
            if record["stage"] not in ("free_space_to_pre_pick", "free_space_transfer"):
                record["commanded_joint_target"] = _array(ik.commanded_joint_target[0]).tolist()
                record["ik_solution_joint_position"] = _array(ik.ik_solution_joint_position[0]).tolist()
            if recorder is None:
                return
            count = recorder.capture(
                cameras,
                render_frame=len(records),
                trajectory_index=len(records) - 1,
                time_s=(len(records) - 1) / task["control_hz"],
                event="NONE",
            )
            if count:
                for frame in recorder.frames[-count:]:
                    frame.update(
                        contact_waypoint=record["contact_waypoint"],
                        execution_stage=record["stage"],
                        joint_position=record["q"],
                        joint_velocity=record["dq"],
                        observed_target_pose_w=record["target_pose_w"],
                    )

        def write_arm(q, dq=None):
            positions = robot.data.joint_pos.clone()
            velocities = robot.data.joint_vel.clone()
            positions[:, arm_ids] = torch.as_tensor(q, dtype=positions.dtype, device=positions.device)
            velocities[:, arm_ids] = torch.as_tensor(
                np.zeros(6) if dq is None else dq, dtype=velocities.dtype, device=velocities.device
            )
            robot.write_joint_state_to_sim(positions, velocities)
            robot.set_joint_position_target(positions[:, arm_ids], joint_ids=arm_ids)
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(env.sim.get_physics_dt())

        def verify_segment(planner, trajectory, start_q, goal_q):
            return validate_trajectory(
                trajectory,
                start_q=start_q,
                goal_q=goal_q,
                position_lower=planner.position_lower,
                position_upper=planner.position_upper,
                velocity_limit=planner.velocity_limit,
                acceleration_limit=EXPERIMENTAL_ACCELERATION_LIMIT,
                control_hz=planner.config.control_hz,
                collision_query=lambda q: planner.collision.query(
                    q, world_ignore_links=planner.world_collision_ignore_links
                ),
            )

        if args.integrated:
            report["scope"] = (
                "One production Isaac run with both CuRobo MotionGen plans, state-replayed free-space segments, "
                "Legacy IK contact, and logical target transfer; integration probe, not final V2 artifact/replay"
            )
            report["stage"] = "curobo_first"
            first_world = actual_world.permit_target_contact()
            report["collision_policy"] = {
                "allowed_world_contact_before_pick": [actual_world.config.target_name],
                "protected_world_meshes": [mesh.name for mesh in first_world.active_meshes],
                "self_collision_hard_gate": True,
            }
            planner = CuroboMvpPlanner(
                frozen_world["collision_model"],
                world=first_world,
                config=PlannerConfig(
                    version="0.7.8", cuda_graph_enabled=False, graph_first=True, max_attempts=4, timeout_s=30
                ),
            )
            planner.warmup()
            first_start = np.asarray(first["start_q"], dtype=np.float64)
            first_trajectory, first_backend = planner.plan(first_start, q_pre)
            first_path = args.output / "first_trajectory.npz"
            save_trajectory(first_path, first_trajectory)
            first_acceptance = verify_segment(planner, load_trajectory(first_path), first_start, q_pre)
            report["curobo_first"] = {"motion_gen": first_backend, "acceptance": first_acceptance}
            if not first_acceptance["passed"]:
                raise RuntimeError("first CuRobo trajectory failed independent collision/kinematic gate")
            for q, dq in zip(first_trajectory.joint_position, first_trajectory.joint_velocity, strict=True):
                write_arm(q, dq)
                env.sim.render()
                records.append(
                    {
                        "stage": "free_space_to_pre_pick",
                        "q": _array(robot.data.joint_pos[0, arm_ids]).tolist(),
                        "dq": _array(robot.data.joint_vel[0, arm_ids]).tolist(),
                        "tcp_position_b": _array(ik.current_tcp_pose_b()[0][0]).tolist(),
                        "gripper_position": list(gripper.state()[0]),
                        "target_position_w": _array(target.data.root_pos_w[0]).tolist(),
                    }
                )
                capture_sample()
        else:
            write_arm(q_pre)
        poses = targets["legacy_ik"]["targets"]
        heading = planar_heading_from_quaternion(poses["pre_pick"]["tcp_pose_b"][3:], TCP_DIRECTION_AXIS_T)
        initial_tcp = _array(ik.current_tcp_pose_b()[0][0])
        initial_error = float(np.linalg.norm(initial_tcp - np.asarray(poses["pre_pick"]["tcp_pose_b"][:3])))
        report.update(
            stage="legacy_approach",
            world_hash=actual_world.world_hash,
            pre_pick_tcp_error_m=initial_error,
            pre_pick_q_error_rad=float(np.max(np.abs(_array(robot.data.joint_pos[0, arm_ids]) - q_pre))),
        )
        if initial_error > 1e-3 or report["pre_pick_q_error_rad"] > 1e-5:
            raise RuntimeError("teleported pre-pick state does not match Legacy TCP target")

        def step_ik(target_b, stage, gripper_command, target_quaternion_b=None):
            contact_reports.begin_control_step()
            current_pose = ik.current_tcp_pose_b()
            current = _array(current_pose[0][0])
            translation = np.clip((target_b - current) / TCP_POSITION_SCALE_M, -1.0, 1.0)
            rotation = np.zeros(3)
            ik.flange_downward_enabled = target_quaternion_b is None
            if target_quaternion_b is not None:
                rotation = np.clip(
                    np.asarray(quaternion_rotation_error_vector(_array(current_pose[1][0]), target_quaternion_b))
                    / TCP_ROTATION_SCALE_RAD,
                    -1.0,
                    1.0,
                )
            raw = torch.as_tensor(
                [*translation, *rotation], dtype=robot.data.joint_pos.dtype, device=robot.data.joint_pos.device
            ).view(1, 6)
            commanded = current + translation * TCP_POSITION_SCALE_M
            ik.downward_yaw_offset_rad = flange_down_yaw_offset_for_tcp_heading(
                commanded,
                heading,
                root_quaternion_wxyz=root_quaternion,
                tcp_heading_axis_local=TCP_DIRECTION_AXIS_T,
                initial_offset_rad=ik.downward_yaw_offset_rad,
            )
            ik.process_actions(raw, action_mask=torch.ones_like(raw), ik_modes=("full_pose_6d",))
            if bool(ik.ik_failed[0]):
                raise RuntimeError(f"Legacy IK failed in {stage}")
            for _ in range(cfg.decimation):
                ik.apply_actions()
                gripper.apply(gripper_command)
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(env.sim.get_physics_dt())
            env.sim.render()
            tcp = _array(ik.current_tcp_pose_b()[0][0])
            record = {
                "stage": stage,
                "q": _array(robot.data.joint_pos[0, arm_ids]).tolist(),
                "dq": _array(robot.data.joint_vel[0, arm_ids]).tolist(),
                "tcp_position_b": tcp.tolist(),
                "requested_tcp_position_b": target_b.tolist(),
                "tcp_quaternion_b": _array(ik.current_tcp_pose_b()[1][0]).tolist(),
                "gripper_position": list(gripper.state()[0]),
                "target_position_w": _array(target.data.root_pos_w[0]).tolist(),
                "ik_position_error_m": float(ik.diagnostics.ik_position_error_m[0]),
                "ik_orientation_error_rad": float(ik.diagnostics.ik_orientation_error_rad[0]),
                "physics_contacts": contact_reports.sample(),
            }
            if target_quaternion_b is not None:
                record["requested_tcp_pose_b"] = [*target_b.tolist(), *target_quaternion_b]
            records.append(record)
            capture_sample()
            return tcp

        def approach(label, target_b, command, tolerance_m=0.01, dwell_steps=3, target_quaternion_b=None):
            initial_count = len(records)
            dwell = 0
            for _ in range(args.max_steps_per_stage):
                tcp = step_ik(target_b, label, command, target_quaternion_b)
                error = float(np.linalg.norm(tcp - target_b))
                rotation_error = (
                    float(
                        np.linalg.norm(
                            quaternion_rotation_error_vector(_array(ik.current_tcp_pose_b()[1][0]), target_quaternion_b)
                        )
                    )
                    if target_quaternion_b is not None
                    else 0.0
                )
                object_rotation_error = (
                    float(
                        np.linalg.norm(
                            quaternion_rotation_error_vector(
                                _array(target.data.root_quat_w[0]),
                                report["placement_calibration"]["desired_object_pose_w"][3:],
                            )
                        )
                    )
                    if target_quaternion_b is not None
                    else 0.0
                )
                dwell = (
                    dwell + 1
                    if error <= tolerance_m and max(rotation_error, object_rotation_error) <= np.deg2rad(1.0)
                    else 0
                )
                if dwell >= dwell_steps:
                    result = {"steps": len(records) - initial_count, "final_tcp_error_m": error}
                    if target_quaternion_b is not None:
                        result["final_tcp_orientation_error_rad"] = rotation_error
                        result["final_object_orientation_error_rad"] = object_rotation_error
                    return result
            raise RuntimeError(f"{label} did not reach TCP tolerance within {args.max_steps_per_stage} steps")

        pick = np.asarray(poses["pick"]["tcp_pose_b"][:3], dtype=np.float64)
        lift = np.asarray(poses["lift"]["tcp_pose_b"][:3], dtype=np.float64)
        contact_points = task.get("contact_waypoint_position_b", {})
        if "pick_entry" in contact_points:
            report["contact_waypoint"] = "state2_pick_entry_xy"
            report["state2_pick_entry"] = approach(
                "legacy_approach_pick", np.asarray(contact_points["pick_entry"]), 1.0,
            )
            report["contact_waypoint"] = "state2_pick_settle_z"
        report["approach"] = approach("legacy_approach_pick", pick, 1.0)
        report["contact_waypoint"] = None
        report["stage"] = "close_gripper"
        close_dwell = 0
        close_steps = 0
        for _ in range(args.max_steps_per_stage):
            step_ik(pick, "close_gripper", -1.0)
            close_steps += 1
            close_error = max(
                abs(actual - desired)
                for actual, desired in zip(gripper.state()[0], gripper.closed_position_tuple, strict=True)
            )
            close_dwell = close_dwell + 1 if close_error <= 0.008 else 0
            if close_dwell >= 3:
                break
        else:
            raise RuntimeError("Legacy gripper did not close within the measured dwell gate")
        report["close_gripper"] = {
            "steps": close_steps,
            "position": list(gripper.state()[0]),
            "position_error": close_error,
        }
        report["target_pose_w_at_pick"] = [
            *_array(target.data.root_pos_w[0]),
            *_array(target.data.root_quat_w[0]),
        ]
        report["stage"] = "legacy_lift"
        if "pick_low_lift" in contact_points:
            report["contact_waypoint"] = "pick_low_lift"
            report["state4_low_lift"] = approach(
                "legacy_lift",
                np.asarray(contact_points["pick_low_lift"]),
                -1.0,
                tolerance_m=0.01,
            )
            report["contact_waypoint"] = "pick_exit_xy"
        report["lift"] = approach("legacy_lift", lift, -1.0)
        report["contact_waypoint"] = None
        report["lift_q"] = _array(robot.data.joint_pos[0, arm_ids]).tolist()
        lift_tcp_position, lift_tcp_quaternion, _ = ik.current_tcp_pose_b()
        report["lift_tcp_pose_b"] = [
            *_array(lift_tcp_position[0]),
            *_array(lift_tcp_quaternion[0]),
        ]
        report["target_pose_w_after_lift"] = [
            *_array(target.data.root_pos_w[0]),
            *_array(target.data.root_quat_w[0]),
        ]
        report["target_lift_distance_m"] = float(
            np.linalg.norm(
                np.asarray(report["target_pose_w_after_lift"][:3]) - np.asarray(report["target_pose_w_at_reset"][:3])
            )
        )
        report["stage"] = "tray_contact_gate"
        target_mesh = next(mesh for mesh in actual_world.config.meshes if mesh.name == actual_world.config.target_name)
        trays = tuple(mesh for mesh in actual_world.config.meshes if mesh.name.startswith("support_tray_"))
        if not trays:
            raise ValueError("production world lacks support tray meshes")
        tray_world = WorldSnapshot(
            WorldConfig((), actual_world.config.target_name, (target_mesh, *trays))
        ).permit_target_contact()
        checker = planner.collision if args.integrated else CuroboCollision(frozen_world["collision_model"], tray_world)
        restore_world = planner.world if args.integrated else None
        tray_diagnostic = {}
        try:
            if args.integrated:
                checker.update_world(tray_world)
            for stage in ("legacy_approach_pick", "close_gripper", "legacy_lift"):
                joints = np.asarray(
                    [item["q"] for item in records if item["stage"] == stage],
                    dtype=np.float32,
                )
                if len(joints) == 0:
                    raise RuntimeError(f"{stage} has no measured frames for tray gate")
                scores = []
                for offset in range(0, len(joints), 64):
                    collision = checker.query(
                        joints[offset : offset + 64],
                        world_ignore_links=("Base_Link",),
                    )
                    scores.extend(np.asarray(collision["world"]).reshape(-1).tolist())
                tray_diagnostic[stage] = {
                    "frames": len(joints),
                    "max_robot_tray_score": float(max(scores)),
                    "positive_frames": int(sum(value > 0 for value in scores)),
                    "colliding_trays": {},
                }
                if tray_diagnostic[stage]["positive_frames"]:
                    for tray in trays:
                        one_world = WorldSnapshot(
                            WorldConfig((), actual_world.config.target_name, (target_mesh, tray))
                        ).permit_target_contact()
                        checker.update_world(one_world)
                        peak = 0.0
                        for offset in range(0, len(joints), 64):
                            collision = checker.query(
                                joints[offset : offset + 64],
                                world_ignore_links=("Base_Link",),
                            )
                            peak = max(peak, float(np.max(collision["world"])))
                        if peak > 0:
                            tray_diagnostic[stage]["colliding_trays"][tray.name] = peak
                    checker.update_world(tray_world)
        finally:
            if restore_world is not None:
                checker.update_world(restore_world)
        report["tray_collision_diagnostic"] = {
            "hard_gate": args.contact_policy == "strict",
            "contact_policy": args.contact_policy,
            "allowed_world_contact": [actual_world.config.target_name],
            "protected_trays": [mesh.name for mesh in trays],
            "segments": tray_diagnostic,
            "passed": all(item["max_robot_tray_score"] <= 0 for item in tray_diagnostic.values()),
        }
        if args.contact_policy == "strict" and not report["tray_collision_diagnostic"]["passed"]:
            names = sorted({name for item in tray_diagnostic.values() for name in item["colliding_trays"]})
            raise RuntimeError(f"measured pick/lift collided with support trays: {names}")
        if args.stop_after_lift:
            if report["target_lift_distance_m"] < 0.1:
                raise RuntimeError("target was not observed moving with the gripper")
            report["record_count"] = len(records)
            report["scope"] = "measured Isaac pick/lift for attached-object CuRobo transfer-goal search"
            report["stage"] = "complete"
            report["passed"] = True
            return 0
        if args.place_high_q is not None:
            high_q = np.asarray(args.place_high_q, dtype=np.float64)
            if args.integrated:
                report["stage"] = "curobo_transfer"
                target_mesh = next(mesh for mesh in actual_world.config.meshes if mesh.name == "target")
                vertices_b = np.asarray(trimesh.load_mesh(target_mesh.file_path, process=False).vertices)
                sphere = attached_sphere_from_measured_target(
                    vertices_b,
                    base_pose_w=report["base_pose_w"],
                    object_pose_w_at_reset=report["target_pose_w_at_reset"],
                    object_pose_w_current=report["target_pose_w_after_lift"],
                    tcp_pose_b_current=report["lift_tcp_pose_b"],
                )
                planner.attach(actual_world.attach((sphere,)))
                report["attachment"] = {"center_tcp": sphere.center_tcp, "radius_m": sphere.radius_m}
                lift_q = np.asarray(report["lift_q"], dtype=np.float64)
                transfer_trajectory, transfer_backend = planner.plan(lift_q, high_q)
                transfer_path = args.output / "transfer_trajectory.npz"
                save_trajectory(transfer_path, transfer_trajectory)
                transfer_acceptance = verify_segment(planner, load_trajectory(transfer_path), lift_q, high_q)
                report["curobo_transfer"] = {"motion_gen": transfer_backend, "acceptance": transfer_acceptance}
                if not transfer_acceptance["passed"]:
                    raise RuntimeError("transfer CuRobo trajectory failed independent collision/kinematic gate")

                def transform(pose):
                    values = np.asarray(pose, dtype=np.float64)
                    return pose_matrix(values[:3], values[3:] / np.linalg.norm(values[3:]))

                world_from_base = transform(report["base_pose_w"])
                world_from_tcp_lift = world_from_base @ transform(report["lift_tcp_pose_b"])
                tcp_from_object = np.linalg.inv(world_from_tcp_lift) @ transform(report["target_pose_w_after_lift"])
                for q, dq in zip(
                    transfer_trajectory.joint_position[1:], transfer_trajectory.joint_velocity[1:], strict=True
                ):
                    write_arm(q, dq)
                    tcp_position, tcp_quaternion, _ = ik.current_tcp_pose_b()
                    tcp_pose = [*_array(tcp_position[0]), *_array(tcp_quaternion[0])]
                    object_pose = matrix_pose_wxyz(world_from_base @ transform(tcp_pose) @ tcp_from_object)
                    target.write_root_pose_to_sim(
                        torch.as_tensor(
                            object_pose, dtype=target.data.root_pos_w.dtype, device=target.data.root_pos_w.device
                        ).view(1, 7)
                    )
                    target.write_root_velocity_to_sim(torch.zeros((1, 6), device=target.data.root_pos_w.device))
                    env.scene.write_data_to_sim()
                    env.sim.forward()
                    env.scene.update(env.sim.get_physics_dt())
                    env.sim.render()
                    records.append(
                        {
                            "stage": "free_space_transfer",
                            "q": _array(robot.data.joint_pos[0, arm_ids]).tolist(),
                            "dq": _array(robot.data.joint_vel[0, arm_ids]).tolist(),
                            "tcp_position_b": _array(tcp_position[0]).tolist(),
                            "gripper_position": list(gripper.state()[0]),
                            "target_position_w": _array(target.data.root_pos_w[0]).tolist(),
                        }
                    )
                    capture_sample()
                report["target_pose_w_after_transfer"] = [
                    *_array(target.data.root_pos_w[0]),
                    *_array(target.data.root_quat_w[0]),
                ]
            else:
                report["scope"] = (
                    "Isaac Legacy pick/lift and place/open probes; teleports at pre-pick and place-high; "
                    "not a complete V2 run or physical transfer"
                )
                write_arm(high_q)
            place_high = np.asarray(poses["place_high"]["tcp_pose_b"][:3], dtype=np.float64)
            actual_high = _array(ik.current_tcp_pose_b()[0][0])
            report["place_high_tcp_error_m"] = float(np.linalg.norm(actual_high - place_high))
            if report["place_high_tcp_error_m"] > 1e-3:
                raise RuntimeError("teleported place-high state does not match Legacy TCP target")
            heading = planar_heading_from_quaternion(poses["place"]["tcp_pose_b"][3:], TCP_DIRECTION_AXIS_T)
            place = np.asarray(poses["place"]["tcp_pose_b"][:3], dtype=np.float64)
            place_quaternion = None
            if task.get("state4_routing", {}).get("place_measured_grasp") and task["destination_index"] == 4:
                report["placement_calibration"] = measured_state4_placement(
                    task,
                    lift_tcp_pose_b=report["lift_tcp_pose_b"],
                    object_pose_w_after_lift=report["target_pose_w_after_lift"],
                )
                calibrated = report["placement_calibration"]["tcp_waypoint_pose_b"]
                place = np.asarray(calibrated["place"][:3])
                place_quaternion = calibrated["place"][3:]
                contact_points = {name: pose[:3] for name, pose in calibrated.items() if name != "place"}
            report["stage"] = "legacy_place_approach"
            if "place_descent" in contact_points:
                report["contact_waypoint"] = "place_descent_outside_roof"
                report["state4_place_descent"] = approach(
                    "legacy_place_approach",
                    np.asarray(contact_points["place_descent"]),
                    -1.0,
                    target_quaternion_b=place_quaternion,
                )
                report["contact_waypoint"] = "place_insert_xy"
            if "place_entry" in contact_points:
                report["state4_place_entry"] = approach(
                    "legacy_place_approach",
                    np.asarray(contact_points["place_entry"]),
                    -1.0,
                    target_quaternion_b=place_quaternion,
                )
                report["contact_waypoint"] = "place_settle_z"
            report["place_approach"] = approach(
                "legacy_place_approach", place, -1.0, target_quaternion_b=place_quaternion
            )
            report["contact_waypoint"] = None
            report["stage"] = "open_gripper"
            open_dwell = 0
            for step in range(args.max_steps_per_stage):
                step_ik(place, "open_gripper", 1.0, place_quaternion)
                open_error = max(
                    abs(actual - desired)
                    for actual, desired in zip(gripper.state()[0], gripper.open_position_tuple, strict=True)
                )
                open_dwell = open_dwell + 1 if open_error <= 0.008 else 0
                if open_dwell >= 3:
                    break
            else:
                raise RuntimeError("Legacy gripper did not open within the measured dwell gate")
            report["open_gripper"] = {
                "steps": step + 1,
                "position": list(gripper.state()[0]),
                "position_error": open_error,
            }
        report["record_count"] = len(records)
        report["passed"] = True
        report["stage"] = "complete"
    except Exception as error:
        report["failure_reason"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    finally:
        report["record_count"] = len(records)
        if contact_reports is not None:
            report["physics_contact_reporting"] = contact_reports.finish()
        trace_path = args.output / "trace.json"
        trace_path.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if recorder is not None:
            report["visualization"] = recorder.finish(source_frames=len(records), render_calls=len(records))
            manifest_path = recorder.output_dir / "frames.json"
            camera_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            camera_manifest.update(
                scope="live Isaac execution capture; a failed execution is not an accepted trajectory",
                source_trace=str(trace_path),
                source_trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                execution_contract=report["execution_contract"],
                execution_passed=report["passed"],
            )
            manifest_path.write_text(json.dumps(camera_manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if env is not None:
            env.close()
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


try:
    raise SystemExit(main())
finally:
    launcher.app.close()
