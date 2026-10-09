"""Replay every V2 hybrid frame in the Legacy production Isaac scene."""

import argparse
import json
import math
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from DataPipe.pipelines.curobo_bootstrap.measured_placement import executed_endpoint_poses
from DataPipe.pipelines.curobo_bootstrap.pick_place_artifact import (
    save_json,
    sha256_file,
    verify_hybrid_artifact,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--no-realtime", action="store_true")
parser.add_argument("--report-name", default="replay_report.json")
parser.add_argument("--visualization-dir", type=Path, help="New directory for three-view sparse RGB frames")
parser.add_argument("--camera-width", type=int, default=128)
parser.add_argument("--camera-height", type=int, default=128)
args_pre, _ = parser.parse_known_args()
artifact = verify_hybrid_artifact(args_pre.run)
from DataPipe.sim_backend import add_launcher_args, configure_cameras, ik_class, launch, trajectory_classes

add_launcher_args(parser)
args = parser.parse_args()
if args.camera_width <= 0 or args.camera_height <= 0:
    parser.error("camera dimensions must be positive")
args.enable_cameras = True
launcher = launch(args)

import torch

from configs.assets import SCENE_ENTITY_AUBO, SCENE_ENTITY_TARGET

AuboIKAction = ik_class()
TcpDockingTrajectoryEnv, TcpDockingTrajectoryEnvCfg = trajectory_classes()

from DataPipe.isaac_gripper import DatasetGripperController
from DataPipe.pipelines.curobo_bootstrap.production_world import extract_production_world
from DataPipe.pipelines.curobo_bootstrap.replay import matrix_pose_wxyz
from DataPipe.pipelines.curobo_bootstrap.replay_visualization import ReplayFrameRecorder
from DataPipe.pipelines.curobo_bootstrap.robot_adapter import pose_matrix
from DataPipe.pipelines.curobo_bootstrap.route_experiment import require_height
from DataPipe.pipelines.curobo_bootstrap.source_target import set_source_target_pose


def _array(tensor):
    return tensor.detach().cpu().numpy().astype(np.float64)


def _pose_matrix_measured(values):
    values = np.asarray(values, dtype=np.float64)
    return pose_matrix(values[:3], values[3:] / np.linalg.norm(values[3:]))


def _quat_error(a, b):
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    return float(2 * math.acos(min(1.0, abs(np.dot(left, right)))))


def main():
    data = artifact["trajectory"]
    plan = artifact["report"]
    snapshot = artifact["snapshot"]
    task = snapshot["task"]
    endpoint_poses = executed_endpoint_poses(task, plan)
    report = {
        "schema": "curobo_pick_place_v2_replay_report/1",
        "passed": False,
        "stage": "startup",
        "source_frames": len(data["time_s"]),
        "source_index": task["source_index"],
        "destination_index": task["destination_index"],
        "source_name": task["source"]["name"],
        "destination_name": task["destination"]["name"],
        "task_hash": task["task_hash"],
        "lift_clearance_m": task["lift_clearance_m"],
        "mode": "headless" if args.headless else "gui",
        "state_replay_only": True,
        "endpoint_tcp_reference_b": endpoint_poses,
        "source_sha256": {
            "trajectory.npz": sha256_file(args.run / "trajectory.npz"),
            "scene_snapshot.json": sha256_file(args.run / "scene_snapshot.json"),
            "plan_report.json": sha256_file(args.run / "plan_report.json"),
        },
    }
    env = None
    recorder = None
    try:
        require_height(task, task["lift_clearance_m"])
        cfg = TcpDockingTrajectoryEnvCfg()
        cfg.scene.num_envs = 1
        cfg.sim.device = args.device
        cfg.seed = task["seed"]
        cfg.actions.joint_trajectory.open_position = (0.0115, -0.0115)
        cfg.actions.joint_trajectory.closed_position = (0.0, 0.0)
        cfg, camera_specs = configure_cameras(cfg, args.camera_width, args.camera_height)
        env = TcpDockingTrajectoryEnv(cfg)
        env.reset()
        if args.backend == "isaacsim" and env.camera_specs:
            camera_specs = env.camera_specs
        report["source_target_pose_w"] = set_source_target_pose(env, task)
        report["stage"] = "scene_identity"
        with tempfile.TemporaryDirectory(prefix="curobo_v2_replay_world_") as directory:
            world, _ = extract_production_world(env, directory)
            if args.backend == "isaacsim":
                from DataPipe.native.world_identity import reconcile_world

                path_map = args.run / "asset_path_map.json"
                mapped = (
                    {k: args.run / v for k, v in json.loads(path_map.read_text())["paths"].items()}
                    if path_map.exists()
                    else {}
                )
                world, report["native_world_geometry"] = reconcile_world(world, snapshot["world"], asset_paths=mapped)
            if world.world_hash != snapshot["world_hash"]:
                raise ValueError("replay production world differs from plan snapshot")
            report["world_hash"] = world.world_hash
        robot = env.scene[SCENE_ENTITY_AUBO]
        target = env.scene[SCENE_ENTITY_TARGET]
        cameras = {spec.name: env.scene[spec.name] for spec in camera_specs}
        arm_ids, arm_names = robot.find_joints(
            ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Flange"], preserve_order=True
        )
        if tuple(arm_names) != tuple(data["joint_names"]):
            raise ValueError("replay Isaac six-joint order differs from NPZ")
        gripper = DatasetGripperController(robot, open_position=(0.0115, -0.0115), closed_position=(0.0, 0.0))
        ik = AuboIKAction(
            robot, env.sim.get_physics_dt(), cfg.decimation, diagnostics_enabled=True, flange_downward_enabled=True
        )
        base_pose = [*_array(robot.data.root_pos_w[0]), *_array(robot.data.root_quat_w[0])]
        report["base_position_error_m"] = float(
            np.linalg.norm(np.asarray(base_pose[:3]) - np.asarray(snapshot["base_pose_w"][:3]))
        )
        report["base_rotation_error_rad"] = _quat_error(base_pose[3:], snapshot["base_pose_w"][3:])
        report["reset_q_error_rad"] = float(
            np.max(np.abs(_array(robot.data.joint_pos[0, arm_ids]) - np.asarray(plan["measured"]["reset_q"])))
        )
        source_pose = snapshot["target_pose_w_at_reset"]
        report["source_target_position_error_m"] = float(
            np.linalg.norm(_array(target.data.root_pos_w[0]) - np.asarray(source_pose[:3]))
        )
        if (
            report["base_position_error_m"] > 1e-5
            or report["base_rotation_error_rad"] > 1e-4
            or report["reset_q_error_rad"] > 1e-5
            or report["source_target_position_error_m"] > 1e-4
        ):
            raise ValueError("replay reset Base, robot or target differs from frozen plan")
        world_from_base = _pose_matrix_measured(base_pose)
        world_from_tcp_lift = world_from_base @ _pose_matrix_measured(plan["measured"]["lift_tcp_pose_b"])
        tcp_from_object = np.linalg.inv(world_from_tcp_lift) @ _pose_matrix_measured(
            plan["measured"]["target_pose_w_after_lift"]
        )
        destination = task["destination"]
        world_from_environment = np.asarray(task["frames"]["T_WE"], dtype=np.float64)
        destination_pose = matrix_pose_wxyz(
            world_from_environment @ pose_matrix(destination["position_e"], destination["rotation_wxyz"])
        )
        waypoint_frames = {
            "pre_pick": plan["trajectory_acceptance"]["segment_ranges"]["free_space_to_pre_pick"][1] - 1,
            "place_high": plan["trajectory_acceptance"]["segment_ranges"]["free_space_transfer"][1] - 1,
            "place": plan["trajectory_acceptance"]["segment_ranges"]["legacy_place_approach"][1] - 1,
        }
        endpoint_errors = {}
        max_q_error = max_dq_error = max_target_error = max_gripper_error = 0.0
        max_base_position_drift = max_base_rotation_drift = 0.0
        written_frames = render_calls = 0
        observed_events = []
        attachment_transitions = []
        last_attached = False
        report["stage"] = "replay"
        visualization_dir = args.visualization_dir or args.run / "replay_visualization"
        recorder = ReplayFrameRecorder(visualization_dir, camera_specs)
        report["visualization_dir"] = str(recorder.output_dir)
        start_time = time.monotonic()
        for index in range(len(data["time_s"])):
            if not launcher.app.is_running():
                raise RuntimeError(f"Isaac app stopped before frame {index}")
            if not args.no_realtime:
                deadline = start_time + float(data["time_s"][index])
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
            positions = robot.data.joint_pos.clone()
            velocities = robot.data.joint_vel.clone()
            positions[:, arm_ids] = torch.as_tensor(
                data["joint_position"][index], dtype=positions.dtype, device=positions.device
            )
            velocities[:, arm_ids] = torch.as_tensor(
                data["joint_velocity"][index], dtype=velocities.dtype, device=velocities.device
            )
            positions[:, gripper.joint_ids] = torch.as_tensor(
                data["gripper_target"][index], dtype=positions.dtype, device=positions.device
            )
            velocities[:, gripper.joint_ids] = 0
            robot.write_joint_state_to_sim(positions, velocities)
            robot.set_joint_position_target(positions[:, arm_ids], joint_ids=arm_ids)
            robot.set_joint_position_target(positions[:, gripper.joint_ids], joint_ids=gripper.joint_ids)
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(env.sim.get_physics_dt())
            attached = bool(data["object_attached"][index])
            if attached != last_attached:
                attachment_transitions.append({"frame": index, "attached": attached})
                last_attached = attached
            if attached:
                tcp_position, tcp_quaternion, _ = ik.current_tcp_pose_b()
                tcp_pose = [*_array(tcp_position[0]), *_array(tcp_quaternion[0])]
                object_pose = matrix_pose_wxyz(world_from_base @ _pose_matrix_measured(tcp_pose) @ tcp_from_object)
            elif data["event_code"][index] == "PLACE" or index > int(plan["trajectory_acceptance"]["place_frame"]):
                object_pose = destination_pose
            else:
                object_pose = source_pose
            target.write_root_pose_to_sim(
                torch.as_tensor(
                    object_pose, dtype=target.data.root_pos_w.dtype, device=target.data.root_pos_w.device
                ).view(1, 7)
            )
            target.write_root_velocity_to_sim(torch.zeros((1, 6), device=target.data.root_pos_w.device))
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(env.sim.get_physics_dt())
            written_frames += 1
            max_q_error = max(
                max_q_error,
                float(np.max(np.abs(_array(robot.data.joint_pos[0, arm_ids]) - data["joint_position"][index]))),
            )
            max_dq_error = max(
                max_dq_error,
                float(np.max(np.abs(_array(robot.data.joint_vel[0, arm_ids]) - data["joint_velocity"][index]))),
            )
            max_target_error = max(
                max_target_error, float(np.linalg.norm(_array(target.data.root_pos_w[0]) - np.asarray(object_pose[:3])))
            )
            max_gripper_error = max(
                max_gripper_error,
                float(
                    np.max(np.abs(_array(robot.data.joint_pos[0, gripper.joint_ids]) - data["gripper_target"][index]))
                ),
            )
            max_base_position_drift = max(
                max_base_position_drift,
                float(np.linalg.norm(_array(robot.data.root_pos_w[0]) - np.asarray(base_pose[:3]))),
            )
            max_base_rotation_drift = max(
                max_base_rotation_drift, _quat_error(_array(robot.data.root_quat_w[0]), base_pose[3:])
            )
            event = str(data["event_code"][index])
            if event != "NONE":
                observed_events.append({"frame": index, "event": event})
            if index in waypoint_frames.values():
                name = next(key for key, value in waypoint_frames.items() if value == index)
                tcp_position, tcp_quaternion, _ = ik.current_tcp_pose_b()
                expected = endpoint_poses[name]
                endpoint_errors[name] = {
                    "position_m": float(np.linalg.norm(_array(tcp_position[0]) - np.asarray(expected[:3]))),
                    "rotation_rad": _quat_error(_array(tcp_quaternion[0]), expected[3:]),
                }
            env.sim.render()
            render_calls += 1
            recorder.capture(
                cameras,
                render_frame=render_calls,
                trajectory_index=index,
                time_s=float(data["time_s"][index]),
                event=event,
            )
        report["visualization"] = recorder.finish(source_frames=len(data["time_s"]), render_calls=render_calls)
        report.update(
            stage="complete",
            written_frames=written_frames,
            render_calls=render_calls,
            max_joint_position_write_error_rad=max_q_error,
            max_joint_velocity_write_error_rad_s=max_dq_error,
            max_target_position_write_error_m=max_target_error,
            max_gripper_position_write_error_m=max_gripper_error,
            max_base_position_drift_m=max_base_position_drift,
            max_base_rotation_drift_rad=max_base_rotation_drift,
            endpoint_errors=endpoint_errors,
            observed_events=observed_events,
            attachment_transitions=attachment_transitions,
            final_target_position_w=_array(target.data.root_pos_w[0]).tolist(),
            final_joint_position=_array(robot.data.joint_pos[0, arm_ids]).tolist(),
            elapsed_wall_s=time.monotonic() - start_time,
        )
        report["final_destination_position_error_m"] = float(
            np.linalg.norm(np.asarray(report["final_target_position_w"]) - np.asarray(destination_pose[:3]))
        )
        report["passed"] = bool(
            written_frames == render_calls == len(data["time_s"])
            and max_q_error <= 1e-5
            and max_dq_error <= 1e-5
            and max_target_error <= 1e-5
            and max_gripper_error <= 1e-5
            and max_base_position_drift <= 1e-6
            and max_base_rotation_drift <= 1e-6
            and report["final_destination_position_error_m"] <= 1e-5
            and endpoint_errors["pre_pick"]["position_m"] <= 1e-3
            and endpoint_errors["place_high"]["position_m"] <= 1e-3
            and endpoint_errors["place"]["position_m"] <= 0.01
            and endpoint_errors["pre_pick"]["rotation_rad"] <= 1e-3
            and endpoint_errors["place_high"]["rotation_rad"] <= 1e-3
            and endpoint_errors["place"]["rotation_rad"] <= 0.05
            and observed_events
            == [
                {"frame": int(plan["trajectory_acceptance"]["pick_frame"]), "event": "PICK"},
                {"frame": int(plan["trajectory_acceptance"]["place_frame"]), "event": "PLACE"},
            ]
            and attachment_transitions
            == [
                {"frame": int(plan["trajectory_acceptance"]["pick_frame"]), "attached": True},
                {"frame": int(plan["trajectory_acceptance"]["place_frame"]), "attached": False},
            ]
        )
        if not report["passed"]:
            report["failure_reason"] = "replay completion gates failed"
    except Exception as error:
        report["failure_reason"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    finally:
        report["backend"] = args.backend
        report["isaaclab_modules_loaded"] = sorted(
            n for n in sys.modules if n == "isaaclab" or n.startswith("isaaclab.")
        )
        if args.backend == "isaacsim" and report["isaaclab_modules_loaded"]:
            report["passed"] = False
            report["failure_reason"] = "native backend imported Isaac Lab"
        if env is not None:
            env.close()
        save_json(args.run / args.report_name, report)
    print(json.dumps(report, ensure_ascii=True, allow_nan=False))
    return 0 if report["passed"] else 1


try:
    raise SystemExit(main())
finally:
    launcher.app.close()
