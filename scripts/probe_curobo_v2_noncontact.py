"""Probe one V2 non-contact MotionGen segment in the frozen production mesh world."""

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.curobo.planner import PlannerConfig
from configs.curobo.world import MeshConfig, WorldConfig
from DataPipe.pipelines.curobo_bootstrap.attachment_geometry import attached_sphere_from_measured_target
from DataPipe.pipelines.curobo_bootstrap.planner import (
    EXPERIMENTAL_ACCELERATION_LIMIT,
    CuroboMvpPlanner,
    TrajectoryData,
    load_trajectory,
    save_trajectory,
    validate_trajectory,
)
from DataPipe.pipelines.curobo_bootstrap.robot_adapter import pose_matrix
from DataPipe.pipelines.curobo_bootstrap.route_identity import require_report_identity, task_identity
from DataPipe.pipelines.curobo_bootstrap.world_adapter import WorldSnapshot, attached_sphere_from_local_aabb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-report", type=Path, required=True)
    parser.add_argument("--targets-report", type=Path)
    parser.add_argument("--legacy-contact-report", type=Path)
    parser.add_argument("--start-q", type=float, nargs=6, required=True)
    parser.add_argument("--goal-q", type=float, nargs=6, required=True)
    parser.add_argument("--segment", choices=("free_space_to_pre_pick", "free_space_transfer"), required=True)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--graph-first", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--graph-only-diagnostic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    report = {
        "schema": "curobo_v2_noncontact_probe/1",
        "passed": False,
        "segment": args.segment,
        "start_q": args.start_q,
        "goal_q": args.goal_q,
        "world_report": str(args.world_report.resolve()),
        "scope": "one non-contact MotionGen segment only; not a complete V2 run",
        "cuda_graph_enabled": not args.disable_cuda_graph,
        "graph_first": args.graph_first,
        "max_attempts": args.max_attempts,
        "timeout_s": args.timeout_s,
    }
    try:
        source = json.loads(args.world_report.read_text(encoding="utf-8"))
        if args.targets_report is not None:
            targets_report = json.loads(args.targets_report.read_text(encoding="utf-8"))
            task = targets_report["task_snapshot"]
            report.update(task_identity(task))
            require_report_identity(targets_report, task, label="targets")
            require_report_identity(source, task, label="world", require_passed=False)
            if not source.get("world_geometry_valid"):
                raise ValueError("selected world lacks validated geometry evidence")
        world_data = source["world"]
        if world_data["cuboids"]:
            raise ValueError("probe expects mesh-only production world")
        meshes = tuple(MeshConfig(**item) for item in world_data["meshes"])
        for mesh in meshes:
            if hashlib.sha256(Path(mesh.file_path).read_bytes()).hexdigest() != mesh.sha256:
                raise ValueError(f"world mesh evidence mismatch: {mesh.name}")
        world = WorldSnapshot(WorldConfig((), world_data["target_name"], meshes))
        if args.targets_report is not None and source["world_hash"] != world.world_hash:
            raise ValueError("selected world hash differs from world report")
        planning_world = (
            world.permit_target_contact() if args.segment == "free_space_to_pre_pick" else world
        )
        report["collision_policy"] = {
            "allowed_world_contact": (
                [world.config.target_name] if args.segment == "free_space_to_pre_pick" else []
            ),
            "protected_world_meshes": [mesh.name for mesh in planning_world.active_meshes],
            "self_collision_hard_gate": True,
        }
        planner = CuroboMvpPlanner(
            source["collision_model"],
            world=planning_world,
            config=PlannerConfig(
                version="0.7.8",
                cuda_graph_enabled=not args.disable_cuda_graph,
                graph_first=args.graph_first,
                max_attempts=args.max_attempts,
                timeout_s=args.timeout_s,
            ),
        )
        if args.segment == "free_space_transfer":
            target_mesh = next(mesh for mesh in meshes if mesh.name == world.config.target_name)
            if not np.allclose(target_mesh.pose_b, (0, 0, 0, 1, 0, 0, 0), atol=1e-8):
                raise ValueError("transfer probe expects Base-frame target mesh vertices")
            vertices_b = np.asarray(trimesh.load_mesh(target_mesh.file_path, process=False).vertices, dtype=np.float64)
            if args.legacy_contact_report is not None:
                legacy = json.loads(args.legacy_contact_report.read_text(encoding="utf-8"))
                if not legacy["passed"] or legacy["world_hash"] != world.world_hash:
                    raise ValueError("Legacy contact report must pass and match frozen world")
                if args.targets_report is not None:
                    require_report_identity(legacy, task, label="Legacy contact", world_hash=world.world_hash)
                if np.max(np.abs(np.asarray(args.start_q) - np.asarray(legacy["lift_q"]))) > 1e-5:
                    raise ValueError("transfer start must match measured Legacy lift joint state")
                if legacy["target_lift_distance_m"] < 0.1:
                    raise ValueError("target was not observed moving with the gripper")
                sphere = attached_sphere_from_measured_target(
                    vertices_b,
                    base_pose_w=legacy["base_pose_w"],
                    object_pose_w_at_reset=legacy["target_pose_w_at_reset"],
                    object_pose_w_current=legacy["target_pose_w_after_lift"],
                    tcp_pose_b_current=legacy["lift_tcp_pose_b"],
                )
                attachment_source = "frozen mesh transformed by measured Isaac target and TCP poses"
            else:
                if args.targets_report is None:
                    raise ValueError("transfer probe requires --legacy-contact-report or --targets-report")
                targets = json.loads(args.targets_report.read_text(encoding="utf-8"))
                if not targets["passed"]:
                    raise ValueError("targets report must have passed")
                pick_pose = targets["legacy_ik"]["targets"]["pick"]["tcp_pose_b"]
                tcp_from_base = np.linalg.inv(pose_matrix(pick_pose[:3], pick_pose[3:]))
                vertices_tcp = (tcp_from_base @ np.column_stack((vertices_b, np.ones(len(vertices_b)))).T).T[:, :3]
                sphere = attached_sphere_from_local_aabb(vertices_tcp.min(axis=0), vertices_tcp.max(axis=0))
                attachment_source = "frozen target mesh and nominal Legacy pick TCP pose; formal run must remeasure"
            planner.attach(world.attach((sphere,)))
            report["attachment"] = {
                "source": attachment_source,
                "legacy_contact_report": str(args.legacy_contact_report.resolve())
                if args.legacy_contact_report
                else None,
                "center_tcp": list(sphere.center_tcp),
                "radius_m": sphere.radius_m,
                "target_removed_from_static_world": world.config.target_name
                not in (mesh.name for mesh in planner.world.active_meshes),
            }
        report.update(world_hash=world.world_hash, curobo_version=planner.version)
        start_q = np.asarray(args.start_q, dtype=np.float64)
        goal_q = np.asarray(args.goal_q, dtype=np.float64)
        report["endpoint_checks"] = planner.endpoint_checks(start_q, goal_q)
        if not all(
            report["endpoint_checks"][key]
            for key in ("within_position_limits", "world_collision_free", "self_collision_free")
        ):
            raise ValueError("non-contact endpoint failed world/self collision gate")
        endpoints = planner.tensor_args.to_device(np.stack((start_q, goal_q)).astype(np.float32))
        report["graph_endpoint_mask_before_warmup"] = (
            planner.motion_gen.graph_planner.mask_samples(endpoints).detach().cpu().tolist()
        )
        planner.warmup()
        report["graph_endpoint_mask_after_warmup"] = (
            planner.motion_gen.graph_planner.mask_samples(endpoints).detach().cpu().tolist()
        )
        trajectory, backend = planner.plan(start_q, goal_q)
        report["motion_gen"] = backend
        trajectory_path = output / "trajectory.npz"
        save_trajectory(trajectory_path, trajectory)
        reloaded = load_trajectory(trajectory_path)
        acceptance = validate_trajectory(
            reloaded,
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
        report["acceptance"] = acceptance
        report["passed"] = bool(backend["success"] and acceptance["passed"])
    except Exception as error:
        report["failure_reason"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        if args.graph_only_diagnostic and "planner" in locals():
            try:
                from curobo.types.state import JointState
                from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

                start = JointState.from_position(
                    planner.tensor_args.to_device(np.asarray(args.start_q, dtype=np.float32)).unsqueeze(0),
                    joint_names=list(planner.collision.joint_names),
                )
                goal = JointState.from_position(
                    planner.tensor_args.to_device(np.asarray(args.goal_q, dtype=np.float32)).unsqueeze(0),
                    joint_names=list(planner.collision.joint_names),
                )
                graph = planner.motion_gen.plan_single_js(
                    start,
                    goal,
                    MotionGenPlanConfig(
                        max_attempts=1,
                        timeout=args.timeout_s,
                        enable_graph=True,
                        enable_opt=False,
                        check_start_validity=True,
                    ),
                )
                report["graph_only_diagnostic"] = {
                    "success": bool(graph.success is not None and graph.success.item()),
                    "status": str(graph.status),
                    "valid_query": bool(graph.valid_query),
                    "used_graph": bool(graph.used_graph),
                    "graph_time_s": float(graph.graph_time),
                    "scope": "diagnostic only; no trajectory acceptance or V2 pass",
                }
                if graph.success is not None and graph.success.item() and graph.interpolated_plan is not None:
                    path = graph.get_interpolated_plan()
                    fields = {name: getattr(path, name) for name in ("position", "velocity", "acceleration")}
                    report["graph_only_diagnostic"]["available_fields"] = [
                        name for name, value in fields.items() if value is not None
                    ]
                    if all(value is not None for value in fields.values()):
                        arrays = {
                            name: value.detach().cpu().numpy().astype(np.float64) for name, value in fields.items()
                        }
                        arrays = {name: value[0] if value.ndim == 3 else value for name, value in arrays.items()}
                        dt = float(graph.interpolation_dt)
                        diagnostic_path = TrajectoryData(
                            np.arange(len(arrays["position"])) * dt,
                            arrays["position"],
                            arrays["velocity"],
                            arrays["acceleration"],
                            tuple(planner.collision.joint_names),
                        )
                        report["graph_only_diagnostic"]["acceptance"] = validate_trajectory(
                            diagnostic_path,
                            start_q=np.asarray(args.start_q),
                            goal_q=np.asarray(args.goal_q),
                            position_lower=planner.position_lower,
                            position_upper=planner.position_upper,
                            velocity_limit=planner.velocity_limit,
                            acceleration_limit=EXPERIMENTAL_ACCELERATION_LIMIT,
                            control_hz=planner.config.control_hz,
                            collision_query=lambda q: planner.collision.query(
                                q, world_ignore_links=planner.world_collision_ignore_links
                            ),
                        )
            except Exception as graph_error:
                report["graph_only_diagnostic_error"] = f"{type(graph_error).__name__}: {graph_error}"
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
