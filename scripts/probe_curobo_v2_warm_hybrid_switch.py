"""Probe both V2 free-space segments on one warm MotionGen with target attachment."""

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
from DataPipe.pipelines.curobo_bootstrap.planner import (
    EXPERIMENTAL_ACCELERATION_LIMIT,
    CuroboMvpPlanner,
    load_trajectory,
    save_trajectory,
    validate_trajectory,
)
from DataPipe.pipelines.curobo_bootstrap.robot_adapter import pose_matrix
from DataPipe.pipelines.curobo_bootstrap.world_adapter import WorldSnapshot, attached_sphere_from_local_aabb


def _verify_segment(planner, trajectory, start_q, goal_q):
    return validate_trajectory(
        trajectory,
        start_q=start_q,
        goal_q=goal_q,
        position_lower=planner.position_lower,
        position_upper=planner.position_upper,
        velocity_limit=planner.velocity_limit,
        acceleration_limit=EXPERIMENTAL_ACCELERATION_LIMIT,
        control_hz=planner.config.control_hz,
        collision_query=lambda q: planner.collision.query(q, world_ignore_links=planner.world_collision_ignore_links),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-probe", type=Path, required=True)
    parser.add_argument("--transfer-probe", type=Path, required=True)
    parser.add_argument("--world-report", type=Path, required=True)
    parser.add_argument("--targets-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report = {
        "schema": "curobo_v2_warm_hybrid_switch_probe/1",
        "passed": False,
        "scope": "two CuRobo segments and warm attachment switch only; Legacy contact execution and replay absent",
    }
    try:
        first = json.loads(args.first_probe.read_text(encoding="utf-8"))
        transfer = json.loads(args.transfer_probe.read_text(encoding="utf-8"))
        source = json.loads(args.world_report.read_text(encoding="utf-8"))
        targets = json.loads(args.targets_report.read_text(encoding="utf-8"))
        if not first["passed"] or not transfer["passed"] or not targets["passed"]:
            raise ValueError("source probes and target report must have passed")
        if first["segment"] != "free_space_to_pre_pick" or transfer["segment"] != "free_space_transfer":
            raise ValueError("source probe segment order mismatch")
        if not (first["world_report"] == transfer["world_report"] == str(args.world_report.resolve())):
            raise ValueError("source probes use different production world reports")
        world_data = source["world"]
        meshes = tuple(MeshConfig(**mesh) for mesh in world_data["meshes"])
        for mesh in meshes:
            if hashlib.sha256(Path(mesh.file_path).read_bytes()).hexdigest() != mesh.sha256:
                raise ValueError(f"world mesh evidence mismatch: {mesh.name}")
        world = WorldSnapshot(WorldConfig((), world_data["target_name"], meshes))
        planner = CuroboMvpPlanner(
            source["collision_model"],
            world=world,
            config=PlannerConfig(
                version="0.7.8", cuda_graph_enabled=False, graph_first=True, max_attempts=4, timeout_s=30
            ),
        )
        planner.warmup()
        for name, source_probe in (("first", first), ("transfer", transfer)):
            if name == "transfer":
                pick_pose = targets["legacy_ik"]["targets"]["pick"]["tcp_pose_b"]
                target_mesh = next(mesh for mesh in meshes if mesh.name == world.config.target_name)
                vertices_b = np.asarray(
                    trimesh.load_mesh(target_mesh.file_path, process=False).vertices, dtype=np.float64
                )
                tcp_from_base = np.linalg.inv(pose_matrix(pick_pose[:3], pick_pose[3:]))
                vertices_tcp = (tcp_from_base @ np.column_stack((vertices_b, np.ones(len(vertices_b)))).T).T[:, :3]
                sphere = attached_sphere_from_local_aabb(vertices_tcp.min(axis=0), vertices_tcp.max(axis=0))
                planner.attach(world.attach((sphere,)))
                report["attachment"] = {"center_tcp": list(sphere.center_tcp), "radius_m": sphere.radius_m}
                if not np.allclose(
                    (*sphere.center_tcp, sphere.radius_m),
                    (*transfer["attachment"]["center_tcp"], transfer["attachment"]["radius_m"]),
                    rtol=0,
                    atol=1e-9,
                ):
                    raise ValueError("warm attachment differs from standalone transfer probe")
            start_q = np.asarray(source_probe["start_q"], dtype=np.float64)
            goal_q = np.asarray(source_probe["goal_q"], dtype=np.float64)
            trajectory, backend = planner.plan(start_q, goal_q)
            path = args.output / f"{name}_trajectory.npz"
            save_trajectory(path, trajectory)
            acceptance = _verify_segment(planner, load_trajectory(path), start_q, goal_q)
            report[name] = {"motion_gen": backend, "acceptance": acceptance, "trajectory": str(path.resolve())}
            if not backend["success"] or not acceptance["passed"]:
                raise RuntimeError(f"{name} segment failed after warm planning")
        report["passed"] = True
    except Exception as error:
        report["failure_reason"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
