"""Audit production Isaac collision-profile bounds against a CuRobo primitive world."""

import argparse
import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from DataPipe.runtime import ensure_curriculumrl_available

ensure_curriculumrl_available()
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(f"Refusing to overwrite audit: {args.output}")
launcher = AppLauncher(args)

import numpy as np
import torch
from CurriculumRL.configs.assets import SCENE_ENTITY_TARGET, TEST_SUPPORT_TRAY_INDICES
from CurriculumRL.tasks.tcp_docking.trajectory_env_cfg import TcpDockingTrajectoryEnv, TcpDockingTrajectoryEnvCfg

import omni.usd
from curobo.geom.sdf.world import CollisionQueryBuffer, WorldCollisionConfig, WorldPrimitiveCollision
from curobo.geom.types import WorldConfig as CuroboWorldConfig
from curobo.types.base import TensorDeviceType
from pxr import Usd, UsdGeom

from configs.coordinate_frames import FLANGE_TO_TCP_ROTATION_F_WXYZ, FLANGE_TO_TCP_TRANSLATION_F
from configs.curobo.world import WorldConfig
from DataPipe.pipelines.curobo_bootstrap.robot_adapter import pose_matrix
from DataPipe.pipelines.curobo_bootstrap.world_adapter import (
    WorldSnapshot,
    aabb_bounds_in_frame,
    attached_sphere_from_local_aabb,
    cuboid_from_world_aabb,
    curobo_world_dict,
)


def _world_aligned_range(cache, prim):
    bound = cache.ComputeWorldBound(prim)
    value = bound.ComputeAlignedRange()
    minimum = np.asarray(value.GetMin(), dtype=np.float64)
    maximum = np.asarray(value.GetMax(), dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,) or not np.isfinite((minimum, maximum)).all():
        raise ValueError(f"invalid bound for {prim.GetPath()}")
    if np.any(maximum <= minimum):
        raise ValueError(f"empty bound for {prim.GetPath()}: {minimum}, {maximum}")
    return minimum, maximum


def main():
    cfg = TcpDockingTrajectoryEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.seed = 7
    cfg.actions.joint_trajectory.open_position = (0.0115, -0.0115)
    cfg.actions.joint_trajectory.closed_position = (0.0, 0.0)
    env = TcpDockingTrajectoryEnv(cfg)
    try:
        env.reset()
        controller = env.action_manager.get_term("joint_trajectory").controller
        robot = controller.robot
        base_position = robot.data.root_pos_w[0].detach().cpu().numpy().astype(float)
        base_quaternion = robot.data.root_quat_w[0].detach().cpu().numpy().astype(float)
        base_quaternion_norm = float(np.linalg.norm(base_quaternion))
        base_quaternion /= base_quaternion_norm
        base_pose_w = pose_matrix(base_position, base_quaternion)
        flange_ids, flange_names = robot.find_bodies(["Flange"], preserve_order=True)
        if flange_names != ["Flange"]:
            raise ValueError("flange body resolution failed")
        flange_position_w = robot.data.body_pos_w[0, flange_ids[0]].detach().cpu().numpy().astype(float)
        flange_quaternion_wxyz = robot.data.body_quat_w[0, flange_ids[0]].detach().cpu().numpy().astype(float)
        flange_quaternion_raw_norm = float(np.linalg.norm(flange_quaternion_wxyz))
        flange_quaternion_wxyz /= flange_quaternion_raw_norm
        flange_pose_w = pose_matrix(flange_position_w, flange_quaternion_wxyz)
        tcp_pose_w = flange_pose_w @ pose_matrix(FLANGE_TO_TCP_TRANSLATION_F, FLANGE_TO_TCP_ROTATION_F_WXYZ)

        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        env_root = str(env.scene.env_prim_paths[0])
        roots = {
            "glass": f"{env_root}/station/static/workstation/WorkStation/Glass/M_Glass",
            "workstation": f"{env_root}/station/static/workstation/WorkStation/WorkStation/M_WorkStation",
            **{
                f"support_tray_{index:02d}": f"{env_root}/station/static/M_SupportTray_{index:02d}"
                for index in TEST_SUPPORT_TRAY_INDICES
            },
        }
        target_path = f"{env_root}/station/interactive/ws_interactive_reagent_01_sample_bottle"
        sources = {}
        cuboids = []
        for name, path in roots.items():
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise ValueError(f"production obstacle prim missing: {path}")
            minimum, maximum = _world_aligned_range(cache, prim)
            cuboid = cuboid_from_world_aabb(name, minimum, maximum, base_pose_w)
            cuboids.append(cuboid)
            sources[name] = {"prim_path": path, "minimum_w": minimum.tolist(), "maximum_w": maximum.tolist()}

        target_prim = stage.GetPrimAtPath(target_path)
        if not target_prim.IsValid():
            raise ValueError(f"production target prim missing: {target_path}")
        target_minimum_w, target_maximum_w = _world_aligned_range(cache, target_prim)
        target = cuboid_from_world_aabb("target", target_minimum_w, target_maximum_w, base_pose_w)
        cuboids.append(target)
        target_object = env.scene[SCENE_ENTITY_TARGET]
        target_position_w = target_object.data.root_pos_w[0].detach().cpu().numpy().astype(float)
        target_quaternion_wxyz = target_object.data.root_quat_w[0].detach().cpu().numpy().astype(float)
        target_quaternion_raw_norm = float(np.linalg.norm(target_quaternion_wxyz))
        target_quaternion_wxyz /= target_quaternion_raw_norm
        target_pose_w = pose_matrix(target_position_w, target_quaternion_wxyz)
        target_minimum_local, target_maximum_local = aabb_bounds_in_frame(
            target_minimum_w, target_maximum_w, target_pose_w
        )
        target_minimum_tcp, target_maximum_tcp = aabb_bounds_in_frame(target_minimum_w, target_maximum_w, tcp_pose_w)
        target_sphere = attached_sphere_from_local_aabb(target_minimum_tcp, target_maximum_tcp)

        snapshot = WorldSnapshot(WorldConfig(tuple(cuboids), "target"))
        attached = snapshot.attach((target_sphere,))
        tensor_args = TensorDeviceType(device=torch.device(args.device))
        model = CuroboWorldConfig.from_dict(curobo_world_dict(snapshot))
        checker = WorldPrimitiveCollision(WorldCollisionConfig(tensor_args=tensor_args, world_model=model))
        points = [[*cuboid.pose_b[:3], 0.001] for cuboid in snapshot.active_cuboids]
        spheres = tensor_args.to_device(points).view(1, 1, -1, 4)
        buffer = CollisionQueryBuffer.initialize_from_shape(spheres.shape, tensor_args, checker.collision_types)
        distances = (
            checker.get_sphere_distance(spheres, buffer, tensor_args.to_device([1]), tensor_args.to_device([0]))
            .view(-1)
            .cpu()
            .tolist()
        )
        passed = (
            len(distances) == len(cuboids)
            and all(value > 0 for value in distances)
            and len(attached.active_cuboids) == len(cuboids) - 1
            and all(cuboid.name != "target" for cuboid in attached.active_cuboids)
            and abs(base_quaternion_norm - 1) <= 1e-6
            and abs(flange_quaternion_raw_norm - 1) <= 1e-6
            and abs(target_quaternion_raw_norm - 1) <= 1e-6
        )
        report = {
            "passed": passed,
            "python": sys.executable,
            "cuda_device": torch.cuda.get_device_name(torch.device(args.device)),
            "env_root": env_root,
            "base_pose_w": {"position": base_position.tolist(), "quaternion_wxyz": base_quaternion.tolist()},
            "base_quaternion_raw_norm": base_quaternion_norm,
            "tcp_pose_w": tcp_pose_w.tolist(),
            "flange_quaternion_raw_norm": flange_quaternion_raw_norm,
            "world_hash": snapshot.world_hash,
            "world": asdict(snapshot.config),
            "source_bounds": sources,
            "target": {
                "prim_path": target_path,
                "minimum_w": target_minimum_w.tolist(),
                "maximum_w": target_maximum_w.tolist(),
                "pose_w": {
                    "position": target_position_w.tolist(),
                    "quaternion_wxyz": target_quaternion_wxyz.tolist(),
                    "quaternion_raw_norm": target_quaternion_raw_norm,
                },
                "minimum_object": list(target_minimum_local),
                "maximum_object": list(target_maximum_local),
                "minimum_tcp_at_audit": list(target_minimum_tcp),
                "maximum_tcp_at_audit": list(target_maximum_tcp),
                "conservative_attached_sphere_at_audit": asdict(target_sphere),
            },
            "curobo_center_distances_m": dict(zip((item.name for item in cuboids), distances, strict=True)),
            "attach_target_removed": len(attached.active_cuboids) == len(cuboids) - 1,
            "scope": (
                "production profile root bounds; conservative AABBs, not per-mesh exact collision geometry; "
                "TCP-frame attached sphere must be recomputed from measured poses at attach time"
            ),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items() if key not in {"world", "source_bounds"}}))
        if not passed:
            raise RuntimeError(report)
    finally:
        env.close()


try:
    main()
except Exception:
    traceback.print_exc()
    raise
finally:
    launcher.app.close()
