"""Isaac Lab scene used only for scene-consistent CuRobo state replay.

Import this module only after ``AppLauncher`` has started Kit.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from CurriculumRL.configs.assets import ROBOT_PRIM_CONTRACT
from CurriculumRL.runtime.xform_spawner import XformSpawnerCfg
from CurriculumRL.tasks.tcp_docking.trajectory_env_cfg import TcpDockingTrajectoryEnvCfg

from pxr import Usd, UsdPhysics

from isaaclab import sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from configs.coordinate_frames import FLANGE_TO_TCP_ROTATION_F_WXYZ, FLANGE_TO_TCP_TRANSLATION_F
from DataPipe.config import GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POSITION
from DataPipe.pipelines.curobo_bootstrap.replay import matrix_pose_wxyz, pose_b_to_w, pose_w_to_b
from DataPipe.pipelines.curobo_bootstrap.robot_adapter import UrdfChain, pose_error, pose_matrix

BASE_POSE_W = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
POSITION_TOLERANCE_M = 1e-6
ROTATION_TOLERANCE_RAD = 1e-6
SIZE_TOLERANCE_M = 1e-6
LIMIT_TOLERANCE = 1e-6
FK_POSITION_TOLERANCE_M = 1e-5
FK_ROTATION_TOLERANCE_RAD = 1e-3


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _unit_pose(position, quaternion):
    position = _array(position)
    quaternion = _array(quaternion)
    if (
        position.shape != (3,)
        or quaternion.shape != (4,)
        or not np.isfinite(position).all()
        or not np.isfinite(quaternion).all()
    ):
        raise ValueError("Isaac returned an invalid pose")
    if not np.isclose(quaternion @ quaternion, 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("Isaac returned a non-unit quaternion")
    return tuple(float(value) for value in (*position, *quaternion))


def _scene_cfg(spec):
    source = TcpDockingTrajectoryEnvCfg().scene
    cfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=False)
    robot_usd = deepcopy(source.AUBObot_usd)
    robot = deepcopy(source.AUBObot)
    robot_usd.prim_path = "{ENV_REGEX_NS}/AUBObot"
    robot.prim_path = f"{{ENV_REGEX_NS}}/AUBObot/{ROBOT_PRIM_CONTRACT.articulation_prim}"
    robot_usd.init_state.pos = BASE_POSE_W[:3]
    robot_usd.init_state.rot = BASE_POSE_W[3:]
    robot.init_state.pos = BASE_POSE_W[:3]
    robot.init_state.rot = BASE_POSE_W[3:]
    initial = dict(robot.init_state.joint_pos)
    initial.update(
        dict(
            zip(
                spec.trajectory.joint_names,
                (float(value) for value in spec.trajectory.joint_position[0]),
                strict=True,
            )
        )
    )
    initial.update(dict(zip(GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POSITION, strict=True)))
    robot.init_state.joint_pos = initial
    setattr(cfg, "AUBObot_usd", robot_usd)
    setattr(cfg, "AUBObot", robot)
    setattr(
        cfg,
        "dome_light",
        AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
        ),
    )
    setattr(
        cfg,
        "planning_world",
        AssetBaseCfg(prim_path="{ENV_REGEX_NS}/PlanningWorld", spawn=XformSpawnerCfg()),
    )
    world_keys = {}
    for index, cuboid in enumerate(spec.world.config.cuboids):
        if "/" in cuboid.name:
            raise ValueError(f"cuboid name is not a single USD path component: {cuboid.name}")
        key = f"planning_cuboid_{index}"
        pose_w = matrix_pose_wxyz(pose_b_to_w(BASE_POSE_W, cuboid.pose_b))
        setattr(
            cfg,
            key,
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/PlanningWorld/{cuboid.name}",
                spawn=sim_utils.CuboidCfg(
                    size=cuboid.dimensions_m,
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.55, 0.9)),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=pose_w[:3], rot=pose_w[3:]),
            ),
        )
        world_keys[cuboid.name] = key
    return cfg, world_keys


class CuroboReplayScene:
    """Own one minimal Isaac scene and expose strict replay measurements."""

    def __init__(self, spec, *, device: str):
        self.spec = spec
        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=2, device=device)
        )
        cfg, self.world_keys = _scene_cfg(spec)
        self.scene = InteractiveScene(cfg)
        self.sim.reset()
        self.scene.reset()
        self.scene.update(self.sim.get_physics_dt())
        self.robot = self.scene.articulations["AUBObot"]
        self.arm_ids, arm_names = self.robot.find_joints(list(spec.trajectory.joint_names), preserve_order=True)
        self.gripper_ids, gripper_names = self.robot.find_joints(list(GRIPPER_JOINT_NAMES), preserve_order=True)
        self.flange_ids, flange_names = self.robot.find_bodies([ROBOT_PRIM_CONTRACT.flange_body], preserve_order=True)
        if tuple(arm_names) != spec.trajectory.joint_names:
            raise ValueError(f"Isaac arm joint order differs: {arm_names}")
        if tuple(gripper_names) != GRIPPER_JOINT_NAMES or flange_names != [ROBOT_PRIM_CONTRACT.flange_body]:
            raise ValueError("Isaac gripper/flange contract differs")
        self._flange_to_tcp = pose_matrix(FLANGE_TO_TCP_TRANSLATION_F, FLANGE_TO_TCP_ROTATION_F_WXYZ)
        self._baseline = None

    def set_camera_view(self, eye=(2.1, 1.8, 1.6), target=(0.0, 0.0, 0.75)):
        self.sim.set_camera_view(eye, target)

    def _base_pose(self):
        return _unit_pose(self.robot.data.root_pos_w[0], self.robot.data.root_quat_w[0])

    def _cuboid_poses(self):
        result = {}
        for name, key in self.world_keys.items():
            position, quaternion = self.scene.extras[key].get_world_poses()
            result[name] = _unit_pose(position[0], quaternion[0])
        return result

    def _cuboid_sizes(self):
        result = {}
        stage = self.scene.stage
        for name, key in self.world_keys.items():
            root_path = str(self.scene.extras[key].prim_paths[0])
            mesh = stage.GetPrimAtPath(root_path + "/geometry/mesh")
            size = float(mesh.GetAttribute("size").Get())
            scale = np.asarray(mesh.GetAttribute("xformOp:scale").Get(), dtype=np.float64)
            result[name] = tuple(float(value) for value in size * scale)
        return result

    def _collision_prims(self):
        enabled = []
        predicate = Usd.TraverseInstanceProxies()
        for prim in Usd.PrimRange.Stage(self.scene.stage, predicate):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                api = UsdPhysics.CollisionAPI(prim)
                if api.GetCollisionEnabledAttr().Get() is not False:
                    enabled.append(str(prim.GetPath()))
        return sorted(set(enabled))

    def _tcp_pose_b(self):
        base = pose_matrix(self._base_pose()[:3], self._base_pose()[3:])
        flange_pose = _unit_pose(
            self.robot.data.body_pos_w[0, self.flange_ids[0]],
            self.robot.data.body_quat_w[0, self.flange_ids[0]],
        )
        flange = pose_matrix(flange_pose[:3], flange_pose[3:])
        return np.linalg.inv(base) @ flange @ self._flange_to_tcp

    def write_frame(self, q, dq):
        q = torch.as_tensor(q, dtype=self.robot.data.joint_pos.dtype, device=self.robot.data.joint_pos.device).view(
            1, 6
        )
        dq = torch.as_tensor(dq, dtype=self.robot.data.joint_vel.dtype, device=self.robot.data.joint_vel.device).view(
            1, 6
        )
        positions = self.robot.data.joint_pos.clone()
        velocities = self.robot.data.joint_vel.clone()
        positions[:, self.arm_ids] = q
        velocities[:, self.arm_ids] = dq
        gripper = torch.as_tensor(GRIPPER_OPEN_POSITION, dtype=positions.dtype, device=positions.device).view(
            1, len(self.gripper_ids)
        )
        positions[:, self.gripper_ids] = gripper
        velocities[:, self.gripper_ids] = 0.0
        self.robot.write_joint_state_to_sim(positions, velocities)
        self.robot.set_joint_position_target(q, joint_ids=self.arm_ids)
        self.robot.set_joint_position_target(gripper, joint_ids=self.gripper_ids)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(self.sim.get_physics_dt())
        actual_q = _array(self.robot.data.joint_pos[0, self.arm_ids])
        actual_dq = _array(self.robot.data.joint_vel[0, self.arm_ids])
        return actual_q, actual_dq

    def initialize_physics_replay(self, q, dq):
        """Write the sole initial state before target-driven physics stepping."""
        return self.write_frame(q, dq)

    def step_position_target(self, q, *, physics_steps: int = 2):
        """Apply one arm target and advance an integral number of physics steps."""
        if physics_steps < 1:
            raise ValueError("physics_steps must be positive")
        target = torch.as_tensor(
            q, dtype=self.robot.data.joint_pos.dtype, device=self.robot.data.joint_pos.device
        ).view(1, 6)
        gripper = torch.as_tensor(
            GRIPPER_OPEN_POSITION,
            dtype=self.robot.data.joint_pos.dtype,
            device=self.robot.data.joint_pos.device,
        ).view(1, len(self.gripper_ids))
        self.robot.set_joint_position_target(target, joint_ids=self.arm_ids)
        self.robot.set_joint_position_target(gripper, joint_ids=self.gripper_ids)
        for _ in range(physics_steps):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.sim.get_physics_dt())
        return (
            _array(self.robot.data.joint_pos[0, self.arm_ids]),
            _array(self.robot.data.joint_vel[0, self.arm_ids]),
        )

    def validate_scene(self):
        report_limits = self.spec.plan_report["limits"]["motion_gen_loaded"]
        actual_limits = _array(self.robot.data.joint_pos_limits[0, self.arm_ids])
        actual_velocity = _array(self.robot.data.joint_vel_limits[0, self.arm_ids])
        expected_lower = np.asarray(report_limits["position_lower_rad"], dtype=np.float64)
        expected_upper = np.asarray(report_limits["position_upper_rad"], dtype=np.float64)
        expected_velocity = np.asarray(report_limits["velocity_rad_s"], dtype=np.float64)
        all_q = np.asarray(self.spec.trajectory.joint_position)
        base_pose = self._base_pose()
        base_position_error, base_rotation_error = pose_error(
            pose_matrix(BASE_POSE_W[:3], BASE_POSE_W[3:]), pose_matrix(base_pose[:3], base_pose[3:])
        )
        cuboid_poses = self._cuboid_poses()
        cuboid_sizes = self._cuboid_sizes()
        pose_errors = {}
        size_errors = {}
        for cuboid in self.spec.world.config.cuboids:
            measured_b = pose_w_to_b(base_pose, cuboid_poses[cuboid.name])
            position_error, rotation_error = pose_error(pose_matrix(cuboid.pose_b[:3], cuboid.pose_b[3:]), measured_b)
            pose_errors[cuboid.name] = {"position_m": position_error, "rotation_rad": rotation_error}
            size_errors[cuboid.name] = float(
                np.max(np.abs(np.asarray(cuboid_sizes[cuboid.name]) - np.asarray(cuboid.dimensions_m)))
            )
        collision_prims = self._collision_prims()
        robot_prefix = f"{self.scene.env_prim_paths[0]}/AUBObot"
        world_prefix = f"{self.scene.env_prim_paths[0]}/PlanningWorld/"
        unexpected = [
            path for path in collision_prims if not (path.startswith(robot_prefix) or path.startswith(world_prefix))
        ]
        chain = UrdfChain(self.spec.urdf_path, "AUBO_E5", self.spec.ee_link)
        fk = {}
        for label, q in (("start", all_q[0]), ("goal", all_q[-1])):
            self.write_frame(q, np.zeros(6))
            position_error, rotation_error = pose_error(chain.fk(q), self._tcp_pose_b())
            fk[label] = {"position_m": position_error, "rotation_rad": rotation_error}
        checks = {
            "single_articulation": len(self.scene.articulations) == 1,
            "joint_order": tuple(self.spec.trajectory.joint_names) == tuple(ROBOT_PRIM_CONTRACT.arm_joints),
            "position_limits": bool(
                np.allclose(actual_limits[:, 0], expected_lower, rtol=0.0, atol=LIMIT_TOLERANCE)
                and np.allclose(actual_limits[:, 1], expected_upper, rtol=0.0, atol=LIMIT_TOLERANCE)
            ),
            "velocity_limits": bool(np.allclose(actual_velocity, expected_velocity, rtol=0.0, atol=LIMIT_TOLERANCE)),
            "trajectory_within_limits": bool(
                np.all(all_q >= actual_limits[:, 0] - LIMIT_TOLERANCE)
                and np.all(all_q <= actual_limits[:, 1] + LIMIT_TOLERANCE)
            ),
            "base_pose": base_position_error <= POSITION_TOLERANCE_M and base_rotation_error <= ROTATION_TOLERANCE_RAD,
            "cuboid_names": set(cuboid_poses) == {item.name for item in self.spec.world.config.cuboids},
            "cuboid_poses": all(
                item["position_m"] <= POSITION_TOLERANCE_M and item["rotation_rad"] <= ROTATION_TOLERANCE_RAD
                for item in pose_errors.values()
            ),
            "cuboid_sizes": all(value <= SIZE_TOLERANCE_M for value in size_errors.values()),
            "no_unexpected_collision_prims": not unexpected,
            "fk_start": fk["start"]["position_m"] <= FK_POSITION_TOLERANCE_M
            and fk["start"]["rotation_rad"] <= FK_ROTATION_TOLERANCE_RAD,
            "fk_goal": fk["goal"]["position_m"] <= FK_POSITION_TOLERANCE_M
            and fk["goal"]["rotation_rad"] <= FK_ROTATION_TOLERANCE_RAD,
        }
        self._baseline = {"base": base_pose, "cuboids": cuboid_poses}
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "configured_base_pose_w": list(BASE_POSE_W),
            "measured_base_pose_w": list(base_pose),
            "base_pose_error": {"position_m": base_position_error, "rotation_rad": base_rotation_error},
            "actual_position_limits_rad": actual_limits.tolist(),
            "actual_velocity_limits_rad_s": actual_velocity.tolist(),
            "expected_cuboids": [item.name for item in self.spec.world.config.cuboids],
            "spawned_cuboids": sorted(cuboid_poses),
            "cuboid_pose_errors": pose_errors,
            "cuboid_size_errors_m": size_errors,
            "collision_prims": collision_prims,
            "unexpected_collision_prims": unexpected,
            "fk": fk,
        }

    def invariant_drift(self):
        if self._baseline is None:
            raise RuntimeError("validate_scene must run before replay")
        base = self._base_pose()
        root_position, root_rotation = pose_error(
            pose_matrix(self._baseline["base"][:3], self._baseline["base"][3:]),
            pose_matrix(base[:3], base[3:]),
        )
        world_position = 0.0
        world_rotation = 0.0
        for name, current in self._cuboid_poses().items():
            position, rotation = pose_error(
                pose_matrix(self._baseline["cuboids"][name][:3], self._baseline["cuboids"][name][3:]),
                pose_matrix(current[:3], current[3:]),
            )
            world_position = max(world_position, position)
            world_rotation = max(world_rotation, rotation)
        return {
            "root_position_m": root_position,
            "root_rotation_rad": root_rotation,
            "world_position_m": world_position,
            "world_rotation_rad": world_rotation,
        }

    def close(self):
        """Release scene views and the SimulationContext in Isaac's required order."""
        if getattr(self, "sim", None) is None:
            return
        if hasattr(self, "robot"):
            del self.robot
        if hasattr(self, "scene"):
            del self.scene
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()
        self.sim = None


__all__ = ["CuroboReplayScene"]
