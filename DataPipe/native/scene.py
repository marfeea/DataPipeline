"""Production USD scene driven by Isaac Sim Core, without Isaac Lab.

The small adapters retain the measured-state contract consumed by the existing
CuRobo/IK collectors. They expose live PhysX tensors, never synthetic readbacks.
"""

from dataclasses import replace
from math import sqrt
from time import monotonic
from types import SimpleNamespace as NS

import numpy as np
import torch

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, RigidPrim
from isaacsim.sensors.camera import Camera
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics

from configs.assets import ROBOT_PRIM_CONTRACT, SCENE_ENTITY_TARGET, TEST_SUPPORT_TRAY_INDICES, asset_by_key, asset_path
from configs.scene import (
    ARM_ACTUATOR,
    ARM_INITIAL_JOINT_POSITIONS,
    AUBO_2_LOCAL_POSITION_STATION,
    AUBO_LOCAL_POSITION_STATION,
    ENABLE_SELF_COLLISIONS,
    GRIPPER_ACTUATOR,
    GRIPPER_INITIAL_JOINT_POSITIONS,
    SAMPLE_BOTTLE_MASS_KG,
    SOLVER_POSITION_ITERATIONS,
    SOLVER_VELOCITY_ITERATIONS,
    WORKSTATION_POSE_E,
)
from configs.task import TARGET_STATES
from configs.vision_diagnostics import DIAGNOSTIC_CAMERA_MODEL
from DataPipe.camera_config import project_diagnostic_camera_sensor_specs
from DataPipe.config import CameraSensorSpec
from DataPipe.native.tcp_kinematics import quaternion_multiply
from DataPipe.pipelines.curobo_bootstrap.replay import matrix_pose_wxyz
from DataPipe.pipelines.curobo_bootstrap.robot_adapter import pose_matrix

ENV_ROOT = "/World/envs/env_0"


class NativeTrajectoryEnvCfg:
    def __init__(self):
        self.decimation = 2
        self.seed = 7
        self.camera_resolution = None
        self.sim = NS(
            dt=1 / 120,
            render_interval=2,
            device="cuda:0",
            physx=NS(
                solver_type=1,
                min_velocity_iteration_count=0,
                max_velocity_iteration_count=255,
                enable_external_forces_every_iteration=False,
            ),
        )
        self.scene = NS(num_envs=1)
        for name in ("AUBObot", "AUBObot_2"):
            limits = dict(zip(ROBOT_PRIM_CONTRACT.arm_joints, (2.617994,) * 3 + (3.14,) * 3))
            setattr(self.scene, name, NS(actuators={"arm": NS(velocity_limit_sim=limits)}))
            setattr(self.scene, name + "_usd", NS(spawn=NS(activate_contact_sensors=True)))
        setattr(self.scene, SCENE_ENTITY_TARGET, NS(spawn=NS(activate_contact_sensors=True)))
        self.actions = NS(joint_trajectory=NS(open_position=(0.0115, -0.0115), closed_position=(0.0, 0.0)))


class RobotAdapter:
    def __init__(self, view):
        self.view = view
        self.root_physx_view = view._physics_view
        self.data = NS()
        self.refresh()

    def refresh(self):
        v = self.root_physx_view
        d = self.data
        d.joint_pos = v.get_dof_positions().clone()
        d.joint_vel = v.get_dof_velocities().clone()
        d.joint_pos_limits = v.get_dof_limits().clone().to(d.joint_pos.device)
        d.joint_vel_limits = v.get_dof_max_velocities().clone().to(d.joint_pos.device)
        p, q = self.view.get_world_poses()
        d.root_pos_w, d.root_quat_w = p.clone(), q.clone()
        links = v.get_link_transforms().clone()
        d.body_pos_w = links[..., :3]
        d.body_quat_w = links[..., [6, 3, 4, 5]]

    def find_joints(self, names, preserve_order=True):
        return [self.view.get_dof_index(name) for name in names], list(names)

    def find_bodies(self, names, preserve_order=True):
        return [self.view.get_body_index(name) for name in names], list(names)

    def write_joint_state_to_sim(self, positions, velocities):
        self.view.set_joint_positions(positions)
        self.view.set_joint_velocities(velocities)

    def set_joint_position_target(self, target, joint_ids):
        self.view.set_joint_position_targets(target, joint_indices=joint_ids)


class TargetAdapter:
    def __init__(self, view):
        self.view = view
        self.root_physx_view = view._physics_view
        self.data = NS()
        self.refresh()

    def refresh(self):
        p, q = self.view.get_world_poses()
        self.data.root_pos_w, self.data.root_quat_w = p.clone(), q.clone()

    def write_root_pose_to_sim(self, pose):
        self.view.set_world_poses(pose[:, :3], pose[:, 3:])

    def write_root_velocity_to_sim(self, velocity):
        self.view.set_velocities(velocity)


class CameraAdapter:
    def __init__(self, camera, device, *, robot=None, mount_transform=None, mount_body_name=None):
        import omni.replicator.core as rep

        self.camera, self.device = camera, device
        self.parameters = rep.AnnotatorRegistry.get_annotator("camera_params")
        self.parameters.attach([camera.get_render_product_path()])
        self.robot, self.mount_transform = robot, mount_transform
        self.mount_body_name = mount_body_name
        self._data = NS()

    @property
    def data(self):
        self.refresh()
        return self._data

    def refresh(self):
        rgb = self.camera.get_rgb()
        if rgb is None or np.asarray(rgb).size == 0:
            raise RuntimeError(f"Camera has no rendered RGB: {self.camera.prim_path}")
        parameters = self.parameters.get_data()
        if not isinstance(parameters, dict) or "cameraViewTransform" not in parameters:
            raise RuntimeError("Camera has no rendered parameters")
        view = np.asarray(parameters["cameraViewTransform"]).reshape(4, 4)
        # Renderer view matrices use USD row vectors. Read the inverse view
        # from the completed RGB render rather than creating Fabric overrides
        # on a camera beneath a moving articulation.
        transform = np.linalg.inv(view).T
        pose = torch.as_tensor(matrix_pose_wxyz(transform), dtype=torch.float32, device=self.device)
        p, q_gl = pose[:3], pose[3:]
        if self.robot is not None:
            body_id = self.robot.find_bodies([self.mount_body_name])[0][0]
            body_p = self.robot.data.body_pos_w[0, body_id].detach().cpu().numpy()
            body_q = self.robot.data.body_quat_w[0, body_id].detach().cpu().numpy()
            expected = pose_matrix(body_p, body_q / np.linalg.norm(body_q)) @ self.mount_transform
            error = np.linalg.norm(p.detach().cpu().numpy() - expected[:3, 3])
            rotation_error = np.max(np.abs(transform[:3, :3] - expected[:3, :3]))
            if error > 1e-5 or rotation_error > 1e-5:
                raise RuntimeError(
                    f"Wrist camera rendered pose is stale: error={error}m rotation_error={rotation_error} "
                    f"body={self.mount_body_name} actual={p.tolist()} expected={expected[:3, 3].tolist()} "
                    f"view={view.tolist()}"
                )
        flip = torch.tensor((0.0, 1.0, 0.0, 0.0), dtype=q_gl.dtype, device=q_gl.device)
        q_ros = quaternion_multiply(q_gl, flip)

        def tensor(value):
            if isinstance(value, torch.Tensor):
                return value.detach().clone().to(self.device).unsqueeze(0)
            return torch.as_tensor(np.asarray(value).copy(), device=self.device).unsqueeze(0)

        self._data = NS(
            output={"rgb": tensor(rgb)},
            pos_w=tensor(p),
            quat_w_ros=tensor(q_ros),
            quat_w_opengl=tensor(q_gl),
            intrinsic_matrices=tensor(self.camera.get_intrinsics_matrix()),
        )

    def render_reference(self):
        frame = self.camera._fabric_time_annotator.get_data()
        if not isinstance(frame, dict) or "referenceTimeNumerator" not in frame:
            raise RuntimeError("Native camera has no completed render reference")
        return {key: int(frame[key]) for key in ("referenceTimeNumerator", "referenceTimeDenominator")}


class SceneAdapter(dict):
    def __init__(self, stage, robot, target, device):
        super().__init__(AUBObot=robot, **{SCENE_ENTITY_TARGET: target})
        self.stage = stage
        self.env_prim_paths = [ENV_ROOT]
        self.env_origins = torch.zeros((1, 3), device=device)

    def write_data_to_sim(self):
        # Core target setters write to PhysX immediately.
        pass

    def update(self, dt):
        for value in self.values():
            if isinstance(value, RobotAdapter | TargetAdapter):
                value.refresh()


class SimAdapter:
    def __init__(self, owner):
        self.owner = owner

    def get_physics_dt(self):
        return self.owner.world.get_physics_dt()

    def forward(self):
        self.owner.world.physics_sim_view.update_articulations_kinematic()
        from omni.physxfabric import get_physx_fabric_interface

        # Match the original forward() flush: zero dt/time disables pose
        # interpolation and publishes state writes without a physics step.
        get_physx_fabric_interface().force_update(0.0, 0.0)

    def step(self, render=False):
        self.owner.world.step(render=render)

    def render(self):
        started = monotonic()
        retries = 0
        while True:
            self.forward()
            self.owner.world.render()
            self.owner.scene.update(self.get_physics_dt())
            if not self.owner.cameras_ready:
                return
            # Material streaming can invalidate an earlier ready image. Keep
            # the physical state fixed while waiting for a usable render.
            missing = [
                name
                for name, camera in self.owner.cameras.items()
                if camera.camera.get_rgb() is None or np.std(camera.camera.get_rgb()) <= 1
            ]
            if not missing:
                return
            if monotonic() - started > 180:
                raise RuntimeError(f"Native render did not become usable: {missing}")
            retries += 1
            if retries % 100 == 0:
                print(f"NATIVE_RENDER_WARMUP retries={retries} cameras={missing}", flush=True)


def _pose(stage, path, position, quaternion):
    prim = stage.GetPrimAtPath(path)
    xf = UsdGeom.Xformable(prim)
    # Match Isaac Lab's XFormPrim authoring, replacing the referenced root ops.
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*position))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(quaternion[0], Gf.Vec3f(*quaternion[1:])))
    xf.AddScaleOp().Set(Gf.Vec3f(1, 1, 1))


class NativeTrajectoryEnv:
    def __init__(self, cfg):
        if cfg.scene.num_envs != 1:
            raise ValueError("Native collection currently uses one production environment")
        self.cfg = cfg
        self.cameras = {}
        self.cameras_ready = False
        self.world = World(
            stage_units_in_meters=1,
            physics_dt=cfg.sim.dt,
            rendering_dt=cfg.sim.dt * cfg.decimation,
            backend="torch",
            device=cfg.sim.device,
        )
        stage = self.world.stage
        import carb

        settings = carb.settings.get_settings()
        # Preserve the original GPU-to-Fabric path. With suppressed PhysX
        # readback, the CPU transform cache does not reflect tensor writes.
        settings.set_bool("/physics/fabricUseGPUInterop", True)
        settings.set_bool("/physics/fabricEnabled", True)
        settings.set_bool("/physics/fabricUpdateTransformations", True)
        for key in (
            "updateToUsd",
            "fabricUpdateVelocities",
            "fabricUpdateForceSensors",
            "fabricUpdateJointStates",
        ):
            settings.set_bool("/physics/" + key, False)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        for path in (ENV_ROOT, ENV_ROOT + "/station/static", ENV_ROOT + "/station/interactive"):
            UsdGeom.Xform.Define(stage, path)

        def asset(key, path, p, q):
            # Match Lab's raw USD reference composition. Core's convenience
            # add_reference_to_stage runs MetricsAssembler and rescales these
            # already-authored project joints by 0.01, changing the gripper.
            stage.DefinePrim(path, "Xform").GetReferences().AddReference(str(asset_path(asset_by_key(key))))
            _pose(stage, path, p, q)

        w = WORKSTATION_POSE_E
        asset("workstation", ENV_ROOT + "/station/static/workstation", w.position, w.rotation_wxyz)
        asset("laboratory", ENV_ROOT + "/Laboratory", (0, 0, 0), (1, 0, 0, 0))
        for i in TEST_SUPPORT_TRAY_INDICES:
            key = f"support_tray_{i:02d}"
            asset(key, ENV_ROOT + f"/station/static/M_SupportTray_{i:02d}", w.position, w.rotation_wxyz)
        rotation = (sqrt(0.5), 0, 0, -sqrt(0.5))
        # Preserve the legacy rounded workstation transform used for translation.
        qw, qx, qy, qz = w.rotation_wxyz

        def robot_position(local):
            v = np.asarray(local)
            u = np.asarray((qx, qy, qz))
            t = 2 * np.cross(u, v)
            return np.asarray(w.position) + v + qw * t + np.cross(u, t)

        self.robot_views = []
        for name, local in (("AUBObot", AUBO_LOCAL_POSITION_STATION), ("AUBObot_2", AUBO_2_LOCAL_POSITION_STATION)):
            path = ENV_ROOT + "/" + name
            p = robot_position(local)
            asset("aubo_with_gripper", path, p, rotation)
            root = stage.GetPrimAtPath(path + "/AUBO_E5")
            props = PhysxSchema.PhysxArticulationAPI.Apply(root)
            props.CreateEnabledSelfCollisionsAttr(ENABLE_SELF_COLLISIONS)
            props.CreateSolverPositionIterationCountAttr(SOLVER_POSITION_ITERATIONS)
            props.CreateSolverVelocityIterationCountAttr(
                max(SOLVER_VELOCITY_ITERATIONS, cfg.sim.physx.min_velocity_iteration_count)
            )
            view = Articulation(
                path + "/AUBO_E5",
                name=name,
                positions=torch.as_tensor(np.array([p]), dtype=torch.float32, device=cfg.sim.device),
                orientations=torch.tensor([rotation], device=cfg.sim.device),
            )
            self.world.scene.add(view)
            self.robot_views.append((name, view))
        target_path = ENV_ROOT + "/station/interactive/" + SCENE_ENTITY_TARGET
        state = TARGET_STATES[0]
        asset("sample_bottle", target_path, state.position_e, state.rotation_wxyz)
        rigid = [p for p in Usd.PrimRange(stage.GetPrimAtPath(target_path)) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(rigid) != 1:
            raise ValueError(f"Expected one bottle rigid body, found {len(rigid)}")
        body = rigid[0]
        UsdPhysics.RigidBodyAPI(body).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.RigidBodyAPI(body).CreateKinematicEnabledAttr(False)
        UsdPhysics.MassAPI.Apply(body).CreateMassAttr(SAMPLE_BOTTLE_MASS_KG)
        props = PhysxSchema.PhysxRigidBodyAPI.Apply(body)
        props.CreateDisableGravityAttr(False)
        props.CreateSolverPositionIterationCountAttr(16)
        props.CreateSolverVelocityIterationCountAttr(max(4, cfg.sim.physx.min_velocity_iteration_count))
        self.target_view = RigidPrim(str(body.GetPath()), name=SCENE_ENTITY_TARGET)
        self.world.scene.add(self.target_view)
        for prim in Usd.PrimRange.Stage(stage):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
        light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        light.CreateColorAttr(Gf.Vec3f(0.9, 0.9, 0.9))
        light.CreateIntensityAttr(500)
        self.world.get_physics_context().set_solver_type("TGS")
        self.world.get_physics_context().set_gravity(-9.81)
        self.world.reset()
        for name, view in self.robot_views:
            ids = [view.get_dof_index(n) for n in ROBOT_PRIM_CONTRACT.arm_joints + ROBOT_PRIM_CONTRACT.gripper_joints]
            q = [ARM_INITIAL_JOINT_POSITIONS[n] for n in ROBOT_PRIM_CONTRACT.arm_joints]
            q += [GRIPPER_INITIAL_JOINT_POSITIONS[n] for n in ROBOT_PRIM_CONTRACT.gripper_joints]
            view.set_joint_positions(torch.tensor([q], device=cfg.sim.device), joint_indices=ids)
            view.set_joint_velocities(torch.zeros((1, len(ids)), device=cfg.sim.device), joint_indices=ids)
            view.set_joint_position_targets(torch.tensor([q], device=cfg.sim.device), joint_indices=ids)
            for names, act in (
                (ROBOT_PRIM_CONTRACT.arm_joints, ARM_ACTUATOR),
                (ROBOT_PRIM_CONTRACT.gripper_joints, GRIPPER_ACTUATOR),
            ):
                joint_ids = [view.get_dof_index(n) for n in names]

                def tensor(value):
                    return torch.full((1, len(joint_ids)), value, device=cfg.sim.device)

                view.set_gains(kps=tensor(act["stiffness"]), kds=tensor(act["damping"]), joint_indices=joint_ids)
                view.set_max_efforts(tensor(act["effort_limit_sim"]), joint_indices=joint_ids)
                if names == ROBOT_PRIM_CONTRACT.arm_joints:
                    limits = getattr(cfg.scene, name).actuators["arm"].velocity_limit_sim
                    view.set_max_joint_velocities(
                        torch.tensor([[limits[n] for n in names]], device=cfg.sim.device), joint_indices=joint_ids
                    )
        self.world.physics_sim_view.update_articulations_kinematic()
        self.robot = RobotAdapter(self.robot_views[0][1])
        self.target = TargetAdapter(self.target_view)
        self.scene = SceneAdapter(stage, self.robot, self.target, cfg.sim.device)
        self.sim = SimAdapter(self)
        self.action_manager = NS(
            get_term=lambda name: NS(
                controller=NS(robot=self.robot, arm_ids=self.robot.find_joints(ROBOT_PRIM_CONTRACT.arm_joints)[0])
            )
        )
        self.camera_specs = ()
        if cfg.camera_resolution:
            self._create_cameras(cfg.camera_resolution)

    def _create_cameras(self, resolution):
        specs = project_diagnostic_camera_sensor_specs(resolution=resolution, data_types=("rgb",))
        root = self.world.stage.GetPrimAtPath(ENV_ROOT + "/AUBObot")
        matches = [p for p in Usd.PrimRange(root) if p.GetName() == "HandCamera" and p.IsA(UsdGeom.Camera)]
        if len(matches) != 1:
            raise ValueError("Robot USD must contain exactly one HandCamera")
        prim = matches[0]
        transform = Gf.Transform(UsdGeom.Xformable(prim).GetLocalTransformation())
        q = transform.GetRotation().GetQuat()
        hand = CameraSensorSpec(
            "HandCamera",
            str(prim.GetPath()),
            resolution,
            ("rgb",),
            tuple(transform.GetTranslation()),
            (q.GetReal(), *q.GetImaginary()),
            "opengl",
            (
                f"{asset_path(asset_by_key('aubo_with_gripper'))}:{prim.GetPath()}; "
                f"pose relative to {prim.GetParent().GetPath()}"
            ),
        )
        self.camera_specs = (
            hand,
            *(
                replace(s, name=name, prim_path=s.prim_path.replace("{ENV_REGEX_NS}", ENV_ROOT))
                for s, name in zip(specs[1:], ("CameraSensor_2", "CameraSensor_3"), strict=True)
            ),
        )
        cache = UsdGeom.XformCache()
        mount_body = prim.GetParent()
        while mount_body and not mount_body.HasAPI(UsdPhysics.RigidBodyAPI):
            mount_body = mount_body.GetParent()
        if not mount_body:
            raise ValueError("HandCamera must be mounted under a robot rigid body")
        flange_matrix = np.array(cache.GetLocalToWorldTransform(mount_body)).T
        hand_matrix = np.array(cache.GetLocalToWorldTransform(prim)).T
        mount_transform = np.linalg.inv(flange_matrix) @ hand_matrix
        for spec in self.camera_specs:
            camera = Camera(spec.prim_path, name=spec.name, resolution=resolution)
            if spec.name == "HandCamera":
                # Core normalizes xform ops during construction. Restore the
                # authored local calibration beneath the moving rigid body.
                camera.set_local_pose(
                    translation=np.array(spec.position_e), orientation=np.array(spec.rotation_wxyz), camera_axes="usd"
                )
            else:
                camera.set_world_pose(np.array(spec.position_e), np.array(spec.rotation_wxyz), camera_axes="usd")
                optics = UsdGeom.Camera(self.world.stage.GetPrimAtPath(spec.prim_path))
                m = DIAGNOSTIC_CAMERA_MODEL
                optics.CreateFocalLengthAttr(m.focal_length_mm)
                optics.CreateHorizontalApertureAttr(m.horizontal_aperture_mm)
                optics.CreateVerticalApertureAttr(m.horizontal_aperture_mm * resolution[1] / resolution[0])
                optics.CreateFocusDistanceAttr(m.focus_distance_mm)
                optics.CreateClippingRangeAttr(Gf.Vec2f(*m.clipping_range_m))
            camera.initialize()
            adapter = CameraAdapter(
                camera,
                self.cfg.sim.device,
                robot=self.robot if spec.name == "HandCamera" else None,
                mount_transform=mount_transform if spec.name == "HandCamera" else None,
                mount_body_name=mount_body.GetName() if spec.name == "HandCamera" else None,
            )
            self.cameras[spec.name] = adapter
            self.scene[spec.name] = adapter
        last_error = None
        warmup_started = monotonic()
        warmup_renders = 0
        while monotonic() - warmup_started < 180:
            self.sim.render()
            warmup_renders += 1
            try:
                for adapter in self.cameras.values():
                    adapter.refresh()
                    adapter.render_reference()
                    # A newly created render product can return a valid gray
                    # placeholder while geometry/materials are still loading.
                    if np.std(adapter.camera.get_rgb()) <= 5:
                        raise RuntimeError(f"Camera render is still warming up: {adapter.camera.prim_path}")
                break
            except RuntimeError as error:
                if "rendered pose is stale" in str(error):
                    raise
                last_error = error
                if warmup_renders % 200 == 0:
                    print(f"NATIVE_CAMERA_WARMUP renders={warmup_renders} last_error={error}", flush=True)
        else:
            raise RuntimeError(f"Native camera warmup did not complete: {last_error}")
        self.cameras_ready = True

    def reset(self):
        # The constructor establishes the same reset pose without physics settling.
        self.scene.update(self.cfg.sim.dt)
        return {}, {}

    def close(self):
        for adapter in self.cameras.values():
            adapter.camera.pause()
            adapter.parameters.detach([adapter.camera.get_render_product_path()])
        self.world.stop()
        self.world.clear_instance()
