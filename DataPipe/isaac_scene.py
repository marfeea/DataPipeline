"""Shared Isaac scene configuration for dataset collection and replay."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from configs.training import SIMULATION_DT_S
from configs.vision_diagnostics import DIAGNOSTIC_CAMERA_MODEL
from CurriculumRL.tasks.tcp_docking.dynamic_scene_cfg import TcpDockingDynamicSceneCfg
from CurriculumRL.tasks.tcp_docking.ik_env_cfg import TcpDockingIKEnvCfg

from isaaclab import sim as sim_utils
from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils import configclass

from .camera_config import project_diagnostic_camera_sensor_specs
from .config import GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POSITION


def _camera_cfg(spec: Any) -> CameraCfg:
    width, height = spec.resolution
    return CameraCfg(
        prim_path=spec.prim_path,
        update_period=0.0,
        height=height,
        width=width,
        data_types=list(spec.data_types),
        offset=CameraCfg.OffsetCfg(
            pos=spec.position_e,
            rot=spec.rotation_wxyz,
            convention=spec.pose_convention,
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=DIAGNOSTIC_CAMERA_MODEL.focal_length_mm,
            focus_distance=DIAGNOSTIC_CAMERA_MODEL.focus_distance_mm,
            horizontal_aperture=DIAGNOSTIC_CAMERA_MODEL.horizontal_aperture_mm,
            clipping_range=DIAGNOSTIC_CAMERA_MODEL.clipping_range_m,
        ),
    )


def make_dataset_ik_env_cfg(
    width: int,
    height: int,
    *,
    control_hz: float,
    target_state_index: int,
    downward_yaw_offset_rad: float = 0.0,
    flange_downward_enabled: bool = True,
    isolate_collection_sequence: bool = False,
    episode_length_s: float | None = None,
) -> tuple[Any, tuple[Any, ...]]:
    """Build the same dynamic scene used by collection and Isaac replay."""

    if width <= 0 or height <= 0:
        raise ValueError("camera width and height must be positive")
    if control_hz <= 0.0:
        raise ValueError("control_hz must be positive")
    if not np.isfinite(downward_yaw_offset_rad):
        raise ValueError("downward_yaw_offset_rad must be finite")
    if episode_length_s is not None and (not np.isfinite(episode_length_s) or episode_length_s <= 0.0):
        raise ValueError("episode_length_s must be finite and positive")
    camera_specs = project_diagnostic_camera_sensor_specs(resolution=(width, height))
    front_camera_cfg, side_camera_cfg, top_camera_cfg = tuple(_camera_cfg(spec) for spec in camera_specs)

    @configclass
    class DatasetSceneCfg(TcpDockingDynamicSceneCfg):
        camera_cfg = front_camera_cfg
        camera_cfg_2 = side_camera_cfg
        camera_cfg_3 = top_camera_cfg

    @configclass
    class DatasetIKEnvCfg(TcpDockingIKEnvCfg):
        scene: DatasetSceneCfg = DatasetSceneCfg(num_envs=1, env_spacing=4.0)

    cfg = DatasetIKEnvCfg()
    dataset_robot_cfg = deepcopy(cfg.scene.AUBObot)
    dataset_joint_pos = dict(dataset_robot_cfg.init_state.joint_pos)
    dataset_joint_pos.update(dict(zip(GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POSITION, strict=True)))
    dataset_robot_cfg.init_state.joint_pos = dataset_joint_pos
    cfg.scene.AUBObot = dataset_robot_cfg
    cfg.curriculum_enabled = False
    if episode_length_s is not None:
        cfg.episode_length_s = float(episode_length_s)
    cfg.evaluation_target_state_index = target_state_index
    cfg.evaluation_curriculum_level = 4
    decimation_f = 1.0 / (control_hz * SIMULATION_DT_S)
    decimation = int(round(decimation_f))
    if decimation < 1 or not np.isclose(decimation, decimation_f):
        raise ValueError("control_hz must map to an integer number of physics steps")
    cfg.decimation = decimation
    cfg.sim.render_interval = decimation
    cfg.trajectory_recording_enabled = True
    # 数据采集与静态 IK 场景共享同一几何语义：Flange 局部 -X 指向世界 -Z。
    # seed 仍取每个物理闭环策略步开始时的实际关节角，安全调度保持在 action term 内。
    cfg.actions.ik.flange_downward_enabled = flange_downward_enabled
    cfg.actions.ik.downward_yaw_offset_rad = downward_yaw_offset_rad
    cfg.actions.ik.flange_down_position_tolerance_m = 1.0e-2
    cfg.actions.ik.flange_down_orientation_tolerance_rad = 5.0e-3
    cfg.actions.ik.ik_seed_strategy = "current_joint_position"
    cfg.actions.ik.diagnostics_enabled = True
    if isolate_collection_sequence:
        # Dataset sequence success is owned by datapipe. Keep collision,
        # workspace, joint-limit and timeout terminations, but prevent formal
        # parking or intentional target interaction from resetting the env
        # before align/advance/close stages finish.
        cfg.terminations.curriculum_success = None
        cfg.terminations.parking_success = None
        cfg.terminations.target_disturbed = None
        cfg.rewards.success_terminal_reward = None
        cfg.rewards.safety_terminal_reward = None
    return cfg, camera_specs


__all__ = ["make_dataset_ik_env_cfg"]
