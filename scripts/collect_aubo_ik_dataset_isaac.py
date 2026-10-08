"""使用正式 AUBO IK ManagerBased 环境采集一小批真实 Isaac episode。

该入口只在 Isaac Lab Python 中运行。采集 HandCamera、CameraSensor_2 和 CameraSensor_3，
HDF5 保存每个控制步的完整数据；每 12 步导出 RGB PNG 和控制信息到
test/result/<批次>/<shard>/<episode>/，仅用于可视化验证。
"""

from __future__ import annotations

# ------------------------------ 启动前引导 ------------------------------
# Isaac Lab 必须先完成工程路径和包路径引导，再创建 AppLauncher。
# 这个顺序很重要：Isaac/Kit 相关模块不能像普通 Python 模块一样随意延后初始化。
import argparse
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_SOURCE_ROOT))

from DataPipe.collection_output import resolve_collection_output  # noqa: E402
from DataPipe.runtime import ensure_curriculumrl_available  # noqa: E402

ensure_curriculumrl_available()

from isaaclab.app import AppLauncher  # noqa: E402

from configs.task import NORMAL_ADVANCE_DISTANCE_M  # noqa: E402
from DataPipe.source2_routing import add_source2_routing_args, source2_routing_from_args  # noqa: E402
from DataPipe.state4_routing import add_state4_routing_args, routing_from_args  # noqa: E402

# ------------------------------ Isaac 采集参数 ------------------------------
# 这里集中定义输出目录、episode 数量、轨迹规划、控制频率和相机分辨率。
# AppLauncher 的参数会一并注入，故可继续使用 --headless、--device 等 Isaac 参数。
# ``parser`` 定义采集入口的命令行契约；AppLauncher 参数会在下方一并注入。
parser = argparse.ArgumentParser(description=__doc__)
add_state4_routing_args(parser)
add_source2_routing_args(parser)
parser.set_defaults(state4_routing=True)
# 输出与批次规模：``output-dir`` 为 HDF5 目标目录，``episodes``/``max-steps`` 控制采集上限。
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--max-steps", type=int, default=960, help="每条 episode 的最大 Isaac 控制步数")
parser.add_argument(
    "--waypoint-position-tolerance-m",
    type=float,
    default=0.01,
    help="实际 TCP 进入该距离后才推进到下一 waypoint，单位 m",
)
# 可复现性与目标：``seed`` 是批次基准随机种子，``target-state-index`` 选择正式目标状态。
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--target-state-index", type=int, default=0)
parser.add_argument("--run-tag", default="isaac-ik-batch")
# 频率：粗规划在 ``planning-hz`` 生成路径，正式环境按 ``control-hz`` 发送 TCP action。
parser.add_argument("--planning-hz", type=float, default=4.0)
parser.add_argument("--control-hz", type=float, default=12.0)
# 轨迹形状：粗规划点数可固定或在 min/max 间采样，``xy-fraction`` 控制横向阶段比例。
parser.add_argument("--coarse-min-waypoints", type=int, default=30)
parser.add_argument("--coarse-max-waypoints", type=int, default=40)
parser.add_argument("--planning-waypoints", type=int, default=None)
parser.add_argument("--xy-fraction-min", type=float, default=0.40)
parser.add_argument("--xy-fraction-max", type=float, default=0.60)
parser.add_argument("--trajectory-mode", choices=("direct", "staged"), default="staged")
parser.add_argument("--high-clearance-z", type=float, default=None)
parser.add_argument("--base-clockwise-deg", type=float, default=45.0)
parser.add_argument("--downward-yaw-offset-deg", type=float, default=0.0)
parser.add_argument(
    "--sequence-mode",
    choices=("pick-place",),
    default="pick-place",
    help="采集抓取、抬升、放置和松爪的完整序列",
)
parser.add_argument(
    "--log-sequence-stage-transitions",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="打印序列进入阶段的日志；默认开启，可用 --no-log-sequence-stage-transitions 关闭",
)
parser.add_argument(
    "--log-place-final-xy-conditions",
    action="store_true",
    help="逐控制步打印 PLACE_FINAL_Z -> PLACE_FINAL_XY 的判定条件；默认关闭",
)
parser.add_argument(
    "--normal-advance-distance-m",
    type=float,
    default=NORMAL_ADVANCE_DISTANCE_M,
    help="preposition 到闭爪抓取 TCP 点的距离，必须与 task.py 配置一致，单位 m",
)
parser.add_argument(
    "--place-state-index",
    type=int,
    default=None,
    help="必填；将物体放置到 TARGET_STATES 中的目标状态索引",
)
parser.add_argument(
    "--grasp-height-m",
    type=float,
    default=None,
    help="覆盖所有 pick/place TCP 点的机器人根系 B Z 高度，单位 m；用于统一抓取高度测试",
)
parser.add_argument(
    "--lift-clearance-m",
    type=float,
    default=None,
    help="必填；抬升安全高度为 max(抓取 Z, 放置 Z) 加该间隙",
)
parser.add_argument(
    "--normal-angle-tolerance-deg",
    type=float,
    default=5.0,
    help="抓取和放置的 XOY 平面 heading 到位容差，单位度",
)
parser.add_argument("--normal-dwell-steps", type=int, default=3)
parser.add_argument("--gripper-dwell-steps", type=int, default=3)
parser.add_argument("--gripper-position-tolerance", type=float, default=0.008)
parser.add_argument(
    "--gripper-open-position",
    type=float,
    nargs=2,
    metavar=("UPPER", "DOWN"),
    default=(0.0115, -0.0115),
)
parser.add_argument(
    "--gripper-closed-position",
    type=float,
    nargs=2,
    metavar=("UPPER", "DOWN"),
    default=None,
    help="必填；两个夹爪关节闭合目标，运行时按 USD 限位校验",
)
# 视觉与 Kit 缓存：三路相机共用分辨率；缓存目录隔离 OptiX 和 Kit 用户配置。
parser.add_argument("--camera-width", type=int, default=128)
parser.add_argument("--camera-height", type=int, default=128)
parser.add_argument("--kit-cache-dir", type=Path, default=Path("data/isaac_kit_cache"))
# ``args_cli`` 是解析后的所有命令行参数；``enable_cameras`` 强制打开 CameraSensor 输出。
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
requested_output_dir = args_cli.output_dir
args_cli.output_dir = resolve_collection_output(requested_output_dir)
if args_cli.output_dir != requested_output_dir:
    print(
        f"DATASET_OUTPUT_REDIRECT reason=legacy_v3 requested={requested_output_dir} "
        f"output={args_cli.output_dir}",
        flush=True,
    )
args_cli.enable_cameras = True
# ``kit_cache_dir`` 是绝对缓存目录；``user_config_path`` 供 Kit 启动参数指向可写配置文件。
kit_cache_dir = args_cli.kit_cache_dir.resolve()
kit_cache_dir.mkdir(parents=True, exist_ok=True)
# OptiX 缓存放在本次采集专用目录，减少不同运行之间的缓存冲突。
os.environ["OPTIX_CACHE_PATH"] = str(kit_cache_dir / "optix")
user_config_path = kit_cache_dir / "user.config.json"
# ``kit_args`` 合并用户传入的 Kit 参数、本脚本生成的可写缓存目录和 userConfigPath。
# Material Library 使用 Kit 的 ``${cache}`` token；仅设置 userConfigPath 不会改变它，
# 因而默认把整个 Kit cache 根目录也指向 ``--kit-cache-dir``，避免写入安装目录失败。
kit_args = args_cli.kit_args or ""
if "--/app/cache=" not in kit_args:
    kit_args = f"{kit_args} --/app/cache={kit_cache_dir}".strip()
args_cli.kit_args = " ".join(
    argument for argument in (kit_args, f"--/app/userConfigPath={user_config_path}") if argument
)

# ------------------------------ 启动 Isaac Sim/Kit ------------------------------
# CameraSensor 需要在 Isaac 图形/渲染运行时中创建；即使无窗口采集，也要显式启用相机。
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------ 正式任务与数据管线依赖 ------------------------------
# 以下模块必须在 Isaac App 启动后导入。采集器消费正式任务环境的 action、reward、termination
# 和 AUBO IK controller，避免采集脚本自行复制一套与训练不一致的任务逻辑。
import CurriculumRL.tasks  # noqa: F401, E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from CurriculumRL.logic.aubo_ik import (  # noqa: E402
    FLANGE_DOWNWARD_AXIS_LOCAL,
    flange_down_tcp_target_pose,
    flange_down_yaw_offset_for_tcp_heading,
    tcp_heading_decomposition,
)
from CurriculumRL.tasks.tcp_docking.aubo_ik_action import AuboIKActionTerm  # noqa: E402
from CurriculumRL.tasks.tcp_docking.mdp.runtime_state import auxiliary_reward_scales, compute_step  # noqa: E402

from configs.assets import SCENE_ENTITY_AUBO, SCENE_ENTITY_TARGET  # noqa: E402
from configs.coordinate_frames import (  # noqa: E402
    FLANGE_NORMAL_AXIS_F,
    FLANGE_TO_TCP_ROTATION_F_WXYZ,
    FLANGE_TO_TCP_TRANSLATION_F,
    TCP_DIRECTION_AXIS_T,
    target_direction_in_world,
)
from configs.render import TEST_RENDER_CFG  # noqa: E402
from configs.task import (  # noqa: E402
    TARGET_STATES,
    TOOL_FORWARD_AXIS,
)
from configs.training import SIMULATION_DT_S  # noqa: E402
from DataPipe import (  # noqa: E402
    PLACE_TRANSFER_HEADING_FRACTION,
    CameraCaptureAdapter,
    CameraFrame,
    ControlState,
    DatasetConfig,
    DatasetSequence,
    EpisodeBuilder,
    EpisodeOutcome,
    EpisodeSupervisor,
    SequenceConfig,
    SequenceObservation,
    SequenceStage,
    StateCaptureAdapter,
    TerminationReason,
    planar_heading_from_quaternion,
    wrap_angle_rad,
)
from DataPipe.alignment import pose_matrix
from DataPipe.collection_alignment import aligned_snapshot, applied_action, xyzw_pose
from DataPipe.collection_cameras import (  # noqa: E402
    CameraExportingTrajectoryWriter,
    add_camera_world_poses,
    configure_collection_cameras,
)
from DataPipe.config import ALIGNED_SCHEMA_VERSION, TCP_HEADING_SCALE_RAD, TCP_POSITION_SCALE_M  # noqa: E402
from DataPipe.episode_naming import next_episode_index  # noqa: E402
from DataPipe.isaac_gripper import DatasetGripperController  # noqa: E402
from DataPipe.isaac_scene import make_dataset_ik_env_cfg  # noqa: E402
from DataPipe.pipelines.curobo_bootstrap.replay import pose_w_to_b as measured_pose_w_to_b  # noqa: E402
from DataPipe.pipelines.curobo_bootstrap.task_targets import (  # noqa: E402
    environment_position_to_base,
    pick_place_positions_b,
)
from DataPipe.trajectory import advance_reached_waypoints, make_plan  # noqa: E402

# ------------------------------ 写入的奖励分量 ------------------------------
# 这些名字与正式 runtime reward 对齐；每一步同时保存总 reward 和分量，便于重播验收、
# reward 诊断以及后续 reward prediction loss 训练。
RECORDED_REWARD_COMPONENTS = (
    "position_reward",
    "progress_reward",
    "best_progress_reward",
    "stall_penalty",
    "far_move_reward",
    "near_inefficiency_penalty",
    "time_penalty",
    "joint_soft_limit_penalty",
    "tool_axis_progress",
    "path_reference_progress",
    "path_reference_reached",
    "success_terminal_reward",
    "timeout_terminal_reward",
    "safety_terminal_reward",
)

# 同一个正式控制器只需核对一次，避免每个控制步都把 GPU 标定张量复制到 CPU。
_VALIDATED_COORDINATE_CONTROLLER_IDS: set[int] = set()


# ------------------------------ 场景配置与单环境适配 ------------------------------
# 数据集采集使用共享场景配置：AUBO、目标、三路 CameraSensor 以及控制频率在此统一生成。
def _make_env_cfg(
    width: int,
    height: int,
    *,
    control_hz: float,
    target_state_index: int,
    downward_yaw_offset_rad: float,
    isolate_collection_sequence: bool,
    episode_length_s: float | None,
) -> tuple[Any, tuple[Any, ...]]:
    """创建单环境采集配置，并返回配置对象与三路相机规格。

    ``width``/``height`` 是三路相机共用的输出分辨率；``control_hz``
    决定环境每次 ``step`` 对应多少个 Isaac 物理步；``target_state_index``
    选择正式任务中的目标停靠状态。场景实体、IK action term、奖励和终止
    逻辑都由 ``make_dataset_ik_env_cfg`` 统一提供，本函数只做入口层转发。
    """

    cfg, camera_specs = make_dataset_ik_env_cfg(
        width,
        height,
        control_hz=control_hz,
        target_state_index=target_state_index,
        downward_yaw_offset_rad=downward_yaw_offset_rad,
        flange_downward_enabled=True,
        isolate_collection_sequence=isolate_collection_sequence,
        episode_length_s=episode_length_s,
    )
    return configure_collection_cameras(cfg, camera_specs)


def _first_env(value: Any) -> Any:
    """从 batch 维取出第 0 个环境，同时保留嵌套字典结构。

    Isaac 环境只有一个并行环境，但其张量通常仍带有首维 ``num_envs``。
    递归处理字典是为了让观测中的命名字段保持原有结构；标量张量不再
    索引，普通 Python 对象则原样返回。
    """

    if isinstance(value, dict):
        return {str(key): _first_env(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value[0] if value.ndim > 0 else value
    return value


# ------------------------------ 相机帧采集 ------------------------------
# 从三路 CameraSensor 同步取出配置要求的全部模态，并记录仿真步号和时间戳。
# 后续 _snapshot 会把这些模态映射到 HDF5 的 image/depth/mask 字段。
def _camera_observation(
    cameras: dict[str, Any],
    camera_specs: tuple[Any, ...],
    sim_step: int,
    timestamp_s: float,
) -> tuple[tuple[CameraFrame, ...], dict[str, dict[str, Any]]]:
    """在同一个 Isaac 控制步读取三路相机，并整理为两种消费格式。

    ``frames`` 是交给 ``CameraCaptureAdapter`` 做同步校验的帧对象序列；
    ``values`` 是交给 ``_snapshot`` 写入观测的
    ``camera_name -> data_type -> tensor`` 字典。相机输出的第 0 个环境
    被取出，但不会在这里转 NumPy，具体转换延迟到快照构造阶段完成。
    """

    # ``frames`` 保存带相机名、仿真步和时间戳的完整帧元数据。
    frames: list[CameraFrame] = []
    # ``values`` 保存当前控制步各相机的 RGB/depth/mask 原始张量。
    values: dict[str, dict[str, Any]] = {}
    # ``spec`` 是一条相机的静态契约，决定要读取哪些 data_type。
    for spec in camera_specs:
        # Isaac CameraSensor 的输出字典；每个 value 的首维是并行环境维。
        outputs = cameras[spec.name].data.output
        # 配置要求但传感器未提供的输出类型；缺失时立即失败，避免静默写坏数据。
        missing = set(spec.data_types).difference(outputs)
        if missing:
            raise RuntimeError(f"CameraSensor {spec.name} 缺少输出：{sorted(missing)}")
        # 去掉单环境 batch 维，保留当前相机要求的所有图像/掩码字段。
        camera_values = {name: outputs[name][0] for name in spec.data_types}
        # 用 RGB（若存在）作为 shape/dtype 的代表；否则使用第一个输出字段。
        representative = camera_values.get("rgb", next(iter(camera_values.values())))
        # 构造供同步适配器校验的不可变帧描述。
        frames.append(
            CameraFrame(
                camera_name=spec.name,
                sim_step=sim_step,
                timestamp_s=timestamp_s,
                data=camera_values,
                shape=tuple(representative.shape),
                dtype=str(representative.dtype),
            )
        )
        # 同一份原始数据也按相机名保存，供后续 HDF5 observation 快照使用。
        values[spec.name] = camera_values
    return tuple(frames), values


# ------------------------------ 物理状态与目标坐标快照 ------------------------------
# Isaac 张量在这里转换为 CPU NumPy，形成 writer 可序列化的显式字段：q、dq、TCP、目标和相机。
def _numpy(value: Any) -> np.ndarray:
    """将 Isaac/Torch 值复制为 CPU NumPy 数组，避免保留计算图或 GPU 引用。"""

    if isinstance(value, torch.Tensor):
        # ``detach`` 切断 autograd，``cpu`` 保证 writer 可序列化，``copy`` 隔离底层存储。
        return value.detach().cpu().numpy().copy()
    # 兼容已是 NumPy/标量/数组协议对象的输入，同样复制一份独立数据。
    return np.asarray(value).copy()



def _measured_base_pose_w(robot: Any) -> tuple[float, ...]:
    """Normalize Isaac's measured root quaternion before strict frame conversion."""

    position = _numpy(robot.data.root_pos_w[0]).astype(np.float64)
    quaternion = _numpy(robot.data.root_quat_w[0]).astype(np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(position).all() or not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("invalid measured robot root pose")
    quaternion /= norm
    return (*position, *quaternion)


def _environment_position_b(unwrapped: Any, position_e: tuple[float, float, float]) -> np.ndarray:
    """Convert one environment-frame position to robot-root B."""

    robot = unwrapped.scene[SCENE_ENTITY_AUBO]
    return environment_position_to_base(
        position_e,
        environment_origin_w=_numpy(unwrapped.scene.env_origins[0]).astype(np.float64),
        base_pose_w=_measured_base_pose_w(robot),
    )


def _target_position_b(
    unwrapped: Any, target_state_index: int, grasp_height_m: float | None = None
) -> np.ndarray:
    """把目标状态中的末端位置从环境世界系转换到机器人根坐标系 ``B``。"""

    # ``target_state`` 是正式任务目标表中的一个目标状态；其 ``preposition_e`` 为环境系位置。
    target_state = TARGET_STATES[target_state_index]
    position_b = _environment_position_b(unwrapped, target_state.preposition_e)
    if grasp_height_m is not None:
        position_b[2] = grasp_height_m
    return position_b


def _object_position_b(unwrapped: Any, target_state_index: int) -> np.ndarray:
    """Return the prescribed target object's position in robot-root B."""

    return _environment_position_b(unwrapped, TARGET_STATES[target_state_index].position_e)


def _grasp_tcp_position_b(
    unwrapped: Any, target_state_index: int, grasp_height_m: float | None = None
) -> np.ndarray:
    """Return the selected state's configured grasp TCP position in robot-root B."""

    position_b = _environment_position_b(unwrapped, TARGET_STATES[target_state_index].grasp_tcp_position_e)
    if grasp_height_m is not None:
        position_b[2] = grasp_height_m
    return position_b


def _target_tcp_direction_w(target_state_index: int) -> np.ndarray:
    """由目标物配置计算 TCP 应对齐的世界系方向，不掺入机器人 Base 坐标系。"""

    return np.asarray(
        target_direction_in_world(TARGET_STATES[target_state_index].rotation_wxyz),
        dtype=np.float64,
    )


def _pick_place_positions_b(
    unwrapped: Any,
    grasp_state_index: int,
    place_state_index: int,
    lift_clearance_m: float,
    grasp_height_m: float | None = None,
    state4_routing=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plan lift, high-XY and final place points from source/destination states."""

    if grasp_height_m is not None:
        # The V2 contract intentionally has no height override.  Preserve the
        # Legacy CLI behavior until that compatibility option is retired.
        grasp_b = _environment_position_b(unwrapped, TARGET_STATES[grasp_state_index].grasp_tcp_position_e)
        place_b = _environment_position_b(unwrapped, TARGET_STATES[place_state_index].grasp_tcp_position_e)
        grasp_b[2] = grasp_height_m
        place_b[2] = grasp_height_m
        safe_z_b = max(grasp_b[2], place_b[2]) + lift_clearance_m
        lift_b, place_high_b = grasp_b.copy(), place_b.copy()
        lift_b[2] = place_high_b[2] = safe_z_b
        return lift_b, place_high_b, place_b
    robot = unwrapped.scene[SCENE_ENTITY_AUBO]
    positions = pick_place_positions_b(
        environment_origin_w=_numpy(unwrapped.scene.env_origins[0]).astype(np.float64),
        base_pose_w=_measured_base_pose_w(robot),
        source_index=grasp_state_index,
        destination_index=place_state_index,
        lift_clearance_m=lift_clearance_m,
        state4_routing=state4_routing,
    )
    return positions["lift"], positions["place_high"], positions["place"]


def _validate_cli_args(args: argparse.Namespace) -> tuple[SequenceConfig, str]:
    """Validate collection arguments and return the sequence configuration."""

    if not 0 <= args.target_state_index < len(TARGET_STATES):
        raise ValueError(f"target-state-index must be in [0, {len(TARGET_STATES) - 1}]")
    if args.place_state_index is not None and not 0 <= args.place_state_index < len(TARGET_STATES):
        raise ValueError(f"place-state-index must be in [0, {len(TARGET_STATES) - 1}]")
    if args.planning_hz <= 0.0 or args.control_hz <= 0.0:
        raise ValueError("planning-hz and control-hz must be positive")
    if args.coarse_min_waypoints <= 0 or args.coarse_max_waypoints < args.coarse_min_waypoints:
        raise ValueError("coarse waypoint bounds are invalid")
    if not 0.0 < args.xy_fraction_min <= args.xy_fraction_max < 1.0:
        raise ValueError("xy-fraction bounds must be in (0, 1)")
    if args.planning_waypoints is not None and args.planning_waypoints <= 0:
        raise ValueError("planning-waypoints must be positive")
    if args.episodes <= 0 or args.max_steps <= 0:
        raise ValueError("episodes 和 max_steps 必须为正数")
    if not np.isfinite(args.waypoint_position_tolerance_m) or args.waypoint_position_tolerance_m <= 0.0:
        raise ValueError("waypoint-position-tolerance-m 必须为正数")
    if not np.isfinite(args.normal_advance_distance_m) or args.normal_advance_distance_m <= 0.0:
        raise ValueError("normal-advance-distance-m 必须为正数")
    if not np.isclose(args.normal_advance_distance_m, NORMAL_ADVANCE_DISTANCE_M, rtol=0.0, atol=1.0e-9):
        raise ValueError("normal-advance-distance-m 必须与 task.py 的 NORMAL_ADVANCE_DISTANCE_M 一致")
    if args.gripper_closed_position is None:
        raise ValueError("必须显式提供 --gripper-closed-position UPPER DOWN")
    if args.place_state_index is None:
        raise ValueError("必须显式提供 --place-state-index INDEX")
    invalid_lift_clearance = (
        args.lift_clearance_m is None or not np.isfinite(args.lift_clearance_m) or args.lift_clearance_m <= 0.0
    )
    if invalid_lift_clearance:
        raise ValueError("必须显式提供正的 --lift-clearance-m")
    sequence_config = SequenceConfig(
        position_tolerance_m=args.waypoint_position_tolerance_m,
        normal_angle_tolerance_rad=math.radians(args.normal_angle_tolerance_deg),
        normal_dwell_steps=args.normal_dwell_steps,
        advance_distance_m=float(args.normal_advance_distance_m),
        gripper_position_tolerance=args.gripper_position_tolerance,
        gripper_dwell_steps=args.gripper_dwell_steps,
        log_place_final_xy_conditions=args.log_place_final_xy_conditions,
    )
    sequence_config.validate()
    if not np.isfinite(args.downward_yaw_offset_deg):
        raise ValueError("downward-yaw-offset-deg 必须为有限值")
    if args.camera_width <= 0 or args.camera_height <= 0:
        raise ValueError("camera width/height 必须为正数")
    if args.grasp_height_m is not None and (
        not np.isfinite(args.grasp_height_m) or args.grasp_height_m <= 0.0
    ):
        raise ValueError("grasp-height-m 必须为正数")
    return sequence_config, TARGET_STATES[args.place_state_index].name


def _snapshot(
    unwrapped: Any,
    controller: Any,
    target_position_b: np.ndarray,
    target_quaternion_b: np.ndarray,
    camera_payload: dict[str, dict[str, Any]],
    gripper_controller: DatasetGripperController,
    object_position_b: np.ndarray,
    desired_approach_axis_b: np.ndarray,
    advance_target_position_b: np.ndarray,
    semantic_stage_id: int = 0,
    waypoint_index: int = 0,
) -> dict[str, Any]:
    """生成一个观测快照，并把 Isaac 状态映射为 HDF5 的显式字段。

    位置/姿态字段使用机器人根坐标系 ``B``；``q``、``dq`` 和 IK 输出只
    保留正式控制器选择的六个关节。相机数据仍按 ``camera_payload`` 中的
    data_type 映射为稳定的数据集字段名。语义阶段与路径点索引分别记录。
    """

    # 从正式场景读取 AUBO 实体，而不是在采集层维护另一份机器人状态。
    robot = unwrapped.scene[SCENE_ENTITY_AUBO]
    # 控制器返回当前 TCP 在机器人根系 B 中的位置、四元数和可能的附加值。
    tcp_position_b, tcp_quaternion_b, _ = controller.current_tcp_pose_b()
    gripper_position, gripper_velocity = gripper_controller.state()
    # ``value`` 是一帧 observation 的字段字典，最终由 EpisodeBuilder 写入 writer。
    value: dict[str, Any] = {
        "q": _numpy(robot.data.joint_pos[0, controller.joint_ids]).astype(np.float32),
        "dq": _numpy(robot.data.joint_vel[0, controller.joint_ids]).astype(np.float32),
        "tcp_pose_wxyz": np.concatenate(
            (_numpy(tcp_position_b[0]), _numpy(tcp_quaternion_b[0]).astype(np.float32))
        ).astype(np.float32),
        "target_pose_wxyz": np.concatenate((target_position_b, target_quaternion_b)).astype(np.float32),
        "gripper_position": np.asarray(gripper_position, dtype=np.float32),
        "gripper_velocity": np.asarray(gripper_velocity, dtype=np.float32),
        "semantic_stage_id": np.asarray(semantic_stage_id, dtype=np.uint8),
        "waypoint_index": np.asarray(waypoint_index, dtype=np.uint16),
        "preposition_position_b": np.asarray(target_position_b, dtype=np.float32),
        "object_position_b": np.asarray(object_position_b, dtype=np.float32),
        "desired_approach_axis_b": np.asarray(desired_approach_axis_b, dtype=np.float32),
        "advance_target_position_b": np.asarray(advance_target_position_b, dtype=np.float32),
    }
    root_quaternion = _numpy(robot.data.root_quat_w[0])
    root_rotation = pose_matrix(xyzw_pose((0, 0, 0), root_quaternion))[:3, :3]
    grasp_position = _grasp_tcp_position_b(unwrapped, args_cli.target_state_index, args_cli.grasp_height_m)
    direction_b = root_rotation.T @ _target_tcp_direction_w(args_cli.target_state_index)
    heading = math.atan2(direction_b[1], direction_b[0])
    grasp_yaw = flange_down_yaw_offset_for_tcp_heading(
        grasp_position, heading, root_quaternion_wxyz=root_quaternion,
        tcp_heading_axis_local=TCP_DIRECTION_AXIS_T,
    )
    _, grasp_quaternion = flange_down_tcp_target_pose(
        grasp_position, root_quaternion_wxyz=root_quaternion, downward_yaw_offset_rad=grasp_yaw,
    )
    value.update(aligned_snapshot(
        robot, controller, gripper_controller, env_origin=_numpy(unwrapped.scene.env_origins[0]),
        source_state=TARGET_STATES[args_cli.target_state_index], goal_state=TARGET_STATES[args_cli.place_state_index],
        tcp_position_b=_numpy(tcp_position_b[0]), tcp_quaternion_b=_numpy(tcp_quaternion_b[0]),
        grasp_position_b=grasp_position, grasp_quaternion_b=grasp_quaternion,
    ))
    # 将 Isaac data_type 名称转换为数据集字段名；未列出的类型不写盘。
    camera_key_names = {
        "rgb": "image",
        "distance_to_image_plane": "depth",
        "semantic_segmentation": "foreground_mask",
        "instance_segmentation_fast": "instance_mask",
    }
    # ``camera_name`` 是场景实体名，``values`` 是该相机当前步的全部输出。
    for camera_name, values in camera_payload.items():
        # ``data_type`` 是 CameraSensor 输出类型，``data`` 是单环境原始张量。
        for data_type, data in values.items():
            # ``field_name`` 是最终写入的 image/depth/mask 字段；None 表示忽略。
            field_name = camera_key_names.get(data_type)
            if field_name is not None:
                # 字段保留相机来源名，例如 image_HandCamera、depth_CameraSensor_2。
                value[f"{field_name}_{camera_name}"] = _numpy(data)
    add_camera_world_poses(value, unwrapped, camera_payload)
    return value


# ------------------------------ 正式 IK 控制器状态 ------------------------------
# 读取正式 AuboIKActionTerm 的 raw/processed action、IK 输出和实际关节状态，
# 使每条 transition 同时保留“策略动作”和“控制器执行结果”。
def _ik_controller(unwrapped: Any) -> Any:
    """获取正式 IK 控制器；reset 后首帧尚未产生 diagnostics。"""

    controller_term = unwrapped.action_manager.get_term("ik")
    if not isinstance(controller_term, AuboIKActionTerm):
        raise RuntimeError("正式 IK 环境的 action term 不是 AuboIKActionTerm")
    controller = controller_term.controller
    controller_id = id(controller)
    if controller_id not in _VALIDATED_COORDINATE_CONTROLLER_IDS:
        _validate_controller_coordinate_contract(controller)
        _VALIDATED_COORDINATE_CONTROLLER_IDS.add(controller_id)
    return controller


def _validate_controller_coordinate_contract(controller: Any) -> None:
    """发现 DataPipe 与 CurriculumRL 的几何定义漂移时立即终止采集。"""

    checks = (
        ("flange normal -X_F", FLANGE_DOWNWARD_AXIS_LOCAL, FLANGE_NORMAL_AXIS_F),
        (
            "flange-to-TCP translation p_FT^F",
            controller.flange_to_tcp_translation_f,
            FLANGE_TO_TCP_TRANSLATION_F,
        ),
    )
    for name, actual, expected in checks:
        actual_array = _numpy(actual).reshape(-1, 3)[0]
        if not np.allclose(actual_array, np.asarray(expected, dtype=np.float64), atol=1.0e-7):
            raise RuntimeError(f"坐标契约不一致：{name}: controller={actual_array.tolist()}, central={expected}")

    actual_q = _numpy(controller.flange_to_tool_rotation_f).reshape(-1, 4)[0]
    actual_q = actual_q / np.linalg.norm(actual_q)
    expected_q = np.asarray(FLANGE_TO_TCP_ROTATION_F_WXYZ, dtype=np.float64)
    if not math.isclose(abs(float(np.dot(actual_q, expected_q))), 1.0, abs_tol=1.0e-7):
        raise RuntimeError(
            "坐标契约不一致：flange-to-TCP rotation q_FT: "
            f"controller={actual_q.tolist()}, central={FLANGE_TO_TCP_ROTATION_F_WXYZ}"
        )


def _control_state(
    unwrapped: Any,
    sim_step: int,
    waypoint_tcp_position_b: np.ndarray | None = None,
    *,
    dataset_action: np.ndarray | None = None,
    gripper_controller: DatasetGripperController | None = None,
    semantic_stage_id: int = 0,
    waypoint_index: int = 0,
) -> tuple[Any, ControlState]:
    """读取正式 IK action term 的输入、输出和实测六关节状态。

    返回的第一个值是底层控制器对象，后续快照需要用它读取 TCP 位姿；
    第二个值是数据层 ``ControlState``，记录同一 ``sim_step`` 的动作与
    实际关节状态。这里的 ``raw_action``/``normalized_action`` 沿用正式
    控制器当前接口的定义，TCP 位移来自 ``processed_action``，IK 输出
    来自 ``desired_joint_target``。
    """

    # ``controller`` 是 AuboIKActionTerm 持有的 Differential IK 控制器实例。
    controller = _ik_controller(unwrapped)
    # ``robot`` 用于读取控制器选中关节的真实位置和速度。
    robot = unwrapped.scene[SCENE_ENTITY_AUBO]
    actual_tcp_position_b, _, _ = controller.current_tcp_pose_b()
    diagnostics = controller.diagnostics
    if waypoint_tcp_position_b is None:
        waypoint_tcp_position_b = _numpy(controller.target_tcp_position_b[0]).astype(np.float64)
    if dataset_action is None:
        dataset_action = np.concatenate((_numpy(controller.raw_action[0]), np.asarray((0.0, 1.0))))
    if gripper_controller is None:
        raise ValueError("gripper_controller is required by the sequence-v3 data contract")
    gripper_position, gripper_velocity = gripper_controller.state()
    joint_position_array = _numpy(robot.data.joint_pos[0, controller.joint_ids]).astype(np.float64)
    heading = tcp_heading_decomposition(
        joint_position_array,
        tcp_heading_axis_local=TCP_DIRECTION_AXIS_T,
    )
    heading_target_rad = float(dataset_action[6] * TCP_HEADING_SCALE_RAD)
    # ``state`` 将 GPU 张量转成 Python 标量元组，形成可序列化的控制步快照。
    state = ControlState(
        # ``sim_step``/``sim_time_s`` 标识这批动作和状态对应的 Isaac 时间点。
        sim_step=sim_step,
        sim_time_s=float(sim_step * unwrapped.step_dt),
        action_applied=applied_action(
            controller, float(dataset_action[-1]), TCP_POSITION_SCALE_M, TCP_HEADING_SCALE_RAD
        ),
        raw_action=tuple(float(value) for value in dataset_action),
        normalized_action=tuple(float(value) for value in dataset_action),
        tcp_displacement=tuple(float(value) for value in controller.processed_action[0].detach().cpu().tolist()),
        ik_output=tuple(float(value) for value in controller.desired_joint_target[0].detach().cpu().tolist()),
        joint_position=tuple(float(value) for value in joint_position_array),
        joint_velocity=tuple(
            float(value) for value in robot.data.joint_vel[0, controller.joint_ids].detach().cpu().tolist()
        ),
        observation=None,
        waypoint_tcp_position_b=tuple(float(value) for value in waypoint_tcp_position_b),
        controller_target_tcp_position_b=tuple(
            float(value) for value in controller.target_tcp_position_b[0].detach().cpu().tolist()
        ),
        ik_seed_joint_position=tuple(
            float(value) for value in controller.ik_seed_joint_position[0].detach().cpu().tolist()
        ),
        ik_solution_joint_position=tuple(
            float(value) for value in controller.ik_solution_joint_position[0].detach().cpu().tolist()
        ),
        limited_joint_target=tuple(
            float(value) for value in controller.desired_joint_target[0].detach().cpu().tolist()
        ),
        commanded_joint_target=tuple(
            float(value) for value in controller.commanded_joint_target[0].detach().cpu().tolist()
        ),
        actual_tcp_position_b=tuple(float(value) for value in actual_tcp_position_b[0].detach().cpu().tolist()),
        gripper_joint_position=gripper_position,
        gripper_joint_velocity=gripper_velocity,
        tcp_heading_target_rad=heading_target_rad,
        tcp_heading_actual_rad=heading.total_heading_rad,
        tcp_heading_first_five_rad=heading.first_five_heading_rad,
        tcp_heading_sixth_axis_rad=heading.sixth_axis_heading_rad,
        tcp_heading_error_rad=wrap_angle_rad(heading_target_rad - heading.total_heading_rad),
        gripper_command=float(dataset_action[-1]),
        semantic_stage_id=semantic_stage_id,
        waypoint_index=waypoint_index,
        ik_success=bool(diagnostics.ik_success[0].item()),
        delta_limited=bool(diagnostics.delta_limited[0].item()),
        velocity_limited=bool(diagnostics.velocity_limited[0].item()),
        position_limited=bool(diagnostics.position_limited[0].item()),
    )
    return controller, state


# ------------------------------ 正式奖励分量读取 ------------------------------
# 直接调用 runtime_state.compute_step，并按项目 RewardManager 的缩放关系还原每步分量。
# 这里不在采集层重新实现奖励公式，保证录制值与训练环境一致。
def _reward_components(unwrapped: Any) -> dict[str, float]:
    """读取正式 runtime reward 分量；不在采集层重算任务奖励。

    ``compute_step`` 返回正式 MDP 计算结果，``auxiliary_reward_scales``
    还原 RewardManager 实际累加到 ``reward_buf`` 的每步量。这样数据集
    同时保存总 reward 与各正式分量，且不会复制一套容易漂移的奖励公式。
    """

    # ``_`` 是正式计算返回但本函数不需要的 step metrics；``rewards`` 包含各分量张量。
    _, rewards = compute_step(unwrapped)
    # mdp reward terms divide by step_dt and RewardManager multiplies by step_dt;
    # values*auxiliary_scale is therefore the same per-step quantity accumulated
    # in env.reward_buf, while preserving each formal component separately.
    # ``name`` 是要记录的正式奖励分量名；每个结果取出单环境标量并转为 float。
    return {
        name: (
            float((getattr(rewards, name) * auxiliary_reward_scales(unwrapped, name))[0].item())
            if getattr(unwrapped.cfg.rewards, name, None) is not None
            else 0.0
        )
        for name in RECORDED_REWARD_COMPONENTS
    }


@dataclass
class PendingTerminal:
    """保存 reset 之前的 terminal transition，避免环境 reset 后丢失终止帧。"""

    # reset 前控制器已经执行的动作记录；由 StateCaptureAdapter 生成。
    action: Any
    # Isaac reward_buf 中与该终局动作对应的总奖励标量。
    reward: float
    # reset 前最后一帧显式 AUBO/相机 observation。
    observation: Any
    # 正式环境给出的 success/timeout/safety_failure 终局结果。
    outcome: EpisodeOutcome
    # 与总奖励对应的正式 reward 分量，便于离线审计和预测损失训练。
    reward_components: dict[str, float]


# ------------------------------ 运行日志 ------------------------------
# 日志格式集中在采集流程之外，主流程只调用带有明确语义的日志函数。
def _print_log(message: str, *, error: bool = False) -> None:
    """输出一条采集日志；``error`` 仅用于需要进入标准错误流的中止信息。"""

    print(message, file=sys.stderr if error else sys.stdout, flush=True)


def _log_dataset_cameras(camera_specs: tuple[Any, ...]) -> None:
    """打印本次数据集使用的相机配置。"""

    camera_summary = ",".join(
        f"{spec.name}@{spec.position_e}:{spec.resolution[0]}x{spec.resolution[1]}" for spec in camera_specs
    )
    # 输出相机配置摘要，供启动阶段核对视觉输入。
    _print_log(f"DATASET_CAMERAS={camera_summary}")


def _log_collection_plan(start_index: int, requested: int, existing: int) -> None:
    """打印断点续采的起始编号和本次采集计划。"""

    # 输出断点续采计划，供批次启动时核对编号范围。
    _print_log(f"COLLECTION_PLAN start_index={start_index} requested={requested} existing={existing}")


def _log_episode_start(collection_index: int, total: int) -> None:
    """打印当前 episode 的批次进度。"""

    # 输出当前 episode 进度，供长时间采集时确认循环状态。
    _print_log(f"episode {collection_index + 1}/{total} 开始收集")


def _log_episode_collected(episode_id: str, steps: int, termination: str) -> None:
    """打印一条 episode 写盘完成信息。"""

    # 输出单条 episode 结果，供采集完成后核对长度和终止原因。
    _print_log(f"COLLECTED episode={episode_id} steps={steps} termination={termination}")


def _log_sequence_stage_transition(
    episode_id: str,
    control_step: int,
    previous_stage: SequenceStage,
    current_stage: SequenceStage,
) -> None:
    """打印状态机进入新阶段的日志，避免按控制步重复输出。"""

    _print_log(
        "SEQUENCE_STAGE_ENTER "
        f"episode={episode_id} control_step={control_step} "
        f"from={previous_stage.name}({previous_stage.value}) "
        f"stage={current_stage.name}({current_stage.value})"
    )


def _log_collection_complete(episodes: int, output_dir: Path) -> None:
    """打印整批数据集写盘完成信息。"""

    # 输出整批完成摘要，供收尾阶段确认数据集位置和数量。
    _print_log(f"COLLECTION_COMPLETE episodes={episodes} output={output_dir}")


def _log_collection_aborted(completed: int, requested: int, error: BaseException) -> None:
    """打印采集中止摘要和完整异常栈。"""

    # 输出简洁中止摘要，供错误处理阶段识别已完成数量和异常原因。
    _print_log(
        f"COLLECTION_ABORTED completed={completed} requested={requested} error={type(error).__name__}: {error}",
        error=True,
    )
    # CUDA/PyTorch 异常经常只在后续同步点暴露；不能只保留一行摘要，
    # 否则无法判断真正触发错误的采集步骤和调用链。
    traceback.print_exception(
        type(error),
        error,
        error.__traceback__,
        file=sys.stderr,
    )
    sys.stderr.flush()


def main() -> int:
    """按 episode 生成 TCP 位移动作，驱动正式 Isaac IK 环境并写出数据集。

    主流程可以按四段理解：校验 CLI 并创建环境、挂接 reset 前终局捕获、
    对每条 episode 做 reset/规划/逐步采集、最后关闭环境和 Kit。局部变量
    尽量保持“环境对象 / 当前 episode 状态 / 当前控制步状态 / 写盘对象”
    四类职责分离，便于沿着一条 transition 追踪数据来源。
    """

    # ------------------------------ 参数校验与环境创建 ------------------------------
    # 先检查轨迹规划和相机参数，再创建正式 CurriculumRL IK 环境。
    sequence_config, place_state_name = _validate_cli_args(args_cli)

    # ``args_cli.seed`` 是整批采集的基准种子；每条 episode 会在此基础上加索引。
    torch.manual_seed(args_cli.seed)

    # ``env_cfg`` 是正式单环境 IK 配置；``camera_specs`` 描述场景中三路相机及其输出模态。
    env_cfg, camera_specs = _make_env_cfg(
        args_cli.camera_width,
        args_cli.camera_height,
        control_hz=args_cli.control_hz,
        target_state_index=args_cli.target_state_index,
        downward_yaw_offset_rad=math.radians(args_cli.downward_yaw_offset_deg),
        isolate_collection_sequence=True,
        episode_length_s=(args_cli.max_steps + 1) / args_cli.control_hz,
    )
    # 将用户指定的 Isaac device 写入配置，避免环境默认选到另一张设备。
    env_cfg.sim.device = args_cli.device
    # 非 headless 采集时固定主视窗到项目 IK 调试视角；这只影响 GUI viewport，
    # 不会改变三路 CameraSensor 的外参、HDF5 图像内容或 IK 控制目标。
    env_cfg.viewer.eye = TEST_RENDER_CFG.viewport_camera_eye
    env_cfg.viewer.lookat = TEST_RENDER_CFG.viewport_camera_target
    # ``env`` 是 Gymnasium 包装后的正式任务环境，负责 reset/step 和终止语义。
    env = gym.make("CurriculumRL-TcpDocking-IK-v0", cfg=env_cfg)
    # ``unwrapped`` 是去掉 Gym wrapper 后的 CurriculumRL 环境，便于访问 scene、reward_buf 等内部状态。
    unwrapped = env.unwrapped
    robot = unwrapped.scene[SCENE_ENTITY_AUBO]
    gripper_controller = DatasetGripperController(
        robot,
        open_position=args_cli.gripper_open_position,
        closed_position=args_cli.gripper_closed_position,
    )
    # ``cameras`` 按相机规格名索引实际 CameraSensor 实体，后续每步从中读取输出。
    cameras = {spec.name: unwrapped.scene[spec.name] for spec in camera_specs}
    # ``camera_adapter`` 检查三路相机是否与同一仿真步同步。
    camera_adapter = CameraCaptureAdapter(camera_specs)
    # ``writer`` 负责将完整 Episode 序列化为 HDF5 manifest/shard，并保存批次元数据。
    writer = CameraExportingTrajectoryWriter(
        args_cli.output_dir,
        DatasetConfig(
            schema_version=ALIGNED_SCHEMA_VERSION,
            episodes_per_shard=2,
            min_free_disk_bytes=0,
            cameras=camera_specs,
            control_frequency_hz=args_cli.control_hz,
            physics_dt_s=SIMULATION_DT_S,
        ),
        dataset_metadata={
            "task_id": "CurriculumRL-DatasetSequence-IK-v4",
            "source_task_id": "CurriculumRL-TcpDocking-IK-v0",
            "robot_model": "AUBO_E5",
            "sequence_mode": args_cli.sequence_mode,
            "success_semantics": "pick_place_sequence_complete",
            "target_state_index": args_cli.target_state_index,
            "target_state_name": TARGET_STATES[args_cli.target_state_index].name,
            "place_state_index": args_cli.place_state_index,
            "place_state_name": place_state_name,
            "trajectory_mode": args_cli.trajectory_mode,
            "run_tag": args_cli.run_tag,
            "reward_target_field": "rewards",
            "reward_semantics": "not_defined; original environment values retained for diagnostics only",
            "reward_prediction_loss": "masked_smooth_l1_requires_reward_valid",
            "reward_component_fields": ",".join(RECORDED_REWARD_COMPONENTS),
        },
    )
    # 打印相机配置，确认本次采集的数据集视觉输入契约。
    _log_dataset_cameras(camera_specs)
    # ``state_adapter`` 把控制器状态快照转为数据层 ActionRecord/CaptureRecord。
    state_adapter = StateCaptureAdapter()
    # ``pending`` 暂存 reset hook 捕获到的终局 transition；通常每次 step 最多一项。
    pending: list[PendingTerminal] = []
    # 当前 episode 的目标位置和姿态，使用机器人根坐标系 B；它们会写入每帧快照。
    active_target_position_b = np.zeros(3, dtype=np.float64)
    active_object_position_b = np.zeros(3, dtype=np.float64)
    active_desired_axis_b = np.zeros(3, dtype=np.float64)
    active_advance_target_position_b = np.zeros(3, dtype=np.float64)
    active_target_quaternion_b = np.zeros(4, dtype=np.float64)
    active_waypoint_position_b = np.zeros(3, dtype=np.float64)
    active_waypoint_index = 0
    active_semantic_stage_id = int(SequenceStage.HIGH_XY_APPROACH)
    active_dataset_action = np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), dtype=np.float32)
    # 当前 episode 的正式 IK 控制器；reset/终局后置 None，防止误读上一条 episode。
    active_controller: Any | None = None

    # ------------------------------ reset 前终止帧捕获 ------------------------------
    # Isaac 环境通常在 step 内部触发 reset；reset 会覆盖终止状态，因此先挂钩 _reset_idx，
    # 把 terminal observation、terminal reward 和 reward components 暂存到 pending。
    def capture_before_reset(env_ids: torch.Tensor) -> None:
        # ``env_ids`` 是本次即将 reset 的并行环境索引；本脚本只采集环境 0。
        # 没有环境 0 的 reset，或当前没有活动控制器时，不应生成终局记录。
        if not bool(torch.any(env_ids == 0)) or active_controller is None:
            return
        # ``metrics`` 是正式终止/成功指标；第二个返回值为此处不需要的奖励对象。
        metrics, _ = compute_step(unwrapped)
        # reset hook 内的 ``sim_step`` 是终局帧所属的仿真步号，必须在 reset 前读取。
        sim_step = int(unwrapped.common_step_counter)
        # 读取终局步的控制器输出和真实关节状态；此处只使用 state 的动作记录。
        _, state = _control_state(
            unwrapped,
            sim_step,
            active_waypoint_position_b,
            dataset_action=active_dataset_action,
            gripper_controller=gripper_controller,
            semantic_stage_id=active_semantic_stage_id,
            waypoint_index=active_waypoint_index,
        )
        # 读取终局时刻三路相机，并用同一个仿真步/时间戳建立同步关系。
        camera_frames, camera_payload = _camera_observation(
            cameras, camera_specs, sim_step, sim_step * unwrapped.step_dt
        )
        # 校验相机帧确实属于终局 ``sim_step``，然后生成 reset 前的最后 observation。
        camera_adapter.capture(camera_frames, expected_sim_step=sim_step)
        # ``terminal_snapshot`` 必须在原始 reset 执行前构造，否则机器人状态会被新 episode 覆盖。
        terminal_snapshot = _snapshot(
            unwrapped,
            active_controller,
            active_target_position_b,
            active_target_quaternion_b,
            camera_payload,
            gripper_controller,
            active_object_position_b,
            active_desired_axis_b,
            active_advance_target_position_b,
            semantic_stage_id=active_semantic_stage_id,
            waypoint_index=active_waypoint_index,
        )
        # ``outcome`` 只消费正式环境的 success/timeout/safety_failure，不在采集层复制阈值逻辑。
        collection_safety_failure = metrics.outside_workspace | metrics.illegal_contact | metrics.joint_hard_limit
        outcome = EpisodeOutcome(
            success=False,
            timeout=bool(unwrapped.reset_time_outs[0].item()),
            safety_failure=bool(collection_safety_failure[0].item()),
        )
        # ``pending`` 中的这一项等待外层 step 处理，再由 EpisodeSupervisor 提交 Episode。
        pending.append(
            PendingTerminal(
                action=state_adapter.capture(state).action,
                reward=float(unwrapped.reward_buf[0].item()),
                observation=terminal_snapshot,
                outcome=outcome,
                reward_components=_reward_components(unwrapped),
            )
        )

    # 保存正式 reset 实现；包装函数只增加“reset 前捕获”，随后仍调用原逻辑。
    original_reset_idx = unwrapped._reset_idx

    def reset_idx_with_capture(env_ids: torch.Tensor) -> None:
        # 先抓终局，再执行环境原有 reset，保证 terminal observation 不被覆盖。
        capture_before_reset(env_ids)
        original_reset_idx(env_ids)

    # 将包装后的 reset 函数挂回正式环境；这不会改变 reset 的原有行为顺序。
    unwrapped._reset_idx = reset_idx_with_capture
    try:
        # ------------------------------ 收集逻辑 ------------------------------
        # 每条 episode 使用独立 seed，生成一条确定的 staged/direct TCP 轨迹，并写入 HDF5。
        # 初次 reset 负责初始化 Isaac 场景；返回的 ``observation`` 是 Gym 观测，
        # 采集脚本不直接解析它，而是从正式环境对象构造显式状态快照。
        observation, _ = env.reset(seed=args_cli.seed)
        # 初次 reset 可能触发 reset hook，但它不属于待写入 episode，清除其暂存项。
        pending.clear()
        # CameraSensor 在 reset 后重新开始自己的帧序列；采集适配器也必须清除上一条
        # episode 的连续性状态，因为 Isaac 的全局 common_step_counter 不会随 reset 递增。
        camera_adapter.reset_episode()
        # 同一 output-dir 和 run-tag/seed 重启时，从已有 episode 编号之后继续。
        # 每次命令都生成完整的 ``--episodes`` 条新记录，避免固定 ep0000...ep0004 导致只新增一条。
        episode_prefix = f"{args_cli.run_tag}-seed{args_cli.seed}-ep"
        first_episode_index = next_episode_index(writer.episode_ids, episode_prefix)
        # 打印采集计划，确认断点续采编号和本次请求数量。
        _log_collection_plan(first_episode_index, args_cli.episodes, len(writer.episode_ids))

        for collection_index in range(args_cli.episodes):
            # ------------------------------ episode 循环 ------------------------------

            episode_index = first_episode_index + collection_index
            # 使用 f-string 将整数索引安全转换为文本；显示 1-based 进度便于确认多 episode 循环仍在继续。
            # 打印当前 episode 进度，便于确认批次循环正在推进。
            _log_episode_start(collection_index, args_cli.episodes)
            # ``episode_id`` 是可复现且可用于断点续采的稳定标识。
            episode_id = f"{episode_prefix}{episode_index:04d}"
            # ``episode_seed`` 为每条 episode 派生独立随机流，保证轨迹规划可复现且彼此不同。
            episode_seed = args_cli.seed + episode_index
            # reset 后重新获取本条 episode 的环境初始状态和正式 IK 控制器。
            active_controller = None
            observation, _ = env.reset(seed=episode_seed)
            # 丢弃 reset 过程中不属于当前 episode 的历史终局暂存项。
            pending.clear()
            # reset 后允许新的 episode 使用相同的全局 sim_step，但仍保持本 episode 内的陈旧帧/重复帧检查。
            camera_adapter.reset_episode()
            # ``sim_step`` 是 episode 首帧的 Isaac 全局仿真步号。
            sim_step = int(unwrapped.common_step_counter)
            # ``active_controller`` 是本条 episode 后续 step 和 snapshot 共用的正式 IK 控制器。
            # reset 后尚未执行 action，IK diagnostics 仍为 None；首帧只需控制器。
            active_controller = _ik_controller(unwrapped)
            # 控制器当前 TCP 的位置/四元数位于机器人根坐标系 B；第三个返回值不用于规划。
            current_position_b, _, _ = active_controller.current_tcp_pose_b()
            # ``current_position_b`` 是 NumPy float64 三维起点，供无梯度轨迹规划使用。
            current_position_b = _numpy(current_position_b[0]).astype(np.float64)
            # ``active_target_position_b`` 是目标状态转换到同一根坐标系 B 后的位置。
            active_target_position_b = _target_position_b(
                unwrapped, args_cli.target_state_index, args_cli.grasp_height_m
            )
            active_object_position_b = _object_position_b(unwrapped, args_cli.target_state_index)
            # 抓取 TCP 终点来自每个 TargetState 的显式 E 系任务真值。
            active_grasp_tcp_position_b = _grasp_tcp_position_b(
                unwrapped, args_cli.target_state_index, args_cli.grasp_height_m
            )
            place_state = TARGET_STATES[args_cli.place_state_index]
            place_tcp_position_e = place_state.grasp_tcp_position_e
            place_tcp_direction_w = _target_tcp_direction_w(args_cli.place_state_index)
            lift_position_b, place_high_position_b, place_position_b = _pick_place_positions_b(
                unwrapped,
                args_cli.target_state_index,
                args_cli.place_state_index,
                float(args_cli.lift_clearance_m),
                args_cli.grasp_height_m,
                routing_from_args(args_cli),
            )
            # 固定目标姿态与 action term 的 Flange-down 几何约束一致，而不是沿用 reset 姿态。
            robot = unwrapped.scene[SCENE_ENTITY_AUBO]
            _, active_target_quaternion_b = flange_down_tcp_target_pose(
                active_target_position_b,
                root_quaternion_wxyz=_numpy(robot.data.root_quat_w[0]).astype(np.float64),
                downward_yaw_offset_rad=math.radians(args_cli.downward_yaw_offset_deg),
            )
            sequence = DatasetSequence(
                preposition_b=active_target_position_b,
                grasp_tcp_position_b=active_grasp_tcp_position_b,
                grasp_tcp_direction_w=_target_tcp_direction_w(args_cli.target_state_index),
                lift_position_b=lift_position_b,
                place_high_position_b=place_high_position_b,
                place_position_b=place_position_b,
                place_tcp_direction_w=place_tcp_direction_w,
                config=sequence_config,
                place_state_index=args_cli.place_state_index,
                source_state_index=args_cli.target_state_index,
                place_preposition_b=_environment_position_b(unwrapped, place_state.preposition_e),
                place_object_pose_b=measured_pose_w_to_b(
                    _measured_base_pose_w(robot),
                    [
                        *(_numpy(unwrapped.scene.env_origins[0]) + np.asarray(place_state.position_e)),
                        *place_state.rotation_wxyz,
                    ],
                ),
                state4_routing=routing_from_args(args_cli),
                source2_routing=source2_routing_from_args(args_cli),
                tool_forward_axis=TOOL_FORWARD_AXIS,
                tcp_heading_axis=TCP_DIRECTION_AXIS_T,
                initial_tcp_heading_rad=planar_heading_from_quaternion(
                    active_target_quaternion_b, TCP_DIRECTION_AXIS_T
                ),
            )
            active_desired_axis_b = np.asarray(sequence.desired_axis_b, dtype=np.float64)
            if sequence.source2_routing is not None:
                active_target_position_b = np.asarray(sequence.preposition_b, dtype=np.float64)
                _, active_target_quaternion_b = flange_down_tcp_target_pose(
                    active_target_position_b,
                    root_quaternion_wxyz=_numpy(robot.data.root_quat_w[0]).astype(np.float64),
                    downward_yaw_offset_rad=math.radians(args_cli.downward_yaw_offset_deg),
                )
            lift_position_b = np.asarray(sequence.lift_position_b, dtype=np.float64)
            place_high_position_b = np.asarray(sequence.place_high_position_b, dtype=np.float64)
            active_advance_target_position_b = np.asarray(sequence.advance_target_b, dtype=np.float64)
            episode_goal_position_b = np.asarray(place_position_b, dtype=np.float64)
            active_semantic_stage_id = int(SequenceStage.HIGH_XY_APPROACH)
            active_dataset_action = np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), dtype=np.float32)
            active_controller.flange_downward_enabled = True
            gripper_controller.apply(1.0)

            # ------------------------------ 轨迹规划与元数据 ------------------------------
            # 规划阶段只决定期望 TCP 路径；真正的 action 仍由下面的控制循环逐步送入正式 IK。
            # ``rng`` 是本条 episode 专属的 NumPy 随机数生成器，不影响其他 episode 的随机序列。
            rng = np.random.default_rng(episode_seed)

            # ``planning_waypoint_count`` 是粗规划点数；若未显式指定，则在配置范围内采样。
            planning_waypoint_count = (
                args_cli.planning_waypoints
                if args_cli.planning_waypoints is not None
                else int(rng.integers(args_cli.coarse_min_waypoints, args_cli.coarse_max_waypoints + 1))
            )
            # ``xy_fraction`` 决定 staged 轨迹中横向接近阶段所占比例。
            xy_fraction = float(rng.uniform(args_cli.xy_fraction_min, args_cli.xy_fraction_max))
            # ``plan`` 同时包含粗规划点、控制频率展开后的 waypoint 以及插值倍数。
            plan = make_plan(
                current_position_b,
                active_target_position_b,
                planning_waypoint_count,
                planning_hz=args_cli.planning_hz,
                control_hz=args_cli.control_hz,
                trajectory_mode=args_cli.trajectory_mode,
                high_clearance_z=args_cli.high_clearance_z,
                xy_fraction=xy_fraction,
                base_clockwise_deg=args_cli.base_clockwise_deg,
            )
            # ``interpolation_substeps`` 是每个粗规划点展开成控制点的整数倍数。
            interpolation_substeps = plan.interpolation_substeps
            # ``control_waypoints`` 是实际逐控制步跟踪的 (phase, position_B) 序列。
            control_waypoints = plan.control_waypoints
            # ``episode_metadata`` 保存复现轨迹和解释数据所需的配置，不参与控制决策。
            episode_metadata = {
                "target_object_id": SCENE_ENTITY_TARGET,
                "tcp_frame_name": "controller_TCP",
                "selected_grasp_id": TARGET_STATES[args_cli.target_state_index].name,
                "grasp_definition_source": "task_config_position_and_flange_down_heading_solver",
                "grasp_reference_frame": "object",
                "object_reference_frame": "target_config_root",
                "action_frame": "robot_base",
                "action_position_scale_m": TCP_POSITION_SCALE_M,
                "action_rotation_scale_rad": TCP_HEADING_SCALE_RAD,
                "action_applied_source": "FK_of_final_transmitted_joint_target_minus_policy_start_FK",
                "goal_position_tolerance_m": sequence_config.position_tolerance_m,
                "goal_rotation_tolerance_rad": sequence_config.normal_angle_tolerance_rad,
                "goal_tolerance_source": "sequence_tcp_tolerances; object_success_not_evaluated",
                "gripper_force_semantics": (
                    "physx_projected_joint_force_along_prismatic_axes; not_fingertip_contact_force"
                ),

                "seed": episode_seed,
                "target_state_index": args_cli.target_state_index,
                "target_state_name": TARGET_STATES[args_cli.target_state_index].name,
                "trajectory_mode": args_cli.trajectory_mode,
                "trajectory_phase_labels": "direct,high_xy_approach,base_swing,base_arc,base_high_xy,"
                "high_pose_alignment,vertical_descent,target_settle",
                "planning_waypoints": planning_waypoint_count,
                "control_waypoints": len(control_waypoints),
                "interpolation_substeps": interpolation_substeps,
                "planning_hz": args_cli.planning_hz,
                "control_hz": args_cli.control_hz,
                "waypoint_progression": "actual_tcp_arrival_gated",
                "waypoint_position_tolerance_m": args_cli.waypoint_position_tolerance_m,
                "ik_constraint_mode": "flange_down_6d",
                "flange_downward_axis_local": "-X",
                "flange_downward_axis_world": "-Z",
                "downward_yaw_offset_deg": args_cli.downward_yaw_offset_deg,
                "ik_seed_strategy": "current_joint_position_at_policy_step",
                "action_semantics": "normalized_xyz_plus_absolute_heading_plus_gripper_in_legacy8_layout",
                "action_rotation_semantics": (
                    "six-dimensional TCP increment is submitted to the existing IK adapter; "
                    "Flange-down mode zeros incremental rotation and resolves orientation from tcp_heading_target"
                ),
                "tcp_heading_semantics": (
                    "normalized XOY heading terminal value; "
                    "ALIGN_NORMAL and ALIGN_PLACE_DIRECTION use bounded heading steps; "
                    "Flange normal remains downward in every sequence stage"
                ),
                "tcp_heading_step_rad": sequence_config.heading_step_rad,
                "tcp_heading_axis_local": TCP_DIRECTION_AXIS_T,
                "tcp_heading_physical_axis": "ClawTool_C/ClawTool_C:+Y",
                "tcp_heading_transfer_fraction": PLACE_TRANSFER_HEADING_FRACTION,
                "xy_fraction": xy_fraction,
                "initial_position_b": tuple(float(value) for value in current_position_b),
                "target_position_b": tuple(float(value) for value in active_target_position_b),
                "object_position_b": tuple(float(value) for value in active_object_position_b),
                "object_rotation_wxyz": TARGET_STATES[args_cli.target_state_index].rotation_wxyz,
                "desired_approach_axis_b": tuple(float(value) for value in active_desired_axis_b),
                "advance_target_position_b": tuple(float(value) for value in active_advance_target_position_b),
                "grasp_tcp_direction_w": tuple(float(value) for value in sequence.grasp_tcp_direction_w),
                "grasp_tcp_position_e": TARGET_STATES[args_cli.target_state_index].grasp_tcp_position_e,
                "grasp_tcp_position_b": tuple(float(value) for value in active_grasp_tcp_position_b),
                "place_tcp_position_e": place_tcp_position_e,
                "place_tcp_direction_w": tuple(float(value) for value in place_tcp_direction_w),
                "place_state_index": args_cli.place_state_index,
                "place_state_name": place_state.name,
                "place_object_position_e": place_state.position_e,
                "place_object_rotation_wxyz": place_state.rotation_wxyz,
                "lift_clearance_m": args_cli.lift_clearance_m,
                "grasp_height_m": args_cli.grasp_height_m,
                "lift_position_b": tuple(float(value) for value in lift_position_b),
                "state4_routing_enabled": bool(args_cli.state4_routing),
                "state2_pre_pick_side_m": args_cli.state2_pre_pick_side_m,
                "state4_pick_lift_m": args_cli.state4_pick_lift_m,
                "state4_pick_exit_m": args_cli.state4_pick_exit_m,
                "state4_place_standoff_m": args_cli.state4_place_standoff_m,
                "state4_place_entry_lift_m": args_cli.state4_place_entry_lift_m,
                "state4_place_measured_grasp": args_cli.state4_place_measured_grasp,
                "lift_contact_waypoints_b": sequence.lift_waypoints,
                "place_descent_position_b": sequence.place_descent_position_b,
                "place_high_position_b": tuple(float(value) for value in place_high_position_b),
                "place_position_b": tuple(float(value) for value in place_position_b),
                "sequence_mode": args_cli.sequence_mode,
                "semantic_stage_labels": ",".join(stage.name.lower() for stage in SequenceStage),
                "normal_angle_tolerance_deg": args_cli.normal_angle_tolerance_deg,
                "normal_advance_distance_m": sequence_config.advance_distance_m,
                "gripper_open_position": tuple(float(value) for value in args_cli.gripper_open_position),
                "gripper_closed_position": gripper_controller.closed_position_tuple,
                "reward_target_field": "rewards",
                "reward_component_fields": ",".join(RECORDED_REWARD_COMPONENTS),
            }
            # 采集首帧相机数据；它与初始机器人状态属于同一个 ``sim_step``。
            camera_frames, camera_payload = _camera_observation(
                cameras, camera_specs, sim_step, sim_step * unwrapped.step_dt
            )
            # 校验首帧三路相机同步性后，生成 episode 的 ``o0`` observation。
            camera_adapter.capture(camera_frames, expected_sim_step=sim_step)
            # ``initial_snapshot`` 是 reset 后、首个 action 前的观测帧。
            initial_snapshot = _snapshot(
                unwrapped,
                active_controller,
                active_target_position_b,
                active_target_quaternion_b,
                camera_payload,
                gripper_controller,
                active_object_position_b,
                active_desired_axis_b,
                active_advance_target_position_b,
                semantic_stage_id=active_semantic_stage_id,
                waypoint_index=0,
            )
            # ``builder`` 按 o0-a0-r0-o1-... 顺序组装本条 episode。
            builder = EpisodeBuilder(episode_id, reward_valid=False)
            builder.start(initial_snapshot)
            # ``supervisor`` 确保 reset hook 捕获的正式终局只提交一次。
            supervisor = EpisodeSupervisor()
            # ``completed`` 标记是否已由正式环境终局并写出；False 表示达到采集上限。
            completed = False
            # ``last_snapshot`` 保存最近一个非 reset 观测，必要时作为 collection-cap 终局观测。
            last_snapshot = initial_snapshot

            # ------------------------------ Isaac 控制步与 transition 录制 ------------------------------
            # 每一步根据当前 TCP 与路径 waypoint 生成归一化位移 action，调用 env.step，
            # 然后读取实际控制状态、相机、总 reward 和分项 reward，统一交给 EpisodeBuilder。
            # waypoint 按实际 TCP 到达状态推进；路径终点到达后继续保持，等待正式 dwell/终止判定。
            waypoint_index = 0
            path_complete = False
            waypoint_advances = 0
            max_waypoint_error_m = 0.0

            # max_steps为当前上限规划步数
            for control_step in range(args_cli.max_steps):
                # 重新读取当前 TCP，而不是假设机器人精确到达上一步期望位置。
                current_position_b, current_quaternion_b, _ = active_controller.current_tcp_pose_b()
                # 转成 CPU float64 三维位置，用于计算本步期望位移。
                current_position_b = _numpy(current_position_b[0]).astype(np.float64)
                current_quaternion_b = _numpy(current_quaternion_b[0]).astype(np.float64)
                previous_waypoint_index = waypoint_index
                waypoint_index, reached_final = advance_reached_waypoints(
                    control_waypoints,
                    waypoint_index,
                    current_position_b,
                    args_cli.waypoint_position_tolerance_m,
                )
                waypoint_advances += waypoint_index - previous_waypoint_index
                path_complete |= reached_final
                # ``desired_position_b`` 在未到达时保持不变，不随控制时钟强制推进。
                path_waypoint_label, path_waypoint_position_b = control_waypoints[waypoint_index]
                desired_position_b = np.asarray(path_waypoint_position_b, dtype=np.float64)
                active_waypoint_index = waypoint_index
                waypoint_error_m = float(np.linalg.norm(desired_position_b - current_position_b))
                max_waypoint_error_m = max(max_waypoint_error_m, waypoint_error_m)
                gripper_position, _ = gripper_controller.state()
                observed_object = unwrapped.scene[SCENE_ENTITY_TARGET]
                observed_object_pose_b = measured_pose_w_to_b(
                    _measured_base_pose_w(unwrapped.scene[SCENE_ENTITY_AUBO]),
                    [*_numpy(observed_object.data.root_pos_w[0]), *_numpy(observed_object.data.root_quat_w[0])],
                )
                sequence_command = sequence.command(
                    SequenceObservation(
                        tuple(float(value) for value in current_position_b),
                        tuple(float(value) for value in current_quaternion_b),
                        gripper_position,
                        tuple(
                            float(value)
                            for value in _numpy(unwrapped.scene[SCENE_ENTITY_AUBO].data.root_quat_w[0]).astype(
                                np.float64
                            )
                        ),
                        tuple(float(value) for value in observed_object_pose_b),
                    ),
                    active_waypoint_b=desired_position_b,
                    active_waypoint_label=path_waypoint_label,
                    path_complete=path_complete,
                    closed_gripper_position=gripper_controller.closed_position_tuple,
                    open_gripper_position=gripper_controller.open_position_tuple,
                )
                previous_sequence_stage = SequenceStage(active_semantic_stage_id)
                active_waypoint_position_b = np.asarray(sequence_command.target_position_b, dtype=np.float64)
                active_semantic_stage_id = int(sequence_command.stage)
                if args_cli.log_sequence_stage_transitions and sequence_command.stage is not previous_sequence_stage:
                    _log_sequence_stage_transition(
                        episode_id,
                        control_step,
                        previous_sequence_stage,
                        sequence_command.stage,
                    )
                active_dataset_action = np.asarray(sequence_command.dataset_action, dtype=np.float32)
                # Flange-down uses only heading; remove ignored legacy rotation channels before submission.
                if sequence_command.flange_downward_enabled:
                    active_dataset_action[3:6] = 0.0
                # 新增的 TCP 平面朝向量只影响 Flange-down 目标姿态；现有六维
                # TCP action 仍按原接口送入 IK，不改变 IK 数学求解器。
                commanded_position_b = (
                    current_position_b + np.clip(active_dataset_action[:3], -1.0, 1.0) * TCP_POSITION_SCALE_M
                )
                active_controller.downward_yaw_offset_rad = flange_down_yaw_offset_for_tcp_heading(
                    commanded_position_b,
                    float(active_dataset_action[6] * TCP_HEADING_SCALE_RAD),
                    root_quaternion_wxyz=_numpy(unwrapped.scene[SCENE_ENTITY_AUBO].data.root_quat_w[0]).astype(
                        np.float64
                    ),
                    tcp_heading_axis_local=TCP_DIRECTION_AXIS_T,
                    initial_offset_rad=active_controller.downward_yaw_offset_rad,
                )
                active_controller.flange_downward_enabled = sequence_command.flange_downward_enabled
                gripper_controller.apply(sequence_command.gripper_command)
                raw_action = active_dataset_action[:6]
                # ``action`` 添加单环境 batch 维 (1, 6)，并放到正式环境所在 device 上。
                action = torch.as_tensor(raw_action, dtype=torch.float32, device=unwrapped.device).view(1, 6)
                # 清除上一个控制步的终局暂存，避免把旧 terminal 误认为本步 reset 结果。
                pending.clear()
                # ``next_observation`` 是 env.step 后的 Gym 观测；``reward``、``terminated``、``truncated``
                # 分别表示本步总奖励、正式终止信号和时间截断信号，最后一个返回值此处不使用。
                next_observation, reward, terminated, truncated, _ = env.step(action)
                if pending:
                    # ``terminal`` 是 reset hook 在环境自动 reset 前保存的终局 transition。
                    terminal = pending.pop(0)
                    terminal_position_b = np.asarray(terminal.action.actual_tcp_position_b, dtype=np.float64)
                    previous_waypoint_index = waypoint_index
                    waypoint_index, reached_final = advance_reached_waypoints(
                        control_waypoints,
                        waypoint_index,
                        terminal_position_b,
                        args_cli.waypoint_position_tolerance_m,
                    )
                    waypoint_advances += waypoint_index - previous_waypoint_index
                    path_complete |= reached_final
                    # 将 reset 前终局动作/奖励/观测补进时间轴，形成最后一个控制步。
                    builder.append(
                        terminal.action,
                        terminal.reward,
                        terminal.observation,
                        terminal.reward_components,
                    )
                    # ``episode`` 是经正式 outcome 判定并冻结后的完整 episode。
                    episode = supervisor.submit(builder, terminal.outcome, terminal.observation)
                    episode_metadata.update(
                        executed_control_steps=control_step + 1,
                        final_waypoint_index=waypoint_index,
                        waypoint_advances=waypoint_advances,
                        path_complete=path_complete,
                        max_waypoint_error_m=max_waypoint_error_m,
                        final_target_error_m=float(np.linalg.norm(episode_goal_position_b - terminal_position_b)),
                    )
                    # 终局 episode 立即写盘，避免后续 reset 覆盖任何终局字段。
                    writer.write(episode, metadata=episode_metadata)
                    # 打印 episode 写盘结果，确认正式终止记录已经提交。
                    _log_episode_collected(episode_id, episode.length, episode.termination_reason.value)
                    # 保留 Gym 返回值供下一轮 reset 对齐；终局后不再使用旧控制器。
                    observation = next_observation
                    active_controller = None
                    completed = True
                    break
                # 没有触发 reset 时，读取 env.step 完成后的新控制步号。
                sim_step = int(unwrapped.common_step_counter)
                # ``state`` 是正式控制器对本步 action 的处理结果和实际关节状态。
                _, state = _control_state(
                    unwrapped,
                    sim_step,
                    active_waypoint_position_b,
                    dataset_action=active_dataset_action,
                    gripper_controller=gripper_controller,
                    semantic_stage_id=active_semantic_stage_id,
                    waypoint_index=active_waypoint_index,
                )
                actual_position_after_step = np.asarray(state.actual_tcp_position_b, dtype=np.float64)
                previous_waypoint_index = waypoint_index
                waypoint_index, reached_final = advance_reached_waypoints(
                    control_waypoints,
                    waypoint_index,
                    actual_position_after_step,
                    args_cli.waypoint_position_tolerance_m,
                )
                waypoint_advances += waypoint_index - previous_waypoint_index
                path_complete |= reached_final
                # 读取与新状态同一仿真步的相机帧和原始模态数据。
                camera_frames, camera_payload = _camera_observation(
                    cameras, camera_specs, sim_step, sim_step * unwrapped.step_dt
                )
                # 确保相机帧与控制状态共享 ``sim_step``，再构造下一个 observation。
                camera_adapter.capture(camera_frames, expected_sim_step=sim_step)
                # ``last_snapshot`` 是本次 transition 的 next_observation，phase 从 1 开始递增。
                last_snapshot = _snapshot(
                    unwrapped,
                    active_controller,
                    active_target_position_b,
                    active_target_quaternion_b,
                    camera_payload,
                    gripper_controller,
                    active_object_position_b,
                    active_desired_axis_b,
                    active_advance_target_position_b,
                    semantic_stage_id=active_semantic_stage_id,
                    waypoint_index=active_waypoint_index,
                )
                # 将正式控制器状态、env 总 reward、next observation 和分项 reward 追加到时间轴。
                builder.append(
                    state_adapter.capture(state).action,
                    float(reward[0].item()),
                    last_snapshot,
                    _reward_components(unwrapped),
                )
                if sequence_command.complete:
                    episode = builder.close(TerminationReason.SUCCESS, last_snapshot)
                    episode_metadata.update(
                        executed_control_steps=episode.length,
                        final_waypoint_index=waypoint_index,
                        waypoint_advances=waypoint_advances,
                        path_complete=path_complete,
                        final_semantic_stage=sequence_command.stage.name.lower(),
                        max_waypoint_error_m=max_waypoint_error_m,
                        final_target_error_m=float(
                            np.linalg.norm(episode_goal_position_b - actual_position_after_step)
                        ),
                    )
                    writer.write(episode, metadata=episode_metadata)
                    _log_episode_collected(episode_id, episode.length, episode.termination_reason.value)
                    active_controller = None
                    completed = True
                    break
                # 更新 Gym 观测引用；脚本主要使用显式 snapshot，但保留该值以保持 env 流程完整。
                observation = next_observation
                # 若环境报告终止却没有经过 reset hook，说明终局捕获契约被破坏，应立即报错。
                if bool(terminated[0].item()) or bool(truncated[0].item()):
                    raise RuntimeError("环境已终止但未在 reset hook 捕获 terminal observation")
            if not completed:
                # 达到 ``max_steps`` 或 waypoint 数量上限时，按采集层上限关闭 episode。
                # 该 timeout 是 collection cap，不冒充正式环境的成功/安全终止。
                episode = builder.close(TerminationReason.TIMEOUT, last_snapshot)
                final_tcp_position_b, _, _ = active_controller.current_tcp_pose_b()
                final_tcp_position_b = _numpy(final_tcp_position_b[0]).astype(np.float64)
                episode_metadata.update(
                    executed_control_steps=episode.length,
                    final_waypoint_index=waypoint_index,
                    waypoint_advances=waypoint_advances,
                    path_complete=path_complete,
                    max_waypoint_error_m=max_waypoint_error_m,
                    final_target_error_m=float(np.linalg.norm(episode_goal_position_b - final_tcp_position_b)),
                )
                # 写出未由 reset hook 产生的 collection-cap episode。
                writer.write(episode, metadata=episode_metadata)
                # 打印采集上限结束结果，区分 collection cap 与正式环境终止。
                _log_episode_collected(episode_id, episode.length, "timeout-collection-cap")
                # 关闭本条 episode 的控制器引用，避免下一次 reset hook 误捕获它。
                active_controller = None
                # 为下一条 episode 对齐环境状态并清理可能残留的终局暂存。
                observation, _ = env.reset(seed=episode_seed)
                pending.clear()
        # ------------------------------ 收尾与资源释放 ------------------------------
        # HDF5 writer 已完成 manifest/shard 写入；finally 负责关闭 Isaac 环境和 Kit。
        # 打印整批完成信息，确认 writer 已完成 manifest/shard 写入。
        _log_collection_complete(len(writer.episode_ids), args_cli.output_dir)
        return 0
    except BaseException as exc:
        # 打印中止摘要，保留异常类型和原因；异常栈由外层运行时统一输出。
        _log_collection_aborted(len(writer.episode_ids), args_cli.episodes, exc)
        raise
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
