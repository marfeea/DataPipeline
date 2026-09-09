"""Replay one or more HDF5 AUBO episodes in the same Isaac scene used for collection.

``state`` writes recorded joint states back into the simulated articulation one
frame at a time. ``action`` resets the formal IK environment and executes the
recorded normalized eight-dimensional TCP-heading-plus-gripper actions. The default is GUI mode;
``--headless`` remains useful for smoke testing the replay path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PACKAGE_SOURCE_ROOT = Path(__file__).resolve().parents[1]
# 支持未 editable install 的本地工作树。
sys.path.insert(0, str(PACKAGE_SOURCE_ROOT))

from DataPipe.runtime import ensure_curriculumrl_available  # noqa: E402

ensure_curriculumrl_available()

# 先加载纯 HDF5 读取接口；真正依赖 Isaac/Kit 的模块放到 main() 中延迟导入。
from CurriculumRL.logic.aubo_ik import flange_down_yaw_offset_for_tcp_heading  # noqa: E402

from DataPipe.config import GRIPPER_OPEN_POSITION, TCP_HEADING_SCALE_RAD, TCP_POSITION_SCALE_M  # noqa: E402
from DataPipe.hdf5_writer import HDF5TrajectoryWriter  # noqa: E402
from DataPipe.records import Episode  # noqa: E402

# GUI replay 的目标节拍；``--frame-delay`` 默认使用其倒数。
REPLAY_FRAME_RATE_HZ = 60.0
REPLAY_OBSERVATION_KEYS = (
    "q",
    "dq",
    "gripper_position",
    "gripper_velocity",
    "semantic_stage_id",
    "waypoint_index",
)

# ------------------------------ 命令行参数 ------------------------------
# output-dir 和 episode-id 使用命名参数，避免目录或 ID 的位置发生变化时误解析。
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, required=True)
episode_selection = parser.add_mutually_exclusive_group(required=True)
episode_selection.add_argument(
    "--episode-id",
    action="append",
    help="要重播的 episode ID；可重复传入多个 ID，顺序即播放顺序。",
)
episode_selection.add_argument(
    "--all-episodes",
    action="store_true",
    help="按 manifest 顺序重播 output-dir 中的全部 episode。",
)
parser.add_argument("--mode", choices=("state", "action"), default="state")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--stop", type=int, default=None)
parser.add_argument(
    "--frame-delay",
    type=float,
    default=1.0 / REPLAY_FRAME_RATE_HZ,
    help="GUI frame delay in seconds; default is 1/60 s (60 FPS)",
)
parser.add_argument("--step", action="store_true", help="wait for Enter after each displayed frame")
parser.add_argument("--camera-width", type=int, default=None)
parser.add_argument("--camera-height", type=int, default=None)
parser.add_argument("--kit-cache-dir", type=Path, default=Path("data/isaac_kit_cache"))
# AppLauncher 会补充 --headless、--device、--kit_args 等 Isaac 启动参数。
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# replay 需要加载采集场景中的 CameraSensor；主视窗 viewport 相机另由 TEST_RENDER_CFG 设置。
args_cli.enable_cameras = True
kit_cache_dir = args_cli.kit_cache_dir.resolve()
kit_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["OPTIX_CACHE_PATH"] = str(kit_cache_dir / "optix")
user_config_path = kit_cache_dir / "user.config.json"
args_cli.kit_args = " ".join(
    argument for argument in (args_cli.kit_args, f"--/app/userConfigPath={user_config_path}") if argument
)


def _load_dataset(
    output_dir: Path,
    *,
    episode_ids: tuple[str, ...] | None = None,
) -> tuple[tuple[Episode, ...], dict[str, dict[str, Any]], dict[str, Any]]:
    """读取整个 HDF5 数据集，并返回 episode、demo 属性和数据集属性。

    ``episodes`` 按 writer 的 manifest/shard 顺序返回，保证 ``--all-episodes``
    的播放顺序稳定；``metadata_by_id`` 保存每个 demo 的 seed、目标状态和
    控制频率；``dataset_attrs`` 保存相机标定等整个数据集共享的属性。
    """

    # writer.load_episodes() 会先校验 manifest 中的 SHA-256，再把 T/T+1 数据恢复为 Episode。
    writer = HDF5TrajectoryWriter(output_dir)
    episodes = writer.load_episodes(
        episode_ids=episode_ids,
        observation_keys=REPLAY_OBSERVATION_KEYS,
    )
    # Episode reader 不携带 HDF5 attrs，因此这里再打开对应分片读取 seed、目标和频率等元数据。
    import h5py

    metadata_by_id: dict[str, dict[str, Any]] = {}
    dataset_attrs: dict[str, Any] = {}
    manifest = writer._manifest  # noqa: SLF001 - the writer already verified shard checksums above.
    for shard in manifest.get("shards", ()):
        path = output_dir / str(shard["path"])
        with h5py.File(path, "r") as handle:
            data = handle["data"]
            # 每个 shard 都带同一份数据集级属性；只需读取第一份，避免后续 shard 覆盖结果。
            if not dataset_attrs:
                dataset_attrs = {str(key): data.attrs[key] for key in data.attrs}
            for demo_name in data:
                if not demo_name.startswith("demo_"):
                    continue
                demo = data[demo_name]
                episode_id_value = demo.attrs.get("episode_id", demo_name.removeprefix("demo_"))
                if isinstance(episode_id_value, bytes):
                    episode_id = episode_id_value.decode("utf-8")
                else:
                    episode_id = str(episode_id_value)
                if episode_ids is None or episode_id in episode_ids:
                    metadata_by_id[episode_id] = {str(key): demo.attrs[key] for key in demo.attrs}
    if len(episodes) != len(metadata_by_id):
        raise ValueError(
            f"HDF5 episode 与 demo metadata 数量不一致：episodes={len(episodes)} metadata={len(metadata_by_id)}"
        )
    return episodes, metadata_by_id, dataset_attrs


def _attribute_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _camera_resolution(dataset_attrs: dict[str, Any]) -> tuple[int, int]:
    # replay 默认沿用采集数据的相机分辨率；缺少旧数据属性时回退到项目默认值。
    value = dataset_attrs.get("camera_calibration")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        calibration = json.loads(str(value))
        width, height = calibration[0]["resolution"]
        return int(width), int(height)
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return 160, 120


def _as_float_array(value: Any, *, name: str, size: int) -> list[float]:
    values = [float(item) for item in value]
    if len(values) != size:
        raise ValueError(f"{name} must have {size} values, got {len(values)}")
    return values


def _format_float_array(value: Any, *, name: str, size: int) -> str:
    values = _as_float_array(value, name=name, size=size)
    return "[" + ", ".join(f"{item:.4f}" for item in values) + "]"


def _print_frame(
    *,
    index: int,
    total: int,
    observation: dict[str, Any],
    action: Any = None,
    reward: float | None = None,
    runtime_done: tuple[bool, bool] | None = None,
) -> None:
    # 这里打印的是当前重播帧的可审计信息，便于同时核对记录状态和 Isaac 实际状态。
    q = observation.get("q", (float("nan"),) * 6)
    details = [
        f"FRAME {index}/{total - 1}",
        f"stage={observation.get('semantic_stage_id', 0)}",
        f"waypoint={observation.get('waypoint_index', 0)}",
        f"q={_format_float_array(q, name='q', size=6)}",
    ]
    if action is not None:
        details.append(f"action={_format_float_array(action, name='action', size=8)}")
    if reward is not None:
        details.append(f"reward={reward:.4f}")
    if runtime_done is not None:
        details.append(f"runtime_terminated={runtime_done[0]} runtime_truncated={runtime_done[1]}")
    print(" ".join(details), flush=True)


def _pause() -> None:
    # --step 用人工回车控制；普通模式用 frame-delay 控制显示节奏。
    if args_cli.step:
        input("按 Enter 播放下一帧...")
    elif args_cli.frame_delay > 0.0:
        time.sleep(args_cli.frame_delay)


def main() -> int:
    # ------------------------------ 数据准备 ------------------------------
    # 先在启动 Kit 前读取并校验数据，文件不存在或 checksum 错误时不会启动仿真。
    if args_cli.start < 0 or args_cli.frame_delay < 0.0:
        raise ValueError("start and frame-delay must be non-negative")
    # ``episodes`` 是 reader 从所有 HDF5 shard 恢复出的完整 episode 序列。
    requested_episode_ids = None if args_cli.all_episodes else tuple(args_cli.episode_id or ())
    episodes, metadata_by_id, dataset_attrs = _load_dataset(
        args_cli.output_dir,
        episode_ids=requested_episode_ids,
    )
    episodes_by_id = {episode.episode_id: episode for episode in episodes}
    if args_cli.all_episodes:
        # ``replay_items`` 按 manifest/shard 顺序播放全部记录，而不是只取第一项。
        replay_items = tuple((episode, metadata_by_id[episode.episode_id]) for episode in episodes)
    else:
        # 重复的 --episode-id 按命令行顺序播放，便于选择一个子集或从中途继续。
        requested_ids = tuple(args_cli.episode_id or ())
        missing_ids = tuple(episode_id for episode_id in requested_ids if episode_id not in episodes_by_id)
        if missing_ids:
            raise KeyError(f"episode-id not found in manifest: {', '.join(missing_ids)}")
        replay_items = tuple((episodes_by_id[episode_id], metadata_by_id[episode_id]) for episode_id in requested_ids)
    if not replay_items:
        raise ValueError("没有可重播的 episode")

    # 当前单个 Isaac 环境在一次运行中固定目标状态和控制频率；多条记录必须共享这两个配置。
    base_episode, base_metadata = replay_items[0]
    target_state_index = _attribute_int(base_metadata.get("target_state_index"), 0)
    control_hz = float(base_metadata.get("control_hz", 12.0))
    ik_constraint_mode = str(base_metadata.get("ik_constraint_mode", "runtime_curriculum"))
    downward_yaw_offset_deg = float(base_metadata.get("downward_yaw_offset_deg", 0.0))
    for episode, metadata in replay_items:
        episode_target_state_index = _attribute_int(metadata.get("target_state_index"), 0)
        episode_control_hz = float(metadata.get("control_hz", control_hz))
        episode_ik_constraint_mode = str(metadata.get("ik_constraint_mode", "runtime_curriculum"))
        episode_downward_yaw_offset_deg = float(metadata.get("downward_yaw_offset_deg", 0.0))
        if (
            episode_target_state_index != target_state_index
            or episode_control_hz != control_hz
            or episode_ik_constraint_mode != ik_constraint_mode
            or episode_downward_yaw_offset_deg != downward_yaw_offset_deg
        ):
            raise ValueError(
                "一次 replay 的 episode 必须共享 target/control/IK constraint 配置："
                f"{base_episode.episode_id}({target_state_index}, {control_hz}, {ik_constraint_mode}, "
                f"{downward_yaw_offset_deg}) vs {episode.episode_id}({episode_target_state_index}, "
                f"{episode_control_hz}, {episode_ik_constraint_mode}, {episode_downward_yaw_offset_deg})"
            )
    recorded_width, recorded_height = _camera_resolution(dataset_attrs)
    camera_width = recorded_width if args_cli.camera_width is None else args_cli.camera_width
    camera_height = recorded_height if args_cli.camera_height is None else args_cli.camera_height

    # ------------------------------ Isaac 启动 ------------------------------
    # 不传 --headless 时启动 GUI；传入 --headless 可只验证重播执行链路。
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        import CurriculumRL.tasks  # noqa: F401
        import gymnasium as gym
        import torch
        from CurriculumRL.tasks.tcp_docking.aubo_ik_action import AuboIKActionTerm

        from configs.assets import SCENE_ENTITY_AUBO
        from configs.render import TEST_RENDER_CFG
        from DataPipe.isaac_gripper import DatasetGripperController
        from DataPipe.isaac_scene import make_dataset_ik_env_cfg

        # 采集和重播使用同一个动态场景配置，避免机器人、目标物和 CameraSensor 发生漂移。
        sequence_mode = str(base_metadata.get("sequence_mode", "position-only"))
        env_cfg, _ = make_dataset_ik_env_cfg(
            camera_width,
            camera_height,
            control_hz=control_hz,
            target_state_index=target_state_index,
            downward_yaw_offset_rad=math.radians(downward_yaw_offset_deg),
            flange_downward_enabled=ik_constraint_mode == "flange_down_6d",
            isolate_collection_sequence=sequence_mode in {"grasp", "pick-place"},
            episode_length_s=(max(len(episode.steps) for episode, _ in replay_items) + 1) / control_hz,
        )
        # 配置 Isaac viewer 的初始位置；这不是 CameraSensor 的位置配置。
        env_cfg.viewer.eye = TEST_RENDER_CFG.viewport_camera_eye
        env_cfg.viewer.lookat = TEST_RENDER_CFG.viewport_camera_target
        env_cfg.sim.device = args_cli.device
        env = gym.make("CurriculumRL-TcpDocking-IK-v0", cfg=env_cfg)
        try:
            unwrapped = env.unwrapped
            robot = unwrapped.scene[SCENE_ENTITY_AUBO]
            ik_term = unwrapped.action_manager.get_term("ik")
            if not isinstance(ik_term, AuboIKActionTerm):
                raise RuntimeError("formal replay requires AuboIKActionTerm")
            controller = ik_term.controller
            joint_ids = controller.joint_ids
            gripper_controller = DatasetGripperController(
                robot,
                open_position=_as_float_array(
                    base_metadata.get("gripper_open_position", GRIPPER_OPEN_POSITION),
                    name="gripper_open_position",
                    size=2,
                ),
                closed_position=_as_float_array(
                    base_metadata.get("gripper_closed_position", GRIPPER_OPEN_POSITION),
                    name="gripper_closed_position",
                    size=2,
                ),
            )

            # ------------------------------ 运行时状态 ------------------------------
            # 用 Isaac 当前关节张量覆盖记录值，打印结果反映实际仿真状态而不是 HDF5 中的参考状态。
            def runtime_observation(reference: dict[str, Any]) -> dict[str, Any]:
                value = dict(reference)
                value["q"] = robot.data.joint_pos[0, joint_ids].detach().cpu().tolist()
                value["gripper_position"] = list(gripper_controller.state()[0])
                return value

            # state 模式的核心：直接写回关节状态，不执行记录动作，也不重新计算任务轨迹。
            def write_recorded_state(observation: dict[str, Any]) -> None:
                q = torch.as_tensor(
                    _as_float_array(observation["q"], name="q", size=6),
                    dtype=robot.data.joint_pos.dtype,
                    device=robot.data.joint_pos.device,
                ).view(1, 6)
                dq = torch.as_tensor(
                    _as_float_array(observation["dq"], name="dq", size=6),
                    dtype=robot.data.joint_vel.dtype,
                    device=robot.data.joint_vel.device,
                ).view(1, 6)
                joint_pos = robot.data.joint_pos.clone()
                joint_vel = robot.data.joint_vel.clone()
                joint_pos[:, joint_ids] = q
                joint_vel[:, joint_ids] = dq
                gripper_q = torch.as_tensor(
                    _as_float_array(observation["gripper_position"], name="gripper_position", size=2),
                    dtype=robot.data.joint_pos.dtype,
                    device=robot.data.joint_pos.device,
                ).view(1, 2)
                gripper_dq = torch.as_tensor(
                    _as_float_array(observation["gripper_velocity"], name="gripper_velocity", size=2),
                    dtype=robot.data.joint_vel.dtype,
                    device=robot.data.joint_vel.device,
                ).view(1, 2)
                joint_pos[:, gripper_controller.joint_ids] = gripper_q
                joint_vel[:, gripper_controller.joint_ids] = gripper_dq
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.set_joint_position_target(q, joint_ids=joint_ids)
                unwrapped.scene.write_data_to_sim()
                unwrapped.sim.step(render=True)
                unwrapped.scene.update(unwrapped.step_dt)

            # ------------------------------ 多 episode 重播 ------------------------------
            # 只创建一次 Isaac 环境；每条 episode 开始时重新 reset，结束后继续遍历下一条记录。
            print(
                f"REPLAY_START episodes={len(replay_items)} mode={args_cli.mode} frame_rate={REPLAY_FRAME_RATE_HZ:.1f}",
                flush=True,
            )
            for replay_index, (episode, episode_metadata) in enumerate(replay_items, start=1):
                # 旧数据第 7 维表示 IK TCP +Y；新数据显式保存物理朝向轴的 T 系表达。
                heading_axis = _as_float_array(
                    episode_metadata.get("tcp_heading_axis_local", (0.0, 1.0, 0.0)),
                    name="tcp_heading_axis_local", size=3,
                )
                # 每条 episode 的 seed、帧数边界独立读取，避免沿用上一条记录的元数据。
                seed = _attribute_int(episode_metadata.get("seed"), 0)
                total_frames = len(episode.observations)
                start = min(args_cli.start, total_frames - 1)
                stop = total_frames if args_cli.stop is None else min(max(args_cli.stop, start + 1), total_frames)

                # action 模式必须从该 episode 的初始状态 reset；state 模式也 reset 后再写首帧。
                env.reset(seed=seed)
                # reset 可能重建 viewer 状态，因此每条 episode 开始时都重新设置 viewport 相机。
                unwrapped.sim.set_camera_view(
                    TEST_RENDER_CFG.viewport_camera_eye,
                    TEST_RENDER_CFG.viewport_camera_target,
                )
                print(
                    f"REPLAY_EPISODE_START index={replay_index}/{len(replay_items)} "
                    f"episode={episode.episode_id} seed={seed} frames={total_frames}",
                    flush=True,
                )

                if args_cli.mode == "state":
                    # state 模式用于检查记录的每一帧姿态是否能在同一场景中恢复到视窗。
                    for index in range(start, stop):
                        observation = episode.observations[index].value
                        write_recorded_state(observation)
                        _print_frame(index=index, total=total_frames, observation=runtime_observation(observation))
                        _pause()
                else:
                    # action 模式从本条 episode reset 后逐步执行记录动作，并输出正式终止/误差信息。
                    action_stop = min(stop - 1, len(episode.steps))
                    if start > 0:
                        write_recorded_state(episode.observations[start].value)
                    for index in range(start, action_stop):
                        observation = episode.observations[index].value
                        recorded_action = _as_float_array(episode.steps[index].action.action, name="action", size=8)
                        if ik_constraint_mode == "flange_down_6d":
                            # 新 schema 全程保持 Flange-down；动作第 7 维只用于生成其平面 yaw 目标。
                            current_position_b, _, _ = controller.current_tcp_pose_b()
                            current_position_b = current_position_b[0].detach().cpu().numpy()
                            commanded_position_b = (
                                current_position_b
                                + np.asarray(recorded_action[:3], dtype=np.float64) * TCP_POSITION_SCALE_M
                            )
                            root_quaternion_wxyz = (
                                controller.robot.data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
                            )
                            controller.downward_yaw_offset_rad = flange_down_yaw_offset_for_tcp_heading(
                                commanded_position_b,
                                recorded_action[6] * TCP_HEADING_SCALE_RAD,
                                root_quaternion_wxyz=root_quaternion_wxyz,
                                tcp_heading_axis_local=heading_axis,
                                initial_offset_rad=controller.downward_yaw_offset_rad,
                            )
                            controller.flange_downward_enabled = True
                        gripper_controller.apply(recorded_action[-1])
                        action = torch.as_tensor(
                            recorded_action[:6],
                            dtype=torch.float32,
                            device=unwrapped.device,
                        ).view(1, 6)
                        _, reward, terminated, truncated, _ = env.step(action)
                        runtime_done = (bool(terminated[0].item()), bool(truncated[0].item()))
                        next_observation = runtime_observation(episode.observations[index + 1].value)
                        _print_frame(
                            index=index + 1,
                            total=total_frames,
                            observation=next_observation,
                            action=episode.steps[index].action.action,
                            reward=float(reward[0].item()),
                            runtime_done=runtime_done if any(runtime_done) else None,
                        )
                        _pause()
                        if any(runtime_done) and index + 1 < len(episode.steps):
                            print(
                                f"REPLAY_EARLY_TERMINAL episode={episode.episode_id} recorded_step={index} "
                                f"recorded_termination={episode.termination_reason.value}",
                                flush=True,
                            )
                            break
                print(
                    f"REPLAY_EPISODE_COMPLETE index={replay_index}/{len(replay_items)} episode={episode.episode_id}",
                    flush=True,
                )
            # 在关闭 Isaac App 前输出汇总；Kit close 可能终止或吞掉其后的标准输出。
            print(
                f"REPLAY_COMPLETE episodes={len(replay_items)} mode={args_cli.mode} "
                f"episode_ids={','.join(episode.episode_id for episode, _ in replay_items)}",
                flush=True,
            )
        finally:
            # 无论重播正常结束还是中途报错，都先关闭 Gym 环境再关闭 Kit 应用。
            env.close()
    finally:
        # 释放 Isaac Sim/Kit 资源，避免下次 replay 继承残留窗口或渲染状态。
        simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
