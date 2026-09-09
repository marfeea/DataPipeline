"""P1-4 单控制步采集适配器的纯逻辑边界。"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from .config import CameraSensorSpec
from .records import ActionRecord


@dataclass(frozen=True, slots=True)
class ControlState:
    """Isaac 适配层读取后交给数据层的最小状态快照。"""

    sim_step: int
    sim_time_s: float
    raw_action: tuple[float, ...]
    normalized_action: tuple[float, ...]
    tcp_displacement: tuple[float, ...]
    ik_output: tuple[float, ...]
    joint_position: tuple[float, ...]
    joint_velocity: tuple[float, ...]
    observation: Any
    waypoint_tcp_position_b: tuple[float, ...] = (0.0, 0.0, 0.0)
    controller_target_tcp_position_b: tuple[float, ...] = (0.0, 0.0, 0.0)
    ik_seed_joint_position: tuple[float, ...] = (0.0,) * 6
    ik_solution_joint_position: tuple[float, ...] = (0.0,) * 6
    limited_joint_target: tuple[float, ...] = (0.0,) * 6
    commanded_joint_target: tuple[float, ...] = (0.0,) * 6
    actual_tcp_position_b: tuple[float, ...] = (0.0, 0.0, 0.0)
    gripper_joint_position: tuple[float, ...] = (0.0, 0.0)
    gripper_joint_velocity: tuple[float, ...] = (0.0, 0.0)
    tcp_heading_target_rad: float = 0.0
    tcp_heading_actual_rad: float = 0.0
    tcp_heading_first_five_rad: float = 0.0
    tcp_heading_sixth_axis_rad: float = 0.0
    tcp_heading_error_rad: float = 0.0
    gripper_command: float | None = None
    semantic_stage_id: int = 0
    waypoint_index: int = 0
    ik_success: bool = True
    delta_limited: bool = False
    velocity_limited: bool = False
    position_limited: bool = False

    def action_record(self) -> ActionRecord:
        """将控制器各层字段转换为 P0-3 固定记录。"""

        if self.sim_step < 0 or not isfinite(self.sim_time_s):
            raise ValueError("sim_step 和 sim_time_s 必须有效")
        return ActionRecord(
            raw_action=self.raw_action,
            action=self.normalized_action,
            tcp_displacement=self.tcp_displacement,
            ik_output=self.ik_output,
            joint_position=self.joint_position,
            joint_velocity=self.joint_velocity,
            waypoint_tcp_position_b=self.waypoint_tcp_position_b,
            controller_target_tcp_position_b=self.controller_target_tcp_position_b,
            ik_seed_joint_position=self.ik_seed_joint_position,
            ik_solution_joint_position=self.ik_solution_joint_position,
            limited_joint_target=self.limited_joint_target,
            commanded_joint_target=self.commanded_joint_target,
            actual_tcp_position_b=self.actual_tcp_position_b,
            gripper_joint_position=self.gripper_joint_position,
            gripper_joint_velocity=self.gripper_joint_velocity,
            tcp_heading_target_rad=self.tcp_heading_target_rad,
            tcp_heading_actual_rad=self.tcp_heading_actual_rad,
            tcp_heading_first_five_rad=self.tcp_heading_first_five_rad,
            tcp_heading_sixth_axis_rad=self.tcp_heading_sixth_axis_rad,
            tcp_heading_error_rad=self.tcp_heading_error_rad,
            gripper_command=self.gripper_command,
            semantic_stage_id=self.semantic_stage_id,
            waypoint_index=self.waypoint_index,
            ik_success=self.ik_success,
            delta_limited=self.delta_limited,
            velocity_limited=self.velocity_limited,
            position_limited=self.position_limited,
        )


@dataclass(frozen=True, slots=True)
class CaptureRecord:
    """同一控制步的状态、动作、观测和同步标识。"""

    sim_step: int
    sim_time_s: float
    action: ActionRecord
    observation: Any


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """一个 CameraSensor 输出帧及其同步标识。"""

    camera_name: str
    sim_step: int
    timestamp_s: float
    data: Any
    shape: tuple[int, ...]
    dtype: str


class CameraCaptureAdapter:
    """检查相机配置、控制步同步和连续帧时间单调性。"""

    def __init__(self, specs: tuple[CameraSensorSpec, ...]) -> None:
        for spec in specs:
            spec.validate()
        self._specs = {spec.name: spec for spec in specs}
        self._last_step: dict[str, int] = {}
        self._last_timestamp: dict[str, float] = {}

    def reset_episode(self) -> None:
        """Reset cross-frame checks at an environment episode boundary.

        Isaac's global simulation step and the camera sensor timestamp do not
        have to increase across ``env.reset()``. The camera sensor itself is
        reset at that boundary, so the next frame starts a new local sequence.
        """

        self._last_step.clear()
        self._last_timestamp.clear()

    def capture(self, frames: tuple[CameraFrame, ...], expected_sim_step: int) -> tuple[CameraFrame, ...]:
        """返回当前控制步完整相机帧；缺帧、陈旧帧或时间倒退均快速失败。"""

        frame_names = [frame.camera_name for frame in frames]
        if len(frame_names) != len(set(frame_names)):
            raise ValueError("当前控制步包含重复 camera frame")
        if len(frames) != len(self._specs) or set(frame_names) != set(self._specs):
            raise ValueError("当前控制步的相机集合与配置不一致")
        for frame in frames:
            if frame.sim_step != expected_sim_step:
                raise ValueError(f"camera={frame.camera_name} 与 sim_step 不同步")
            if not isfinite(frame.timestamp_s):
                raise ValueError(f"camera={frame.camera_name} timestamp 无效")
            if not frame.shape or any(value <= 0 for value in frame.shape) or not frame.dtype:
                raise ValueError(f"camera={frame.camera_name} shape/dtype 无效")
            if frame.camera_name in self._last_step and frame.sim_step <= self._last_step[frame.camera_name]:
                raise ValueError(f"camera={frame.camera_name} 出现陈旧或重复帧")
            if (
                frame.camera_name in self._last_timestamp
                and frame.timestamp_s <= self._last_timestamp[frame.camera_name]
            ):
                raise ValueError(f"camera={frame.camera_name} timestamp 不递增")
            self._last_step[frame.camera_name] = frame.sim_step
            self._last_timestamp[frame.camera_name] = frame.timestamp_s
        return frames


class StateCaptureAdapter:
    """检查所有字段共享同一个仿真步后再交给 writer。"""

    def capture(self, state: ControlState) -> CaptureRecord:
        action = state.action_record()
        return CaptureRecord(state.sim_step, state.sim_time_s, action, state.observation)


__all__ = ["CameraCaptureAdapter", "CameraFrame", "CaptureRecord", "ControlState", "StateCaptureAdapter"]
