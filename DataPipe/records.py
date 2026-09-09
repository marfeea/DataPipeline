"""P0 episode、边界和动作字段的纯 Python 实现。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any

from .config import (
    ACTION_DIMENSION,
    ARM_ACTION_DIMENSION,
    DATASET_SCHEMA_VERSION,
    NORMALIZED_ACTION_HIGH,
    NORMALIZED_ACTION_LOW,
    TCP_HEADING_SCALE_RAD,
)


def _fixed_vector(name: str, values: Sequence[float], *, size: int = ACTION_DIMENSION) -> tuple[float, ...]:
    """规范化并检查一个固定维度的有限浮点向量。"""

    result = tuple(float(value) for value in values)
    if len(result) != size:
        raise ValueError(f"{name} 必须是 {size} 维，实际为 {len(result)} 维")
    if not all(isfinite(value) for value in result):
        raise ValueError(f"{name} 不能包含 NaN 或 Inf")
    return result


class TerminationReason(str, Enum):
    """P0 需要区分的终局类别。"""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    SAFETY_FAILURE = "safety_failure"

    @property
    def is_terminal(self) -> bool:
        """成功和安全失败需要写入 terminal，超时不需要。"""

        return self is not TerminationReason.TIMEOUT


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """一个控制步的动作和机械臂状态快照。

    ``action`` 是 R2Dreamer 主动作键，固定等于归一化动作；
    ``raw_action`` 保留归一化前输出，便于审计裁剪或缩放行为。
    """

    raw_action: Sequence[float]
    action: Sequence[float]
    tcp_displacement: Sequence[float]
    ik_output: Sequence[float]
    joint_position: Sequence[float]
    joint_velocity: Sequence[float]
    waypoint_tcp_position_b: Sequence[float] = field(default_factory=lambda: (0.0,) * 3)
    controller_target_tcp_position_b: Sequence[float] = field(default_factory=lambda: (0.0,) * 3)
    ik_seed_joint_position: Sequence[float] = field(default_factory=lambda: (0.0,) * 6)
    ik_solution_joint_position: Sequence[float] = field(default_factory=lambda: (0.0,) * 6)
    limited_joint_target: Sequence[float] = field(default_factory=lambda: (0.0,) * 6)
    commanded_joint_target: Sequence[float] = field(default_factory=lambda: (0.0,) * 6)
    actual_tcp_position_b: Sequence[float] = field(default_factory=lambda: (0.0,) * 3)
    gripper_joint_position: Sequence[float] = field(default_factory=lambda: (0.0,) * 2)
    gripper_joint_velocity: Sequence[float] = field(default_factory=lambda: (0.0,) * 2)
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

    def __post_init__(self) -> None:
        raw_action = _fixed_vector("raw_action", self.raw_action)
        action = _fixed_vector("action", self.action)
        tcp_displacement = _fixed_vector("tcp_displacement", self.tcp_displacement, size=ARM_ACTION_DIMENSION)
        ik_output = _fixed_vector("ik_output", self.ik_output, size=ARM_ACTION_DIMENSION)
        joint_position = _fixed_vector("joint_position", self.joint_position, size=ARM_ACTION_DIMENSION)
        joint_velocity = _fixed_vector("joint_velocity", self.joint_velocity, size=ARM_ACTION_DIMENSION)
        waypoint_tcp_position_b = _fixed_vector("waypoint_tcp_position_b", self.waypoint_tcp_position_b, size=3)
        controller_target_tcp_position_b = _fixed_vector(
            "controller_target_tcp_position_b", self.controller_target_tcp_position_b, size=3
        )
        ik_seed_joint_position = _fixed_vector(
            "ik_seed_joint_position", self.ik_seed_joint_position, size=ARM_ACTION_DIMENSION
        )
        ik_solution_joint_position = _fixed_vector(
            "ik_solution_joint_position", self.ik_solution_joint_position, size=ARM_ACTION_DIMENSION
        )
        limited_joint_target = _fixed_vector(
            "limited_joint_target", self.limited_joint_target, size=ARM_ACTION_DIMENSION
        )
        commanded_joint_target = _fixed_vector(
            "commanded_joint_target", self.commanded_joint_target, size=ARM_ACTION_DIMENSION
        )
        actual_tcp_position_b = _fixed_vector("actual_tcp_position_b", self.actual_tcp_position_b, size=3)
        gripper_joint_position = _fixed_vector("gripper_joint_position", self.gripper_joint_position, size=2)
        gripper_joint_velocity = _fixed_vector("gripper_joint_velocity", self.gripper_joint_velocity, size=2)
        scalar_fields = {
            "tcp_heading_target_rad": float(self.tcp_heading_target_rad),
            "tcp_heading_actual_rad": float(self.tcp_heading_actual_rad),
            "tcp_heading_first_five_rad": float(self.tcp_heading_first_five_rad),
            "tcp_heading_sixth_axis_rad": float(self.tcp_heading_sixth_axis_rad),
            "tcp_heading_error_rad": float(self.tcp_heading_error_rad),
        }
        if not all(isfinite(value) for value in scalar_fields.values()):
            raise ValueError("TCP heading fields 不能包含 NaN 或 Inf")
        gripper_command = action[-1] if self.gripper_command is None else float(self.gripper_command)
        semantic_stage_id = int(self.semantic_stage_id)
        waypoint_index = int(self.waypoint_index)
        if not isfinite(gripper_command) or not NORMALIZED_ACTION_LOW <= gripper_command <= NORMALIZED_ACTION_HIGH:
            raise ValueError("gripper_command must be finite and within [-1, 1]")
        if semantic_stage_id < 0 or waypoint_index < 0:
            raise ValueError("semantic_stage_id and waypoint_index must be non-negative")
        if any(value < NORMALIZED_ACTION_LOW or value > NORMALIZED_ACTION_HIGH for value in action):
            raise ValueError("action 必须位于 [-1, 1]")
        if abs(action[-1] - gripper_command) > 1.0e-6:
            raise ValueError("action gripper component must equal gripper_command")
        if abs(action[-2] * TCP_HEADING_SCALE_RAD - scalar_fields["tcp_heading_target_rad"]) > 1.0e-6:
            raise ValueError("action TCP heading component must equal tcp_heading_target_rad")
        object.__setattr__(self, "raw_action", raw_action)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "tcp_displacement", tcp_displacement)
        object.__setattr__(self, "ik_output", ik_output)
        object.__setattr__(self, "joint_position", joint_position)
        object.__setattr__(self, "joint_velocity", joint_velocity)
        object.__setattr__(self, "waypoint_tcp_position_b", waypoint_tcp_position_b)
        object.__setattr__(self, "controller_target_tcp_position_b", controller_target_tcp_position_b)
        object.__setattr__(self, "ik_seed_joint_position", ik_seed_joint_position)
        object.__setattr__(self, "ik_solution_joint_position", ik_solution_joint_position)
        object.__setattr__(self, "limited_joint_target", limited_joint_target)
        object.__setattr__(self, "commanded_joint_target", commanded_joint_target)
        object.__setattr__(self, "actual_tcp_position_b", actual_tcp_position_b)
        object.__setattr__(self, "gripper_joint_position", gripper_joint_position)
        object.__setattr__(self, "gripper_joint_velocity", gripper_joint_velocity)
        for name, value in scalar_fields.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "gripper_command", gripper_command)
        object.__setattr__(self, "semantic_stage_id", semantic_stage_id)
        object.__setattr__(self, "waypoint_index", waypoint_index)
        object.__setattr__(self, "ik_success", bool(self.ik_success))
        object.__setattr__(self, "delta_limited", bool(self.delta_limited))
        object.__setattr__(self, "velocity_limited", bool(self.velocity_limited))
        object.__setattr__(self, "position_limited", bool(self.position_limited))

    @property
    def normalized_action(self) -> tuple[float, ...]:
        """返回与 R2Dreamer ``action`` 完全相同的归一化动作。"""

        return self.action

    def as_dict(self) -> dict[str, Any]:
        """返回不依赖具体写盘后端的字段快照。"""

        return {
            "raw_action": self.raw_action,
            "action": self.action,
            "normalized_action": self.normalized_action,
            "tcp_displacement": self.tcp_displacement,
            "ik_output": self.ik_output,
            "joint_position": self.joint_position,
            "joint_velocity": self.joint_velocity,
            "waypoint_tcp_position_b": self.waypoint_tcp_position_b,
            "controller_target_tcp_position_b": self.controller_target_tcp_position_b,
            "ik_seed_joint_position": self.ik_seed_joint_position,
            "ik_solution_joint_position": self.ik_solution_joint_position,
            "limited_joint_target": self.limited_joint_target,
            "commanded_joint_target": self.commanded_joint_target,
            "actual_tcp_position_b": self.actual_tcp_position_b,
            "gripper_joint_position": self.gripper_joint_position,
            "gripper_joint_velocity": self.gripper_joint_velocity,
            "tcp_heading_target_rad": self.tcp_heading_target_rad,
            "tcp_heading_actual_rad": self.tcp_heading_actual_rad,
            "tcp_heading_first_five_rad": self.tcp_heading_first_five_rad,
            "tcp_heading_sixth_axis_rad": self.tcp_heading_sixth_axis_rad,
            "tcp_heading_error_rad": self.tcp_heading_error_rad,
            "gripper_command": self.gripper_command,
            "semantic_stage_id": self.semantic_stage_id,
            "waypoint_index": self.waypoint_index,
            "ik_success": self.ik_success,
            "delta_limited": self.delta_limited,
            "velocity_limited": self.velocity_limited,
            "position_limited": self.position_limited,
        }


@dataclass(frozen=True, slots=True)
class BoundaryFlags:
    """观测帧边界标志。"""

    is_first: bool = False
    is_last: bool = False
    is_terminal: bool = False

    def __post_init__(self) -> None:
        if self.is_terminal and not self.is_last:
            raise ValueError("is_terminal=True 时必须同时 is_last=True")


@dataclass(frozen=True, slots=True)
class ObservationFrame:
    """episode 中的一个观测帧，长度按 T+1 组织。"""

    timestep: int
    value: Any
    boundary: BoundaryFlags


@dataclass(frozen=True, slots=True)
class StepRecord:
    """一个 action/reward 控制步，长度按 T 组织。"""

    timestep: int
    action: ActionRecord
    reward: float
    reward_components: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestep < 0:
            raise ValueError("timestep 不能为负数")
        if not isfinite(float(self.reward)):
            raise ValueError("reward 不能为 NaN 或 Inf")
        object.__setattr__(self, "reward", float(self.reward))
        components = {str(name): float(value) for name, value in self.reward_components.items()}
        if not all(isfinite(value) for value in components.values()):
            raise ValueError("reward_components 不能包含 NaN 或 Inf")
        object.__setattr__(self, "reward_components", components)


@dataclass(frozen=True, slots=True)
class Episode:
    """已提交的完整 episode。

    不变量：``len(observations) == len(steps) + 1``，因此时间轴严格为
    ``o0-a0-r0-o1-...-a(T-1)-r(T-1)-oT``。
    """

    episode_id: str
    observations: tuple[ObservationFrame, ...]
    steps: tuple[StepRecord, ...]
    termination_reason: TerminationReason
    schema_version: str = DATASET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.validate()

    @property
    def length(self) -> int:
        """返回动作/奖励步数 T。"""

        return len(self.steps)

    @property
    def terminal_observation(self) -> Any:
        """返回 reset 前必须落盘的终局观测。"""

        return self.observations[-1].value

    def validate(self) -> None:
        """验证 P0-1 至 P0-3 的结构和边界不变量。"""

        if not self.episode_id:
            raise ValueError("episode_id 不能为空")
        if self.schema_version != DATASET_SCHEMA_VERSION:
            raise ValueError(f"不支持的数据 schema：{self.schema_version}")
        if len(self.observations) != len(self.steps) + 1:
            raise ValueError("observation 长度必须等于 action/reward 长度加一")
        if not self.observations:
            raise ValueError("episode 至少需要一个首观测")
        for expected_timestep, frame in enumerate(self.observations):
            if frame.timestep != expected_timestep:
                raise ValueError("观测 timestep 必须从 0 连续递增")
        for expected_timestep, step in enumerate(self.steps):
            if step.timestep != expected_timestep:
                raise ValueError("action/reward timestep 必须从 0 连续递增")
        if not self.observations[0].boundary.is_first:
            raise ValueError("首观测必须设置 is_first=True")
        if any(frame.boundary.is_first for frame in self.observations[1:]):
            raise ValueError("只有首观测可以设置 is_first=True")
        if not self.observations[-1].boundary.is_last:
            raise ValueError("终局观测必须设置 is_last=True")
        if any(frame.boundary.is_last for frame in self.observations[:-1]):
            raise ValueError("只有终局观测可以设置 is_last=True")
        if self.observations[-1].boundary.is_terminal != self.termination_reason.is_terminal:
            raise ValueError("终局 is_terminal 与 termination_reason 不一致")

    def timeline(self) -> tuple[str, ...]:
        """返回可直接用于 P0-1 验收的时间轴 token。"""

        tokens: list[str] = []
        for timestep, step in enumerate(self.steps):
            tokens.extend((f"o{timestep}", f"a{timestep}", f"r{timestep}"))
        tokens.append(f"o{len(self.steps)}")
        return tuple(tokens)

    def step_fields(self, timestep: int) -> dict[str, Any]:
        """并排返回指定控制步的动作、IK 和实际关节字段。"""

        step = self.steps[timestep]
        return {"timestep": timestep, "reward": step.reward, **step.action.as_dict()}

    def as_record(self) -> dict[str, Any]:
        """返回供 writer 序列化的结构化记录。"""

        return {
            "episode_id": self.episode_id,
            "schema_version": self.schema_version,
            "termination_reason": self.termination_reason.value,
            "observations": [
                {
                    "timestep": frame.timestep,
                    "value": frame.value,
                    "is_first": frame.boundary.is_first,
                    "is_last": frame.boundary.is_last,
                    "is_terminal": frame.boundary.is_terminal,
                }
                for frame in self.observations
            ],
            "steps": [
                {
                    "timestep": step.timestep,
                    "reward": step.reward,
                    "reward_components": dict(step.reward_components),
                    **step.action.as_dict(),
                }
                for step in self.steps
            ],
        }


_UNSET = object()


class EpisodeBuilder:
    """按先观测、后控制步、再终局观测的顺序装配 episode。"""

    def __init__(self, episode_id: str) -> None:
        self.episode_id = episode_id
        self._active = False
        self._observations: list[ObservationFrame] = []
        self._steps: list[StepRecord] = []
        self._last_episode: Episode | None = None

    @property
    def is_active(self) -> bool:
        """是否存在尚未提交的 episode。"""

        return self._active

    def start(self, observation: Any) -> None:
        """写入首观测；调用前一个 episode 必须已经 close。"""

        if self._active:
            raise RuntimeError("当前 episode 尚未 close，不能开始下一条轨迹")
        self._observations = [ObservationFrame(0, observation, BoundaryFlags(is_first=True))]
        self._steps = []
        self._active = True

    def append(
        self,
        action: ActionRecord,
        reward: float,
        next_observation: Any,
        reward_components: Mapping[str, float] | None = None,
    ) -> None:
        """追加一个 action/reward/next-observation 三元组。"""

        if not self._active:
            raise RuntimeError("必须先 start episode")
        timestep = len(self._steps)
        self._steps.append(StepRecord(timestep, action, reward, reward_components or {}))
        self._observations.append(ObservationFrame(timestep + 1, next_observation, BoundaryFlags()))

    def close(self, termination_reason: TerminationReason, terminal_observation: Any = _UNSET) -> Episode:
        """在 reset 前提交终局观测，并冻结为完整 episode。"""

        if not self._active:
            raise RuntimeError("没有可 close 的活动 episode")
        if terminal_observation is _UNSET:
            raise ValueError("close 必须显式提供 terminal observation，确保 reset 前落盘")
        final_timestep = len(self._steps)
        self._observations[-1] = ObservationFrame(
            final_timestep,
            terminal_observation,
            BoundaryFlags(is_last=True, is_terminal=termination_reason.is_terminal),
        )
        episode = Episode(
            episode_id=self.episode_id,
            observations=tuple(self._observations),
            steps=tuple(self._steps),
            termination_reason=termination_reason,
        )
        self._last_episode = episode
        self._active = False
        return episode

    def reset(self) -> None:
        """清空 builder；活动 episode 未 close 时拒绝 reset。"""

        if self._active:
            raise RuntimeError("必须先 close 并提交 terminal observation，再 reset")
        self._observations = []
        self._steps = []
        self._last_episode = None


__all__ = [
    "ActionRecord",
    "BoundaryFlags",
    "Episode",
    "EpisodeBuilder",
    "ObservationFrame",
    "StepRecord",
    "TerminationReason",
]
