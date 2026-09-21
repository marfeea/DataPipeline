"""Episode JSON 序列化和反序列化。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .records import ActionRecord, BoundaryFlags, Episode, ObservationFrame, StepRecord, TerminationReason


def json_safe(value: Any) -> Any:
    """将常见 numpy/Torch 标量和数组转换为 JSON 值。"""

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("数据不能包含 NaN 或 Inf")
        return value
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    raise TypeError(f"无法 JSON 序列化字段类型：{type(value).__name__}")


def episode_to_record(episode: Episode) -> dict[str, Any]:
    """转换为 JSON-safe episode 记录。"""

    return json_safe(episode.as_record())


def episode_from_record(record: Mapping[str, Any]) -> Episode:
    """从 writer 记录恢复 Episode 并重新执行 P0 校验。"""

    observations = tuple(
        ObservationFrame(
            timestep=int(item["timestep"]),
            value=item.get("value"),
            boundary=BoundaryFlags(
                is_first=bool(item["is_first"]),
                is_last=bool(item["is_last"]),
                is_terminal=bool(item["is_terminal"]),
            ),
        )
        for item in record["observations"]
    )
    steps = tuple(
        StepRecord(
            timestep=int(item["timestep"]),
            reward=float(item["reward"]),
            action=ActionRecord(
                raw_action=item["raw_action"],
                action=item["action"],
                tcp_displacement=item["tcp_displacement"],
                ik_output=item["ik_output"],
                joint_position=item["joint_position"],
                joint_velocity=item["joint_velocity"],
                waypoint_tcp_position_b=item.get("waypoint_tcp_position_b", (0.0,) * 3),
                controller_target_tcp_position_b=item.get("controller_target_tcp_position_b", (0.0,) * 3),
                ik_seed_joint_position=item.get("ik_seed_joint_position", (0.0,) * 6),
                ik_solution_joint_position=item.get("ik_solution_joint_position", item["ik_output"]),
                limited_joint_target=item.get("limited_joint_target", item["ik_output"]),
                commanded_joint_target=item.get("commanded_joint_target", item["ik_output"]),
                actual_tcp_position_b=item.get("actual_tcp_position_b", (0.0,) * 3),
                gripper_joint_position=item.get("gripper_joint_position", (0.0,) * 2),
                gripper_joint_velocity=item.get("gripper_joint_velocity", (0.0,) * 2),
                tcp_heading_target_rad=item.get("tcp_heading_target_rad", 0.0),
                tcp_heading_actual_rad=item.get("tcp_heading_actual_rad", 0.0),
                tcp_heading_first_five_rad=item.get("tcp_heading_first_five_rad", 0.0),
                tcp_heading_sixth_axis_rad=item.get("tcp_heading_sixth_axis_rad", 0.0),
                tcp_heading_error_rad=item.get("tcp_heading_error_rad", 0.0),
                gripper_command=item.get("gripper_command", item["action"][-1]),
                semantic_stage_id=item.get("semantic_stage_id", 0),
                waypoint_index=item.get("waypoint_index", 0),
                ik_success=item.get("ik_success", True),
                delta_limited=item.get("delta_limited", False),
                velocity_limited=item.get("velocity_limited", False),
                position_limited=item.get("position_limited", False),
                action_applied=item.get("action_applied"),
            ),
            reward_components=item.get("reward_components", {}),
            reward_valid=item.get("reward_valid", True),
        )
        for item in record["steps"]
    )
    return Episode(
        episode_id=str(record["episode_id"]),
        observations=observations,
        steps=steps,
        termination_reason=TerminationReason(str(record["termination_reason"])),
        schema_version=str(record["schema_version"]),
    )


__all__ = ["episode_from_record", "episode_to_record", "json_safe"]
