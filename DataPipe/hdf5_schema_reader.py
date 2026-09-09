"""Read formal HDF5 demos back into the pure Python episode contract."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import numpy as np

from .hdf5_schema import xyzw_to_wxyz
from .records import ActionRecord, BoundaryFlags, Episode, ObservationFrame, StepRecord, TerminationReason


def _python_value(value: np.ndarray) -> Any:
    value = np.asarray(value)
    if value.ndim == 0:
        return value.item()
    return value.tolist()


def episode_from_hdf5_group(
    group: Any,
    *,
    observation_keys: Collection[str] | None = None,
) -> Episode:
    """Reconstruct one episode and validate its ``T/T+1`` semantics.

    ``observation_keys`` allows callers such as replay to skip unused visual
    fields instead of expanding them into Python lists.
    """

    action_count = int(group["actions"].shape[0])
    observation_group = group["obs"]
    if observation_keys is None:
        selected_observation_keys = tuple(observation_group)
    else:
        selected_observation_keys = tuple(str(name) for name in observation_keys)
        if len(selected_observation_keys) != len(set(selected_observation_keys)):
            raise ValueError("observation_keys must not contain duplicates")
        missing_keys = set(selected_observation_keys).difference(observation_group)
        if missing_keys:
            raise KeyError(f"observation fields are missing: {sorted(missing_keys)}")
    observations = []
    for index in range(action_count + 1):
        value = {}
        for name in selected_observation_keys:
            field_value = _python_value(observation_group[name][index])
            if name.endswith("pose") and np.asarray(field_value).shape == (7,):
                field_value = xyzw_to_wxyz(field_value)
                value[name + "_wxyz"] = field_value
            else:
                value[name] = field_value
        observations.append(
            ObservationFrame(
                index,
                value,
                BoundaryFlags(is_first=index == 0, is_last=index == action_count, is_terminal=False),
            )
        )
    reason = TerminationReason(str(group.attrs["termination_reason"]))
    observations[-1] = ObservationFrame(
        action_count,
        observations[-1].value,
        BoundaryFlags(is_last=True, is_terminal=reason.is_terminal),
    )
    diagnostics = group["diagnostics"]
    reward_components_group = diagnostics.get("reward_components")
    steps = []
    for index in range(action_count):
        reward_components = {}
        if reward_components_group is not None:
            reward_components = {name: float(reward_components_group[name][index]) for name in reward_components_group}
        steps.append(
            StepRecord(
                index,
                ActionRecord(
                    raw_action=diagnostics["raw_action"][index],
                    action=group["actions"][index],
                    tcp_displacement=diagnostics["tcp_displacement"][index],
                    ik_output=diagnostics["ik_output"][index],
                    joint_position=diagnostics["joint_position"][index],
                    joint_velocity=diagnostics["joint_velocity"][index],
                    waypoint_tcp_position_b=(
                        diagnostics["waypoint_tcp_position_b"][index]
                        if "waypoint_tcp_position_b" in diagnostics
                        else (0.0,) * 3
                    ),
                    controller_target_tcp_position_b=(
                        diagnostics["controller_target_tcp_position_b"][index]
                        if "controller_target_tcp_position_b" in diagnostics
                        else (0.0,) * 3
                    ),
                    ik_seed_joint_position=(
                        diagnostics["ik_seed_joint_position"][index]
                        if "ik_seed_joint_position" in diagnostics
                        else (0.0,) * 6
                    ),
                    ik_solution_joint_position=(
                        diagnostics["ik_solution_joint_position"][index]
                        if "ik_solution_joint_position" in diagnostics
                        else diagnostics["ik_output"][index]
                    ),
                    limited_joint_target=(
                        diagnostics["limited_joint_target"][index]
                        if "limited_joint_target" in diagnostics
                        else diagnostics["ik_output"][index]
                    ),
                    commanded_joint_target=(
                        diagnostics["commanded_joint_target"][index]
                        if "commanded_joint_target" in diagnostics
                        else diagnostics["ik_output"][index]
                    ),
                    actual_tcp_position_b=(
                        diagnostics["actual_tcp_position_b"][index]
                        if "actual_tcp_position_b" in diagnostics
                        else (0.0,) * 3
                    ),
                    gripper_joint_position=(
                        diagnostics["gripper_joint_position"][index]
                        if "gripper_joint_position" in diagnostics
                        else (0.0,) * 2
                    ),
                    gripper_joint_velocity=(
                        diagnostics["gripper_joint_velocity"][index]
                        if "gripper_joint_velocity" in diagnostics
                        else (0.0,) * 2
                    ),
                    gripper_command=(
                        float(diagnostics["gripper_command"][index])
                        if "gripper_command" in diagnostics
                        else float(group["actions"][index][-1])
                    ),
                    tcp_heading_target_rad=(
                        float(diagnostics["tcp_heading_target_rad"][index])
                        if "tcp_heading_target_rad" in diagnostics
                        else 0.0
                    ),
                    tcp_heading_actual_rad=(
                        float(diagnostics["tcp_heading_actual_rad"][index])
                        if "tcp_heading_actual_rad" in diagnostics
                        else 0.0
                    ),
                    tcp_heading_first_five_rad=(
                        float(diagnostics["tcp_heading_first_five_rad"][index])
                        if "tcp_heading_first_five_rad" in diagnostics
                        else 0.0
                    ),
                    tcp_heading_sixth_axis_rad=(
                        float(diagnostics["tcp_heading_sixth_axis_rad"][index])
                        if "tcp_heading_sixth_axis_rad" in diagnostics
                        else 0.0
                    ),
                    tcp_heading_error_rad=(
                        float(diagnostics["tcp_heading_error_rad"][index])
                        if "tcp_heading_error_rad" in diagnostics
                        else 0.0
                    ),
                    semantic_stage_id=int(group["phases"][index]),
                    waypoint_index=(int(group["waypoint_indices"][index]) if "waypoint_indices" in group else 0),
                    ik_success=bool(diagnostics["ik_success"][index]) if "ik_success" in diagnostics else True,
                    delta_limited=(
                        bool(diagnostics["delta_limited"][index]) if "delta_limited" in diagnostics else False
                    ),
                    velocity_limited=(
                        bool(diagnostics["velocity_limited"][index]) if "velocity_limited" in diagnostics else False
                    ),
                    position_limited=(
                        bool(diagnostics["position_limited"][index]) if "position_limited" in diagnostics else False
                    ),
                ),
                float(group["rewards"][index]),
                reward_components,
            )
        )
    return Episode(
        episode_id=str(group.attrs["episode_id"]),
        observations=tuple(observations),
        steps=tuple(steps),
        termination_reason=reason,
    )


__all__ = ["episode_from_hdf5_group"]
