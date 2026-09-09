"""Thin Isaac adapter for the collection-only two-joint gripper command."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from configs.assets import ROBOT_PRIM_CONTRACT


class DatasetGripperController:
    """Map normalized open/close commands to validated joint targets."""

    def __init__(
        self,
        robot: Any,
        *,
        open_position: Sequence[float],
        closed_position: Sequence[float],
    ) -> None:
        self.robot = robot
        self.joint_ids, joint_names = robot.find_joints(list(ROBOT_PRIM_CONTRACT.gripper_joints), preserve_order=True)
        if tuple(joint_names) != ROBOT_PRIM_CONTRACT.gripper_joints:
            raise RuntimeError(f"gripper joint contract mismatch: {tuple(joint_names)}")
        self.open_position = self._target(open_position, name="open_position")
        self.closed_position = self._target(closed_position, name="closed_position")
        limits = robot.data.joint_pos_limits[0, self.joint_ids]
        for name, target in (("open_position", self.open_position), ("closed_position", self.closed_position)):
            if bool(torch.any(target < limits[:, 0])) or bool(torch.any(target > limits[:, 1])):
                raise ValueError(f"{name} is outside the USD gripper joint limits")

    def _target(self, values: Sequence[float], *, name: str) -> torch.Tensor:
        result = tuple(float(value) for value in values)
        if len(result) != 2:
            raise ValueError(f"{name} must contain UpperFinger and DownFinger targets")
        return torch.as_tensor(
            result,
            dtype=self.robot.data.joint_pos.dtype,
            device=self.robot.data.joint_pos.device,
        )

    def apply(self, command: float) -> None:
        command = max(-1.0, min(1.0, float(command)))
        open_fraction = 0.5 * (command + 1.0)
        target = self.closed_position + open_fraction * (self.open_position - self.closed_position)
        self.robot.set_joint_position_target(target.view(1, 2), joint_ids=self.joint_ids)

    def state(self) -> tuple[tuple[float, float], tuple[float, float]]:
        position = tuple(float(value) for value in self.robot.data.joint_pos[0, self.joint_ids].detach().cpu())
        velocity = tuple(float(value) for value in self.robot.data.joint_vel[0, self.joint_ids].detach().cpu())
        return position, velocity  # type: ignore[return-value]

    @property
    def closed_position_tuple(self) -> tuple[float, float]:
        return tuple(float(value) for value in self.closed_position.detach().cpu())  # type: ignore[return-value]

    @property
    def open_position_tuple(self) -> tuple[float, float]:
        return tuple(float(value) for value in self.open_position.detach().cpu())  # type: ignore[return-value]


__all__ = ["DatasetGripperController"]
