"""AUBO 数值 IK 的独立 Manager-Based 动作适配链。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from configs.assets import ROBOT_PRIM_CONTRACT
from configs.aubo_ik_control import (
    ACTION_CLIP,
    HARD_LIMIT_TERMINATION_MARGIN_RAD,
    IK_AXIS_TOLERANCE_RAD,
    IK_DAMPING,
    IK_INTERNAL_MAX_JOINT_STEP_RAD,
    IK_ORIENTATION_TOLERANCE_RAD,
    IK_POSITION_TOLERANCE_M,
    JOINT_TARGET_MARGIN_RAD,
    MAX_IK_ITERATIONS,
    MAX_JOINT_VELOCITY_RAD_S,
    MAX_POLICY_JOINT_DELTA_RAD,
    POSITION_INCREMENT_SCALE_M,
    ROTATION_INCREMENT_SCALE_RAD,
)
from configs.task import FLANGE_TO_TCP_TRANSLATION_F, FLANGE_TO_TOOL_ROTATION_F

from .aubo_ik import (
    IKMode,
    create_ik_solver,
    flange_down_tcp_target_pose,
    solve_flange_down_ik,
)
from .differential_ik import apply_tcp_increment, scale_policy_action
from .tcp_kinematics import (
    quaternion_conjugate,
    quaternion_multiply,
    rotate_vector,
    world_to_root_position,
)

_DIAGNOSTIC_TOLERANCE_RAD = 1.0e-6
_FLANGE_DOWN_IK_MODE = "flange_down_6d"
_CURRENT_JOINT_SEED_STRATEGY = "current_joint_position"


@dataclass(frozen=True)
class AuboIKDiagnostics:
    raw_action: torch.Tensor
    processed_action: torch.Tensor
    target_tcp_position_b: torch.Tensor
    target_tcp_quaternion_b: torch.Tensor
    ik_mode: tuple[str, ...]
    ik_success: torch.Tensor
    ik_reason: tuple[str, ...]
    ik_iterations: torch.Tensor
    ik_position_error_m: torch.Tensor
    ik_orientation_error_rad: torch.Tensor
    ik_seed_joint_position: torch.Tensor
    ik_solution_joint_position: torch.Tensor
    requested_joint_delta: torch.Tensor
    applied_joint_delta: torch.Tensor
    delta_limited: torch.Tensor
    velocity_limited: torch.Tensor
    position_limited: torch.Tensor
    desired_joint_target: torch.Tensor
    joint_target: torch.Tensor


class AuboIKAction:
    """把归一化 TCP 增量交给当前 AUBO 数值 IK，并安全跟踪关节解。"""

    def __init__(
        self,
        robot: Any,
        physics_dt_s: float,
        policy_decimation: int,
        *,
        diagnostics_enabled: bool = False,
        flange_downward_enabled: bool = False,
        downward_yaw_offset_rad: float = 0.0,
        flange_down_position_tolerance_m: float = IK_POSITION_TOLERANCE_M,
        flange_down_orientation_tolerance_rad: float = IK_ORIENTATION_TOLERANCE_RAD,
        ik_seed_strategy: str = _CURRENT_JOINT_SEED_STRATEGY,
    ) -> None:
        if physics_dt_s <= 0.0 or policy_decimation <= 0:
            raise ValueError("物理步长与策略 decimation 必须为正")
        self.robot = robot
        self.physics_dt_s = physics_dt_s
        self.policy_decimation = policy_decimation
        self.diagnostics_enabled = diagnostics_enabled
        self.flange_downward_enabled = flange_downward_enabled
        if not np.isfinite(downward_yaw_offset_rad):
            raise ValueError("downward_yaw_offset_rad 必须是有限值")
        self.downward_yaw_offset_rad = float(downward_yaw_offset_rad)
        if (
            not np.isfinite(flange_down_position_tolerance_m)
            or not np.isfinite(flange_down_orientation_tolerance_rad)
            or flange_down_position_tolerance_m <= 0.0
            or flange_down_orientation_tolerance_rad <= 0.0
        ):
            raise ValueError("Flange-down IK 容差必须为正")
        self.flange_down_position_tolerance_m = float(flange_down_position_tolerance_m)
        self.flange_down_orientation_tolerance_rad = float(flange_down_orientation_tolerance_rad)
        if ik_seed_strategy != _CURRENT_JOINT_SEED_STRATEGY:
            raise ValueError(
                "正式物理闭环只支持 current_joint_position IK seed，"
                f"实际为 {ik_seed_strategy!r}"
            )
        self.ik_seed_strategy = ik_seed_strategy
        self.joint_ids, joint_names = robot.find_joints(list(ROBOT_PRIM_CONTRACT.arm_joints), preserve_order=True)
        flange_ids, flange_names = robot.find_bodies([ROBOT_PRIM_CONTRACT.flange_body], preserve_order=True)
        if tuple(joint_names) != ROBOT_PRIM_CONTRACT.arm_joints:
            raise RuntimeError(
                f"机械臂关节解析不一致：期望 {ROBOT_PRIM_CONTRACT.arm_joints}，实际 {tuple(joint_names)}"
            )
        if tuple(flange_names) != (ROBOT_PRIM_CONTRACT.flange_body,):
            raise RuntimeError(f"Flange body 必须唯一匹配，实际 {tuple(flange_names)}")
        if len(set(self.joint_ids)) != 6:
            raise RuntimeError(f"机械臂关节索引必须是六个唯一值，实际 {self.joint_ids}")
        self.flange_body_id = flange_ids[0]

        num_envs = robot.data.joint_pos.shape[0]
        device = robot.data.joint_pos.device
        dtype = robot.data.joint_pos.dtype
        self.flange_to_tcp_translation_f = torch.tensor(FLANGE_TO_TCP_TRANSLATION_F, dtype=dtype, device=device).expand(
            num_envs, -1
        )
        self.flange_to_tool_rotation_f = torch.tensor(FLANGE_TO_TOOL_ROTATION_F, dtype=dtype, device=device).expand(
            num_envs, -1
        )
        self.max_policy_joint_delta_rad = torch.tensor(MAX_POLICY_JOINT_DELTA_RAD, dtype=dtype, device=device)
        self.raw_action = torch.zeros((num_envs, 6), dtype=dtype, device=device)
        self.processed_action = torch.zeros_like(self.raw_action)
        self.target_tcp_position_b = torch.zeros((num_envs, 3), dtype=dtype, device=device)
        self.target_tcp_quaternion_b = torch.zeros((num_envs, 4), dtype=dtype, device=device)
        self.policy_start_joint_position = torch.zeros_like(self.raw_action)
        self.ik_seed_joint_position = torch.zeros_like(self.raw_action)
        self.ik_solution_joint_position = torch.zeros_like(self.raw_action)
        self.desired_joint_target = torch.zeros_like(self.raw_action)
        self.commanded_joint_target = torch.zeros_like(self.raw_action)
        self._requested_joint_delta = torch.zeros_like(self.raw_action)
        self._applied_joint_delta = torch.zeros_like(self.raw_action)
        self._trajectory_substep = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._ik_success = torch.ones(num_envs, dtype=torch.bool, device=device)
        self._ik_iterations = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._ik_position_error_m = torch.zeros(num_envs, dtype=dtype, device=device)
        self._ik_orientation_error_rad = torch.zeros(num_envs, dtype=dtype, device=device)
        self._delta_limited = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._velocity_limited = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._position_limited = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._hard_limit_reached = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._ik_modes = tuple(IKMode.FULL_POSE_6D.value for _ in range(num_envs))
        self._ik_reasons = tuple("" for _ in range(num_envs))
        self._solvers = {mode.value: create_ik_solver(mode) for mode in IKMode}
        self._diagnostics: AuboIKDiagnostics | None = None
        self.reset()

    @property
    def diagnostics(self) -> AuboIKDiagnostics | None:
        return self._diagnostics if self.diagnostics_enabled else None

    @property
    def hard_limit_reached(self) -> torch.Tensor:
        return self._hard_limit_reached

    @property
    def ik_failed(self) -> torch.Tensor:
        return ~self._ik_success

    def current_tcp_pose_b(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flange_position_w = self.robot.data.body_pos_w[:, self.flange_body_id]
        flange_quaternion_w = self.robot.data.body_quat_w[:, self.flange_body_id]
        root_position_w = self.robot.data.root_pos_w
        root_quaternion_w = self.robot.data.root_quat_w
        offset_w = rotate_vector(flange_quaternion_w, self.flange_to_tcp_translation_f)
        tcp_position_b = world_to_root_position(flange_position_w + offset_w, root_position_w, root_quaternion_w)
        flange_quaternion_b = quaternion_multiply(quaternion_conjugate(root_quaternion_w), flange_quaternion_w)
        tcp_quaternion_b = quaternion_multiply(flange_quaternion_b, self.flange_to_tool_rotation_f)
        return tcp_position_b, tcp_quaternion_b, flange_quaternion_b

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.raw_action.shape[0], device=self.raw_action.device)
        current = self.robot.data.joint_pos[:, self.joint_ids]
        position_b, quaternion_b, _ = self.current_tcp_pose_b()
        self.raw_action[env_ids] = 0.0
        self.processed_action[env_ids] = 0.0
        self.target_tcp_position_b[env_ids] = position_b[env_ids]
        self.target_tcp_quaternion_b[env_ids] = quaternion_b[env_ids]
        self.policy_start_joint_position[env_ids] = current[env_ids]
        self.ik_seed_joint_position[env_ids] = current[env_ids]
        self.ik_solution_joint_position[env_ids] = current[env_ids]
        self.desired_joint_target[env_ids] = current[env_ids]
        self.commanded_joint_target[env_ids] = current[env_ids]
        self._requested_joint_delta[env_ids] = 0.0
        self._applied_joint_delta[env_ids] = 0.0
        self._trajectory_substep[env_ids] = 0
        self._ik_success[env_ids] = True
        self._ik_iterations[env_ids] = 0
        self._ik_position_error_m[env_ids] = 0.0
        self._ik_orientation_error_rad[env_ids] = 0.0
        self._delta_limited[env_ids] = False
        self._velocity_limited[env_ids] = False
        self._position_limited[env_ids] = False
        self._hard_limit_reached[env_ids] = False
        self._diagnostics = None

    def process_actions(
        self,
        raw_action: torch.Tensor,
        *,
        action_mask: torch.Tensor,
        ik_modes: Sequence[str],
    ) -> None:
        if raw_action.shape != self.raw_action.shape:
            raise ValueError(f"动作形状必须为 {tuple(self.raw_action.shape)}，实际 {tuple(raw_action.shape)}")
        if action_mask.shape != raw_action.shape:
            raise ValueError(f"动作掩码形状必须为 {tuple(raw_action.shape)}，实际 {tuple(action_mask.shape)}")
        if len(ik_modes) != raw_action.shape[0]:
            raise ValueError(f"IK mode 数量必须为 {raw_action.shape[0]}，实际 {len(ik_modes)}")
        invalid_modes = sorted(set(ik_modes) - set(self._solvers))
        if invalid_modes:
            raise ValueError(f"课程给出了未知 IK mode：{invalid_modes}")

        self.raw_action.copy_(raw_action)
        self.processed_action.copy_(
            scale_policy_action(
                raw_action * action_mask,
                action_clip=ACTION_CLIP,
                position_scale_m=POSITION_INCREMENT_SCALE_M,
                rotation_scale_rad=ROTATION_INCREMENT_SCALE_RAD,
            )
        )
        if self.flange_downward_enabled:
            # Flange-down 是动作项配置施加的几何约束；策略旋转增量不再改变姿态目标。
            self.processed_action[:, 3:] = 0.0
        current_position_b, current_quaternion_b, _ = self.current_tcp_pose_b()
        target_position_b, target_quaternion_b = apply_tcp_increment(
            current_position_b, current_quaternion_b, self.processed_action
        )
        self.target_tcp_position_b.copy_(target_position_b)
        self.target_tcp_quaternion_b.copy_(target_quaternion_b)

        current = self.robot.data.joint_pos[:, self.joint_ids]
        self.policy_start_joint_position.copy_(current)
        self.ik_seed_joint_position.copy_(current)
        target_axis_b = rotate_vector(
            target_quaternion_b,
            torch.tensor((0.0, 0.0, 1.0), dtype=target_position_b.dtype, device=target_position_b.device).expand(
                target_position_b.shape[0], -1
            ),
        )
        current_np = self._to_numpy(current)
        target_position_np = self._to_numpy(target_position_b)
        target_quaternion_np = self._to_numpy(target_quaternion_b)
        target_axis_np = self._to_numpy(target_axis_b)
        root_quaternion_np = self._to_numpy(self.robot.data.root_quat_w)
        requested_np = np.zeros_like(current_np)
        solution_np = current_np.copy()
        success = np.zeros(raw_action.shape[0], dtype=np.bool_)
        iterations = np.zeros(raw_action.shape[0], dtype=np.int64)
        position_errors = np.full(raw_action.shape[0], np.inf, dtype=np.float64)
        orientation_errors = np.full(raw_action.shape[0], np.inf, dtype=np.float64)
        reasons: list[str] = []

        resolved_modes: list[str] = []
        for env_id, mode in enumerate(ik_modes):
            kwargs: dict[str, object] = {
                "seed": current_np[env_id],
                "max_iterations": MAX_IK_ITERATIONS,
                "position_tolerance_m": IK_POSITION_TOLERANCE_M,
                "damping": IK_DAMPING,
                "max_joint_step_rad": IK_INTERNAL_MAX_JOINT_STEP_RAD,
            }
            if self.flange_downward_enabled:
                flange_target_position, flange_down_tcp_quaternion = flange_down_tcp_target_pose(
                    target_position_np[env_id],
                    root_quaternion_wxyz=root_quaternion_np[env_id],
                    downward_yaw_offset_rad=self.downward_yaw_offset_rad,
                )
                target_quaternion_np[env_id] = flange_down_tcp_quaternion
                result = solve_flange_down_ik(
                    flange_target_position,
                    root_quaternion_wxyz=root_quaternion_np[env_id],
                    downward_yaw_offset_rad=self.downward_yaw_offset_rad,
                    orientation_tolerance_rad=self.flange_down_orientation_tolerance_rad,
                    **{**kwargs, "position_tolerance_m": self.flange_down_position_tolerance_m},
                )
                resolved_modes.append(_FLANGE_DOWN_IK_MODE)
            elif mode == IKMode.TOOL_AXIS_5D.value:
                kwargs["target_axis_b"] = target_axis_np[env_id]
                kwargs["axis_tolerance_rad"] = IK_AXIS_TOLERANCE_RAD
                result = self._solvers[mode].solve(target_position_np[env_id], **kwargs)
                resolved_modes.append(mode)
            elif mode == IKMode.FULL_POSE_6D.value:
                kwargs["target_quaternion_b_wxyz"] = target_quaternion_np[env_id]
                kwargs["orientation_tolerance_rad"] = IK_ORIENTATION_TOLERANCE_RAD
                result = self._solvers[mode].solve(target_position_np[env_id], **kwargs)
                resolved_modes.append(mode)
            else:
                result = self._solvers[mode].solve(target_position_np[env_id], **kwargs)
                resolved_modes.append(mode)
            success[env_id] = result.success and result.q is not None
            iterations[env_id] = result.iterations
            position_errors[env_id] = result.position_error_m
            orientation_errors[env_id] = result.orientation_error_rad
            reasons.append(result.reason)
            if success[env_id]:
                solution_np[env_id] = result.q
                requested_np[env_id] = result.q - current_np[env_id]

        if self.flange_downward_enabled:
            self.target_tcp_quaternion_b.copy_(
                torch.as_tensor(target_quaternion_np, dtype=current.dtype, device=current.device)
            )

        requested = torch.as_tensor(requested_np, dtype=current.dtype, device=current.device)
        if self.flange_downward_enabled:
            # 对完整 IK 解统一缩放六关节增量，既保持求解方向，又保证每关节不超过安全上限。
            absolute_requested = torch.abs(requested)
            scale_candidates = torch.where(
                absolute_requested > self.max_policy_joint_delta_rad,
                self.max_policy_joint_delta_rad / absolute_requested.clamp_min(torch.finfo(requested.dtype).eps),
                torch.ones_like(requested),
            )
            applied = requested * torch.amin(scale_candidates, dim=-1, keepdim=True)
        else:
            applied = torch.maximum(
                torch.minimum(requested, self.max_policy_joint_delta_rad), -self.max_policy_joint_delta_rad
            )
        unconstrained_desired = current + applied
        limits = self.robot.data.joint_pos_limits[:, self.joint_ids]
        desired = torch.maximum(
            torch.minimum(unconstrained_desired, limits[..., 1] - JOINT_TARGET_MARGIN_RAD),
            limits[..., 0] + JOINT_TARGET_MARGIN_RAD,
        )
        success_tensor = torch.as_tensor(success, dtype=torch.bool, device=current.device)
        desired = torch.where(success_tensor.unsqueeze(-1), desired, current)
        applied = desired - current

        self.desired_joint_target.copy_(desired)
        self.ik_solution_joint_position.copy_(
            torch.as_tensor(solution_np, dtype=current.dtype, device=current.device)
        )
        self.commanded_joint_target.copy_(current)
        self._requested_joint_delta.copy_(requested)
        self._applied_joint_delta.copy_(applied)
        self._ik_success.copy_(success_tensor)
        self._ik_iterations.copy_(torch.as_tensor(iterations, dtype=torch.long, device=current.device))
        self._ik_position_error_m.copy_(torch.as_tensor(position_errors, dtype=current.dtype, device=current.device))
        self._ik_orientation_error_rad.copy_(
            torch.as_tensor(orientation_errors, dtype=current.dtype, device=current.device)
        )
        self._delta_limited.copy_(torch.any(torch.abs(applied - requested) > _DIAGNOSTIC_TOLERANCE_RAD, dim=-1))
        self._velocity_limited.zero_()
        self._position_limited.copy_(
            torch.any(torch.abs(desired - unconstrained_desired) > _DIAGNOSTIC_TOLERANCE_RAD, dim=-1)
        )
        self._hard_limit_reached.zero_()
        self._record_hard_limit_state()
        self._trajectory_substep.zero_()
        self._ik_modes = tuple(resolved_modes)
        self._ik_reasons = tuple(reasons)

    def apply_actions(self) -> torch.Tensor:
        self._record_hard_limit_state()
        self._trajectory_substep.clamp_max_(self.policy_decimation - 1).add_(1)
        ratio = self._trajectory_substep.to(dtype=self.raw_action.dtype).unsqueeze(-1) / float(self.policy_decimation)
        reference_target = self.policy_start_joint_position + self._applied_joint_delta * ratio
        requested_target_step = reference_target - self.commanded_joint_target
        scheduled_step_limit = self.max_policy_joint_delta_rad / float(self.policy_decimation)
        scheduled_step = torch.maximum(
            torch.minimum(requested_target_step, scheduled_step_limit), -scheduled_step_limit
        )
        self._delta_limited |= torch.any(
            torch.abs(scheduled_step - requested_target_step) > _DIAGNOSTIC_TOLERANCE_RAD, dim=-1
        )
        velocity_step_limit = MAX_JOINT_VELOCITY_RAD_S * self.physics_dt_s
        velocity_step = scheduled_step.clamp(-velocity_step_limit, velocity_step_limit)
        self._velocity_limited |= torch.any(
            torch.abs(velocity_step - scheduled_step) > _DIAGNOSTIC_TOLERANCE_RAD, dim=-1
        )
        limits = self.robot.data.joint_pos_limits[:, self.joint_ids]
        lower = torch.minimum(limits[..., 0] + JOINT_TARGET_MARGIN_RAD, self.commanded_joint_target)
        upper = torch.maximum(limits[..., 1] - JOINT_TARGET_MARGIN_RAD, self.commanded_joint_target)
        unconstrained_target = self.commanded_joint_target + velocity_step
        joint_target = torch.maximum(torch.minimum(unconstrained_target, upper), lower)
        self._position_limited |= torch.any(
            torch.abs(joint_target - unconstrained_target) > _DIAGNOSTIC_TOLERANCE_RAD, dim=-1
        )
        self.commanded_joint_target.copy_(joint_target)
        self.robot.set_joint_position_target(joint_target, joint_ids=self.joint_ids)
        if self.diagnostics_enabled:
            self._diagnostics = AuboIKDiagnostics(
                raw_action=self.raw_action.clone(),
                processed_action=self.processed_action.clone(),
                target_tcp_position_b=self.target_tcp_position_b.clone(),
                target_tcp_quaternion_b=self.target_tcp_quaternion_b.clone(),
                ik_mode=self._ik_modes,
                ik_success=self._ik_success.clone(),
                ik_reason=self._ik_reasons,
                ik_iterations=self._ik_iterations.clone(),
                ik_position_error_m=self._ik_position_error_m.clone(),
                ik_orientation_error_rad=self._ik_orientation_error_rad.clone(),
                ik_seed_joint_position=self.ik_seed_joint_position.clone(),
                ik_solution_joint_position=self.ik_solution_joint_position.clone(),
                requested_joint_delta=self._requested_joint_delta.clone(),
                applied_joint_delta=self._applied_joint_delta.clone(),
                delta_limited=self._delta_limited.clone(),
                velocity_limited=self._velocity_limited.clone(),
                position_limited=self._position_limited.clone(),
                desired_joint_target=self.desired_joint_target.clone(),
                joint_target=joint_target.clone(),
            )
        return joint_target

    def _record_hard_limit_state(self) -> None:
        current = self.robot.data.joint_pos[:, self.joint_ids]
        limits = self.robot.data.joint_pos_limits[:, self.joint_ids]
        clearance = torch.minimum(current - limits[..., 0], limits[..., 1] - current)
        self._hard_limit_reached |= torch.any(
            clearance <= HARD_LIMIT_TERMINATION_MARGIN_RAD,
            dim=-1,
        )

    @staticmethod
    def _to_numpy(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().numpy().astype(np.float64, copy=True)

