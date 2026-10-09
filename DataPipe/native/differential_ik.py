"""不依赖 Isaac API 的六维增量位姿和阻尼最小二乘求解。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .tcp_kinematics import (
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_multiply,
    rotate_vector,
)


@dataclass(frozen=True)
class DlsResult:
    joint_delta: torch.Tensor
    minimum_singular_value: torch.Tensor
    damping: torch.Tensor
    singular: torch.Tensor


@dataclass(frozen=True)
class JointProjectionResult:
    joint_target: torch.Tensor
    joint_delta: torch.Tensor
    delta_limited: torch.Tensor
    velocity_limited: torch.Tensor
    position_limited: torch.Tensor


def scale_policy_action(
    raw_action: torch.Tensor,
    *,
    action_clip: float,
    position_scale_m: float,
    rotation_scale_rad: float,
) -> torch.Tensor:
    """裁剪归一化六维动作，并分别缩放平移和旋转向量。"""

    if raw_action.ndim != 2 or raw_action.shape[-1] != 6:
        raise ValueError(f"raw_action 必须为 (env, 6)，实际 {tuple(raw_action.shape)}")
    if not torch.isfinite(raw_action).all():
        raise ValueError("raw_action 包含 NaN/Inf")
    clipped = raw_action.clamp(-action_clip, action_clip)
    scale = raw_action.new_tensor(
        [
            position_scale_m,
            position_scale_m,
            position_scale_m,
            rotation_scale_rad,
            rotation_scale_rad,
            rotation_scale_rad,
        ]
    )
    return clipped * scale


def rotation_vector_to_quaternion(rotation_vector: torch.Tensor) -> torch.Tensor:
    """将旋转向量转换为 ``wxyz`` 四元数，小角度处保持有限梯度。"""

    if rotation_vector.shape[-1:] != (3,):
        raise ValueError("rotation_vector 最后一维必须为 3")
    angle = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    eps = torch.finfo(rotation_vector.dtype).eps
    scale = torch.where(angle > eps, torch.sin(half_angle) / angle, 0.5 - angle.square() / 48.0)
    return normalize_quaternion(torch.cat((torch.cos(half_angle), rotation_vector * scale), dim=-1))


def quaternion_to_rotation_vector(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """将 ``wxyz`` 四元数转换为最短路径旋转向量。"""

    quaternion_wxyz = normalize_quaternion(quaternion_wxyz)
    quaternion_wxyz = torch.where(quaternion_wxyz[..., :1] < 0.0, -quaternion_wxyz, quaternion_wxyz)
    real = quaternion_wxyz[..., :1].clamp(-1.0, 1.0)
    imaginary = quaternion_wxyz[..., 1:]
    sin_half = torch.linalg.vector_norm(imaginary, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, real)
    eps = torch.finfo(quaternion_wxyz.dtype).eps
    scale = torch.where(sin_half > eps, angle / sin_half, 2.0 + sin_half.square() / 3.0)
    return imaginary * scale


def apply_tcp_increment(
    position_b: torch.Tensor, quaternion_b_wxyz: torch.Tensor, scaled_action_b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """在根坐标系 B 中平移，并用 B 系旋转向量左乘当前 TCP 姿态。"""

    target_position_b = position_b + scaled_action_b[..., :3]
    delta_quaternion_b = rotation_vector_to_quaternion(scaled_action_b[..., 3:])
    target_quaternion_b = quaternion_multiply(delta_quaternion_b, quaternion_b_wxyz)
    return target_position_b, target_quaternion_b


def pose_error(
    current_position_b: torch.Tensor,
    current_quaternion_b_wxyz: torch.Tensor,
    target_position_b: torch.Tensor,
    target_quaternion_b_wxyz: torch.Tensor,
    *,
    position_gain: float,
    rotation_gain: float,
) -> torch.Tensor:
    """返回 B 系六维位置/最短旋转向量误差。"""

    position_error = (target_position_b - current_position_b) * position_gain
    relative_quaternion = quaternion_multiply(target_quaternion_b_wxyz, quaternion_conjugate(current_quaternion_b_wxyz))
    rotation_error = quaternion_to_rotation_vector(relative_quaternion) * rotation_gain
    return torch.cat((position_error, rotation_error), dim=-1)


def tcp_jacobian_from_flange(
    flange_jacobian_b: torch.Tensor,
    flange_quaternion_b_wxyz: torch.Tensor,
    flange_to_tcp_translation_f: torch.Tensor,
) -> torch.Tensor:
    """将法兰 Jacobian 平移到 TCP；输入/输出均表达在根坐标系 B。"""

    if flange_jacobian_b.ndim != 3 or flange_jacobian_b.shape[-2] != 6:
        raise ValueError(f"flange_jacobian_b 必须为 (env, 6, joint)，实际 {tuple(flange_jacobian_b.shape)}")
    offset_b = rotate_vector(flange_quaternion_b_wxyz, flange_to_tcp_translation_f)
    angular = flange_jacobian_b[:, 3:, :]
    tangential = torch.linalg.cross(
        angular.transpose(1, 2), offset_b.unsqueeze(1).expand(-1, angular.shape[-1], -1), dim=-1
    ).transpose(1, 2)
    return torch.cat((flange_jacobian_b[:, :3, :] + tangential, angular), dim=1)


def damped_least_squares(
    jacobian: torch.Tensor,
    task_error: torch.Tensor,
    *,
    damping: float,
    singular_value_threshold: float,
    singular_damping: float,
) -> DlsResult:
    """以 ``Jᵀ(JJᵀ+λ²I)⁻¹e`` 求解，并在近奇异时提高阻尼。"""

    if jacobian.ndim != 3 or jacobian.shape[1] != 6 or task_error.shape != jacobian.shape[:2]:
        raise ValueError(
            f"期望 jacobian=(env,6,joint)、task_error=(env,6)，实际 {tuple(jacobian.shape)} / {tuple(task_error.shape)}"
        )
    if not torch.isfinite(jacobian).all() or not torch.isfinite(task_error).all():
        raise ValueError("Jacobian 或任务误差包含 NaN/Inf")
    singular_values = torch.linalg.svdvals(jacobian)
    minimum_singular_value = singular_values[..., -1]
    singular = minimum_singular_value < singular_value_threshold
    damping_tensor = torch.where(
        singular,
        minimum_singular_value.new_full(minimum_singular_value.shape, singular_damping),
        minimum_singular_value.new_full(minimum_singular_value.shape, damping),
    )
    identity = torch.eye(6, dtype=jacobian.dtype, device=jacobian.device).expand(jacobian.shape[0], -1, -1)
    system = jacobian @ jacobian.transpose(1, 2) + damping_tensor[:, None, None].square() * identity
    joint_delta = jacobian.transpose(1, 2) @ torch.linalg.solve(system, task_error.unsqueeze(-1))
    joint_delta = joint_delta.squeeze(-1)
    if not torch.isfinite(joint_delta).all():
        raise ValueError("DLS 关节增量包含 NaN/Inf")
    return DlsResult(joint_delta, minimum_singular_value, damping_tensor, singular)


def project_joint_target(
    current_joint_position: torch.Tensor,
    requested_joint_delta: torch.Tensor,
    joint_position_limits: torch.Tensor,
    joint_velocity_limits: torch.Tensor,
    *,
    physics_dt_s: float,
    max_joint_delta_rad: float,
    max_joint_velocity_rad_s: float,
    joint_position_margin_rad: float,
) -> JointProjectionResult:
    """依次执行单步、速度和位置限幅，返回逐环境保护标志。"""

    if current_joint_position.shape != requested_joint_delta.shape:
        raise ValueError("current_joint_position 与 requested_joint_delta 形状必须一致")
    if joint_position_limits.shape != (*current_joint_position.shape, 2):
        raise ValueError("joint_position_limits 形状必须为 (env, joint, 2)")
    requested = requested_joint_delta
    step_limited_delta = requested.clamp(-max_joint_delta_rad, max_joint_delta_rad)
    delta_limited = torch.any(step_limited_delta != requested, dim=-1)
    effective_velocity_limit = torch.minimum(
        joint_velocity_limits.abs(), joint_velocity_limits.new_full((), max_joint_velocity_rad_s)
    )
    velocity_delta_limit = effective_velocity_limit * physics_dt_s
    velocity_limited_delta = torch.maximum(
        torch.minimum(step_limited_delta, velocity_delta_limit), -velocity_delta_limit
    )
    velocity_limited = torch.any(velocity_limited_delta != step_limited_delta, dim=-1)
    lower = joint_position_limits[..., 0] + joint_position_margin_rad
    upper = joint_position_limits[..., 1] - joint_position_margin_rad
    if torch.any(lower > upper):
        raise ValueError("关节位置限位区间小于两倍安全 margin")
    unconstrained_target = current_joint_position + velocity_limited_delta
    joint_target = torch.maximum(torch.minimum(unconstrained_target, upper), lower)
    position_limited = torch.any(joint_target != unconstrained_target, dim=-1)
    joint_delta = joint_target - current_joint_position
    if not torch.isfinite(joint_target).all():
        raise ValueError("最终关节目标包含 NaN/Inf")
    return JointProjectionResult(
        joint_target=joint_target,
        joint_delta=joint_delta,
        delta_limited=delta_limited,
        velocity_limited=velocity_limited,
        position_limited=position_limited,
    )
