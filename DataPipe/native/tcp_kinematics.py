"""不依赖 Isaac API 的 TCP 位姿、速度和坐标变换。"""

from __future__ import annotations

import torch


def _require_last_dim(tensor: torch.Tensor, size: int, name: str) -> None:
    if tensor.shape[-1:] != (size,):
        raise ValueError(f"{name} 最后一维必须为 {size}，实际为 {tuple(tensor.shape)}")


def normalize_quaternion(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """归一化 ``wxyz`` 四元数，并拒绝零范数输入。"""

    _require_last_dim(quaternion_wxyz, 4, "quaternion_wxyz")
    norm = torch.linalg.vector_norm(quaternion_wxyz, dim=-1, keepdim=True)
    if torch.any(norm <= torch.finfo(quaternion_wxyz.dtype).eps):
        raise ValueError("四元数范数必须大于零")
    return quaternion_wxyz / norm


def quaternion_conjugate(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    quaternion_wxyz = normalize_quaternion(quaternion_wxyz)
    return torch.cat((quaternion_wxyz[..., :1], -quaternion_wxyz[..., 1:]), dim=-1)


def quaternion_multiply(left_wxyz: torch.Tensor, right_wxyz: torch.Tensor) -> torch.Tensor:
    """计算可广播的 Hamilton 乘积 ``left * right``。"""

    left_wxyz = normalize_quaternion(left_wxyz)
    right_wxyz = normalize_quaternion(right_wxyz)
    left_wxyz, right_wxyz = torch.broadcast_tensors(left_wxyz, right_wxyz)
    lw, lx, ly, lz = left_wxyz.unbind(dim=-1)
    rw, rx, ry, rz = right_wxyz.unbind(dim=-1)
    result = torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )
    return normalize_quaternion(result)


def rotate_vector(quaternion_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """用 ``wxyz`` 四元数旋转三维向量。"""

    quaternion_wxyz = normalize_quaternion(quaternion_wxyz)
    _require_last_dim(vector, 3, "vector")
    quaternion_wxyz, vector = torch.broadcast_tensors(quaternion_wxyz, torch.cat((vector, vector[..., :1]), -1))
    vector = vector[..., :3]
    real = quaternion_wxyz[..., :1]
    imaginary = quaternion_wxyz[..., 1:]
    return vector + 2.0 * (
        real * torch.linalg.cross(imaginary, vector, dim=-1)
        + torch.linalg.cross(imaginary, torch.linalg.cross(imaginary, vector, dim=-1), dim=-1)
    )


def flange_to_tcp_state(
    flange_position_w: torch.Tensor,
    flange_quaternion_wxyz: torch.Tensor,
    flange_linear_velocity_w: torch.Tensor,
    flange_angular_velocity_w: torch.Tensor,
    flange_to_tcp_translation_f: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算 TCP 世界位置及含 ``omega × r`` 切向项的世界线速度。"""

    offset_w = rotate_vector(flange_quaternion_wxyz, flange_to_tcp_translation_f)
    tcp_position_w = flange_position_w + offset_w
    tcp_linear_velocity_w = flange_linear_velocity_w + torch.linalg.cross(flange_angular_velocity_w, offset_w, dim=-1)
    return tcp_position_w, tcp_linear_velocity_w


def world_to_root_position(
    position_w: torch.Tensor, root_position_w: torch.Tensor, root_quaternion_wxyz: torch.Tensor
) -> torch.Tensor:
    """将世界位置转换到机器人根坐标系。"""

    return rotate_vector(quaternion_conjugate(root_quaternion_wxyz), position_w - root_position_w)


def world_to_root_vector(vector_w: torch.Tensor, root_quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """将世界自由向量转换到机器人根坐标系，不引入环境原点平移。"""

    return rotate_vector(quaternion_conjugate(root_quaternion_wxyz), vector_w)
