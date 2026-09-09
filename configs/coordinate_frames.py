"""机器人任务中的坐标系、方向轴与固定/运行时变换契约。

命名规则
========

``W``：Isaac 世界坐标系（World）。
``B``：机械臂根/基座坐标系（Base）。
``F``：机械臂末端法兰坐标系（Flange）。
``T``：IK 控制器根据固定 q_FT 计算的 TCP 坐标系（不是 USD 子 prim 坐标系）。
``C``：USD 中 ClawTool_C/ClawTool_C 子 prim 的局部坐标系。
以上四个是本任务重点检查的坐标系。另用 ``G`` 标记目标物配置中的局部轴
（Goal/Target object local）；它只用于把配置方向转到 W，不属于机器人
``W->B->F->T`` 运动学链。

后缀表示“向量在哪个坐标系中表达”，例如 ``direction_w`` 是在 W 系
表达的方向；四元数 ``q_wb`` 表示 B 系在 W 系中的姿态，可把 B 系向量
旋转到 W 系。四元数顺序统一为 ``wxyz``，旋转使用 Hamilton 乘法。

本文件只描述几何语义，不依赖 Isaac、Torch 或 NumPy，方便独立检查和修改。
"""

from __future__ import annotations

from collections.abc import Sequence
from math import acos, isfinite, sqrt
from typing import Final

Vector3 = tuple[float, float, float]
QuaternionWxyz = tuple[float, float, float, float]

# -----------------------------------------------------------------------------
# 坐标系标识
# -----------------------------------------------------------------------------

WORLD_FRAME: Final = "W"
ROBOT_BASE_FRAME: Final = "B"
FLANGE_FRAME: Final = "F"
TCP_FRAME: Final = "T"
TARGET_FRAME: Final = "G"

# -----------------------------------------------------------------------------
# 三类任务方向
# -----------------------------------------------------------------------------

# 世界重力向下方向，在世界坐标系 W 中表达。
WORLD_DOWN_DIRECTION_W: Final[Vector3] = (0.0, 0.0, -1.0)

# 当前 AUBO USD/URDF 代码约定的 Flange 工作面法向，在 F 系中表达。
# CurriculumRL.logic.aubo_ik.FLANGE_DOWNWARD_AXIS_LOCAL 当前也是 (-1, 0, 0)。
FLANGE_NORMAL_AXIS_F: Final[Vector3] = (-1.0, 0.0, 0.0)

# 物理朝向由子 prim +Y_C 定义。USD q_FC 将 +Y_C 映射到 +Z_F，
# IK q_FT 则将 -X_T 映射到 +Z_F，故控制与判定必须使用 -X_T。
# 不能把子 prim 的 (0,1,0) 直接作为 IK TCP 的局部轴。
CLAW_DIRECTION_AXIS_C: Final[Vector3] = (0.0, 1.0, 0.0)
CLAW_CHILD_PRIM_RELATIVE_PATH: Final = "ClawTool_C/ClawTool_C"
FLANGE_TO_CLAW_ROTATION_F_WXYZ: Final[QuaternionWxyz] = (0.5, 0.5, 0.5, 0.5)
TCP_DIRECTION_AXIS_T: Final[Vector3] = (-1.0, 0.0, 0.0)

# 旧任务中用于 tool-axis reward/5D IK 的工具前向轴。保留并明确其语义，
# 禁止在 ALIGN_PLACE_DIRECTION 中与 TCP_DIRECTION_AXIS_T 静默互换。
TCP_TOOL_FORWARD_AXIS_T: Final[Vector3] = (0.0, 0.0, 1.0)

# 抓取目标要求的停靠方向轴，在目标物自身坐标系 G 中表达。
TARGET_DOCKING_AXIS_G: Final[Vector3] = (0.0, -1.0, 0.0)

# -----------------------------------------------------------------------------
# 两个固定变换：F -> T 与 G -> T
# -----------------------------------------------------------------------------

# TCP 原点相对 Flange 原点的位置，在 F 系中表达，单位 m：p_FT^F。
FLANGE_TO_TCP_TRANSLATION_F: Final[Vector3] = (-0.17086, 0.0, 0.11305)

# TCP 坐标系 T 在 Flange 坐标系 F 中的姿态 q_FT；把 T 系方向旋转到 F 系。
FLANGE_TO_TCP_ROTATION_F_WXYZ: Final[QuaternionWxyz] = (sqrt(0.5), 0.0, sqrt(0.5), 0.0)

# 旧 full-pose/reset 逻辑使用的 TCP 坐标系 T 在目标物坐标系 G 中的姿态 q_GT；
# 它把 +Z_T 映射到 -Y_G，不能用作子 prim 朝向的标定。
# ALIGN_PLACE_DIRECTION 比较 +Y_C（即 -X_T）的世界表达与目标 -Y_G 的世界表达。
TARGET_TO_TCP_ROTATION_G_WXYZ: Final[QuaternionWxyz] = (sqrt(0.5), sqrt(0.5), 0.0, 0.0)

IDENTITY_QUATERNION_WXYZ: Final[QuaternionWxyz] = (1.0, 0.0, 0.0, 0.0)


def _vector3(values: Sequence[float], name: str) -> Vector3:
    result = tuple(float(value) for value in values)
    if len(result) != 3 or not all(isfinite(value) for value in result):
        raise ValueError(f"{name} must be a finite vector of length 3")
    return result  # type: ignore[return-value]


def normalize_direction(values: Sequence[float], name: str = "direction") -> Vector3:
    """返回单位方向；方向不包含坐标原点，因此不应用平移。"""

    vector = _vector3(values, name)
    magnitude = sqrt(sum(value * value for value in vector))
    if magnitude <= 1.0e-9:
        raise ValueError(f"{name} must have non-zero length")
    return tuple(value / magnitude for value in vector)  # type: ignore[return-value]


def normalize_quaternion_wxyz(values: Sequence[float], name: str = "quaternion_wxyz") -> QuaternionWxyz:
    quaternion = tuple(float(value) for value in values)
    if len(quaternion) != 4 or not all(isfinite(value) for value in quaternion):
        raise ValueError(f"{name} must be a finite quaternion in wxyz order")
    magnitude = sqrt(sum(value * value for value in quaternion))
    if magnitude <= 1.0e-9:
        raise ValueError(f"{name} must have non-zero norm")
    return tuple(value / magnitude for value in quaternion)  # type: ignore[return-value]


def quaternion_conjugate_wxyz(quaternion_wxyz: Sequence[float]) -> QuaternionWxyz:
    w, x, y, z = normalize_quaternion_wxyz(quaternion_wxyz)
    return (w, -x, -y, -z)


def quaternion_multiply_wxyz(left_wxyz: Sequence[float], right_wxyz: Sequence[float]) -> QuaternionWxyz:
    """Hamilton 乘积；``q_ac = q_ab * q_bc``。"""

    lw, lx, ly, lz = normalize_quaternion_wxyz(left_wxyz, "left_wxyz")
    rw, rx, ry, rz = normalize_quaternion_wxyz(right_wxyz, "right_wxyz")
    return normalize_quaternion_wxyz(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def rotate_vector_wxyz(quaternion_parent_child_wxyz: Sequence[float], vector_child: Sequence[float]) -> Vector3:
    """用 ``q_parent_child`` 把 child 系自由向量旋转到 parent 系，不改变长度。"""

    w, x, y, z = normalize_quaternion_wxyz(quaternion_parent_child_wxyz)
    vx, vy, vz = _vector3(vector_child, "vector_child")
    tx, ty, tz = 2.0 * (y * vz - z * vy), 2.0 * (z * vx - x * vz), 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    )


def rotate_direction_wxyz(quaternion_parent_child_wxyz: Sequence[float], direction_child: Sequence[float]) -> Vector3:
    """用 ``q_parent_child`` 把 child 系单位方向旋转到 parent 系。"""

    return normalize_direction(rotate_vector_wxyz(quaternion_parent_child_wxyz, direction_child))


# -----------------------------------------------------------------------------
# 运行时变换一：世界坐标系 W <-> 机械臂 Base 坐标系 B
# -----------------------------------------------------------------------------


def base_to_world_direction(direction_b: Sequence[float], q_wb_wxyz: Sequence[float]) -> Vector3:
    """``d_W = R_WB d_B``；方向变换不使用 Base 的世界位置。"""

    return rotate_direction_wxyz(q_wb_wxyz, direction_b)


def world_to_base_direction(direction_w: Sequence[float], q_wb_wxyz: Sequence[float]) -> Vector3:
    """``d_B = R_WB.T d_W``；是 :func:`base_to_world_direction` 的逆。"""

    return rotate_direction_wxyz(quaternion_conjugate_wxyz(q_wb_wxyz), direction_w)


def world_to_base_position(
    position_w: Sequence[float], position_wb_w: Sequence[float], q_wb_wxyz: Sequence[float]
) -> Vector3:
    """``p_B = R_WB.T (p_W - p_WB^W)``；位置变换包含 Base 原点平移。"""

    point = _vector3(position_w, "position_w")
    origin = _vector3(position_wb_w, "position_wb_w")
    return rotate_vector_wxyz(
        quaternion_conjugate_wxyz(q_wb_wxyz),
        tuple(value - base for value, base in zip(point, origin, strict=True)),
    )


def base_to_world_position(
    position_b: Sequence[float], position_wb_w: Sequence[float], q_wb_wxyz: Sequence[float]
) -> Vector3:
    """``p_W = p_WB^W + R_WB p_B``；是 :func:`world_to_base_position` 的逆。"""

    point_w = rotate_vector_wxyz(q_wb_wxyz, _vector3(position_b, "position_b"))
    origin_w = _vector3(position_wb_w, "position_wb_w")
    return tuple(point + origin for point, origin in zip(point_w, origin_w, strict=True))  # type: ignore[return-value]


# -----------------------------------------------------------------------------
# 运行时变换二：机械臂 Base 坐标系 B -> Flange 坐标系 F
# -----------------------------------------------------------------------------


def flange_pose_in_world(
    position_wb_w: Sequence[float],
    q_wb_wxyz: Sequence[float],
    position_bf_b: Sequence[float],
    q_bf_wxyz: Sequence[float],
) -> tuple[Vector3, QuaternionWxyz]:
    """由机械臂正运动学给出的 B->F 位姿，组合得到 Flange 世界位姿。

    ``p_BF^B`` 与 ``q_BF`` 不是固定标定量，而是随关节角变化的 FK 输出：
    ``p_WF^W = p_WB^W + R_WB p_BF^B``，``q_WF = q_WB * q_BF``。
    """

    return (
        base_to_world_position(position_bf_b, position_wb_w, q_wb_wxyz),
        quaternion_multiply_wxyz(q_wb_wxyz, q_bf_wxyz),
    )


# -----------------------------------------------------------------------------
# 固定变换三：Flange 坐标系 F -> TCP 坐标系 T
# -----------------------------------------------------------------------------


def tcp_quaternion_in_world(q_wb_wxyz: Sequence[float], q_bt_wxyz: Sequence[float]) -> QuaternionWxyz:
    """组合 B->T 与 W->B：``q_WT = q_WB * q_BT``。"""

    return quaternion_multiply_wxyz(q_wb_wxyz, q_bt_wxyz)


def flange_to_tcp_pose_world(
    position_wf_w: Sequence[float],
    q_wf_wxyz: Sequence[float],
    translation_ft_f: Sequence[float] = FLANGE_TO_TCP_TRANSLATION_F,
    q_ft_wxyz: Sequence[float] = FLANGE_TO_TCP_ROTATION_F_WXYZ,
) -> tuple[Vector3, QuaternionWxyz]:
    """由 Flange 世界位姿得到 TCP 世界位姿。

    ``p_WT = p_WF + R_WF p_FT^F``，``q_WT = q_WF * q_FT``。
    """

    flange_position = _vector3(position_wf_w, "position_wf_w")
    offset_w = rotate_vector_wxyz(q_wf_wxyz, translation_ft_f)
    tcp_position = tuple(a + b for a, b in zip(flange_position, offset_w, strict=True))
    return tcp_position, quaternion_multiply_wxyz(q_wf_wxyz, q_ft_wxyz)


def tcp_pose_in_world_from_base_flange(
    position_wb_w: Sequence[float],
    q_wb_wxyz: Sequence[float],
    position_bf_b: Sequence[float],
    q_bf_wxyz: Sequence[float],
    translation_ft_f: Sequence[float] = FLANGE_TO_TCP_TRANSLATION_F,
    q_ft_wxyz: Sequence[float] = FLANGE_TO_TCP_ROTATION_F_WXYZ,
) -> tuple[Vector3, QuaternionWxyz]:
    """按 ``W->B->F->T`` 完整链路由 FK 输出计算 TCP 世界位姿。"""

    position_wf_w, q_wf_wxyz = flange_pose_in_world(
        position_wb_w,
        q_wb_wxyz,
        position_bf_b,
        q_bf_wxyz,
    )
    return flange_to_tcp_pose_world(position_wf_w, q_wf_wxyz, translation_ft_f, q_ft_wxyz)


def tcp_direction_in_world(
    q_wb_wxyz: Sequence[float],
    q_bt_wxyz: Sequence[float],
    direction_axis_t: Sequence[float] = TCP_DIRECTION_AXIS_T,
) -> Vector3:
    """把 TCP 的朝向轴从 T 系经 B 系转换到世界系 W。"""

    return rotate_direction_wxyz(tcp_quaternion_in_world(q_wb_wxyz, q_bt_wxyz), direction_axis_t)


def target_direction_in_world(
    q_wg_wxyz: Sequence[float],
    docking_axis_g: Sequence[float] = TARGET_DOCKING_AXIS_G,
) -> Vector3:
    """由目标物世界姿态计算抓取/放置目标方向 ``d_W = R_WG d_G``。"""

    return rotate_direction_wxyz(q_wg_wxyz, docking_axis_g)


def direction_error_rad(actual_direction_w: Sequence[float], target_direction_w: Sequence[float]) -> float:
    """返回两个世界系单位方向之间的无符号夹角，范围 ``[0, pi]``。"""

    actual = normalize_direction(actual_direction_w, "actual_direction_w")
    target = normalize_direction(target_direction_w, "target_direction_w")
    cosine = max(-1.0, min(1.0, sum(a * b for a, b in zip(actual, target, strict=True))))
    return acos(cosine)


__all__ = [
    "CLAW_DIRECTION_AXIS_C",
    "CLAW_CHILD_PRIM_RELATIVE_PATH",
    "FLANGE_TO_CLAW_ROTATION_F_WXYZ",
    "FLANGE_FRAME",
    "FLANGE_NORMAL_AXIS_F",
    "FLANGE_TO_TCP_ROTATION_F_WXYZ",
    "FLANGE_TO_TCP_TRANSLATION_F",
    "IDENTITY_QUATERNION_WXYZ",
    "QuaternionWxyz",
    "ROBOT_BASE_FRAME",
    "TARGET_DOCKING_AXIS_G",
    "TARGET_FRAME",
    "TARGET_TO_TCP_ROTATION_G_WXYZ",
    "TCP_DIRECTION_AXIS_T",
    "TCP_FRAME",
    "TCP_TOOL_FORWARD_AXIS_T",
    "Vector3",
    "WORLD_DOWN_DIRECTION_W",
    "WORLD_FRAME",
    "base_to_world_direction",
    "base_to_world_position",
    "direction_error_rad",
    "flange_pose_in_world",
    "flange_to_tcp_pose_world",
    "normalize_direction",
    "normalize_quaternion_wxyz",
    "quaternion_conjugate_wxyz",
    "quaternion_multiply_wxyz",
    "rotate_direction_wxyz",
    "rotate_vector_wxyz",
    "target_direction_in_world",
    "tcp_direction_in_world",
    "tcp_pose_in_world_from_base_flange",
    "tcp_quaternion_in_world",
    "world_to_base_direction",
    "world_to_base_position",
]
