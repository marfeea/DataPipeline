"""Current AUBO E5 kinematics and NumPy inverse-kinematics adapter.

The read-only ``reference`` solver uses a Unity DH convention that does not
match the AUBO asset currently loaded by this project.  This module keeps the
reference idea (DH forward kinematics plus a downward-pose convenience
solver), but derives the active chain from ``Asset/AUBOE5_URDF/AUBOE5.urdf``.

The six DH rows describe active joints only.  ``Flange`` is represented by a
fixed transform after the DH chain; the project TCP is a second fixed
transform after ``Flange``.  Consequently the no-TCP and TCP models share the
same joint geometry while exposing different endpoint poses and Jacobians.
Lengths are metres, angles are radians, and quaternions use ``wxyz``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from math import pi
from typing import Final, Protocol

import numpy as np

from configs.task import FLANGE_TO_TCP_TRANSLATION_F, FLANGE_TO_TOOL_ROTATION_F, TCP_HEADING_AXIS_LOCAL

# =============================================================================
# 第一块：公共约定与模型常量
# =============================================================================
#
# 这里集中声明本模块所有外部可见的坐标约定，检查 IK 时建议先从这一块开始：
#
# 1. B 系是 AUBO 机器人根坐标系，所有 FK/IK 位置和姿态都在 B 系表达。
# 2. 关节角顺序固定为 Joint1、Joint2、Joint3、Joint4、Joint5、Flange。
# 3. 四元数顺序固定为 wxyz；长度单位为 m；角度单位为 rad。
# 4. DH 表只描述 6 个主动关节，Flange 和 TCP 都通过末端固定变换表达。
# 5. ``solve_ik`` 默认只服从调用者给出的目标位姿，不会自动猜测“末端向下”。

JOINT_COUNT: Final = 6
WORLD_DOWN_AXIS_W: Final = np.array((0.0, 0.0, -1.0), dtype=np.float64)
# 当前 AUBO USD 中，Flange 朝向面的法向对应局部 -X，而不是 +Y/+Z。
# 初始姿态下局部 +X 已指向世界 +Z，因此法向 -X 指向世界 -Z。
FLANGE_DOWNWARD_AXIS_LOCAL: Final = np.array((-1.0, 0.0, 0.0), dtype=np.float64)
JOINT_LIMITS_RAD: Final = np.array(
    [(-3.0543262, 3.0543262)] * JOINT_COUNT,
    dtype=np.float64,
)

# =============================================================================
# 第二块：当前 USD/URDF 对应的 DH 参数
# =============================================================================
#
# 每一行的顺序是 [a, d, alpha, theta_offset]。这不是 reference 目录中
# Unity 求解器的原始参数，而是根据当前 AUBOE5.urdf 的 z-x-x-x-z-x
# 关节轴链重新适配的参数。
#
# ``AUBO_E5_DH_NO_TCP`` 和 ``AUBO_E5_DH_WITH_TCP`` 目前关节几何相同；
# 两者的区别在于后续 endpoint 固定变换不同，而不是再增加一个“虚拟关节”。
AUBO_E5_DH_NO_TCP: Final = np.array(
    [
        [0.0, 0.1425132, pi / 2.0, pi / 2.0],
        [0.4079251, -0.1203119, 0.0, pi / 2.0],
        [0.3757994, 0.0, 0.0, 0.0],
        [0.0, 0.0, pi / 2.0, pi / 2.0],
        [0.0, 0.1025002, -pi / 2.0, 0.0],
        [0.0, -0.0540166, 0.0, 0.0],
    ],
    dtype=np.float64,
)
AUBO_E5_DH_WITH_TCP: Final = AUBO_E5_DH_NO_TCP.copy()
_BASE_TO_DH_TRANSLATION_B: Final = np.array((0.0000784, 0.0000151, 0.0), dtype=np.float64)

# Frame6 是 DH 链的辅助末端坐标系。该固定旋转把 Frame6 映射到当前
# URDF/USD 的 Flange 坐标系，校准点为零关节角。
_FRAME6_TO_FLANGE_ROTATION: Final = np.array(
    ((0.0, -1.0, 0.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0)),
    dtype=np.float64,
)


# =============================================================================
# 第三块：数值输入、旋转和齐次变换工具
# =============================================================================


def _readonly(array: np.ndarray) -> np.ndarray:
    """把模型常量设为只读，避免求解过程中被意外修改。"""

    array.setflags(write=False)
    return array


_readonly(JOINT_LIMITS_RAD)
_readonly(WORLD_DOWN_AXIS_W)
_readonly(FLANGE_DOWNWARD_AXIS_LOCAL)
_readonly(AUBO_E5_DH_NO_TCP)
_readonly(AUBO_E5_DH_WITH_TCP)
_readonly(_BASE_TO_DH_TRANSLATION_B)
_readonly(_FRAME6_TO_FLANGE_ROTATION)


def _quaternion_to_rotation_wxyz(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    """将 wxyz 四元数转换为旋转矩阵。"""

    q = _vector(quaternion_wxyz, "quaternion_wxyz", 4)
    norm = np.linalg.norm(q)
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("quaternion_wxyz must not be zero")
    w, x, y, z = q / norm
    return np.array(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _rotation_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    """将 3x3 旋转矩阵转换为 wxyz 四元数。"""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {matrix.shape}")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = 2.0 * np.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0))
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0))
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = 2.0 * np.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0))
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float64,
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return quaternion


def _homogeneous(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    """构造齐次变换矩阵。"""

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = _vector(translation, "translation", 3)
    return transform


def _dh_transform(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    """按本模块的标准 DH 行参数构造单节变换。"""

    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        ((ct, -st * ca, st * sa, a * ct), (st, ct * ca, -ct * sa, a * st), (0.0, sa, ca, d), (0.0, 0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def _vector(value: Sequence[float], name: str, size: int) -> np.ndarray:
    """统一检查一维有限向量，避免 NaN/Inf 进入 IK。"""

    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


# =============================================================================
# 第四块：Flange/TCP endpoint 模型
# =============================================================================


@dataclass(frozen=True)
class AuboKinematicModel:
    """一条主动关节链和一个固定 endpoint 变换。"""

    name: str
    dh: np.ndarray
    frame6_to_endpoint: np.ndarray

    def __post_init__(self) -> None:
        dh = np.asarray(self.dh, dtype=np.float64)
        endpoint = np.asarray(self.frame6_to_endpoint, dtype=np.float64)
        if dh.shape != (JOINT_COUNT, 4):
            raise ValueError(f"dh must have shape (6, 4), got {dh.shape}")
        if endpoint.shape != (4, 4):
            raise ValueError(f"frame6_to_endpoint must have shape (4, 4), got {endpoint.shape}")
        if not np.all(np.isfinite(dh)) or not np.all(np.isfinite(endpoint)):
            raise ValueError("kinematic model contains NaN/Inf")


@dataclass(frozen=True)
class ForwardKinematicsSplit:
    """将六轴 FK 拆成前五轴基准、第六轴变化、Flange 和 TCP 位姿。"""

    first_five_frame_b: np.ndarray
    sixth_axis_transform_b: np.ndarray
    flange_transform_b: np.ndarray
    tcp_transform_b: np.ndarray
    first_five_tcp_reference_b: np.ndarray


@dataclass(frozen=True)
class TCPHeadingDecomposition:
    """TCP 平面朝向轴在 B 系 XOY 平面的朝向及其两部分 FK 来源。"""

    total_heading_rad: float
    first_five_heading_rad: float
    sixth_axis_heading_rad: float


def _endpoint_transform(flange_to_tcp: bool) -> np.ndarray:
    """构造 Frame6 到 Flange 或 TCP 的固定末端变换。"""

    flange = _homogeneous(_FRAME6_TO_FLANGE_ROTATION, (0.0, 0.0, 0.0))
    if not flange_to_tcp:
        return flange
    tcp_rotation = _quaternion_to_rotation_wxyz(FLANGE_TO_TOOL_ROTATION_F)
    tcp = _homogeneous(tcp_rotation, FLANGE_TO_TCP_TRANSLATION_F)
    return flange @ tcp


AUBO_E5_FLANGE_MODEL: Final = AuboKinematicModel(
    "aubo_e5_flange",
    AUBO_E5_DH_NO_TCP,
    _endpoint_transform(False),
)
AUBO_E5_TCP_MODEL: Final = AuboKinematicModel(
    "aubo_e5_tcp",
    AUBO_E5_DH_WITH_TCP,
    _endpoint_transform(True),
)


@dataclass(frozen=True)
class IKResult:
    """一次 IK 查询的关节解、收敛状态和末端残差。"""

    success: bool
    q: np.ndarray | None
    reason: str = ""
    iterations: int = 0
    position_error_m: float = float("inf")
    orientation_error_rad: float = float("inf")

    @property
    def q_degrees(self) -> np.ndarray | None:
        return None if self.q is None else np.rad2deg(self.q)


class IKMode(str, Enum):
    """可热插拔的末端任务维度。"""

    POSITION_3D = "position_3d"
    TOOL_AXIS_5D = "tool_axis_5d"
    FULL_POSE_6D = "full_pose_6d"


class IKPlugin(Protocol):
    """三种 IK 插件共享的最小调用契约。"""

    mode: IKMode

    def solve(self, target_position_b: Sequence[float], **kwargs: object) -> IKResult:
        """针对一个目标调用当前插件。"""


# =============================================================================
# 第五块：正向运动学与 Jacobian
# =============================================================================


def forward_kinematics(q: Sequence[float], model: AuboKinematicModel = AUBO_E5_TCP_MODEL) -> np.ndarray:
    """计算指定 endpoint 在机器人根坐标系 B 中的 4x4 位姿。"""

    q_array = _vector(q, "q", JOINT_COUNT)
    transform = _homogeneous(np.eye(3), _BASE_TO_DH_TRANSLATION_B)
    for row, angle in zip(model.dh, q_array, strict=True):
        transform = transform @ _dh_transform(row[0], row[1], row[2], row[3] + angle)
    return transform @ model.frame6_to_endpoint


def forward_kinematics_split(
    q: Sequence[float], model: AuboKinematicModel = AUBO_E5_TCP_MODEL
) -> ForwardKinematicsSplit:
    """返回前五轴基准与第六轴变化拆开的 FK 结果。

    ``first_five_tcp_reference_b`` 使用相同的前五轴角度、但将第六轴置零，
    因而它是“前五轴造成的 TCP 基准姿态”；完整 TCP 位姿再由实际第六轴
    变换叠加得到。该函数只做正向运动学，不改变任何 IK 求解路径。
    """

    q_array = _vector(q, "q", JOINT_COUNT)
    transform = _homogeneous(np.eye(3), _BASE_TO_DH_TRANSLATION_B)
    for row, angle in zip(model.dh[:5], q_array[:5], strict=True):
        transform = transform @ _dh_transform(row[0], row[1], row[2], row[3] + angle)
    first_five_frame_b = transform.copy()
    sixth_row = model.dh[5]
    sixth_axis_transform_b = _dh_transform(
        sixth_row[0], sixth_row[1], sixth_row[2], sixth_row[3] + q_array[5]
    )
    frame6_transform_b = first_five_frame_b @ sixth_axis_transform_b
    flange_transform_b = frame6_transform_b @ _homogeneous(_FRAME6_TO_FLANGE_ROTATION, (0.0, 0.0, 0.0))
    tcp_transform_b = frame6_transform_b @ model.frame6_to_endpoint
    first_five_tcp_reference_b = (
        first_five_frame_b
        @ _dh_transform(sixth_row[0], sixth_row[1], sixth_row[2], sixth_row[3])
        @ model.frame6_to_endpoint
    )
    return ForwardKinematicsSplit(
        first_five_frame_b=first_five_frame_b,
        sixth_axis_transform_b=sixth_axis_transform_b,
        flange_transform_b=flange_transform_b,
        tcp_transform_b=tcp_transform_b,
        first_five_tcp_reference_b=first_five_tcp_reference_b,
    )


def _planar_heading_rad(axis_b: np.ndarray) -> float:
    """计算向量在 B 系 XOY 平面的 atan2 朝向。"""

    planar = np.asarray(axis_b, dtype=np.float64).copy()
    planar[2] = 0.0
    norm = float(np.linalg.norm(planar))
    if norm <= 1.0e-9:
        raise ValueError("TCP 平面朝向轴在 XOY 平面的投影不能为零")
    return float(np.arctan2(planar[1] / norm, planar[0] / norm))


def _wrap_angle_rad(angle_rad: float) -> float:
    """将角度包装到 [-pi, pi)。"""

    return float((angle_rad + pi) % (2.0 * pi) - pi)


def tcp_heading_decomposition(
    q: Sequence[float],
    model: AuboKinematicModel = AUBO_E5_TCP_MODEL,
    tcp_heading_axis_local: Sequence[float] = TCP_HEADING_AXIS_LOCAL,
) -> TCPHeadingDecomposition:
    """返回 TCP 平面朝向以及前五轴/第六轴的 FK 分解贡献。"""

    axis_local = _vector(tcp_heading_axis_local, "tcp_heading_axis_local", 3)
    split = forward_kinematics_split(q, model)
    total_axis_b = split.tcp_transform_b[:3, :3] @ axis_local
    first_five_axis_b = split.first_five_tcp_reference_b[:3, :3] @ axis_local
    total_heading = _planar_heading_rad(total_axis_b)
    first_five_heading = _planar_heading_rad(first_five_axis_b)
    return TCPHeadingDecomposition(
        total_heading_rad=total_heading,
        first_five_heading_rad=first_five_heading,
        sixth_axis_heading_rad=_wrap_angle_rad(total_heading - first_five_heading),
    )


def forward_pose(q: Sequence[float], model: AuboKinematicModel = AUBO_E5_TCP_MODEL) -> tuple[np.ndarray, np.ndarray]:
    """返回 endpoint 在 B 系中的位置和旋转矩阵。"""

    transform = forward_kinematics(q, model)
    return transform[:3, 3].copy(), transform[:3, :3].copy()


def jacobian(q: Sequence[float], model: AuboKinematicModel = AUBO_E5_TCP_MODEL) -> np.ndarray:
    """计算 endpoint 在 B 系中的几何 Jacobian，形状为 (6, 6)。"""

    q_array = _vector(q, "q", JOINT_COUNT)
    transform = _homogeneous(np.eye(3), _BASE_TO_DH_TRANSLATION_B)
    joint_axes: list[np.ndarray] = []
    joint_origins: list[np.ndarray] = []
    for row, angle in zip(model.dh, q_array, strict=True):
        joint_axes.append(transform[:3, 2].copy())
        joint_origins.append(transform[:3, 3].copy())
        transform = transform @ _dh_transform(row[0], row[1], row[2], row[3] + angle)
    endpoint = transform @ model.frame6_to_endpoint
    position = endpoint[:3, 3]
    linear = np.column_stack([np.cross(axis, position - origin) for axis, origin in zip(joint_axes, joint_origins)])
    angular = np.column_stack(joint_axes)
    return np.vstack((linear, angular))


# =============================================================================
# 第六块：姿态误差与“末端向下”目标姿态
# =============================================================================


def downward_rotation(yaw: float) -> np.ndarray:
    """构造 endpoint 的向下姿态，使 endpoint 局部 +Z 指向 B 系 -Z。

    ``yaw`` 只决定向下姿态绕 B 系 Z 轴的旋转。这个矩阵本身就是
    endpoint 的目标旋转，不应再乘一次 ``frame6_to_endpoint``。
    """

    c, s = np.cos(yaw), np.sin(yaw)
    return np.array(((c, s, 0.0), (s, -c, 0.0), (0.0, 0.0, -1.0)), dtype=np.float64)


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """构造绕单位轴旋转 ``angle`` 的 SO(3) 矩阵。"""

    x, y, z = axis
    skew = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    return identity + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def downward_endpoint_rotation(
    target_position_b: Sequence[float],
    down_axis_b: Sequence[float],
    endpoint_axis_local: Sequence[float] = (0.0, 0.0, 1.0),
    yaw_offset_rad: float = 0.0,
) -> np.ndarray:
    """构造 endpoint 旋转，使其局部 +Z 对齐 ``down_axis_b``。

    x 轴取目标点相对 DH 基准点的水平投影，保证向下姿态仍有稳定的
    yaw；y 轴由右手系补齐。这样不仅能处理 B 系 -Z 向下，也能处理
    机器人根坐标系相对世界坐标系存在任意旋转的情况。
    """

    target_position = _vector(target_position_b, "target_position_b", 3)
    down_axis = _vector(down_axis_b, "down_axis_b", 3).copy()
    down_norm = np.linalg.norm(down_axis)
    if down_norm <= np.finfo(np.float64).eps:
        raise ValueError("down_axis_b must not be zero")
    down_axis /= down_norm
    local_axis = _vector(endpoint_axis_local, "endpoint_axis_local", 3).copy()
    local_norm = np.linalg.norm(local_axis)
    if local_norm <= np.finfo(np.float64).eps:
        raise ValueError("endpoint_axis_local must not be zero")
    local_axis /= local_norm
    radial = target_position - _BASE_TO_DH_TRANSLATION_B
    heading = radial - np.dot(radial, down_axis) * down_axis
    heading_norm = np.linalg.norm(heading)
    if heading_norm <= 1.0e-8:
        raise ValueError("target_position_b is on the down-axis and has no stable yaw")
    target_u = heading / heading_norm
    target_v = np.cross(down_axis, target_u)

    # 构造两个右手正交基：局部 (u, v, constrained_axis) 和目标
    # (u, v, down)。矩阵乘积保证 R @ endpoint_axis_local == down_axis_b。
    reference = np.array((1.0, 0.0, 0.0), dtype=np.float64)
    if abs(float(np.dot(reference, local_axis))) > 0.9:
        reference = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    local_u = reference - np.dot(reference, local_axis) * local_axis
    local_u /= np.linalg.norm(local_u)
    local_v = np.cross(local_axis, local_u)
    target_basis = np.column_stack((target_u, target_v, down_axis))
    local_basis = np.column_stack((local_u, local_v, local_axis))
    target_rotation = target_basis @ local_basis.T
    if not np.isfinite(yaw_offset_rad):
        raise ValueError("yaw_offset_rad must be finite")
    if abs(yaw_offset_rad) > np.finfo(np.float64).eps:
        target_rotation = _axis_angle_rotation(down_axis, yaw_offset_rad) @ target_rotation
    return target_rotation


def _rotation_log(rotation: np.ndarray) -> np.ndarray:
    """计算 SO(3) 旋转矩阵的旋转向量对数映射。"""

    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1.0e-8:
        return np.array(
            (
                0.5 * (rotation[2, 1] - rotation[1, 2]),
                0.5 * (rotation[0, 2] - rotation[2, 0]),
                0.5 * (rotation[1, 0] - rotation[0, 1]),
            ),
            dtype=np.float64,
        )
    if pi - angle < 1.0e-6:
        diagonal = np.maximum((np.diag(rotation) + 1.0) * 0.5, 0.0)
        axis_index = int(np.argmax(diagonal))
        axis = np.zeros(3, dtype=np.float64)
        axis[axis_index] = np.sqrt(diagonal[axis_index])
        next_index, last_index = (axis_index + 1) % 3, (axis_index + 2) % 3
        if axis[axis_index] > 1.0e-8:
            axis[next_index] = rotation[next_index, axis_index] / (2.0 * axis[axis_index])
            axis[last_index] = rotation[last_index, axis_index] / (2.0 * axis[axis_index])
        return angle * axis
    axis = np.array(
        (rotation[2, 1] - rotation[1, 2], rotation[0, 2] - rotation[2, 0], rotation[1, 0] - rotation[0, 1]),
        dtype=np.float64,
    ) / (2.0 * np.sin(angle))
    return angle * axis


def _pose_error(
    current_position: np.ndarray,
    current_rotation: np.ndarray,
    target_position: np.ndarray,
    target_rotation: np.ndarray | None,
) -> np.ndarray:
    """拼接位置误差和可选的姿态误差，供 DLS 使用。"""

    position_error = target_position - current_position
    if target_rotation is None:
        return position_error
    rotation_error = _rotation_log(target_rotation @ current_rotation.T)
    return np.concatenate((position_error, rotation_error))


def _normalized_axis(value: Sequence[float], name: str) -> np.ndarray:
    """检查并归一化一个工具轴方向。"""

    axis = _vector(value, name, 3).copy()
    norm = float(np.linalg.norm(axis))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError(f"{name} must not be zero")
    return axis / norm


def _axis_perpendicular_basis(axis: np.ndarray) -> np.ndarray:
    """返回垂直于 ``axis`` 的 3x2 正交基，用于表达工具轴的 2D 误差。"""

    reference = np.array((1.0, 0.0, 0.0), dtype=np.float64)
    if abs(float(np.dot(reference, axis))) > 0.9:
        reference = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    first = reference - np.dot(reference, axis) * axis
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    return np.column_stack((first, second))


def _tool_axis_error(
    current_rotation: np.ndarray,
    target_axis_b: np.ndarray,
    endpoint_axis_local: np.ndarray,
    axis_basis_b: np.ndarray,
) -> tuple[np.ndarray, float]:
    """计算工具轴的 2D 最短旋转误差及其角度。"""

    current_axis_b = current_rotation @ endpoint_axis_local
    cosine = float(np.clip(np.dot(current_axis_b, target_axis_b), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    cross = np.cross(current_axis_b, target_axis_b)
    sine = float(np.linalg.norm(cross))
    if sine > 1.0e-8:
        rotation_error_b = cross * (angle / sine)
    elif cosine >= 0.0:
        rotation_error_b = cross
    else:
        # 反向共线时旋转轴不唯一，选择固定的垂直基保证结果确定。
        rotation_error_b = axis_basis_b[:, 0] * pi
    return axis_basis_b.T @ rotation_error_b, angle


# =============================================================================
# 第七块：通用阻尼最小二乘 IK
# =============================================================================


def solve_ik(
    target_position_b: Sequence[float],
    target_quaternion_b_wxyz: Sequence[float] | None = None,
    *,
    model: AuboKinematicModel = AUBO_E5_TCP_MODEL,
    seed: Sequence[float] | None = None,
    max_iterations: int = 200,
    position_tolerance_m: float = 1.0e-5,
    orientation_tolerance_rad: float = 1.0e-4,
    damping: float = 0.02,
    max_joint_step_rad: float = 0.15,
    preferred_joint_position: Sequence[float] | None = None,
    nullspace_gain: float = 0.0,
) -> IKResult:
    """用带关节限位和单步限幅的 DLS 求解位置或完整位姿。

    注意：
    - ``target_quaternion_b_wxyz=None`` 时只求位置，不约束姿态；
    - 传入四元数时才求 6 维完整位姿；
    - 本函数不会额外施加“TCP 轴必须向下”等不等式约束；
    - 若需要向下姿态，应调用下面的 ``solve_down_ik``，或由调用方显式
      构造满足向下条件的目标四元数后调用本函数。
    """

    target_position = _vector(target_position_b, "target_position_b", 3)
    target_rotation = (
        None if target_quaternion_b_wxyz is None else _quaternion_to_rotation_wxyz(target_quaternion_b_wxyz)
    )
    if max_iterations <= 0 or damping <= 0.0 or max_joint_step_rad <= 0.0:
        raise ValueError("max_iterations, damping and max_joint_step_rad must be positive")
    if nullspace_gain < 0.0 or not np.isfinite(nullspace_gain):
        raise ValueError("nullspace_gain must be finite and non-negative")
    if seed is None:
        q = np.zeros(JOINT_COUNT, dtype=np.float64)
    else:
        q = _vector(seed, "seed", JOINT_COUNT).copy()
    preferred = None
    if preferred_joint_position is not None:
        preferred = _vector(preferred_joint_position, "preferred_joint_position", JOINT_COUNT)
    lower, upper = JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1]
    q = np.clip(q, lower, upper)
    task_size = 3 if target_rotation is None else 6

    for iteration in range(1, max_iterations + 1):
        current_position, current_rotation = forward_pose(q, model)
        error = _pose_error(current_position, current_rotation, target_position, target_rotation)
        position_error = float(np.linalg.norm(error[:3]))
        orientation_error = float(np.linalg.norm(error[3:])) if task_size == 6 else 0.0
        task_satisfied = position_error <= position_tolerance_m and orientation_error <= orientation_tolerance_rad
        if task_satisfied and (preferred is None or nullspace_gain <= 0.0):
            return IKResult(
                True, q, iterations=iteration, position_error_m=position_error, orientation_error_rad=orientation_error
            )
        current_jacobian = jacobian(q, model)[:task_size, :]
        system = current_jacobian @ current_jacobian.T + damping * damping * np.eye(task_size)
        task_inverse = current_jacobian.T @ np.linalg.inv(system)
        delta = task_inverse @ error
        if preferred is not None and nullspace_gain > 0.0:
            nullspace_projector = np.eye(JOINT_COUNT) - task_inverse @ current_jacobian
            projected_preference = nullspace_projector @ (preferred - q)
            if task_satisfied and np.linalg.norm(projected_preference) <= 1.0e-3:
                return IKResult(
                    True,
                    q,
                    iterations=iteration,
                    position_error_m=position_error,
                    orientation_error_rad=orientation_error,
                )
            delta += nullspace_gain * projected_preference
        delta_norm = float(np.linalg.norm(delta))
        if not np.isfinite(delta_norm):
            return IKResult(
                False, None, "DLS produced a non-finite joint step", iteration, position_error, orientation_error
            )
        if delta_norm > max_joint_step_rad:
            delta *= max_joint_step_rad / delta_norm
        q = np.clip(q + delta, lower, upper)

    current_position, current_rotation = forward_pose(q, model)
    residual = _pose_error(current_position, current_rotation, target_position, target_rotation)
    position_error = float(np.linalg.norm(residual[:3]))
    orientation_error = float(np.linalg.norm(residual[3:])) if task_size == 6 else 0.0
    return IKResult(False, None, "maximum iterations reached", max_iterations, position_error, orientation_error)


def solve_tool_axis_ik(
    target_position_b: Sequence[float],
    target_axis_b: Sequence[float],
    *,
    model: AuboKinematicModel = AUBO_E5_TCP_MODEL,
    endpoint_axis_local: Sequence[float] = (0.0, 0.0, 1.0),
    seed: Sequence[float] | None = None,
    max_iterations: int = 200,
    position_tolerance_m: float = 1.0e-5,
    axis_tolerance_rad: float = 1.0e-4,
    damping: float = 0.02,
    max_joint_step_rad: float = 0.15,
    preferred_joint_position: Sequence[float] | None = None,
    nullspace_gain: float = 0.15,
) -> IKResult:
    """用 5D DLS 求解位置和工具轴方向，不约束绕工具轴的旋转。"""

    target_position = _vector(target_position_b, "target_position_b", 3)
    target_axis = _normalized_axis(target_axis_b, "target_axis_b")
    endpoint_axis = _normalized_axis(endpoint_axis_local, "endpoint_axis_local")
    axis_basis = _axis_perpendicular_basis(target_axis)
    if max_iterations <= 0 or damping <= 0.0 or max_joint_step_rad <= 0.0:
        raise ValueError("max_iterations, damping and max_joint_step_rad must be positive")
    if nullspace_gain < 0.0 or not np.isfinite(nullspace_gain):
        raise ValueError("nullspace_gain must be finite and non-negative")
    if seed is None:
        q = np.zeros(JOINT_COUNT, dtype=np.float64)
    else:
        q = _vector(seed, "seed", JOINT_COUNT).copy()
    preferred = None
    if preferred_joint_position is not None:
        preferred = _vector(preferred_joint_position, "preferred_joint_position", JOINT_COUNT)
    lower, upper = JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1]
    q = np.clip(q, lower, upper)

    for iteration in range(1, max_iterations + 1):
        current_position, current_rotation = forward_pose(q, model)
        axis_error, axis_error_rad = _tool_axis_error(current_rotation, target_axis, endpoint_axis, axis_basis)
        position_error = float(np.linalg.norm(target_position - current_position))
        if position_error <= position_tolerance_m and axis_error_rad <= axis_tolerance_rad:
            return IKResult(
                True,
                q,
                iterations=iteration,
                position_error_m=position_error,
                orientation_error_rad=axis_error_rad,
            )

        current_jacobian = jacobian(q, model)
        task_jacobian = np.vstack((current_jacobian[:3, :], axis_basis.T @ current_jacobian[3:, :]))
        error = np.concatenate((target_position - current_position, axis_error))
        system = task_jacobian @ task_jacobian.T + damping * damping * np.eye(5)
        task_inverse = task_jacobian.T @ np.linalg.inv(system)
        delta = task_inverse @ error
        if preferred is not None and nullspace_gain > 0.0:
            nullspace_projector = np.eye(JOINT_COUNT) - task_inverse @ task_jacobian
            delta += nullspace_gain * (nullspace_projector @ (preferred - q))
        delta_norm = float(np.linalg.norm(delta))
        if not np.isfinite(delta_norm):
            return IKResult(
                False, None, "5D DLS produced a non-finite joint step", iteration, position_error, axis_error_rad
            )
        if delta_norm > max_joint_step_rad:
            delta *= max_joint_step_rad / delta_norm
        q = np.clip(q + delta, lower, upper)

    current_position, current_rotation = forward_pose(q, model)
    _, axis_error_rad = _tool_axis_error(current_rotation, target_axis, endpoint_axis, axis_basis)
    position_error = float(np.linalg.norm(target_position - current_position))
    return IKResult(False, None, "maximum iterations reached", max_iterations, position_error, axis_error_rad)


def solve_position_ik(
    target_position_b: Sequence[float],
    *,
    model: AuboKinematicModel = AUBO_E5_TCP_MODEL,
    seed: Sequence[float] | None = None,
    **kwargs: object,
) -> IKResult:
    """3D 插件：只约束 endpoint 位置。"""

    return solve_ik(target_position_b, None, model=model, seed=seed, **kwargs)


def solve_full_pose_ik(
    target_position_b: Sequence[float],
    target_quaternion_b_wxyz: Sequence[float],
    *,
    model: AuboKinematicModel = AUBO_E5_TCP_MODEL,
    seed: Sequence[float] | None = None,
    **kwargs: object,
) -> IKResult:
    """6D 插件：同时约束 endpoint 位置和完整姿态。"""

    return solve_ik(target_position_b, target_quaternion_b_wxyz, model=model, seed=seed, **kwargs)


class PositionIKSolver:
    """可插拔的 3D 位置 IK。"""

    mode = IKMode.POSITION_3D

    def __init__(self, model: AuboKinematicModel = AUBO_E5_TCP_MODEL) -> None:
        self.model = model

    def solve(self, target_position_b: Sequence[float], **kwargs: object) -> IKResult:
        return solve_position_ik(target_position_b, model=self.model, **kwargs)


class ToolAxisIKSolver:
    """可插拔的 5D 位置 + 工具轴方向 IK。"""

    mode = IKMode.TOOL_AXIS_5D

    def __init__(self, model: AuboKinematicModel = AUBO_E5_TCP_MODEL) -> None:
        self.model = model

    def solve(self, target_position_b: Sequence[float], **kwargs: object) -> IKResult:
        target_axis_b = kwargs.pop("target_axis_b", None)
        if target_axis_b is None:
            raise ValueError("tool_axis_5d requires target_axis_b")
        return solve_tool_axis_ik(target_position_b, target_axis_b, model=self.model, **kwargs)


class FullPoseIKSolver:
    """可插拔的 6D 位置 + 完整姿态 IK。"""

    mode = IKMode.FULL_POSE_6D

    def __init__(self, model: AuboKinematicModel = AUBO_E5_TCP_MODEL) -> None:
        self.model = model

    def solve(self, target_position_b: Sequence[float], **kwargs: object) -> IKResult:
        target_quaternion = kwargs.pop("target_quaternion_b_wxyz", None)
        if target_quaternion is None:
            raise ValueError("full_pose_6d requires target_quaternion_b_wxyz")
        return solve_full_pose_ik(target_position_b, target_quaternion, model=self.model, **kwargs)


def create_ik_solver(
    mode: IKMode | str,
    *,
    model: AuboKinematicModel = AUBO_E5_TCP_MODEL,
) -> IKPlugin:
    """按模式创建 IK 插件；上层可在运行时替换返回的 solver。"""

    selected_mode = mode if isinstance(mode, IKMode) else IKMode(mode)
    if selected_mode is IKMode.POSITION_3D:
        return PositionIKSolver(model)
    if selected_mode is IKMode.TOOL_AXIS_5D:
        return ToolAxisIKSolver(model)
    if selected_mode is IKMode.FULL_POSE_6D:
        return FullPoseIKSolver(model)
    raise ValueError(f"unsupported IK mode: {mode}")


def solve_ik_mode(
    mode: IKMode | str,
    target_position_b: Sequence[float],
    *,
    model: AuboKinematicModel = AUBO_E5_TCP_MODEL,
    **kwargs: object,
) -> IKResult:
    """一次性按模式路由到可插拔 IK。"""

    return create_ik_solver(mode, model=model).solve(target_position_b, **kwargs)


# =============================================================================
# 第八块：Flange/TCP 两套公开 IK 封装
# =============================================================================


def solve_flange_ik(
    target_position_b: Sequence[float], target_quaternion_b_wxyz: Sequence[float] | None = None, **kwargs: object
) -> IKResult:
    """求解 endpoint 为机器人 Flange 的位置或完整位姿。"""

    return solve_ik(target_position_b, target_quaternion_b_wxyz, model=AUBO_E5_FLANGE_MODEL, **kwargs)


def solve_tcp_ik(
    target_position_b: Sequence[float], target_quaternion_b_wxyz: Sequence[float] | None = None, **kwargs: object
) -> IKResult:
    """求解 endpoint 为项目 task TCP 的位置或完整位姿。"""

    return solve_ik(target_position_b, target_quaternion_b_wxyz, model=AUBO_E5_TCP_MODEL, **kwargs)


# =============================================================================
# 第九块：显式末端向下 IK
# =============================================================================


def _world_down_axis_in_root(root_quaternion_wxyz: Sequence[float] | None) -> np.ndarray:
    """将世界向下轴转换到机器人根坐标系 B。"""

    if root_quaternion_wxyz is None:
        return WORLD_DOWN_AXIS_W.copy()
    root_rotation_w = _quaternion_to_rotation_wxyz(root_quaternion_wxyz)
    return root_rotation_w.T @ WORLD_DOWN_AXIS_W


def tcp_target_to_flange_down_position(
    target_tcp_position_b: Sequence[float],
    *,
    root_quaternion_wxyz: Sequence[float] | None = None,
    downward_yaw_offset_rad: float = 0.0,
    max_iterations: int = 50,
    tolerance_m: float = 1.0e-10,
) -> np.ndarray:
    """把 TCP 目标点逆变换为保持 Flange 法向向下时的 Flange 原点目标。

    Flange 目标 yaw 由其目标位置决定，因此通过固定点迭代同时满足：
    ``p_tcp = p_flange + R_flange @ t_flange_to_tcp``，以及 Flange 局部
    ``-X`` 指向世界 ``-Z``。返回值可直接交给 ``solve_flange_down_ik``。
    """

    target_tcp_position = _vector(target_tcp_position_b, "target_tcp_position_b", 3)
    if max_iterations <= 0 or tolerance_m <= 0.0 or not np.isfinite(tolerance_m):
        raise ValueError("max_iterations and tolerance_m must be positive")
    down_axis_b = _world_down_axis_in_root(root_quaternion_wxyz)
    flange_to_tcp_translation = _vector(
        FLANGE_TO_TCP_TRANSLATION_F,
        "FLANGE_TO_TCP_TRANSLATION_F",
        3,
    )
    flange_position = target_tcp_position.copy()
    for _ in range(max_iterations):
        flange_rotation = downward_endpoint_rotation(
            flange_position,
            down_axis_b,
            FLANGE_DOWNWARD_AXIS_LOCAL,
            yaw_offset_rad=downward_yaw_offset_rad,
        )
        updated_position = target_tcp_position - flange_rotation @ flange_to_tcp_translation
        if np.linalg.norm(updated_position - flange_position) <= tolerance_m:
            return updated_position
        flange_position = updated_position

    # 某些大 yaw 与 TCP 偏置组合不满足简单固定点迭代的收缩条件，但原方程
    # 仍有解。对 r(p_f) = p_f + R(p_f)t_f_tcp - p_tcp 做有限差分 Newton，
    # 并用回溯线搜索保证每次都降低位置残差。
    flange_position = target_tcp_position.copy()

    def residual(position: np.ndarray) -> np.ndarray:
        rotation = downward_endpoint_rotation(
            position,
            down_axis_b,
            FLANGE_DOWNWARD_AXIS_LOCAL,
            yaw_offset_rad=downward_yaw_offset_rad,
        )
        return position + rotation @ flange_to_tcp_translation - target_tcp_position

    finite_difference_m = max(1.0e-6, tolerance_m * 100.0)
    identity = np.eye(3, dtype=np.float64)
    for _ in range(max_iterations):
        current_residual = residual(flange_position)
        residual_norm = float(np.linalg.norm(current_residual))
        if residual_norm <= tolerance_m:
            return flange_position
        residual_jacobian = np.column_stack(
            [
                (
                    residual(flange_position + finite_difference_m * direction)
                    - residual(flange_position - finite_difference_m * direction)
                )
                / (2.0 * finite_difference_m)
                for direction in identity
            ]
        )
        try:
            step = np.linalg.solve(residual_jacobian, -current_residual)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(residual_jacobian, -current_residual, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            break
        scale = 1.0
        accepted = False
        for _ in range(14):
            candidate = flange_position + scale * step
            if float(np.linalg.norm(residual(candidate))) < residual_norm:
                flange_position = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break
    raise RuntimeError("TCP-to-Flange inverse transform did not converge")


def flange_down_tcp_target_pose(
    target_tcp_position_b: Sequence[float],
    *,
    root_quaternion_wxyz: Sequence[float] | None = None,
    downward_yaw_offset_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 Flange-down 对应的 Flange 位置与 TCP 目标四元数 ``wxyz``。"""

    flange_position = tcp_target_to_flange_down_position(
        target_tcp_position_b,
        root_quaternion_wxyz=root_quaternion_wxyz,
        downward_yaw_offset_rad=downward_yaw_offset_rad,
    )
    flange_rotation = downward_endpoint_rotation(
        flange_position,
        _world_down_axis_in_root(root_quaternion_wxyz),
        FLANGE_DOWNWARD_AXIS_LOCAL,
        yaw_offset_rad=downward_yaw_offset_rad,
    )
    tcp_rotation = flange_rotation @ _quaternion_to_rotation_wxyz(FLANGE_TO_TOOL_ROTATION_F)
    return flange_position, _rotation_to_quaternion_wxyz(tcp_rotation)


def flange_down_yaw_offset_for_tcp_heading(
    target_tcp_position_b: Sequence[float],
    target_heading_rad: float,
    *,
    root_quaternion_wxyz: Sequence[float] | None = None,
    tcp_heading_axis_local: Sequence[float] = TCP_HEADING_AXIS_LOCAL,
    initial_offset_rad: float = 0.0,
    tolerance_rad: float = 1.0e-6,
    max_iterations: int = 20,
    finite_difference_rad: float = 1.0e-4,
    max_step_rad: float = pi / 4.0,
) -> float:
    """求解给定 TCP 平面朝向对应的 Flange-down yaw offset。

    TCP 相对 Flange 存在平移偏置，改变 yaw 时，为保持同一个 TCP 位置，
    Flange 目标位置也会随之变化。因此 heading 与 yaw offset 在大角度范围
    内不是一次线性换算。本函数使用有限差分 Newton 迭代，并限制每次角度
    更新，供采集和回放共同使用同一份几何真值。
    """

    target_position = _vector(target_tcp_position_b, "target_tcp_position_b", 3)
    heading_axis = _vector(tcp_heading_axis_local, "tcp_heading_axis_local", 3)
    heading_axis_norm = float(np.linalg.norm(heading_axis))
    if heading_axis_norm <= np.finfo(np.float64).eps:
        raise ValueError("tcp_heading_axis_local must not be zero")
    heading_axis /= heading_axis_norm
    if not np.isfinite(target_heading_rad) or not np.isfinite(initial_offset_rad):
        raise ValueError("target_heading_rad and initial_offset_rad must be finite")
    if tolerance_rad <= 0.0 or finite_difference_rad <= 0.0 or max_step_rad <= 0.0:
        raise ValueError("heading solver tolerances and max_step_rad must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    def heading_at(offset_rad: float) -> float:
        _, tcp_quaternion = flange_down_tcp_target_pose(
            target_position,
            root_quaternion_wxyz=root_quaternion_wxyz,
            downward_yaw_offset_rad=offset_rad,
        )
        tcp_rotation = _quaternion_to_rotation_wxyz(tcp_quaternion)
        return _planar_heading_rad(tcp_rotation @ heading_axis)

    offset = _wrap_angle_rad(float(initial_offset_rad))
    residual = float("inf")
    for _ in range(max_iterations):
        current_heading = heading_at(offset)
        residual = _wrap_angle_rad(float(target_heading_rad) - current_heading)
        if abs(residual) <= tolerance_rad:
            return offset
        plus_heading = heading_at(offset + finite_difference_rad)
        minus_heading = heading_at(offset - finite_difference_rad)
        slope = _wrap_angle_rad(plus_heading - minus_heading) / (2.0 * finite_difference_rad)
        if abs(slope) <= 1.0e-6 or not np.isfinite(slope):
            raise RuntimeError("Flange-down yaw offset 对 TCP 平面朝向没有有效映射")
        step = float(np.clip(residual / slope, -max_step_rad, max_step_rad))
        offset = _wrap_angle_rad(offset + step)
    raise RuntimeError(
        "Flange-down yaw offset 未收敛到目标 TCP 平面朝向："
        f"residual={residual:.6g} rad"
    )


def solve_down_ik(
    target_position_b: Sequence[float],
    *,
    model: AuboKinematicModel = AUBO_E5_TCP_MODEL,
    seed: Sequence[float] | None = None,
    root_quaternion_wxyz: Sequence[float] | None = None,
    endpoint_axis_local: Sequence[float] = (0.0, 0.0, 1.0),
    downward_yaw_offset_rad: float = 0.0,
    **kwargs: object,
) -> IKResult:
    """求解 endpoint 局部 +Z 指向世界坐标系向下方向的位置姿态。

    这里的“末端”由 ``model`` 决定：
    - ``AUBO_E5_FLANGE_MODEL``：由 ``endpoint_axis_local`` 指定，当前模型使用局部 -X；
    - ``AUBO_E5_TCP_MODEL``：TCP 的 +Z 向下。

    Isaac Sim 世界坐标采用 Z 轴向上、重力沿世界 -Z，因此世界向下轴
    固定为 ``WORLD_DOWN_AXIS_W = (0, 0, -1)``。如果提供
    ``root_quaternion_wxyz``，函数会把这个世界轴转换到机器人 B 系；
    若不提供，则按当前项目静态场景的约定使用 B 系 ``(0, 0, -1)``。

    当前工作站姿态只绕世界 Z 轴旋转，所以 B 系的 -Z 与世界 -Z
    重合；显式传入根姿态仍然是更安全的调用方式。
    """

    target_position = _vector(target_position_b, "target_position_b", 3)
    down_axis_b = _world_down_axis_in_root(root_quaternion_wxyz)
    # 这里直接生成所选 endpoint 的目标旋转，不能再乘
    # model.frame6_to_endpoint，否则约束会转移到辅助 Frame6。
    target_rotation = downward_endpoint_rotation(
        target_position,
        down_axis_b,
        endpoint_axis_local,
        yaw_offset_rad=downward_yaw_offset_rad,
    )
    return solve_ik(target_position, _rotation_to_quaternion_wxyz(target_rotation), model=model, seed=seed, **kwargs)


def solve_flange_down_ik(
    target_position_b: Sequence[float],
    *,
    seed: Sequence[float] | None = None,
    root_quaternion_wxyz: Sequence[float] | None = None,
    downward_yaw_offset_rad: float = 0.0,
    **kwargs: object,
) -> IKResult:
    """求解 Flange 位置，并约束其法向（局部 ``-X``）朝世界 ``-Z``。

    当前 AUBO 资产中，Flange 的工作面法向由局部 ``-X`` 表示；TCP 的
    局部 ``+Z`` 是另一套坐标约定，不能在 Flange IK 中混用。该封装固定
    使用 Flange endpoint 和对应局部法向，调用方无需在场景脚本中重复
    组装姿态目标或选择模型。
    """

    return solve_down_ik(
        target_position_b,
        model=AUBO_E5_FLANGE_MODEL,
        seed=seed,
        root_quaternion_wxyz=root_quaternion_wxyz,
        endpoint_axis_local=FLANGE_DOWNWARD_AXIS_LOCAL,
        downward_yaw_offset_rad=downward_yaw_offset_rad,
        **kwargs,
    )


__all__ = [
    "AUBO_E5_DH_NO_TCP",
    "AUBO_E5_DH_WITH_TCP",
    "AUBO_E5_FLANGE_MODEL",
    "AUBO_E5_TCP_MODEL",
    "FLANGE_DOWNWARD_AXIS_LOCAL",
    "AuboKinematicModel",
    "ForwardKinematicsSplit",
    "IKResult",
    "IKMode",
    "IKPlugin",
    "TCPHeadingDecomposition",
    "PositionIKSolver",
    "ToolAxisIKSolver",
    "FullPoseIKSolver",
    "WORLD_DOWN_AXIS_W",
    "downward_endpoint_rotation",
    "downward_rotation",
    "forward_kinematics",
    "forward_pose",
    "flange_down_tcp_target_pose",
    "flange_down_yaw_offset_for_tcp_heading",
    "forward_kinematics_split",
    "jacobian",
    "create_ik_solver",
    "solve_down_ik",
    "solve_flange_down_ik",
    "solve_flange_ik",
    "solve_ik",
    "solve_ik_mode",
    "solve_position_ik",
    "solve_tool_axis_ik",
    "solve_full_pose_ik",
    "solve_tcp_ik",
    "tcp_target_to_flange_down_position",
    "tcp_heading_decomposition",
]
