"""TCP 标定、目标状态、最终成功条件和安全阈值的唯一事实源。"""

from dataclasses import dataclass
from math import pi
from typing import Final

from .coordinate_frames import (
    FLANGE_TO_TCP_ROTATION_F_WXYZ,
    TARGET_DOCKING_AXIS_G,
    TARGET_TO_TCP_ROTATION_G_WXYZ,
    TCP_DIRECTION_AXIS_T,
    TCP_TOOL_FORWARD_AXIS_T,
)
from .coordinate_frames import (
    FLANGE_TO_TCP_TRANSLATION_F as COORDINATE_FLANGE_TO_TCP_TRANSLATION_F,
)


@dataclass(frozen=True)
class TargetState:
    """环境局部坐标中的离散目标状态。"""

    name: str
    # 目标物实体根位置，坐标系为环境局部 E，单位 m。
    position_e: tuple[float, float, float]
    # 目标物实体姿态，四元数顺序为 wxyz。
    rotation_wxyz: tuple[float, float, float, float]
    # TCP 停靠/接近位置，坐标系为环境局部 E，单位 m。
    preposition_e: tuple[float, float, float]
    # 夹爪闭合并抓取目标物时的 TCP 位置，坐标系为环境局部 E，单位 m。
    grasp_tcp_position_e: tuple[float, float, float]


# 兼容旧调用方的名称；唯一几何事实源位于 configs.coordinate_frames。
FLANGE_TO_TCP_TRANSLATION_F: Final = COORDINATE_FLANGE_TO_TCP_TRANSLATION_F
FLANGE_TO_TOOL_ROTATION_F: Final = FLANGE_TO_TCP_ROTATION_F_WXYZ
TARGET_TO_TOOL_ROTATION_T: Final = TARGET_TO_TCP_ROTATION_G_WXYZ
TOOL_FORWARD_AXIS: Final = TCP_TOOL_FORWARD_AXIS_T
TCP_HEADING_AXIS_LOCAL: Final = TCP_DIRECTION_AXIS_T
TARGET_DOCKING_AXIS: Final = TARGET_DOCKING_AXIS_G
# TCP 停靠点相对目标物根位置的环境局部 E 坐标 Z 轴抬高量，单位 m。
TCP_DOCKING_Z_OFFSET_E: Final = 0.017
# TCP 从 preposition 沿目标物方向推进到闭爪抓取点的距离，单位 m。
# 该值同时约束采集器的 normal advance 参数，避免推进过深造成夹爪碰撞。
NORMAL_ADVANCE_DISTANCE_M: Final = 0.075

TARGET_STATES: Final = (
    TargetState(
        "sample_bottle_state_01",
        (1.537, 0.203, 0.94),
        (0.0, 0.0, 0.0, 1.0),
        (1.537, 0.083, 0.94 + TCP_DOCKING_Z_OFFSET_E),
        (1.537, 0.158, 0.94 + TCP_DOCKING_Z_OFFSET_E),
    ),
    TargetState(
        "sample_bottle_state_02",
        (0.91167, 0.1753, 0.96789),
        (0.70710678, 0.0, 0.0, -0.70710678),
        (1.03167, 0.1753, 0.96789 + TCP_DOCKING_Z_OFFSET_E),
        (0.95667, 0.1753, 0.96789 + TCP_DOCKING_Z_OFFSET_E),
    ),
    TargetState(
        "sample_bottle_state_03",
        (0.91167, 0.03036, 0.96676),
        (0.70710678, 0.0, 0.0, -0.70710678),
        (1.03167, 0.03036, 0.96676 + TCP_DOCKING_Z_OFFSET_E),
        (0.95667, 0.03036, 0.96676 + TCP_DOCKING_Z_OFFSET_E),
    ),
    TargetState(
        "sample_bottle_state_04",
        (0.91235, -0.18557, 0.99091),
        (0.70710678, 0.0, 0.0, -0.70710678),
        (1.03235, -0.18557, 0.99091 + TCP_DOCKING_Z_OFFSET_E),
        (0.95735, -0.18557, 0.99091 + TCP_DOCKING_Z_OFFSET_E),
    ),
    TargetState(
        "sample_bottle_state_05",
        (0.90264, -0.50461, 1.0915),
        (1.0, 0.0, 0.0, 0.0),
        (0.90264, -0.38461, 1.0915 + TCP_DOCKING_Z_OFFSET_E),
        (0.90264, -0.45961, 1.0915 + TCP_DOCKING_Z_OFFSET_E),
    ),
)

PARKING_ENTER_DISTANCE_M: Final = 0.04
PARKING_EXIT_DISTANCE_M: Final = 0.055
PARKING_MAX_TCP_SPEED_M_S: Final = 0.03
PARKING_MAX_ORIENTATION_ERROR_RAD: Final = 10.0 * pi / 180.0
PARKING_DWELL_POLICY_STEPS: Final = 2

TCP_WORKSPACE_B: Final = ((-0.80, 0.80), (-0.80, 0.80), (0.05, 1.20))
ILLEGAL_CONTACT_FORCE_N: Final = 50.0
MAX_TARGET_DISPLACEMENT_M: Final = 0.03
MAX_TARGET_LINEAR_SPEED_M_S: Final = 0.05
