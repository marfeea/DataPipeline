"""阶段 A 静态场景的位姿和执行器基线。"""

from dataclasses import dataclass
from math import pi
from typing import Final


@dataclass(frozen=True)
class Pose:
    """环境局部位姿，四元数顺序固定为 wxyz。"""

    position: tuple[float, float, float]
    rotation_wxyz: tuple[float, float, float, float]


WORKSTATION_POSE_E: Final = Pose(
    position=(1.3, 0.0, 0.0),
    rotation_wxyz=(0.70711, 0.0, 0.0, -0.70711),
)
AUBO_LOCAL_POSITION_STATION: Final = (0.034, -0.013, 0.816)
AUBO_2_LOCAL_POSITION_STATION: Final = (-0.81, 0.21, 0.816)

ARM_INITIAL_JOINT_POSITIONS: Final = {
    "Joint1": 0.0,
    "Joint2": -30.0 * pi / 180.0,
    "Joint3": 70.0 * pi / 180.0,
    "Joint4": 45.0 * pi / 180.0,
    "Joint5": 90.0 * pi / 180.0,
    "Flange": 0.0,
}
GRIPPER_INITIAL_JOINT_POSITIONS: Final = {"UpperFinger": 0.0115, "DownFinger": -0.0115}

ARM_ACTUATOR: Final = {
    "effort_limit_sim": 2400.0,
    "velocity_limit_sim": 3.14,
    "stiffness": 6000.0,
    "damping": 600.0,
}
GRIPPER_ACTUATOR: Final = {
    "effort_limit_sim": 100.0,
    "stiffness": 2000.0,
    "damping": 100.0,
}

# pick-place 测试使用的低质量动态目标；保持为正值，避免无质量刚体的不稳定行为。
SAMPLE_BOTTLE_MASS_KG: Final = 0.01

ENABLE_SELF_COLLISIONS: Final = False
SOLVER_POSITION_ITERATIONS: Final = 8
SOLVER_VELOCITY_ITERATIONS: Final = 1
