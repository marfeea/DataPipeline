"""Differential IK 六维动作、阻尼和安全限幅的唯一配置入口。"""

from math import pi
from typing import Final

from .scene import ARM_ACTUATOR

ACTION_DIMENSION: Final = 6
ACTION_FRAME: Final = "robot_root_B"
ROTATION_INCREMENT_REPRESENTATION: Final = "rotation_vector"
ACTION_CLIP: Final = 1.0
POSITION_INCREMENT_SCALE_M: Final = 0.02
ROTATION_INCREMENT_SCALE_RAD: Final = 5.0 * pi / 180.0

POSITION_ERROR_GAIN: Final = 1.0
ROTATION_ERROR_GAIN: Final = 1.0
DLS_DAMPING: Final = 0.05
SINGULAR_VALUE_THRESHOLD: Final = 0.05
SINGULAR_DAMPING: Final = 0.20

MAX_JOINT_DELTA_RAD: Final = 0.02
MAX_JOINT_VELOCITY_RAD_S: Final = float(ARM_ACTUATOR["velocity_limit_sim"])
JOINT_POSITION_MARGIN_RAD: Final = 1.0 * pi / 180.0
