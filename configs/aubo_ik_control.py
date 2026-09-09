"""AUBO 数值 IK 训练链的动作、求解和安全跟踪参数。"""

from math import pi
from typing import Final

from .scene import ARM_ACTUATOR

ACTION_DIMENSION: Final = 6
ACTION_REPRESENTATION: Final = "normalized_tcp_increment_to_aubo_ik"
ACTION_FRAME: Final = "robot_root_B"
ACTION_CLIP: Final = 1.0
ACTION_SPACE_LOW: Final = -1.0
ACTION_SPACE_HIGH: Final = 1.0
POSITION_INCREMENT_SCALE_M: Final = 0.02
ROTATION_INCREMENT_SCALE_RAD: Final = 5.0 * pi / 180.0

MAX_IK_ITERATIONS: Final = 80
IK_POSITION_TOLERANCE_M: Final = 1.0e-4
IK_ORIENTATION_TOLERANCE_RAD: Final = 1.0e-3
IK_AXIS_TOLERANCE_RAD: Final = 1.0e-3
IK_DAMPING: Final = 0.02
IK_INTERNAL_MAX_JOINT_STEP_RAD: Final = 0.15

# 数值 IK 的完整解可能跨构型；正式训练只允许每个 4 Hz 策略步小幅推进。
MAX_POLICY_JOINT_DELTA_RAD: Final = (0.08, 0.08, 0.08, 0.08, 0.08, 0.08)
MAX_JOINT_VELOCITY_RAD_S: Final = float(ARM_ACTUATOR["velocity_limit_sim"])
JOINT_TARGET_MARGIN_RAD: Final = 0.1 * pi / 180.0
HARD_LIMIT_TERMINATION_MARGIN_RAD: Final = 0.02 * pi / 180.0
