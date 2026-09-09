"""PPO 六轴增量关节角动作和安全跟踪参数。"""

from math import pi
from typing import Final

from .scene import ARM_ACTUATOR

ACTION_DIMENSION: Final = 6
ACTION_REPRESENTATION: Final = "normalized_incremental_joint_position"
ACTION_UNIT: Final = "rad"
ACTION_CLIP: Final = 1.0
ACTION_SPACE_LOW: Final = -1.0
ACTION_SPACE_HIGH: Final = 1.0

# 每个 4 Hz 策略步允许各关节移动的最大角度；用六维向量保留后续逐关节标定能力。
MAX_POLICY_JOINT_DELTA_RAD: Final = (0.03, 0.03, 0.03, 0.03, 0.03, 0.03)
MAX_JOINT_VELOCITY_RAD_S: Final = float(ARM_ACTUATOR["velocity_limit_sim"])

# 距硬限位 1° 内是允许进入的低代价软区；动作目标最多进入到距硬限位 0.1°。
SOFT_LIMIT_MARGIN_RAD: Final = 1.0 * pi / 180.0
JOINT_TARGET_MARGIN_RAD: Final = 0.1 * pi / 180.0
# 实际关节进入距硬限位 0.02° 的数值容差即视为硬限位安全失败。
HARD_LIMIT_TERMINATION_MARGIN_RAD: Final = 0.02 * pi / 180.0
