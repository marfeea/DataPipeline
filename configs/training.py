"""仿真、策略频率和 episode 时长基线。"""

from typing import Final

SIMULATION_DT_S: Final = 1.0 / 120.0
POLICY_DECIMATION: Final = 30
POLICY_FREQUENCY_HZ: Final = 4.0
EPISODE_LENGTH_S: Final = 40.0
MAX_POLICY_STEPS: Final = 160
