"""TCP 停靠奖励参数；数值均按一个 4 Hz 策略步解释。"""

from typing import Final

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardParameters:
    """一个策略步上的奖励参数，距离和 TCP 位移单位均为米。"""

    position_weight: float
    position_scale_m: float
    progress_weight_per_m: float
    target_best_progress_total_reward: float
    stall_weight_per_m: float
    stall_far_distance_m: float
    stall_far_min_progress_m: float
    stall_near_min_progress_m: float
    far_move_weight: float
    far_distance_scale_m: float
    near_inefficiency_weight: float
    useful_progress_multiplier: float
    time_penalty_weight: float
    quality_progress_scale_m: float
    quality_action_squared_gain_per_m2: float
    quality_jitter_squared_gain_per_m2: float
    success_terminal_reward: float
    timeout_terminal_reward: float
    collision_terminal_reward: float
    limit_terminal_reward: float
    tool_axis_best_progress_weight: float
    path_reference_progress_weight_per_m: float
    path_reference_reached_reward: float
    joint_soft_limit_max_penalty: float


REWARD_PARAMETERS: Final = RewardParameters(
    # 正线性当前位置奖励：目标点为 0.03，距离 1 m 及以上为 0。
    position_weight=0.03,
    position_scale_m=1.0,
    # 主奖励保持连续有符号：接近 1 cm 得 +0.05，远离 1 cm 得 -0.05。
    progress_weight_per_m=5.0,
    # 每个 episode 从初始距离刷新到目标点的历史前沿总预算固定为 +10。
    target_best_progress_total_reward=10.0,
    stall_weight_per_m=-2.0,
    stall_far_distance_m=0.15,
    stall_far_min_progress_m=0.015,
    # 正式停车要求 TCP 速度 <= 0.03 m/s；4 Hz 下 0.008 m 的旧阈值
    # 已高于 0.0075 m/步，因此近区下调，避免惩罚合法的精细停车动作。
    stall_near_min_progress_m=0.002,
    # 以远区 progress=actNorm=0.02 m 的有效动作贡献约 0.02~0.03 为标定点；
    # 原 0.10 仅产生约 2.8e-4，训练占比接近零，无法承担移动有效性引导。
    far_move_weight=10.0,
    far_distance_scale_m=0.15,
    near_inefficiency_weight=-1.2,
    useful_progress_multiplier=6.0,
    # 旧 30 步回合末端系数为 -0.8；先按 160 步缩放为 -0.16，随后因
    # far-move 放大将当前值收紧为 -0.22。
    # 移动有效性放大后同步略增时间代价，保证“早到近区再等待超时”的
    # gamma=0.99 折扣回报仍为负，且不改变正常有效移动的正向排序。
    time_penalty_weight=-0.22,
    quality_progress_scale_m=0.02,
    quality_action_squared_gain_per_m2=8.0,
    quality_jitter_squared_gain_per_m2=12.0,
    # 四类互斥训练终止信用；安全失败覆盖成功，成功覆盖同一步超时。
    success_terminal_reward=10.0,
    timeout_terminal_reward=-15.0,
    collision_terminal_reward=-25.0,
    limit_terminal_reward=-10.0,
    tool_axis_best_progress_weight=0.5,
    # 多路径训练已取消；历史路径分量固定关闭。
    path_reference_progress_weight_per_m=0.0,
    path_reference_reached_reward=0.0,
    joint_soft_limit_max_penalty=0.01,
)

# RewardManager 会再乘策略步长 dt=0.25 s。访问器返回“每秒率”抵消该乘法，
# 因而这里和纯逻辑测试中的数值就是策略实际收到的逐步奖励。
REWARD_TERM_WEIGHT: Final = 1.0
