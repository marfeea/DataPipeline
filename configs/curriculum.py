"""阶段 E 的版本化课程配置；不复制或修改最终任务阈值。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

FINAL_TASK_THRESHOLDS_ARE_IMMUTABLE: Final = True
"""课程不得修改 ``configs.task`` 中的停靠或安全真值。"""

CURRICULUM_CONFIG_VERSION: Final = "curriculum-v4-joint-delta-p7-position-reward"
"""P7 落地位置到达奖励模型与逐等级课程成功终止。"""

CURRICULUM_V4_SCHEMA_VERSION: Final = "curriculum-v4-joint-delta-p1"
"""有界六轴关节增量动作谱系的固定策略观测接口版本。"""


@dataclass(frozen=True)
class AuxiliaryRewardScales:
    """按课程等级缩放当前实际装配的塑形项；终局项始终保持原权重。"""

    position_reward: float = 1.0
    progress_reward: float = 1.0
    best_progress_reward: float = 1.0
    stall_penalty: float = 1.0
    far_move_reward: float = 1.0
    near_inefficiency_penalty: float = 1.0
    time_penalty: float = 1.0
    joint_soft_limit_penalty: float = 1.0
    tool_axis_progress: float = 1.0
    # 多路径方案已取消；保留字段仅用于兼容固定奖励分量接口。
    path_reference_progress: float = 0.0
    path_reference_reached: float = 0.0


@dataclass(frozen=True)
class StageSuccessCfg:
    """课程阶段成功门槛；不替代 ``configs.task`` 的正式停车真值。"""

    max_position_error_m: float
    max_tcp_speed_m_s: float | None
    required_dwell_steps: int
    max_orientation_error_rad: float | None = None
    require_formal_parking_success: bool = False
    require_path_reference_reached: bool = False


@dataclass(frozen=True)
class CollisionProfileCfg:
    """课程碰撞档的身份和接触过滤范围；不修改 USD 的 CollisionAPI。"""

    identifier: str
    one_hot_index: int
    filter_prim_paths_expr: tuple[str, ...]


@dataclass(frozen=True)
class CurriculumLevelCfg:
    """一个等级可控制的训练分布和辅助奖励，不含任务真值。"""

    target_state_probabilities: tuple[float, ...]
    path_mode_probabilities: tuple[float, float, float]
    action_mask: tuple[float, float, float, float, float, float]
    collision_profile_id: str
    collision_clearance_enabled_mask: tuple[float, float, float]
    auxiliary_rewards: AuxiliaryRewardScales
    stage_success: StageSuccessCfg
    previous_level_sampling_probability: float = 0.0
    approved_randomization: tuple[str, ...] = ()

    @property
    def joint_action_mask(self) -> tuple[float, float, float, float, float, float]:
        """旧 Joint 文档与测试的兼容名称；新代码统一使用 ``action_mask``。"""

        return self.action_mask


@dataclass(frozen=True)
class CurriculumTransitionCfg:
    """滚动窗口、升降级和冷却规则。"""

    rolling_window_episodes: int = 64
    min_episodes_for_transition: int = 32
    promote_success_rate: float = 0.75
    max_promote_safety_failure_rate: float = 0.05
    demote_success_rate: float = 0.30
    max_demote_safety_failure_rate: float = 0.20
    cooldown_episodes: int = 32


@dataclass(frozen=True)
class CurriculumCfg:
    version: str
    observation_schema_version: str
    levels: tuple[CurriculumLevelCfg, ...]
    transition: CurriculumTransitionCfg


@dataclass(frozen=True)
class ObservationFieldCfg:
    """一项固定策略观测的张量契约。"""

    name: str
    dimension: int
    unit: str
    coordinate_frame: str
    normalization: str
    enabled_levels: tuple[int, ...]


@dataclass(frozen=True)
class CurriculumOutcomeSchemaCfg:
    """课程统计与正式任务真值的结果字段及其安全优先级。"""

    curriculum_success_field: str
    formal_parking_success_field: str
    safety_failure_field: str
    precedence: tuple[str, ...]


@dataclass(frozen=True)
class PathModeCfg:
    """历史路径模式接口；当前训练只启用 direct，保留三维槽位兼容观测 shape。"""

    name: str
    one_hot_index: int
    sampling_probability: float
    midpoint_offset_b: tuple[float, float, float]


CURRICULUM_V2_PATH_MODES: Final = (
    PathModeCfg("direct", 0, 1.0, (0.0, 0.0, 0.0)),
    PathModeCfg("lateral_positive", 1, 0.0, (0.0, 0.12, 0.0)),
    PathModeCfg("lateral_negative", 2, 0.0, (0.0, -0.12, 0.0)),
)

DIRECT_ONLY_PATH_MODE_PROBABILITIES: Final = (1.0, 0.0, 0.0)
"""多路径训练已取消；所有课程等级固定使用 direct 占位模式。"""

CURRICULUM_COLLISION_PROFILES: Final = (
    CollisionProfileCfg(
        "C1", 0, ("{ENV_REGEX_NS}/station/static/workstation/WorkStation/Glass/M_Glass",)
    ),
    CollisionProfileCfg(
        "C2", 1, ("{ENV_REGEX_NS}/station/static/workstation/WorkStation/WorkStation/M_WorkStation",)
    ),
    CollisionProfileCfg(
        "C3",
        2,
        (
            "{ENV_REGEX_NS}/station/static/workstation/WorkStation/Glass/M_Glass",
            "{ENV_REGEX_NS}/station/static/workstation/WorkStation/WorkStation/M_WorkStation",
        ),
    ),
)

PATH_REFERENCE_REACH_TOLERANCE_M: Final = 0.04
"""历史路径诊断容差；当前不参与奖励、阶段成功或课程门禁。"""


CURRICULUM_V4_OBSERVATION_SCHEMA: Final = (
    # 关节增量动作谱系保持 52 维形状，但以新 schema 版本拒绝旧 checkpoint。
    ObservationFieldCfg("arm_joint_position", 6, "rad", "joint", "joint_limit_relative", (1, 2, 3, 4, 5)),
    ObservationFieldCfg("arm_joint_velocity", 6, "rad/s", "joint", "velocity_limit_relative", (1, 2, 3, 4, 5)),
    ObservationFieldCfg("tcp_position_error", 3, "m", "B", "workspace_relative", (1, 2, 3, 4, 5)),
    ObservationFieldCfg("tcp_orientation_error", 3, "rad", "B", "pi_relative", (1, 2, 3, 4, 5)),
    ObservationFieldCfg("tcp_linear_velocity", 3, "m/s", "B", "velocity_limit_relative", (1, 2, 3, 4, 5)),
    ObservationFieldCfg("tcp_angular_velocity", 3, "rad/s", "B", "velocity_limit_relative", (1, 2, 3, 4, 5)),
    ObservationFieldCfg("target_state_one_hot", 4, "one_hot", "none", "identity", (1, 2, 3, 4, 5)),
    ObservationFieldCfg("curriculum_level_normalized", 1, "ratio", "none", "[0,1]", (1, 2, 3, 4, 5)),
    # path_mode 保留三维固定占位以维持 52 维 schema；当前始终为 direct=[1,0,0]。
    ObservationFieldCfg("path_mode_one_hot", 3, "one_hot", "B", "identity", (1, 2, 3, 4, 5)),
    ObservationFieldCfg("collision_profile_one_hot", 3, "one_hot", "none", "identity", (1, 2, 3, 4, 5)),
    ObservationFieldCfg("collision_group_min_clearance", 3, "m", "E", "clearance_cap_relative", (4, 5)),
    ObservationFieldCfg("collision_clearance_enabled_mask", 3, "binary", "none", "identity", (1, 2, 3, 4, 5)),
    ObservationFieldCfg("joint_action_mask", 6, "binary", "joint", "identity", (1, 2, 3, 4, 5)),
    ObservationFieldCfg("curriculum_stage_one_hot", 5, "one_hot", "none", "identity", (1, 2, 3, 4, 5)),
)

CURRICULUM_V4_OBSERVATION_DIMENSION: Final = sum(field.dimension for field in CURRICULUM_V4_OBSERVATION_SCHEMA)

CURRICULUM_V4_OUTCOME_SCHEMA: Final = CurriculumOutcomeSchemaCfg(
    curriculum_success_field="curriculum_success",
    formal_parking_success_field="formal_parking_success",
    safety_failure_field="safety_failure",
    precedence=("safety_failure", "formal_parking_success", "curriculum_success"),
)


CURRICULUM_CFG: Final = CurriculumCfg(
    version=CURRICULUM_CONFIG_VERSION,
    observation_schema_version=CURRICULUM_V4_SCHEMA_VERSION,
    levels=(
        CurriculumLevelCfg(
            target_state_probabilities=(1.0, 0.0, 0.0, 0.0, 0.0),
            path_mode_probabilities=DIRECT_ONLY_PATH_MODE_PROBABILITIES,
            action_mask=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            collision_profile_id="C1",
            collision_clearance_enabled_mask=(0.0, 0.0, 0.0),
            # L0 只训练位置到达；不注入工具轴姿态塑形。
            auxiliary_rewards=AuxiliaryRewardScales(tool_axis_progress=0.0),
            # L0 阶段成功只要求位置进入 8 cm 范围并连续保持 3 个策略步。
            stage_success=StageSuccessCfg(0.08, None, 3),
        ),
        CurriculumLevelCfg(
            target_state_probabilities=(1.0, 0.0, 0.0, 0.0, 0.0),
            path_mode_probabilities=DIRECT_ONLY_PATH_MODE_PROBABILITIES,
            action_mask=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            collision_profile_id="C1",
            collision_clearance_enabled_mask=(0.0, 0.0, 0.0),
            auxiliary_rewards=AuxiliaryRewardScales(tool_axis_progress=1.0),
            stage_success=StageSuccessCfg(0.06, 0.06, 2, max_orientation_error_rad=0.3490658503988659),
            previous_level_sampling_probability=0.25,
        ),
        CurriculumLevelCfg(
            target_state_probabilities=(0.2, 0.2, 0.2, 0.2, 0.2),
            path_mode_probabilities=DIRECT_ONLY_PATH_MODE_PROBABILITIES,
            action_mask=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            collision_profile_id="C2",
            collision_clearance_enabled_mask=(0.0, 0.0, 0.0),
            auxiliary_rewards=AuxiliaryRewardScales(),
            stage_success=StageSuccessCfg(0.05, 0.05, 2, max_orientation_error_rad=0.2617993877991494),
            previous_level_sampling_probability=0.25,
        ),
        CurriculumLevelCfg(
            target_state_probabilities=(0.2, 0.2, 0.2, 0.2, 0.2),
            path_mode_probabilities=DIRECT_ONLY_PATH_MODE_PROBABILITIES,
            action_mask=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            collision_profile_id="C3",
            # P4 接入并验证几何最近点前，三个净空字段均视为未启用。
            collision_clearance_enabled_mask=(0.0, 0.0, 0.0),
            auxiliary_rewards=AuxiliaryRewardScales(),
            stage_success=StageSuccessCfg(0.05, 0.05, 2, max_orientation_error_rad=0.2617993877991494),
            previous_level_sampling_probability=0.25,
        ),
        CurriculumLevelCfg(
            target_state_probabilities=(0.2, 0.2, 0.2, 0.2, 0.2),
            path_mode_probabilities=DIRECT_ONLY_PATH_MODE_PROBABILITIES,
            action_mask=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            collision_profile_id="C3",
            # P4 接入并验证几何最近点前，三个净空字段均视为未启用。
            collision_clearance_enabled_mask=(0.0, 0.0, 0.0),
            auxiliary_rewards=AuxiliaryRewardScales(),
            stage_success=StageSuccessCfg(
                0.04,
                0.03,
                2,
                max_orientation_error_rad=0.17453292519943295,
                require_formal_parking_success=True,
            ),
            previous_level_sampling_probability=0.25,
            # 尚无经资产/接触验证批准的物理随机化，故明确保持为空。
            approved_randomization=(),
        ),
    ),
    transition=CurriculumTransitionCfg(),
)
