"""AUBO 数值 IK 训练链的独立课程与观测版本。"""

from __future__ import annotations

from typing import Final

from .curriculum import (
    CURRICULUM_COLLISION_PROFILES,
    CURRICULUM_V2_PATH_MODES,
    DIRECT_ONLY_PATH_MODE_PROBABILITIES,
    AuxiliaryRewardScales,
    CurriculumCfg,
    CurriculumLevelCfg,
    CurriculumOutcomeSchemaCfg,
    CurriculumTransitionCfg,
    ObservationFieldCfg,
    StageSuccessCfg,
)

IK_CURRICULUM_CONFIG_VERSION: Final = "curriculum-aubo-ik-v1"
IK_OBSERVATION_SCHEMA_VERSION: Final = "aubo-ik-observation-v1"

# L0 只解位置，L1 解位置与工具轴，L2-L4 恢复完整 6D 位姿。
IK_MODE_BY_LEVEL: Final = (
    "position_3d",
    "tool_axis_5d",
    "full_pose_6d",
    "full_pose_6d",
    "full_pose_6d",
)

IK_OBSERVATION_SCHEMA: Final = (
    ObservationFieldCfg("arm_joint_position", 6, "rad", "joint", "joint_limit_relative", (0, 1, 2, 3, 4)),
    ObservationFieldCfg("arm_joint_velocity", 6, "rad/s", "joint", "velocity_limit_relative", (0, 1, 2, 3, 4)),
    ObservationFieldCfg("tcp_position_error", 3, "m", "B", "workspace_relative", (0, 1, 2, 3, 4)),
    ObservationFieldCfg("tcp_orientation_error", 3, "rad", "B", "pi_relative", (0, 1, 2, 3, 4)),
    ObservationFieldCfg("tcp_linear_velocity", 3, "m/s", "B", "velocity_limit_relative", (0, 1, 2, 3, 4)),
    ObservationFieldCfg("tcp_angular_velocity", 3, "rad/s", "B", "velocity_limit_relative", (0, 1, 2, 3, 4)),
    ObservationFieldCfg("target_state_one_hot", 4, "one_hot", "none", "identity", (0, 1, 2, 3, 4)),
    ObservationFieldCfg("curriculum_level_normalized", 1, "ratio", "none", "[0,1]", (0, 1, 2, 3, 4)),
    ObservationFieldCfg("path_mode_one_hot", 3, "one_hot", "B", "identity", (0, 1, 2, 3, 4)),
    ObservationFieldCfg("collision_profile_one_hot", 3, "one_hot", "none", "identity", (0, 1, 2, 3, 4)),
    ObservationFieldCfg("collision_group_min_clearance", 3, "m", "E", "clearance_cap_relative", (3, 4)),
    ObservationFieldCfg("collision_clearance_enabled_mask", 3, "binary", "none", "identity", (0, 1, 2, 3, 4)),
    ObservationFieldCfg("ik_action_mask", 6, "binary", "B_task", "identity", (0, 1, 2, 3, 4)),
    ObservationFieldCfg("curriculum_stage_one_hot", 5, "one_hot", "none", "identity", (0, 1, 2, 3, 4)),
)
IK_OBSERVATION_DIMENSION: Final = sum(field.dimension for field in IK_OBSERVATION_SCHEMA)

IK_OUTCOME_SCHEMA: Final = CurriculumOutcomeSchemaCfg(
    curriculum_success_field="curriculum_success",
    formal_parking_success_field="formal_parking_success",
    safety_failure_field="safety_failure",
    precedence=("safety_failure", "formal_parking_success", "curriculum_success"),
)

IK_CURRICULUM_CFG: Final = CurriculumCfg(
    version=IK_CURRICULUM_CONFIG_VERSION,
    observation_schema_version=IK_OBSERVATION_SCHEMA_VERSION,
    levels=(
        CurriculumLevelCfg(
            target_state_probabilities=(1.0, 0.0, 0.0, 0.0),
            path_mode_probabilities=DIRECT_ONLY_PATH_MODE_PROBABILITIES,
            action_mask=(1.0, 1.0, 1.0, 0.0, 0.0, 0.0),
            collision_profile_id="C1",
            collision_clearance_enabled_mask=(0.0, 0.0, 0.0),
            auxiliary_rewards=AuxiliaryRewardScales(tool_axis_progress=0.0),
            stage_success=StageSuccessCfg(0.08, None, 3),
        ),
        CurriculumLevelCfg(
            target_state_probabilities=(1.0, 0.0, 0.0, 0.0),
            path_mode_probabilities=DIRECT_ONLY_PATH_MODE_PROBABILITIES,
            action_mask=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            collision_profile_id="C1",
            collision_clearance_enabled_mask=(0.0, 0.0, 0.0),
            auxiliary_rewards=AuxiliaryRewardScales(tool_axis_progress=1.0),
            stage_success=StageSuccessCfg(0.06, 0.06, 2, max_orientation_error_rad=0.3490658503988659),
            previous_level_sampling_probability=0.25,
        ),
        CurriculumLevelCfg(
            target_state_probabilities=(0.25, 0.25, 0.25, 0.25),
            path_mode_probabilities=DIRECT_ONLY_PATH_MODE_PROBABILITIES,
            action_mask=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            collision_profile_id="C2",
            collision_clearance_enabled_mask=(0.0, 0.0, 0.0),
            auxiliary_rewards=AuxiliaryRewardScales(),
            stage_success=StageSuccessCfg(0.05, 0.05, 2, max_orientation_error_rad=0.2617993877991494),
            previous_level_sampling_probability=0.25,
        ),
        CurriculumLevelCfg(
            target_state_probabilities=(0.25, 0.25, 0.25, 0.25),
            path_mode_probabilities=DIRECT_ONLY_PATH_MODE_PROBABILITIES,
            action_mask=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            collision_profile_id="C3",
            collision_clearance_enabled_mask=(0.0, 0.0, 0.0),
            auxiliary_rewards=AuxiliaryRewardScales(),
            stage_success=StageSuccessCfg(0.05, 0.05, 2, max_orientation_error_rad=0.2617993877991494),
            previous_level_sampling_probability=0.25,
        ),
        CurriculumLevelCfg(
            target_state_probabilities=(0.25, 0.25, 0.25, 0.25),
            path_mode_probabilities=DIRECT_ONLY_PATH_MODE_PROBABILITIES,
            action_mask=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            collision_profile_id="C3",
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
        ),
    ),
    transition=CurriculumTransitionCfg(),
)

__all__ = [
    "CURRICULUM_COLLISION_PROFILES",
    "CURRICULUM_V2_PATH_MODES",
    "IK_CURRICULUM_CFG",
    "IK_CURRICULUM_CONFIG_VERSION",
    "IK_MODE_BY_LEVEL",
    "IK_OBSERVATION_DIMENSION",
    "IK_OBSERVATION_SCHEMA",
    "IK_OBSERVATION_SCHEMA_VERSION",
    "IK_OUTCOME_SCHEMA",
]
