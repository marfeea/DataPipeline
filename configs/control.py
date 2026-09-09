"""TCP 停靠双控制链的任务身份与版本化契约。"""

from __future__ import annotations

from typing import Any, Final

LEGACY_JOINT_TASK_ID: Final = "CurriculumRL-TcpDocking-v0"
JOINT_TASK_ID: Final = "CurriculumRL-TcpDocking-Joint-v0"
AUBO_IK_TASK_ID: Final = "CurriculumRL-TcpDocking-IK-v0"

JOINT_CONTROLLER_FAMILY: Final = "joint"
AUBO_IK_CONTROLLER_FAMILY: Final = "aubo_ik"

JOINT_ACTION_SCHEMA_VERSION: Final = "joint-delta-v1"
AUBO_IK_ACTION_SCHEMA_VERSION: Final = "aubo-ik-tcp-increment-v1"
POLICY_ACTION_DIMENSION: Final = 6
POLICY_ACTION_LOW: Final = -1.0
POLICY_ACTION_HIGH: Final = 1.0

TCP_DOCKING_TASK_IDS: Final = frozenset((LEGACY_JOINT_TASK_ID, JOINT_TASK_ID, AUBO_IK_TASK_ID))
JOINT_TASK_IDS: Final = frozenset((LEGACY_JOINT_TASK_ID, JOINT_TASK_ID))


def is_tcp_docking_task(task_id: str) -> bool:
    return task_id in TCP_DOCKING_TASK_IDS


def controller_family_for_task(task_id: str) -> str:
    if task_id in JOINT_TASK_IDS:
        return JOINT_CONTROLLER_FAMILY
    if task_id == AUBO_IK_TASK_ID:
        return AUBO_IK_CONTROLLER_FAMILY
    raise ValueError(f"未知 TCP 停靠任务：{task_id}")


def action_schema_for_controller(controller_family: str) -> str:
    if controller_family == JOINT_CONTROLLER_FAMILY:
        return JOINT_ACTION_SCHEMA_VERSION
    if controller_family == AUBO_IK_CONTROLLER_FAMILY:
        return AUBO_IK_ACTION_SCHEMA_VERSION
    raise ValueError(f"未知控制器谱系：{controller_family}")


def curriculum_config_for_controller(controller_family: str) -> Any:
    """延迟取得对应课程，避免配置模块形成循环导入。"""

    if controller_family == JOINT_CONTROLLER_FAMILY:
        from .curriculum import CURRICULUM_CFG

        return CURRICULUM_CFG
    if controller_family == AUBO_IK_CONTROLLER_FAMILY:
        from .curriculum_ik import IK_CURRICULUM_CFG

        return IK_CURRICULUM_CFG
    raise ValueError(f"未知控制器谱系：{controller_family}")


__all__ = [
    "AUBO_IK_ACTION_SCHEMA_VERSION",
    "AUBO_IK_CONTROLLER_FAMILY",
    "AUBO_IK_TASK_ID",
    "JOINT_ACTION_SCHEMA_VERSION",
    "JOINT_CONTROLLER_FAMILY",
    "JOINT_TASK_ID",
    "JOINT_TASK_IDS",
    "LEGACY_JOINT_TASK_ID",
    "POLICY_ACTION_DIMENSION",
    "POLICY_ACTION_HIGH",
    "POLICY_ACTION_LOW",
    "TCP_DOCKING_TASK_IDS",
    "action_schema_for_controller",
    "controller_family_for_task",
    "curriculum_config_for_controller",
    "is_tcp_docking_task",
]
