"""P0-4 当前 IK 场景基线的可审计描述。"""

from __future__ import annotations

from dataclasses import dataclass

from .config import CONTROL_FREQUENCY_HZ


@dataclass(frozen=True, slots=True)
class SceneBaseline:
    """描述采集入口必须复用的现有静态场景事实。"""

    task_id: str
    scene_cfg: str
    source_script: str
    target_state_names: tuple[str, ...]
    control_frequency_hz: float = CONTROL_FREQUENCY_HZ
    target_selection: str = "TARGET_STATES[target_state_index]"
    ik_execution_semantics: str = "TCP B 系路径 -> Flange 向下 IK -> 关节目标 -> 物理步执行"

    def validate(self) -> None:
        """检查基线描述没有出现空契约。"""

        if not self.task_id or not self.scene_cfg or not self.source_script:
            raise ValueError("场景基线必须包含任务、场景配置和来源脚本")
        if not self.target_state_names or len(set(self.target_state_names)) != len(self.target_state_names):
            raise ValueError("target_state_names 必须非空且唯一")
        if self.control_frequency_hz <= 0.0:
            raise ValueError("场景基线控制频率必须为正数")


STATIC_IK_SCENE_BASELINE = SceneBaseline(
    task_id="CurriculumRL-TcpDocking-IK-v0",
    scene_cfg="CurriculumRL.tasks.tcp_docking.static_scene_cfg.TcpDockingStaticSceneCfg",
    source_script="scripts/test_aubo_ik_scene.py",
    target_state_names=(
        "sample_bottle_state_01",
        "sample_bottle_state_02",
        "sample_bottle_state_03",
        "sample_bottle_state_04",
        "sample_bottle_state_05",
    ),
)
STATIC_IK_SCENE_BASELINE.validate()


__all__ = ["STATIC_IK_SCENE_BASELINE", "SceneBaseline"]
