"""P0-6 终局监督适配：只消费正式环境结果，不复制阈值。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .records import TerminationReason


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    """正式环境在 reset 前给出的终局结果。"""

    success: bool
    timeout: bool
    safety_failure: bool
    metrics: dict[str, float] = field(default_factory=dict)
    primary_reason: str | None = None

    def termination_reason(self) -> TerminationReason:
        """按正式终止优先级映射为 P0 终局类别。"""

        if self.safety_failure:
            return TerminationReason.SAFETY_FAILURE
        if self.success:
            return TerminationReason.SUCCESS
        if self.timeout:
            return TerminationReason.TIMEOUT
        raise ValueError("终局结果必须至少包含 success、timeout 或 safety_failure")


class EpisodeSupervisor:
    """将正式环境 outcome 交给 EpisodeBuilder。"""

    def __init__(self) -> None:
        self._submitted = False

    @property
    def submitted(self) -> bool:
        """当前 episode 是否已经提交终局。"""

        return self._submitted

    def submit(self, builder: Any, outcome: EpisodeOutcome, terminal_observation: Any):
        """只消费结果并提交一次，不在采集层重算阈值。"""

        if self._submitted:
            raise RuntimeError("同一 episode 不能重复提交终局")
        episode = builder.close(outcome.termination_reason(), terminal_observation)
        self._submitted = True
        return episode

    def reset(self) -> None:
        """为下一条 episode 清空提交门闩。"""

        self._submitted = False


__all__ = ["EpisodeOutcome", "EpisodeSupervisor"]
