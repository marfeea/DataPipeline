"""R2Dreamer 所需的 episode batch 适配层。

仓库未包含 R2Dreamer 训练代码，因此这里提供不绑定外部库的 B x T 结构，
并在边界上保留后续 TensorDict loader 所需的键名。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .records import Episode


@dataclass(frozen=True, slots=True)
class R2DreamerBatch:
    """一批 episode 序列；每个 episode 可以独立保留变长 T。"""

    observations: tuple[tuple[Any, ...], ...]
    actions: tuple[tuple[tuple[float, ...], ...], ...]
    rewards: tuple[tuple[float, ...], ...]
    is_first: tuple[tuple[bool, ...], ...]
    is_last: tuple[tuple[bool, ...], ...]
    is_terminal: tuple[tuple[bool, ...], ...]
    reward_valid: tuple[tuple[bool, ...], ...]

    @classmethod
    def from_episodes(cls, episodes: tuple[Episode, ...] | list[Episode]) -> R2DreamerBatch:
        """从已校验 episode 生成 batch。"""

        return cls(
            observations=tuple(tuple(frame.value for frame in episode.observations) for episode in episodes),
            actions=tuple(tuple(step.action.action for step in episode.steps) for episode in episodes),
            rewards=tuple(tuple(step.reward for step in episode.steps) for episode in episodes),
            reward_valid=tuple(tuple(step.reward_valid for step in episode.steps) for episode in episodes),
            is_first=tuple(tuple(frame.boundary.is_first for frame in episode.observations) for episode in episodes),
            is_last=tuple(tuple(frame.boundary.is_last for frame in episode.observations) for episode in episodes),
            is_terminal=tuple(
                tuple(frame.boundary.is_terminal for frame in episode.observations) for episode in episodes
            ),
        )

    @property
    def batch_size(self) -> int:
        return len(self.observations)

    def validate(self) -> None:
        """检查 B、T/T+1 和边界长度。"""

        for index in range(self.batch_size):
            t_plus_one = len(self.observations[index])
            if len(self.reward_valid[index]) != len(self.actions[index]):
                raise ValueError(f"batch episode {index} reward_valid must have T samples")
            if len(self.actions[index]) + 1 != t_plus_one or len(self.rewards[index]) != len(self.actions[index]):
                raise ValueError(f"batch episode {index} 的 T/T+1 长度不一致")
            if any(
                len(values) != t_plus_one
                for values in (self.is_first[index], self.is_last[index], self.is_terminal[index])
            ):
                raise ValueError(f"batch episode {index} 的边界长度不一致")

    def as_mapping(self) -> dict[str, Any]:
        """返回 TensorDict loader 对齐的键名。"""

        self.validate()
        return {
            "observation": self.observations,
            "action": self.actions,
            "reward": self.rewards,
            "reward_valid": self.reward_valid,
            "is_first": self.is_first,
            "is_last": self.is_last,
            "is_terminal": self.is_terminal,
        }


__all__ = ["R2DreamerBatch"]
