"""数据产线统一 schema 入口。"""

from __future__ import annotations

from typing import Any

from .records import Episode


def validate_episode(episode: Episode) -> None:
    """执行 P0-1 至 P0-3 的 episode 校验。"""

    episode.validate()


def validate_step_fields(episode: Episode, timestep: int) -> dict[str, Any]:
    """返回并校验单控制步字段，供诊断和验收入口使用。"""

    if not 0 <= timestep < episode.length:
        raise IndexError(f"timestep 超出 episode 范围：{timestep}")
    return episode.step_fields(timestep)


__all__ = ["validate_episode", "validate_step_fields"]
