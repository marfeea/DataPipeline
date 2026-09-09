"""轨迹选择和 state/action 两种模式的无渲染 replay 数据接口。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .hdf5_writer import HDF5TrajectoryWriter
from .records import Episode


@dataclass(frozen=True, slots=True)
class ReplaySelection:
    """回放器选择出的单条完整轨迹。"""

    episode: Episode
    mode: str

    def frames(self):
        """state 模式逐帧返回观测。"""

        if self.mode != "state":
            raise ValueError("frames 只适用于 state 模式")
        return self.episode.observations

    def steps(self):
        """action 模式逐步返回动作/奖励。"""

        if self.mode != "action":
            raise ValueError("steps 只适用于 action 模式")
        return self.episode.steps


def select_episode(output_dir: str | Path, episode_id: str, mode: str = "state") -> ReplaySelection:
    """按 manifest 选择轨迹并检查 replay mode。"""

    if mode not in {"state", "action"}:
        raise ValueError("mode 必须是 state 或 action")
    episodes = HDF5TrajectoryWriter(output_dir).load_episodes()
    for episode in episodes:
        if episode.episode_id == episode_id:
            return ReplaySelection(episode, mode)
    raise KeyError(f"manifest 中没有 episode：{episode_id}")


__all__ = ["ReplaySelection", "select_episode"]
