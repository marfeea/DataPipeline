"""Helpers for allocating stable episode identifiers across collection runs."""

from __future__ import annotations

from collections.abc import Iterable


def next_episode_index(existing_episode_ids: Iterable[str], episode_prefix: str) -> int:
    """Return the first index after existing numeric IDs with ``episode_prefix``."""

    indices = [
        int(episode_id[len(episode_prefix) :])
        for episode_id in existing_episode_ids
        if episode_id.startswith(episode_prefix) and episode_id[len(episode_prefix) :].isdigit()
    ]
    return max(indices, default=-1) + 1


__all__ = ["next_episode_index"]
