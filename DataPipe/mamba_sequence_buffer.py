"""Fixed-window HDF5 loader for the AUBO/R2Dreamer boundary contract."""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .config import ALIGNED_SCHEMA_VERSION


def window_count(transition_count: int, horizon: int = 16) -> int:
    """Return the number of legal fixed windows for one episode."""

    if transition_count < 0 or horizon <= 0:
        raise ValueError("transition_count must be non-negative and horizon must be positive")
    return max(0, transition_count - horizon + 1)


class MambaSequenceBuffer:
    """Return ``horizon + 1`` observations and ``horizon`` transitions.

    ``_max_start`` is intentionally ``T - horizon``.  Therefore an episode
    with exactly 16 transitions exposes one legal window at start 0.
    """

    def __init__(
        self,
        dataset: str | Path | Sequence[str | Path],
        *,
        horizon: int = 16,
        split: str | None = None,
        state_keys: Sequence[str] = (),
        image_keys: Sequence[str] = (),
        seed: int = 0,
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.horizon = int(horizon)
        self.split = split
        self.state_keys = tuple(state_keys)
        self.image_keys = tuple(image_keys)
        self._random = random.Random(seed)
        self._h5py = self._import_h5py()
        self.paths = self._resolve_paths(dataset)
        self._files = [self._h5py.File(path, "r") for path in self.paths]
        self._demo_lookup = self._build_demo_lookup()
        if not self._demo_lookup:
            self.close()
            raise ValueError("dataset contains no demos for the selected split")
        self.window_indices = tuple(
            (file_index, name, start)
            for file_index, name in self._demo_lookup
            for start in range(self._max_start(file_index, name) + 1)
        )
        if not self.window_indices:
            self.close()
            raise ValueError(f"no episode has horizon={self.horizon} transitions")

    @staticmethod
    def _import_h5py():
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("MambaSequenceBuffer requires h5py") from exc
        return h5py

    def _resolve_paths(self, dataset: str | Path | Sequence[str | Path]) -> tuple[Path, ...]:
        if isinstance(dataset, str | Path):
            root = Path(dataset)
            if root.is_dir():
                manifest = root / "manifest.json"
                if manifest.exists():
                    import json

                    payload = json.loads(manifest.read_text(encoding="utf-8"))
                    paths = tuple(root / shard["path"] for shard in payload.get("shards", ()))
                else:
                    paths = tuple(sorted(root.glob("shard-*.hdf5")))
            else:
                paths = (root,)
        else:
            paths = tuple(Path(item) for item in dataset)
        if not paths or any(not path.exists() for path in paths):
            raise FileNotFoundError("one or more HDF5 dataset paths do not exist")
        return paths

    def _build_demo_lookup(self) -> tuple[tuple[int, str], ...]:
        lookup = []
        for file_index, handle in enumerate(self._files):
            for name in sorted(handle["data"].keys()):
                demo = handle["data"][name]
                if handle["data"].attrs.get("schema_version") == ALIGNED_SCHEMA_VERSION:
                    from .aligned_schema import validate_reward_mask

                    try:
                        validate_reward_mask(demo)
                    except ValueError:
                        self.close()
                        raise
                if self.split is not None and str(demo.attrs.get("split", "")) != self.split:
                    continue
                lookup.append((file_index, name))
        return tuple(lookup)

    def _demo(self, file_index: int, name: str) -> Any:
        return self._files[file_index]["data"][name]

    def _max_start(self, file_index: int, name: str) -> int:
        length = int(self._demo(file_index, name)["actions"].shape[0])
        return length - self.horizon

    def close(self) -> None:
        for handle in getattr(self, "_files", ()):
            handle.close()

    def __enter__(self) -> MambaSequenceBuffer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def sample(self, batch_size: int = 1) -> dict[str, Any]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        windows = [self._random.choice(self.window_indices) for _ in range(batch_size)]
        return self._stack(windows)

    def iter_fixed_batches(self, batch_size: int = 1, max_batches: int = 0) -> Iterator[dict[str, Any]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        windows: Iterable[tuple[int, str, int]] = self.window_indices
        if max_batches > 0:
            windows = tuple(windows)[: batch_size * max_batches]
        windows = tuple(windows)
        for offset in range(0, len(windows), batch_size):
            yield self._stack(windows[offset : offset + batch_size])

    def _stack(self, windows: Sequence[tuple[int, str, int]]) -> dict[str, Any]:
        examples = [self._sample_window(*window) for window in windows]
        observation_keys = tuple(examples[0]["observations"])
        observations = {
            key: np.stack([example["observations"][key] for example in examples], axis=0) for key in observation_keys
        }
        result: dict[str, Any] = {
            "observations": observations,
            "actions": np.stack([example["actions"] for example in examples], axis=0),
            "rewards": np.stack([example["rewards"] for example in examples], axis=0)[..., None],
            "reward_valid": np.stack([example["reward_valid"] for example in examples], axis=0)[..., None],
            "dones": np.stack([example["dones"] for example in examples], axis=0)[..., None],
            "timeouts": np.stack([example["timeouts"] for example in examples], axis=0)[..., None],
            "is_terminal": np.stack([example["is_terminal"] for example in examples], axis=0)[..., None],
        }
        if self.state_keys:
            state_values = [observations[key].reshape(len(examples), self.horizon + 1, -1) for key in self.state_keys]
            result["state"] = np.concatenate(state_values, axis=-1).astype(np.float32)
        if self.image_keys:
            result["image"] = np.concatenate([observations[key] for key in self.image_keys], axis=-1)
        return result

    def _sample_window(self, file_index: int, name: str, start: int) -> dict[str, Any]:
        demo = self._demo(file_index, name)
        if start < 0 or start > self._max_start(file_index, name):
            raise IndexError("window start is outside the legal 0..T-horizon range")
        observation_group = demo["obs"]
        observations = {
            key: np.asarray(observation_group[key][start : start + self.horizon + 1])
            for key in observation_group
            if key != "camera_calibration"
        }
        return {
            "observations": observations,
            "actions": np.asarray(demo["actions"][start : start + self.horizon], dtype=np.float32),
            "rewards": np.asarray(demo["rewards"][start : start + self.horizon], dtype=np.float32),
            "reward_valid": (
                np.asarray(demo["reward_valid"][start : start + self.horizon], dtype=bool)
                if "reward_valid" in demo
                else np.ones(self.horizon, dtype=bool)
            ),
            "dones": np.asarray(demo["dones"][start : start + self.horizon], dtype=np.uint8),
            "timeouts": np.asarray(demo["timeouts"][start : start + self.horizon], dtype=np.uint8),
            "is_terminal": np.asarray(demo["is_terminal"][start : start + self.horizon], dtype=np.uint8),
        }


__all__ = ["MambaSequenceBuffer", "window_count"]
