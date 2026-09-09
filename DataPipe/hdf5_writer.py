"""Atomic, resumable HDF5 shard writer for formal training data."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

from .config import DATASET_SCHEMA_VERSION, DatasetConfig
from .hdf5_schema import observation_keys, write_episode_group
from .records import Episode


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class HDF5TrajectoryWriter:
    """Write episode groups to ``shard-xxxxx.hdf5`` and atomically update a manifest."""

    def __init__(
        self,
        output_dir: str | Path,
        config: DatasetConfig | None = None,
        *,
        dataset_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or DatasetConfig()
        self.config.validate()
        self.dataset_metadata = dict(dataset_metadata or {})
        self.manifest_path = self.output_dir / "manifest.json"
        self._manifest = self._load_manifest()
        self._episode_ids = set(self._manifest.get("episode_ids", ()))

    @property
    def episode_ids(self) -> frozenset[str]:
        return frozenset(self._episode_ids)

    def write(self, episode: Episode, *, metadata: Mapping[str, Any] | None = None) -> str:
        """Append an episode, rewriting only the current shard via ``.partial``."""

        episode.validate()
        observation_keys(episode)
        if episode.episode_id in self._episode_ids:
            raise ValueError(f"duplicate episode_id: {episode.episode_id}")
        self._check_disk()
        try:
            import h5py
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("HDF5 export requires h5py") from exc

        shards = list(self._manifest.get("shards", ()))
        if shards and len(shards[-1]["episode_ids"]) < self.config.episodes_per_shard:
            shard = dict(shards[-1])
            shard_index = len(shards) - 1
            existing_path = self.output_dir / shard["path"]
            shards[shard_index] = shard
        else:
            shard_index = len(shards)
            shard = {"path": f"shard-{shard_index:05d}.hdf5", "episode_ids": []}
            existing_path = None
            shards.append(shard)

        final_path = self.output_dir / shard["path"]
        partial_path = final_path.with_name(final_path.name + ".partial")
        if partial_path.exists():
            partial_path.unlink()
        with h5py.File(partial_path, "w") as handle:
            data = handle.create_group("data")
            attrs = {
                "schema_version": self.config.schema_version,
                "episodes": len(shard["episode_ids"]) + 1,
                "control_frequency_hz": self.config.control_frequency_hz,
                "physics_dt_s": self.config.physics_dt_s,
                "action_position_scale_m": self.config.action_position_scale_m,
                "action_rotation_scale_rad": self.config.action_rotation_scale_rad,
                "action_schema": "aubo_tcp_delta_xyzrxyz_plus_tcp_heading_plus_gripper_normalized8",
                "observation_schema": "explicit_aubo_physical_fields",
                "reward_type": self.config.reward_type,
                "camera_calibration": json.dumps(
                    [
                        {
                            "name": camera.name,
                            "prim_path": camera.prim_path,
                            "resolution": camera.resolution,
                            "data_types": camera.data_types,
                            "position_e": camera.position_e,
                            "rotation_wxyz": camera.rotation_wxyz,
                            "pose_convention": camera.pose_convention,
                        }
                        for camera in self.config.cameras
                    ],
                    ensure_ascii=False,
                ),
                **self.dataset_metadata,
            }
            for key, value in attrs.items():
                if value is not None:
                    data.attrs[str(key)] = value
            if existing_path is not None:
                with h5py.File(existing_path, "r") as previous:
                    for name in previous["data"]:
                        previous["data"].copy(name, data)
            demo_name = f"demo_{episode.episode_id}"
            if demo_name in data:
                raise ValueError(f"duplicate episode_id in shard: {episode.episode_id}")
            write_episode_group(data.create_group(demo_name), episode, metadata=metadata)
            data.attrs["episodes"] = len(data) - 0
            data.attrs["total"] = sum(int(data[name].attrs["num_samples"]) for name in data)
            handle.flush()
        os.replace(partial_path, final_path)

        shard["episode_ids"] = [*shard["episode_ids"], episode.episode_id]
        shard["sha256"] = _sha256(final_path)
        self._episode_ids.add(episode.episode_id)
        self._manifest = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "episode_count": len(self._episode_ids),
            "episode_ids": sorted(self._episode_ids),
            "shards": shards,
        }
        self._atomic_json_write(self.manifest_path, self._manifest)
        return shard["path"]

    def load_episodes(
        self,
        *,
        episode_ids: Collection[str] | None = None,
        observation_keys: Collection[str] | None = None,
    ) -> tuple[Episode, ...]:
        """Load formal episodes, optionally filtering IDs and observation fields."""

        from .hdf5_schema_reader import episode_from_hdf5_group

        selected_episode_ids = None if episode_ids is None else frozenset(str(value) for value in episode_ids)
        episodes: list[Episode] = []
        for shard in self._manifest.get("shards", ()):
            path = self.output_dir / shard["path"]
            if _sha256(path) != shard.get("sha256"):
                raise ValueError(f"checksum mismatch: {path}")
            import h5py

            with h5py.File(path, "r") as handle:
                for name in handle["data"]:
                    group = handle["data"][name]
                    episode_id = group.attrs.get("episode_id", name.removeprefix("demo_"))
                    if isinstance(episode_id, bytes):
                        episode_id = episode_id.decode("utf-8")
                    if selected_episode_ids is not None and str(episode_id) not in selected_episode_ids:
                        continue
                    episodes.append(episode_from_hdf5_group(group, observation_keys=observation_keys))
        return tuple(episodes)

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"schema_version": self.config.schema_version, "episode_ids": [], "shards": []}
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest_schema = manifest.get("schema_version", "<missing>")
        if manifest_schema != self.config.schema_version:
            raise ValueError(
                f"incompatible dataset schema in {self.manifest_path}: "
                f"found {manifest_schema!r}, expected {self.config.schema_version!r}. "
                "Use a new empty output directory for the current schema, "
                "or migrate the old dataset explicitly; schemas must not be mixed."
            )
        return manifest

    def _check_disk(self) -> None:
        if shutil.disk_usage(self.output_dir).free < self.config.min_free_disk_bytes:
            raise OSError("insufficient free disk space for HDF5 shard")

    @staticmethod
    def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
        partial = path.with_name(path.name + ".partial")
        partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(partial, path)


__all__ = ["HDF5TrajectoryWriter"]
