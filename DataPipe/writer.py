"""P1-8 原子 JSONL 分片、manifest、校验和、磁盘门禁与续采。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DatasetConfig
from .records import Episode
from .serialization import episode_from_record, episode_to_record


@dataclass(frozen=True, slots=True)
class ShardInfo:
    path: str
    episode_ids: tuple[str, ...]
    sha256: str


class TrajectoryWriter:
    """以可恢复的 JSONL 分片写入完成 episode。"""

    def __init__(self, output_dir: str | Path, config: DatasetConfig | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or DatasetConfig()
        self.config.validate()
        self.manifest_path = self.output_dir / "manifest.json"
        self._manifest: dict[str, Any] = self._load_manifest()
        self._episode_ids = set(self._manifest.get("episode_ids", ()))

    @property
    def episode_ids(self) -> frozenset[str]:
        """已提交 episode ID。"""

        return frozenset(self._episode_ids)

    def write(self, episode: Episode) -> ShardInfo:
        """校验并原子追加一条 episode，重复 ID 直接拒绝。"""

        episode.validate()
        if episode.episode_id in self._episode_ids:
            raise ValueError(f"episode_id 重复：{episode.episode_id}")
        self._check_disk()
        shards = list(self._manifest.get("shards", ()))
        if shards and len(shards[-1]["episode_ids"]) < self.config.episodes_per_shard:
            shard = shards[-1]
            shard_path = self.output_dir / shard["path"]
            records = self._read_records(shard_path)
        else:
            shard = {"path": f"shard-{len(shards):05d}.jsonl", "episode_ids": []}
            shard_path = self.output_dir / shard["path"]
            records = []
            shards.append(shard)
        records.append(episode_to_record(episode))
        payload = "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records)
        self._atomic_write(shard_path, payload)
        checksum = _sha256(shard_path)
        shard["episode_ids"] = [record["episode_id"] for record in records]
        shard["sha256"] = checksum
        self._episode_ids.add(episode.episode_id)
        self._manifest = {
            "schema_version": self.config.schema_version,
            "control_frequency_hz": self.config.control_frequency_hz,
            "episode_count": len(self._episode_ids),
            "episode_ids": sorted(self._episode_ids),
            "shards": shards,
        }
        self._write_manifest()
        return ShardInfo(shard["path"], tuple(shard["episode_ids"]), checksum)

    def load_episodes(self) -> tuple[Episode, ...]:
        """读取 manifest 中的全部已完成 episode。"""

        episodes: list[Episode] = []
        for shard in self._manifest.get("shards", ()):
            path = self.output_dir / shard["path"]
            if _sha256(path) != shard.get("sha256"):
                raise ValueError(f"分片校验和不匹配：{path}")
            episodes.extend(episode_from_record(record) for record in self._read_records(path))
        return tuple(episodes)

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"schema_version": self.config.schema_version, "episode_ids": [], "shards": []}
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != self.config.schema_version:
            raise ValueError("manifest schema 与当前配置不一致")
        return manifest

    def _check_disk(self) -> None:
        free_bytes = shutil.disk_usage(self.output_dir).free
        if free_bytes < self.config.min_free_disk_bytes:
            raise OSError(f"磁盘剩余空间不足：{free_bytes} < {self.config.min_free_disk_bytes}")

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"manifest 引用的分片不存在：{path}")
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _atomic_write(self, path: Path, payload: str) -> None:
        partial = path.with_name(path.name + ".partial")
        partial.write_text(payload, encoding="utf-8", newline="\n")
        os.replace(partial, path)

    def _write_manifest(self) -> None:
        payload = json.dumps(self._manifest, ensure_ascii=False, indent=2) + "\n"
        self._atomic_write(self.manifest_path, payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["ShardInfo", "TrajectoryWriter"]
