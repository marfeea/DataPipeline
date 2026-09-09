"""独立数据集校验器。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DATASET_SCHEMA_VERSION
from .hdf5_schema import validate_episode_group
from .serialization import episode_from_record
from .writer import _sha256


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """validator 的结构化结果。"""

    episode_count: int
    shard_count: int
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors


def validate_dataset(output_dir: str | Path) -> ValidationReport:
    """检查 manifest、分片、episode 时序、边界和动作范围。"""

    root = Path(output_dir)
    errors: list[str] = []
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return ValidationReport(0, 0, ("缺少 manifest.json",))
    try:
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - 具体 JSON 错误由报告承载
        return ValidationReport(0, 0, (f"manifest 无法解析：{exc}",))
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        errors.append("manifest schema_version 不匹配")
    seen: set[str] = set()
    episode_count = 0
    for shard in manifest.get("shards", ()):
        path = root / shard.get("path", "")
        if not path.exists():
            errors.append(f"分片不存在：{path}")
            continue
        if _sha256(path) != shard.get("sha256"):
            errors.append(f"分片校验和不匹配：{path.name}")
        try:
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except Exception as exc:
            errors.append(f"分片无法解析 {path.name}：{exc}")
            continue
        for record in records:
            episode_count += 1
            episode_id = str(record.get("episode_id", "<missing>"))
            if episode_id in seen:
                errors.append(f"重复 episode_id：{episode_id}")
            seen.add(episode_id)
            try:
                episode_from_record(record)
            except Exception as exc:
                errors.append(f"episode={episode_id} 校验失败：{exc}")
    if episode_count != int(manifest.get("episode_count", -1)):
        errors.append("manifest episode_count 与分片数量不一致")
    if seen != set(manifest.get("episode_ids", ())):
        errors.append("manifest episode_ids 与分片内容不一致")
    return ValidationReport(episode_count, len(manifest.get("shards", ())), tuple(errors))


def validate_hdf5_dataset(output_dir: str | Path) -> ValidationReport:
    """Validate formal HDF5 shards, manifest checksums, and T/T+1 axes."""

    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return ValidationReport(0, 0, ("missing manifest.json",))
    errors: list[str] = []
    try:
        import h5py

        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - dependency/IO specific
        return ValidationReport(0, 0, (f"cannot read HDF5 manifest: {exc}",))
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        errors.append("manifest schema_version mismatch")
    seen: set[str] = set()
    episode_count = 0
    for shard in manifest.get("shards", ()):
        path = root / str(shard.get("path", ""))
        if path.suffix != ".hdf5":
            errors.append(f"non-HDF5 shard in formal manifest: {path.name}")
            continue
        if not path.exists():
            errors.append(f"missing HDF5 shard: {path}")
            continue
        if _sha256(path) != shard.get("sha256"):
            errors.append(f"HDF5 checksum mismatch: {path.name}")
        try:
            with h5py.File(path, "r") as handle:
                data = handle["data"]
                if data.attrs.get("schema_version") != DATASET_SCHEMA_VERSION:
                    errors.append(f"{path.name}: data schema_version mismatch")
                for name in data:
                    episode_count += 1
                    if name in seen:
                        errors.append(f"duplicate demo group: {name}")
                    seen.add(name.removeprefix("demo_"))
                    errors.extend(f"{name}: {error}" for error in validate_episode_group(data[name]))
        except Exception as exc:
            errors.append(f"cannot read {path.name}: {exc}")
    if episode_count != int(manifest.get("episode_count", -1)):
        errors.append("manifest episode_count mismatch")
    if seen != set(manifest.get("episode_ids", ())):
        errors.append("manifest episode_ids mismatch")
    return ValidationReport(episode_count, len(manifest.get("shards", ())), tuple(errors))


__all__ = ["ValidationReport", "validate_dataset", "validate_hdf5_dataset"]
