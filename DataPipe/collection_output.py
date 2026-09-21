"""Select a version-isolated output directory before starting Isaac."""

from __future__ import annotations

import json
from pathlib import Path

from .config import ALIGNED_SCHEMA_VERSION, DATASET_SCHEMA_VERSION


def _schema(directory: Path) -> str | None:
    manifest = directory / "manifest.json"
    if manifest.exists():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            version = payload["schema_version"]
            if not isinstance(version, str):
                raise ValueError("schema_version must be a string")
            return version
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Cannot read dataset schema from {manifest}: {exc}") from exc
    if directory.exists():
        if not directory.is_dir():
            raise ValueError(f"Dataset output is not a directory: {directory}")
        if any(directory.glob("*.hdf5*")):
            raise ValueError(f"Dataset shards exist without manifest.json in {directory}; choose a new directory")
    return None


def resolve_collection_output(output_dir: str | Path) -> Path:
    """Resume v4 or redirect a known v3 dataset to a stable sibling ending in _v4.

    This function only reads metadata. It never migrates or modifies old data.
    Malformed or unknown manifests and occupied incompatible siblings fail closed.
    """
    requested = Path(output_dir)
    version = _schema(requested)
    if version in (None, ALIGNED_SCHEMA_VERSION):
        return requested
    if version != DATASET_SCHEMA_VERSION:
        raise ValueError(f"Unsupported dataset schema {version!r} in {requested}; choose a new output directory")
    selected = requested.with_name(requested.name + "_v4")
    selected_version = _schema(selected)
    if selected_version not in (None, ALIGNED_SCHEMA_VERSION):
        raise ValueError(
            f"Cannot isolate v4 output: {selected} already contains schema {selected_version!r}; "
            "choose a new output directory"
        )
    return selected
