"""Locate subprocess stages shared by historical tools and the production chain."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def script_path(name: str | Path) -> Path:
    """Resolve one unambiguous project script without importing or launching it."""
    name = Path(name)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError(f"Expected a path relative to scripts/: {name}")
    candidates = [ROOT / directory / name for directory in ("scripts", "testscripts")]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(f"Expected exactly one script for {name}, found {found}")
    return found[0]
