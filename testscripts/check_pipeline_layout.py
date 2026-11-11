"""Check production/experiment boundaries offline, without starting Isaac Sim."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from testscripts.script_paths import script_path


def module_path(name):
    path = ROOT.joinpath(*name.split("."))
    if path.with_suffix(".py").is_file():
        return path.with_suffix(".py")
    if (path / "__init__.py").is_file():
        return path / "__init__.py"
    return None


def check(*, verify_cleanup_hashes=False):
    manifest = json.loads((ROOT / "docs/pipeline_layout_20261015.json").read_text(encoding="utf-8"))
    errors = []
    retained = {item["path"] for item in manifest["retained"]}
    actual = {p.relative_to(ROOT).as_posix() for p in (ROOT / "scripts").rglob("*.py")}
    if actual != retained:
        errors.append(f"Production inventory changed: added={sorted(actual - retained)}, missing={sorted(retained - actual)}")
    configuration = {item["path"] for item in manifest.get("configuration", [])}
    actual_configuration = {p.relative_to(ROOT).as_posix() for p in (ROOT / "configs").rglob("*.py")}
    if configuration and actual_configuration != configuration:
        errors.append(f"Configuration inventory changed: added={sorted(actual_configuration - configuration)}, missing={sorted(configuration - actual_configuration)}")
    relocations = manifest["moved"] + manifest.get("configuration_moves", [])
    for item in relocations:
        if (ROOT / item["from"]).exists() or not (ROOT / item["to"]).is_file():
            errors.append(f"Incomplete relocation: {item['from']} -> {item['to']}")
    for item in manifest["retained"] + manifest.get("configuration", []):
        path = ROOT / item["path"]
        if verify_cleanup_hashes and path.is_file():
            if hashlib.sha256(path.read_bytes()).hexdigest() != item.get("sha256_current", item.get("sha256_before")):
                errors.append(f"Production source differs from recorded layout baseline: {item['path']}")

    old_paths = [item["from"] for item in relocations]
    moved_modules = {".".join(Path(old).with_suffix("").parts) for old in old_paths}
    sources = [p for folder in ("scripts", "testscripts", "configs") for p in (ROOT / folder).rglob("*.py")]
    for path in sources:
        relative = path.relative_to(ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except SyntaxError as error:
            errors.append(f"{relative}: {error}")
            continue
        package = ".".join(path.relative_to(ROOT).parts[:-1])
        production = relative.startswith(("scripts/", "configs/"))
        for node in ast.walk(tree):
            imports = []
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = "." * node.level + (node.module or "")
                imports = [importlib.util.resolve_name(base, package) if node.level else base]
            for name in imports:
                if production and name.split(".")[0] == "testscripts":
                    errors.append(f"{relative}:{node.lineno}: production imports experimental module {name}")
                if relative.startswith("configs/") and name.split(".")[0] == "scripts":
                    errors.append(f"{relative}:{node.lineno}: configuration imports runtime module {name}")
                if name.split(".")[0] in ("scripts", "testscripts", "configs") and module_path(name) is None:
                    errors.append(f"{relative}:{node.lineno}: missing local import {name}")
            if isinstance(node, ast.ImportFrom):
                base = "." * node.level + (node.module or "")
                base = importlib.util.resolve_name(base, package) if node.level else base
                for alias in node.names:
                    if base + "." + alias.name in moved_modules:
                        errors.append(f"{relative}:{node.lineno}: obsolete import of moved module {alias.name}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for old in old_paths:
                    if re.search(r"(?<![\w])" + re.escape(old) + r"(?![\w])", node.value):
                        errors.append(f"{relative}:{node.lineno}: obsolete script path {old}")
                if production and re.search(r"(?<![\w])testscripts[/.]", node.value):
                    errors.append(f"{relative}:{node.lineno}: production references experiments")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "script_path":
                if node.args and isinstance(node.args[0], ast.Constant):
                    try:
                        script_path(node.args[0].value)
                    except (ValueError, FileNotFoundError) as error:
                        errors.append(f"{relative}:{node.lineno}: {error}")

    # Verify the exact subprocess names consumed by the production orchestrator.
    tree = ast.parse((ROOT / "scripts/collect_isaacsim.py").read_text(encoding="utf-8"))
    stages = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "invoke":
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                stages.add(node.args[1].value)
    for name in stages:
        if not (ROOT / "scripts" / name).is_file():
            errors.append(f"Production subprocess missing: {name}")
    return dict(passed=not errors, production_python_files=len(actual), scanned_python_files=len(sources),
                configuration_python_files=len(actual_configuration),
                relocated_configuration_files=len(manifest.get("configuration_moves", [])),
                moved_python_files=len(manifest["moved"]), production_stages=sorted(stages),
                cleanup_hashes_verified=verify_cleanup_hashes, errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-cleanup-hashes", action="store_true", help="Check byte identity against the current recorded layout baseline")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = check(verify_cleanup_hashes=args.verify_cleanup_hashes)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
