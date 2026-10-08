"""Copy accepted run003 graph artifacts and their visual evidence into one archive."""

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from DataPipe.pipelines.curobo_bootstrap.pick_place_artifact import sha256_file, verify_hybrid_artifact
from DataPipe.pipelines.curobo_bootstrap.route_identity import require_report_identity

PROJECT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    PROJECT / "data/curobo_pick_place_graph_run003_20261003",
    PROJECT / "data/curobo_pick_place_graph_run003_20261003_r2",
    PROJECT / "data/curobo_pick_place_graph_run003_fix_20261004",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_original_baseline():
    baseline = read(PROJECT / "docs/run003_failure_analysis_20261004/offline_diagnostics.json")
    for name, expected in baseline["source_sha256"].items():
        if sha256_file(PROJECT.parent / name) != expected:
            raise ValueError(f"baseline source changed: {name}")
    return len(baseline["source_sha256"])


def inventory():
    accepted, rejected = [], []
    recovered = {}
    for priority, root in enumerate(SOURCE_ROOTS):
        for path in sorted(root.rglob("plan_report.json")):
            if not read(path).get("passed") or not (path.parent / "trajectory.npz").exists():
                continue
            run = path.parent
            try:
                snapshot = read(run / "scene_snapshot.json")
                references = [
                    (snapshot["collision_model"], snapshot["collision_model_sha256"]),
                    (snapshot["urdf"], snapshot["urdf_sha256"]),
                    *((mesh["file_path"], mesh["sha256"]) for mesh in snapshot["world"]["meshes"]),
                ]
                assets = []
                for original, digest in references:
                    resolved = Path(original)
                    missing = not resolved.is_file()
                    if missing:
                        key = (resolved.name, digest)
                        if key not in recovered:
                            recovered[key] = next(
                                (
                                    candidate
                                    for source in SOURCE_ROOTS
                                    for candidate in source.rglob(resolved.name)
                                    if candidate.is_file() and sha256_file(candidate) == digest
                                ),
                                None,
                            )
                        if recovered[key] is None:
                            raise FileNotFoundError(f"no hash-matched dependency found: {original}")
                        resolved = recovered[key]
                    assets.append(
                        dict(
                            original_path=original, sha256=digest, source_path=str(resolved), recovered_missing=missing
                        )
                    )
                artifact = verify_hybrid_artifact(
                    run, asset_paths={a["original_path"]: a["source_path"] for a in assets}
                )
                task = artifact["snapshot"]["task"]
                replay = read(run / "replay_report.json")
                require_report_identity(
                    replay, task, label="archive source replay", world_hash=artifact["snapshot"]["world_hash"]
                )
                if not replay["passed"]:
                    raise ValueError("replay did not pass")
                for name, digest in replay["source_sha256"].items():
                    if sha256_file(run / name) != digest:
                        raise ValueError(f"replay source hash differs: {name}")
                core_hashes = {
                    name: sha256_file(run / name)
                    for name in ("trajectory.npz", "scene_snapshot.json", "plan_report.json", "replay_report.json")
                }
                version = hashlib.sha256(json.dumps(core_hashes, sort_keys=True).encode()).hexdigest()
                route_report = next(
                    (
                        p
                        for p in (
                            run.with_name(run.name + "_evidence") / "run_report.json",
                            root / "evidence" / (run.name + "_evidence") / "run_report.json",
                        )
                        if p.exists()
                    ),
                    None,
                )
                accepted.append(
                    dict(
                        source_run=str(run),
                        source_root=str(root),
                        priority=priority,
                        version=version,
                        route=f"{task['source_index']}_to_{task['destination_index']}",
                        source_index=task["source_index"],
                        destination_index=task["destination_index"],
                        task_hash=replay["task_hash"],
                        world_hash=replay["world_hash"],
                        replay_mode=replay["mode"],
                        samples=artifact["acceptance"]["samples"],
                        replay_images=replay["visualization"]["image_count"],
                        lift_clearance_m=task["lift_clearance_m"],
                        state4_routing=task.get("state4_routing"),
                        source_route_report=str(route_report) if route_report else None,
                        core_sha256=core_hashes,
                        assets=assets,
                    )
                )
            except Exception as error:
                rejected.append(dict(source_run=str(run), reason=f"{type(error).__name__}: {error}"))
    return accepted, rejected


def rank(entry):
    return entry["priority"], entry["replay_mode"] == "gui", entry["source_run"]


def copy_checked(source, destination):
    if source.is_dir():
        shutil.copytree(source, destination)
        files = [p for p in source.rglob("*") if p.is_file()]
    else:
        shutil.copy2(source, destination)
        files = [source]
    for path in files:
        copied = destination / path.relative_to(source) if source.is_dir() else destination
        if sha256_file(path) != sha256_file(copied):
            raise ValueError(f"copied file hash mismatch: {copied}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "data/curobo_pick_place_graph_run003_success_20261004")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument(
        "--append", action="store_true", help="Verify and preserve existing versions, then add new ones"
    )
    args = parser.parse_args()
    baseline_count = check_original_baseline()
    accepted, rejected = inventory()
    if args.inventory_only:
        print(
            json.dumps(
                dict(
                    accepted_records=len(accepted),
                    successful_directions=len({r["route"] for r in accepted}),
                    rejected=rejected,
                    records=accepted,
                ),
                indent=2,
            )
        )
        return
    output = args.output.resolve()
    if output.exists() and not args.append:
        raise FileExistsError("archive already exists; choose a fresh output directory")
    if not output.is_relative_to(PROJECT / "data"):
        raise ValueError("archive must stay within this project's data directory")
    if not accepted:
        raise ValueError("no accepted artifacts to archive")
    if any(output.is_relative_to(root) for root in SOURCE_ROOTS):
        raise ValueError("archive cannot be nested inside an input run")
    previous_records = []
    if output.exists():
        previous = read(output / "archive_manifest.json")
        if previous.get("schema") != "run003_success_archive/1":
            raise ValueError("unrecognized existing archive")
        for record in previous["records"]:
            directory = (output / record["archive_directory"]).resolve()
            if not directory.is_relative_to(output):
                raise ValueError("archive directory escapes output")
            for name, digest in record["files_sha256"].items():
                file = (directory / name).resolve()
                if not file.is_relative_to(directory) or sha256_file(file) != digest:
                    raise ValueError(f"existing archive file differs: {file}")
            verify_hybrid_artifact(directory)
        if sha256_file(output / "route_presets.json") != previous["parameter_config_sha256"]:
            raise ValueError("existing archived parameter configuration differs")
        previous_records = previous["records"]
        history = output / "history"
        history.mkdir(exist_ok=True)
        for name in ("archive_manifest.json", "route_presets.json", "README.md"):
            source = output / name
            destination = history / (source.stem + "_" + sha256_file(source) + source.suffix)
            if not destination.exists():
                copy_checked(source, destination)
    else:
        output.mkdir(parents=True)
    grouped = defaultdict(list)
    for entry in accepted:
        grouped[entry["route"]].append(entry)
    copied_records = list(previous_records)
    existing_versions = {(r["route"], r["version"]) for r in previous_records}
    existing_routes = {r["route"] for r in previous_records}
    for route, entries in sorted(grouped.items()):
        recommended = max(entries, key=rank)
        versions = defaultdict(list)
        for entry in entries:
            versions[entry["version"]].append(entry)
        for version, sources in sorted(versions.items()):
            if (route, version) in existing_versions:
                continue
            selected = max(sources, key=rank)
            canonical = route not in existing_routes and version == recommended["version"]
            relative = Path(route) if canonical else Path("variants") / route / version[:12]
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            copy_checked(Path(selected["source_run"]), destination)
            source_run = Path(selected["source_run"])
            for suffix, name in (
                ("_gui_preview.jpg", "gui_preview.jpg"),
                ("_visual_inspection_samples.json", "visual_inspection_samples.json"),
            ):
                preview = source_run.with_name(source_run.name + suffix)
                if preview.exists():
                    copy_checked(preview, destination / name)
            asset_directory = destination / "referenced_assets"
            asset_directory.mkdir()
            asset_map = {}
            for asset in selected["assets"]:
                source = Path(asset["source_path"])
                relative_asset = Path("referenced_assets") / (asset["sha256"] + source.suffix)
                if not (destination / relative_asset).exists():
                    copy_checked(source, destination / relative_asset)
                asset_map[asset["original_path"]] = str(relative_asset)
            (destination / "asset_path_map.json").write_text(
                json.dumps(dict(schema="curobo_artifact_asset_paths/1", paths=asset_map), indent=2) + "\n",
                encoding="utf-8",
            )
            if selected["source_route_report"]:
                source_report = Path(selected["source_route_report"])
                copy_checked(source_report, destination / "source_run_report.json")
                run_report = read(source_report)
                integrated = Path(run_report.get("integrated_run", ""))
                if run_report.get("passed") and integrated.is_dir() and "integrated_run" in run_report:
                    native = destination / "contact_execution"
                    native.mkdir()
                    for name in (
                        "report.json",
                        "trace.json",
                        "capture_audit.json",
                        "live_gui_preview.jpg",
                        "execution_visualization",
                    ):
                        if (integrated / name).exists():
                            copy_checked(integrated / name, native / name)
            verify_hybrid_artifact(destination)
            file_hashes = {
                str(p.relative_to(destination)): sha256_file(p) for p in destination.rglob("*") if p.is_file()
            }
            copied_records.append(
                dict(
                    selected,
                    archive_directory=str(relative),
                    recommended=canonical,
                    equivalent_sources=[s["source_run"] for s in sources],
                    files_sha256=file_hashes,
                )
            )
    copy_checked(PROJECT / "configs/curobo/route_presets.json", output / "route_presets.json")
    manifest = dict(
        schema="run003_success_archive/1",
        scope="current run003 graph, r2 baseline and repair batches; accepted artifact and replay",
        successful_direction_count=len({r["route"] for r in copied_records}),
        source_record_count=len(accepted),
        archived_version_count=len(copied_records),
        original_directories_preserved=True,
        baseline_files_verified=check_original_baseline(),
        successful_routes=sorted({r["route"] for r in copied_records}),
        records=copied_records,
        excluded_reported_successes=rejected,
        parameter_config_sha256=sha256_file(output / "route_presets.json"),
        dependencies=(
            "Referenced verification assets are copied and hash checked; asset_path_map.json preserves frozen reports. "
            "Original evidence paths remain for provenance."
        ),
    )
    assert manifest["baseline_files_verified"] == baseline_count
    (output / "archive_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    rows = [
        "# run003 双向图成功轨迹归档",
        "",
        f"成功方向：{manifest['successful_direction_count']}；成功版本：{len(copied_records)}。",
        "",
        "各方向根目录为推荐版本；其他已验收成功版本保存在 variants 下。原记录保留，复制文件逐一核对 SHA256。",
        "",
        "| 方向 | 推荐目录 | 回放模式 | 轨迹点数 | 回放图片 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record in sorted((r for r in copied_records if r["recommended"]), key=lambda r: r["route"]):
        rows.append(
            f"| {record['route']} | [{record['route']}]({record['archive_directory']}/) "
            f"| {record['replay_mode']} | {record['samples']} | {record['replay_images']} |"
        )
    rows += [
        "",
        "4→0 成功参数：局部抬升 15 mm，水平退出 180 mm，统一转运净空 300 mm；物理 120 Hz，控制 60 Hz。",
        "项目配置：configs/curobo/route_presets.json；本目录保存同一配置副本。",
        "现场图片：各路线 contact_execution/execution_visualization；回放图片：各路线 replay_visualization。",
        "有头重跑：python scripts/plan_curobo_pick_place_v2.py --route-preset 4_to_0 --output <新的目录>。"
        "其他已验证路线使用同名预设。",
        "",
        "成功口径沿用 run003 混合规划/接触执行和状态回放，接触段托盘碰撞仍为诊断项。",
        "校验所需网格、URDF、碰撞模型均在 referenced_assets 中；asset_path_map.json 指向其归档副本。",
        "冻结报告保留原路径用于追溯。重跑 Isaac 仍需本项目运行环境。",
    ]
    (output / "README.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(
        json.dumps({k: v for k, v in manifest.items() if k not in ("records", "excluded_reported_successes")}, indent=2)
    )
    if rejected:
        print(json.dumps(dict(excluded_reported_successes=rejected), indent=2))


if __name__ == "__main__":
    main()
