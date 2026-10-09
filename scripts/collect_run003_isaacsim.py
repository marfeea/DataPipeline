"""Re-execute diverse accepted run003 routes using the native Isaac Sim backend."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DataPipe.pipelines.curobo_bootstrap.pick_place_artifact import sha256_file, verify_hybrid_artifact


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bundle_completed_routes(status, output):
    from archive_run003_successes import copy_checked

    archive_records = []
    for record in status["routes"]:
        if not record.get("passed"):
            continue
        directory = Path(record["artifact"])
        integrated = Path(record["integrated_run"])
        if not all(p.resolve().is_relative_to(output.resolve()) for p in (directory, integrated)):
            raise ValueError("bundle paths must stay inside the collection output")

        def copy_file(source, destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if sha256_file(source) != sha256_file(destination):
                    raise ValueError(f"existing bundle file differs: {destination}")
            else:
                copy_checked(source, destination)

        for name in ("report.json", "trace.json", "execution_visualization"):
            source = integrated / name
            destination = directory / "contact_execution" / name
            if source.is_dir():
                for file in source.rglob("*"):
                    if file.is_file():
                        copy_file(file, destination / file.relative_to(source))
            else:
                copy_file(source, destination)
        snapshot = verify_hybrid_artifact(directory)["snapshot"]
        references = [
            (snapshot["collision_model"], snapshot["collision_model_sha256"]),
            (snapshot["urdf"], snapshot["urdf_sha256"]),
            *((mesh["file_path"], mesh["sha256"]) for mesh in snapshot["world"]["meshes"]),
        ]
        mapping = {}
        for original, digest in references:
            source = Path(original)
            if sha256_file(source) != digest:
                raise ValueError(f"bundle dependency hash differs: {source}")
            relative = Path("referenced_assets") / (digest + source.suffix)
            copy_file(source, directory / relative)
            mapping[original] = relative.as_posix()
        (directory / "asset_path_map.json").write_text(
            json.dumps(
                {
                    "schema": "curobo_artifact_asset_paths/1",
                    "paths": mapping,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (directory / "source_run_report.json").write_text(
            json.dumps(
                {
                    "schema": "run003_isaacsim_route/1",
                    "passed": True,
                    "backend": "isaacsim",
                    "integrated_run": str(integrated),
                    "source_run": str(directory),
                    "reference": record["reference"],
                    "stages": [
                        {**info, "stage": "integrated" if info["stage"] == "execution" else info["stage"]}
                        for info in record["commands"]
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        verify_hybrid_artifact(directory)
        record["archive_layout_verified"] = True
        core_names = (
            "trajectory.npz",
            "scene_snapshot.json",
            "plan_report.json",
            "replay_report.json",
            "asset_path_map.json",
            "source_run_report.json",
        )
        files = [directory / name for name in core_names]
        for name in ("contact_execution", "referenced_assets", "replay_visualization"):
            files.extend(p for p in (directory / name).rglob("*") if p.is_file())
        archive_records.append(
            {
                "route": record["route"],
                "recommended": True,
                "archive_directory": directory.name,
                "source_index": snapshot["task"]["source_index"],
                "destination_index": snapshot["task"]["destination_index"],
                "files_sha256": {p.relative_to(directory).as_posix(): sha256_file(p) for p in files},
            }
        )
    (output / "archive_manifest.json").write_text(
        json.dumps(
            {
                "schema": "run003_success_archive/1",
                "backend": "isaacsim",
                "successful_directions": len(archive_records),
                "records": archive_records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def select_routes(archive, count):
    manifest = read(archive / "archive_manifest.json")
    entries = [
        e
        for e in manifest["records"]
        if e["recommended"] and (archive / e["archive_directory"] / "source_run_report.json").is_file()
    ]
    features = []
    for entry in entries:
        artifact = verify_hybrid_artifact(archive / entry["archive_directory"])
        q = artifact["trajectory"]["joint_position"]
        features.append(
            np.stack([np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, len(q)), q[:, j]) for j in range(6)], axis=-1)
        )
    f = np.stack(features)
    distances = np.sqrt(np.mean((f[:, None] - f[None, :]) ** 2, axis=(2, 3)))
    chosen = [int(np.argmax(np.mean(distances, axis=1)))]
    while len(chosen) < len(entries):
        chosen.append(
            max((i for i in range(len(entries)) if i not in chosen), key=lambda i: min(distances[i, j] for j in chosen))
        )
    return [entries[i] for i in chosen], {
        "method": "farthest point sampling of 100 normalized-time six-joint samples; RMS radians",
        "candidate_routes": [e["route"] for e in entries],
        "ranking": [entries[i]["route"] for i in chosen],
        "initial_selection": [entries[i]["route"] for i in chosen[:count]],
        "initial_pairwise_rms_rad": distances[np.ix_(chosen[:count], chosen[:count])].tolist(),
    }


def resume_replays(status, output, environment, save):
    """Reuse completed physical execution and rebuild failed visual evidence."""
    from audit_run003_isaacsim import inspect_frames, reference_calibration

    for record in status["routes"]:
        if record.get("passed") or not record.get("stage", "").startswith("replay"):
            continue
        evidence = output / (record["route"] + "_evidence")
        artifact_dir = output / record["route"]
        if not all(path.resolve().is_relative_to(output.resolve()) for path in (evidence, artifact_dir)):
            raise ValueError("resumed route paths must stay inside the collection output")
        contact = read(evidence / "integrated/report.json")
        if not contact.get("passed") or contact.get("backend") != "isaacsim" or contact.get("isaaclab_modules_loaded"):
            continue
        artifact = verify_hybrid_artifact(artifact_dir)
        reference_frames = reference_calibration(Path(record["reference"]) / "replay_visualization")

        def quality():
            replay = read(artifact_dir / "replay_report.json")
            if not replay.get("passed") or replay.get("backend") != "isaacsim" or replay.get("isaaclab_modules_loaded"):
                raise ValueError("native replay runtime gates failed")
            return inspect_frames(
                artifact_dir / "replay_visualization",
                source_count=len(artifact["trajectory"]["time_s"]),
                reference_frames=reference_frames,
            )

        try:
            visual_quality = quality()
        except (OSError, ValueError, KeyError, AssertionError):
            attempt = 1
            while any(
                path.exists()
                for path in (
                    artifact_dir / f"replay_visualization_failed_{attempt}",
                    artifact_dir / f"replay_report_failed_{attempt}.json",
                    evidence / f"replay_resume_{attempt}.log",
                )
            ):
                attempt += 1
            for name in ("replay_visualization", "replay_report.json"):
                source = artifact_dir / name
                if source.exists():
                    destination = artifact_dir / (
                        f"replay_visualization_failed_{attempt}"
                        if name == "replay_visualization"
                        else f"replay_report_failed_{attempt}.json"
                    )
                    source.rename(destination)
            cmd = next(info["command"] for info in reversed(record["commands"]) if info["stage"].startswith("replay"))
            log = evidence / f"replay_resume_{attempt}.log"
            info = {"stage": f"replay_resume_{attempt}", "command": cmd, "log": str(log)}
            record["commands"].append(info)
            record["stage"] = info["stage"]
            save()
            print(
                f"NATIVE_ROUTE route={record['route']} stage={info['stage']} reuse_physical_execution=true", flush=True
            )
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(cmd, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            info["returncode"] = result.returncode
            save()
            try:
                visual_quality = quality()
            except (OSError, ValueError, KeyError, AssertionError) as error:
                record["failure_reason"] = f"resumed replay quality: {type(error).__name__}: {error}"
                save()
                continue
        trace = read(evidence / "integrated/trace.json")
        record["execution_visual_quality"] = inspect_frames(
            evidence / "integrated/execution_visualization",
            source_count=len(trace),
            trace=trace,
            reference_frames=reference_frames,
        )
        record["replay_visual_quality"] = visual_quality
        record.pop("failure_reason", None)
        record.update(
            stage="complete", passed=True, artifact=str(artifact_dir), integrated_run=str(evidence / "integrated")
        )
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/curobo_pick_place_graph_run003_success_20261004")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.count < 6:
        parser.error("the migration acceptance requires at least six directions")
    archive = args.archive.resolve()
    output = args.output_root.resolve()
    if output == archive or output.is_relative_to(archive):
        parser.error("output must be separate from the reference archive")
    if output.exists() and not args.resume:
        raise FileExistsError("choose a new output directory or explicitly resume")
    output.mkdir(parents=True, exist_ok=True)
    entries, selection = select_routes(archive, args.count)
    manifest = read(archive / "archive_manifest.json")
    status = {
        "schema": "run003_isaacsim_collection/1",
        "passed": False,
        "backend": "isaacsim",
        "archive": str(archive),
        "requested_count": args.count,
        "selection": selection,
        "routes": [],
    }
    if args.resume and (output / "collection_report.json").exists():
        status["routes"] = read(output / "collection_report.json")["routes"]

    def save():
        (output / "collection_report.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    save()
    if args.select_only:
        print(json.dumps(selection, indent=2))
        return 0
    runtime = Path(sys.executable).parent
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join(
        [str(runtime), str(runtime / "Library/bin"), str(runtime / "Scripts"), environment.get("PATH", "")]
    )
    environment["HEADLESS"] = "0" if args.gui else "1"
    if args.resume:
        resume_replays(status, output, environment, save)
    for entry in entries:
        if sum(r.get("passed", False) for r in status["routes"]) >= args.count:
            break
        route = entry["route"]
        previous = next((r for r in status["routes"] if r["route"] == route), None)
        if previous is not None:
            if previous.get("passed"):
                verify_hybrid_artifact(Path(previous["artifact"]))
            continue
        reference = archive / entry["archive_directory"]
        origin = read(reference / "source_run_report.json")
        command = next(
            s["command"]
            for s in origin["stages"]
            if s["stage"] == Path(origin["integrated_run"]).name and s["returncode"] == 0
        )
        evidence = output / (route + "_evidence")
        evidence.mkdir()
        record = {
            "route": route,
            "passed": False,
            "stage": "execution",
            "reference": str(reference),
            "reference_trajectory_sha256": sha256_file(reference / "trajectory.npz"),
            "commands": [],
        }
        status["routes"].append(record)
        save()
        native_command = [sys.executable, str(ROOT / "scripts/probe_curobo_v2_legacy_contact_isaac.py"), *command[2:]]
        native_command[native_command.index("--output") + 1] = str(evidence / "integrated")
        native_command += ["--backend", "isaacsim", "--kit-cache-dir", str(output / "kit_cache")]
        if not args.gui and "--headless" not in native_command:
            native_command.append("--headless")
        if "--capture-execution" not in native_command:
            native_command.append("--capture-execution")
        native_command += ["--camera-width", "640", "--camera-height", "480"]

        def invoke(stage, cmd):
            record["stage"] = stage
            log = evidence / (stage + ".log")
            print(f"NATIVE_ROUTE route={route} stage={stage}", flush=True)
            info = {"stage": stage, "command": cmd, "log": str(log)}
            record["commands"].append(info)
            save()
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(cmd, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            info["returncode"] = result.returncode
            save()
            return result.returncode == 0

        if not invoke("execution", native_command):
            report_path = evidence / "integrated/report.json"
            record["failure_reason"] = (
                read(report_path).get("failure_reason") if report_path.exists() else "runtime failure; see log"
            )
            save()
            continue
        contact_path = evidence / "integrated/report.json"
        if not contact_path.exists():
            record["failure_reason"] = "execution exited without a completed report; see execution log"
            save()
            continue
        contact = read(contact_path)
        if not contact.get("passed"):
            record["failure_reason"] = contact.get("failure_reason")
            save()
            continue
        if contact.get("backend") != "isaacsim" or contact.get("isaaclab_modules_loaded") != []:
            raise RuntimeError("native execution dependency gate failed")
        targets = command[command.index("--targets-report") + 1]
        world = command[command.index("--world-report") + 1]
        artifact = output / route
        build = [
            sys.executable,
            str(ROOT / "scripts/build_curobo_pick_place_v2_artifact.py"),
            "--integrated-run",
            str(evidence / "integrated"),
            "--targets-report",
            targets,
            "--world-report",
            world,
            "--output",
            str(artifact),
            "--contact-policy",
            "run003",
        ]
        if not invoke("artifact", build):
            record["failure_reason"] = read(artifact / "plan_report.json").get("failure_reason")
            save()
            continue
        verify_hybrid_artifact(artifact)
        replay = [
            sys.executable,
            str(ROOT / "scripts/replay_isaacsim.py"),
            "--backend",
            "isaacsim",
            "--run",
            str(artifact),
            "--no-realtime",
            "--camera-width",
            "640",
            "--camera-height",
            "480",
            "--kit-cache-dir",
            str(output / "kit_cache"),
        ]
        if not args.gui:
            replay.append("--headless")
        if not invoke("replay", replay):
            record["failure_reason"] = read(artifact / "replay_report.json").get("failure_reason")
            save()
            continue
        replay_report = read(artifact / "replay_report.json")
        if (
            not replay_report.get("passed")
            or replay_report.get("backend") != "isaacsim"
            or replay_report.get("isaaclab_modules_loaded") != []
        ):
            record["failure_reason"] = replay_report.get("failure_reason", "native replay gates failed")
            save()
            continue
        try:
            from audit_run003_isaacsim import inspect_frames, reference_calibration

            trace = read(evidence / "integrated/trace.json")
            reference_frames = reference_calibration(reference / "replay_visualization")
            record["execution_visual_quality"] = inspect_frames(
                evidence / "integrated/execution_visualization",
                source_count=len(trace),
                trace=trace,
                reference_frames=reference_frames,
            )
            native_artifact = verify_hybrid_artifact(artifact)
            record["replay_visual_quality"] = inspect_frames(
                artifact / "replay_visualization",
                source_count=len(native_artifact["trajectory"]["time_s"]),
                reference_frames=reference_frames,
            )
        except Exception as error:
            record["failure_reason"] = f"visual quality audit: {type(error).__name__}: {error}"
            save()
            continue
        record.update(
            stage="complete", passed=True, artifact=str(artifact), integrated_run=str(evidence / "integrated")
        )
        save()
    bundle_completed_routes(status, output)
    changed = []
    for entry in manifest["records"]:
        for name, expected in entry["files_sha256"].items():
            path = archive / entry["archive_directory"] / name
            if sha256_file(path) != expected:
                changed.append(str(path))
    status["reference_files_changed"] = changed
    status["successful_count"] = sum(r.get("passed", False) for r in status["routes"])
    status["passed"] = status["successful_count"] >= args.count and not changed
    save()
    return 0 if status["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
