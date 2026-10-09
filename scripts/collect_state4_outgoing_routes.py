"""Collect and independently audit the three outgoing state4 routes from a frozen batch."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_sources(manifest):
    for name, expected in manifest["source_sha256"].items():
        path = Path(name)
        if not path.is_absolute():
            path = PROJECT / path
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen collection source changed: {name}")


def run_script(script, arguments, log):
    print(f"STATE4_BATCH stage={script} log={log}", flush=True)
    environment = dict(os.environ, HEADLESS="0")
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            [sys.executable, str(PROJECT / "scripts" / script), *map(str, arguments)],
            cwd=PROJECT,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"{script} exited {result.returncode}; see {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args()
    batch = args.batch.resolve()
    manifest = read(batch / "experiment_manifest.json")
    assert manifest["schema"] == "state4_outgoing_collection/1"
    assert manifest["routes"] == ["4_to_1", "4_to_2", "4_to_3"]
    verify_sources(manifest)
    parameters = manifest["parameters"]
    records = []
    for destination in (1, 2, 3):
        route = f"4_to_{destination}"
        run = batch / route
        evidence = batch / (route + "_evidence")
        record = dict(route=route, passed=False, run=str(run))
        try:
            report_path = evidence / "run_report.json"
            if not report_path.exists():
                arguments = ["--source-index", "4", "--destination-index", str(destination)]
                for key in (
                    "lift_clearance_m", "state4_pick_lift_m", "state4_pick_exit_m", "state4_place_standoff_m",
                    "physics_hz", "seed_count", "max_candidates", "contact_policy", "collision_model",
                ):
                    arguments += ["--" + key.replace("_", "-"), str(parameters[key])]
                assert parameters["gui"] and parameters["capture_execution"] and parameters["state4_routing"]
                arguments += ["--gui", "--capture-execution", "--state4-routing", "--output", str(run)]
                run_script("collect_isaacsim.py", arguments, batch / (route + "_collection.log"))
            report = read(report_path)
            if not report.get("passed"):
                raise ValueError(f"collection failed at {report.get('stage')}: {report.get('error')}")
            replay_path = run / "replay_report.json"
            if not replay_path.exists():
                run_script(
                    "replay_isaacsim.py",
                    ["--run", run, "--device", "cuda:0", "--camera-width", "640", "--camera-height", "480"],
                    run / "replay_gui.log",
                )
            assert read(replay_path)["passed"] and read(replay_path)["mode"] == "gui"
            captures = []
            for frames_path in sorted(evidence.glob("*/execution_visualization/frames.json")):
                if not read(frames_path)["frames"]:
                    continue
                probe = frames_path.parents[1]
                run_script("audit_contact_execution_visuals.py", ["--run", probe], probe / "capture_audit.log")
                capture = read(probe / "capture_audit.json")
                assert capture["capture_audit_passed"]
                captures.append(dict(run=str(probe), images=len(read(frames_path)["frames"])))
            integrated = Path(report["integrated_run"])
            accepted_capture = read(integrated / "capture_audit.json")
            assert accepted_capture["capture_audit_passed"]
            run_script("preview_run003_stage1_gui.py", ["--run", run], run / "preview_audit.log")
            run_script("audit_state4_routed_run.py", ["--run", run], run / "state4_routing_audit.log")
            audit = read(run / "state4_routing_audit.json")
            assert audit["audit_passed"]
            record.update(
                passed=True,
                samples=report["samples"],
                task_hash=report["task_hash"],
                world_hash=report["world_hash"],
                integrated_run=str(integrated),
                accepted_candidate=integrated.name.removeprefix("integrated_"),
                final_target_position_error_m=audit["final_target_error_m"],
                native_images=len(read(integrated / "execution_visualization/frames.json")["frames"]),
                replay_images=audit["replay_images"],
                audited_captures=captures,
                state4_routing=audit["state4_routing"],
            )
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        records.append(record)
        verify_sources(manifest)
        result = dict(
            schema="state4_outgoing_collection_result/1", derived_from_preset=manifest["derived_from_preset"],
            parameters=parameters, records=records, frozen_sources_verified=True,
            scope="run003 mixed planning/contact execution and GUI state replay; contact tray collision is diagnostic",
        )
        (batch / "collection_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("STATE4_BATCH_RESULT " + json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
