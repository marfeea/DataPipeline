"""Collect all twenty directed routes with native Isaac Sim GUI and retry startup failures."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from testscripts.script_paths import script_path


import argparse
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pipelines.curobo_bootstrap.pick_place_artifact import verify_hybrid_artifact


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--routes", nargs="*")
    parser.add_argument("--source2-recovery", action="store_true", help="Test the existing lateral/raised source2 approach with slower wrist drives")
    args = parser.parse_args()
    batch = args.output_root.resolve()
    batch.mkdir(parents=True, exist_ok=True)
    runtime = Path(sys.executable).parent
    os.environ["PATH"] = os.pathsep.join([str(runtime), str(runtime / "Scripts"), str(runtime / "Library/bin"), os.environ.get("PATH", "")])
    os.environ["HEADLESS"] = "0"
    presets = read(ROOT / "configs/curobo/route_presets.json")["routes"]
    routes = args.routes or list(presets) + [f"{s}_to_{d}" for s in range(5) for d in range(5) if s != d and f"{s}_to_{d}" not in presets]
    status_path = batch / "graph_summary.json"
    status = read(status_path) or dict(schema="five_state_isaacsim_gui_test/1", backend="isaacsim", mode="gui", routes={})

    def save():
        status["successful_routes"] = sum(bool(r.get("passed")) for r in status["routes"].values())
        status["passed"] = len(status["routes"]) == 20 and status["successful_routes"] == 20
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def invoke(script, arguments, log):
        command = [sys.executable, str(script_path(script)), *map(str, arguments)]
        assert "--headless" not in command and "--no-gui" not in command
        print("COMMAND " + subprocess.list2cmdline(command), flush=True)
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
        return result.returncode

    def startup_failure(report, log):
        latest = {stage["stage"]: stage for stage in report.get("stages", [])}
        text = str(report.get("failure_reason") or report.get("error") or "")
        if not latest:
            text += log.read_text(encoding="utf-8", errors="replace")
        for stage in latest.values():
            path = Path(stage.get("log", ""))
            if path.is_file() and not stage.get("passed"):
                text += path.read_text(encoding="utf-8", errors="replace")
        return any(marker in text for marker in ("Native camera warmup", "Native render did not", "Wrist camera rendered pose is stale", "Camera has no rendered", "Crash detected", "[Fatal]", "DLL load failed", "WinError 126", "WinError 32", "Ninja is required", "Failed to load the shared library"))

    for route in routes:
        if status["routes"].get(route, {}).get("passed"):
            continue
        source, destination = map(int, route.split("_to_"))
        assert 0 <= source < 5 and 0 <= destination < 5 and source != destination
        record = status["routes"].setdefault(route, dict(passed=False, attempts=[]))
        attempt = len(record["attempts"])
        while True:
            attempt += 1
            run = batch / (route if attempt == 1 else f"{route}_retry_{attempt}")
            evidence = run.with_name(run.name + "_evidence")
            if run.exists() or evidence.exists():
                continue
            log = batch / f"{run.name}_collection.log"
            info = dict(attempt=attempt, run=str(run), log=str(log), stage="collection")
            record["attempts"].append(info)
            save()
            parameters = ["--route-preset", route] if route in presets else [
                "--source-index", str(source), "--destination-index", str(destination),
                "--lift-clearance-m", "0.3" if 4 in (source, destination) or (args.source2_recovery and source == 2) else "0.4",
                "--seed-count", "200", "--max-candidates", "8", "--contact-policy", "run003",
                "--collision-model", "data/curobo_pick_place_graph_run003_20261003_r2/collision_model.json",
            ]
            if args.source2_recovery and source == 2:
                parameters += ["--state2-pre-pick-side-m", "0.06", "--state2-pre-pick-lift-m", "0.04", "--contact-wrist-velocity-limit-rad-s", "0.3"]
            if route == "2_to_4":
                parameters += ["--state4-routing", "--state4-place-standoff-m", "0.24", "--state4-place-entry-lift-m", "0.012", "--state4-place-measured-grasp"]
            info["returncode"] = invoke("collect_isaacsim.py", [*parameters, "--execution-mode", "hybrid", "--backend", "isaacsim", "--gui", "--capture-execution", "--output", run], log)
            report = read(evidence / "run_report.json")
            info["report"] = str(evidence / "run_report.json")
            if not report.get("passed"):
                info["failure_reason"] = report.get("failure_reason", "missing passing run report")
                info["retryable"] = startup_failure(report, log)
                save()
                print(f"FAILED {route} attempt={attempt} retryable={info['retryable']} {info['failure_reason']}", flush=True)
                if info["retryable"]:
                    time.sleep(3)
                    continue
                record["failure_reason"] = info["failure_reason"]
                break
            verify_hybrid_artifact(run)
            integrated = Path(report["integrated_run"])
            audit_log = batch / f"{run.name}_capture_audit.log"
            if invoke("audit_contact_execution_visuals.py", ["--run", integrated], audit_log):
                record["failure_reason"] = f"capture audit failed: {audit_log}"
                break
            info["native_images"] = len(read(integrated / "execution_visualization/frames.json")["frames"])
            info["stage"] = "replay"
            save()
            replay_attempt = 0
            while True:
                replay_attempt += 1
                replay_log = batch / f"{run.name}_replay_{replay_attempt}.log"
                visual = run / f"replay_visualization_{replay_attempt}"
                report_name = f"replay_report_{replay_attempt}.json"
                code = invoke("replay_isaacsim.py", ["--run", run, "--backend", "isaacsim", "--device", "cuda:0", "--camera-width", "640", "--camera-height", "480", "--visualization-dir", visual, "--report-name", report_name], replay_log)
                replay = read(run / report_name)
                if code == 0 and replay.get("passed") and replay.get("mode") == "gui":
                    frames = read(visual / "frames.json")["frames"]
                    assert frames and all((visual / frame["rgb"]).is_file() for frame in frames)
                    (run / "replay_report.json").write_text(json.dumps(replay, indent=2) + "\n", encoding="utf-8")
                    record.update(passed=True, run=str(run), integrated_run=str(integrated), samples=report["samples"], native_images=info["native_images"], replay_images=len(frames), replay_visualization=str(visual))
                    record.pop("failure_reason", None)
                    info["stage"] = "complete"
                    break
                if not startup_failure(replay, replay_log):
                    record["failure_reason"] = f"replay failed: {replay_log}"
                    break
                print(f"RETRY CAMERA {route} replay={replay_attempt}", flush=True)
                time.sleep(3)
            break
        save()
        print(f"RESULT {route} passed={record['passed']} successful={status['successful_routes']}/20", flush=True)
    return 0 if status["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
