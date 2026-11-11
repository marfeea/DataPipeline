"""Recover existing GUI replays, then recollect the route with invalid native images."""

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
    args = parser.parse_args()
    batch = args.output_root.resolve()
    runtime = Path(sys.executable).parent
    os.environ["HEADLESS"] = "0"
    os.environ["PYTHONPATH"] = str(ROOT)
    os.environ["PATH"] = os.pathsep.join([str(runtime), str(runtime / "Scripts"), str(runtime / "Library/bin"), os.environ.get("PATH", "")])
    summary_path = batch / "graph_summary.json"
    summary = read(summary_path)

    def save():
        summary["successful_routes"] = sum(bool(r.get("passed")) for r in summary["routes"].values())
        summary["passed"] = len(summary["routes"]) == 20 and summary["successful_routes"] == 20
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    def invoke(script, arguments, log):
        command = [sys.executable, str(script_path(script)), *map(str, arguments)]
        assert "--headless" not in command and "--no-gui" not in command
        print("COMMAND " + subprocess.list2cmdline(command), flush=True)
        with log.open("w", encoding="utf-8") as stream:
            code = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT).returncode
        return command, code

    for name in ("4_to_2", "3_to_1"):
        record = summary["routes"][name]
        run = Path(record.get("run") or record["attempts"][-1]["run"])
        verify_hybrid_artifact(run)
        collection = read(run.with_name(run.name + "_evidence") / "run_report.json")
        assert collection["passed"] and collection["mode"] == "gui"
        record["passed"] = False
        record["recovery_stage"] = "gui_replay"
        save()
        attempt = 0
        while True:
            attempt += 1
            log = batch / f"{run.name}_replay_{attempt}.log"
            visual = run / f"replay_visualization_{attempt}"
            report_name = f"replay_report_{attempt}.json"
            if log.exists() or visual.exists() or (run / report_name).exists():
                continue
            command, code = invoke("replay_isaacsim.py", ["--run", run, "--backend", "isaacsim", "--device", "cuda:0", "--camera-width", "640", "--camera-height", "480", "--visualization-dir", visual, "--report-name", report_name], log)
            replay = read(run / report_name)
            record.setdefault("replay_recovery_attempts", []).append(dict(command=command, returncode=code, report=str(run / report_name), log=str(log)))
            save()
            if code == 0 and replay.get("passed") and replay.get("mode") == "gui":
                frames = read(visual / "frames.json")["frames"]
                assert frames and all((visual / frame["rgb"]).is_file() for frame in frames)
                (run / "replay_report.json").write_text(json.dumps(replay, indent=2) + "\n", encoding="utf-8")
                integrated = Path(collection["integrated_run"])
                record.update(passed=True, run=str(run), integrated_run=str(integrated), samples=collection["samples"], native_images=len(read(integrated / "execution_visualization/frames.json")["frames"]), replay_images=len(frames), replay_visualization=str(visual), recovery_stage="complete")
                record.pop("failure_reason", None)
                save()
                subprocess.run([sys.executable, str(ROOT / "testscripts/audit_five_state_gui_test.py"), "--output-root", str(batch), "--routes", name], cwd=ROOT, env=os.environ.copy())
                assert read(batch / "independent_audit.json")["routes"][name]["passed"]
                print(f"RECOVERED {name}", flush=True)
                break
            text = str(replay.get("failure_reason", "")) + log.read_text(encoding="utf-8", errors="replace")
            if not any(marker in text for marker in ("Native camera warmup", "Native render did not", "Camera has no rendered", "Wrist camera rendered pose is stale", "[Fatal]", "Crash detected", "DLL load failed", "WinError")):
                raise RuntimeError(f"non-startup replay failure: {log}")
            print(f"RETRY GUI {name} attempt={attempt}", flush=True)
            time.sleep(3)

    summary = read(summary_path)
    summary["routes"]["4_to_0"]["passed"] = False
    summary["routes"]["4_to_0"]["failure_reason"] = "original native and replay images failed scene visibility audit; recollecting"
    save()
    command, code = invoke("collect_five_state_gui_test.py", ["--output-root", batch, "--routes", "4_to_0"], batch / "4_to_0_recovery_batch.log")
    current = read(summary_path)
    if not current["routes"]["4_to_0"].get("passed"):
        raise RuntimeError(f"4_to_0 recollection failed, exit={code}")
    subprocess.run([sys.executable, str(ROOT / "testscripts/audit_five_state_gui_test.py"), "--output-root", str(batch), "--routes", "4_to_0"], cwd=ROOT, env=os.environ.copy())
    assert read(batch / "independent_audit.json")["routes"]["4_to_0"]["passed"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
