"""Run the two stage1 routes from scratch with visible Isaac and 640x480 replay."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError("GUI retest requires an empty output directory")
    model = project / "data/curobo_pick_place_graph_run003_20261003_r2/collision_model.json"
    baseline = json.loads(
        (project / "docs/run003_failure_analysis_20261004/offline_diagnostics.json").read_text(encoding="utf-8")
    )
    assert all(digest(project.parent / name) == expected for name, expected in baseline["source_sha256"].items())
    sources = (
        "DataPipe/pipelines/curobo_bootstrap/task_targets.py",
        "scripts/collect_isaacsim.py",
        "scripts/audit_run003_stage1.py",
        "scripts/retest_run003_stage1_gui.py",
        "tests/curobo_bootstrap/test_task_targets.py",
        "tests/curobo_bootstrap/test_route_execution_mode.py",
    )
    manifest = dict(
        scope="stage1 GUI retest after user-requested deletion of the two headless route records",
        routes=["3_to_0", "3_to_1"],
        mode="gui",
        camera_resolution=[640, 480],
        lift_clearance_m=0.4,
        seed=7,
        pre_pick_random_seed_count=200,
        max_candidates=8,
        target_random_seed_count_per_waypoint=16,
        collision_model=str(model),
        collision_model_sha256=digest(model),
        implementation_sha256={str(project / name): digest(project / name) for name in sources},
        targeted_tests=dict(passed=30),
        original_evidence_unchanged=True,
    )
    (root / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["HEADLESS"] = "0"
    environment["PYTHONIOENCODING"] = "utf-8"
    runtime = Path(sys.executable).parent
    environment["PATH"] = os.pathsep.join(
        [str(runtime), str(runtime / "Scripts"), str(runtime / "Library/bin"), environment.get("PATH", "")]
    )
    status = dict(mode="gui", resolution=[640, 480], stage="starting", passed=False, runs={})

    def save():
        (root / "stage1_gui_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    def invoke(key, command):
        assert "--headless" not in command
        status["stage"] = key
        status["runs"][key] = dict(command=command, headless_environment="0", stage="running")
        save()
        print(f"GUI_STAGE {key}", flush=True)
        with (root / (key + ".log")).open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=project, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        status["runs"][key].update(stage="complete", exit_code=result.returncode)
        save()
        if result.returncode:
            raise RuntimeError(f"GUI test failed at {key}; see its log")

    save()
    try:
        for destination in (0, 1):
            name = f"3_to_{destination}"
            run = root / name
            invoke(
                name + "_planning",
                [
                    sys.executable,
                    str(project / "scripts/collect_isaacsim.py"),
                    "--source-index",
                    "3",
                    "--destination-index",
                    str(destination),
                    "--gui",
                    "--contact-policy",
                    "run003",
                    "--collision-model",
                    str(model),
                    "--seed-count",
                    "200",
                    "--max-candidates",
                    "8",
                    "--output",
                    str(run),
                ],
            )
            invoke(
                name + "_replay",
                [
                    sys.executable,
                    str(project / "scripts/replay_isaacsim.py"),
                    "--run",
                    str(run),
                    "--device",
                    "cuda:0",
                    "--camera-width",
                    "640",
                    "--camera-height",
                    "480",
                ],
            )
            report = json.loads((run / "replay_report.json").read_text(encoding="utf-8"))
            assert report["passed"] and report["mode"] == "gui"
        invoke(
            "independent_audit",
            [
                sys.executable,
                str(project / "scripts/audit_run003_stage1.py"),
                "--output-root",
                str(root),
            ],
        )
        status.update(stage="complete", passed=True)
    except Exception as error:
        status["failure_reason"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
