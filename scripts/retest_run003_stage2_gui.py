"""Test every state4 route at explicit 0.30 m height, then collect GUI RGB evidence."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from DataPipe.pipelines.curobo_bootstrap.route_experiment import freeze_experiment, require_height


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = args.output_root.resolve()
    original = project / "data/curobo_pick_place_graph_run003_20261003_r2"
    model = original / "collision_model.json"
    routes = ["0_to_4", "4_to_0", "1_to_4", "4_to_1", "3_to_4", "4_to_3", "4_to_2", "2_to_4"]
    sources = [
        "DataPipe/pipelines/curobo_bootstrap/task_targets.py",
        "DataPipe/pipelines/curobo_bootstrap/route_experiment.py",
        "DataPipe/pipelines/curobo_bootstrap/pick_place_artifact.py",
        "scripts/collect_curobo_pick_place_v2_graph.py",
        "scripts/collect_isaacsim.py",
        "scripts/dry_run_curobo_pick_place_v2_targets_isaac.py",
        "scripts/probe_curobo_v2_legacy_contact_isaac.py",
        "scripts/replay_isaacsim.py",
        "scripts/audit_run003_graph.py",
        "scripts/audit_run003_stage2.py",
        "scripts/retest_run003_stage2_gui.py",
        "scripts/preview_run003_stage1_gui.py",
        "tests/curobo_bootstrap/test_route_height.py",
    ]
    baseline = read(project / "docs/run003_failure_analysis_20261004/offline_diagnostics.json")
    original_hashes = {str(project.parent / name): expected for name, expected in baseline["source_sha256"].items()}
    assert all(digest(Path(name)) == expected for name, expected in original_hashes.items())
    manifest = dict(
        schema="run003_stage2_height_gui/1",
        routes=routes,
        mode="gui",
        camera_resolution=[640, 480],
        route_lift_clearance_m={name: 0.3 for name in routes},
        unchanged_default_lift_clearance_m=0.4,
        seed=7,
        pre_pick_random_seed_count=200,
        max_candidates=8,
        contact_policy="run003",
        collision_model=str(model),
        collision_model_sha256=digest(model),
        original_root=str(original),
        original_evidence_sha256=original_hashes,
        implementation_sha256={str(project / name): digest(project / name) for name in sources},
        targeted_tests=dict(passed=44),
    )
    freeze_experiment(root, manifest)
    status_path = root / "stage2_gui_status.json"
    status = read(status_path) if status_path.exists() else dict(mode="gui", stage="starting", runs={}, routes={})
    if status.get("stage") == "complete":
        raise FileExistsError("This GUI experiment is already complete")
    environment = os.environ.copy()
    environment.update(HEADLESS="0", PYTHONIOENCODING="utf-8")
    runtime = Path(sys.executable).parent
    environment["PATH"] = os.pathsep.join(
        [str(runtime), str(runtime / "Scripts"), str(runtime / "Library/bin"), environment.get("PATH", "")]
    )

    def save():
        status_path.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def invoke(key, command):
        assert "--headless" not in command
        status.update(stage=key)
        status["runs"][key] = dict(command=command, headless_environment="0", stage="running")
        save()
        print("GUI_STAGE " + key, flush=True)
        with (root / (key + ".log")).open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=project, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        status["runs"][key].update(stage="complete", exit_code=result.returncode)
        save()
        return result.returncode

    try:
        for name in routes:
            if status["routes"].get(name, {}).get("terminal"):
                continue
            source, destination = name.split("_to_")
            run = root / name
            evidence = root / (name + "_evidence")
            if not evidence.exists():
                invoke(
                    name + "_planning",
                    [
                        sys.executable,
                        str(project / "scripts/collect_isaacsim.py"),
                        "--source-index",
                        source,
                        "--destination-index",
                        destination,
                        "--lift-clearance-m",
                        "0.30",
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
            route_report = read(evidence / "run_report.json")
            targets = read(evidence / "targets.json")
            require_height(targets["task_snapshot"], 0.3)
            result = dict(
                planning_passed=route_report["passed"], target_ik_passed=targets["passed"], stage=route_report["stage"]
            )
            if route_report["passed"]:
                if not (run / "replay_report.json").exists():
                    exit_code = invoke(
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
                    if exit_code:
                        raise RuntimeError(name + " GUI replay failed")
                replay = read(run / "replay_report.json")
                assert replay["passed"] and replay["mode"] == "gui"
                if invoke(
                    name + "_preview",
                    [sys.executable, str(project / "scripts/preview_run003_stage1_gui.py"), "--run", str(run)],
                ):
                    raise RuntimeError(name + " preview generation failed")
                result.update(
                    replay_passed=True, frames=replay["source_frames"], images=replay["visualization"]["image_count"]
                )
            else:
                result["failure_reason"] = route_report.get("failure_reason")
            result["terminal"] = True
            status["routes"][name] = result
            save()
            print("GUI_RESULT " + json.dumps({name: result}), flush=True)
        if invoke(
            "independent_audit",
            [sys.executable, str(project / "scripts/audit_run003_stage2.py"), "--output-root", str(root)],
        ):
            raise RuntimeError("stage2 independent audit failed")
        status.update(stage="complete", audit_passed=True)
    except Exception as error:
        status["failure_reason"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
