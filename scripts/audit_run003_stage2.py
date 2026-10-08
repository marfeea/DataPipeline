"""Independently audit all eight state4 height experiments and GUI captures."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_run003_graph import audit_route
from PIL import Image

from DataPipe.pipelines.curobo_bootstrap.pick_place_artifact import verify_hybrid_artifact
from DataPipe.pipelines.curobo_bootstrap.route_experiment import require_height
from DataPipe.pipelines.curobo_bootstrap.route_identity import require_report_identity


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    manifest = read(root / "experiment_manifest.json")
    expected_routes = {f"{s}_to_{d}" for s in range(5) for d in range(5) if s != d and 4 in (s, d)}
    assert set(manifest["routes"]) == expected_routes
    assert manifest["mode"] == "gui" and manifest["camera_resolution"] == [640, 480]
    assert set(manifest["route_lift_clearance_m"].values()) == {0.3}
    model = Path(manifest["collision_model"])
    assert digest(model) == manifest["collision_model_sha256"]
    for filename, expected in {**manifest["implementation_sha256"], **manifest["original_evidence_sha256"]}.items():
        assert digest(Path(filename)) == expected, filename
    routes = []
    for name in manifest["routes"]:
        source, destination = (int(item) for item in name.split("_to_"))
        evidence = root / (name + "_evidence")
        run = read(evidence / "run_report.json")
        targets = read(evidence / "targets.json")
        task = targets["task_snapshot"]
        require_height(task, manifest["route_lift_clearance_m"][name])
        require_report_identity(targets, task, label=name + " targets")
        require_report_identity(run, task, label=name + " run", require_passed=False)
        assert run["mode"] == "gui" and run["lift_clearance_m"] == 0.3
        assert run["collision_model"] == str(model)
        assert all("--headless" not in stage["command"] for stage in run["stages"])
        command = next(stage["command"] for stage in run["stages"] if stage["stage"] == "targets")
        assert float(command[command.index("--lift-clearance-m") + 1]) == 0.3
        original = read(Path(manifest["original_root"]) / "evidence" / (name + "_evidence") / "targets.json")
        assert task["task_hash"] != original["task_hash"]
        for waypoint in ("pre_pick", "pick", "place"):
            assert task["waypoint_position_b"][waypoint] == original["task_snapshot"]["waypoint_position_b"][waypoint]
        assert len(targets["legacy_ik"]["targets"]) == 5 and targets["legacy_ik"]["passed"]
        lift_w = np.asarray(task["frames"]["T_WB"]) @ np.array([*task["waypoint_position_b"]["lift"], 1])
        entry = audit_route(root, source, destination, expected_lift_clearance_m=0.3)
        assert entry["verified"] and entry["status"] in ("collected_and_replayed", "planning_failed"), entry
        entry.update(target_ik_passed=True, lift_clearance_m=0.3, lift_world_z_m=float(lift_w[2]), mode="gui")
        if run["passed"]:
            artifact = verify_hybrid_artifact(root / name)
            require_height(artifact["snapshot"]["task"], 0.3)
            assert artifact["snapshot"]["collision_model_sha256"] == manifest["collision_model_sha256"]
            replay = read(root / name / "replay_report.json")
            require_report_identity(replay, task, label=name + " replay", world_hash=run["world_hash"])
            assert replay["mode"] == "gui" and replay["lift_clearance_m"] == 0.3
            for report_path in (run["contact_probe"], str(Path(run["integrated_run"]) / "report.json")):
                contact = read(Path(report_path))
                assert contact["passed"] and contact["lift_clearance_m"] == 0.3
                require_report_identity(contact, task, label=name + " contact", world_hash=run["world_hash"])
            frames = read(root / name / "replay_visualization/frames.json")["frames"]
            for frame in frames:
                with Image.open(root / name / "replay_visualization" / frame["rgb"]) as image:
                    assert list(image.size) == manifest["camera_resolution"]
            entry["samples"] = artifact["acceptance"]["samples"]
        routes.append(entry)
    report = dict(
        audit_passed=True,
        scope="all eight state4 routes: height IK, full planning outcome and GUI state replay/capture",
        original_evidence_unchanged=True,
        all_height_targets_passed=True,
        recovered_route_count=sum(r["status"] == "collected_and_replayed" for r in routes),
        blocked_route_count=sum(r["status"] == "planning_failed" for r in routes),
        routes=routes,
        experiment_manifest_sha256=digest(root / "experiment_manifest.json"),
    )
    output = root / "stage2_audit.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "routes"}))


if __name__ == "__main__":
    main()
