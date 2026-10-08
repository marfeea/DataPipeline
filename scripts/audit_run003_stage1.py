"""Verify the two stage1 routes against the frozen experiment and source evidence."""

import argparse
import hashlib
import json
from pathlib import Path

from audit_run003_graph import audit_route
from PIL import Image

from DataPipe.pipelines.curobo_bootstrap.pick_place_artifact import verify_hybrid_artifact


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    project = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["routes"] == ["3_to_0", "3_to_1"]
    assert manifest["lift_clearance_m"] == 0.4 and manifest["seed"] == 7
    assert digest(Path(manifest["collision_model"])) == manifest["collision_model_sha256"]
    for filename, expected in manifest["implementation_sha256"].items():
        assert digest(Path(filename)) == expected
    baseline = json.loads(
        (project / "docs/run003_failure_analysis_20261004/offline_diagnostics.json").read_text(encoding="utf-8")
    )
    assert all(digest(project.parent / name) == expected for name, expected in baseline["source_sha256"].items())
    routes = []
    for destination in (0, 1):
        entry = audit_route(root, 3, destination)
        assert entry["verified"] and entry["status"] == "collected_and_replayed", entry
        run = root / entry["selected_run"]
        artifact = verify_hybrid_artifact(run)
        entry["samples"] = artifact["acceptance"]["samples"]
        if manifest.get("mode") == "gui":
            route_report = json.loads(
                (root / (entry["selected_run"] + "_evidence") / "run_report.json").read_text(encoding="utf-8")
            )
            replay = json.loads((run / "replay_report.json").read_text(encoding="utf-8"))
            assert route_report["mode"] == replay["mode"] == "gui"
            assert all("--headless" not in stage["command"] for stage in route_report["stages"])
            frames = json.loads((run / "replay_visualization/frames.json").read_text(encoding="utf-8"))["frames"]
            for frame in frames:
                with Image.open(run / "replay_visualization" / frame["rgb"]) as image:
                    assert list(image.size) == manifest["camera_resolution"]
            entry.update(mode="gui", resolution=manifest["camera_resolution"])
        targets = json.loads(
            (root / (entry["selected_run"] + "_evidence") / "targets.json").read_text(encoding="utf-8")
        )
        entry["target_seed_selection"] = {
            name: dict(seed_source=target["seed_source"], attempt_count=len(target["attempts"]))
            for name, target in targets["legacy_ik"]["targets"].items()
        }
        if destination == 0:
            high = targets["legacy_ik"]["targets"]["place_high"]
            assert not high["attempts"][0]["passed"] and high["selected_attempt_index"] > 0
            assert targets["task_hash"] == "37791d56822b1e11c9bfc90d2ec21ba87347d8c09a84658f24d2b337f9c4f198"
        routes.append(entry)
    report = dict(
        passed=True,
        scope="3_to_0 recovery and 3_to_1 regression in run003 planning/state replay",
        original_evidence_unchanged=True,
        experiment_manifest_sha256=digest(root / "experiment_manifest.json"),
        routes=routes,
    )
    path = root / "stage1_audit.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
