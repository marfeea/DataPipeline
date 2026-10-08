"""Independently audit terminal run_003 routes; incomplete routes remain pending."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collect_curobo_pick_place_v2_graph import has_environment_failure

from DataPipe.pipelines.curobo_bootstrap.pick_place_artifact import verify_hybrid_artifact
from DataPipe.pipelines.curobo_bootstrap.route_experiment import require_height


def read(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def live_target_evidence(value):
    if isinstance(value, dict):
        if value.get("pose_source") == "measured rigid root + USD local geometry":
            return value
        values = value.values()
    elif isinstance(value, list):
        values = value
    else:
        return None
    return next((found for item in values if (found := live_target_evidence(item))), None)


def audit_route(root, source, destination, *, expected_lift_clearance_m=0.4):
    name = f"{source}_to_{destination}"
    paths = [root / name, root / (name + "_retry")]
    chosen = next(
        (
            p
            for p in reversed(paths)
            if (read(p.with_name(p.name + "_evidence") / "run_report.json") or {}).get("passed")
        ),
        next((p for p in reversed(paths) if p.with_name(p.name + "_evidence").exists()), paths[0]),
    )
    evidence = chosen.with_name(chosen.name + "_evidence")
    run = read(evidence / "run_report.json")
    entry = {"route": name, "selected_run": chosen.name, "verified": False}
    if not run or not (run.get("passed") or run.get("failure_reason")):
        return dict(entry, status="pending", stage=(run or {}).get("stage"))
    assert run["source_index"] == source and run["destination_index"] == destination
    assert run["contact_policy"] == "run003"
    assert not has_environment_failure(run), "unresolved runtime failure requires retry"
    targets = read(evidence / "targets.json")
    assert targets["source_index"] == source and targets["destination_index"] == destination
    assert targets["config"]["lift_clearance_m"] == expected_lift_clearance_m, "height differs from frozen experiment"
    assert targets["task_snapshot"]["lift_clearance_m"] == expected_lift_clearance_m
    require_height(targets["task_snapshot"], expected_lift_clearance_m)
    assert targets["config"]["control_hz"] == 60.0 and targets["config"]["seed"] == 7
    assert targets["task_snapshot"]["source_index"] == source
    assert targets["task_snapshot"]["destination_index"] == destination
    assert targets["task_hash"] == run["task_hash"]
    entry["stage"] = run["stage"]
    if not run["passed"]:
        if run["stage"] == "targets":
            failures = {k: v["reason"] for k, v in targets["legacy_ik"]["targets"].items() if not v["passed"]}
            assert failures
            entry["failed_targets"] = failures
        elif run["stage"] == "pre_pick_branches":
            branches = read(evidence / "pre_pick_branches.json")
            assert branches["collision_free_count"] == 0 and branches["candidate_count"] > 0
            entry["candidate_count"] = branches["candidate_count"]
            entry["minimum_self_collision_score"] = min(c["self_collision_score"] for c in branches["candidates"])
        else:
            assert run.get("stages"), "failure has no stage evidence"
            entry["failure_reason"] = run["failure_reason"]
        return dict(entry, status="planning_failed", verified=True)
    artifact = verify_hybrid_artifact(chosen)
    plan = artifact["report"]
    replay = read(chosen / "replay_report.json")
    if not replay:
        return dict(entry, status="pending_replay")
    assert replay["passed"] and replay["state_replay_only"]
    assert plan["contact_policy"] == "run003" and not plan["legacy_collision_diagnostic"]["hard_gate"]
    for filename, digest in replay["source_sha256"].items():
        assert hashlib.sha256((chosen / filename).read_bytes()).hexdigest() == digest
    world = read(evidence / "world/report.json")
    live = live_target_evidence(world)
    assert live and live["live_root_pose_w"] == world["source_target_pose_w"]
    for mesh in world["world"]["meshes"]:
        assert hashlib.sha256(Path(mesh["file_path"]).read_bytes()).hexdigest() == mesh["sha256"]
    attachment = plan["attachment"]
    assert np.linalg.norm(attachment["center_tcp"]) < 0.15, "fixed run_003 grasp has a displaced bottle"
    assert 0.05 < attachment["radius_m"] < 0.15
    manifest = read(chosen / "replay_visualization/frames.json")
    frames = manifest["frames"]
    assert len(frames) == replay["render_calls"] // 12 * 3 == replay["visualization"]["image_count"]
    assert {f["camera"] for f in frames} == {"HandCamera", "CameraSensor_2", "CameraSensor_3"}
    for frame in frames:
        assert frame["render_frame"] == frame["trajectory_index"] + 1
        with Image.open(chosen / "replay_visualization" / frame["rgb"]) as image:
            image.verify()
    entry.update(
        image_count=len(frames),
        attachment=attachment,
        world_hash=replay["world_hash"],
        tray_collision_free=plan["legacy_collision_diagnostic"].get("tray_collision_free"),
        visual_content_scope="PNG decoding verified; scene content requires separate visual inspection",
    )
    return dict(entry, status="collected_and_replayed", verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    experiment = read(root / "experiment_manifest.json")
    routes = []
    for source in range(5):
        for destination in range(5):
            if source == destination:
                continue
            try:
                name = f"{source}_to_{destination}"
                clearance = (
                    experiment["route_lift_clearance_m"][name]
                    if experiment and "route_lift_clearance_m" in experiment
                    else 0.4
                )
                routes.append(audit_route(root, source, destination, expected_lift_clearance_m=clearance))
            except Exception as error:
                routes.append(
                    {
                        "route": f"{source}_to_{destination}",
                        "verified": False,
                        "status": "audit_failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    report = {
        "scope": "all 20 directed routes; run_003 planning and state replay, "
        "not physics execution or strict tray compliance",
        "verified_count": sum(r["verified"] for r in routes),
        "collected_and_replayed_count": sum(r["status"] == "collected_and_replayed" for r in routes),
        "routes": routes,
    }
    report["complete"] = report["verified_count"] == 20
    (root / "independent_graph_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "routes"}, ensure_ascii=False))
    for route in routes:
        if route["status"] == "audit_failed":
            print(json.dumps(route, ensure_ascii=False))


if __name__ == "__main__":
    main()
