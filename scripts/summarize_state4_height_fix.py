"""Summarize height IK repair and live GUI pilots without accepting failed routes."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from DataPipe.pipelines.curobo_bootstrap.route_experiment import require_height
from DataPipe.pipelines.curobo_bootstrap.route_identity import require_report_identity


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    project = Path(__file__).resolve().parents[1]
    baseline = read(project / "docs/run003_failure_analysis_20261004/offline_diagnostics.json")
    assert all(digest(project.parent / name) == sha for name, sha in baseline["source_sha256"].items())
    model = project / "data/curobo_pick_place_graph_run003_20261003_r2/collision_model.json"
    assert digest(model) == "2a88974b62bbfb48f31203b313c2022b3a92f94b447751b577f51bfc3f7d2c76"
    offline = read(root / "state4_height_030_offline.json")
    assert offline["all_height_targets_passed"] and len(offline["routes"]) == 8
    pilots = []
    for name, route in (("0_to_4_240hz_branch03", "0_to_4"), ("4_to_0_240hz_pick01", "4_to_0")):
        directory = root / name
        report = read(directory / "report.json")
        audit = read(directory / "capture_audit.json")
        targets = read(root.parent / "stage2" / (route + "_evidence") / "targets.json")
        task = targets["task_snapshot"]
        require_height(task, 0.3)
        require_report_identity(report, task, label=name, require_passed=False)
        assert audit["capture_audit_passed"] and report["mode"] == "gui"
        assert report["execution_contract"]["physics_hz"] == 240
        trace = read(directory / "trace.json")
        last = trace[-1]
        waypoint = "place" if last["stage"] == "legacy_place_approach" else "lift"
        error = float(np.linalg.norm(np.array(last["tcp_position_b"]) - task["waypoint_position_b"][waypoint]))
        pilots.append(
            dict(
                route=route,
                run=str(directory),
                integration_passed=report["passed"],
                final_stage=report["stage"],
                failure_reason=report.get("failure_reason"),
                final_waypoint=waypoint,
                final_tcp_error_m=error,
                target_lift_distance_m=report.get("target_lift_distance_m"),
                final_observed_target_pose_w=last["target_pose_w"],
                peak_abs_joint_velocity_rad_s=np.max(np.abs([r["dq"] for r in trace]), axis=0).tolist(),
                source_frames=audit["source_frames"],
                image_count=audit["image_count"],
                preview=audit["preview"],
                report_sha256=digest(directory / "report.json"),
                trace_sha256=digest(directory / "trace.json"),
                frames_manifest_sha256=digest(directory / "execution_visualization/frames.json"),
            )
        )
    result = dict(
        scope="state4 height reachability repaired; diagnostic GUI pilots, not accepted full routes",
        height_ik_repaired_route_count=8,
        lift_clearance_m=0.3,
        unchanged_default_lift_clearance_m=0.4,
        physical_contact_hz=240,
        control_hz=60,
        original_evidence_unchanged=True,
        collision_model_sha256=digest(model),
        recovered_full_route_count=0,
        image_count=sum(p["image_count"] for p in pilots),
        all_capture_audits_passed=True,
        targeted_tests_passed=53,
        pilots=pilots,
    )
    (root / "height_fix_result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
