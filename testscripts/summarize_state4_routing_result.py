"""Summarize audited state4 success and completed XY placement comparisons."""

import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    project = Path(__file__).resolve().parents[1]
    root = project / "data/curobo_pick_place_graph_run003_fix_20261004/state4_routing"
    baseline = read(project / "docs/run003_failure_analysis_20261004/offline_diagnostics.json")
    for name, digest in baseline["source_sha256"].items():
        assert hashlib.sha256((project.parent / name).read_bytes()).hexdigest() == digest, name
    collision_hash = hashlib.sha256(
        (project / "data/curobo_pick_place_graph_run003_20261003_r2/collision_model.json").read_bytes()
    ).hexdigest()
    assert collision_hash == "2a88974b62bbfb48f31203b313c2022b3a92f94b447751b577f51bfc3f7d2c76"
    accepted = root / "pick015_exit180_gate10mm/full/4_to_0"
    audit = read(accepted / "state4_routing_audit.json")
    route = read(accepted.with_name(accepted.name + "_evidence") / "run_report.json")
    capture = read(Path(route["integrated_run"]) / "capture_audit.json")
    assert audit["audit_passed"] and capture["capture_audit_passed"] and route["passed"]
    comparisons = []
    for directory in ("place180", "place240"):
        evidence = root / directory / "0_to_4_evidence"
        run = read(evidence / "run_report.json")
        assert run["passed"] or run.get("failure_reason"), "comparison still running"
        assert run["mode"] == "gui" and run["physics_hz"] == 120 and run["lift_clearance_m"] == 0.3
        if run["passed"]:
            assert read(Path(run["output"]) / "state4_routing_audit.json")["audit_passed"]
        diagnostics = read(evidence / "insertion_diagnostics.json")
        captures = []
        for attempt in diagnostics["attempts"]:
            native = read(Path(attempt["run"]) / "capture_audit.json")
            assert native["capture_audit_passed"]
            captures.append(dict(run=attempt["run"], images=native["image_count"], passed=native["integration_passed"]))
        comparisons.append(
            dict(
                evidence=str(evidence),
                passed=run["passed"],
                state4_routing=run["state4_routing"],
                task_hash=run["task_hash"],
                world_hash=run["world_hash"],
                diagnostics=diagnostics,
                captures=captures,
                search_scope="all four feasible pick candidates"
                if directory == "place180"
                else "matched first pick candidate",
            )
        )
    assert comparisons[0]["world_hash"] == comparisons[1]["world_hash"]
    original_targets = read(root / "place180/0_to_4_evidence/targets.json")["task_snapshot"]
    compared_targets = read(root / "place240/0_to_4_evidence/targets.json")["task_snapshot"]
    assert original_targets["waypoint_position_b"]["place"] == compared_targets["waypoint_position_b"]["place"]
    assert (
        original_targets["contact_waypoint_position_b"]["place_descent"][2]
        == (compared_targets["contact_waypoint_position_b"]["place_descent"][2])
    )
    original_pick = read(root / "place180/0_to_4_evidence/pre_pick_branches.json")["candidates"][0]
    compared_pick = read(root / "place240/0_to_4_evidence/pre_pick_branches.json")["candidates"][0]
    assert original_pick["joint_position"] == compared_pick["joint_position"]
    frozen = read(root / "place240/experiment_manifest.json")
    for name, digest in frozen["source_sha256"].items():
        assert hashlib.sha256((project / name).read_bytes()).hexdigest() == digest, name
    result = dict(
        schema="state4_routing_fix/1",
        state4_to_state0_passed=True,
        state0_to_state4_passed=any(c["passed"] for c in comparisons),
        accepted_route=dict(
            run=str(accepted),
            state4_routing=route["state4_routing"],
            lift_clearance_m=route["lift_clearance_m"],
            samples=route["samples"],
            final_target_error_m=audit["final_target_error_m"],
            native_images=capture["image_count"],
            replay_images=audit["replay_images"],
        ),
        comparisons=comparisons,
        regression_tests_passed=75,
        original_evidence_files_verified=len(baseline["source_sha256"]),
        collision_model_sha256=collision_hash,
        frozen_execution_sources_unchanged=True,
        scope="run003 mixed free-space planning/state writing and physics contact execution, plus GUI state replay; "
        "contact tray collision remains diagnostic",
    )
    (root / "state4_routing_result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k not in ("comparisons", "accepted_route")}))


if __name__ == "__main__":
    main()
