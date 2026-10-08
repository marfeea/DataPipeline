"""Audit state4 contact waypoints, accepted artifact, and GUI replay."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from audit_run003_graph import audit_route

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from DataPipe.pipelines.curobo_bootstrap.measured_placement import measured_state4_placement
from DataPipe.pipelines.curobo_bootstrap.pick_place_artifact import verify_hybrid_artifact
from DataPipe.pipelines.curobo_bootstrap.robot_adapter import pose_matrix
from DataPipe.pipelines.curobo_bootstrap.route_experiment import require_height
from DataPipe.pipelines.curobo_bootstrap.route_identity import require_report_identity
from DataPipe.pipelines.curobo_bootstrap.task_targets import solve_legacy_joint_targets
from DataPipe.runtime import ensure_curriculumrl_available
from DataPipe.sequence import quaternion_rotation_error_vector


def audit_measured_placement(task, targets, report, records):
    """Reconstruct same-run calibration and independently solve its full poses."""
    expected = measured_state4_placement(
        task,
        lift_tcp_pose_b=report["lift_tcp_pose_b"],
        object_pose_w_after_lift=report["target_pose_w_after_lift"],
    )
    actual = report["placement_calibration"]
    assert actual["schema"] == expected["schema"]
    for field in ("tcp_from_object", "desired_object_pose_w", "lift_tcp_pose_b", "object_pose_w_after_lift"):
        assert np.allclose(actual[field], expected[field], atol=1e-12, rtol=0)
    for field in ("tcp_waypoint_pose_b", "expected_object_waypoint_pose_w"):
        assert actual[field].keys() == expected[field].keys()
        assert all(np.allclose(actual[field][name], pose, atol=1e-12, rtol=0) for name, pose in expected[field].items())
    ensure_curriculumrl_available()
    from CurriculumRL.logic.aubo_ik import JOINT_LIMITS_RAD, forward_pose, solve_tcp_ik

    rng = np.random.default_rng(task["seed"])
    seeds = [
        np.asarray([r for r in records if r["stage"] == "free_space_transfer"][-1]["q"]),
        np.asarray(targets["legacy_ik"]["q_start"]),
        np.zeros(6),
    ]
    seeds += [rng.uniform(JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1]) for _ in range(16)]
    checks = {}
    for name, pose in expected["tcp_waypoint_pose_b"].items():
        for seed in seeds:
            solved = solve_tcp_ik(pose[:3], pose[3:], seed=seed)
            if not solved.success or solved.q is None:
                continue
            q = np.asarray(solved.q)
            position, rotation = forward_pose(q)
            target_matrix = pose_matrix(pose[:3], pose[3:])
            p_error = float(np.linalg.norm(position - target_matrix[:3, 3]))
            r_error = float(np.arccos(np.clip((np.trace(target_matrix[:3, :3].T @ rotation) - 1) / 2, -1, 1)))
            if (
                np.all((q >= JOINT_LIMITS_RAD[:, 0]) & (q <= JOINT_LIMITS_RAD[:, 1]))
                and p_error <= 1e-5
                and r_error <= 1e-4
            ):
                checks[name] = dict(q=q.tolist(), fk_position_error_m=p_error, fk_orientation_error_rad=r_error)
                break
        assert name in checks, f"calibrated {name} failed independent full-pose IK"
    labels = dict(place_descent="place_descent_outside_roof", place_entry="place_insert_xy", place="place_settle_z")
    for name, label in labels.items():
        samples = [r for r in records if r.get("contact_waypoint") == label]
        assert samples and name in expected["tcp_waypoint_pose_b"]
        pose = expected["tcp_waypoint_pose_b"][name]
        assert all(np.allclose(r["requested_tcp_pose_b"], pose, atol=1e-12, rtol=0) for r in samples)
        assert np.linalg.norm(np.asarray(samples[-1]["tcp_position_b"]) - pose[:3]) <= 0.01
        assert np.linalg.norm(
            quaternion_rotation_error_vector(samples[-1]["tcp_quaternion_b"], pose[3:])
        ) <= np.deg2rad(1)
        assert np.linalg.norm(
            quaternion_rotation_error_vector(samples[-1]["target_pose_w"][3:], expected["desired_object_pose_w"][3:])
        ) <= np.deg2rad(1)
    return dict(calibration=expected, independent_full_pose_ik=checks)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    evidence = run.with_name(run.name + "_evidence")
    route = read(evidence / "run_report.json")
    targets = read(evidence / "targets.json")
    task = targets["task_snapshot"]
    require_report_identity(route, task, label="routed route")
    require_height(task, route["lift_clearance_m"])
    assert route["state4_routing"] == task["state4_routing"] and route["mode"] == "gui"
    artifact = verify_hybrid_artifact(run)
    assert artifact["snapshot"]["task"] == task
    replay = read(run / "replay_report.json")
    require_report_identity(replay, task, label="GUI replay", world_hash=route["world_hash"])
    assert replay["mode"] == "gui" and replay["passed"]
    graph_audit = audit_route(
        run.parent,
        task["source_index"],
        task["destination_index"],
        expected_lift_clearance_m=route["lift_clearance_m"],
    )
    assert graph_audit["verified"] and graph_audit["status"] == "collected_and_replayed"
    assert graph_audit["selected_run"] == run.name
    integrated = Path(route["integrated_run"])
    report = read(integrated / "report.json")
    records = read(integrated / "trace.json")
    assert report["execution_contract"]["physics_hz"] == route["physics_hz"]
    if route.get("contact_velocity_iterations"):
        solver = report["execution_contract"]["physx_solver"]
        assert solver["solver_type"] == 1 and solver["enable_external_forces_every_iteration"]
        assert solver["min_velocity_iteration_count"] >= route["contact_velocity_iterations"]
    if route.get("contact_flange_velocity_limit_rad_s") is not None:
        flange = report["execution_contract"]["flange_drive"]
        assert flange["joint"] == "Flange"
        assert flange["velocity_limit_rad_s"] == route["contact_flange_velocity_limit_rad_s"]
        assert np.isclose(flange["physx_velocity_limit_rad_s"], flange["velocity_limit_rad_s"], atol=1e-6, rtol=0)
    if route.get("contact_wrist_velocity_limit_rad_s") is not None:
        wrist = report["execution_contract"]["wrist_drive"]
        assert wrist["joint_names"] == ["Joint4", "Joint5", "Flange"]
        assert wrist["velocity_limit_rad_s"] == route["contact_wrist_velocity_limit_rad_s"]
        assert all(
            np.isclose(value, wrist["velocity_limit_rad_s"], atol=1e-6, rtol=0)
            for value in wrist["physx_velocity_limits_rad_s"].values()
        )
    contact_ik = {}
    measured_placement_audit = None
    if task["destination_index"] == 4 and task["state4_routing"].get("place_measured_grasp"):
        measured_placement_audit = audit_measured_placement(task, targets, report, records)
    for name, point in task["contact_waypoint_position_b"].items():
        heading_name = "lift" if name == "pick_low_lift" else ("pick" if name.startswith("pick_") else "place")
        single = dict(task, waypoint_position_b={heading_name: point})
        ik = solve_legacy_joint_targets(single, targets["legacy_ik"]["q_start"])
        assert ik["passed"]
        contact_ik[name] = ik
    if "pick_entry" in task["contact_waypoint_position_b"]:
        assert task["source_index"] == 2
        entry_records = [r for r in records if r.get("contact_waypoint") == "state2_pick_entry_xy"]
        settle_records = [r for r in records if r.get("contact_waypoint") == "state2_pick_settle_z"]
        close = [r for r in records if r["stage"] == "close_gripper"]
        assert entry_records and settle_records and close
        assert records.index(entry_records[-1]) < records.index(settle_records[0])
        assert records.index(settle_records[-1]) < records.index(close[0])
        assert report["state2_pick_entry"]["final_tcp_error_m"] <= 0.01
        assert report["approach"]["final_tcp_error_m"] <= 0.01
        for rows, goal in (
            (entry_records, task["contact_waypoint_position_b"]["pick_entry"]),
            (settle_records, task["waypoint_position_b"]["pick"]),
        ):
            assert all(np.allclose(r["requested_tcp_position_b"], goal, atol=1e-12, rtol=0) for r in rows)
        assert np.linalg.norm(
            np.asarray(settle_records[-1]["tcp_position_b"]) - task["waypoint_position_b"]["pick"]
        ) <= 0.01
    if task["source_index"] == 4:
        low = [r for r in records if r.get("contact_waypoint") == "pick_low_lift"]
        exit_records = [r for r in records if r.get("contact_waypoint") == "pick_exit_xy"]
        assert low and exit_records
        assert records.index(low[-1]) < records.index(exit_records[0])
        assert report["state4_low_lift"]["final_tcp_error_m"] <= 0.01
        assert report["target_lift_distance_m"] >= 0.1
        assert (
            abs(task["waypoint_position_b"]["lift"][2] - task["contact_waypoint_position_b"]["pick_low_lift"][2])
            < 1e-12
        )
    if task["destination_index"] == 4:
        descent = [r for r in records if r.get("contact_waypoint") == "place_descent_outside_roof"]
        insert = [r for r in records if r.get("contact_waypoint") == "place_insert_xy"]
        assert descent and insert
        assert records.index(descent[-1]) < records.index(insert[0])
        entry = task["contact_waypoint_position_b"].get("place_entry", task["waypoint_position_b"]["place"])
        assert task["contact_waypoint_position_b"]["place_descent"][2] == entry[2]
        assert report["state4_place_descent"]["final_tcp_error_m"] <= 0.01
        if "place_entry" in task["contact_waypoint_position_b"]:
            settle = [r for r in records if r.get("contact_waypoint") == "place_settle_z"]
            assert settle and records.index(insert[-1]) < records.index(settle[0])
            assert report["state4_place_entry"]["final_tcp_error_m"] <= 0.01
    expected = np.asarray(task["frames"]["T_WE"]) @ [*task["destination"]["position_e"], 1]
    final_error = float(np.linalg.norm(np.asarray(records[-1]["target_position_w"]) - expected[:3]))
    assert final_error <= 0.02
    result = dict(
        audit_passed=True,
        source_index=task["source_index"],
        destination_index=task["destination_index"],
        state4_routing=task["state4_routing"],
        contact_ik=contact_ik,
        measured_placement_audit=measured_placement_audit,
        artifact_acceptance=artifact["acceptance"],
        final_target_error_m=final_error,
        scope="same-run mixed planning/contact execution and GUI state replay; run003 contact policy retained",
        replay_images=replay["visualization"]["image_count"],
        independent_artifact_world_image_audit=graph_audit,
    )
    (run / "state4_routing_audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k not in ("contact_ik", "artifact_acceptance")}))


if __name__ == "__main__":
    main()
