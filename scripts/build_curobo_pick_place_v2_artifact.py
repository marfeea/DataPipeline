"""Freeze one passed same-run Isaac/CuRobo V2 integration trace as a validated artifact."""

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.curobo.world import MeshConfig, WorldConfig
from DataPipe.pipelines.curobo_bootstrap.collision_model import load_collision_model
from DataPipe.pipelines.curobo_bootstrap.curobo_collision import CuroboCollision
from DataPipe.pipelines.curobo_bootstrap.measured_placement import require_measured_calibration
from DataPipe.pipelines.curobo_bootstrap.pick_place_artifact import (
    assemble_hybrid_trajectory,
    save_hybrid_trajectory,
    save_json,
    sha256_file,
    validate_hybrid_trajectory,
    verify_hybrid_artifact,
)
from DataPipe.pipelines.curobo_bootstrap.planner import load_trajectory
from DataPipe.pipelines.curobo_bootstrap.robot_adapter import pose_error, pose_matrix
from DataPipe.pipelines.curobo_bootstrap.route_identity import require_report_identity, task_identity
from DataPipe.pipelines.curobo_bootstrap.world_adapter import AttachedSphere, WorldSnapshot


def _executed_fk_residual(waypoints, targets):
    from DataPipe.native.aubo_ik import forward_pose

    residuals = {}
    for name in ("pre_pick", "pick", "lift", "place_high", "place"):
        position, rotation = forward_pose(waypoints[name])
        actual = np.eye(4)
        actual[:3, :3] = rotation
        actual[:3, 3] = position
        pose = targets[name]["tcp_pose_b"]
        position_error, rotation_error = pose_error(pose_matrix(pose[:3], pose[3:]), actual)
        residuals[name] = {
            "position_m": position_error,
            "rotation_rad": rotation_error,
            "passed": bool(position_error <= 0.01 and rotation_error <= 0.05),
        }
    if not all(item["passed"] for item in residuals.values()):
        raise ValueError("executed Legacy waypoint FK residual exceeded 10 mm / 0.05 rad")
    return residuals


def _legacy_collision_diagnostic(arrays, ranges, world_data, collision_model, attachment):
    meshes = tuple(MeshConfig(**mesh) for mesh in world_data["meshes"])
    world = WorldSnapshot(WorldConfig((), world_data["target_name"], meshes))
    target = next(mesh for mesh in meshes if mesh.name == world.config.target_name)
    trays = tuple(mesh for mesh in meshes if mesh.name.startswith("support_tray_"))
    if not trays:
        raise ValueError("production world lacks support tray meshes")
    tray_world = WorldSnapshot(WorldConfig((), world.config.target_name, (target, *trays)))
    checker = CuroboCollision(collision_model, world)
    diagnostic = {
        "hard_gate": True,
        "scope": "robot versus support-tray hard gate; carried-bottle versus tray scores are separate diagnostics",
        "allowed_world_contact": [world.config.target_name],
        "protected_trays": [mesh.name for mesh in trays],
        "before_pick_target_model": "frozen source target mesh for full-world diagnostic; excluded for tray hard gate",
        "after_pick_target_model": (
            "full-world diagnostic uses measured TCP sphere; tray hard gate excludes the carried sphere"
        ),
        "segments": {},
    }

    def query(name, *, tray_only):
        start, end = ranges[name]
        q = arrays["joint_position"][start:end]
        world_scores = []
        self_scores = []
        for offset in range(0, len(q), 64):
            scores = checker.query(q[offset : offset + 64].astype(np.float32), world_ignore_links=("Base_Link",))
            world_scores.extend(np.max(scores["world"].reshape(len(scores["world"]), -1), axis=1).tolist())
            self_scores.extend(np.max(scores["self"].reshape(len(scores["self"]), -1), axis=1).tolist())
        metrics = {
            "frames": len(q),
            "max_world_score": float(max(world_scores)),
            "max_self_score": float(max(self_scores)),
            "world_positive_frames": int(sum(value > 0 for value in world_scores)),
            "self_positive_frames": int(sum(value > 0 for value in self_scores)),
        }
        if tray_only:
            diagnostic["segments"][name]["tray_max_world_score"] = metrics["max_world_score"]
            diagnostic["segments"][name]["tray_positive_frames"] = metrics["world_positive_frames"]
        else:
            diagnostic["segments"][name] = metrics

    before = ("legacy_approach_pick", "close_gripper")
    after = ("legacy_lift", "legacy_place_approach", "open_gripper")
    for name in before:
        query(name, tray_only=False)
    checker.update_world(tray_world.permit_target_contact())
    for name in before:
        query(name, tray_only=True)

    def identify_trays(names, *, attached_sphere=None):
        for name in names:
            if diagnostic["segments"][name]["tray_positive_frames"] == 0:
                diagnostic["segments"][name]["colliding_trays"] = {}
                continue
            start, end = ranges[name]
            joints = arrays["joint_position"][start:end]
            contacts = {}
            for tray in trays:
                one_world = WorldSnapshot(WorldConfig((), world.config.target_name, (target, tray)))
                one_world = (
                    one_world.permit_target_contact()
                    if attached_sphere is None
                    else one_world.attach((attached_sphere,))
                )
                checker.update_world(one_world)
                peak = 0.0
                for offset in range(0, len(joints), 64):
                    scores = checker.query(
                        joints[offset : offset + 64].astype(np.float32),
                        world_ignore_links=("Base_Link",),
                    )
                    peak = max(peak, float(np.max(scores["world"])))
                if peak > 0:
                    contacts[tray.name] = peak
            diagnostic["segments"][name]["colliding_trays"] = contacts

    identify_trays(before)
    sphere = AttachedSphere(tuple(attachment["center_tcp"]), float(attachment["radius_m"]))
    checker.attach(world.attach((sphere,)))
    for name in after:
        query(name, tray_only=False)
    checker.update_world(tray_world.attach((sphere,)))
    for name in after:
        query(name, tray_only=True)
        metrics = diagnostic["segments"][name]
        metrics["robot_and_carried_bottle_tray_max_world_score"] = metrics["tray_max_world_score"]
        metrics["robot_and_carried_bottle_tray_positive_frames"] = metrics["tray_positive_frames"]
    checker.detach(tray_world.permit_target_contact())
    for name in after:
        query(name, tray_only=True)
    identify_trays(after)
    diagnostic["tray_collision_free"] = all(
        item["tray_max_world_score"] <= 0 for item in diagnostic["segments"].values()
    )
    return diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrated-run", type=Path, required=True)
    parser.add_argument("--targets-report", type=Path, required=True)
    parser.add_argument("--world-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-policy", choices=("strict", "run003"), default="strict")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    report = {
        "schema": "curobo_pick_place_v2_plan_report/1",
        "passed": False,
        "stage": "preflight",
        "contact_policy": args.contact_policy,
        "scope": "Same-run Isaac/CuRobo selected-route hybrid plan and state replay; no separate production replay yet",
    }
    try:
        integrated_path = args.integrated_run.resolve()
        integrated = json.loads((integrated_path / "report.json").read_text(encoding="utf-8"))
        targets = json.loads(args.targets_report.read_text(encoding="utf-8"))
        world = json.loads(args.world_report.read_text(encoding="utf-8"))
        task = targets["task_snapshot"]
        report.update(task_identity(task))
        require_report_identity(targets, task, label="targets")
        require_report_identity(world, task, label="world", require_passed=False)
        if not world.get("world_geometry_valid"):
            raise ValueError("selected world lacks validated geometry evidence")
        require_report_identity(integrated, task, label="integrated run", world_hash=world["world_hash"])
        if not targets["legacy_ik"]["passed"]:
            raise ValueError("Legacy target IK did not pass")
        if not all(
            integrated[key]["motion_gen"]["success"] and integrated[key]["acceptance"]["passed"]
            for key in ("curobo_first", "curobo_transfer")
        ):
            raise ValueError("both same-run MotionGen segments must pass full independent acceptance")
        for key in ("curobo_first", "curobo_transfer"):
            metrics = integrated[key]["acceptance"]["metrics"]
            if metrics["max_world_collision_score"] > 0 or metrics["max_self_collision_score"] > 0:
                raise ValueError(f"{key} source collision score is positive")
        if integrated["world_hash"] != world["world_evidence"]["world_hash"]:
            raise ValueError("integrated run world identity differs from frozen production world")
        if "reset_q" not in integrated:
            raise ValueError("integrated run lacks measured same-run Isaac reset q")
        if task["control_hz"] != 60:
            raise ValueError("V2 hybrid trajectory requires 60 Hz")
        collision_model = Path(world["collision_model"])
        asset = load_collision_model(collision_model)
        urdf = Path(asset["urdf"])
        trace_path = integrated_path / "trace.json"
        records = json.loads(trace_path.read_text(encoding="utf-8"))
        first_path = integrated_path / "first_trajectory.npz"
        transfer_path = integrated_path / "transfer_trajectory.npz"
        arrays = assemble_hybrid_trajectory(records, load_trajectory(first_path), load_trajectory(transfer_path))
        acceptance = validate_hybrid_trajectory(arrays, urdf=urdf)
        if acceptance["samples"] != integrated["record_count"]:
            raise ValueError("integrated source frame count differs from assembled trajectory")
        ranges = acceptance["segment_ranges"]
        executed_waypoints = {
            "q_start": integrated["reset_q"],
            "pre_pick": records[ranges["free_space_to_pre_pick"][1] - 1]["q"],
            "pick": records[ranges["legacy_approach_pick"][1] - 1]["q"],
            "lift": records[ranges["legacy_lift"][1] - 1]["q"],
            "place_high": records[ranges["free_space_transfer"][1] - 1]["q"],
            "place": records[ranges["legacy_place_approach"][1] - 1]["q"],
        }
        executed_targets = dict(targets["legacy_ik"]["targets"])
        placement_calibration = None
        if task.get("state4_routing", {}).get("place_measured_grasp") and task["destination_index"] == 4:
            placement_calibration = require_measured_calibration(
                task,
                integrated["placement_calibration"],
                lift_tcp_pose_b=integrated["lift_tcp_pose_b"],
                object_pose_w_after_lift=integrated["target_pose_w_after_lift"],
            )
            executed_targets["place"] = dict(tcp_pose_b=placement_calibration["tcp_waypoint_pose_b"]["place"])
        executed_residual = _executed_fk_residual(executed_waypoints, executed_targets)
        legacy_collision = _legacy_collision_diagnostic(
            arrays, ranges, world["world"], collision_model, integrated["attachment"]
        )
        report["legacy_collision_diagnostic"] = legacy_collision
        legacy_collision["hard_gate"] = args.contact_policy == "strict"
        legacy_collision["scope"] = (
            "robot versus support-tray hard gate; carried-bottle scores separate"
            if args.contact_policy == "strict"
            else "run_003 Legacy contact diagnostic only; not a collision-free claim"
        )
        report["contact_policy"] = args.contact_policy
        report["collision_policy"] = {
            "allowed_world_contact": [world["world"]["target_name"]],
            "protected_trays": legacy_collision["protected_trays"],
            "self_collision_hard_gate": True,
            "tray_collision_hard_gate": args.contact_policy == "strict",
        }
        if args.contact_policy == "strict" and not legacy_collision["tray_collision_free"]:
            colliding_stages = [
                name for name, metrics in legacy_collision["segments"].items() if metrics["tray_max_world_score"] > 0
            ]
            raise ValueError(f"V2 Legacy contact collided with support trays: {colliding_stages}")
        target_w = np.asarray(task["frames"]["T_WE"], dtype=np.float64) @ np.asarray(
            [*task["destination"]["position_e"], 1.0], dtype=np.float64
        )
        final_target_error_m = float(np.linalg.norm(np.asarray(records[-1]["target_position_w"]) - target_w[:3]))
        if final_target_error_m > 0.02:
            raise ValueError("final observed target is more than 20 mm from selected destination")
        snapshot = {
            "schema": "curobo_pick_place_v2_scene_snapshot/1",
            "task": task,
            "world": world["world"],
            "world_hash": integrated["world_hash"],
            "base_pose_w": integrated["base_pose_w"],
            "target_pose_w_at_reset": integrated["target_pose_w_at_reset"],
            "collision_model": str(collision_model.resolve()),
            "collision_model_sha256": sha256_file(collision_model),
            "urdf": str(urdf.resolve()),
            "urdf_sha256": sha256_file(urdf),
            "tcp_contract": "Legacy AuboIKAction / calibrated TCP, xyz+wxyz, E/W/B frames",
        }
        if placement_calibration is not None:
            snapshot["placement_calibration"] = placement_calibration
        report.update(
            stage="readback",
            source_index=task["source_index"],
            destination_index=task["destination_index"],
            source_name=task["source"]["name"],
            destination_name=task["destination"]["name"],
            world_hash=integrated["world_hash"],
            task_hash=task["task_hash"],
            device="cuda:0",
            seed=task["seed"],
            curobo_version="0.7.8",
            legacy_ik_targets=targets["legacy_ik"]["targets"],
            executed_waypoint_q=executed_waypoints,
            executed_waypoint_fk_residual=executed_residual,
            executed_waypoint_tcp_pose_b={name: executed_targets[name]["tcp_pose_b"] for name in executed_targets},
            placement_calibration=placement_calibration,
            legacy_collision_diagnostic=legacy_collision,
            measured={
                "reset_q": integrated["reset_q"],
                "lift_q": integrated["lift_q"],
                "lift_tcp_pose_b": integrated["lift_tcp_pose_b"],
                "target_pose_w_after_lift": integrated["target_pose_w_after_lift"],
                "target_pose_w_after_transfer": integrated["target_pose_w_after_transfer"],
                "final_target_position_w": records[-1]["target_position_w"],
                "final_target_position_error_m": final_target_error_m,
                "pre_pick_tcp_error_m": integrated["pre_pick_tcp_error_m"],
                "place_high_tcp_error_m": integrated["place_high_tcp_error_m"],
                "approach": integrated["approach"],
                "close_gripper": integrated["close_gripper"],
                "lift": integrated["lift"],
                "place_approach": integrated["place_approach"],
                "open_gripper": integrated["open_gripper"],
            },
            motion_gen={key: integrated[key] for key in ("curobo_first", "curobo_transfer")},
            attachment=integrated["attachment"],
            trajectory_acceptance=acceptance,
            source_sha256={
                "integrated_report": sha256_file(integrated_path / "report.json"),
                "integrated_trace": sha256_file(trace_path),
                "first_motion_gen_npz": sha256_file(first_path),
                "transfer_motion_gen_npz": sha256_file(transfer_path),
                "targets_report": sha256_file(args.targets_report),
                "world_report": sha256_file(args.world_report),
            },
        )
        snapshot_path = output / "scene_snapshot.json"
        trajectory_path = output / "trajectory.npz"
        save_json(snapshot_path, snapshot)
        save_hybrid_trajectory(trajectory_path, arrays)
        report["sha256"] = {
            "scene_snapshot.json": sha256_file(snapshot_path),
            "trajectory.npz": sha256_file(trajectory_path),
            "collision_model": snapshot["collision_model_sha256"],
            "urdf": snapshot["urdf_sha256"],
        }
        report["passed"] = True
        report["stage"] = "complete"
        save_json(output / "plan_report.json", report)
        verified = verify_hybrid_artifact(output)
        if verified["acceptance"] != acceptance:
            raise RuntimeError("independent V2 artifact readback changed acceptance")
    except Exception as error:
        report["passed"] = False
        report["failure_reason"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        save_json(output / "plan_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
