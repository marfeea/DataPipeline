"""Read frozen run003 artifacts and write isolated, invalid-for-training reward sidecars."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pick_place_reward_candidate import VERSION, PickPlaceReward, RewardObservation, RewardTask

PHASES = {
    "free_space_to_pre_pick": "reach",
    "legacy_approach_pick": "reach",
    "close_gripper": "close",
    "legacy_lift": "lift",
    "free_space_transfer": "transport",
    "legacy_place_approach": "place",
    "open_gripper": "release",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def transform_position(matrix, xyz):
    return tuple((np.asarray(matrix) @ np.r_[xyz, 1.0])[:3])


def evaluate_route(dataset_root, route):
    run = dataset_root / route
    plan = read_json(run / "plan_report.json")
    snapshot = read_json(run / "scene_snapshot.json")
    task = snapshot["task"]
    if route != f"{task['source_index']}_to_{task['destination_index']}":
        raise ValueError("directory/task identity mismatch")
    matches = []
    for trace_path in (dataset_root / (route + "_evidence")).glob("integrated*/trace.json"):
        report_path = trace_path.with_name("report.json")
        if (
            sha256(trace_path) == plan["source_sha256"]["integrated_trace"]
            and sha256(report_path) == plan["source_sha256"]["integrated_report"]
        ):
            matches.append((trace_path, report_path))
    if len(matches) != 1:
        raise ValueError("expected exactly one SHA256-matching execution trace/report")
    trace_path, report_path = matches[0]
    report, records = read_json(report_path), read_json(trace_path)
    if report["task_hash"] != task["task_hash"] or plan["task_hash"] != task["task_hash"]:
        raise ValueError("task hashes do not agree")
    input_paths = [run / name for name in ("trajectory.npz", "scene_snapshot.json", "plan_report.json")]
    input_paths.extend((trace_path, report_path))
    before = {str(path.resolve()): sha256(path) for path in input_paths}
    with np.load(run / "trajectory.npz", allow_pickle=False) as arrays:
        times = arrays["time_s"].copy()
        if times.shape != (len(records),) or len(times) < 2 or not np.isfinite(times).all():
            raise ValueError("trace/time dimensions invalid")
        if not np.allclose(arrays["joint_position"], [record["q"] for record in records], rtol=0, atol=1e-12):
            raise ValueError("artifact joint positions differ from source trace")
        if not np.allclose(arrays["joint_velocity"], [record["dq"] for record in records], rtol=0, atol=1e-12):
            raise ValueError("artifact joint velocities differ from source trace")
    hz = task["control_hz"]
    if not np.allclose(np.diff(times), 1 / hz, rtol=0, atol=1e-9):
        raise ValueError("nonuniform or inconsistent control time")
    desired = report.get("placement_calibration", {}).get("desired_object_pose_w")
    if desired is None:
        t_we = np.asarray(task["frames"]["T_WE"])
        if not np.allclose(t_we[:3, :3], np.eye(3), rtol=0, atol=1e-10):
            raise ValueError("rotated environment needs explicit desired_object_pose_w calibration")
        dest = task["destination"]
        desired = [*transform_position(t_we, dest["position_e"]), *dest["rotation_wxyz"]]
    routing = report.get("state4_routing")
    reward_task = RewardTask(
        grasp_position_w=transform_position(task["frames"]["T_WE"],
                                            task["source"].get("pre_grasp_tcp_position_e")
                                            or task["source"]["grasp_tcp_position_e"]),
        source_position_w=tuple(report["target_pose_w_at_reset"][:3]),
        goal_pose_w=tuple(desired),
        lift_target_m=routing["pick_lift_m"] if routing else 0.05,
    )
    tray_segments = report["tray_collision_diagnostic"]["segments"]
    overlap_segments = {name for name, values in tray_segments.items() if values["positive_frames"] > 0}
    # Segment-level geometric diagnostics do not locate the first colliding frame.
    # Attribute the one-off risk charge at segment end, explicitly for offline review.
    overlap_indices = {
        max(index for index, record in enumerate(records) if record["stage"] == name) for name in overlap_segments
    }
    observations = []
    for index, record in enumerate(records):
        observations.append(
            RewardObservation(
                time_s=float(times[index]),
                tcp_position_w=transform_position(task["frames"]["T_WB"], record["tcp_position_b"]),
                object_pose_w=tuple(record["target_pose_w"]),
                joint_velocity=tuple(record["dq"]),
                gripper_position=tuple(record["gripper_position"]),
                phase=PHASES[record["stage"]],
                geometry_overlap=index in overlap_indices,
                # Existing run003 has logical transfer, missing contact/support evidence.
                evidence_valid=False,
                physical_dynamics=False,
            )
        )
    model = PickPlaceReward(reward_task, observations[0])
    rows = []
    for index, obs in enumerate(observations[1:], start=1):
        result = model.step(obs)
        rows.append(
            {
                "transition_index": index - 1,
                "from_observation": index - 1,
                "to_observation": index,
                "time_s": obs.time_s,
                "execution_stage": records[index]["stage"],
                **result,
            }
        )
    totals = {key: float(sum(row["terms"][key] for row in rows)) for key in rows[0]["terms"]}
    stage_totals = {
        name: {
            "transitions": sum(row["execution_stage"] == name for row in rows),
            "return": float(sum(row["reward"] for row in rows if row["execution_stage"] == name)),
        }
        for name in dict.fromkeys(record["stage"] for record in records)
    }
    after = {str(path.resolve()): sha256(path) for path in input_paths}
    if before != after:
        raise ValueError("inputs changed during evaluation")
    summary = {
        "schema": VERSION,
        "route": route,
        "reward_config": asdict(model.config),
        "reward_task": asdict(reward_task),
        "input_sha256": before,
        "inputs_unchanged": True,
        "observations": len(records),
        "transitions": len(rows),
        "duration_s": float(times[-1] - times[0]),
        "diagnostic_return_undiscounted": float(sum(row["reward"] for row in rows)),
        "mean_reward": float(np.mean([row["reward"] for row in rows])),
        "last_reward": rows[-1]["reward"],
        "component_returns": totals,
        "stage_returns": stage_totals,
        "final_position_error_m": rows[-1]["position_error_m"],
        "final_rotation_error_deg": float(np.rad2deg(rows[-1]["rotation_error_rad"])),
        "physical_lift_vertical_m": report["target_pose_w_after_lift"][2] - report["target_pose_w_at_reset"][2],
        "reported_3d_lift_displacement_m": report["target_lift_distance_m"],
        "valid_training_rewards": sum(row["reward_valid"] for row in rows),
        "confirmed_success": False,
        "success_evidence_status": "unknown",
        "confirmed_success_bonus": totals["success"],
        "invalid_reasons": [
            "logical object transfer / free-space state writing",
            "no observed grasp/contact/support evidence",
            "release only lasts a few frames; no 0.25 s settled post-release evidence",
            "geometric tray overlap; first frame unknown (risk charge at segment end)",
        ],
        "semantics": "N measured states -> N-1 transitions; not the existing NPZ command-row contract",
    }
    return summary, rows


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--routes", nargs="+", default=["0_to_2", "4_to_1"])
    args = parser.parse_args()
    root, output = args.dataset_root.resolve(), args.output_dir.resolve()
    if output == root or root in output.parents:
        raise ValueError("reward output must be separate from source dataset")
    if output.exists():
        raise ValueError("output already exists; select a fresh output directory")
    if any(not route.replace("_to_", "").isdigit() or route.count("_to_") != 1 for route in args.routes):
        raise ValueError("route names must use integer_to_integer")
    evaluated = [(route, *evaluate_route(root, route)) for route in args.routes]
    output.mkdir(parents=True)
    for route, summary, rows in evaluated:
        write_json(output / (route + "_summary.json"), summary)
        write_json(output / (route + "_transitions.json"), rows)
        with (output / (route + "_rewards.csv")).open("w", newline="", encoding="utf-8") as stream:
            keys = ["transition_index", "time_s", "execution_stage", "reward", "reward_valid", *rows[0]["terms"]]
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({**{key: row[key] for key in keys[:5]}, **row["terms"]})
    write_json(output / "summary.json", [summary for _, summary, _ in evaluated])
    print(
        json.dumps(
            [
                {
                    key: summary[key]
                    for key in (
                        "route",
                        "transitions",
                        "diagnostic_return_undiscounted",
                        "mean_reward",
                        "last_reward",
                        "final_position_error_m",
                        "final_rotation_error_deg",
                        "valid_training_rewards",
                    )
                }
                for _, summary, _ in evaluated
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
