"""Score frozen, diagnosed execution failures with the unchanged reward v1.

No simulated transitions are invented for exceptions or offline artifact gates.
An explicit execution timeout is attributed to the last captured transition.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from testscripts.evaluate_pick_place_rewards import PHASES, read_json, sha256, transform_position, write_json
from scripts.pick_place_reward_candidate import VERSION, PickPlaceReward, RewardObservation, RewardTask


def safe_input(root, name):
    path = (root / name).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError(f"input must be an existing workspace file: {name}")
    return path


def verify_inputs(root, case):
    required = {case[name] for name in ("trace", "report", "targets", "diagnostic", "analysis_document")}
    if not required.issubset(case["input_sha256"]):
        raise ValueError("all selected evidence must have frozen hashes")
    for name, digest in case["input_sha256"].items():
        if sha256(safe_input(root, name)) != digest:
            raise ValueError(f"frozen failure evidence changed: {name}")


def failure_diagnostic(root, case, report):
    """Check the diagnosis against its original report instead of trusting labels."""
    kind = case["failure_kind"]
    if kind not in {
        "velocity_gate",
        "acceleration_gate",
        "world_endpoint_gate",
        "ik_failure",
        "timeout",
        "noncarry_gate",
    }:
        raise ValueError("unsupported failure diagnosis")
    if kind in {"velocity_gate", "acceleration_gate"}:
        diagnostic = read_json(safe_input(root, case["diagnostic"]))
        route = next(item for item in diagnostic["routes"] if item["route"] == case["route"])
        item = next(item for item in route["integrated"] if item["candidate"] == case["candidate"])
        if item["artifact_failure_reason"] != case["failure_reason"]:
            raise ValueError("artifact rejection does not match selected diagnosis")
        if item["trace_sha256"] != case["input_sha256"][case["trace"]]:
            raise ValueError("diagnostic trace hash differs")
        gate = "velocity_limit_passed" if kind == "velocity_gate" else "legacy_acceleration_passed"
        if item[gate] is not False or report.get("passed") is not True:
            raise ValueError("diagnosis requires a completed execution and failed artifact gate")
        return item
    if report.get("passed") is not False or report.get("failure_reason") != case["failure_reason"]:
        raise ValueError("execution report does not establish selected failure")
    return {"failure_reason": report["failure_reason"], "stage": report["stage"]}


def evaluate_case(root, case, *, model_factory=PickPlaceReward, model_version=VERSION):
    verify_inputs(root, case)
    records = read_json(safe_input(root, case["trace"]))
    report = read_json(safe_input(root, case["report"]))
    targets = read_json(safe_input(root, case["targets"]))
    task = targets["task_snapshot"]
    if len(records) < 2 or report["record_count"] != len(records):
        raise ValueError("insufficient or inconsistent measured trace")
    if targets["task_hash"] != report["task_hash"] or task["task_hash"] != report["task_hash"]:
        raise ValueError("execution/target task identity mismatch")
    if case["route"] != f"{report['source_index']}_to_{report['destination_index']}":
        raise ValueError("selected route differs from execution")
    diagnostic = failure_diagnostic(root, case, report)
    hz = float(task["control_hz"])
    if hz <= 0 or not np.isfinite(hz) or report["execution_contract"]["control_hz"] != hz:
        raise ValueError("control frequency mismatch")
    # Frozen capture contract: record i has time i/control_hz (no per-record clocks).
    t_we = np.asarray(task["frames"]["T_WE"])
    if not np.allclose(t_we[:3, :3], np.eye(3), rtol=0, atol=1e-10):
        raise ValueError("rotated environment needs a full goal-pose transform adapter")
    destination = task["destination"]
    desired = report.get("placement_calibration", {}).get("desired_object_pose_w")
    if desired is None:
        desired = (*transform_position(t_we, destination["position_e"]), *destination["rotation_wxyz"])
    routing = report.get("state4_routing")
    reward_task = RewardTask(
        grasp_position_w=transform_position(t_we, task["source"].get("pre_grasp_tcp_position_e")
                                            or task["source"]["grasp_tcp_position_e"]),
        source_position_w=tuple(report["target_pose_w_at_reset"][:3]),
        goal_pose_w=tuple(desired),
        lift_target_m=routing["pick_lift_m"] if routing else 0.05,
    )
    segments = report.get("tray_collision_diagnostic", {}).get("segments", {})
    indices = set()
    for name, values in segments.items():
        if values["positive_frames"] > 0:
            matching = [i for i, record in enumerate(records) if record["stage"] == name]
            if not matching:
                raise ValueError("geometric diagnostic segment missing from trace")
            indices.add(matching[-1])
    timeout = case["failure_kind"] == "timeout"
    if timeout:
        deadline = re.fullmatch(
            r"RuntimeError: (legacy_(?:approach_pick|lift|place_approach)) "
            r"did not reach TCP tolerance within [1-9]\d* steps",
            report.get("failure_reason", ""),
        )
        if deadline is None:
            raise ValueError("timeout flag requires an explicit recorded execution deadline")
        allowed_stages = {deadline[1]}
        if deadline[1] == "legacy_approach_pick":
            allowed_stages.add("legacy_approach")  # Historical report label before the approach loop.
        if records[-1]["stage"] != deadline[1] or report["stage"] not in allowed_stages:
            raise ValueError("timeout report and captured execution phase differ")
    observations = [
        RewardObservation(
            time_s=i / hz,
            tcp_position_w=transform_position(task["frames"]["T_WB"], record["tcp_position_b"]),
            object_pose_w=tuple(record["target_pose_w"]),
            joint_velocity=tuple(record["dq"]),
            gripper_position=tuple(record["gripper_position"]),
            phase=PHASES[record["stage"]],
            geometry_overlap=i in indices,
            timeout=timeout and i == len(records) - 1,
            evidence_valid=False,
            physical_dynamics=False,
        )
        for i, record in enumerate(records)
    ]
    model = model_factory(reward_task, observations[0])
    rows = []
    for i, obs in enumerate(observations[1:], 1):
        result = model.step(obs)
        rows.append(
            {
                "transition_index": i - 1,
                "from_observation": i - 1,
                "to_observation": i,
                "time_s": obs.time_s,
                "execution_stage": records[i]["stage"],
                "contact_waypoint": records[i].get("contact_waypoint"),
                **result,
            }
        )
    totals = {name: float(sum(row["terms"][name] for row in rows)) for name in rows[0]["terms"]}
    summary = {
        "schema": model_version,
        "case_id": case["case_id"],
        "route": case["route"],
        "candidate": case["candidate"],
        "failure_kind": case["failure_kind"],
        "failure_reason": case["failure_reason"],
        "original_execution_passed": report["passed"],
        "original_failure_stage": report["stage"],
        "diagnostic": diagnostic,
        "reward_config": asdict(model.config),
        "reward_task": asdict(reward_task),
        "input_sha256": case["input_sha256"],
        "source_trace": case["trace"],
        "analysis_document": case["analysis_document"],
        "inputs_unchanged": True,
        "observations": len(records),
        "transitions": len(rows),
        "observed_duration_s": (len(records) - 1) / hz,
        "recorded_prefix_only": records[-1]["stage"] != "open_gripper",
        "starts_after_unrecorded_pre_pick": records[0]["stage"] != "free_space_to_pre_pick",
        "time_basis": "frozen capture convention i/control_hz; first captured observation at zero",
        "diagnostic_return_undiscounted": float(sum(row["reward"] for row in rows)),
        "mean_reward": float(np.mean([row["reward"] for row in rows])),
        "last_reward": rows[-1]["reward"],
        "component_returns": totals,
        "stage_returns": {
            name: {
                "transitions": sum(row["execution_stage"] == name for row in rows),
                "return": float(sum(row["reward"] for row in rows if row["execution_stage"] == name)),
            }
            for name in dict.fromkeys(record["stage"] for record in records)
        },
        "final_position_error_m": rows[-1]["position_error_m"],
        "final_rotation_error_deg": float(np.rad2deg(rows[-1]["rotation_error_rad"])),
        "final_vertical_lift_m": rows[-1]["vertical_lift_m"],
        "last_waypoint": records[-1].get("contact_waypoint"),
        "geometry_diagnostic_available": bool(segments),
        "recognized_failure_event": rows[-1]["failure"],
        "valid_training_rewards": 0,
        "confirmed_success_bonus": totals["success"],
        "event_mapping": (
            "explicit timeout -> last observed transition, -15; no additional transition"
            if timeout
            else "no v1 event matches this diagnosis; no invented generic failure penalty"
        ),
        "invalid_reasons": [
            "offline provisional score; incomplete physical grasp/support/contact evidence",
            "logical transfer/free-space state writing where present",
            "segment overlap charges are retrospectively attributed; missing diagnostics remain unknown",
        ],
    }
    verify_inputs(root, case)
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    selection = read_json(args.cases)
    if selection["model_version"] != VERSION:
        raise ValueError("selected reward version differs")
    output = args.output_dir.resolve()
    if output.exists() or root not in output.parents:
        raise ValueError("select a fresh output directory inside workspace")
    evaluated = []
    for case in selection["cases"]:
        key = case["case_id"]
        if not key.replace("_", "").isalnum():
            raise ValueError("invalid case id")
        for name in case["input_sha256"]:
            if safe_input(root, name).parent == output or output in safe_input(root, name).parents:
                raise ValueError("output cannot contain source evidence")
        evaluated.append(evaluate_case(root, case))
    output.mkdir(parents=True)
    for summary, rows in evaluated:
        key = summary["case_id"]
        write_json(output / f"{key}_summary.json", summary)
        write_json(output / f"{key}_transitions.json", rows)
        with (output / f"{key}_rewards.csv").open("w", encoding="utf-8", newline="") as stream:
            keys = ["transition_index", "time_s", "execution_stage", "reward", "reward_valid", *rows[0]["terms"]]
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({**{name: row[name] for name in keys[:5]}, **row["terms"]})
    write_json(output / "summary.json", [summary for summary, _ in evaluated])
    print(
        json.dumps(
            [
                {
                    name: summary[name]
                    for name in (
                        "case_id",
                        "transitions",
                        "diagnostic_return_undiscounted",
                        "last_reward",
                        "final_position_error_m",
                        "final_rotation_error_deg",
                        "recognized_failure_event",
                    )
                }
                for summary, _ in evaluated
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
