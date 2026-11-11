"""Score frozen failures under v2, retaining separate controller-event rewards."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from testscripts.evaluate_failed_pick_place_rewards import evaluate_case as evaluate_v1
from testscripts.evaluate_failed_pick_place_rewards import safe_input
from testscripts.evaluate_pick_place_rewards import read_json, write_json
from scripts.pick_place_reward_candidate_v2 import VERSION, PickPlaceRewardV2


def evaluate_case(root, case):
    models = []

    def factory(task, initial):
        model = PickPlaceRewardV2(task, initial)
        models.append(model)
        return model

    summary, rows = evaluate_v1(root, case, model_factory=factory, model_version=VERSION)
    events = []
    if case["failure_kind"] == "ik_failure":
        match = re.fullmatch(
            r"RuntimeError: Legacy IK failed in (legacy_approach_pick|legacy_lift|legacy_place_approach)",
            case["failure_reason"],
        )
        if match is None or summary["original_failure_stage"] != match[1]:
            raise ValueError("IK cost requires an explicit, stage-matching controller exception")
        event = models[0].abort_ik_failure(match[1])
        event.update(
            {
                "after_observation": summary["observations"] - 1,
                "attribution": "reported after last captured state; before rejected attempt physics",
                "source_report": case["report"],
            }
        )
        events.append(event)
    if case["failure_kind"] == "noncarry_gate" and case["failure_reason"] != (
        "RuntimeError: target was not observed moving with the gripper"
    ):
        raise ValueError("noncarry diagnosis must match the original displacement gate")
    summary["observed_transition_return"] = summary["diagnostic_return_undiscounted"]
    summary["controller_event_return"] = sum(event["reward"] for event in events)
    summary["component_returns"]["ik_failure"] = summary["controller_event_return"]
    summary["diagnostic_return_undiscounted"] += summary["controller_event_return"]
    summary["controller_events"] = len(events)
    summary["recognized_failure_event"] = "ik_failed" if events else summary["recognized_failure_event"]
    summary["last_observed_transition_reward"] = summary.pop("last_reward")
    summary["mean_observed_transition_reward"] = summary.pop("mean_reward")
    summary["return_contract"] = "sum(N-1 observed transition rewards) + sum(separate controller-event rewards)"
    summary["event_mapping"] = (
        "explicit IK abort -> separate -15 controller event; no fabricated frame or transition"
        if events
        else summary["event_mapping"]
    )
    summary["overspeed_event_cost"] = summary["overacceleration_event_cost"] = 0.0
    summary["noncarry_gate_is_confirmed_drop"] = False
    return summary, rows, events


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
        raise ValueError("select a fresh workspace output directory")
    keys = [case["case_id"] for case in selection["cases"]]
    if len(set(keys)) != len(keys) or any(not key.replace("_", "").isalnum() for key in keys):
        raise ValueError("case ids must be unique safe names")
    for case in selection["cases"]:
        for name in case["input_sha256"]:
            if output in safe_input(root, name).parents:
                raise ValueError("output cannot contain original evidence")
    evaluated = [evaluate_case(root, case) for case in selection["cases"]]
    output.mkdir(parents=True)
    for summary, rows, events in evaluated:
        key = summary["case_id"]
        write_json(output / f"{key}_summary.json", summary)
        write_json(output / f"{key}_transitions.json", rows)
        write_json(output / f"{key}_controller_events.json", events)
        with (output / f"{key}_rewards.csv").open("w", encoding="utf-8", newline="") as stream:
            fields = ["transition_index", "time_s", "execution_stage", "reward", "reward_valid", *rows[0]["terms"]]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({**{name: row[name] for name in fields[:5]}, **row["terms"]})
    write_json(output / "summary.json", [summary for summary, _, _ in evaluated])
    print(
        json.dumps(
            [
                {
                    name: summary[name]
                    for name in (
                        "case_id",
                        "transitions",
                        "diagnostic_return_undiscounted",
                        "controller_event_return",
                        "recognized_failure_event",
                    )
                }
                for summary, _, _ in evaluated
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
