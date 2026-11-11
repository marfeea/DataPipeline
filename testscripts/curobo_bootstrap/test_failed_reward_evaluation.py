"""Evidence checks for offline failure scoring, without Isaac or historical data."""

import json

import numpy as np
import pytest

from testscripts.evaluate_failed_pick_place_rewards import evaluate_case
from testscripts.evaluate_pick_place_rewards import sha256


def frozen_case(root, kind="ik_failure"):
    reason = {
        "ik_failure": "RuntimeError: Legacy IK failed in legacy_lift",
        "timeout": "RuntimeError: legacy_place_approach did not reach TCP tolerance within 600 steps",
        "world_endpoint_gate": "ValueError: invalid fixed planning endpoints: world collision",
        "acceleration_gate": "ValueError: V2 Legacy measured acceleration envelope failure",
        "noncarry_gate": "RuntimeError: target was not observed moving with the gripper",
    }[kind]
    records = [
        {
            "stage": stage,
            "tcp_position_b": [x, 0, 0],
            "target_pose_w": [0, 0, z, 1, 0, 0, 0],
            "dq": [0] * 6,
            "gripper_position": [0, 0],
        }
        for stage, x, z in [
            ("legacy_approach_pick", 1, 0),
            ("legacy_lift", 0, 0.05),
            ("legacy_place_approach" if kind == "timeout" else "legacy_lift", 0, 0.05),
        ]
    ]
    report = {
        "record_count": 3,
        "task_hash": "frozen-task",
        "source_index": 2,
        "destination_index": 0,
        "execution_contract": {"control_hz": 60},
        "target_pose_w_at_reset": [0, 0, 0, 1, 0, 0, 0],
        "passed": kind == "acceleration_gate",
        "stage": "complete" if kind == "acceleration_gate" else records[-1]["stage"],
        "failure_reason": None if kind == "acceleration_gate" else reason,
    }
    targets = {
        "task_hash": "frozen-task",
        "task_snapshot": {
            "task_hash": "frozen-task",
            "control_hz": 60,
            "frames": {"T_WE": np.eye(4).tolist(), "T_WB": np.eye(4).tolist()},
            "destination": {"position_e": [1, 0, 0], "rotation_wxyz": [1, 0, 0, 0]},
            "source": {"grasp_tcp_position_e": [0, 0, 0]},
        },
    }
    for name, value in [("trace.json", records), ("report.json", report), ("targets.json", targets)]:
        (root / name).write_text(json.dumps(value), encoding="utf-8")
    (root / "analysis.md").write_text("Recorded failure diagnosis", encoding="utf-8")
    diagnostic = "report.json"
    if kind == "acceleration_gate":
        diagnostic = "measured_dynamics.json"
        (root / diagnostic).write_text(
            json.dumps(
                {
                    "routes": [
                        {
                            "route": "2_to_0",
                            "integrated": [
                                {
                                    "candidate": "example",
                                    "artifact_failure_reason": reason,
                                    "trace_sha256": sha256(root / "trace.json"),
                                    "legacy_acceleration_passed": False,
                                }
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
    return {
        "case_id": "fixture",
        "route": "2_to_0",
        "candidate": "example",
        "failure_kind": kind,
        "failure_reason": reason,
        "trace": "trace.json",
        "report": "report.json",
        "targets": "targets.json",
        "diagnostic": diagnostic,
        "analysis_document": "analysis.md",
        "input_sha256": {
            name: sha256(root / name)
            for name in {"trace.json", "report.json", "targets.json", "analysis.md", diagnostic}
        },
    }


def test_explicit_timeout_charged_once_on_existing_last_transition(tmp_path):
    case = frozen_case(tmp_path, "timeout")
    summary, rows = evaluate_case(tmp_path, case)
    assert summary["transitions"] == len(rows) == 2
    assert [row["terms"]["failure"] for row in rows] == [0, -15]
    assert summary["recognized_failure_event"] == "timeout"
    assert summary["diagnostic_return_undiscounted"] == pytest.approx(sum(row["reward"] for row in rows))
    assert all(row["reward_valid"] is False for row in rows)


@pytest.mark.parametrize("kind", ["ik_failure", "world_endpoint_gate", "acceleration_gate"])
def test_diagnosis_without_v1_event_does_not_invent_physical_collision_or_penalty(tmp_path, kind):
    summary, rows = evaluate_case(tmp_path, frozen_case(tmp_path, kind))
    assert summary["recognized_failure_event"] is None
    assert summary["component_returns"]["failure"] == 0
    assert summary["geometry_diagnostic_available"] is False
    assert summary["recorded_prefix_only"] is True
    assert summary["starts_after_unrecorded_pre_pick"] is True
    assert summary["confirmed_success_bonus"] == 0
    assert summary["valid_training_rewards"] == 0
    assert all(row["reward_valid"] is False for row in rows)


def test_changed_evidence_is_rejected(tmp_path):
    case = frozen_case(tmp_path)
    (tmp_path / "trace.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence changed"):
        evaluate_case(tmp_path, case)


def test_unhashed_trace_is_rejected(tmp_path):
    case = frozen_case(tmp_path)
    del case["input_sha256"]["trace.json"]
    with pytest.raises(ValueError, match="frozen hashes"):
        evaluate_case(tmp_path, case)


def test_task_identity_mismatch_is_rejected_even_with_refreshed_file_hash(tmp_path):
    case = frozen_case(tmp_path)
    targets = json.loads((tmp_path / "targets.json").read_text(encoding="utf-8"))
    targets["task_hash"] = "other-task"
    (tmp_path / "targets.json").write_text(json.dumps(targets), encoding="utf-8")
    case["input_sha256"]["targets.json"] = sha256(tmp_path / "targets.json")
    with pytest.raises(ValueError, match="identity mismatch"):
        evaluate_case(tmp_path, case)


def test_artifact_diagnostic_must_reference_selected_trace(tmp_path):
    case = frozen_case(tmp_path, "acceleration_gate")
    path = tmp_path / "measured_dynamics.json"
    diagnostic = json.loads(path.read_text(encoding="utf-8"))
    diagnostic["routes"][0]["integrated"][0]["trace_sha256"] = "wrong-trace"
    path.write_text(json.dumps(diagnostic), encoding="utf-8")
    case["input_sha256"]["measured_dynamics.json"] = sha256(path)
    with pytest.raises(ValueError, match="trace hash differs"):
        evaluate_case(tmp_path, case)


def test_v2_ik_penalty_is_a_separate_event_preserving_real_transitions(tmp_path):
    from testscripts.evaluate_failed_pick_place_rewards_v2 import evaluate_case as evaluate_v2

    case = frozen_case(tmp_path)
    old, old_rows = evaluate_case(tmp_path, case)
    new, rows, events = evaluate_v2(tmp_path, case)
    assert rows == old_rows
    assert len(rows) == new["transitions"] == 2
    assert new["controller_event_return"] == -15
    assert new["diagnostic_return_undiscounted"] == pytest.approx(old["diagnostic_return_undiscounted"] - 15)
    assert len(events) == 1
    assert events[0]["after_observation"] == 2
    assert events[0]["physical_transition_created"] is False
    assert events[0]["event_time_s"] is None
    assert events[0]["reward_valid"] is False


@pytest.mark.parametrize("kind", ["acceleration_gate", "world_endpoint_gate", "timeout", "noncarry_gate"])
def test_v2_does_not_reclassify_other_failures_as_ik_or_drop(tmp_path, kind):
    from testscripts.evaluate_failed_pick_place_rewards_v2 import evaluate_case as evaluate_v2

    case = frozen_case(tmp_path, kind)
    old, old_rows = evaluate_case(tmp_path, case)
    new, rows, events = evaluate_v2(tmp_path, case)
    assert rows == old_rows
    assert events == []
    assert new["diagnostic_return_undiscounted"] == old["diagnostic_return_undiscounted"]
    assert new["overspeed_event_cost"] == new["overacceleration_event_cost"] == 0
    assert new["noncarry_gate_is_confirmed_drop"] is False


def test_v2_ik_label_requires_original_controller_exception(tmp_path):
    from testscripts.evaluate_failed_pick_place_rewards_v2 import evaluate_case as evaluate_v2

    case = frozen_case(tmp_path, "world_endpoint_gate")
    case["failure_kind"] = "ik_failure"
    with pytest.raises(ValueError, match="explicit, stage-matching"):
        evaluate_v2(tmp_path, case)


@pytest.mark.parametrize("stage", ["legacy_approach_pick", "legacy_lift", "legacy_place_approach"])
def test_deadline_in_each_execution_phase_is_a_timeout(tmp_path, stage):
    case = frozen_case(tmp_path, "timeout")
    report_path = tmp_path / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["stage"] = stage
    report["failure_reason"] = case["failure_reason"] = (
        f"RuntimeError: {stage} did not reach TCP tolerance within 600 steps"
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")
    case["input_sha256"]["report.json"] = sha256(report_path)
    trace_path = tmp_path / "trace.json"
    records = json.loads(trace_path.read_text(encoding="utf-8"))
    records[-1]["stage"] = stage
    trace_path.write_text(json.dumps(records), encoding="utf-8")
    case["input_sha256"]["trace.json"] = sha256(trace_path)
    summary, rows = evaluate_case(tmp_path, case)
    assert summary["recognized_failure_event"] == "timeout"
    assert rows[-1]["terms"]["failure"] == -15
