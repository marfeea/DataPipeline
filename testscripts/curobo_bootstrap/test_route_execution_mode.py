import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import collect_isaacsim
from scripts.collect_isaacsim import _run_stage, _stage_command


def test_gui_removes_headless_from_isaac_stage_and_preserves_task_arguments():
    command = _stage_command(
        "python.exe",
        "targets.py",
        ("--source-index", 3, "--destination-index", 0, "--output", "targets.json", "--headless"),
        gui=True,
    )
    assert command == [
        "python.exe",
        "targets.py",
        "--source-index",
        "3",
        "--destination-index",
        "0",
        "--output",
        "targets.json",
    ]


def test_default_execution_keeps_headless_flag():
    assert _stage_command("python.exe", "contact.py", ("--headless",), gui=False) == [
        "python.exe",
        "contact.py",
        "--headless",
    ]


def test_gui_does_not_change_non_isaac_planner_arguments():
    arguments = ("--disable-cuda-graph", "--graph-first", "--max-attempts", 4)
    assert _stage_command("python.exe", "probe.py", arguments, gui=True) == _stage_command(
        "python.exe",
        "probe.py",
        arguments,
        gui=False,
    )


def simulate_stage(monkeypatch, tmp_path, results):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    output = evidence / "contact_probe_01"
    pending = iter(results)

    def run(command, **kwargs):
        assert Path(command[command.index("--output") + 1]) == output
        # A retry must archive the previous output before starting another process.
        assert not output.exists()
        output.mkdir()
        (output / "report.json").write_text(json.dumps(next(pending)), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(collect_isaacsim.subprocess, "run", run)
    monkeypatch.setattr(collect_isaacsim.time, "sleep", lambda _: None)
    records = []
    result = _run_stage(
        ["python.exe", "probe.py", "--output", str(output)],
        evidence=evidence, stage="contact_probe_01", route="4_to_1", records=records,
        save_status=lambda: None,
    )
    return result, evidence, output, records


def test_camera_timeout_retries_same_candidate_even_with_zero_exit_code(monkeypatch, tmp_path):
    failure = {"passed": False, "failure_reason": "RuntimeError: Native camera warmup did not complete"}
    result, evidence, output, records = simulate_stage(
        monkeypatch, tmp_path, [failure, {"passed": True}]
    )
    assert result == 0
    assert [r["runtime_attempt"] for r in records] == [1, 2]
    assert [r["passed"] for r in records] == [False, True]
    archived = evidence / "runtime_retries/contact_probe_01_1/contact_probe_01/report.json"
    assert json.loads(archived.read_text()) == failure
    assert json.loads((output / "report.json").read_text())["passed"]


def test_camera_retries_stop_after_three_attempts_and_retain_all_failures(monkeypatch, tmp_path):
    failure = {"passed": False, "failure_reason": "Native render did not become usable: ['HandCamera']"}
    _, evidence, output, records = simulate_stage(monkeypatch, tmp_path, [failure] * 3)
    assert len(records) == 3
    assert all(not record["passed"] for record in records)
    assert all(record["failure_reason"] == failure["failure_reason"] for record in records)
    assert len(list((evidence / "runtime_retries").glob("*/*/report.json"))) == 2
    assert json.loads((output / "report.json").read_text()) == failure


def test_ik_failure_is_reported_without_runtime_retry(monkeypatch, tmp_path):
    failure = {"passed": False, "failure_reason": "Legacy IK failed in legacy_lift"}
    _, _, _, records = simulate_stage(monkeypatch, tmp_path, [failure])
    assert len(records) == 1
    assert not records[0]["retryable_runtime_failure"]
    assert records[0]["failure_reason"] == failure["failure_reason"]


def test_runtime_retry_cannot_move_output_outside_route_evidence(monkeypatch, tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    output = tmp_path / "unrelated"
    output.mkdir()
    failure = {"passed": False, "failure_reason": "Native camera warmup did not complete"}
    (output / "report.json").write_text(json.dumps(failure), encoding="utf-8")
    monkeypatch.setattr(collect_isaacsim.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    with pytest.raises(ValueError, match="inside this route's evidence"):
        _run_stage(["python.exe", "probe.py", "--output", str(output)], evidence=evidence,
                   stage="contact_probe_01", route="4_to_1", records=[], save_status=lambda: None)
    assert json.loads((output / "report.json").read_text()) == failure
