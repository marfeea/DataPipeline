"""Main-entry routing and audit gates for physical outcomes, including failures."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from configs.curobo.robot import DEFAULT_COLLISION_MODEL
from scripts import collect_isaacsim
from scripts.pipelines.curobo_bootstrap.task_targets import PickPlaceV2Config, task_snapshot


def setup_reference(monkeypatch, tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    task = task_snapshot(config=PickPlaceV2Config(source_index=0, destination_index=2),
                         environment_origin_w=(0, 0, 0), base_pose_w=(1.287, -0.034, 0.816, 1, 0, 0, 0))
    (reference / "scene_snapshot.json").write_text(json.dumps({"task": task}), encoding="utf-8")
    identity = dict(task_hash=task["task_hash"], source_index=0, destination_index=2,
                    source_name="source", destination_name="destination")
    monkeypatch.setattr(collect_isaacsim, "verify_hybrid_artifact", lambda _: {"snapshot": {"task": task}})
    monkeypatch.setattr(collect_isaacsim, "task_identity", lambda _: identity)
    return reference, identity


def test_historical_reference_is_rejected_before_simulation_or_output(monkeypatch, tmp_path):
    reference, _ = setup_reference(monkeypatch, tmp_path)
    snapshot = json.loads((reference / "scene_snapshot.json").read_text())
    del snapshot["task"]["source"]["pre_grasp_tcp_position_e"]
    monkeypatch.setattr(collect_isaacsim, "verify_hybrid_artifact", lambda _: {"snapshot": snapshot})
    monkeypatch.setattr(collect_isaacsim, "_run_stage", lambda *args, **kwargs: pytest.fail("no simulator launch"))
    output = tmp_path / "physical"
    with pytest.raises(SystemExit):
        collect_isaacsim.main(["--reference-run", str(reference), "--output", str(output), "--gui"])
    assert not output.exists()


@pytest.mark.parametrize("outcome", ["success", "dropped", "grasp_verified_task_incomplete"])
def test_main_reference_collects_and_audits_every_outcome(monkeypatch, tmp_path, outcome):
    reference, identity = setup_reference(monkeypatch, tmp_path)
    output = tmp_path / "physical"
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if Path(command[1]).name == "collect_physical_failure_isaacsim.py":
            output.mkdir()
            execution = dict(identity, passed=True, task_succeeded=outcome == "success", outcome=outcome,
                             samples=100, transitions=99, valid_reward_transitions=0)
            (output / "report.json").write_text(json.dumps(execution), encoding="utf-8")
            return SimpleNamespace(returncode=0)
        assert Path(command[1]).name == "audit_physical_failure.py"
        Path(command[command.index("--report") + 1]).write_text('{"passed": true}', encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(collect_isaacsim.subprocess, "run", run)
    assert collect_isaacsim.main([
        "--reference-run", str(reference), "--output", str(output), "--no-gui",
        "--scenario", "open_after_grasp", "--gripper-effort-scale", "0.5",
        "--physics-hz", "240", "--lift-hold-s", "0.4", "--observation-tail-s", "0.8",
        "--contact-velocity-iterations", "4", "--contact-wrist-velocity-limit-rad-s", "0.3",
    ]) == 0
    assert len(commands) == 2  # Frozen references do not execute planning stages.
    execution_command = commands[0]
    for flag, value in (("--scenario", "open_after_grasp"), ("--gripper-effort-scale", "0.5"),
                        ("--physics-hz", "240"), ("--lift-hold-s", "0.4"), ("--observation-tail-s", "0.8"),
                        ("--contact-velocity-iterations", "4"),
                        ("--contact-wrist-velocity-limit-rad-s", "0.3")):
        assert execution_command[execution_command.index(flag) + 1] == value
    assert "--headless" in execution_command
    assert "--stable-placement" in execution_command
    assert "--continue-on-task-failure" in execution_command
    report = json.loads((tmp_path / "physical_evidence/run_report.json").read_text())
    assert report["passed"] and report["execution_passed"]
    assert report["task_succeeded"] == (outcome == "success")
    assert report["outcome"] == outcome
    assert report["execution_mode"] == "physical" and report["backend"] == "isaacsim"
    assert report["valid_reward_transitions"] == 0 and not report["demonstration_eligible"]


def test_gui_collection_can_explicitly_restore_stop_on_failure(monkeypatch, tmp_path):
    reference, _ = setup_reference(monkeypatch, tmp_path)
    commands = []

    def stop(command, **kwargs):
        commands.append(command)
        raise RuntimeError("stop before simulator launch")

    monkeypatch.setattr(collect_isaacsim, "_run_stage", stop)
    assert collect_isaacsim.main([
        "--reference-run", str(reference), "--output", str(tmp_path / "physical"),
        "--gui", "--no-continue-on-task-failure",
    ]) == 1
    assert "--no-continue-on-task-failure" in commands[0]
    assert "--continue-on-task-failure" not in commands[0]
    assert "--headless" not in commands[0]


def test_audit_failure_keeps_recording_and_rejects_main_collection(monkeypatch, tmp_path):
    reference, identity = setup_reference(monkeypatch, tmp_path)
    output = tmp_path / "physical"

    def run(command, **kwargs):
        if "--output" in command:
            output.mkdir()
            (output / "report.json").write_text(json.dumps(dict(
                identity, passed=True, task_succeeded=True, outcome="success",
                samples=100, transitions=99, valid_reward_transitions=0,
            )), encoding="utf-8")
            return SimpleNamespace(returncode=0)
        Path(command[command.index("--report") + 1]).write_text('{"passed": false}', encoding="utf-8")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(collect_isaacsim.subprocess, "run", run)
    assert collect_isaacsim.main(["--reference-run", str(reference), "--output", str(output)]) == 1
    assert (output / "report.json").exists()
    status = json.loads((tmp_path / "physical_evidence/run_report.json").read_text())
    assert not status["passed"] and not status["demonstration_eligible"]
    assert status["stage"] == "physical_audit"


@pytest.mark.parametrize("arguments", [
    ["--source-index", "1"],
    ["--execution-mode", "hybrid"],
    ["--backend", "isaaclab"],
    ["--physics-hz", "121"],
    ["--gripper-effort-scale", "nan"],
    ["--observation-tail-s", "0.1"],
    ["--contact-velocity-iterations", "256"],
    ["--contact-wrist-velocity-limit-rad-s", "nan"],
    ["--contact-wrist-velocity-limit-rad-s", "0.3", "--contact-flange-velocity-limit-rad-s", "0.3"],
])
def test_invalid_physical_settings_fail_before_creating_output(monkeypatch, tmp_path, arguments):
    reference, _ = setup_reference(monkeypatch, tmp_path)
    output = tmp_path / "physical"
    with pytest.raises(SystemExit):
        collect_isaacsim.main(["--reference-run", str(reference), "--output", str(output), *arguments])
    assert not output.exists() and not (tmp_path / "physical_evidence").exists()


@pytest.mark.parametrize("mode,backend", [("physical", "isaacsim"), ("hybrid", "isaaclab")])
def test_new_routes_select_physical_by_default_and_keep_explicit_hybrid(monkeypatch, tmp_path, mode, backend):
    def stop_after_first_stage(command, **kwargs):
        assert command[command.index("--backend") + 1] == backend
        raise RuntimeError("stop before simulator launch")

    monkeypatch.setattr(collect_isaacsim, "_run_stage", stop_after_first_stage)
    argv = ["--source-index", "0", "--destination-index", "2", "--output", str(tmp_path / "run")]
    if mode == "hybrid":
        argv += ["--execution-mode", mode]
    assert collect_isaacsim.main(argv) == 1
    report = json.loads((tmp_path / "run_evidence/run_report.json").read_text())
    assert report["execution_mode"] == mode and report["backend"] == backend


@pytest.mark.parametrize("mode", ["physical", "hybrid"])
def test_planned_route_publishes_the_selected_data_family(monkeypatch, tmp_path, mode):
    """Walk the full orchestrator: planning artifacts must not become physical results."""
    task = dict(source_index=0, destination_index=2, lift_clearance_m=0.4, task_hash="planned-task")
    identity = dict(task_hash=task["task_hash"], source_index=0, destination_index=2,
                    source_name="source", destination_name="destination")
    output = tmp_path / "run"
    stages = []
    monkeypatch.setattr(collect_isaacsim, "task_identity", lambda _: identity)
    monkeypatch.setattr(collect_isaacsim, "require_height", lambda *_: None)

    def check_identity(report, frozen, **kwargs):
        assert report["task_hash"] == frozen["task_hash"]

    monkeypatch.setattr(collect_isaacsim, "require_report_identity", check_identity)

    def verify(path):
        assert (path / "scene_snapshot.json").exists()
        return dict(snapshot={"task": task}, report=dict(identity, passed=True), acceptance={"samples": 75})

    monkeypatch.setattr(collect_isaacsim, "verify_hybrid_artifact", verify)

    def stage(command, **kwargs):
        name = kwargs["stage"]
        stages.append(name)
        flag = "--output" if "--output" in command else "--report"
        path = Path(command[command.index(flag) + 1])
        report = dict(identity, passed=True)
        if name == "targets":
            report.update(task_snapshot=task, config={"lift_clearance_m": 0.4}, legacy_ik={"q_start": [0] * 6})
        elif name == "world":
            assert Path(command[command.index("--collision-model") + 1]) == DEFAULT_COLLISION_MODEL
            report.update(world_geometry_valid=True, world_hash="world", world_evidence={"world_hash": "world"})
        elif "branches" in name:
            report["candidates"] = [{"collision_free": True, "joint_position": [0] * 6}]
        elif name.startswith("contact_probe"):
            report["lift_q"] = [0] * 6
        elif name == "physical_execution":
            assert not output.exists()
            reference = Path(command[command.index("--reference-run") + 1])
            assert reference == tmp_path / "run_evidence/planning_reference"
            verify(reference)
            report.update(task_succeeded=False, outcome="dropped", samples=100, transitions=99,
                          valid_reward_transitions=0)
        if path.suffix == ".json":
            path.write_text(json.dumps(report), encoding="utf-8")
        else:
            path.mkdir()
            (path / "report.json").write_text(json.dumps(report), encoding="utf-8")
            if name.startswith("artifact"):
                (path / "scene_snapshot.json").write_text(json.dumps({"task": task}), encoding="utf-8")
        return 0

    monkeypatch.setattr(collect_isaacsim, "_run_stage", stage)
    assert collect_isaacsim.main([
        "--source-index", "0", "--destination-index", "2", "--output", str(output),
        "--execution-mode", mode,
    ]) == 0
    status = json.loads((tmp_path / "run_evidence/run_report.json").read_text())
    assert status["passed"] and output.exists()
    if mode == "physical":
        assert stages[-2:] == ["physical_execution", "physical_audit"]
        assert not status["task_succeeded"] and status["outcome"] == "dropped"
        assert not (output / "scene_snapshot.json").exists()  # Mock physical output differs from planning output.
    else:
        assert "physical_execution" not in stages and "physical_audit" not in stages
        assert (output / "scene_snapshot.json").exists()
