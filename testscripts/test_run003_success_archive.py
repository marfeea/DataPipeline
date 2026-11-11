"""An incremental archive must preserve old evidence and reject corrupted inputs."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def archive_case(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "testscripts/archive_run003_successes.py"
    spec = importlib.util.spec_from_file_location("archive_run003_successes_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = tmp_path / "configs/curobo/route_presets.json"
    config.parent.mkdir(parents=True)
    config.write_text('{"new_config": true}\n', encoding="utf-8")
    output = tmp_path / "data/archive"
    route = output / "4_to_0"
    route.mkdir(parents=True)
    evidence = route / "trajectory.npz"
    evidence.write_bytes(b"unchanged evidence")
    archived_config = output / "route_presets.json"
    archived_config.write_text('{"old_config": true}\n', encoding="utf-8")
    record = dict(
        route="4_to_0", version="existing", priority=2, replay_mode="gui", source_run="original",
        archive_directory="4_to_0", recommended=True, samples=548, replay_images=135,
        files_sha256={"trajectory.npz": module.sha256_file(evidence)},
    )
    manifest = dict(
        schema="run003_success_archive/1", records=[record],
        parameter_config_sha256=module.sha256_file(archived_config),
    )
    (output / "archive_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (output / "README.md").write_text("old index", encoding="utf-8")
    monkeypatch.setattr(module, "PROJECT", tmp_path)
    monkeypatch.setattr(module, "SOURCE_ROOTS", ())
    monkeypatch.setattr(module, "check_original_baseline", lambda: 39)
    monkeypatch.setattr(module, "inventory", lambda: ([record], []))
    monkeypatch.setattr(module, "verify_hybrid_artifact", lambda directory: None)
    monkeypatch.setattr(sys, "argv", [str(script), "--output", str(output), "--append"])
    return module, output, evidence


def test_append_preserves_existing_version_and_configuration_history(archive_case):
    module, output, evidence = archive_case
    original = evidence.read_bytes()
    old_manifest = (output / "archive_manifest.json").read_bytes()
    module.main()
    manifest = json.loads((output / "archive_manifest.json").read_text(encoding="utf-8"))
    assert manifest["archived_version_count"] == 1
    assert manifest["successful_direction_count"] == 1
    assert evidence.read_bytes() == original
    history = list((output / "history").glob("archive_manifest_*.json"))
    assert len(history) == 1 and history[0].read_bytes() == old_manifest
    assert json.loads(next((output / "history").glob("route_presets_*.json")).read_text()) == {"old_config": True}


def test_append_rejects_changed_old_evidence_before_writing(archive_case):
    module, output, evidence = archive_case
    original_manifest = (output / "archive_manifest.json").read_bytes()
    evidence.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="existing archive file differs"):
        module.main()
    assert (output / "archive_manifest.json").read_bytes() == original_manifest
    assert not (output / "history").exists()


def test_append_rejects_paths_outside_archive(archive_case):
    module, output, _ = archive_case
    manifest_path = output / "archive_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["records"][0]["files_sha256"] = {"../../outside": "unused"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="existing archive file differs"):
        module.main()
    assert not (output / "history").exists()
