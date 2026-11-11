"""Read-only current-asset checks; no Isaac scene or trajectory is executed."""

import json

import pytest

from configs.curobo.robot import DEFAULT_COLLISION_MODEL
from scripts.pipelines.curobo_bootstrap.collision_model import derive_collision_model, load_collision_model

REFRESHED_MODEL = DEFAULT_COLLISION_MODEL.with_name("collision_model_v2_mesh_refresh_20261014.json")


def test_current_default_model_passes_strict_source_validation_and_is_fully_regenerated():
    model = load_collision_model(REFRESHED_MODEL)
    old_path = DEFAULT_COLLISION_MODEL.with_name("collision_model_v2.json")
    old = json.loads(old_path.read_text(encoding="utf-8"))
    parameters = {name: old.get(name) for name in (
        "max_cell_m", "padding_m", "attachment_slots", "gripper_mode", "refined_flange_cell_m", "gripper_cell_m",
    )}
    assert model == derive_collision_model(old["urdf"], **parameters)
    assert sum(len(s) for s in model["collision_spheres"].values()) == 391
    assert model["self_collision_ignore"] == old["self_collision_ignore"]
    for before, after in zip(old["mesh_evidence"], model["mesh_evidence"], strict=True):
        assert before["path"] == after["path"] and before["sha256"] != after["sha256"]


def test_source_validation_still_rejects_a_stale_mesh_hash(tmp_path):
    model = load_collision_model(REFRESHED_MODEL)
    old = json.loads(DEFAULT_COLLISION_MODEL.with_name("collision_model_v2.json").read_text(encoding="utf-8"))
    model["mesh_evidence"][0]["sha256"] = old["mesh_evidence"][0]["sha256"]
    modified = tmp_path / "stale.json"
    modified.write_text(json.dumps(model), encoding="utf-8")
    with pytest.raises(ValueError, match="collision mesh evidence no longer matches"):
        load_collision_model(modified)
