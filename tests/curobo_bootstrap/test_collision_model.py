import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from DataPipe.pipelines.curobo_bootstrap.collision_model import (
    CollisionSphere,
    aabb_cover_spheres,
    curobo_robot_dict,
    derive_collision_model,
    load_collision_model,
    write_collision_model,
)

URDF = Path("configs/curobo/assets/aubo_e5_project/aubo_e5_project.urdf")


def corners(bounds):
    return np.array(list(__import__("itertools").product(*zip(*bounds, strict=True))))


def test_aabb_grid_is_deterministic_and_covers_corners():
    low = np.array([-0.1, -0.03, 0.2])
    high = np.array([0.15, 0.04, 0.31])
    first = aabb_cover_spheres(low, high, max_cell_m=0.08, padding_m=0.002)
    assert first == aabb_cover_spheres(low, high, max_cell_m=0.08, padding_m=0.002)
    for corner in corners((low, high)):
        assert min(np.linalg.norm(corner - sphere.center) - sphere.radius for sphere in first) <= 1e-12


@pytest.mark.parametrize(
    "minimum,maximum,kwargs",
    [
        ([0, 0], [1, 1], {}),
        ([0, 0, 0], [0, 1, 1], {}),
        ([0, 0, 0], [1, 1, 1], {"max_cell_m": 0}),
        ([0, 0, 0], [1, 1, 1], {"padding_m": -1}),
    ],
)
def test_aabb_grid_rejects_invalid_input(minimum, maximum, kwargs):
    with pytest.raises(ValueError):
        aabb_cover_spheres(minimum, maximum, **kwargs)


def test_collision_sphere_validation():
    with pytest.raises(ValueError):
        CollisionSphere((0, 0), 1)
    with pytest.raises(ValueError):
        CollisionSphere((0, 0, 0), 0)


def test_derived_model_covers_every_recorded_bound_corner():
    model = derive_collision_model(URDF)
    assert model["collision_link_names"][-1] == "gripper_base"
    assert model["attachment_parent"] == "project_tcp"
    assert model["gripper_policy"].startswith("fixed envelope")
    assert model["urdf_sha256"] == hashlib.sha256(URDF.read_bytes()).hexdigest()
    for name, bounds in model["geometry_bounds"].items():
        spheres = [CollisionSphere(tuple(item["center"]), item["radius"]) for item in model["collision_spheres"][name]]
        for corner in corners((bounds["minimum"], bounds["maximum"])):
            assert min(np.linalg.norm(corner - sphere.center) - sphere.radius for sphere in spheres) <= 1e-12


def test_v2_open_gripper_keeps_components_separate_and_refines_flange():
    legacy = derive_collision_model(URDF)
    model = derive_collision_model(
        URDF, gripper_mode="fixed_open_components", refined_flange_cell_m=0.025, padding_m=0.001
    )
    assert model["gripper_policy"].startswith("fixed open-position")
    assert len(model["collision_spheres"]["AUBO_E5_Flange"]) > len(legacy["collision_spheres"]["AUBO_E5_Flange"])
    assert len(model["collision_spheres"]["gripper_base"]) > len(legacy["collision_spheres"]["gripper_base"])


def test_v2_gripper_resolution_is_independent_of_flange():
    coarse = derive_collision_model(
        URDF, gripper_mode="fixed_open_components", refined_flange_cell_m=0.025, padding_m=0.001
    )
    fine = derive_collision_model(
        URDF,
        gripper_mode="fixed_open_components",
        refined_flange_cell_m=0.025,
        gripper_cell_m=0.0125,
        padding_m=0.001,
    )
    assert fine["collision_spheres"]["AUBO_E5_Flange"] == coarse["collision_spheres"]["AUBO_E5_Flange"]
    assert len(fine["collision_spheres"]["gripper_base"]) > len(coarse["collision_spheres"]["gripper_base"])
    assert max(item["radius"] for item in fine["collision_spheres"]["gripper_base"]) < max(
        item["radius"] for item in coarse["collision_spheres"]["gripper_base"]
    )
    with pytest.raises(ValueError, match="gripper_cell_m"):
        derive_collision_model(URDF, gripper_cell_m=0)


def test_model_round_trip_and_source_verification(tmp_path):
    output = tmp_path / "collision_model.json"
    written = write_collision_model(URDF, output)
    assert load_collision_model(output) == written
    with pytest.raises(FileExistsError):
        write_collision_model(URDF, output)
    damaged = json.loads(output.read_text())
    damaged["urdf_sha256"] = "0" * 64
    output.write_text(json.dumps(damaged))
    with pytest.raises(ValueError, match="URDF evidence"):
        load_collision_model(output)


def test_curobo_payload_has_inactive_attachment_slots_and_no_limit_claims():
    model = derive_collision_model(URDF, attachment_slots=3)
    config = curobo_robot_dict(model)["kinematics"]
    assert config["extra_collision_spheres"] == {"attached_object": 3}
    assert config["extra_links"]["attached_object"]["parent_link_name"] == "project_tcp"
    assert config["collision_link_names"][-1] == "attached_object"
    assert "cspace" not in config
    assert "attached_object" not in config["collision_spheres"]


def test_payload_rejects_a_different_urdf(tmp_path):
    model = derive_collision_model(URDF)
    other = tmp_path / "other.urdf"
    other.write_text("<robot name='other'/>")
    with pytest.raises(ValueError, match="does not match"):
        curobo_robot_dict(model, urdf=other)
