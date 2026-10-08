from dataclasses import asdict

import numpy as np
import pytest

from configs.curobo.world import CuboidConfig, MeshConfig, WorldConfig
from DataPipe.pipelines.curobo_bootstrap.contracts import config_hash
from DataPipe.pipelines.curobo_bootstrap.world_adapter import (
    AttachedSphere,
    AttachmentState,
    WorldSnapshot,
    aabb_bounds_in_frame,
    attached_sphere_from_local_aabb,
    cuboid_clearance,
    cuboid_from_world_aabb,
    curobo_world_dict,
)


def world():
    return WorldConfig(
        (
            CuboidConfig("table", (0, 0, -0.1, 1, 0, 0, 0), (2, 2, 0.2)),
            CuboidConfig("target", (0.3, 0, 0.05, 1, 0, 0, 0), (0.04, 0.04, 0.1)),
        ),
        "target",
    )


def test_attach_detach_is_atomic_and_static_hash_is_stable():
    before = WorldSnapshot(world())
    attached = before.attach((AttachedSphere((0, 0, 0.03), 0.02),))
    assert attached.target_state is AttachmentState.ATTACHED
    assert [item.name for item in attached.active_cuboids] == ["table"]
    legacy_payload = asdict(world())
    legacy_payload.pop("meshes")
    assert attached.world_hash == before.world_hash == config_hash(legacy_payload)
    assert attached.detach() == before
    with pytest.raises(RuntimeError):
        before.detach()
    with pytest.raises(RuntimeError):
        attached.attach((AttachedSphere((0, 0, 0), 0.1),))


def test_attached_spheres_transform_from_tcp_to_base():
    snapshot = WorldSnapshot(world()).attach((AttachedSphere((0.1, 0, 0), 0.02),))
    actual = snapshot.attached_spheres_b(np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]]))
    np.testing.assert_allclose(actual, [[1, 2.1, 3, 0.02]])


def test_curobo_payload_removes_only_attached_target():
    snapshot = WorldSnapshot(world())
    assert list(curobo_world_dict(snapshot)["cuboid"]) == ["table", "target"]
    attached = snapshot.attach((AttachedSphere((0, 0, 0), 0.01),))
    assert list(curobo_world_dict(attached)["cuboid"]) == ["table"]


def test_mesh_world_dict_and_hash_ignore_run_specific_file_path(tmp_path):
    digest = "a" * 64
    first = MeshConfig("station", (0, 0, 0, 1, 0, 0, 0), str((tmp_path / "a.obj").resolve()), digest, "/World/M")
    second = MeshConfig("station", (0, 0, 0, 1, 0, 0, 0), str((tmp_path / "b.obj").resolve()), digest, "/World/M")
    a = WorldSnapshot(WorldConfig(world().cuboids, "target", (first,)))
    b = WorldSnapshot(WorldConfig(world().cuboids, "target", (second,)))
    assert a.world_hash == b.world_hash
    assert curobo_world_dict(a)["mesh"]["station"]["file_path"] == first.file_path


def test_target_contact_policy_keeps_trays_and_self_collision_model(tmp_path):
    digest = "b" * 64
    target = MeshConfig("target", (0, 0, 0, 1, 0, 0, 0), str((tmp_path / "target.obj").resolve()), digest, "/World/sample_bottle")
    tray = MeshConfig("support_tray_01", (0, 0, 0, 1, 0, 0, 0), str((tmp_path / "tray.obj").resolve()), digest, "/World/tray")
    station = MeshConfig("workstation", (0, 0, 0, 1, 0, 0, 0), str((tmp_path / "station.obj").resolve()), digest, "/World/station")
    world = WorldSnapshot(WorldConfig((), "target", (target, tray, station)))
    contact = world.permit_target_contact()
    assert [mesh.name for mesh in contact.active_meshes] == ["support_tray_01", "workstation"]
    assert list(curobo_world_dict(contact)["mesh"]) == ["support_tray_01", "workstation"]
    assert contact.world_hash == world.world_hash
    assert contact.target_state is AttachmentState.WORLD
    attached = contact.attach((AttachedSphere((0, 0, 0), 0.01),))
    assert [mesh.name for mesh in attached.active_meshes] == ["support_tray_01", "workstation"]


def test_target_contact_policy_cannot_exempt_a_support_tray(tmp_path):
    digest = "c" * 64
    tray = MeshConfig("support_tray_01", (0, 0, 0, 1, 0, 0, 0), str((tmp_path / "tray.obj").resolve()), digest, "/World/tray")
    snapshot = WorldSnapshot(WorldConfig((), tray.name, (tray,)))
    with pytest.raises(ValueError, match="support trays"):
        snapshot.permit_target_contact()


def test_attaching_a_mesh_target_removes_only_that_mesh(tmp_path):
    digest = "b" * 64
    target = MeshConfig(
        "target", (0, 0, 0, 1, 0, 0, 0), str((tmp_path / "target.obj").resolve()), digest, "/World/T"
    )
    station = MeshConfig(
        "station", (0, 0, 0, 1, 0, 0, 0), str((tmp_path / "station.obj").resolve()), digest, "/World/S"
    )
    snapshot = WorldSnapshot(WorldConfig((), "target", (target, station)))
    attached = snapshot.attach((AttachedSphere((0, 0, 0), 0.01),))
    assert list(curobo_world_dict(snapshot)["mesh"]) == ["station", "target"]
    assert list(curobo_world_dict(attached)["mesh"]) == ["station"]


def test_oriented_box_clearance_has_expected_sign():
    cuboid = CuboidConfig("box", (0, 0, 0, 0.7071067811865476, 0, 0, 0.7071067811865476), (2, 1, 1))
    assert cuboid_clearance((0, 0, 0), cuboid) == pytest.approx(-0.5)
    assert cuboid_clearance((0, 1.1, 0), cuboid) == pytest.approx(0.1)


@pytest.mark.parametrize("value", [0, -1, float("nan")])
def test_invalid_attached_sphere_rejected(value):
    with pytest.raises(ValueError):
        AttachedSphere((0, 0, 0), value)


def test_snapshot_rejects_partial_state_mutation():
    with pytest.raises(ValueError):
        WorldSnapshot(world(), AttachmentState.ATTACHED, ())
    with pytest.raises(ValueError):
        WorldSnapshot(world(), AttachmentState.WORLD, (AttachedSphere((0, 0, 0), 0.1),))


def test_world_aabb_is_expressed_as_base_oriented_box():
    base_pose_w = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
    cuboid = cuboid_from_world_aabb("obstacle", (1, 3, 3), (3, 5, 5), base_pose_w)
    np.testing.assert_allclose(cuboid.pose_b[:3], (2, -1, 1), atol=1e-12)
    np.testing.assert_allclose(cuboid.dimensions_m, (2, 2, 2), atol=1e-12)
    np.testing.assert_allclose(np.abs(cuboid.pose_b[3:]), (np.sqrt(0.5), 0, 0, np.sqrt(0.5)), atol=1e-12)


def test_conservative_attached_sphere_contains_all_aabb_corners():
    sphere = attached_sphere_from_local_aabb((-0.01, -0.02, -0.03), (0.03, 0.02, 0.09), padding_m=0.001)
    corners = np.array([[x, y, z] for x in (-0.01, 0.03) for y in (-0.02, 0.02) for z in (-0.03, 0.09)])
    distances = np.linalg.norm(corners - np.asarray(sphere.center_tcp), axis=1)
    assert float(distances.max()) == pytest.approx(sphere.radius_m - 0.001)
    assert np.all(distances <= sphere.radius_m)


def test_world_aabb_corners_are_rebounded_in_object_frame():
    frame_pose_w = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
    minimum, maximum = aabb_bounds_in_frame((1, 2, 3), (3, 6, 5), frame_pose_w)
    np.testing.assert_allclose(minimum, (0, -2, 0), atol=1e-12)
    np.testing.assert_allclose(maximum, (4, 0, 2), atol=1e-12)
