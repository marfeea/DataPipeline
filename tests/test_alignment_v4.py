from dataclasses import replace
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from test_datapipe_p0_r import _episode

from DataPipe.alignment import matrix_pose, pose_displacement, pose_matrix, rigid_transform, transform_pose
from DataPipe.collection_alignment import aligned_snapshot, relative_pose_in_axes
from DataPipe.config import ALIGNED_SCHEMA_VERSION, DatasetConfig
from DataPipe.hdf5_schema import validate_episode_group
from DataPipe.hdf5_writer import HDF5TrajectoryWriter
from DataPipe.mamba_sequence_buffer import MambaSequenceBuffer
from DataPipe.r2dreamer import R2DreamerBatch
from DataPipe.records import TerminationReason
from DataPipe.reward import RewardLossConfig, compute_reward_prediction_loss
from DataPipe.validator import validate_hdf5_dataset


def aligned_episode(name="aligned", *, terminal=False):
    episode = _episode(name, steps=2)
    base = pose_matrix((0.3, -0.2, 0.1, 0, 0, np.sqrt(0.5), np.sqrt(0.5)))
    frames = []
    for i, frame in enumerate(episode.observations):
        value = dict(frame.value)
        tcp = np.array((0.01 * i, 0.2, 0.3, 0, 0, 0, 1))
        grasp = np.array((0.4, 0.5, 0.6, 0, 0, 0, 1))
        goal = np.array((0.7, 0.8, 0.9, 0, 0, 1, 0))
        value.update(
            tcp_pose_b=tcp,
            tcp_pose_w=transform_pose(base, tcp),
            grasp_pose_o=grasp,
            grasp_pose_b=grasp,
            grasp_pose_w=transform_pose(base, grasp),
            goal_object_pose_b=goal,
            goal_object_pose_w=transform_pose(base, goal),
            ee_twist_b=(0.12, 0, 0, 0, 0, 0),
            ee_twist_w=(0, 0.12, 0, 0, 0, 0),
            gripper_force=(1.0, -1.0),
            T_w_b=base,
            grasp_in_ee_pose_b=relative_pose_in_axes(tcp, grasp),
            grasp_in_ee_pose_w=relative_pose_in_axes(transform_pose(base, tcp), transform_pose(base, grasp)),
        )
        frames.append(replace(frame, value=value))
    action = (0.5, 0, 0, 0, 0, 0, 0.25, 0.25)
    steps = tuple(replace(s, action=replace(s.action, action=action, action_applied=action)) for s in episode.steps)
    if terminal:
        frames[-1] = replace(frames[-1], boundary=replace(frames[-1].boundary, is_terminal=True))
    return replace(
        episode,
        observations=tuple(frames),
        steps=steps,
        termination_reason=TerminationReason.SUCCESS if terminal else TerminationReason.TIMEOUT,
    )


def writer(path):
    return HDF5TrajectoryWriter(path, DatasetConfig(schema_version=ALIGNED_SCHEMA_VERSION, min_free_disk_bytes=0))


@pytest.mark.parametrize("q", [(0, 0, 0, 1), (1, 0, 0, 0), (0, 1, 0, 0), (0.5, 0.5, 0.5, 0.5)])
def test_quaternion_transform_roundtrip(q):
    pose = (1.0, 2.0, 3.0, *q)
    np.testing.assert_allclose(pose_matrix(matrix_pose(pose_matrix(pose))), pose_matrix(pose), atol=1e-12)
    base = pose_matrix((0.1, 0.2, -0.1, 0, 0, np.sqrt(0.5), np.sqrt(0.5)))
    world = transform_pose(base, pose)
    np.testing.assert_allclose(pose_matrix(transform_pose(np.linalg.inv(base), world)), pose_matrix(pose), atol=1e-12)


def test_reflection_and_non_unit_quaternion_rejected():
    invalid = np.eye(4)
    invalid[0, 0] = -1
    with pytest.raises(ValueError, match="rigid"):
        rigid_transform(invalid)
    with pytest.raises(ValueError, match="unit norm"):
        pose_matrix((0, 0, 0, 0, 0, 0, 2))


def test_measured_rotation_displacement_uses_shortest_branch():
    first = (0, 0, 0, 0, 0, 0, 1)
    last = (0.02, 0, 0, 0, 0, -np.sin(0.1), -np.cos(0.1))
    np.testing.assert_allclose(pose_displacement(first, last), (0.02, 0, 0, 0, 0, 0.2), atol=1e-10)


def test_v4_roundtrip_success_timeout_and_masked_training(tmp_path):
    output = writer(tmp_path)
    output.write(aligned_episode("timeout"))
    output.write(aligned_episode("success", terminal=True))
    report = validate_hdf5_dataset(tmp_path)
    assert report.passed, report.errors
    loaded = output.load_episodes()
    assert all(not step.reward_valid for episode in loaded for step in episode.steps)
    assert loaded[0].steps[0].action.action_applied is not None
    assert "T_w_b" in loaded[0].observations[0].value
    batch = R2DreamerBatch.from_episodes(loaded).as_mapping()
    assert not np.any(batch["reward_valid"])
    with MambaSequenceBuffer(tmp_path, horizon=2) as buffer:
        data = buffer.sample(2)
        assert data["reward_valid"].shape == (2, 2, 1)
        assert (
            compute_reward_prediction_loss(np.ones((2, 2, 1)), data["rewards"], reward_valid=data["reward_valid"]) == 0
        )
    with h5py.File(tmp_path / "shard-00000.hdf5") as handle:
        group = handle["data/demo_timeout"]
        assert group["transforms/T_w_b"].shape == (4, 4)
        assert group["actions"].id == group["action_commanded"].id
        np.testing.assert_allclose(group["diagnostics/tcp_displacement"][:, 0], 0.01, atol=1e-7)
        assert not group.attrs["object_state_is_dynamic"]


@pytest.mark.parametrize("field", ["gripper_force", "ee_twist_w", "grasp_pose_o", "T_w_b"])
def test_v4_missing_sensor_rejected_before_partial_write(tmp_path, field):
    episode = aligned_episode()
    frames = tuple(replace(f, value={k: v for k, v in f.value.items() if k != field}) for f in episode.observations)
    with pytest.raises(ValueError, match="missing"):
        writer(tmp_path).write(replace(episode, observations=frames))
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize(
    "path,value,expected",
    [
        ("reward_valid", True, "reward_valid=false"),
        ("obs/tcp_pose_w", 0.0, "unit norm"),
        ("diagnostics/tcp_displacement", 9.0, "adjacent observations"),
        ("action_applied", np.nan, "finite"),
    ],
)
def test_corruption_reports_contract_error(tmp_path, path, value, expected):
    writer(tmp_path).write(aligned_episode())
    with h5py.File(tmp_path / "shard-00000.hdf5", "r+") as handle:
        group = handle["data/demo_aligned"]
        group[path][...] = value
        assert any(expected in e for e in validate_episode_group(group))


def test_masked_reward_uses_only_valid_targets():
    assert compute_reward_prediction_loss([0.0, 10.0], [1.0, 0.0], reward_valid=[True, False]) == 0.5
    assert (
        compute_reward_prediction_loss(
            [0.0, 10.0], [1.0, 0.0], RewardLossConfig(positive_weight=3), reward_valid=[True, False]
        )
        == 0.5
    )
    with pytest.raises(ValueError, match="boolean"):
        compute_reward_prediction_loss([0.0], [1.0], reward_valid=[1])


def test_com_velocity_is_shifted_to_tcp_not_flange_origin():
    data = SimpleNamespace(
        root_pos_w=np.zeros((1, 3)),
        root_quat_w=np.array([[1, 0, 0, 0]]),
        body_com_pos_w=np.array([[[0.5, 0, 0]]]),
        body_com_vel_w=np.array([[[0, 0, 0, 0, 0, 2.0]]]),
        applied_torque=np.array([[3.0, -4.0]]),
    )
    state = SimpleNamespace(position_e=(0, 0, 0), rotation_wxyz=(1, 0, 0, 0))
    value = aligned_snapshot(
        SimpleNamespace(
            data=data, root_physx_view=SimpleNamespace(get_dof_projected_joint_forces=lambda: np.array([[3.0, -4.0]]))
        ),
        SimpleNamespace(flange_body_id=0),
        SimpleNamespace(joint_ids=[0, 1]),
        env_origin=(0, 0, 0),
        source_state=state,
        goal_state=state,
        tcp_position_b=(1, 0, 0),
        tcp_quaternion_b=(1, 0, 0, 0),
        grasp_position_b=(1, 0, 0),
        grasp_quaternion_b=(1, 0, 0, 0),
    )
    np.testing.assert_allclose(value["ee_twist_w"], (0, 1, 0, 0, 0, 2))
    np.testing.assert_allclose(value["gripper_force"], (3, -4))


def test_v4_camera_xyzw_roundtrip_and_nested_calibration(tmp_path):
    episode = aligned_episode()
    frames = []
    for frame in episode.observations:
        fields = dict(frame.value)
        fields.update(
            front_intrinsic_matrix=np.eye(3),
            front_position_w=(0, 0, 0),
            front_quat_w_ros=(0.5, 0.5, -0.5, 0.5),
            front_quat_w_opengl=(1.0, 0.0, 0.0, 0.0),
        )
        frames.append(replace(frame, value=fields))
    output = writer(tmp_path)
    output.write(replace(episode, observations=tuple(frames)))
    with h5py.File(tmp_path / "shard-00000.hdf5") as handle:
        group = handle["data/demo_aligned"]
        assert not validate_episode_group(group)
        np.testing.assert_allclose(group["obs/front_quat_w_ros"][0], (0.5, -0.5, 0.5, 0.5))
        assert group["obs/camera_calibration"].id == group["camera_calibration"].id
    loaded = output.load_episodes()[0]
    assert loaded.schema_version == ALIGNED_SCHEMA_VERSION
    np.testing.assert_allclose(loaded.observations[0].value["front_quat_w_ros"], (0.5, 0.5, -0.5, 0.5))
    with MambaSequenceBuffer(tmp_path, horizon=2) as buffer:
        assert buffer.sample()["observations"]["front_quat_w_ros"].shape == (1, 3, 4)


def test_moving_base_transform_axis(tmp_path):
    episode = aligned_episode()
    frames = []
    for i, frame in enumerate(episode.observations):
        value = dict(frame.value)
        transform = np.array(value["T_w_b"], copy=True)
        transform[0, 3] += i * 0.1
        value["T_w_b"] = transform
        for stem in ("tcp_pose", "grasp_pose", "goal_object_pose"):
            value[stem + "_w"] = transform_pose(transform, value[stem + "_b"])
        value["grasp_in_ee_pose_w"] = relative_pose_in_axes(value["tcp_pose_w"], value["grasp_pose_w"])
        frames.append(replace(frame, value=value))
    writer(tmp_path).write(replace(episode, observations=tuple(frames)))
    with h5py.File(tmp_path / "shard-00000.hdf5") as handle:
        group = handle["data/demo_aligned"]
        assert group["transforms/T_w_b"].shape == (3, 4, 4)
        assert not validate_episode_group(group)


def test_training_loader_rejects_missing_v4_reward_mask(tmp_path):
    writer(tmp_path).write(aligned_episode())
    with h5py.File(tmp_path / "shard-00000.hdf5", "r+") as handle:
        del handle["data/demo_aligned/reward_valid"]
    with pytest.raises(ValueError, match="reward_valid"):
        MambaSequenceBuffer(tmp_path, horizon=2)


def test_legacy_reader_keeps_historical_reward_semantics(tmp_path):
    output = HDF5TrajectoryWriter(tmp_path, DatasetConfig(min_free_disk_bytes=0))
    output.write(_episode("legacy", steps=2))
    assert all(step.reward_valid for step in output.load_episodes()[0].steps)
    with MambaSequenceBuffer(tmp_path, horizon=2) as buffer:
        assert np.all(buffer.sample()["reward_valid"])


def test_v4_cannot_silently_downgrade_to_v3(tmp_path):
    output = writer(tmp_path / "v4")
    output.write(aligned_episode())
    legacy = HDF5TrajectoryWriter(tmp_path / "v3", DatasetConfig(min_free_disk_bytes=0))
    with pytest.raises(ValueError, match="downgrade"):
        legacy.write(output.load_episodes()[0])


def test_applied_rotation_rejected_before_partial_write(tmp_path):
    episode = aligned_episode()
    action = list(episode.steps[0].action.action_applied)
    action[3] = 0.1
    steps = (
        replace(episode.steps[0], action=replace(episode.steps[0].action, action_applied=action)),
        *episode.steps[1:],
    )
    with pytest.raises(ValueError, match="action_applied requires zero"):
        writer(tmp_path).write(replace(episode, steps=steps))
    assert not list(tmp_path.glob("*.partial"))
    assert not list(tmp_path.glob("*.hdf5"))


@pytest.mark.parametrize("name", ["action_commanded", "action_applied"])
def test_detached_diagnostic_action_must_match_canonical(tmp_path, name):
    writer(tmp_path).write(aligned_episode())
    with h5py.File(tmp_path / "shard-00000.hdf5", "r+") as handle:
        group = handle["data/demo_aligned"]
        values = group[name][...]
        values[0, 0] += 0.1
        del group["diagnostics"][name]
        group["diagnostics"].create_dataset(name, data=values)
        assert any(f"diagnostics/{name} disagrees" in error for error in validate_episode_group(group))


@pytest.mark.parametrize("key,value", [("quaternion_order", "wxyz"), ("reward_valid", True), ("force_unit", "unknown")])
def test_file_metadata_corruption_rejected(tmp_path, key, value):
    writer(tmp_path).write(aligned_episode())
    with h5py.File(tmp_path / "shard-00000.hdf5", "r+") as handle:
        handle["data"].attrs[key] = value
        assert any(f"file metadata {key}" in error for error in validate_episode_group(handle["data/demo_aligned"]))


def test_persisted_applied_rotation_rejected(tmp_path):
    writer(tmp_path).write(aligned_episode())
    with h5py.File(tmp_path / "shard-00000.hdf5", "r+") as handle:
        group = handle["data/demo_aligned"]
        group["action_applied"][0, 3] = 0.1
        assert any("action_applied requires zero" in error for error in validate_episode_group(group))


def test_applied_translation_can_exceed_normalized_command_range(tmp_path):
    episode = aligned_episode()
    action = list(episode.steps[0].action.action_applied)
    action[0] = 1.5
    steps = (
        replace(episode.steps[0], action=replace(episode.steps[0].action, action_applied=action)),
        *episode.steps[1:],
    )
    writer(tmp_path).write(replace(episode, steps=steps))
    report = validate_hdf5_dataset(tmp_path)
    assert report.passed, report.errors


@pytest.mark.parametrize("path,shape", [
    ("rewards", ()),
    ("rewards", (2, 1)),
    ("timeouts", (0,)),
    ("obs/ee_twist_b", ()),
    ("diagnostics/gripper_command", ()),
])
def test_malformed_axes_return_errors_without_crashing(tmp_path, path, shape):
    writer(tmp_path).write(aligned_episode())
    with h5py.File(tmp_path / "shard-00000.hdf5", "r+") as handle:
        group = handle["data/demo_aligned"]
        del group[path]
        group.create_dataset(path, data=np.zeros(shape))
        assert validate_episode_group(group)


@pytest.mark.parametrize("path", ["actions", "obs", "diagnostics"])
def test_wrong_hdf5_node_type_returns_errors(tmp_path, path):
    writer(tmp_path).write(aligned_episode())
    with h5py.File(tmp_path / "shard-00000.hdf5", "r+") as handle:
        group = handle["data/demo_aligned"]
        del group[path]
        if path == "actions":
            group.create_group(path)
        else:
            group.create_dataset(path, data=np.zeros(2))
        assert validate_episode_group(group)


@pytest.mark.parametrize("path,key,value", [
    ("obs/tcp_pose_b", "unit", "millimeter"),
    ("obs/grasp_pose_o", "axis_order", "x,y,z,qw,qx,qy,qz"),
    ("obs/ee_twist_w", "reference_point", "center_of_mass"),
    ("obs/ee_twist_b", "frame", "world"),
    ("obs/gripper_force", "source", "estimated_torque"),
    ("obs/object_position_b", "is_dynamic", True),
    ("action_commanded", "axis", "T+1"),
    ("transforms/T_w_b", "convention", "source_from_target"),
])
def test_sensor_semantic_metadata_corruption_rejected(tmp_path, path, key, value):
    writer(tmp_path).write(aligned_episode())
    with h5py.File(tmp_path / "shard-00000.hdf5", "r+") as handle:
        group = handle["data/demo_aligned"]
        group[path].attrs[key] = value
        assert any(f"{path} metadata {key}" in error for error in validate_episode_group(group))


@pytest.mark.parametrize("shape,dtype", [((2, 1), bool), ((2,), float), ((1,), bool)])
def test_both_training_readers_reject_malformed_reward_masks(tmp_path, shape, dtype):
    from DataPipe.hdf5_schema_reader import episode_from_hdf5_group

    writer(tmp_path).write(aligned_episode())
    with h5py.File(tmp_path / "shard-00000.hdf5", "r+") as handle:
        group = handle["data/demo_aligned"]
        del group["reward_valid"]
        group.create_dataset("reward_valid", data=np.zeros(shape, dtype=dtype))
        with pytest.raises(ValueError, match="reward_valid"):
            episode_from_hdf5_group(group)
    with pytest.raises(ValueError, match="reward_valid"):
        MambaSequenceBuffer(tmp_path, horizon=2)
