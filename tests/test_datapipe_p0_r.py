from __future__ import annotations

import json
import sys
from dataclasses import replace
from math import pi
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from DataPipe import (  # noqa: E402
    ACTION_DIMENSION,
    ARM_ACTION_DIMENSION,
    ActionRecord,
    DatasetConfig,
    EpisodeBuilder,
    HDF5TrajectoryWriter,
    MambaSequenceBuffer,
    TerminationReason,
    normalized_action_to_tcp_delta,
    tcp_delta_to_normalized_action,
)
from DataPipe.config import DATASET_SCHEMA_VERSION  # noqa: E402
from DataPipe.hdf5_schema import _as_array, validate_episode_group, wxyz_to_xyzw, xyzw_to_wxyz  # noqa: E402
from DataPipe.mamba_sequence_buffer import window_count  # noqa: E402
from DataPipe.validator import validate_hdf5_dataset  # noqa: E402


def _action() -> ActionRecord:
    return ActionRecord(
        raw_action=(0.25,) * ACTION_DIMENSION,
        action=(0.25,) * ACTION_DIMENSION,
        tcp_displacement=normalized_action_to_tcp_delta((0.25,) * ACTION_DIMENSION)[:ARM_ACTION_DIMENSION],
        ik_output=(0.1,) * ARM_ACTION_DIMENSION,
        joint_position=(0.2,) * ARM_ACTION_DIMENSION,
        joint_velocity=(0.3,) * ARM_ACTION_DIMENSION,
        tcp_heading_target_rad=0.25 * pi,
    )


def _observation(step: int) -> dict[str, object]:
    return {
        "image_front": np.full((4, 4, 3), step % 255, dtype=np.uint8),
        "q": np.full((6,), 0.1 * step, dtype=np.float32),
        "dq": np.full((6,), 0.2 * step, dtype=np.float32),
        "tcp_pose_wxyz": (1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0),
        "target_pose_wxyz": (4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0),
        "gripper_position": (0.0115, -0.0115),
        "gripper_velocity": (0.0, 0.0),
        "semantic_stage_id": 0,
        "waypoint_index": step,
        "preposition_position_b": (4.0, 5.0, 6.0),
        "object_position_b": (4.1, 5.0, 6.0),
        "desired_approach_axis_b": (1.0, 0.0, 0.0),
        "advance_target_position_b": (4.05, 5.0, 6.0),
    }


def _episode(episode_id: str, steps: int = 16) -> object:
    builder = EpisodeBuilder(episode_id)
    builder.start(_observation(0))
    for step in range(steps):
        builder.append(
            _action(),
            0.5,
            _observation(step + 1),
            {"position_reward": 0.25, "success_terminal_reward": 0.0},
        )
    return builder.close(TerminationReason.TIMEOUT, _observation(steps))


def test_p0_r4_action_scales_round_trip() -> None:
    normalized = (0.25, -0.5, 1.0, -1.0, 0.0, 0.75, -0.25, -1.0)
    assert tcp_delta_to_normalized_action(normalized_action_to_tcp_delta(normalized)) == normalized


def test_p0_r6_quaternion_disk_order_round_trip() -> None:
    internal = (1.0, 2.0, 3.0, 0.7, 0.1, 0.2, 0.3)
    assert xyzw_to_wxyz(wxyz_to_xyzw(internal)) == internal


def test_depth_observation_preserves_positive_infinity_for_ray_misses() -> None:
    depth = np.asarray([0.25, np.inf], dtype=np.float32)

    np.testing.assert_array_equal(_as_array(depth, name="depth_HandCamera"), depth)


@pytest.mark.parametrize("invalid", [np.nan, -np.inf])
def test_depth_observation_rejects_other_nonfinite_values(invalid: float) -> None:
    with pytest.raises(ValueError, match="NaN or negative Inf"):
        _as_array(np.asarray([0.25, invalid], dtype=np.float32), name="depth_HandCamera")


def test_non_depth_observation_still_rejects_positive_infinity() -> None:
    with pytest.raises(ValueError, match="contains NaN or Inf"):
        _as_array(np.asarray([0.25, np.inf], dtype=np.float32), name="q")


def test_sequence_v3_rejects_missing_required_observation_fields(tmp_path: Path) -> None:
    builder = EpisodeBuilder("missing-gripper-state")
    incomplete = {"q": [0.0] * 6, "dq": [0.0] * 6}
    builder.start(incomplete)
    builder.append(_action(), 0.0, incomplete)
    episode = builder.close(TerminationReason.TIMEOUT, incomplete)
    writer = HDF5TrajectoryWriter(tmp_path, DatasetConfig(min_free_disk_bytes=0))
    with pytest.raises(KeyError, match="required observation fields"):
        writer.write(episode)
    assert not list(tmp_path.glob("*.partial"))


def test_p0_r1_hdf5_schema_writer_and_resume(tmp_path: Path) -> None:
    import h5py

    writer = HDF5TrajectoryWriter(tmp_path, DatasetConfig(episodes_per_shard=2, min_free_disk_bytes=0))
    writer.write(_episode("e0"), metadata={"split": "train", "seed": 7})
    writer.write(_episode("e1"), metadata={"split": "validation", "seed": 8})
    writer.write(_episode("e2"), metadata={"split": "test", "seed": 9})

    assert not list(tmp_path.glob("*.partial"))
    report = validate_hdf5_dataset(tmp_path)
    assert report.passed, report.errors
    with h5py.File(tmp_path / "shard-00000.hdf5", "r") as handle:
        demo = handle["data"]["demo_e0"]
        assert demo.attrs["seed"] == 7
        assert demo["actions"].shape == (16, 8)
        assert demo["obs"]["q"].shape == (17, 6)
        assert demo["obs"]["image_front"].compression == "gzip"
        assert demo["diagnostics"]["reward_components"]["position_reward"].shape == (16,)
        assert demo["diagnostics"]["waypoint_tcp_position_b"].shape == (16, 3)
        assert demo["diagnostics"]["ik_seed_joint_position"].shape == (16, 6)
        assert demo["diagnostics"]["ik_solution_joint_position"].shape == (16, 6)
        assert demo["diagnostics"]["limited_joint_target"].shape == (16, 6)
        assert demo["diagnostics"]["commanded_joint_target"].shape == (16, 6)
        assert demo["diagnostics"]["actual_tcp_position_b"].shape == (16, 3)
        assert demo["diagnostics"]["ik_success"].shape == (16,)
        assert demo["diagnostics"]["delta_limited"].shape == (16,)
        assert demo["phases"].shape == (16,)
        assert demo["waypoint_indices"].shape == (16,)
        assert demo["diagnostics"]["gripper_joint_position"].shape == (16, 2)
        assert tuple(demo["obs"]["tcp_pose"][0]) == wxyz_to_xyzw(_observation(0)["tcp_pose_wxyz"])
        assert not validate_episode_group(demo)
    assert writer.load_episodes()[0].steps[0].reward_components["position_reward"] == 0.25


def test_hdf5_writer_reports_incompatible_manifest_schema(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": "r2dreamer-aubo-ik-sequence-v2"}),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="incompatible dataset schema.*sequence-v2.*sequence-v3.*new empty output directory",
    ):
        HDF5TrajectoryWriter(tmp_path, DatasetConfig(min_free_disk_bytes=0))

    assert DATASET_SCHEMA_VERSION == "r2dreamer-aubo-ik-sequence-v3"


@pytest.mark.parametrize(
    "stages",
    (
        (0, 1, 0, 2, 3, 4, 6, 7, 11, 10, 12, 8, 9),
        (0, 1, 0, 2, 3, 4, 6, 11, 7, 10, 12, 8, 9),
    ),
    ids=("current-move-before-align", "legacy-align-before-move"),
)
def test_sequence_v3_validator_uses_semantic_stage_order_not_numeric_order(
    tmp_path: Path,
    stages: tuple[int, ...],
) -> None:
    import h5py

    builder = EpisodeBuilder("semantic-stage-order")
    builder.start(_observation(0))
    for step, stage in enumerate(stages):
        builder.append(
            replace(_action(), semantic_stage_id=stage),
            0.0,
            _observation(step + 1),
        )
    writer = HDF5TrajectoryWriter(tmp_path, DatasetConfig(min_free_disk_bytes=0))
    writer.write(builder.close(TerminationReason.SUCCESS, _observation(len(stages))))

    with h5py.File(tmp_path / "shard-00000.hdf5", "r") as handle:
        assert not validate_episode_group(handle["data"]["demo_semantic-stage-order"])


def test_sequence_v3_validator_rejects_stage_backtracking_and_heading_mismatch(tmp_path: Path) -> None:
    import h5py

    writer = HDF5TrajectoryWriter(tmp_path, DatasetConfig(min_free_disk_bytes=0))
    writer.write(_episode("invalid-contract", steps=4))
    path = tmp_path / "shard-00000.hdf5"
    with h5py.File(path, "r+") as handle:
        demo = handle["data"]["demo_invalid-contract"]
        demo["phases"][:] = np.asarray((2, 3, 2, 3), dtype=np.uint8)
        demo["diagnostics"]["tcp_heading_target_rad"][:] = 0.0
        errors = validate_episode_group(demo)

    assert "phases must follow the semantic sequence order after path stages 0/1" in errors
    assert "actions TCP heading component must match diagnostics/tcp_heading_target_rad" in errors


def test_p0_r9_r10_fixed_window_boundary(tmp_path: Path) -> None:
    writer = HDF5TrajectoryWriter(tmp_path, DatasetConfig(min_free_disk_bytes=0))
    writer.write(_episode("e16", 16))
    writer.write(_episode("e17", 17))
    assert window_count(15) == 0
    assert window_count(16) == 1
    assert window_count(17) == 2
    with MambaSequenceBuffer(tmp_path, horizon=16) as buffer:
        assert len(buffer.window_indices) == 3
        batch = buffer.sample(batch_size=2)
        assert batch["actions"].shape == (2, 16, 8)
        assert batch["observations"]["q"].shape == (2, 17, 6)
        assert batch["is_terminal"].shape == (2, 16, 1)
