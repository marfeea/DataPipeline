"""Sparse camera export must use post-control-step indices and all three views."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("control_steps, expected_steps", [(11, []), (12, [12]), (25, [12, 24])])
def test_export_step_boundaries_and_three_camera_payloads(tmp_path, monkeypatch, control_steps, expected_steps):
    source = Path(__file__).resolve().parents[1] / "DataPipe" / "collection_cameras.py"
    spec = importlib.util.spec_from_file_location("camera_export_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "RESULT_ROOT", tmp_path)
    cameras = [SimpleNamespace(name=name, prim_path=f"/test/{name}", source_reference="test")
               for name in module.CAMERA_NAMES]
    observations = []
    for step in range(control_steps + 1):
        value = {"semantic_stage_id": 3}
        for camera_index, camera in enumerate(cameras):
            name = camera.name
            value[f"image_{name}"] = np.full((2, 3, 4), step + camera_index, dtype=np.uint8)
            value[f"depth_{name}"] = np.full((2, 3, 1), step / 10, dtype=np.float32)
            value[f"{name}_position_w"] = np.array([step, camera_index, 0.0])
            value[f"{name}_quat_w_ros"] = np.array([1.0, 0.0, 0.0, 0.0])
            value[f"{name}_quat_w_opengl"] = np.array([1.0, 0.0, 0.0, 0.0])
            value[f"{name}_intrinsic_matrix"] = np.eye(3)
        observations.append(SimpleNamespace(value=value))
    steps = [SimpleNamespace(action=SimpleNamespace(as_dict=lambda i=i: {"waypoint_index": i}),
                             reward=float(i), reward_components={}) for i in range(control_steps)]
    episode = SimpleNamespace(episode_id="test_episode", observations=observations, steps=steps,
                              length=control_steps, termination_reason=SimpleNamespace(value="success"))
    assert module.export_episode_frames(episode, cameras, 12.0, hdf5_path=tmp_path / "batch/shard-00000.hdf5") == len(expected_steps) * 3
    result = tmp_path / "batch/shard-00000" / episode.episode_id
    metadata = json.loads((result / "frames.json").read_text())
    assert [(frame["control_step"], frame["camera"]) for frame in metadata["frames"]] == [
        (step, name) for step in expected_steps for name in module.CAMERA_NAMES
    ]
    from PIL import Image

    for frame in metadata["frames"]:
        step = frame["control_step"]
        camera_index = module.CAMERA_NAMES.index(frame["camera"])
        with Image.open(result / frame["rgb"]) as image:
            assert image.mode == "RGB"
            assert np.all(np.asarray(image) == step + camera_index)
        assert "depth" not in frame
        assert frame["time_since_reset_s"] == step / 12
        assert frame["camera_position_w"] == [step, camera_index, 0.0]

    assert not list(result.rglob("*.npy"))
    assert len(list(result.rglob("*.png"))) == len(expected_steps) * 3
    assert [item["control_step"] for item in metadata["controls"]] == expected_steps
    for item in metadata["controls"]:
        assert item["action_index"] == item["observation_index"] - 1
        assert item["control"]["waypoint_index"] == item["action_index"]
        assert item["reward"] == item["action_index"]


def test_writer_full_rate_images_and_resumed_shard_mapping(tmp_path, monkeypatch):
    import h5py
    from dataclasses import replace
    from DataPipe import DatasetConfig
    from DataPipe.config import CameraSensorSpec
    from DataPipe import collection_cameras as module
    from test_datapipe_p0_r import _episode

    monkeypatch.setattr(module, "RESULT_ROOT", tmp_path / "result")
    cameras = tuple(CameraSensorSpec(name=name, prim_path=f"/test/{name}", resolution=(4, 4), data_types=("rgb", "distance_to_image_plane"))
                    for name in module.CAMERA_NAMES)
    config = DatasetConfig(cameras=cameras, episodes_per_shard=2, min_free_disk_bytes=0)
    output = tmp_path / "data/batch"
    for index in range(3):
        # Reopen each time to verify resumed appends and shard rollover.
        writer = module.CameraExportingTrajectoryWriter(output, config)
        episode = _episode(f"ep{index}", steps=25)
        observations = []
        for step, frame in enumerate(episode.observations):
            value = dict(frame.value)
            for name in module.CAMERA_NAMES:
                value[f"image_{name}"] = value["image_front"].copy()
                value[f"depth_{name}"] = np.full((4, 4, 1), step, dtype=np.float32)
                value[f"{name}_position_w"] = np.array([step, 0, 0])
                value[f"{name}_quat_w_ros"] = np.array([1, 0, 0, 0])
                value[f"{name}_quat_w_opengl"] = np.array([1, 0, 0, 0])
                value[f"{name}_intrinsic_matrix"] = np.eye(3)
            observations.append(replace(frame, value=value))
        episode = replace(episode, observations=tuple(observations))
        shard = writer.write(episode)
        assert shard == f"shard-{index // 2:05d}.hdf5"
        result = tmp_path / "result/batch" / Path(shard).stem / episode.episode_id
        metadata = json.loads((result / "frames.json").read_text())
        assert metadata["hdf5_path"] == str((output / shard).resolve())
        assert len(list(result.rglob("*.png"))) == 6
        assert not list(result.rglob("*.npy"))
        with h5py.File(output / shard) as handle:
            group = handle[metadata["hdf5_group"]]
            assert group["actions"].shape[0] == 25
            for name in module.CAMERA_NAMES:
                assert group[f"obs/image_{name}"].shape[0] == 26
                assert group[f"obs/depth_{name}"].shape[0] == 26
                np.testing.assert_array_equal(group[f"obs/image_{name}"][:, 0, 0, 0], np.arange(26))
                np.testing.assert_array_equal(group[f"obs/depth_{name}"][:, 0, 0, 0], np.arange(26))
                calibration = group["camera_calibration"][name]
                assert calibration["intrinsic_matrix"].shape == (26, 3, 3)
                assert calibration["extrinsic_position_w"].shape == (26, 3)
                assert calibration["extrinsic_quaternion_w_ros_wxyz"].shape == (26, 4)
                assert calibration["extrinsic_quaternion_w_opengl_wxyz"].shape == (26, 4)
                np.testing.assert_array_equal(
                    calibration["intrinsic_matrix"][:], group[f"obs/{name}_intrinsic_matrix"][:]
                )
            for item in metadata["controls"]:
                np.testing.assert_allclose(item["control"]["action"], group["actions"][item["action_index"]])
    assert len(list((tmp_path / "result/batch/shard-00000").iterdir())) == 2
