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
    episode = SimpleNamespace(episode_id="test_episode", observations=observations,
                              length=control_steps, termination_reason=SimpleNamespace(value="success"))
    assert module.export_episode_frames(episode, cameras, 12.0) == len(expected_steps) * 3
    result = tmp_path / episode.episode_id
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
        assert np.allclose(np.load(result / frame["depth"]), step / 10)
        assert frame["time_since_reset_s"] == step / 12
        assert frame["camera_position_w"] == [step, camera_index, 0.0]
