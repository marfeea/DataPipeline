"""V2 replay captures the same sparse three-view RGB evidence as Legacy IK."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from DataPipe.collection_cameras import CAMERA_NAMES
from DataPipe.pipelines.curobo_bootstrap.replay_visualization import ReplayFrameRecorder


@pytest.mark.parametrize("render_calls, expected_frames", [(11, 0), (12, 3), (25, 6)])
def test_capture_uses_completed_render_count_and_three_views(tmp_path, render_calls, expected_frames):
    specs = tuple(SimpleNamespace(name=name, resolution=(3, 2), prim_path=f"/test/{name}", source_reference="test")
                  for name in CAMERA_NAMES)
    recorder = ReplayFrameRecorder(tmp_path / "frames", specs)
    cameras = {}
    for number, name in enumerate(CAMERA_NAMES):
        rgb = torch.full((1, 2, 3, 4), number + 10, dtype=torch.uint8)
        cameras[name] = SimpleNamespace(data=SimpleNamespace(
            output={"rgb": rgb},
            pos_w=torch.tensor([[number, 0.0, 0.0]]),
            quat_w_ros=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            quat_w_opengl=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            intrinsic_matrices=torch.eye(3).unsqueeze(0),
        ))
    for render_frame in range(1, render_calls + 1):
        recorder.capture(cameras, render_frame=render_frame, trajectory_index=render_frame - 1,
                         time_s=(render_frame - 1) / 60, event="NONE")
    summary = recorder.finish(source_frames=render_calls, render_calls=render_calls)
    manifest = json.loads((tmp_path / "frames/frames.json").read_text(encoding="utf-8"))
    assert summary["image_count"] == expected_frames
    assert [(frame["render_frame"], frame["trajectory_index"], frame["camera"]) for frame in manifest["frames"]] == [
        (render_frame, render_frame - 1, name)
        for render_frame in range(12, render_calls + 1, 12)
        for name in CAMERA_NAMES
    ]
    for frame in manifest["frames"]:
        with Image.open(tmp_path / "frames" / frame["rgb"]) as image:
            assert image.mode == "RGB"
            assert np.all(np.asarray(image) == CAMERA_NAMES.index(frame["camera"]) + 10)


def test_recorder_does_not_replace_previous_evidence(tmp_path):
    specs = tuple(SimpleNamespace(name=name) for name in CAMERA_NAMES)
    ReplayFrameRecorder(tmp_path / "frames", specs)
    with pytest.raises(FileExistsError):
        ReplayFrameRecorder(tmp_path / "frames", specs)
