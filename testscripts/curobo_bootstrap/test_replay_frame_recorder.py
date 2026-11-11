import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from scripts.collection_cameras import CAMERA_NAMES
from scripts.pipelines.curobo_bootstrap.replay_visualization import ReplayFrameRecorder


class TensorView:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, index):
        return TensorView(self.value[index])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def camera_fixture():
    specs = [SimpleNamespace(name=n, resolution=(8, 8), prim_path=f"/scene/{n}", source_reference="test")
             for n in CAMERA_NAMES]
    cameras = {}
    for i, name in enumerate(CAMERA_NAMES):
        rgb = np.zeros((1, 8, 8, 3), dtype=np.uint8)
        rgb[0, :, :, 0] = 20 + i
        rgb[0, :, :, 1] = np.arange(8)
        data = SimpleNamespace(
            output={"rgb": TensorView(rgb)}, pos_w=TensorView([[0, 0, 1]]),
            quat_w_ros=TensorView([[1, 0, 0, 0]]), quat_w_opengl=TensorView([[1, 0, 0, 0]]),
            intrinsic_matrices=TensorView([np.eye(3)]),
        )
        cameras[name] = SimpleNamespace(data=data)
    return specs, cameras


def test_short_gripper_stage_is_saved_and_periodic_force_does_not_duplicate_frames(tmp_path):
    specs, cameras = camera_fixture()
    recorder = ReplayFrameRecorder(tmp_path / "images", specs)
    for frame in range(1, 14):
        recorder.capture(cameras, render_frame=frame, trajectory_index=frame - 1, time_s=(frame - 1) / 60,
                         event="CLOSE" if frame == 3 else "NONE", force=frame in {3, 12})
    result = recorder.finish(source_frames=13, render_calls=13)
    assert result["image_count"] == 6
    manifest = json.loads((tmp_path / "images/frames.json").read_text())
    assert manifest["forced_render_frames"] == [3]
    assert {f["render_frame"] for f in manifest["frames"]} == {3, 12}
    assert len({(f["render_frame"], f["camera"]) for f in manifest["frames"]}) == 6
    for i, name in enumerate(CAMERA_NAMES):
        short = next(f for f in manifest["frames"] if f["camera"] == name and f["render_frame"] == 3)
        assert short["event"] == "CLOSE" and short["trajectory_index"] == 2
        with Image.open(tmp_path / "images" / short["rgb"]) as picture:
            assert picture.getpixel((0, 0))[0] == 20 + i


def test_default_capture_keeps_historical_periodic_contract(tmp_path):
    specs, cameras = camera_fixture()
    recorder = ReplayFrameRecorder(tmp_path / "images", specs)
    for frame in range(1, 25):
        recorder.capture(cameras, render_frame=frame, trajectory_index=frame - 1,
                         time_s=(frame - 1) / 60, event="NONE")
    assert recorder.finish(source_frames=24, render_calls=24)["image_count"] == 6
    manifest = json.loads((tmp_path / "images/frames.json").read_text())
    assert "forced_render_frames" not in manifest
    assert {f["render_frame"] for f in manifest["frames"]} == {12, 24}
