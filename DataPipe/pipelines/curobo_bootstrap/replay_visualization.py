"""Sparse RGB evidence for the V2 Isaac replay, aligned to rendered frames."""

from __future__ import annotations

import json
from pathlib import Path

from DataPipe.collection_cameras import CAMERA_NAMES, FRAME_INTERVAL, _as_rgb

CAMERA_DIRS = {"HandCamera": "handCamera", "CameraSensor_2": "camera2", "CameraSensor_3": "camera3"}


def _numpy(value):
    return value.detach().cpu().numpy()


class ReplayFrameRecorder:
    """Save all three Legacy IK camera views after each twelfth render call."""

    def __init__(self, output_dir: Path, camera_specs):
        self.output_dir = Path(output_dir).resolve()
        self.camera_specs = tuple(camera_specs)
        if tuple(spec.name for spec in self.camera_specs) != CAMERA_NAMES:
            raise ValueError("replay cameras must match the Legacy IK three-view order")
        self.output_dir.mkdir(parents=True, exist_ok=False)
        for name in CAMERA_NAMES:
            (self.output_dir / CAMERA_DIRS[name]).mkdir()
        self.frames = []
        self.last_render_frame = 0

    def capture(self, cameras, *, render_frame: int, trajectory_index: int, time_s: float, event: str) -> int:
        if render_frame <= self.last_render_frame:
            raise ValueError("render frame numbers must increase")
        self.last_render_frame = render_frame
        if render_frame % FRAME_INTERVAL:
            return 0
        from PIL import Image

        for spec in self.camera_specs:
            camera = cameras[spec.name]
            output = camera.data.output
            if "rgb" not in output:
                raise RuntimeError(f"CameraSensor {spec.name} has no RGB output")
            rgb = _as_rgb(_numpy(output["rgb"][0]))
            if rgb.shape[:2] != (spec.resolution[1], spec.resolution[0]):
                raise RuntimeError(f"CameraSensor {spec.name} has unexpected RGB shape: {rgb.shape}")
            name = f"{spec.name}_render_frame_{render_frame:04d}_rgb.png"
            path = self.output_dir / CAMERA_DIRS[spec.name] / name
            Image.fromarray(rgb).save(path)
            data = camera.data
            self.frames.append({
                "camera": spec.name,
                "render_frame": render_frame,
                "trajectory_index": trajectory_index,
                "time_s": time_s,
                "event": event,
                "rgb": path.relative_to(self.output_dir).as_posix(),
                "camera_position_w": _numpy(data.pos_w[0]).tolist(),
                "camera_quaternion_w_ros_wxyz": _numpy(data.quat_w_ros[0]).tolist(),
                "camera_quaternion_w_opengl_wxyz": _numpy(data.quat_w_opengl[0]).tolist(),
                "intrinsic_matrix": _numpy(data.intrinsic_matrices[0]).tolist(),
            })
        return len(self.camera_specs)

    def finish(self, *, source_frames: int, render_calls: int) -> dict:
        expected = render_calls // FRAME_INTERVAL * len(self.camera_specs)
        if render_calls != source_frames or len(self.frames) != expected:
            raise RuntimeError("sparse replay capture does not match rendered frame count")
        manifest = {
            "schema": "curobo_pick_place_v2_replay_frames/1",
            "source_frames": source_frames,
            "render_calls": render_calls,
            "frame_interval_render_calls": FRAME_INTERVAL,
            "capture_alignment": "render_frame 12 is trajectory_index 11; capture occurs after render",
            "cameras": [
                {"name": spec.name, "prim_path": spec.prim_path, "source_reference": spec.source_reference}
                for spec in self.camera_specs
            ],
            "frames": self.frames,
        }
        (self.output_dir / "frames.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {"directory": str(self.output_dir), "manifest": "frames.json", "image_count": len(self.frames)}
