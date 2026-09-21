"""Collect HandCamera, CameraSensor_2 and CameraSensor_3 with sparse exports.

Both collector entry points use this shared visual pipeline. HandCamera must already exist
in the robot USD; its authored transform and camera optics are preserved.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from DataPipe.hdf5_writer import HDF5TrajectoryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "test" / "result"
FRAME_INTERVAL = 12
CAMERA_NAMES = ("HandCamera", "CameraSensor_2", "CameraSensor_3")
# Both robots reference the same USD. Bind the camera only under the first robot.
HAND_CAMERA_ROBOT_PRIM_PATH = "{ENV_REGEX_NS}/AUBObot"


def configure_collection_cameras(cfg: Any, camera_specs: tuple[Any, ...]) -> tuple[Any, tuple[Any, ...]]:
    """Replace the front camera with the existing hand camera, without respawning it."""

    from pxr import Gf, Usd, UsdGeom
    from isaaclab.sensors.camera import CameraCfg

    from DataPipe.config import CameraSensorSpec

    robot_asset = cfg.scene.AUBObot_usd
    if robot_asset.prim_path != HAND_CAMERA_ROBOT_PRIM_PATH:
        raise ValueError(f"HandCamera must belong to {HAND_CAMERA_ROBOT_PRIM_PATH}, got {robot_asset.prim_path}")
    source_path = str(robot_asset.spawn.usd_path)
    stage = Usd.Stage.Open(source_path)
    root = stage.GetDefaultPrim()
    if not root:
        raise RuntimeError(f"Robot USD has no default prim: {source_path}")
    matches = [prim for prim in Usd.PrimRange(root) if prim.GetName() == "HandCamera" and prim.IsA(UsdGeom.Camera)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one Camera prim named HandCamera in {source_path}; found {len(matches)}. "
            "Save the hand camera into the robot USD before collecting."
        )
    prim = matches[0]
    relative_path = prim.GetPath().MakeRelativePath(root.GetPath())
    prim_path = f"{HAND_CAMERA_ROBOT_PRIM_PATH}/{relative_path}"
    transform = Gf.Transform(UsdGeom.Xformable(prim).GetLocalTransformation())
    rotation = transform.GetRotation().GetQuat()
    hand_spec = CameraSensorSpec(
        name="HandCamera",
        prim_path=prim_path,
        resolution=camera_specs[0].resolution,
        data_types=("rgb", "distance_to_image_plane"),
        # This pose is relative to the camera's parent, as recorded in source_reference.
        position_e=tuple(transform.GetTranslation()),
        rotation_wxyz=(rotation.GetReal(), *rotation.GetImaginary()),
        pose_convention="opengl",
        source_reference=f"{source_path}:{prim.GetPath()}; pose relative to {prim.GetParent().GetPath()}",
    )
    width, height = hand_spec.resolution
    cfg.scene.camera_cfg = None
    cfg.scene.HandCamera = CameraCfg(
        prim_path=prim_path,
        spawn=None,
        width=width,
        height=height,
        data_types=list(hand_spec.data_types),
        update_period=0.0,
        update_latest_camera_pose=True,
    )
    specs = [hand_spec]
    for old_spec, name in zip(camera_specs[1:], CAMERA_NAMES[1:], strict=True):
        camera_cfg = getattr(cfg.scene, old_spec.name)
        camera_cfg.update_latest_camera_pose = True
        setattr(cfg.scene, old_spec.name, None)
        setattr(cfg.scene, name, camera_cfg)
        specs.append(replace(old_spec, name=name))
    return cfg, tuple(specs)


def add_camera_world_poses(value: dict[str, Any], unwrapped: Any, camera_names: Any) -> None:
    """Record current world extrinsics and intrinsics alongside the images."""

    for name in camera_names:
        data = unwrapped.scene[name].data
        for suffix, tensor in (
            ("position_w", data.pos_w),
            ("quat_w_ros", data.quat_w_ros),
            ("quat_w_opengl", data.quat_w_opengl),
            ("intrinsic_matrix", data.intrinsic_matrices),
        ):
            value[f"{name}_{suffix}"] = tensor[0].detach().cpu().numpy().astype(np.float64, copy=True)


def _as_rgb(frame: np.ndarray) -> np.ndarray:
    value = np.asarray(frame)
    if value.ndim != 3 or value.shape[-1] not in (3, 4):
        raise ValueError(f"unexpected camera RGB shape: {value.shape}")
    value = value[..., :3]
    if value.dtype != np.uint8:
        scale = 255.0 if np.issubdtype(value.dtype, np.floating) and value.max(initial=0.0) <= 1.0 else 1.0
        value = np.clip(value * scale, 0.0, 255.0).astype(np.uint8)
    return value


def export_episode_frames(
    episode: Any, camera_specs: Any, control_hz: float, *, hdf5_path: str | Path
) -> int:
    """Export post-step observations 12, 24, ... for all three cameras."""

    from PIL import Image

    hdf5_path = Path(hdf5_path).resolve()
    output_dir = RESULT_ROOT / hdf5_path.parent.name / hdf5_path.stem / episode.episode_id
    output_dir.mkdir(parents=True, exist_ok=False)
    camera_dirs = {"HandCamera": "handCamera", "CameraSensor_2": "camera2", "CameraSensor_3": "camera3"}
    for spec in camera_specs:
        (output_dir / camera_dirs.get(spec.name, spec.name)).mkdir()
    controls = []
    frames = []
    for control_step in range(FRAME_INTERVAL, len(episode.observations), FRAME_INTERVAL):
        observation = episode.observations[control_step].value
        # o_k is the result of a_(k-1), matching HDF5 obs[k] and actions[k-1].
        step = episode.steps[control_step - 1]
        controls.append({
            "control_step": control_step,
            "observation_index": control_step,
            "action_index": control_step - 1,
            "time_since_reset_s": control_step / control_hz,
            "control": step.action.as_dict(),
            "reward": step.reward,
            "reward_valid": getattr(step, "reward_valid", True),
            "reward_components": dict(step.reward_components),
        })
        for spec in camera_specs:
            stem = f"{spec.name}_control_step_{control_step:04d}"
            rgb_path = output_dir / camera_dirs.get(spec.name, spec.name) / f"{stem}_rgb.png"
            Image.fromarray(_as_rgb(observation[f"image_{spec.name}"])).save(rgb_path)
            frames.append({
                "camera": spec.name,
                "control_step": control_step,
                "time_since_reset_s": control_step / control_hz,
                "semantic_stage_id": int(observation["semantic_stage_id"]),
                "rgb": rgb_path.relative_to(output_dir).as_posix(),
                "camera_position_w": observation[f"{spec.name}_position_w"].tolist(),
                "camera_quaternion_w_ros_wxyz": observation[f"{spec.name}_quat_w_ros"].tolist(),
                "camera_quaternion_w_opengl_wxyz": observation[f"{spec.name}_quat_w_opengl"].tolist(),
                "intrinsic_matrix": observation[f"{spec.name}_intrinsic_matrix"].tolist(),
            })
    metadata = {
        "episode": episode.episode_id,
        "executed_control_steps": episode.length,
        "termination_reason": episode.termination_reason.value,
        "frame_interval_control_steps": FRAME_INTERVAL,
        "control_frequency_hz": control_hz,
        "hdf5_path": str(hdf5_path),
        "hdf5_group": f"/data/demo_{episode.episode_id}",
        "control_alignment": "obs[k] is the post-step image for actions[k-1]",
        "controls": controls,
        "cameras": [{"name": spec.name, "prim_path": spec.prim_path,
                     "source_reference": spec.source_reference} for spec in camera_specs],
        "frames": frames,
    }
    (output_dir / "frames.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return len(frames)


class CameraExportingTrajectoryWriter(HDF5TrajectoryWriter):
    """Persist every step to HDF5, then export sparse inspection data."""

    def write(self, episode: Any, *, metadata: Any = None) -> str:
        shard_path = super().write(episode, metadata=metadata)
        export_episode_frames(
            episode, self.config.cameras, self.config.control_frequency_hz,
            hdf5_path=self.output_dir / shard_path,
        )
        return shard_path
