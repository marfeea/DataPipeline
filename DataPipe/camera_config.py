"""从项目三路视觉诊断配置生成数据管线 CameraSensor 契约。"""

from __future__ import annotations

from typing import Protocol

from .config import MIN_FREE_DISK_BYTES, CameraSensorSpec, DatasetConfig


class DiagnosticCameraPoseConfig(Protocol):
    """项目诊断相机位姿所需的最小接口。"""

    scene_name: str
    prim_name: str
    position_e: tuple[float, float, float]
    rotation_wxyz: tuple[float, float, float, float]
    pose_convention: str
    source_reference: str


def camera_sensor_spec_from_diagnostic_pose(
    pose_cfg: DiagnosticCameraPoseConfig,
    *,
    resolution: tuple[int, int],
    data_types: tuple[str, ...] = ("rgb", "distance_to_image_plane"),
) -> CameraSensorSpec:
    """将一条项目诊断相机位姿转换为数据管线静态契约。"""

    return CameraSensorSpec(
        name=str(pose_cfg.scene_name),
        prim_path=f"{{ENV_REGEX_NS}}/{pose_cfg.prim_name}",
        resolution=resolution,
        data_types=data_types,
        position_e=tuple(float(value) for value in pose_cfg.position_e),
        rotation_wxyz=tuple(float(value) for value in pose_cfg.rotation_wxyz),
        pose_convention=str(pose_cfg.pose_convention),
        source_reference=f"configs.vision_diagnostics: {pose_cfg.source_reference}",
    )


def project_diagnostic_camera_sensor_specs(
    *,
    resolution: tuple[int, int] | None = None,
    data_types: tuple[str, ...] = ("rgb", "distance_to_image_plane"),
) -> tuple[CameraSensorSpec, ...]:
    """返回项目既有前视、侧视和俯视三路数据相机配置。"""

    from configs.vision_diagnostics import DIAGNOSTIC_CAMERA_MODEL, DIAGNOSTIC_CAMERA_POSES

    resolved_resolution = resolution or (DIAGNOSTIC_CAMERA_MODEL.width, DIAGNOSTIC_CAMERA_MODEL.height)
    specs = tuple(
        camera_sensor_spec_from_diagnostic_pose(
            pose_cfg,
            resolution=resolved_resolution,
            data_types=data_types,
        )
        for pose_cfg in DIAGNOSTIC_CAMERA_POSES
    )
    if len(specs) != 3:
        raise ValueError(f"项目数据相机配置必须恰好包含三路视角，实际为 {len(specs)}")
    return specs


def dataset_config_with_project_diagnostic_cameras(
    *,
    episodes_per_shard: int = 10,
    min_free_disk_bytes: int = MIN_FREE_DISK_BYTES,
    resolution: tuple[int, int] | None = None,
    data_types: tuple[str, ...] = ("rgb", "distance_to_image_plane"),
) -> DatasetConfig:
    """返回已绑定项目三路诊断相机的 DatasetConfig。"""

    return DatasetConfig(
        episodes_per_shard=episodes_per_shard,
        min_free_disk_bytes=min_free_disk_bytes,
        cameras=project_diagnostic_camera_sensor_specs(resolution=resolution, data_types=data_types),
    )


__all__ = [
    "camera_sensor_spec_from_diagnostic_pose",
    "dataset_config_with_project_diagnostic_cameras",
    "project_diagnostic_camera_sensor_specs",
]
