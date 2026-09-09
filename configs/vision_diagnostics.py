"""视觉语义诊断相机的可复现位姿与成像配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .scene import WORKSTATION_POSE_E

Vec3 = tuple[float, float, float]
QuatWxyz = tuple[float, float, float, float]


@dataclass(frozen=True)
class DiagnosticCameraPoseCfg:
    """相对于工作站平移基准、但与环境轴对齐的相机位姿。"""

    scene_name: str
    prim_name: str
    workstation_offset_e: Vec3
    rotation_wxyz: QuatWxyz
    pose_convention: str
    source_reference: str

    @property
    def position_e(self) -> Vec3:
        """返回环境局部坐标系 ``E`` 下的相机位置。"""

        return tuple(
            origin + offset
            for origin, offset in zip(WORKSTATION_POSE_E.position, self.workstation_offset_e, strict=True)
        )


@dataclass(frozen=True)
class DiagnosticCameraModelCfg:
    """从 Test 项目迁移的针孔相机模型。"""

    width: int = 640
    height: int = 480
    focal_length_mm: float = 24.0
    focus_distance_mm: float = 400.0
    horizontal_aperture_mm: float = 20.955
    clipping_range_m: tuple[float, float] = (0.1, 1.0e5)


DIAGNOSTIC_CAMERA_MODEL: Final = DiagnosticCameraModelCfg()

DIAGNOSTIC_CAMERA_DATA_TYPES: Final = (
    "rgb",
    "distance_to_image_plane",
    "normals",
    "semantic_segmentation",
    "instance_segmentation_fast",
    "instance_id_segmentation_fast",
)

# Test 当前配置中的前、侧视角；俯视相机固定在工作站碰撞范围之外。
DIAGNOSTIC_CAMERA_POSES: Final = (
    DiagnosticCameraPoseCfg(
        scene_name="camera_cfg",
        prim_name="CameraSensor",
        workstation_offset_e=(1.4, 0.0, 1.3),
        rotation_wxyz=(0.5, 0.5, 0.5, 0.5),
        pose_convention="opengl",
        source_reference="Test current camera_cfg",
    ),
    DiagnosticCameraPoseCfg(
        scene_name="camera_cfg_2",
        prim_name="CameraSensor_2",
        workstation_offset_e=(0.0, -0.8, 2.0),
        rotation_wxyz=(0.86603, 0.5, 0.0, 0.0),
        pose_convention="opengl",
        source_reference="Test current camera_cfg_2",
    ),
    DiagnosticCameraPoseCfg(
        scene_name="camera_cfg_3",
        prim_name="CameraSensor_3",
        workstation_offset_e=(0.0, 0.0, 2.8),
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        pose_convention="opengl",
        source_reference="P0 external top-down diagnostic view",
    ),
)
