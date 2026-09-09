"""测试脚本使用的视图相机配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .scene import WORKSTATION_POSE_E

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class TestRenderCfg:
    """相对于工作台基准定义的 Isaac Lab 视口相机。"""

    relative_workstation_offset: Vec3 = (-1.8, 0.2, 1.25)
    look_direction: Vec3 = (1.0, 0.0, 0.0)

    @property
    def viewport_camera_eye(self) -> Vec3:
        """返回环境坐标系中的相机位置。"""

        return tuple(
            origin + offset
            for origin, offset in zip(
                WORKSTATION_POSE_E.position,
                self.relative_workstation_offset,
                strict=True,
            )
        )

    @property
    def viewport_camera_target(self) -> Vec3:
        """返回环境坐标系中的相机朝向目标点。"""

        return tuple(
            eye + direction for eye, direction in zip(self.viewport_camera_eye, self.look_direction, strict=True)
        )


TEST_RENDER_CFG: Final = TestRenderCfg()
