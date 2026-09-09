"""数据产线 P0 的版本化字段和单位契约。

本模块只描述数据格式，不导入 Isaac Lab、Torch 或现有任务运行时。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, radians
from typing import Final

ARM_ACTION_DIMENSION: Final = 6
TCP_HEADING_ACTION_DIMENSION: Final = 1
GRIPPER_ACTION_DIMENSION: Final = 1
ACTION_DIMENSION: Final = ARM_ACTION_DIMENSION + TCP_HEADING_ACTION_DIMENSION + GRIPPER_ACTION_DIMENSION
DATASET_SCHEMA_VERSION: Final = "r2dreamer-aubo-ik-sequence-v3"
CONTROL_FREQUENCY_HZ: Final = 4.0
JOINT_NAMES: Final = ("Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Flange")
NORMALIZED_ACTION_LOW: Final = -1.0
NORMALIZED_ACTION_HIGH: Final = 1.0
TCP_INCREMENT_COMPONENTS: Final = ("x_m", "y_m", "z_m", "rx_rad", "ry_rad", "rz_rad")
ACTION_COMPONENTS: Final = (*TCP_INCREMENT_COMPONENTS, "tcp_heading_target", "gripper")
GRIPPER_JOINT_NAMES: Final = ("UpperFinger", "DownFinger")
GRIPPER_OPEN_POSITION: Final = (0.0115, -0.0115)
TCP_POSITION_SCALE_M: Final = 0.02
TCP_ROTATION_SCALE_RAD: Final = radians(5.0)
TCP_HEADING_SCALE_RAD: Final = pi
PHYSICS_DT_S: Final = 1.0 / 120.0
MIN_FREE_DISK_BYTES: Final = 128 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """一个数据字段的静态描述。"""

    name: str
    shape: str
    dtype: str
    unit: str
    frame: str
    axis_order: tuple[str, ...] = ()
    episode_axis: str = "step"


@dataclass(frozen=True, slots=True)
class CameraSensorSpec:
    """P0-5 相机配置；位姿未提供时不得静默采集。"""

    name: str
    prim_path: str
    resolution: tuple[int, int]
    data_types: tuple[str, ...]
    position_e: tuple[float, float, float] | None = None
    rotation_wxyz: tuple[float, float, float, float] | None = None
    pose_convention: str = "opengl"
    source_reference: str = ""

    @property
    def pose_is_configured(self) -> bool:
        """相机位置和 wxyz 姿态是否同时提供。"""

        return self.position_e is not None and self.rotation_wxyz is not None

    def validate(self) -> None:
        """检查相机的静态字段，不访问 Isaac。"""

        if not self.name or not self.prim_path:
            raise ValueError("camera name 和 prim_path 不能为空")
        if len(self.resolution) != 2 or any(value <= 0 for value in self.resolution):
            raise ValueError(f"camera {self.name} 的 resolution 必须为正二维值")
        if not self.data_types:
            raise ValueError(f"camera {self.name} 至少需要一个 data_type")
        if self.rotation_wxyz is not None and len(self.rotation_wxyz) != 4:
            raise ValueError(f"camera {self.name} 的 rotation_wxyz 必须为四元数")
        if self.position_e is not None and len(self.position_e) != 3:
            raise ValueError(f"camera {self.name} 的 position_e 必须为三维坐标")
        if self.pose_convention not in {"opengl", "ros", "world"}:
            raise ValueError(f"camera {self.name} 的 pose_convention 不支持：{self.pose_convention}")


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """数据产线配置；只保存契约和写入参数，不包含场景查询副作用。"""

    schema_version: str = DATASET_SCHEMA_VERSION
    control_frequency_hz: float = CONTROL_FREQUENCY_HZ
    physics_dt_s: float = PHYSICS_DT_S
    action_position_scale_m: float = TCP_POSITION_SCALE_M
    action_rotation_scale_rad: float = TCP_ROTATION_SCALE_RAD
    reward_type: str = "formal_environment_reward"
    episodes_per_shard: int = 10
    min_free_disk_bytes: int = MIN_FREE_DISK_BYTES
    cameras: tuple[CameraSensorSpec, ...] = ()

    def validate(self, *, require_camera_poses: bool = False) -> None:
        """验证字段、频率、分片和相机配置。"""

        if self.schema_version != DATASET_SCHEMA_VERSION:
            raise ValueError(f"不支持的数据 schema：{self.schema_version}")
        if self.control_frequency_hz <= 0.0:
            raise ValueError("control_frequency_hz 必须为正数")
        if self.physics_dt_s <= 0.0:
            raise ValueError("physics_dt_s 必须为正数")
        if self.action_position_scale_m <= 0.0 or self.action_rotation_scale_rad <= 0.0:
            raise ValueError("action scale 必须为正数")
        if not self.reward_type:
            raise ValueError("reward_type 不能为空")
        if self.episodes_per_shard <= 0:
            raise ValueError("episodes_per_shard 必须为正整数")
        if self.min_free_disk_bytes < 0:
            raise ValueError("min_free_disk_bytes 不能为负数")
        names = [camera.name for camera in self.cameras]
        if len(names) != len(set(names)):
            raise ValueError("camera name 不能重复")
        for camera in self.cameras:
            camera.validate()
            if require_camera_poses and not camera.pose_is_configured:
                raise ValueError(f"camera {camera.name} 缺少 position_e 或 rotation_wxyz")


def p0_field_specs() -> tuple[FieldSpec, ...]:
    """返回 P0-1 至 P0-3 必须具备的字段表。"""

    return (
        FieldSpec("observation", "opaque", "task-defined", "task-defined", "task-defined", episode_axis="T+1"),
        FieldSpec("action", "(8,)", "float32", "normalized [-1, 1]", "policy output", ACTION_COMPONENTS),
        FieldSpec("reward", "()", "float32", "scalar", "task-defined"),
        FieldSpec(
            "raw_action",
            "(8,)",
            "float32",
            "pre-normalization/unitless",
            "policy output",
            ACTION_COMPONENTS,
        ),
        FieldSpec(
            "normalized_action",
            "(8,)",
            "float32",
            "unitless [-1, 1]",
            "policy output",
            ACTION_COMPONENTS,
        ),
        FieldSpec(
            "tcp_displacement",
            "(6,)",
            "float32",
            "xyz m + rxyz rad",
            "robot_root_B",
            TCP_INCREMENT_COMPONENTS,
        ),
        FieldSpec("ik_output", "(6,)", "float32", "rad", "robot joint space", JOINT_NAMES),
        FieldSpec("joint_position", "(6,)", "float32", "rad", "robot joint space", JOINT_NAMES),
        FieldSpec("joint_velocity", "(6,)", "float32", "rad/s", "robot joint space", JOINT_NAMES),
        FieldSpec("gripper_position", "(2,)", "float32", "rad", "gripper joint space", GRIPPER_JOINT_NAMES),
        FieldSpec("gripper_velocity", "(2,)", "float32", "rad/s", "gripper joint space", GRIPPER_JOINT_NAMES),
        FieldSpec("is_first", "()", "bool", "boolean", "episode boundary", episode_axis="T+1"),
        FieldSpec("is_last", "()", "bool", "boolean", "episode boundary", episode_axis="T+1"),
        FieldSpec("is_terminal", "()", "bool", "boolean", "episode boundary", episode_axis="T+1"),
    )


__all__ = [
    "ACTION_DIMENSION",
    "ACTION_COMPONENTS",
    "ARM_ACTION_DIMENSION",
    "CameraSensorSpec",
    "CONTROL_FREQUENCY_HZ",
    "DATASET_SCHEMA_VERSION",
    "DatasetConfig",
    "FieldSpec",
    "GRIPPER_ACTION_DIMENSION",
    "GRIPPER_JOINT_NAMES",
    "GRIPPER_OPEN_POSITION",
    "JOINT_NAMES",
    "MIN_FREE_DISK_BYTES",
    "PHYSICS_DT_S",
    "NORMALIZED_ACTION_HIGH",
    "NORMALIZED_ACTION_LOW",
    "TCP_INCREMENT_COMPONENTS",
    "TCP_POSITION_SCALE_M",
    "TCP_ROTATION_SCALE_RAD",
    "TCP_HEADING_ACTION_DIMENSION",
    "TCP_HEADING_SCALE_RAD",
    "p0_field_specs",
]
