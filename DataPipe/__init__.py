"""R2Dreamer IK 数据产线的 P0 纯逻辑契约。"""

from .baseline import STATIC_IK_SCENE_BASELINE, SceneBaseline
from .camera_config import (
    camera_sensor_spec_from_diagnostic_pose,
    dataset_config_with_project_diagnostic_cameras,
    project_diagnostic_camera_sensor_specs,
)
from .capture import CameraCaptureAdapter, CameraFrame, CaptureRecord, ControlState, StateCaptureAdapter
from .config import (
    ACTION_DIMENSION,
    ARM_ACTION_DIMENSION,
    DATASET_SCHEMA_VERSION,
    GRIPPER_OPEN_POSITION,
    JOINT_NAMES,
    TCP_HEADING_ACTION_DIMENSION,
    CameraSensorSpec,
    DatasetConfig,
    p0_field_specs,
)
from .hdf5_writer import HDF5TrajectoryWriter
from .mamba_sequence_buffer import MambaSequenceBuffer
from .r2dreamer import R2DreamerBatch
from .records import (
    ActionRecord,
    BoundaryFlags,
    Episode,
    EpisodeBuilder,
    ObservationFrame,
    StepRecord,
    TerminationReason,
)
from .reward import RewardLossConfig, compute_reward_prediction_loss, smooth_l1_reward_error
from .sequence import (
    PLACE_TRANSFER_HEADING_FRACTION,
    DatasetSequence,
    SequenceCommand,
    SequenceConfig,
    SequenceObservation,
    SequenceStage,
    heading_to_normalized_action,
    normalized_action_to_heading,
    planar_heading_from_quaternion,
    planar_heading_from_vector,
    wrap_angle_rad,
)
from .state import (
    normalized_action_to_tcp_delta,
    tcp_delta_to_normalized_action,
    validate_explicit_observation,
    validate_normalized_action,
)
from .supervision import EpisodeOutcome, EpisodeSupervisor

__all__ = [
    "ACTION_DIMENSION",
    "ARM_ACTION_DIMENSION",
    "TCP_HEADING_ACTION_DIMENSION",
    "ActionRecord",
    "BoundaryFlags",
    "CameraCaptureAdapter",
    "CameraFrame",
    "CameraSensorSpec",
    "CaptureRecord",
    "ControlState",
    "DATASET_SCHEMA_VERSION",
    "DatasetConfig",
    "GRIPPER_OPEN_POSITION",
    "DatasetSequence",
    "Episode",
    "EpisodeBuilder",
    "EpisodeOutcome",
    "EpisodeSupervisor",
    "HDF5TrajectoryWriter",
    "JOINT_NAMES",
    "ObservationFrame",
    "PLACE_TRANSFER_HEADING_FRACTION",
    "camera_sensor_spec_from_diagnostic_pose",
    "dataset_config_with_project_diagnostic_cameras",
    "project_diagnostic_camera_sensor_specs",
    "R2DreamerBatch",
    "RewardLossConfig",
    "compute_reward_prediction_loss",
    "smooth_l1_reward_error",
    "MambaSequenceBuffer",
    "STATIC_IK_SCENE_BASELINE",
    "SceneBaseline",
    "SequenceCommand",
    "SequenceConfig",
    "SequenceObservation",
    "SequenceStage",
    "heading_to_normalized_action",
    "normalized_action_to_heading",
    "planar_heading_from_quaternion",
    "planar_heading_from_vector",
    "StateCaptureAdapter",
    "StepRecord",
    "TerminationReason",
    "normalized_action_to_tcp_delta",
    "tcp_delta_to_normalized_action",
    "validate_explicit_observation",
    "validate_normalized_action",
    "wrap_angle_rad",
    "p0_field_specs",
]
