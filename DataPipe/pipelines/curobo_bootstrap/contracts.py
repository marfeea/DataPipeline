"""Versioned, standard-library-only contracts. Poses use metres and wxyz."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum, IntFlag
from math import isclose, isfinite
from typing import Any

CONTRACT_VERSION = 1


def canonical_json(value: Any) -> str:
    """Reject NaN/Inf and produce stable UTF-8 JSON for content hashing."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def config_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def finite_vector(value, size: int, name: str) -> tuple[float, ...]:
    result = tuple(float(x) for x in value)
    if len(result) != size or not all(isfinite(x) for x in result):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def positive(value: float, name: str, *, zero: bool = False) -> None:
    if not isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")


@dataclass(frozen=True)
class Pose:
    """Rigid pose with an explicit expression frame; no implicit normalization."""

    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    frame: str

    def __post_init__(self):
        object.__setattr__(self, "position", finite_vector(self.position, 3, "position"))
        q = finite_vector(self.quaternion_wxyz, 4, "quaternion_wxyz")
        if not isclose(sum(x * x for x in q), 1.0, abs_tol=1e-6):
            raise ValueError("quaternion_wxyz must have unit norm")
        object.__setattr__(self, "quaternion_wxyz", q)
        if not isinstance(self.frame, str) or not self.frame.strip():
            raise ValueError("frame must be explicit")


@dataclass(frozen=True)
class ConfigHashes:
    robot: str
    task: str
    scene: str
    world: str
    planner: str

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be a lowercase SHA256 digest")


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    seed: int
    initial_joint_position: tuple[float, ...]
    source_pose: Pose
    place_pose: Pose
    grasp_tcp_pose_b: Pose
    place_tcp_pose_b: Pose
    hashes: ConfigHashes
    pregrasp_m: float = 0.05
    lift_m: float = 0.10
    preplace_m: float = 0.05
    retreat_m: float = 0.05
    gripper_dwell_s: float = 0.5
    contract_version: int = CONTRACT_VERSION

    def __post_init__(self):
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("episode_id must be nonempty")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be a nonnegative 63-bit integer")
        if type(self.contract_version) is not int or self.contract_version != CONTRACT_VERSION:
            raise ValueError("unsupported contract_version")
        object.__setattr__(self, "initial_joint_position", finite_vector(self.initial_joint_position, 6, "q0"))
        for name in ("source_pose", "place_pose", "grasp_tcp_pose_b", "place_tcp_pose_b"):
            if not isinstance(getattr(self, name), Pose):
                raise ValueError(f"{name} must be a Pose")
        if self.source_pose.frame != self.place_pose.frame:
            raise ValueError("source and place object poses must share a frame")
        if self.grasp_tcp_pose_b.frame != "B" or self.place_tcp_pose_b.frame != "B":
            raise ValueError("TCP goals must be expressed in B")
        if not isinstance(self.hashes, ConfigHashes):
            raise ValueError("hashes must be ConfigHashes")
        for name in ("pregrasp_m", "lift_m", "preplace_m", "retreat_m", "gripper_dwell_s"):
            positive(getattr(self, name), name)

    def to_json(self) -> str:
        return canonical_json(asdict(self))

    @classmethod
    def from_json(cls, payload: str) -> EpisodeSpec:
        data = json.loads(payload)
        for key in ("source_pose", "place_pose", "grasp_tcp_pose_b", "place_tcp_pose_b"):
            data[key] = Pose(**data[key])
        data["hashes"] = ConfigHashes(**data["hashes"])
        return cls(**data)


class Stage(str, Enum):
    PREGRASP = "PREGRASP"
    APPROACH_GRASP = "APPROACH_GRASP"
    CLOSE_GRIPPER = "CLOSE_GRIPPER"
    VERIFY_GRASP = "VERIFY_GRASP"
    LIFT = "LIFT"
    TRANSFER = "TRANSFER"
    PREPLACE = "PREPLACE"
    PLACE = "PLACE"
    OPEN_GRIPPER = "OPEN_GRIPPER"
    VERIFY_PLACE = "VERIFY_PLACE"
    RETREAT = "RETREAT"


class EventFlag(IntFlag):
    NONE = 0
    CLOSE_GRIPPER = 1
    OPEN_GRIPPER = 2
    VERIFY_GRASP = 4
    VERIFY_PLACE = 8
    ATTACH = 16
    DETACH = 32


@dataclass(frozen=True)
class MotionSegment:
    stage: Stage
    target_tcp_pose_b: Pose

    def __post_init__(self):
        if not isinstance(self.stage, Stage) or self.stage in {
            Stage.CLOSE_GRIPPER,
            Stage.OPEN_GRIPPER,
            Stage.VERIFY_GRASP,
            Stage.VERIFY_PLACE,
        }:
            raise ValueError("motion requires a motion stage")
        if not isinstance(self.target_tcp_pose_b, Pose) or self.target_tcp_pose_b.frame != "B":
            raise ValueError("motion target must be expressed in B")


@dataclass(frozen=True)
class DiscreteEvent:
    stage: Stage
    flags: EventFlag
    dwell_s: float = 0.0

    def __post_init__(self):
        expected = {
            Stage.CLOSE_GRIPPER: EventFlag.CLOSE_GRIPPER,
            Stage.OPEN_GRIPPER: EventFlag.OPEN_GRIPPER,
            Stage.VERIFY_GRASP: EventFlag.VERIFY_GRASP | EventFlag.ATTACH,
            Stage.VERIFY_PLACE: EventFlag.VERIFY_PLACE | EventFlag.DETACH,
        }
        if not isinstance(self.stage, Stage) or not isinstance(self.flags, EventFlag):
            raise ValueError("typed stage and event flags are required")
        if self.stage not in expected or self.flags != expected[self.stage]:
            raise ValueError("invalid stage/event combination")
        positive(self.dwell_s, "dwell_s", zero=True)


@dataclass(frozen=True)
class TrajectoryProgram:
    episode_id: str
    steps: tuple[MotionSegment | DiscreteEvent, ...]

    def __post_init__(self):
        object.__setattr__(self, "steps", tuple(self.steps))
        if not all(isinstance(step, MotionSegment | DiscreteEvent) for step in self.steps):
            raise ValueError("program requires motion segments or discrete events")
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("episode_id must be nonempty")
        if tuple(step.stage for step in self.steps) != tuple(Stage):
            raise ValueError("program must contain the complete ordered pick-place stages")

    def to_json(self) -> str:
        steps = []
        for step in self.steps:
            data = asdict(step)
            data["kind"] = "motion" if isinstance(step, MotionSegment) else "event"
            steps.append(data)
        return canonical_json({"contract_version": CONTRACT_VERSION, "episode_id": self.episode_id, "steps": steps})

    @classmethod
    def from_json(cls, payload: str) -> TrajectoryProgram:
        data = json.loads(payload)
        version = data.pop("contract_version")
        if type(version) is not int or version != CONTRACT_VERSION:
            raise ValueError("unsupported contract_version")
        steps = []
        for raw in data.pop("steps"):
            kind = raw.pop("kind")
            raw["stage"] = Stage(raw["stage"])
            if kind == "motion":
                raw["target_tcp_pose_b"] = Pose(**raw["target_tcp_pose_b"])
                steps.append(MotionSegment(**raw))
            elif kind == "event":
                raw["flags"] = EventFlag(raw["flags"])
                steps.append(DiscreteEvent(**raw))
            else:
                raise ValueError("unknown program step kind")
        return cls(steps=tuple(steps), **data)


@dataclass(frozen=True)
class FailureDiagnostic:
    episode_id: str
    phase: str
    code: str
    message: str
    segment_index: int | None = None

    def __post_init__(self):
        if self.phase not in {"contract", "planning", "preflight", "execution", "task", "storage"}:
            raise ValueError("unknown failure phase")
        if not all(isinstance(x, str) and x.strip() for x in (self.episode_id, self.code, self.message)):
            raise ValueError("failure requires episode_id, code and message")
        if self.segment_index is not None and (type(self.segment_index) is not int or self.segment_index < 0):
            raise ValueError("segment_index must be nonnegative")

    def to_json(self) -> str:
        return canonical_json(asdict(self))


@dataclass(frozen=True)
class PlannedTrajectory:
    """T commands at interval end times; initial q belongs to EpisodeSpec.

    Observations have T+1 samples. Segment indices address program.steps.
    """

    joint_names: tuple[str, ...]
    control_hz: float
    time_s: tuple[float, ...]
    joint_position: tuple[tuple[float, ...], ...]
    joint_velocity: tuple[tuple[float, ...], ...]
    joint_acceleration: tuple[tuple[float, ...], ...]
    planned_tcp_pose_b: tuple[tuple[float, ...], ...]
    gripper_command: tuple[float, ...]
    segment_index: tuple[int, ...]
    segment_sample_index: tuple[int, ...]
    semantic_stage_id: tuple[int, ...]
    event_flags: tuple[int, ...]
    collision_distance: tuple[float, ...] | None = None
    contract_version: int = CONTRACT_VERSION

    def __post_init__(self):
        if type(self.contract_version) is not int or self.contract_version != CONTRACT_VERSION:
            raise ValueError("unsupported contract_version")
        positive(self.control_hz, "control_hz")
        names = tuple(self.joint_names)
        if len(names) != 6 or len(set(names)) != 6 or not all(isinstance(x, str) and x.strip() for x in names):
            raise ValueError("six distinct joint names required")
        object.__setattr__(self, "joint_names", names)
        times = tuple(float(t) for t in self.time_s)
        count = len(times)
        if not count or any(
            not isfinite(t) or not isclose(t, (i + 1) / self.control_hz, rel_tol=0, abs_tol=1e-9)
            for i, t in enumerate(times)
        ):
            raise ValueError("time_s must be the nonempty uniform command-end timeline")
        object.__setattr__(self, "time_s", times)
        for field, width in (
            ("joint_position", 6),
            ("joint_velocity", 6),
            ("joint_acceleration", 6),
            ("planned_tcp_pose_b", 7),
        ):
            rows = tuple(finite_vector(row, width, field) for row in getattr(self, field))
            if len(rows) != count:
                raise ValueError(f"{field} must have T rows")
            object.__setattr__(self, field, rows)
        for pose in self.planned_tcp_pose_b:
            Pose(pose[:3], pose[3:], "B")
        gripper = finite_vector(self.gripper_command, count, "gripper_command")
        if any(x not in (-1.0, 1.0) for x in gripper):
            raise ValueError("gripper_command must be -1 (close) or +1 (open)")
        object.__setattr__(self, "gripper_command", gripper)
        for field in ("segment_index", "segment_sample_index", "semantic_stage_id", "event_flags"):
            values = tuple(getattr(self, field))
            if len(values) != count or any(type(x) is not int or x < 0 for x in values):
                raise ValueError(f"{field} must contain T nonnegative integers")
            object.__setattr__(self, field, values)
        if any(x >= len(Stage) for x in self.segment_index):
            raise ValueError("segment index outside program")
        for i, (segment, sample) in enumerate(zip(self.segment_index, self.segment_sample_index)):
            if i == 0 or segment != self.segment_index[i - 1]:
                if sample != 0 or (i and segment < self.segment_index[i - 1]):
                    raise ValueError("segments must advance and sample indices must restart at zero")
            elif sample != self.segment_sample_index[i - 1] + 1:
                raise ValueError("segment sample indices must be consecutive")
        if any(x & ~63 for x in self.event_flags):
            raise ValueError("unknown event flag bits")
        if self.collision_distance is not None:
            object.__setattr__(
                self, "collision_distance", finite_vector(self.collision_distance, count, "collision_distance")
            )

    def to_json(self) -> str:
        return canonical_json(asdict(self))

    @classmethod
    def from_json(cls, payload: str) -> PlannedTrajectory:
        return cls(**json.loads(payload))
