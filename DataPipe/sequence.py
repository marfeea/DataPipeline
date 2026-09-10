"""Pure data-collection stage machine and TCP normal geometry."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import IntEnum
from math import acos, atan2, isfinite, pi, sqrt

from configs.coordinate_frames import (
    IDENTITY_QUATERNION_WXYZ,
    TCP_DIRECTION_AXIS_T,
    base_to_world_direction,
    direction_error_rad,
    tcp_direction_in_world,
    world_to_base_direction,
)

from .config import TCP_POSITION_SCALE_M, TCP_ROTATION_SCALE_RAD

Vector3 = tuple[float, float, float]
QuaternionWxyz = tuple[float, float, float, float]
ArmAction = tuple[float, float, float, float, float, float]
ZERO_ARM_ACTION: ArmAction = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class SequenceStage(IntEnum):
    """Stable stage IDs stored in the dataset."""

    HIGH_XY_APPROACH = 0
    MATCH_Z = 1
    ALIGN_NORMAL = 2
    ADVANCE_NORMAL = 3
    CLOSE_GRIPPER = 4
    HOLD_COMPLETE = 5
    LIFT_OBJECT = 6
    ALIGN_PLACE_DIRECTION = 11
    PLACE_HIGH_XY = 7
    PLACE_FINAL_Z = 10
    PLACE_FINAL_XY = 12
    RELEASE_GRIPPER = 8
    PLACE_COMPLETE = 9
    # 兼容旧调用方：旧的 PLACE_OBJECT(7) 现在明确表示高位 XOY 平移阶段。
    PLACE_OBJECT = PLACE_HIGH_XY


# 阶段 ID 为兼容已生成数据保持稳定，不能再用整数大小推断执行顺序。
SEQUENCE_STAGE_ORDER = (
    SequenceStage.HIGH_XY_APPROACH,
    SequenceStage.MATCH_Z,
    SequenceStage.ALIGN_NORMAL,
    SequenceStage.ADVANCE_NORMAL,
    SequenceStage.CLOSE_GRIPPER,
    SequenceStage.LIFT_OBJECT,
    SequenceStage.PLACE_HIGH_XY,
    SequenceStage.ALIGN_PLACE_DIRECTION,
    SequenceStage.PLACE_FINAL_Z,
    SequenceStage.PLACE_FINAL_XY,
    SequenceStage.RELEASE_GRIPPER,
    SequenceStage.PLACE_COMPLETE,
)
SEQUENCE_STAGE_RANK = {stage: rank for rank, stage in enumerate(SEQUENCE_STAGE_ORDER)}
# HOLD_COMPLETE 是 position-only/grasp 分支的终态，与 pick-place 的末态同级。
SEQUENCE_STAGE_RANK[SequenceStage.HOLD_COMPLETE] = len(SEQUENCE_STAGE_ORDER) - 1
LEGACY_SEQUENCE_STAGE_ORDER_V3 = (
    SequenceStage.HIGH_XY_APPROACH,
    SequenceStage.MATCH_Z,
    SequenceStage.ALIGN_NORMAL,
    SequenceStage.ADVANCE_NORMAL,
    SequenceStage.CLOSE_GRIPPER,
    SequenceStage.LIFT_OBJECT,
    SequenceStage.ALIGN_PLACE_DIRECTION,
    SequenceStage.PLACE_HIGH_XY,
    SequenceStage.PLACE_FINAL_Z,
    SequenceStage.PLACE_FINAL_XY,
    SequenceStage.RELEASE_GRIPPER,
    SequenceStage.PLACE_COMPLETE,
)
PLACE_TRANSFER_HEADING_FRACTION = 0.5


@dataclass(frozen=True, slots=True)
class SequenceConfig:
    position_tolerance_m: float = 0.01
    # 字段名沿用旧 CLI；用于抓取和放置的 XOY 平面 heading 到位门限。
    normal_angle_tolerance_rad: float = 5.0 * pi / 180.0
    normal_dwell_steps: int = 3
    advance_distance_m: float = 0.0
    gripper_position_tolerance: float = 0.008
    gripper_dwell_steps: int = 3
    heading_step_rad: float = TCP_ROTATION_SCALE_RAD

    def validate(self, *, grasp_enabled: bool) -> None:
        if self.position_tolerance_m <= 0.0:
            raise ValueError("position_tolerance_m must be positive")
        if not 0.0 < self.normal_angle_tolerance_rad < pi:
            raise ValueError("normal_angle_tolerance_rad must be in (0, pi)")
        if self.normal_dwell_steps <= 0 or self.gripper_dwell_steps <= 0:
            raise ValueError("stage dwell steps must be positive integers")
        if self.advance_distance_m < 0.0:
            raise ValueError("advance_distance_m must be non-negative")
        if self.gripper_position_tolerance <= 0.0:
            raise ValueError("gripper_position_tolerance must be positive")
        if not 0.0 < self.heading_step_rad <= pi:
            raise ValueError("heading_step_rad must be in (0, pi]")


@dataclass(frozen=True, slots=True)
class SequenceState:
    stage: SequenceStage = SequenceStage.HIGH_XY_APPROACH
    dwell_steps: int = 0
    stage_steps: int = 0
    complete: bool = False
    tcp_quaternion_ref_b_wxyz: QuaternionWxyz | None = None
    tcp_heading_ref_rad: float | None = None


@dataclass(frozen=True, slots=True)
class SequenceObservation:
    tcp_position_b: Vector3
    tcp_quaternion_b_wxyz: QuaternionWxyz
    gripper_position: tuple[float, float]
    # q_WB：Base 坐标系 B 在世界坐标系 W 中的姿态，把 B 系方向旋转到 W 系。
    base_quaternion_wxyz: QuaternionWxyz = IDENTITY_QUATERNION_WXYZ


@dataclass(frozen=True, slots=True)
class SequenceCommand:
    stage: SequenceStage
    target_position_b: Vector3
    desired_axis_b: Vector3
    arm_action: ArmAction
    gripper_command: float
    flange_downward_enabled: bool
    complete: bool
    tcp_heading_target_rad: float = 0.0

    @property
    def dataset_action(self) -> tuple[float, ...]:
        return (*self.arm_action, heading_to_normalized_action(self.tcp_heading_target_rad), self.gripper_command)


def _vector(values: Sequence[float], *, size: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size or not all(isfinite(value) for value in result):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return result


def _dot(left: Vector3, right: Vector3) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(vector: Sequence[float]) -> float:
    return sqrt(sum(value * value for value in vector))


def normalize(vector: Sequence[float], *, name: str = "vector") -> Vector3:
    values = _vector(vector, size=3, name=name)
    magnitude = _norm(values)
    if magnitude <= 1.0e-9:
        raise ValueError(f"{name} must have non-zero length")
    return tuple(value / magnitude for value in values)  # type: ignore[return-value]


def wrap_angle_rad(angle_rad: float) -> float:
    """Wrap an angle to [-pi, pi)."""

    if not isfinite(angle_rad):
        raise ValueError("angle_rad must be finite")
    return (angle_rad + pi) % (2.0 * pi) - pi


def planar_heading_from_vector(vector_b: Sequence[float]) -> float:
    """Return the vector heading in the B-frame XOY plane."""

    values = _vector(vector_b, size=3, name="vector_b")
    planar_norm = sqrt(values[0] * values[0] + values[1] * values[1])
    if planar_norm <= 1.0e-9:
        raise ValueError("vector_b must have a non-zero XOY projection")
    return atan2(values[1], values[0])


def planar_heading_from_quaternion(
    quaternion_wxyz: Sequence[float], tool_forward_axis: Sequence[float] = (0.0, 0.0, 1.0)
) -> float:
    """Return the TCP tool-axis heading in the B-frame XOY plane."""

    return planar_heading_from_vector(quaternion_rotate_vector(quaternion_wxyz, tool_forward_axis))


def heading_to_normalized_action(heading_rad: float) -> float:
    """Encode a planar heading in the normalized action range."""

    return max(-1.0, min(1.0, wrap_angle_rad(heading_rad) / pi))


def normalized_action_to_heading(value: float) -> float:
    """Decode a normalized planar heading action to radians."""

    if not isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError("normalized heading must be finite and within [-1, 1]")
    return float(value * pi)


def approach_direction(preposition_b: Sequence[float], target_b: Sequence[float]) -> Vector3:
    """Return the prescribed-point-to-target unit vector in robot-root B."""

    preposition = _vector(preposition_b, size=3, name="preposition_b")
    target = _vector(target_b, size=3, name="target_b")
    return normalize(tuple(end - start for start, end in zip(preposition, target, strict=True)), name="approach")


def quaternion_rotate_vector(quaternion_wxyz: Sequence[float], vector: Sequence[float]) -> Vector3:
    """Rotate a vector by a normalized wxyz quaternion."""

    w, x, y, z = _vector(quaternion_wxyz, size=4, name="quaternion_wxyz")
    magnitude = sqrt(w * w + x * x + y * y + z * z)
    if magnitude <= 1.0e-9:
        raise ValueError("quaternion_wxyz must have non-zero norm")
    w, x, y, z = (value / magnitude for value in (w, x, y, z))
    vx, vy, vz = _vector(vector, size=3, name="vector")
    tx, ty, tz = 2.0 * (y * vz - z * vy), 2.0 * (z * vx - x * vz), 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    )


def quaternion_rotation_error_vector(
    current_quaternion_wxyz: Sequence[float], target_quaternion_wxyz: Sequence[float]
) -> Vector3:
    """Return the shortest rotation vector from the current to target quaternion."""

    current = _vector(current_quaternion_wxyz, size=4, name="current_quaternion_wxyz")
    target = _vector(target_quaternion_wxyz, size=4, name="target_quaternion_wxyz")
    current_norm = sqrt(sum(value * value for value in current))
    target_norm = sqrt(sum(value * value for value in target))
    if current_norm <= 1.0e-9 or target_norm <= 1.0e-9:
        raise ValueError("quaternions must have non-zero length")
    current = tuple(value / current_norm for value in current)
    target = tuple(value / target_norm for value in target)
    cw, cx, cy, cz = current
    tw, tx, ty, tz = target
    # target * conjugate(current), both quaternions use wxyz ordering.
    error = (
        tw * cw + tx * cx + ty * cy + tz * cz,
        -tw * cx + tx * cw - ty * cz + tz * cy,
        -tw * cy + tx * cz + ty * cw - tz * cx,
        -tw * cz - tx * cy + ty * cx + tz * cw,
    )
    if error[0] < 0.0:
        error = tuple(-value for value in error)
    vector_norm = sqrt(sum(value * value for value in error[1:]))
    if vector_norm <= 1.0e-9:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * atan2(vector_norm, max(error[0], 0.0))
    return tuple(angle * value / vector_norm for value in error[1:])  # type: ignore[return-value]


def minimal_rotation_vector(current_axis_b: Sequence[float], desired_axis_b: Sequence[float]) -> Vector3:
    """Return the shortest B-frame rotation vector from current to desired axis."""

    current = normalize(current_axis_b, name="current_axis_b")
    desired = normalize(desired_axis_b, name="desired_axis_b")
    cross = _cross(current, desired)
    cross_norm = _norm(cross)
    dot = max(-1.0, min(1.0, _dot(current, desired)))
    if cross_norm > 1.0e-9:
        angle = atan2(cross_norm, dot)
        return tuple(angle * value / cross_norm for value in cross)  # type: ignore[return-value]
    if dot >= 0.0:
        return (0.0, 0.0, 0.0)
    basis = min(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), key=lambda item: abs(_dot(current, item)))
    axis = normalize(_cross(current, basis), name="antiparallel_axis")
    return tuple(pi * value for value in axis)  # type: ignore[return-value]


def axis_angle_error_rad(current_axis_b: Sequence[float], desired_axis_b: Sequence[float]) -> float:
    current = normalize(current_axis_b, name="current_axis_b")
    desired = normalize(desired_axis_b, name="desired_axis_b")
    return acos(max(-1.0, min(1.0, _dot(current, desired))))


def stage_for_waypoint_label(label: str) -> SequenceStage:
    base_label = label.removesuffix("_interp")
    return (
        SequenceStage.MATCH_Z if base_label in {"vertical_descent", "target_settle"} else SequenceStage.HIGH_XY_APPROACH
    )


def stepped_heading_target(current_heading_rad: float, final_heading_rad: float, max_step_rad: float) -> float:
    """Return a bounded heading target along the wrapped shortest branch."""

    if not 0.0 < max_step_rad <= pi:
        raise ValueError("max_step_rad must be in (0, pi]")
    error = wrap_angle_rad(float(final_heading_rad) - float(current_heading_rad))
    bounded_error = max(-max_step_rad, min(max_step_rad, error))
    return wrap_angle_rad(float(current_heading_rad) + bounded_error)


class DatasetSequence:
    """Monotonic collection-only stage machine; it never changes task logic."""

    def __init__(
        self,
        *,
        preposition_b: Sequence[float],
        target_b: Sequence[float] | None = None,
        grasp_tcp_position_b: Sequence[float] | None = None,
        grasp_tcp_direction_w: Sequence[float] | None = None,
        lift_position_b: Sequence[float] | None = None,
        place_high_position_b: Sequence[float] | None = None,
        place_position_b: Sequence[float] | None = None,
        place_tcp_axis_b: Sequence[float] | None = None,
        place_tcp_direction_w: Sequence[float] | None = None,
        config: SequenceConfig,
        grasp_enabled: bool,
        place_enabled: bool = False,
        tool_forward_axis: Sequence[float] = (0.0, 0.0, 1.0),
        tcp_heading_axis: Sequence[float] = TCP_DIRECTION_AXIS_T,
        initial_tcp_heading_rad: float | None = None,
        initial_state: SequenceState | None = None,
    ) -> None:
        config.validate(grasp_enabled=grasp_enabled)
        self.preposition_b = _vector(preposition_b, size=3, name="preposition_b")
        if grasp_tcp_position_b is not None and target_b is not None:
            raise ValueError("provide grasp_tcp_position_b or legacy target_b, not both")
        legacy_target = grasp_tcp_position_b is None and target_b is not None
        grasp_target = grasp_tcp_position_b if grasp_tcp_position_b is not None else target_b
        if grasp_enabled and grasp_target is None:
            raise ValueError("grasp sequence requires grasp_tcp_position_b")
        if grasp_target is None:
            grasp_target = self.preposition_b
        direction_target_b = _vector(grasp_target, size=3, name="grasp_tcp_position_b")
        self.target_b = direction_target_b
        self.desired_axis_b = (
            approach_direction(self.preposition_b, direction_target_b)
            if direction_target_b != self.preposition_b
            else (0.0, 0.0, 1.0)
        )
        self.grasp_tcp_position_b = (
            tuple(
                value + config.advance_distance_m * axis
                for value, axis in zip(self.preposition_b, self.desired_axis_b, strict=True)
            )
            if legacy_target
            else direction_target_b
        )
        self.grasp_tcp_direction_w = (
            normalize(grasp_tcp_direction_w, name="grasp_tcp_direction_w")
            if grasp_tcp_direction_w is not None
            else None
        )
        self.advance_target_b = self.grasp_tcp_position_b
        if not legacy_target and config.advance_distance_m > 0.0:
            configured_distance = _norm(
                tuple(
                    grasp - preposition
                    for grasp, preposition in zip(self.grasp_tcp_position_b, self.preposition_b, strict=True)
                )
            )
            if abs(configured_distance - config.advance_distance_m) > config.position_tolerance_m:
                raise ValueError("advance_distance_m is inconsistent with grasp_tcp_position_b")
        self.place_enabled = bool(place_enabled)
        if self.place_enabled and not grasp_enabled:
            raise ValueError("place sequence requires grasp_enabled")
        place_positions = (lift_position_b, place_high_position_b, place_position_b)
        if self.place_enabled and any(position is None for position in place_positions):
            raise ValueError("place sequence requires lift, place-high, and place positions")
        self.lift_position_b = (
            _vector(lift_position_b, size=3, name="lift_position_b") if lift_position_b is not None else None
        )
        self.place_high_position_b = (
            _vector(place_high_position_b, size=3, name="place_high_position_b")
            if place_high_position_b is not None
            else None
        )
        self.place_position_b = (
            _vector(place_position_b, size=3, name="place_position_b") if place_position_b is not None else None
        )
        if self.place_enabled and place_tcp_axis_b is None and place_tcp_direction_w is None:
            raise ValueError("place sequence requires place_tcp_direction_w or legacy place_tcp_axis_b")
        # ``place_tcp_direction_w`` 是正式接口：目标方向始终在世界系 W 中表达。
        # ``place_tcp_axis_b`` 只保留给旧数据和纯逻辑调用方，运行时会先转回 W 系再比较。
        self.place_tcp_axis_b = (
            normalize(place_tcp_axis_b, name="place_tcp_axis_b") if place_tcp_axis_b is not None else None
        )
        self.place_tcp_direction_w = (
            normalize(place_tcp_direction_w, name="place_tcp_direction_w")
            if place_tcp_direction_w is not None
            else None
        )
        # TCP 工具轴用于放置方向对齐；Flange-down 约束下只保留 XOY 平面朝向角。
        self.tool_forward_axis = normalize(tool_forward_axis, name="tool_forward_axis")
        self.tcp_heading_axis = normalize(tcp_heading_axis, name="tcp_heading_axis")
        self.initial_tcp_heading_rad = (
            None if initial_tcp_heading_rad is None else wrap_angle_rad(float(initial_tcp_heading_rad))
        )
        self.config = config
        self.grasp_enabled = bool(grasp_enabled)
        self.initial_state = initial_state or SequenceState()
        if self.initial_state.dwell_steps < 0 or self.initial_state.stage_steps < 0:
            raise ValueError("initial sequence counters must be non-negative")
        self.state = self.initial_state

    def reset(self) -> None:
        self.state = self.initial_state

    def _arm_action(
        self,
        observation: SequenceObservation,
        target_position_b: Vector3,
        target_quaternion_b_wxyz: QuaternionWxyz | None = None,
        *,
        target_axis_b: Vector3 | None = None,
    ) -> ArmAction:
        position = _vector(observation.tcp_position_b, size=3, name="tcp_position_b")
        translation = tuple(
            max(-1.0, min(1.0, (target - actual) / TCP_POSITION_SCALE_M))
            for target, actual in zip(target_position_b, position, strict=True)
        )
        if target_quaternion_b_wxyz is not None and target_axis_b is not None:
            raise ValueError("provide target_quaternion_b_wxyz or target_axis_b, not both")
        # 全程由 Isaac action term 施加 Flange-down 约束；抓取前进阶段仍沿用
        # 锁存四元数的姿态保持策略。放置阶段的平面朝向由独立 heading 动作记录，
        # 不再依赖把该自由度混入六维 IK 的旋转增量。
        if target_axis_b is not None:
            current_axis_b = quaternion_rotate_vector(observation.tcp_quaternion_b_wxyz, self.tool_forward_axis)
            rotation_error = minimal_rotation_vector(current_axis_b, target_axis_b)
        else:
            rotation_error = (
                (0.0, 0.0, 0.0)
                if target_quaternion_b_wxyz is None
                else quaternion_rotation_error_vector(observation.tcp_quaternion_b_wxyz, target_quaternion_b_wxyz)
            )
        rotation = tuple(max(-1.0, min(1.0, value / TCP_ROTATION_SCALE_RAD)) for value in rotation_error)
        return (*translation, *rotation)

    def _qualified(self, condition: bool) -> int:
        return self.state.dwell_steps + 1 if condition else 0

    def _place_direction_w(self, observation: SequenceObservation) -> Vector3:
        """返回世界系 W 中的目标 TCP 朝向。"""

        if self.place_tcp_direction_w is not None:
            return self.place_tcp_direction_w
        if self.place_tcp_axis_b is None:
            raise RuntimeError("place stage requires a TCP direction")
        return base_to_world_direction(self.place_tcp_axis_b, observation.base_quaternion_wxyz)

    def _place_direction_b(self, observation: SequenceObservation) -> Vector3:
        """仅在生成 B 系控制目标时，把世界系目标方向转换到 B 系。"""

        return world_to_base_direction(self._place_direction_w(observation), observation.base_quaternion_wxyz)

    def _place_heading_rad(self, observation: SequenceObservation) -> float:
        return planar_heading_from_vector(self._place_direction_b(observation))

    def _tcp_direction_w(self, observation: SequenceObservation) -> Vector3:
        """把子 prim +Y_C 在 IK TCP 系中的表达经 q_BT、q_WB 转到 W。"""

        return tcp_direction_in_world(
            observation.base_quaternion_wxyz,
            observation.tcp_quaternion_b_wxyz,
            self.tcp_heading_axis,
        )

    def _place_direction_ok(self, observation: SequenceObservation) -> bool:
        """下降及松爪前仍检查实测世界方向，避免高位到位后姿态漂移。"""
        return direction_error_rad(
            self._tcp_direction_w(observation), self._place_direction_w(observation)
        ) <= self.config.normal_angle_tolerance_rad

    def _place_transfer_heading(self, observation: SequenceObservation) -> float:
        place_heading_rad = self._place_heading_rad(observation)
        source_heading = self.state.tcp_heading_ref_rad
        if source_heading is None:
            source_heading = self.initial_tcp_heading_rad
        if source_heading is None:
            source_heading = planar_heading_from_quaternion(
                observation.tcp_quaternion_b_wxyz,
                self.tcp_heading_axis,
            )
        return wrap_angle_rad(
            source_heading + PLACE_TRANSFER_HEADING_FRACTION * wrap_angle_rad(place_heading_rad - source_heading)
        )

    def _grasp_heading_rad(self, observation: SequenceObservation) -> float:
        if self.grasp_tcp_direction_w is not None:
            return planar_heading_from_vector(
                world_to_base_direction(self.grasp_tcp_direction_w, observation.base_quaternion_wxyz)
            )
        # 兼容未提供抓取方向的旧调用方。
        if self.initial_tcp_heading_rad is not None:
            return self.initial_tcp_heading_rad
        return planar_heading_from_quaternion(observation.tcp_quaternion_b_wxyz, self.tcp_heading_axis)

    def _heading_target(self, observation: SequenceObservation, stage: SequenceStage) -> float:
        if stage is SequenceStage.ALIGN_NORMAL:
            return stepped_heading_target(
                planar_heading_from_quaternion(observation.tcp_quaternion_b_wxyz, self.tcp_heading_axis),
                self._grasp_heading_rad(observation),
                self.config.heading_step_rad,
            )
        if stage in {SequenceStage.PLACE_HIGH_XY, SequenceStage.ALIGN_PLACE_DIRECTION}:
            current_heading = planar_heading_from_quaternion(
                observation.tcp_quaternion_b_wxyz,
                self.tcp_heading_axis,
            )
            final_heading = (
                self._place_transfer_heading(observation)
                if stage is SequenceStage.PLACE_HIGH_XY
                else self._place_heading_rad(observation)
            )
            return stepped_heading_target(
                current_heading,
                final_heading,
                self.config.heading_step_rad,
            )
        if stage in {
            SequenceStage.PLACE_FINAL_Z,
            SequenceStage.PLACE_FINAL_XY,
            SequenceStage.RELEASE_GRIPPER,
            SequenceStage.PLACE_COMPLETE,
        }:
            return self._place_heading_rad(observation)
        if self.state.tcp_heading_ref_rad is not None:
            return self.state.tcp_heading_ref_rad
        if self.initial_tcp_heading_rad is not None:
            return self.initial_tcp_heading_rad
        return planar_heading_from_quaternion(observation.tcp_quaternion_b_wxyz, self.tcp_heading_axis)

    def _make_command(
        self,
        observation: SequenceObservation,
        stage: SequenceStage,
        target_position_b: Vector3,
        desired_axis_b: Vector3,
        arm_action: ArmAction,
        gripper_command: float,
        _flange_downward_enabled: bool,
        complete: bool,
    ) -> SequenceCommand:
        del _flange_downward_enabled
        return SequenceCommand(
            stage,
            target_position_b,
            desired_axis_b,
            arm_action,
            gripper_command,
            True,
            complete,
            self._heading_target(observation, stage),
        )

    def command(
        self,
        observation: SequenceObservation,
        *,
        active_waypoint_b: Sequence[float],
        active_waypoint_label: str,
        path_complete: bool,
        closed_gripper_position: Sequence[float],
        open_gripper_position: Sequence[float] | None = None,
    ) -> SequenceCommand:
        """Advance stage gates from measured state and return the next command."""

        waypoint = _vector(active_waypoint_b, size=3, name="active_waypoint_b")
        closed = _vector(closed_gripper_position, size=2, name="closed_gripper_position")
        opened = (
            _vector(open_gripper_position, size=2, name="open_gripper_position")
            if open_gripper_position is not None
            else None
        )
        if self.place_enabled and opened is None:
            raise ValueError("place sequence requires open_gripper_position")
        state = self.state

        if state.stage in {SequenceStage.HIGH_XY_APPROACH, SequenceStage.MATCH_Z}:
            if not path_complete:
                waypoint_stage = stage_for_waypoint_label(active_waypoint_label)
                self.state = SequenceState(waypoint_stage, 0, state.stage_steps + 1, False)
                return self._make_command(
                    observation,
                    waypoint_stage,
                    waypoint,
                    self.desired_axis_b,
                    self._arm_action(observation, waypoint),
                    1.0,
                    True,
                    False,
                )
            if not self.grasp_enabled:
                self.state = SequenceState(SequenceStage.HOLD_COMPLETE, 0, 0, True)
            else:
                self.state = SequenceState(SequenceStage.ALIGN_NORMAL)
            state = self.state

        if state.stage is SequenceStage.ALIGN_NORMAL:
            position_ok = (
                _norm(tuple(a - b for a, b in zip(observation.tcp_position_b, self.preposition_b, strict=True)))
                <= self.config.position_tolerance_m
            )
            # 保持预停靠位置和 Flange-down，仅调整 TCP 的 XOY 朝向。
            heading_error = abs(wrap_angle_rad(
                self._grasp_heading_rad(observation)
                - planar_heading_from_quaternion(observation.tcp_quaternion_b_wxyz, self.tcp_heading_axis)
            ))
            dwell = self._qualified(position_ok and heading_error <= self.config.normal_angle_tolerance_rad)
            if dwell >= self.config.normal_dwell_steps:
                self.state = SequenceState(
                    SequenceStage.ADVANCE_NORMAL,
                    tcp_quaternion_ref_b_wxyz=_vector(
                        observation.tcp_quaternion_b_wxyz, size=4, name="tcp_quaternion_b_wxyz"
                    ),
                    tcp_heading_ref_rad=planar_heading_from_quaternion(
                        observation.tcp_quaternion_b_wxyz, self.tcp_heading_axis
                    ),
                )
                state = self.state
            else:
                self.state = replace(state, dwell_steps=dwell, stage_steps=state.stage_steps + 1)
                return self._make_command(
                    observation,
                    state.stage,
                    self.preposition_b,
                    self.desired_axis_b,
                    self._arm_action(observation, self.preposition_b),
                    1.0,
                    True,
                    False,
                )

        if state.stage is SequenceStage.ADVANCE_NORMAL:
            position_ok = (
                _norm(tuple(a - b for a, b in zip(observation.tcp_position_b, self.advance_target_b, strict=True)))
                <= self.config.position_tolerance_m
            )
            dwell = self._qualified(position_ok)
            if dwell >= self.config.normal_dwell_steps:
                self.state = SequenceState(
                    SequenceStage.CLOSE_GRIPPER,
                    tcp_quaternion_ref_b_wxyz=state.tcp_quaternion_ref_b_wxyz,
                    tcp_heading_ref_rad=state.tcp_heading_ref_rad,
                )
                state = self.state
            else:
                self.state = replace(state, dwell_steps=dwell, stage_steps=state.stage_steps + 1)
                return self._make_command(
                    observation,
                    state.stage,
                    self.advance_target_b,
                    self.desired_axis_b,
                    self._arm_action(observation, self.advance_target_b, state.tcp_quaternion_ref_b_wxyz),
                    1.0,
                    False,
                    False,
                )

        if state.stage is SequenceStage.CLOSE_GRIPPER:
            gripper_error = max(
                abs(actual - target) for actual, target in zip(observation.gripper_position, closed, strict=True)
            )
            dwell = self._qualified(gripper_error <= self.config.gripper_position_tolerance)
            if dwell >= self.config.gripper_dwell_steps:
                next_stage = SequenceStage.LIFT_OBJECT if self.place_enabled else SequenceStage.HOLD_COMPLETE
                self.state = SequenceState(
                    next_stage,
                    complete=not self.place_enabled,
                    tcp_quaternion_ref_b_wxyz=state.tcp_quaternion_ref_b_wxyz,
                    tcp_heading_ref_rad=state.tcp_heading_ref_rad,
                )
                state = self.state
            else:
                self.state = replace(state, dwell_steps=dwell, stage_steps=state.stage_steps + 1)
                return self._make_command(
                    observation,
                    state.stage,
                    self.advance_target_b,
                    self.desired_axis_b,
                    ZERO_ARM_ACTION,
                    -1.0,
                    False,
                    False,
                )

        if state.stage is SequenceStage.LIFT_OBJECT:
            assert self.lift_position_b is not None
            position_ok = (
                _norm(tuple(a - b for a, b in zip(observation.tcp_position_b, self.lift_position_b, strict=True)))
                <= self.config.position_tolerance_m
            )
            dwell = self._qualified(position_ok)
            if dwell >= self.config.normal_dwell_steps:
                self.state = SequenceState(
                    SequenceStage.PLACE_HIGH_XY,
                    tcp_quaternion_ref_b_wxyz=state.tcp_quaternion_ref_b_wxyz,
                    tcp_heading_ref_rad=state.tcp_heading_ref_rad,
                )
                state = self.state
            else:
                self.state = replace(state, dwell_steps=dwell, stage_steps=state.stage_steps + 1)
                return self._make_command(
                    observation,
                    state.stage,
                    self.lift_position_b,
                    self.desired_axis_b,
                    self._arm_action(observation, self.lift_position_b, state.tcp_quaternion_ref_b_wxyz),
                    -1.0,
                    False,
                    False,
                )

        if state.stage is SequenceStage.PLACE_HIGH_XY:
            assert self.place_high_position_b is not None and self.place_position_b is not None
            place_target = self.place_high_position_b
            position_ok = (
                _norm(tuple(a - b for a, b in zip(observation.tcp_position_b, place_target, strict=True)))
                <= self.config.position_tolerance_m
            )
            transfer_heading_error = abs(
                wrap_angle_rad(
                    self._place_transfer_heading(observation)
                    - planar_heading_from_quaternion(observation.tcp_quaternion_b_wxyz, self.tcp_heading_axis)
                )
            )
            dwell = self._qualified(position_ok and transfer_heading_error <= self.config.normal_angle_tolerance_rad)
            if dwell >= self.config.normal_dwell_steps:
                self.state = SequenceState(
                    SequenceStage.ALIGN_PLACE_DIRECTION,
                    tcp_quaternion_ref_b_wxyz=state.tcp_quaternion_ref_b_wxyz,
                    tcp_heading_ref_rad=state.tcp_heading_ref_rad,
                )
                state = self.state
            else:
                self.state = replace(state, dwell_steps=dwell, stage_steps=state.stage_steps + 1)
                return self._make_command(
                    observation,
                    state.stage,
                    place_target,
                    self.desired_axis_b,
                    self._arm_action(
                        observation,
                        place_target,
                        state.tcp_quaternion_ref_b_wxyz,
                    ),
                    -1.0,
                    False,
                    False,
                )
            if state.stage is SequenceStage.ALIGN_PLACE_DIRECTION:
                return self._make_command(
                    observation,
                    state.stage,
                    self.place_high_position_b,
                    self.desired_axis_b,
                    self._arm_action(
                        observation,
                        self.place_high_position_b,
                    ),
                    -1.0,
                    False,
                    False,
                )

        if state.stage is SequenceStage.PLACE_FINAL_Z:
            assert self.place_position_b is not None
            position_ok = (
                _norm(tuple(a - b for a, b in zip(observation.tcp_position_b, self.place_position_b, strict=True)))
                <= self.config.position_tolerance_m
            )
            dwell = self._qualified(position_ok and self._place_direction_ok(observation))
            if dwell >= self.config.normal_dwell_steps:
                self.state = SequenceState(
                    SequenceStage.PLACE_FINAL_XY,
                    tcp_quaternion_ref_b_wxyz=state.tcp_quaternion_ref_b_wxyz,
                    tcp_heading_ref_rad=state.tcp_heading_ref_rad,
                )
                return self._make_command(
                    observation,
                    SequenceStage.PLACE_FINAL_XY,
                    self.place_position_b,
                    self.desired_axis_b,
                    self._arm_action(
                        observation,
                        self.place_position_b,
                    ),
                    -1.0,
                    False,
                    False,
                )
            else:
                self.state = replace(state, dwell_steps=dwell, stage_steps=state.stage_steps + 1)
                return self._make_command(
                    observation,
                    state.stage,
                    self.place_position_b,
                    self.desired_axis_b,
                    self._arm_action(
                        observation,
                        self.place_position_b,
                    ),
                    -1.0,
                    False,
                    False,
                )

        if state.stage is SequenceStage.ALIGN_PLACE_DIRECTION:
            assert self.place_high_position_b is not None
            alignment_target = self.place_high_position_b
            position_ok = (
                _norm(tuple(a - b for a, b in zip(observation.tcp_position_b, alignment_target, strict=True)))
                <= self.config.position_tolerance_m
            )
            # 正式朝向判定全部在世界系 W 中进行：先将子 prim 的朝向轴（-X_T）
            # 经 q_BT、q_WB 转到 W，再与目标物配置计算出的世界方向比较。
            orientation_error = direction_error_rad(
                self._tcp_direction_w(observation),
                self._place_direction_w(observation),
            )
            dwell = self._qualified(position_ok and orientation_error <= self.config.normal_angle_tolerance_rad)
            if dwell >= self.config.normal_dwell_steps:
                self.state = SequenceState(
                    SequenceStage.PLACE_FINAL_Z,
                    tcp_quaternion_ref_b_wxyz=state.tcp_quaternion_ref_b_wxyz,
                    tcp_heading_ref_rad=state.tcp_heading_ref_rad,
                )
                return self._make_command(
                    observation,
                    SequenceStage.PLACE_FINAL_Z,
                    self.place_position_b,
                    self.desired_axis_b,
                    self._arm_action(
                        observation,
                        self.place_position_b,
                    ),
                    -1.0,
                    False,
                    False,
                )
            else:
                self.state = replace(state, dwell_steps=dwell, stage_steps=state.stage_steps + 1)
                return self._make_command(
                    observation,
                    state.stage,
                    alignment_target,
                    self.desired_axis_b,
                    self._arm_action(
                        observation,
                        alignment_target,
                    ),
                    -1.0,
                    False,
                    False,
                )

        if state.stage is SequenceStage.PLACE_FINAL_XY:
            assert self.place_position_b is not None
            position_ok = (
                _norm(tuple(a - b for a, b in zip(observation.tcp_position_b, self.place_position_b, strict=True)))
                <= self.config.position_tolerance_m
            )
            dwell = self._qualified(position_ok and self._place_direction_ok(observation))
            if dwell >= self.config.normal_dwell_steps:
                self.state = SequenceState(
                    SequenceStage.RELEASE_GRIPPER,
                    tcp_quaternion_ref_b_wxyz=state.tcp_quaternion_ref_b_wxyz,
                    tcp_heading_ref_rad=state.tcp_heading_ref_rad,
                )
                return self._make_command(
                    observation,
                    SequenceStage.RELEASE_GRIPPER,
                    self.place_position_b,
                    self.desired_axis_b,
                    ZERO_ARM_ACTION,
                    1.0,
                    False,
                    False,
                )
            else:
                self.state = replace(state, dwell_steps=dwell, stage_steps=state.stage_steps + 1)
                return self._make_command(
                    observation,
                    state.stage,
                    self.place_position_b,
                    self.desired_axis_b,
                    self._arm_action(
                        observation,
                        self.place_position_b,
                    ),
                    -1.0,
                    False,
                    False,
                )

        if state.stage is SequenceStage.RELEASE_GRIPPER:
            assert self.place_position_b is not None and opened is not None
            gripper_error = max(
                abs(actual - target) for actual, target in zip(observation.gripper_position, opened, strict=True)
            )
            dwell = self._qualified(gripper_error <= self.config.gripper_position_tolerance)
            if dwell >= self.config.gripper_dwell_steps:
                self.state = SequenceState(
                    SequenceStage.PLACE_COMPLETE,
                    complete=True,
                    tcp_quaternion_ref_b_wxyz=state.tcp_quaternion_ref_b_wxyz,
                    tcp_heading_ref_rad=state.tcp_heading_ref_rad,
                )
                state = self.state
            else:
                self.state = replace(state, dwell_steps=dwell, stage_steps=state.stage_steps + 1)
                return self._make_command(
                    observation,
                    state.stage,
                    self.place_position_b,
                    self.desired_axis_b,
                    ZERO_ARM_ACTION,
                    1.0,
                    False,
                    False,
                )

        if state.stage is SequenceStage.PLACE_COMPLETE:
            assert self.place_position_b is not None
            return self._make_command(
                observation,
                state.stage,
                self.place_position_b,
                self.desired_axis_b,
                self._arm_action(observation, self.place_position_b),
                1.0,
                False,
                True,
            )

        hold_target = self.advance_target_b if self.grasp_enabled else self.preposition_b
        return self._make_command(
            observation,
            SequenceStage.HOLD_COMPLETE,
            hold_target,
            self.desired_axis_b,
            self._arm_action(observation, hold_target, state.tcp_quaternion_ref_b_wxyz),
            -1.0 if self.grasp_enabled else 1.0,
            not self.grasp_enabled,
            True,
        )


__all__ = [
    "DatasetSequence",
    "SequenceCommand",
    "SequenceConfig",
    "SequenceObservation",
    "SequenceStage",
    "LEGACY_SEQUENCE_STAGE_ORDER_V3",
    "PLACE_TRANSFER_HEADING_FRACTION",
    "SEQUENCE_STAGE_ORDER",
    "SEQUENCE_STAGE_RANK",
    "approach_direction",
    "axis_angle_error_rad",
    "heading_to_normalized_action",
    "minimal_rotation_vector",
    "normalized_action_to_heading",
    "planar_heading_from_quaternion",
    "planar_heading_from_vector",
    "quaternion_rotate_vector",
    "quaternion_rotation_error_vector",
    "stage_for_waypoint_label",
    "stepped_heading_target",
    "wrap_angle_rad",
]
