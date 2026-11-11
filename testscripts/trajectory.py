"""与 Isaac 解耦的 IK TCP 轨迹规划内核。

该模块复现当前静态 IK 测试的路径语义，便于测试入口和采集入口共享。
它只生成路径点，不调用求解器、不访问场景、不写文件。
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

Vector3 = tuple[float, float, float]
Waypoint = tuple[str, Vector3]


@dataclass(frozen=True, slots=True)
class TrajectoryPlan:
    """确定性 TCP 轨迹计划，供静态 IK 验证和 Isaac 采集共同消费。"""

    planning_waypoints: tuple[Waypoint, ...]
    control_waypoints: tuple[Waypoint, ...]
    planning_hz: float
    control_hz: float
    interpolation_substeps: int
    xy_fraction: float

    @property
    def duration_s(self) -> float:
        return len(self.planning_waypoints) / self.planning_hz


def _vector(value: Iterable[float]) -> Vector3:
    result = tuple(float(item) for item in value)
    if len(result) != 3 or not all(math.isfinite(item) for item in result):
        raise ValueError("位置必须是三个有限浮点数")
    return result  # type: ignore[return-value]


def _lerp(start: Vector3, end: Vector3, ratio: float) -> Vector3:
    return tuple(a + ratio * (b - a) for a, b in zip(start, end, strict=True))  # type: ignore[return-value]


def resolve_interpolation_substeps(planning_hz: float, control_hz: float) -> int:
    """解析粗规划频率到控制频率的整数展开倍数。"""

    if planning_hz <= 0.0 or control_hz <= 0.0:
        raise ValueError("planning_hz 和 control_hz 必须为正数")
    ratio = control_hz / planning_hz
    substeps = int(round(ratio))
    if substeps < 1 or not math.isclose(ratio, substeps, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("control_hz 必须是 planning_hz 的正整数倍")
    return substeps


def physics_steps_for_control(control_hz: float, physics_dt_s: float) -> int:
    """解析每个控制点对应的物理步数。"""

    if control_hz <= 0.0 or physics_dt_s <= 0.0:
        raise ValueError("control_hz 和 physics_dt_s 必须为正数")
    expected = 1.0 / (control_hz * physics_dt_s)
    steps = int(round(expected))
    if steps < 1 or not math.isclose(expected, steps, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("control_hz 必须映射为整数个物理步")
    return steps


def build_waypoints(
    start_position: Iterable[float],
    target_position: Iterable[float],
    waypoint_count: int,
    *,
    trajectory_mode: str = "staged",
    high_clearance_z: float | None = None,
    xy_fraction: float = 0.5,
    base_clockwise_deg: float = 0.0,
) -> tuple[Waypoint, ...]:
    """生成 direct 或高位横移后分段下降的 TCP B 系路径。"""

    start = _vector(start_position)
    target = _vector(target_position)
    if waypoint_count <= 0:
        raise ValueError("waypoint_count 必须为正数")
    if not 0.0 < xy_fraction < 1.0:
        raise ValueError("xy_fraction 必须位于 (0, 1)")
    if trajectory_mode == "direct":
        return tuple(("direct", _lerp(start, target, index / waypoint_count)) for index in range(1, waypoint_count + 1))
    if trajectory_mode != "staged":
        raise ValueError(f"未知 trajectory_mode：{trajectory_mode}")

    high_z = start[2] if high_clearance_z is None else float(high_clearance_z)
    if high_z < target[2]:
        raise ValueError("high_clearance_z 不能低于目标 z")
    high = (target[0], target[1], high_z)
    if abs(base_clockwise_deg) > 1.0e-9:
        base_route_count = max(3, min(waypoint_count - 3, int(round(xy_fraction * waypoint_count))))
        alignment_count = max(2, int(round(0.10 * waypoint_count)))
        settle_count = max(2, int(round(0.10 * waypoint_count)))
        descent_count = waypoint_count - base_route_count - alignment_count - settle_count
        if descent_count < 1:
            raise ValueError("waypoint_count is too small for staged base route and descent")
        swing_offset = min(0.30, max(0.12, 0.18 * abs(base_clockwise_deg) / 45.0))
        swing = (min(start[0], target[0]) - swing_offset, start[1], high_z)
        arc = (swing[0], 0.5 * (start[1] + target[1]), high_z)
        segment = base_route_count // 3
        route_points = (
            ("base_swing", start, swing, segment),
            ("base_arc", swing, arc, segment),
            ("base_high_xy", arc, high, base_route_count - 2 * segment),
        )
        approach_list = [
            (phase, _lerp(segment_start, segment_end, index / count))
            for phase, segment_start, segment_end, count in route_points
            for index in range(1, count + 1)
        ]
        approach_list.extend(("high_pose_alignment", high) for _ in range(alignment_count))
        approach = tuple(approach_list)
    else:
        approach_count = max(1, min(waypoint_count - 1, int(round(xy_fraction * waypoint_count))))
        descent_count = waypoint_count - approach_count
        approach = tuple(
            ("high_xy_approach", _lerp(start, high, index / approach_count)) for index in range(1, approach_count + 1)
        )
    descent_list = [
        ("vertical_descent", _lerp(high, target, index / descent_count)) for index in range(1, descent_count + 1)
    ]
    if abs(base_clockwise_deg) > 1.0e-9:
        descent_list.extend(("target_settle", target) for _ in range(max(2, int(round(0.10 * waypoint_count)))))
    return approach + tuple(descent_list)


def interpolate_waypoints(
    start_position: Iterable[float], planning_waypoints: Iterable[Waypoint], substeps: int
) -> tuple[Waypoint, ...]:
    """将粗规划点展开为控制频率路径点。"""

    if substeps <= 0:
        raise ValueError("substeps 必须为正数")
    previous = _vector(start_position)
    result: list[Waypoint] = []
    for phase, target in planning_waypoints:
        target_vector = _vector(target)
        for substep in range(1, substeps + 1):
            result.append(
                (
                    phase if substep == substeps else f"{phase}_interp",
                    _lerp(previous, target_vector, substep / substeps),
                )
            )
        previous = target_vector
    return tuple(result)


def advance_reached_waypoints(
    waypoints: Iterable[Waypoint],
    current_index: int,
    actual_position: Iterable[float],
    tolerance_m: float,
) -> tuple[int, bool]:
    """按实际 TCP 到达状态推进 waypoint，并在终点持续保持。

    返回的索引始终可用于选取当前目标；第二个值表示最终 waypoint 已经
    到达。密集插值点若同时落入容差会一次跳过，未到达的点绝不会仅因
    控制时钟推进而被跳过。
    """

    points = tuple(waypoints)
    if not points:
        raise ValueError("waypoints 不能为空")
    if not 0 <= current_index < len(points):
        raise ValueError("current_index 超出 waypoint 范围")
    if tolerance_m <= 0.0 or not math.isfinite(tolerance_m):
        raise ValueError("tolerance_m 必须是有限正数")
    actual = _vector(actual_position)
    index = current_index
    while index < len(points) - 1:
        target = points[index][1]
        error = math.sqrt(sum((target[axis] - actual[axis]) ** 2 for axis in range(3)))
        if error > tolerance_m:
            return index, False
        index += 1
    final = points[-1][1]
    final_error = math.sqrt(sum((final[axis] - actual[axis]) ** 2 for axis in range(3)))
    return index, final_error <= tolerance_m


def make_plan(
    start_position: Iterable[float],
    target_position: Iterable[float],
    waypoint_count: int,
    *,
    planning_hz: float,
    control_hz: float,
    trajectory_mode: str = "staged",
    high_clearance_z: float | None = None,
    xy_fraction: float = 0.5,
    base_clockwise_deg: float = 0.0,
) -> TrajectoryPlan:
    """构造唯一的粗规划与控制频率 TCP 轨迹。"""

    substeps = resolve_interpolation_substeps(planning_hz, control_hz)
    planning = build_waypoints(
        start_position,
        target_position,
        waypoint_count,
        trajectory_mode=trajectory_mode,
        high_clearance_z=high_clearance_z,
        xy_fraction=xy_fraction,
        base_clockwise_deg=base_clockwise_deg,
    )
    return TrajectoryPlan(
        planning_waypoints=planning,
        control_waypoints=interpolate_waypoints(start_position, planning, substeps),
        planning_hz=float(planning_hz),
        control_hz=float(control_hz),
        interpolation_substeps=substeps,
        xy_fraction=float(xy_fraction),
    )


__all__ = [
    "TrajectoryPlan",
    "Waypoint",
    "advance_reached_waypoints",
    "build_waypoints",
    "interpolate_waypoints",
    "make_plan",
    "physics_steps_for_control",
    "resolve_interpolation_substeps",
]
