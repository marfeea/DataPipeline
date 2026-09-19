"""Test directed pick-place edges over the configured target states.

This is a kinematic preflight for the Isaac collection entry.  It deliberately
does not claim that a bottle was physically grasped: it checks that every
sampled TCP waypoint in a pick -> lift -> transfer -> place route has a valid
flange-down IK solution while using one common grasp height.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRICULUM_ROOT = PROJECT_ROOT.parent.parent / "CurriculumRL"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CURRICULUM_SOURCE = CURRICULUM_ROOT / "source" / "CurriculumRL"
if str(CURRICULUM_SOURCE) not in sys.path:
    sys.path.insert(0, str(CURRICULUM_SOURCE))

from CurriculumRL.configs.scene import (  # noqa: E402
    ARM_INITIAL_JOINT_POSITIONS,
    AUBO_LOCAL_POSITION_STATION,
    WORKSTATION_POSE_E,
)
from configs.task import TARGET_STATES  # noqa: E402
from CurriculumRL.logic.aubo_ik import (  # noqa: E402
    AUBO_E5_TCP_MODEL,
    forward_pose,
    solve_flange_down_ik,
    tcp_target_to_flange_down_position,
)
from CurriculumRL.logic.tcp_kinematics import rotate_vector, world_to_root_position  # noqa: E402


@dataclass(frozen=True)
class EdgeResult:
    source_index: int
    source_name: str
    destination_index: int
    destination_name: str
    grasp_height_m: float
    lift_height_m: float
    waypoint_count: int
    reachable: bool
    max_position_error_m: float | None
    max_orientation_error_rad: float | None
    max_joint_step_rad: float | None
    failure_phase: str | None
    failure_waypoint: int | None
    failure_reason: str | None


def _environment_position_to_base(position_e: tuple[float, float, float]) -> np.ndarray:
    """Match the collector's E -> W -> B transform for env 0."""

    root_quaternion = torch.as_tensor(WORKSTATION_POSE_E.rotation_wxyz, dtype=torch.float64).view(1, 4)
    workstation_position = torch.as_tensor(WORKSTATION_POSE_E.position, dtype=torch.float64)
    robot_local_position = torch.as_tensor(AUBO_LOCAL_POSITION_STATION, dtype=torch.float64).view(1, 3)
    root_position = workstation_position + rotate_vector(root_quaternion, robot_local_position)[0]
    position_world = torch.as_tensor(position_e, dtype=torch.float64).view(1, 3)
    return world_to_root_position(position_world, root_position.view(1, 3), root_quaternion)[0].numpy()


def _lerp(a: np.ndarray, b: np.ndarray, count: int) -> list[np.ndarray]:
    return [a + (b - a) * (index / count) for index in range(1, count + 1)]


def _state_xy(index: int, grasp_height_m: float, height_frame: str) -> tuple[np.ndarray, np.ndarray]:
    state = TARGET_STATES[index]
    # Keep the configured approach direction in XY and make the TCP height
    # identical for preposition and grasp, as requested for this preflight.
    if height_frame == "environment":
        grasp_e = (state.grasp_tcp_position_e[0], state.grasp_tcp_position_e[1], grasp_height_m)
        pre_e = (state.preposition_e[0], state.preposition_e[1], grasp_height_m)
        return _environment_position_to_base(pre_e), _environment_position_to_base(grasp_e)
    pre_b = _environment_position_to_base(state.preposition_e)
    grasp_b = _environment_position_to_base(state.grasp_tcp_position_e)
    pre_b[2] = grasp_height_m
    grasp_b[2] = grasp_height_m
    return pre_b, grasp_b


def _route_waypoints(
    source: int,
    destination: int,
    grasp_height_m: float,
    lift_height_m: float,
    samples: int,
    height_frame: str,
) -> list[tuple[str, np.ndarray]]:
    source_pre, source_grasp = _state_xy(source, grasp_height_m, height_frame)
    destination_pre, destination_grasp = _state_xy(destination, grasp_height_m, height_frame)
    source_lift = source_grasp.copy()
    source_lift[2] = lift_height_m
    destination_lift = destination_grasp.copy()
    destination_lift[2] = lift_height_m
    points = (
        ("approach_source", source_pre),
        ("grasp_source", source_grasp),
        ("lift_source", source_lift),
        ("transfer_high", destination_lift),
        ("place_destination", destination_grasp),
    )
    waypoints: list[tuple[str, np.ndarray]] = []
    start = np.asarray(
        forward_pose(
            [ARM_INITIAL_JOINT_POSITIONS[name] for name in ("Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Flange")],
            AUBO_E5_TCP_MODEL,
        )[0],
        dtype=np.float64,
    )
    previous = start
    for phase, target in points:
        for point in _lerp(previous, target, samples):
            waypoints.append((phase, point))
        previous = target
    return waypoints


def _test_edge(
    source: int,
    destination: int,
    grasp_height_m: float,
    lift_clearance_m: float,
    samples: int,
    height_frame: str,
) -> EdgeResult:
    lift_height_m = grasp_height_m + lift_clearance_m
    waypoints = _route_waypoints(source, destination, grasp_height_m, lift_height_m, samples, height_frame)
    q = np.asarray(
        [ARM_INITIAL_JOINT_POSITIONS[name] for name in ("Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Flange")],
        dtype=np.float64,
    )
    max_position_error = 0.0
    max_orientation_error = 0.0
    max_joint_step = 0.0
    failure_phase = None
    failure_waypoint = None
    failure_reason = None
    for waypoint_index, (phase, tcp_position) in enumerate(waypoints, start=1):
        try:
            flange_position = tcp_target_to_flange_down_position(tcp_position)
            result = solve_flange_down_ik(
                flange_position,
                seed=q,
                max_iterations=300,
                max_joint_step_rad=0.20,
                position_tolerance_m=1.0e-2,
                orientation_tolerance_rad=5.0e-3,
            )
        except Exception as exc:  # noqa: BLE001 - preserve the exact failing edge in the report
            failure_phase = phase
            failure_waypoint = waypoint_index
            failure_reason = f"{type(exc).__name__}: {exc}"
            break
        if not result.success or result.q is None:
            failure_phase = phase
            failure_waypoint = waypoint_index
            failure_reason = result.reason
            break
        q_next = np.asarray(result.q, dtype=np.float64)
        if not np.all(np.isfinite(q_next)):
            failure_phase = phase
            failure_waypoint = waypoint_index
            failure_reason = "IK produced NaN/Inf"
            break
        max_position_error = max(max_position_error, float(result.position_error_m))
        max_orientation_error = max(max_orientation_error, float(result.orientation_error_rad))
        max_joint_step = max(max_joint_step, float(np.linalg.norm(q_next - q)))
        q = q_next
    reachable = failure_phase is None
    return EdgeResult(
        source_index=source,
        source_name=TARGET_STATES[source].name,
        destination_index=destination,
        destination_name=TARGET_STATES[destination].name,
        grasp_height_m=grasp_height_m,
        lift_height_m=lift_height_m,
        waypoint_count=len(waypoints),
        reachable=reachable,
        max_position_error_m=max_position_error if reachable else None,
        max_orientation_error_rad=max_orientation_error if reachable else None,
        max_joint_step_rad=max_joint_step if reachable else None,
        failure_phase=failure_phase,
        failure_waypoint=failure_waypoint,
        failure_reason=failure_reason,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grasp-height", type=float, default=0.40)
    parser.add_argument("--height-frame", choices=("base", "environment"), default="base")
    parser.add_argument("--lift-clearance", type=float, default=0.10)
    parser.add_argument("--samples-per-segment", type=int, default=20)
    args = parser.parse_args()
    if not np.isfinite(args.grasp_height) or not np.isfinite(args.lift_clearance) or args.lift_clearance <= 0:
        raise ValueError("grasp-height must be finite and lift-clearance must be positive")
    if args.samples_per_segment <= 0:
        raise ValueError("samples-per-segment must be positive")

    results = [
        _test_edge(
            source,
            destination,
            args.grasp_height,
            args.lift_clearance,
            args.samples_per_segment,
            args.height_frame,
        )
        for source in range(len(TARGET_STATES))
        for destination in range(len(TARGET_STATES))
        if source != destination
    ]
    payload = {
        "test": "kinematic_pick_place_directed_graph",
        "configured_state_count": len(TARGET_STATES),
        "configured_state_names": [state.name for state in TARGET_STATES],
        "self_edges_skipped": True,
        "grasp_height_m": args.grasp_height,
        "height_frame": args.height_frame,
        "lift_clearance_m": args.lift_clearance,
        "samples_per_segment": args.samples_per_segment,
        "edge_count": len(results),
        "reachable_edge_count": sum(result.reachable for result in results),
        "results": [asdict(result) for result in results],
        "note": "Kinematic IK preflight only; physical grasp attachment and camera/HDF5 collection require Isaac runtime smoke.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    print(json.dumps({key: payload[key] for key in ("configured_state_count", "edge_count", "reachable_edge_count", "grasp_height_m", "lift_clearance_m")}, ensure_ascii=False))
    for result in results:
        status = "REACHABLE" if result.reachable else "UNREACHABLE"
        suffix = "" if result.reachable else f" phase={result.failure_phase} reason={result.failure_reason}"
        print(f"{status} {result.source_index}->{result.destination_index} {result.source_name} -> {result.destination_name}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
