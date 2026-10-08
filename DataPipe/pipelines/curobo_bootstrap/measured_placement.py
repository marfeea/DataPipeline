"""Place the measured grasp at the original object goal without changing that goal."""

import numpy as np

from .replay import matrix_pose_wxyz
from .robot_adapter import pose_matrix, rigid_transform


def _pose(value):
    values = np.asarray(value, dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("measured placement requires a finite xyz+wxyz pose")
    norm = np.linalg.norm(values[3:])
    if not np.isclose(norm, 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("measured placement quaternion must be unit length")
    return pose_matrix(values[:3], values[3:] / norm)


def measured_state4_placement(task, *, lift_tcp_pose_b, object_pose_w_after_lift):
    """Calibrate TCP position and rotation from this run's measured rigid grasp.

    Contact offsets still refer to the original nominal TCP waypoints. Applying
    those offsets to the desired object root keeps its final world pose exact.
    No collision or physical execution acceptance is implied by this geometry.
    """
    if task["destination_index"] != 4 or "state4_routing" not in task:
        raise ValueError("measured placement is only defined for routed destination state4")
    world_from_base = rigid_transform(task["frames"]["T_WB"])
    world_from_environment = rigid_transform(task["frames"]["T_WE"])
    base_from_tcp_lift = _pose(lift_tcp_pose_b)
    world_from_object_lift = _pose(object_pose_w_after_lift)
    tcp_from_object = np.linalg.inv(world_from_base @ base_from_tcp_lift) @ world_from_object_lift
    desired = task["destination"]
    base_from_object_goal = (
        np.linalg.inv(world_from_base)
        @ world_from_environment
        @ pose_matrix(desired["position_e"], desired["rotation_wxyz"])
    )
    geometry = measured_tcp_waypoints_b(
        tcp_pose_b=lift_tcp_pose_b,
        object_pose_b=matrix_pose_wxyz(np.linalg.inv(world_from_base) @ world_from_object_lift),
        goal_object_pose_b=matrix_pose_wxyz(base_from_object_goal),
        nominal_final_position_b=task["waypoint_position_b"]["place"],
        contact_points_b=task["contact_waypoint_position_b"],
    )
    object_poses = {
        name: list(matrix_pose_wxyz(world_from_base @ _pose(pose)))
        for name, pose in geometry["expected_object_waypoint_pose_b"].items()
    }
    return dict(
        schema="state4_measured_grasp_placement/1",
        lift_tcp_pose_b=list(lift_tcp_pose_b),
        object_pose_w_after_lift=list(object_pose_w_after_lift),
        tcp_from_object=rigid_transform(tcp_from_object).tolist(),
        desired_object_pose_w=list(matrix_pose_wxyz(world_from_base @ base_from_object_goal)),
        tcp_waypoint_pose_b=geometry["tcp_waypoint_pose_b"],
        expected_object_waypoint_pose_w=object_poses,
    )


def measured_tcp_waypoints_b(
    *, tcp_pose_b, object_pose_b, goal_object_pose_b, nominal_final_position_b, contact_points_b
):
    """Frame-local counterpart used by the physical collection state machine."""
    tcp_from_object = np.linalg.inv(_pose(tcp_pose_b)) @ _pose(object_pose_b)
    goal = _pose(goal_object_pose_b)
    final = np.asarray(nominal_final_position_b, dtype=np.float64)
    if final.shape != (3,) or not np.isfinite(final).all():
        raise ValueError("nominal final TCP must be a finite 3D position")
    points = dict(contact_points_b, place=final.tolist())
    tcp_poses, object_poses = {}, {}
    for name, point in points.items():
        if name.startswith("pick_"):
            continue
        point = np.asarray(point, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("contact waypoints must be finite 3D positions")
        object_waypoint = goal.copy()
        object_waypoint[:3, 3] += point - final
        tcp_poses[name] = list(matrix_pose_wxyz(object_waypoint @ np.linalg.inv(tcp_from_object)))
        object_poses[name] = list(matrix_pose_wxyz(object_waypoint))
    return dict(
        tcp_from_object=rigid_transform(tcp_from_object).tolist(),
        tcp_waypoint_pose_b=tcp_poses,
        expected_object_waypoint_pose_b=object_poses,
    )


def require_measured_calibration(task, saved, *, lift_tcp_pose_b, object_pose_w_after_lift):
    """Reject a changed goal or calibration that does not match measured inputs."""
    expected = measured_state4_placement(
        task, lift_tcp_pose_b=lift_tcp_pose_b, object_pose_w_after_lift=object_pose_w_after_lift
    )

    def matches(first, second):
        if isinstance(first, dict):
            return (
                isinstance(second, dict)
                and first.keys() == second.keys()
                and all(matches(value, second[name]) for name, value in first.items())
            )
        if isinstance(first, str):
            return first == second
        try:
            actual = np.asarray(second, dtype=np.float64)
            reference = np.asarray(first, dtype=np.float64)
        except (TypeError, ValueError):
            return False
        return actual.shape == reference.shape and np.allclose(actual, reference, atol=1e-12, rtol=0)

    if not matches(expected, saved):
        raise ValueError("state4 placement calibration differs from measured grasp or original object goal")
    return expected


def executed_endpoint_poses(task, report):
    """Use the measured-grasp TCP goal while retaining historical endpoint references."""
    poses = {name: list(value["tcp_pose_b"]) for name, value in report["legacy_ik_targets"].items()}
    if task.get("state4_routing", {}).get("place_measured_grasp") and task["destination_index"] == 4:
        expected = require_measured_calibration(
            task,
            report["placement_calibration"],
            lift_tcp_pose_b=report["measured"]["lift_tcp_pose_b"],
            object_pose_w_after_lift=report["measured"]["target_pose_w_after_lift"],
        )
        poses["place"] = expected["tcp_waypoint_pose_b"]["place"]
        saved = report["executed_waypoint_tcp_pose_b"]["place"]
        if not np.allclose(saved, poses["place"], atol=1e-12, rtol=0):
            raise ValueError("executed endpoint differs from calibrated TCP goal")
    return poses
