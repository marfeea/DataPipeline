"""Collection-only bridge from measured Isaac state to the v4 contract."""

from __future__ import annotations

import numpy as np

from .alignment import matrix_pose, pose_matrix


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def xyzw_pose(position, quaternion_wxyz):
    q = array(quaternion_wxyz)
    return np.r_[array(position), q[1:], q[0]]


def relative_pose_in_axes(ee_pose, grasp_pose):
    """Full relative rotation and offset, expressed in the named parent axes.

    Equivalent to inv(T_parent_ee) @ T_parent_grasp, re-expressed from EE axes
    into the parent axes at the EE origin; no position-only approximation.
    """
    ee, grasp = pose_matrix(ee_pose), pose_matrix(grasp_pose)
    result = np.eye(4)
    result[:3, 3] = grasp[:3, 3] - ee[:3, 3]
    result[:3, :3] = grasp[:3, :3] @ ee[:3, :3].T
    return matrix_pose(result)


def aligned_snapshot(
    robot,
    controller,
    gripper,
    *,
    env_origin,
    source_state,
    goal_state,
    tcp_position_b,
    tcp_quaternion_b,
    grasp_position_b,
    grasp_quaternion_b,
):
    """Read actual TCP velocity at its origin, including the COM offset term."""
    base = pose_matrix(xyzw_pose(robot.data.root_pos_w[0], robot.data.root_quat_w[0]))
    tcp_b = xyzw_pose(tcp_position_b, tcp_quaternion_b)
    tcp_w = matrix_pose(base @ pose_matrix(tcp_b))
    grasp_b = xyzw_pose(grasp_position_b, grasp_quaternion_b)
    grasp_w = matrix_pose(base @ pose_matrix(grasp_b))
    object_w = pose_matrix(xyzw_pose(array(env_origin) + source_state.position_e, source_state.rotation_wxyz))
    goal_w = xyzw_pose(array(env_origin) + goal_state.position_e, goal_state.rotation_wxyz)
    body_id = controller.flange_body_id
    # Isaac body_com_vel_w is explicitly at the COM. Shift it to the TCP.
    com_position = array(robot.data.body_com_pos_w[0, body_id])
    com_twist = array(robot.data.body_com_vel_w[0, body_id])
    twist_w = np.r_[com_twist[:3] + np.cross(com_twist[3:], tcp_w[:3] - com_position), com_twist[3:]]
    rotation = base[:3, :3]
    twist_b = np.r_[rotation.T @ twist_w[:3], rotation.T @ twist_w[3:]]
    force = array(robot.root_physx_view.get_dof_projected_joint_forces()[0, gripper.joint_ids])
    return {
        "tcp_pose_b": tcp_b,
        "tcp_pose_w": tcp_w,
        "ee_twist_b": twist_b,
        "ee_twist_w": twist_w,
        "gripper_force": force,
        "T_w_b": base,
        "grasp_pose_o": matrix_pose(np.linalg.inv(object_w) @ pose_matrix(grasp_w)),
        "grasp_pose_b": grasp_b,
        "grasp_pose_w": grasp_w,
        "grasp_in_ee_pose_b": relative_pose_in_axes(tcp_b, grasp_b),
        "grasp_in_ee_pose_w": relative_pose_in_axes(tcp_w, grasp_w),
        "goal_object_pose_b": matrix_pose(np.linalg.inv(base) @ pose_matrix(goal_w)),
        "goal_object_pose_w": goal_w,
    }


def applied_action(controller, gripper_command, position_scale_m, heading_scale_rad):
    """FK of the final transmitted joint target, relative to policy-start FK.

    This represents the final servo substep target, not achieved motion. Values may
    exceed [-1,1] after IK/safety mapping; never clip the diagnostic retrospectively.
    """
    from CurriculumRL.logic.aubo_ik import forward_kinematics, tcp_heading_decomposition

    from configs.coordinate_frames import TCP_DIRECTION_AXIS_T

    start = array(controller.policy_start_joint_position[0])
    final = array(controller.commanded_joint_target[0])
    displacement = forward_kinematics(final)[:3, 3] - forward_kinematics(start)[:3, 3]
    heading = tcp_heading_decomposition(final, tcp_heading_axis_local=TCP_DIRECTION_AXIS_T).total_heading_rad
    return tuple(np.r_[displacement / position_scale_m, np.zeros(3), heading / heading_scale_rad, gripper_command])
