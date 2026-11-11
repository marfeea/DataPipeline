"""CuRobo planning and physical position-servo execution, with no state writes."""
from dataclasses import replace
import itertools
import json
from pathlib import Path

import numpy as np


def line_deviation(points, start, end):
    points, start, end = (np.asarray(x, dtype=float) for x in (points, start, end))
    direction = end - start
    progress = np.clip((points - start) @ direction / (direction @ direction), 0, 1)
    return np.linalg.norm(points - (start + progress[:, None] * direction), axis=1)


def normalized_pose(position, quaternion):
    from scripts.pipelines.curobo_bootstrap.robot_adapter import pose_matrix
    quaternion = np.asarray(quaternion, dtype=float)
    return pose_matrix(position, quaternion / np.linalg.norm(quaternion))


def attached_cover(vertices_base, *, base_pose_w, object_pose_reset, object_pose_current, tcp_pose_b, slots=4):
    """Cover the measured TCP-frame object AABB with a union of smaller spheres."""
    from scripts.pipelines.curobo_bootstrap.world_adapter import AttachedSphere
    def pose(values):
        return normalized_pose(values[:3], values[3:])
    wb = pose(base_pose_w)
    transform = np.linalg.inv(pose(tcp_pose_b)) @ np.linalg.inv(wb) @ pose(object_pose_current) @ np.linalg.inv(pose(object_pose_reset)) @ wb
    vertices = (transform @ np.column_stack((vertices_base, np.ones(len(vertices_base)))).T).T[:, :3]
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    allocations = [np.array(x) for x in itertools.product(range(1,slots+1),repeat=3) if np.prod(x)==slots]
    allocation = min(allocations, key=lambda x: np.linalg.norm((high-low)/x))
    half = (high-low)/allocation/2
    radius = float(np.linalg.norm(half))
    spheres = []
    for index in itertools.product(*(range(int(n)) for n in allocation)):
        center = low + (np.asarray(index)+.5)*(high-low)/allocation
        spheres.append(AttachedSphere(tuple(center), radius))
    if not np.all(np.min([np.linalg.norm(vertices - sphere.center_tcp,axis=1)-sphere.radius_m for sphere in spheres],axis=0) <= 1e-9):
        raise RuntimeError("Attachment cover does not enclose measured vertices")
    return tuple(spheres)


def line_axis_rotation(start, goal, goal_rotation):
    """Fixed virtual TCP rotation: its goal-frame Z axis follows the exact lift line."""
    z = np.asarray(goal_rotation).T @ (np.asarray(goal)-start)
    z /= np.linalg.norm(z)
    helper = np.array([0.,1.,0.]) if abs(z[1]) < .9 else np.array([1.,0.,0.])
    x = np.cross(helper,z)
    x /= np.linalg.norm(x)
    return np.column_stack((x,np.cross(z,x),z))


def run_lift(*, env, robot, target, ik, gripper, arm_ids, cfg, task, args, records, report,
             lift, pick_quaternion, actual_world, frozen_world, capture_sample, contacts):
    import torch
    import trimesh
    from configs.curobo.planner import PlannerConfig
    from curobo.geom.types import WorldConfig as CuroboWorldConfig
    from curobo.rollout.cost.pose_cost import PoseCostMetric
    from curobo.types.math import Pose
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
    from scripts.native.aubo_ik import forward_kinematics, solve_ik
    from scripts.pipelines.curobo_bootstrap.attachment_geometry import attached_sphere_from_measured_target
    from scripts.pipelines.curobo_bootstrap.planner import (
        CuroboMvpPlanner, TrajectoryData, JOINT_NAMES, EXPERIMENTAL_ACCELERATION_LIMIT,
        save_trajectory, validate_trajectory,
    )
    from scripts.pipelines.curobo_bootstrap.world_adapter import WorldSnapshot, curobo_world_dict

    def array(value):
        return value.detach().cpu().numpy().astype(float)

    trajectory_root = Path(args.trajectory_output)
    trajectory_root.mkdir(parents=True, exist_ok=True)
    start_q = array(robot.data.joint_pos[0, arm_ids])
    start_pose = forward_kinematics(start_q)
    goal_q = start_q.copy()
    # IK is used only to identify the same endpoint/branch for the joint-space query.
    for fraction in np.linspace(0, 1, 45)[1:]:
        result = solve_ik(start_pose[:3, 3] * (1 - fraction) + lift * fraction, pick_quaternion,
                          seed=goal_q, max_iterations=400)
        if not result.success:
            raise RuntimeError(f"Endpoint IK branch continuation failed: {result.reason}")
        goal_q = np.asarray(result.q)
    meta = dict(planning_mode=args.curobo_lift_mode, command_hz=60, physics_hz=args.physics_hz,
                actual_state_writes_during_lift=False, velocity_feedforward=False,
                target_anchored_to="time-indexed planned joint reference, not measured TCP",
                time_scale=args.curobo_time_scale, start_q=start_q.tolist(), endpoint_q=goal_q.tolist(),
                start_tcp_pose_b=[*start_pose[:3, 3].tolist(), *array(ik.current_tcp_pose_b()[1][0]).tolist()],
                requested_goal_tcp_pose_b=[*lift.tolist(), *pick_quaternion])
    report["curobo_lift_experiment"] = meta
    report["stage"] = "curobo_lift_planning"
    target_mesh = next(mesh for mesh in actual_world.config.meshes if mesh.name == actual_world.config.target_name)
    sphere = attached_sphere_from_measured_target(
        np.asarray(trimesh.load_mesh(target_mesh.file_path, process=False).vertices),
        base_pose_w=report["base_pose_w"], object_pose_w_at_reset=report["target_pose_w_at_reset"],
        object_pose_w_current=report["target_pose_w_at_pick"], tcp_pose_b_current=meta["start_tcp_pose_b"],
    )
    meta["attachment"] = dict(center_tcp=sphere.center_tcp, radius_m=sphere.radius_m)
    spheres = attached_cover(
        np.asarray(trimesh.load_mesh(target_mesh.file_path, process=False).vertices),
        base_pose_w=report["base_pose_w"], object_pose_reset=report["target_pose_w_at_reset"],
        object_pose_current=report["target_pose_w_at_pick"], tcp_pose_b=meta["start_tcp_pose_b"],
    )
    meta["attachment_spheres"] = [dict(center_tcp=s.center_tcp,radius_m=s.radius_m) for s in spheres]
    meta["attachment_representation"] = "four-sphere conservative AABB cover; original single bounding sphere retained for comparison"
    from scripts.pipelines.curobo_bootstrap import planner as planner_module
    from scripts.pipelines.curobo_bootstrap.replay import matrix_pose_wxyz
    original_robot_dict = planner_module.planning_robot_dict
    axis_rotation = None
    if args.curobo_lift_mode == "line":
        axis_rotation = line_axis_rotation(start_pose[:3,3],lift,normalized_pose(lift,pick_quaternion)[:3,:3])
        fixed = np.eye(4)
        fixed[:3,:3] = axis_rotation
        axis_pose = matrix_pose_wxyz(fixed)
        def virtual_robot_dict(asset):
            payload = original_robot_dict(asset)
            payload["kinematics"]["extra_links"]["lift_axis_tcp"] = dict(
                parent_link_name="project_tcp",link_name="lift_axis_tcp",fixed_transform=axis_pose,
                joint_type="FIXED",joint_name="lift_axis_joint")
            payload["kinematics"]["ee_link"] = "lift_axis_tcp"
            return payload
        planner_module.planning_robot_dict = virtual_robot_dict
        meta["virtual_tcp_axis_rotation"] = axis_rotation.tolist()
        meta["virtual_tcp_changes_physical_goal"] = False
    try:
        planner = CuroboMvpPlanner(frozen_world["collision_model"], world=actual_world.permit_target_contact(),
                                   config=PlannerConfig(version="0.7.8", cuda_graph_enabled=False,
                                                        graph_first=False, max_attempts=6, timeout_s=30))
    finally:
        planner_module.planning_robot_dict = original_robot_dict
    planner.attach(actual_world.attach(spheres))
    meta["runtime_limits"] = planner.runtime_limits
    meta["full_world_endpoint_checks"] = planner.endpoint_checks(start_q, goal_q)
    checks = meta["full_world_endpoint_checks"]
    if not all(checks[key] for key in ("within_position_limits", "world_collision_free", "self_collision_free")):
        # Explicit diagnostic scope: the same known proxy conflict as the Legacy IK trials.
        if not checks["within_position_limits"] or not checks["self_collision_free"]:
            raise RuntimeError(f"Full-world planning endpoints are invalid: {checks}")
        diagnostic = WorldSnapshot(replace(actual_world.config, meshes=tuple(
            mesh for mesh in actual_world.config.meshes if mesh.name != "support_tray_11"))).attach(spheres)
        planner.motion_gen.update_world(CuroboWorldConfig.from_dict(curobo_world_dict(diagnostic)))
        planner.collision.update_world(diagnostic)
        planner.world = diagnostic
        meta["excluded_planning_proxy_meshes"] = ["support_tray_11"]
        meta["physical_scene_changed"] = False
        meta["diagnostic_endpoint_checks"] = planner.endpoint_checks(start_q, goal_q)
        if not all(meta["diagnostic_endpoint_checks"][key] for key in
                   ("within_position_limits", "world_collision_free", "self_collision_free")):
            raise RuntimeError(f"Diagnostic endpoints remain invalid: {meta['diagnostic_endpoint_checks']}")
    else:
        meta["excluded_planning_proxy_meshes"] = []
    print("CUROBO_LIFT_ENDPOINTS " + json.dumps(meta), flush=True)
    if args.curobo_lift_mode == "joint":
        trajectory, planning = planner.plan(start_q, goal_q)
    else:
        start = JointState.from_position(planner.tensor_args.to_device(start_q).unsqueeze(0), joint_names=list(JOINT_NAMES))
        virtual_goal = normalized_pose(lift,pick_quaternion)
        if axis_rotation is not None:
            virtual_goal[:3,:3] = virtual_goal[:3,:3] @ axis_rotation
        goal = Pose(position=planner.tensor_args.to_device(lift).unsqueeze(0),
                    quaternion=planner.tensor_args.to_device(matrix_pose_wxyz(virtual_goal)[3:]).unsqueeze(0))
        weights = [1,1,1,1,1,0] if axis_rotation is not None else [1,1,1,0,1,0]
        metric = PoseCostMetric(hold_partial_pose=True,
                               hold_vec_weight=planner.tensor_args.to_device(weights),
                               project_to_goal_frame=axis_rotation is not None)
        result = planner.motion_gen.plan_single(start, goal, MotionGenPlanConfig(
            max_attempts=6, timeout=30, enable_graph=False, enable_graph_attempt=3,
            pose_cost_metric=metric, check_start_validity=True))
        planning = dict(success=bool(result.success is not None and result.success.item()), status=str(result.status),
                        valid_query=bool(result.valid_query), solve_time_s=float(result.solve_time),
                        interpolation_dt_s=float(result.interpolation_dt), constraint_weights=weights)
        meta["motion_gen"] = planning
        if not planning["success"]:
            raise RuntimeError(f"Pose-constrained CuRobo lift failed: {planning}")
        path = result.get_interpolated_plan()
        q, dq, ddq = [array(getattr(path, name)) for name in ("position", "velocity", "acceleration")]
        if q.ndim == 3:
            q, dq, ddq = q[0], dq[0], ddq[0]
        trajectory = TrajectoryData(np.arange(len(q)) * float(result.interpolation_dt), q, dq, ddq, JOINT_NAMES)
        goal_q = trajectory.joint_position[-1]
    meta["motion_gen"] = planning
    save_trajectory(trajectory_root / "curobo_planned.npz", trajectory)
    meta["plan_validation"] = validate_trajectory(
        trajectory, start_q=start_q, goal_q=goal_q, position_lower=planner.position_lower,
        position_upper=planner.position_upper, velocity_limit=planner.velocity_limit,
        acceleration_limit=EXPERIMENTAL_ACCELERATION_LIMIT,
        collision_query=lambda q: planner.collision.query(q, world_ignore_links=planner.world_collision_ignore_links))
    if not meta["plan_validation"]["passed"]:
        raise RuntimeError(f"CuRobo plan validation failed: {meta['plan_validation']}")
    planned_tcp = np.asarray([forward_kinematics(q) for q in trajectory.joint_position])
    planned_speed = np.linalg.norm(np.diff(planned_tcp[:, :3, 3], axis=0), axis=1) * 60
    meta.update(planned_duration_s=float(trajectory.time_s[-1]), planned_tcp_peak_speed_m_s=float(max(planned_speed)),
                planned_tcp_median_speed_m_s=float(np.median(planned_speed)),
                planned_max_line_deviation_m=float(max(line_deviation(planned_tcp[:, :3, 3], start_pose[:3, 3], lift))),
                planned_max_orientation_deviation_deg=float(max(np.rad2deg(np.arccos(np.clip(
                    (np.trace(start_pose[:3, :3].T @ pose[:3, :3]) - 1) / 2, -1, 1))) for pose in planned_tcp)))
    times = np.arange(int(np.ceil(trajectory.time_s[-1] * args.curobo_time_scale * 60)) + 1) / 60
    q_ref = np.stack([np.interp(times / args.curobo_time_scale, trajectory.time_s, column)
                      for column in trajectory.joint_position.T], axis=1)
    save_trajectory(trajectory_root / "execution_reference.npz", TrajectoryData(
        times, q_ref, np.gradient(q_ref, 1 / 60, axis=0), np.gradient(np.gradient(q_ref, 1 / 60, axis=0), 1 / 60, axis=0), JOINT_NAMES))
    print("CUROBO_LIFT_PLAN " + json.dumps(meta), flush=True)
    last_command = start_q.copy()
    last_reference = start_q.copy()
    initial_count = len(records)

    def execute(q, stage, reference_index):
        nonlocal last_command, last_reference
        previous = last_reference.copy()
        contacts.begin_control_step()
        before = array(ik.current_tcp_pose_b()[0][0])
        rate_limited = False
        for substep in range(cfg.decimation):
            candidate = previous + (q - previous) * ((substep + 1) / cfg.decimation)
            bounded = last_command + np.clip(candidate - last_command, -3.14 / args.physics_hz, 3.14 / args.physics_hz)
            rate_limited |= bool(np.max(np.abs(bounded - candidate)) > 1e-6)
            robot.set_joint_position_target(torch.as_tensor(bounded, dtype=robot.data.joint_pos.dtype,
                                                           device=robot.data.joint_pos.device)[None], joint_ids=arm_ids)
            last_command = bounded
            gripper.apply(-1.0)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.sim.get_physics_dt())
        last_reference = q.copy()
        env.sim.render()
        tcp, quat, _ = ik.current_tcp_pose_b()
        record = dict(stage=stage, curobo_physical=True, q=array(robot.data.joint_pos[0, arm_ids]).tolist(),
                      dq=array(robot.data.joint_vel[0, arm_ids]).tolist(), tcp_position_b=array(tcp[0]).tolist(),
                      tcp_quaternion_b=array(quat[0]).tolist(), current_tcp_position_b=before.tolist(),
                      gripper_position=list(gripper.state()[0]), target_position_w=array(target.data.root_pos_w[0]).tolist(),
                      commanded_joint_target=last_command.tolist(), planned_joint_reference=q.tolist(),
                      reference_tcp_position_b=forward_kinematics(q)[:3, 3].tolist(),
                      reference_index=reference_index, command_hz=60, physical_rate_limited=rate_limited,
                      physics_contacts=contacts.sample())
        records.append(record)
        capture_sample()
        position_error = float(np.linalg.norm(array(tcp[0]) - lift))
        pose_now = normalized_pose(array(tcp[0]), array(quat[0]))
        goal_pose = normalized_pose(lift, pick_quaternion)
        rotation_error = float(np.arccos(np.clip((np.trace(goal_pose[:3, :3].T @ pose_now[:3, :3]) - 1) / 2, -1, 1)))
        return position_error, rotation_error

    report["stage"] = "curobo_lift_execution"
    for index, q in enumerate(q_ref[1:], 1):
        position_error, rotation_error = execute(q, "legacy_lift", index)
    meta["reference_execution_frames"] = len(records) - initial_count
    dwell = 0
    for _ in range(120):
        position_error, rotation_error = execute(q_ref[-1], "legacy_lift", len(q_ref) - 1)
        dwell = dwell + 1 if position_error <= 0.01 and rotation_error <= np.deg2rad(1) else 0
        if dwell >= 3:
            break
    else:
        raise RuntimeError(f"Physical CuRobo endpoint did not settle: position={position_error}, rotation={rotation_error}")
    lift_frames = len(records) - initial_count
    for _ in range(30):
        execute(q_ref[-1], "legacy_lift_hold", len(q_ref) - 1)
    meta["measured_lift_frames"] = lift_frames
    return dict(steps=lift_frames, final_tcp_error_m=position_error,
                final_tcp_orientation_error_rad=rotation_error)
