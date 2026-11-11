"""Physical position-servo lift driven by a precomputed Legacy IK reference."""
import json
from pathlib import Path
import time
import numpy as np

from timed_reference import Settings,plan_reference,ReferenceClock,JointTargetLimiter


def run_lift(*,env,robot,target,ik,gripper,arm_ids,cfg,task,args,records,report,lift,pick_quaternion,
             capture_sample,contacts,**unused):
    import torch
    from scripts.native.aubo_ik import forward_kinematics
    from scripts.pipelines.curobo_bootstrap.planner import TrajectoryData,JOINT_NAMES,save_trajectory
    from scripts.pipelines.curobo_bootstrap.robot_adapter import pose_matrix
    def array(value): return value.detach().cpu().numpy().astype(float)
    settings=Settings(speed_m_s=args.timed_lift_speed,ramp_s=args.timed_lift_ramp)
    started=time.monotonic()
    meta=dict(generator="Legacy IK timed Cartesian reference",settings=settings.__dict__,
              scope="isolated lift experiment",actual_state_writes_during_lift=False,
              velocity_feedforward=False,command_hz=settings.command_hz,physics_hz=args.physics_hz,
              requested_goal_tcp_pose_b=[*lift.tolist(),*pick_quaternion],
              training_configuration_changed=False)
    report["timed_lift_experiment"]=meta
    report["stage"]="legacy_timed_lift_planning"
    start_q=array(robot.data.joint_pos[0,arm_ids])
    meta["start_q"]=start_q.tolist()
    reference=plan_reference(start_q,lift,pick_quaternion,settings)
    meta.update(reference.metadata)
    meta["planning_wall_s"]=time.monotonic()-started
    meta["planned_duration_s"]=reference.duration
    root=Path(args.trajectory_output)
    root.mkdir(parents=True,exist_ok=True)
    times=np.arange(int(np.ceil(reference.duration*60))+1)/60
    q_ref=reference.q(times)
    save_trajectory(root/"planned_reference.npz",TrajectoryData(times,q_ref,reference.q(times,1),reference.q(times,2),JOINT_NAMES))
    np.savez_compressed(root/"reference_geometry.npz",time_s=times,tcp_position_b=reference.position(times),
                        start_q=start_q,goal_tcp_position_b=lift)
    print("TIMED_LIFT_PLAN "+json.dumps(meta),flush=True)
    clock=ReferenceClock(reference,settings)
    limiter=JointTargetLimiter(start_q,meta["velocity_limit_rad_s"],meta["acceleration_limit_rad_s2"],1/args.physics_hz)
    initial_count=len(records)
    execution_started=time.monotonic()
    history=[]
    goal_quaternion=np.asarray(pick_quaternion,float)
    goal_quaternion/=np.linalg.norm(goal_quaternion)
    goal_rotation=pose_matrix(lift,goal_quaternion)[:3,:3]

    def execute(stage):
        before=array(ik.current_tcp_pose_b()[0][0])
        clock_meta=clock.advance(before)
        contacts.begin_control_step()
        limited=False
        for substep in range(cfg.decimation):
            phase=clock_meta["previous_time_s"]+(clock.time-clock_meta["previous_time_s"])*(substep+1)/cfg.decimation
            desired=reference.q(phase)
            command,velocity,acceleration,was_limited=limiter.step(desired)
            limited|=was_limited
            if np.any(command<np.asarray(meta["position_lower_rad"])+.0017) or np.any(command>np.asarray(meta["position_upper_rad"])-.0017):
                raise RuntimeError("Continuous joint command reached position margin")
            history.append(dict(reference_time_s=phase,commanded_joint_target=command.tolist(),
                                command_velocity_rad_s=velocity.tolist(),command_acceleration_rad_s2=acceleration.tolist(),limited=was_limited))
            robot.set_joint_position_target(torch.as_tensor(command,dtype=robot.data.joint_pos.dtype,
                                                           device=robot.data.joint_pos.device)[None],joint_ids=arm_ids)
            gripper.apply(-1.)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.sim.get_physics_dt())
        env.sim.render()
        tcp,quaternion,_=ik.current_tcp_pose_b()
        p,q=array(tcp[0]),array(quaternion[0])
        row=dict(stage=stage,timed_lift=True,q=array(robot.data.joint_pos[0,arm_ids]).tolist(),
                 dq=array(robot.data.joint_vel[0,arm_ids]).tolist(),tcp_position_b=p.tolist(),tcp_quaternion_b=q.tolist(),
                 current_tcp_position_b=before.tolist(),reference_tcp_position_b=reference.position(clock.time).tolist(),
                 commanded_tcp_position_b=forward_kinematics(command)[:3,3].tolist(),
                 commanded_joint_target=command.tolist(),planned_joint_reference=reference.q(clock.time).tolist(),
                 gripper_position=list(gripper.state()[0]),target_position_w=array(target.data.root_pos_w[0]).tolist(),
                 command_hz=60.,physical_rate_limited=limited,physics_contacts=contacts.sample(),
                 execution_wall_s=time.monotonic()-execution_started,**clock_meta)
        records.append(row)
        capture_sample()
        rotation=pose_matrix(p,q/np.linalg.norm(q))[:3,:3]
        return float(np.linalg.norm(p-lift)),float(np.arccos(np.clip((np.trace(goal_rotation.T@rotation)-1)/2,-1,1)))

    report["stage"]="legacy_timed_lift_execution"
    try:
        for _ in range(60*30):
            position_error,rotation_error=execute("legacy_lift")
            if clock.time>=reference.duration-1e-10:
                break
        else:
            raise RuntimeError("Timed lift exceeded 30 s simulation budget")
        meta["reference_execution_frames"]=len(records)-initial_count
        dwell=0
        for _ in range(120):
            position_error,rotation_error=execute("legacy_lift")
            dwell=dwell+1 if position_error<=.01 and rotation_error<=np.deg2rad(1) else 0
            if dwell>=3: break
        else:
            raise RuntimeError(f"Timed lift endpoint did not settle: {position_error} m, {rotation_error} rad")
        lift_frames=len(records)-initial_count
        for _ in range(30): execute("legacy_lift_hold")
        meta["measured_lift_frames"]=lift_frames
        return dict(steps=lift_frames,final_tcp_error_m=position_error,final_tcp_orientation_error_rad=rotation_error)
    finally:
        meta["execution_wall_s"]=time.monotonic()-execution_started
        (root/"physical_commands.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
        (root/"timed_reference_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
