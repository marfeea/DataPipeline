"""Legacy IK Cartesian path with smooth time law and an independent reference clock."""
from dataclasses import dataclass
import hashlib
import math
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class Settings:
    command_hz: float = 60.
    physics_hz: float = 120.
    speed_m_s: float = .25
    ramp_s: float = .25
    step_cap_m: float = .01
    soft_lead_m: float = .04
    hard_lead_m: float = .06
    hold_s: float = .5
    stuck_timeout_s: float = 1.

    def __post_init__(self):
        if not all(np.isfinite(v) and v>0 for v in self.__dict__.values()):
            raise ValueError("Settings must be finite and positive")
        if not self.soft_lead_m < self.hard_lead_m or self.physics_hz != 2*self.command_hz:
            raise ValueError("Expected 120/60 Hz timing and distinct lead thresholds")


class SmoothLine:
    def __init__(self, start, goal, speed, ramp):
        self.start, self.goal = np.asarray(start,float), np.asarray(goal,float)
        self.distance = float(np.linalg.norm(self.goal-self.start))
        if self.distance <= 0 or speed<=0 or ramp<=0:
            raise ValueError("Nonzero line, positive speed/ramp required")
        self.speed, self.ramp = min(speed,self.distance/ramp), ramp
        self.cruise = max(0.,self.distance/self.speed-ramp)
        self.duration = 2*ramp+self.cruise

    @staticmethod
    def integral(u):
        return 2.5*u**4-3*u**5+u**6

    def progress(self, time):
        t=np.clip(np.asarray(time,float),0,self.duration)
        r,v,c=self.ramp,self.speed,self.cruise
        distance=np.where(t<r,v*r*self.integral(np.clip(t/r,0,1)),
                          np.where(t<=r+c,v*r*.5+v*(t-r),
                                   self.distance-v*r*self.integral(np.clip((self.duration-t)/r,0,1))))
        return np.clip(distance/self.distance,0,1)

    def position(self,time):
        return self.start+np.asarray(self.progress(time))[...,None]*(self.goal-self.start)


class JointReference:
    def __init__(self, spline, line, scale, metadata):
        self.spline,self.line,self.scale,self.metadata=spline,line,scale,metadata
        self.duration=line.duration*scale

    def q(self,time,derivative=0):
        t=np.asarray(time,float)
        value=self.spline(np.clip(t,0,self.duration)/self.scale,nu=derivative)/self.scale**derivative
        if derivative:
            value=np.where(((t<0)|(t>self.duration))[...,None],0,value)
        return value

    def position(self,time):
        return self.line.position(np.asarray(time)/self.scale)


def derivative_metrics(spline,duration,scale,settings):
    t=np.linspace(0,duration,max(2,math.ceil(duration*scale*settings.physics_hz)+1))
    return dict(velocity=np.max(np.abs(spline(t,nu=1)),axis=0)/scale,
                acceleration=np.max(np.abs(spline(t,nu=2)),axis=0)/scale**2,
                jerk=np.max(np.abs(spline(t,nu=3)),axis=0)/scale**3)


def plan_reference(start_q,goal,goal_quaternion,settings):
    from configs.aubo_ik_control import JOINT_TARGET_MARGIN_RAD
    from scripts.native.aubo_ik import forward_kinematics,solve_ik
    from scripts.pipelines.curobo_bootstrap.planner import read_urdf_joint_limits
    start_q=np.asarray(start_q,float)
    pose=forward_kinematics(start_q)
    line=SmoothLine(pose[:3,3],goal,settings.speed_m_s,settings.ramp_s)
    times=np.linspace(0,line.duration,math.ceil(line.duration*settings.command_hz)+1)
    rotations=Rotation.from_matrix(pose[:3,:3])
    goal_quaternion=np.asarray(goal_quaternion,float)
    goal_quaternion=goal_quaternion/np.linalg.norm(goal_quaternion)
    end_rotation=Rotation.from_quat(np.roll(goal_quaternion,-1))
    slerp=Slerp([0.,1.],Rotation.from_quat(np.stack((rotations.as_quat(),end_rotation.as_quat()))))
    q=[start_q.copy()]
    max_iterations=0
    for t in times[1:]:
        quaternion=np.roll(slerp(float(line.progress(t))).as_quat(),1)
        answer=solve_ik(line.position(t),quaternion,seed=q[-1],max_iterations=200,
                        position_tolerance_m=1e-8,orientation_tolerance_rad=1e-7)
        if not answer.success:
            raise RuntimeError(f"Legacy IK path failed at t={t}: {answer.reason}")
        q.append(np.asarray(answer.q))
        max_iterations=max(max_iterations,answer.iterations)
    q=np.asarray(q)
    spline=CubicSpline(times,q,axis=0,bc_type=((1,np.zeros(6)),(1,np.zeros(6))))
    # URDF-derived speed values, matching the prior CuRobo comparison.
    from pathlib import Path
    urdf=Path(__file__).resolve().parents[2]/"configs/curobo/assets/aubo_e5_project/aubo_e5_grasp_tcp_20261015.urdf"
    lower,upper,velocity=read_urdf_joint_limits(urdf)
    acceleration=np.full(6,5.)
    jerk=np.full(6,50.)
    scale=1.
    for _ in range(12):
        metrics=derivative_metrics(spline,line.duration,scale,settings)
        ratio=max(1.,np.max(metrics["velocity"]/velocity),np.sqrt(np.max(metrics["acceleration"]/acceleration)),
                  np.cbrt(np.max(metrics["jerk"]/jerk)),settings.speed_m_s/(settings.step_cap_m*settings.command_hz))
        if ratio<=1.000001:
            break
        scale*=ratio*1.015
    else:
        raise RuntimeError("Joint retiming did not converge")
    dense=spline(np.linspace(0,line.duration,math.ceil(line.duration*scale*settings.physics_hz)+1))
    if np.any(dense<lower+JOINT_TARGET_MARGIN_RAD) or np.any(dense>upper-JOINT_TARGET_MARGIN_RAD):
        raise RuntimeError("Preplanned joint reference exceeds position limits")
    dense_tcp=np.asarray([forward_kinematics(x)[:3,3] for x in dense])
    expected=line.position(np.linspace(0,line.duration,len(dense)))
    error=float(np.max(np.linalg.norm(dense_tcp-expected,axis=1)))
    if error>.001:
        raise RuntimeError(f"Interpolated IK path deviates by {error} m")
    metadata=dict(generator="Legacy IK solve_ik + smooth Cartesian time law; no CuRobo planning",
                  nominal_duration_s=line.duration,time_scale=scale,duration_s=line.duration*scale,urdf=str(urdf),
                  urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),solver_position_tolerance_m=1e-8,
                  solver_orientation_tolerance_rad=1e-7,solver_max_iterations=200,
                  samples=len(times),maximum_solver_iterations=int(max_iterations),
                  interpolation_position_error_m=error,settings=settings.__dict__,
                  velocity_limit_rad_s=velocity.tolist(),acceleration_limit_rad_s2=acceleration.tolist(),
                  jerk_limit_rad_s3=jerk.tolist(),position_lower_rad=lower.tolist(),position_upper_rad=upper.tolist(),
                  planned_max_velocity_rad_s=metrics["velocity"].tolist(),
                  planned_max_acceleration_rad_s2=metrics["acceleration"].tolist(),planned_max_jerk_rad_s3=metrics["jerk"].tolist())
    return JointReference(spline,line,scale,metadata)


class ReferenceClock:
    def __init__(self,reference,settings):
        self.reference,self.settings=reference,settings
        self.time=0.
        self.rate=1.
        self.stuck=0.

    def advance(self,actual):
        cfg=self.settings
        dt=1/cfg.command_hz
        old=self.time
        lead=float(np.linalg.norm(self.reference.position(old)-actual))
        predicted_lead=float(np.linalg.norm(self.reference.position(min(self.reference.duration,old+dt))-actual))
        requested=float(np.clip((cfg.hard_lead_m-predicted_lead)/(cfg.hard_lead_m-cfg.soft_lead_m),0,1))
        self.rate=float(np.clip(requested,self.rate-4*dt,self.rate+4*dt))
        candidate=min(self.reference.duration,old+self.rate*dt)
        origin=self.reference.position(old)
        def allowed(t):
            p=self.reference.position(t)
            return np.linalg.norm(p-actual)<=cfg.hard_lead_m+1e-10 and np.linalg.norm(p-origin)<=cfg.step_cap_m+1e-10
        if not allowed(candidate):
            low,high=old,candidate
            for _ in range(32):
                mid=(low+high)/2
                if allowed(mid): low=mid
                else: high=mid
            candidate=low
        self.time=candidate
        self.stuck=self.stuck+dt if self.time<=old+1e-10 and old<self.reference.duration else 0.
        if self.stuck>cfg.stuck_timeout_s:
            raise RuntimeError("Reference clock stalled at tracking lead limit")
        return dict(previous_time_s=old,reference_time_s=self.time,clock_rate=(self.time-old)/dt,
                    tracking_lead_before_m=lead,predicted_tracking_lead_m=predicted_lead,
                    governor_active=predicted_lead>cfg.soft_lead_m)


class JointTargetLimiter:
    """Continuous command state; never reset from measured robot position."""
    def __init__(self,q,velocity_limit,acceleration_limit,dt):
        self.q=np.asarray(q,float).copy()
        self.velocity=np.zeros_like(self.q)
        self.vmax=np.asarray(velocity_limit,float)
        self.amax=np.asarray(acceleration_limit,float)
        self.dt=dt

    def step(self,desired):
        wanted=(np.asarray(desired)-self.q)/self.dt
        bounded=np.clip(wanted,-self.vmax,self.vmax)
        new_velocity=np.clip(bounded,self.velocity-self.amax*self.dt,self.velocity+self.amax*self.dt)
        limited=bool(np.max(np.abs(new_velocity-wanted))>1e-7)
        acceleration=(new_velocity-self.velocity)/self.dt
        self.q=self.q+new_velocity*self.dt
        self.velocity=new_velocity
        return self.q.copy(),new_velocity.copy(),acceleration,limited
