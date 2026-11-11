import sys
from pathlib import Path
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[1])]
from timed_reference import Settings,SmoothLine,ReferenceClock,JointTargetLimiter


def test_smooth_time_law_is_monotone_exact_and_does_not_overshoot():
    path=SmoothLine([1,2,3],[1.3,2.4,3],.25,.25)
    t=np.linspace(0,path.duration,10001)
    p=path.position(t)
    assert np.all(np.diff(path.progress(t))>=-1e-12)
    np.testing.assert_allclose(p[0],[1,2,3])
    np.testing.assert_allclose(p[-1],[1.3,2.4,3])
    assert np.max(np.linalg.norm(np.diff(p,axis=0),axis=1)/np.diff(t))<=.25+1e-9
    assert np.linalg.norm(p[1]-p[0])<1e-10


def test_short_move_uses_lower_peak_speed_instead_of_overshooting():
    path=SmoothLine([0,0,0],[0,0,.001],.25,.25)
    assert path.cruise==0
    assert path.speed==pytest.approx(.004)
    np.testing.assert_allclose(path.position(100),[0,0,.001])


class ToyReference:
    duration=10.
    def position(self,t): return np.array([0.,0.,t*.6])


def test_reference_progress_is_retained_and_each_step_is_bounded():
    clock=ReferenceClock(ToyReference(),Settings())
    actual=np.zeros(3)
    last=clock.reference.position(clock.time)
    for _ in range(4):
        info=clock.advance(actual)
        new=clock.reference.position(clock.time)
        assert np.linalg.norm(new-last)<=.01+1e-9
        last=new
        actual[2]+=.001
    assert new[2]>.03
    assert new[2]-actual[2]>.02


def test_tracking_governor_never_pulls_existing_reference_back():
    clock=ReferenceClock(ToyReference(),Settings())
    clock.time=.1
    old=clock.time
    info=clock.advance(np.zeros(3))
    assert clock.time>=old
    assert clock.time==pytest.approx(old,abs=1e-8)
    assert info["governor_active"]
    for _ in range(58): clock.advance(np.zeros(3))
    with pytest.raises(RuntimeError,match="stalled"):
        for _ in range(5): clock.advance(np.zeros(3))


def test_soft_threshold_predicts_next_reference_instead_of_waiting_one_cycle():
    clock=ReferenceClock(ToyReference(),Settings())
    clock.time=.06
    info=clock.advance(np.zeros(3))
    assert info["tracking_lead_before_m"]<.04
    assert info["predicted_tracking_lead_m"]>.04
    assert info["governor_active"]
    assert info["clock_rate"]<1


def test_joint_command_state_has_velocity_and_acceleration_bounds():
    limiter=JointTargetLimiter(np.zeros(6),np.ones(6)*2,np.ones(6)*5,1/120)
    last=limiter.q.copy()
    old_velocity=np.zeros(6)
    for _ in range(60):
        q,v,a,limited=limiter.step(np.ones(6))
        assert np.max(np.abs(q-last))<=2/120+1e-12
        assert np.max(np.abs((v-old_velocity)*120))<=5+1e-10
        assert np.all(q>=last)
        last,old_velocity=q,v
    assert np.all(last>0)


def test_invalid_timing_and_tracking_thresholds_are_rejected():
    with pytest.raises(ValueError): Settings(physics_hz=100)
    with pytest.raises(ValueError): Settings(soft_lead_m=.07)


@pytest.mark.parametrize("speed,ramp",[(.25,.25),(.35,.45),(.40,.80)])
def test_real_legacy_ik_reference_reaches_original_pose_with_joint_limits(speed,ramp):
    from timed_reference import plan_reference
    from scripts.native.aubo_ik import forward_kinematics
    start=np.array([-2.3734209537506104,-.5979837775230408,2.480827569961548,-.30727264285087585,1.5698314905166626,.8018686175346375])
    goal=np.array([-.23700020926873466,.2500003967767759,.5548898959054966])
    ref=plan_reference(start,goal,[1,0,0,0],Settings(speed_m_s=speed,ramp_s=ramp))
    np.testing.assert_allclose(ref.q(0),start,atol=1e-12)
    final=forward_kinematics(ref.q(ref.duration))
    np.testing.assert_allclose(final[:3,3],goal,atol=1e-6)
    np.testing.assert_allclose(final[:3,:3],np.eye(3),atol=1e-6)
    sample=np.linspace(0,ref.duration,2001)
    assert np.max(np.abs(ref.q(sample,2)))<=5+1e-5
    assert np.max(np.abs(ref.q(sample,3)))<=50+1e-4
    assert ref.metadata["interpolation_position_error_m"]<1e-4
