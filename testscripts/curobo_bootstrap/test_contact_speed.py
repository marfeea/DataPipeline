import numpy as np
import pytest

from scripts.pipelines.curobo_bootstrap import contact_speed
from scripts.pipelines.curobo_bootstrap.planner import JOINT_NAMES


def test_reference_excludes_legacy_motion_and_segment_seams(tmp_path, monkeypatch):
    def linear_fk(q):
        pose = np.eye(4)
        pose[0, 3] = q[0]
        return pose

    monkeypatch.setattr(contact_speed, "forward_kinematics", linear_fk)
    q = np.zeros((7, 6))
    q[:, 0] = [0, 1, 4, 8, 20, 30, 31]
    path = tmp_path / "trajectory.npz"
    np.savez(path, time_s=np.arange(7), joint_position=q, joint_names=JOINT_NAMES,
             segment_source=["curobo", "curobo", "legacy_ik", "legacy_ik", "curobo", "curobo", "curobo"],
             stage_id=["pick", "pick", "pick", "lift", "transfer", "transfer", "transfer"])
    reference = contact_speed.curobo_tcp_speed_reference(path)
    assert reference["interval_count"] == 3
    assert reference["median_m_s"] == 1
    assert reference["max_m_s"] == 10


def test_tcp_step_uses_linear_speed_magnitude_and_preserves_direction():
    delta = contact_speed.tcp_step_toward([0, 0, 0], [0.3, 0.4, 0],
                                         speed_m_s=0.6, control_hz=60, step_limit_m=0.05)
    np.testing.assert_allclose(delta, [0.006, 0.008, 0])
    assert np.linalg.norm(delta) == pytest.approx(0.01)


def test_tcp_step_obeys_cap_and_does_not_overshoot():
    kwargs = dict(speed_m_s=6, control_hz=60, step_limit_m=0.05)
    np.testing.assert_allclose(contact_speed.tcp_step_toward([0, 0, 0], [0.3, 0.4, 0], **kwargs), [0.03, 0.04, 0])
    np.testing.assert_allclose(contact_speed.tcp_step_toward([0, 0, 0], [0.001, 0, 0], **kwargs), [0.001, 0, 0])


def test_stationary_reference_cannot_set_zero_contact_speed(tmp_path):
    path = tmp_path / "trajectory.npz"
    np.savez(path, time_s=[0, 1], joint_position=np.zeros((2, 6)), joint_names=JOINT_NAMES)
    with pytest.raises(ValueError, match="positive"):
        contact_speed.curobo_tcp_speed_reference(path)
