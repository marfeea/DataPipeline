import numpy as np

from scripts.pipelines.curobo_bootstrap.contact_speed import tracked_tcp_target


def test_command_advances_despite_slow_servo_response_and_remains_bounded():
    current = np.zeros(3)
    goal = np.array([0.12, 0, 0])
    command = current.copy()
    for _ in range(120):
        command = tracked_tcp_target(command, current, goal, speed_m_s=0.03, control_hz=60, step_limit_m=0.001)
        assert np.linalg.norm(command - current) <= 0.02 + 1e-12
        current += 0.08 * (command - current)
    # Re-anchoring a 0.5 mm command to actual position would cover only 4.8 mm.
    assert current[0] > 0.04


def test_stalled_robot_cannot_accumulate_an_unbounded_target():
    actual = np.zeros(3)
    command = actual.copy()
    for _ in range(100):
        command = tracked_tcp_target(command, actual, [1, 0, 0], speed_m_s=0.05,
                                     control_hz=60, step_limit_m=0.001)
    np.testing.assert_allclose(command, [0.02, 0, 0], atol=1e-12, rtol=0)
