import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("lift_speed_policy", Path(__file__).with_name("policy.py"))
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def test_diagonal_displacement_has_five_cm_total_limit():
    delta = policy.bounded_lift_delta([0, 0, 0], [0.3, 0.4, 0])
    np.testing.assert_allclose(delta, [0.03, 0.04, 0])
    assert np.isclose(np.linalg.norm(delta) * 4, 0.2)


def test_final_short_move_and_stationary_target_do_not_overshoot():
    np.testing.assert_allclose(policy.bounded_lift_delta([0, 0, 0.4], [0, 0, 0.405]), [0, 0, 0.005])
    np.testing.assert_array_equal(policy.bounded_lift_delta([1, 2, 3], [1, 2, 3]), [0, 0, 0])


def test_reset_uses_robot_adapter_joint_indices_and_zero_velocity():
    from types import SimpleNamespace
    import torch
    runner_spec = importlib.util.spec_from_file_location("lift_sweep", Path(__file__).with_name("run_sweep.py"))
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    initial = torch.arange(8, dtype=torch.float32).view(1, 8)
    calls = []

    def write(q, velocity):
        assert torch.equal(q, initial)
        assert torch.count_nonzero(velocity) == 0
        calls.append("reset")

    def target(q, joint_ids):
        assert joint_ids == list(range(8)) and torch.equal(q, initial)
        calls.append("target")

    owner = SimpleNamespace(robot=SimpleNamespace(write_joint_state_to_sim=write, set_joint_position_target=target),
                            scene=SimpleNamespace(write_data_to_sim=lambda: None, update=lambda dt: None),
                            sim=SimpleNamespace(forward=lambda: None, render=lambda: calls.append("render")))
    runner.reset_robot(owner, initial, 1 / 120)
    assert calls == ["reset", "target", "render"]
