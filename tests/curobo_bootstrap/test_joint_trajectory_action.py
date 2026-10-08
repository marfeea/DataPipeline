"""CPU checks of the actual runtime controller with only Isaac managers stubbed."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from DataPipe.runtime import ensure_curriculumrl_available

torch = pytest.importorskip("torch")


@pytest.fixture
def controller_class(monkeypatch):
    ensure_curriculumrl_available()
    import CurriculumRL.configs.assets as assets

    managers = ModuleType("isaaclab.managers")
    managers.ActionTerm = type("ActionTerm", (), {})
    managers.ActionTermCfg = type("ActionTermCfg", (), {})
    utils = ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", utils)
    path = Path(assets.__file__).parents[1] / "tasks/tcp_docking/joint_trajectory_action.py"
    spec = importlib.util.spec_from_file_location("CurriculumRL.tasks.tcp_docking._trajectory_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.JointTrajectoryAction


class Robot:
    def __init__(self):
        self.data = SimpleNamespace(
            joint_pos=torch.zeros(2, 8),
            joint_pos_limits=torch.tensor([-2.0, 2.0]).expand(2, 8, 2).clone(),
        )
        self.sent = []

    def find_joints(self, names, preserve_order):
        return (list(range(6)) if len(names) == 6 else [6, 7]), names

    def set_joint_position_target(self, target, joint_ids):
        self.sent.append((target.clone(), joint_ids))


def make_controller(cls):
    robot = Robot()
    return robot, cls(robot, open_position=(0.01, -0.01), closed_position=(0, 0))


def test_absolute_targets_hold_without_interpolation(controller_class):
    robot, controller = make_controller(controller_class)
    action = torch.tensor([[1.5] * 6 + [1.0], [-1.5] * 6 + [-1.0]])
    controller.process_actions(action)
    action.zero_()  # Cache must own its storage.
    for _ in range(3):
        robot.data.joint_pos.add_(0.01)
        controller.apply_actions()
    for arm, gripper in zip(robot.sent[::2], robot.sent[1::2]):
        assert torch.equal(arm[0], torch.tensor([[1.5] * 6, [-1.5] * 6]))
        assert torch.allclose(gripper[0], torch.tensor([[0.01, -0.01], [0, 0]]))


@pytest.mark.parametrize("kind", ["nan", "limit", "gripper", "shape", "dtype"])
def test_rejected_batch_does_not_change_targets(controller_class, kind):
    _, controller = make_controller(controller_class)
    valid = torch.zeros(2, 7)
    valid[:, 6] = 1
    controller.process_actions(valid)
    bad = valid.clone()
    if kind == "nan":
        bad[1, 0] = float("nan")
    elif kind == "limit":
        bad[1, 2] = 3
    elif kind == "gripper":
        bad[1, 6] = 0
    elif kind == "shape":
        bad = bad[:, :6]
    else:
        bad = bad.double()
    with pytest.raises(ValueError):
        controller.process_actions(bad)
    assert torch.equal(controller.raw_action, valid)


def test_partial_reset_holds_measured_joints_only_for_selected_env(controller_class):
    robot, controller = make_controller(controller_class)
    command = torch.ones(2, 7)
    controller.process_actions(command)
    robot.data.joint_pos[0] = 0.25
    controller.reset(torch.tensor([0]))
    assert torch.equal(controller.arm_target[0], torch.full((6,), 0.25))
    assert torch.equal(controller.gripper_target[0], torch.full((2,), 0.25))
    assert torch.equal(controller.arm_target[1], torch.ones(6))


def test_gripper_presets_checked_in_every_environment(controller_class):
    robot = Robot()
    robot.data.joint_pos_limits[1, 6, 1] = 0.005
    with pytest.raises(ValueError):
        controller_class(robot, open_position=(0.01, -0.01), closed_position=(0, 0))
