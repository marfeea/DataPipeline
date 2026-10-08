"""P2 real-physics smoke; requires isaaclab, does not require CuRobo."""

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from DataPipe.runtime import ensure_curriculumrl_available

ensure_curriculumrl_available()
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

import CurriculumRL.tasks  # noqa: F401
import gymnasium as gym
import torch
from CurriculumRL.tasks.tcp_docking.trajectory_env_cfg import TcpDockingTrajectoryEnvCfg


def main():
    cfg = TcpDockingTrajectoryEnvCfg()
    cfg.scene.num_envs = 2
    cfg.sim.device = args.device
    cfg.seed = 7
    cfg.actions.joint_trajectory.open_position = (0.0115, -0.0115)
    # Empty-gripper smoke preset only; task-specific grasp calibration is P3/P8.
    cfg.actions.joint_trajectory.closed_position = (0.0, 0.0)
    env = gym.make("CurriculumRL-TcpDocking-Trajectory-v0", cfg=cfg)
    base = env.unwrapped
    try:
        obs, _ = env.reset()
        controller = base.action_manager.get_term("joint_trajectory").controller
        robot = controller.robot
        q0 = robot.data.joint_pos[:, controller.arm_ids].clone()
        assert base.action_manager.total_action_dim == 7
        assert not hasattr(base, "termination_manager")
        assert torch.isfinite(obs["state"]["joint_position"]).all()
        max_sent_error = 0.0
        observation_count = 1
        phase_errors = {}
        # Precomputed fixed trajectory, no changes based on observed tracking.
        commands = []
        for phase in range(3):
            for tick in range(120):
                command = torch.cat((q0.clone(), torch.ones((2, 1), device=base.device)), dim=1)
                if phase == 0:
                    command[:, 0] += 0.02 * min((tick + 1) / 60, 1.0)
                elif phase == 1:
                    command[:, 0] += 0.02
                    command[:, 6] = -1
                else:
                    command[:, 0] += 0.02 * max(1.0 - (tick + 1) / 60, 0.0)
                commands.append(command)
        for index, command in enumerate(commands):
            obs, _ = env.step(command)
            observation_count += 1
            sent = robot.data.joint_pos_target[:, controller.arm_ids]
            max_sent_error = max(max_sent_error, float((sent - command[:, :6]).abs().max()))
            assert torch.equal(sent, command[:, :6]), "environment changed the planned target"
            assert torch.isfinite(obs["state"]["joint_position"]).all()
            if (index + 1) % 120 == 0:
                arm_error = float((robot.data.joint_pos[:, controller.arm_ids] - command[:, :6]).abs().max())
                finger_error = float(
                    (robot.data.joint_pos[:, controller.gripper_ids] - controller.gripper_target).abs().max()
                )
                phase_errors[str(index // 120)] = {"arm_max_rad": arm_error, "gripper_max_m": finger_error}
                assert arm_error < 0.02, phase_errors
                assert finger_error < 0.003, phase_errors
        cached = controller.raw_action.clone()
        bad = cached.clone()
        bad[0, 0] = float("nan")
        try:
            env.step(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("NaN target was accepted")
        assert torch.equal(controller.raw_action, cached)
        # Explicit partial reset must not reset the other environment's target.
        env.reset(env_ids=torch.tensor([0], device=base.device))
        assert torch.equal(controller.raw_action[1], cached[1])
        assert torch.equal(controller.arm_target[0], robot.data.joint_pos[0, controller.arm_ids])
        report = {
            "passed": True,
            "transitions": len(commands),
            "observations": observation_count,
            "num_envs": 2,
            "control_hz": 1 / base.step_dt,
            "max_sent_error_rad": max_sent_error,
            "phase_errors": phase_errors,
            "nan_rejected": True,
            "partial_reset_passed": True,
            "python": sys.executable,
            "scope": "manual trajectory and empty gripper; not CuRobo or pick-place",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report), flush=True)
    finally:
        env.close()


try:
    main()
except Exception:
    traceback.print_exc()
    raise
finally:
    launcher.app.close()
