"""Exercise native robot state, gripper and IK before expensive image collection."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DataPipe.sim_backend import add_launcher_args, ik_class, launch, trajectory_classes

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--cameras", action="store_true")
add_launcher_args(parser)
args = parser.parse_args()
launcher = launch(args)
import torch

from DataPipe.isaac_gripper import DatasetGripperController

Env, Cfg = trajectory_classes()
cfg = Cfg()
cfg.sim.device = args.device
if args.cameras:
    cfg.camera_resolution = (640, 480)
env = None
report = {"passed": False}
try:
    env = Env(cfg)
    env.reset()
    from pxr import UsdGeom

    report["stage_meters_per_unit"] = UsdGeom.GetStageMetersPerUnit(env.scene.stage)
    robot = env.scene["AUBObot"]
    report["joint_names"] = robot.view.dof_names
    report["joint_limits"] = robot.data.joint_pos_limits.tolist()
    gripper = DatasetGripperController(robot, open_position=(0.0115, -0.0115), closed_position=(0, 0))
    ik = ik_class()(robot, cfg.sim.dt, cfg.decimation, diagnostics_enabled=True, flange_downward_enabled=True)
    p, q, _ = ik.current_tcp_pose_b()
    report["initial_tcp_pose_b"] = [*p[0].tolist(), *q[0].tolist()]
    if args.cameras:
        report["hand_camera_before"] = env.cameras["HandCamera"].data.pos_w[0].tolist()
        report["render_reference_before"] = env.cameras["HandCamera"].render_reference()
    ik.process_actions(
        torch.zeros((1, 6), device=args.device),
        action_mask=torch.ones((1, 6), device=args.device),
        ik_modes=("full_pose_6d",),
    )
    for _ in range(cfg.decimation):
        ik.apply_actions()
        gripper.apply(1.0)
        env.sim.step(render=False)
        env.scene.update(cfg.sim.dt)
    report["joint_position"] = robot.data.joint_pos.tolist()
    if args.cameras:
        env.sim.render()
        report["hand_camera_after"] = env.cameras["HandCamera"].data.pos_w[0].tolist()
        report["render_reference_after"] = env.cameras["HandCamera"].render_reference()
    report["passed"] = not bool(ik.ik_failed[0])
except Exception as error:
    import traceback

    report.update(error=str(error), traceback=traceback.format_exc())
finally:
    report["isaaclab_modules_loaded"] = [n for n in sys.modules if n == "isaaclab" or n.startswith("isaaclab.")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if env is not None:
        env.close()
    launcher.app.close()
