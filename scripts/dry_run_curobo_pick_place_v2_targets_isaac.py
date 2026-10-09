"""Extract and audit the fixed 0->1 CuRobo V2 task targets after a production reset."""

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.task import TARGET_STATES
from DataPipe.pipelines.curobo_bootstrap.task_targets import LIFT_CLEARANCE_M, PickPlaceV2Config
from DataPipe.sim_backend import add_launcher_args, launch, trajectory_classes
from DataPipe.source2_routing import add_source2_routing_args, source2_routing_from_args
from DataPipe.state4_routing import add_state4_routing_args, routing_from_args, routing_payload

parser = argparse.ArgumentParser(
    description=__doc__,
    epilog="TARGET_STATES: " + ", ".join(f"{i}={state.name}" for i, state in enumerate(TARGET_STATES)),
)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--source-index", type=int, default=0)
parser.add_argument("--destination-index", type=int, default=1)
parser.add_argument("--lift-clearance-m", type=float, default=LIFT_CLEARANCE_M)
add_launcher_args(parser)
add_state4_routing_args(parser)
add_source2_routing_args(parser)
args = parser.parse_args()
if not 0 <= args.source_index < len(TARGET_STATES) or not 0 <= args.destination_index < len(TARGET_STATES):
    parser.error(f"source-index and destination-index must be in 0..{len(TARGET_STATES) - 1}")
if args.source_index == args.destination_index:
    parser.error("source-index and destination-index must differ")
try:
    selected_config = PickPlaceV2Config(
        source_index=args.source_index,
        destination_index=args.destination_index,
        lift_clearance_m=args.lift_clearance_m,
        state4_routing=routing_from_args(args),
        source2_routing=source2_routing_from_args(args),
    )
except ValueError as error:
    parser.error(str(error))
if args.output.exists():
    raise FileExistsError(f"Refusing to overwrite dry-run report: {args.output}")
launcher = launch(args)

import numpy as np

TcpDockingTrajectoryEnv, TcpDockingTrajectoryEnvCfg = trajectory_classes()

from DataPipe.pipelines.curobo_bootstrap.source_target import set_source_target_pose
from DataPipe.pipelines.curobo_bootstrap.task_targets import (
    PickPlaceV2Config,
    solve_legacy_joint_targets,
    task_snapshot,
)


def _array(value):
    return value.detach().cpu().numpy().astype(np.float64)


def main():
    report = {
        "passed": False,
        "stage": "startup",
        "python": sys.executable,
        "source_index": args.source_index,
        "destination_index": args.destination_index,
        "source_name": TARGET_STATES[args.source_index].name,
        "destination_name": TARGET_STATES[args.destination_index].name,
    }
    env = None
    try:
        config = selected_config
        cfg = TcpDockingTrajectoryEnvCfg()
        cfg.scene.num_envs = 1
        cfg.sim.device = args.device
        cfg.seed = config.seed
        cfg.actions.joint_trajectory.open_position = (0.0115, -0.0115)
        cfg.actions.joint_trajectory.closed_position = (0.0, 0.0)
        env = TcpDockingTrajectoryEnv(cfg)
        env.reset()
        controller = env.action_manager.get_term("joint_trajectory").controller
        robot = controller.robot
        q_start = _array(robot.data.joint_pos[0, controller.arm_ids])
        base_position = _array(robot.data.root_pos_w[0])
        base_quaternion = _array(robot.data.root_quat_w[0])
        base_quaternion_norm = float(np.linalg.norm(base_quaternion))
        if not np.isfinite(base_quaternion_norm) or abs(base_quaternion_norm - 1.0) > 1e-6:
            raise ValueError(f"production Base quaternion is not unit: {base_quaternion_norm}")
        base_quaternion /= base_quaternion_norm
        environment_origin = _array(env.scene.env_origins[0])
        snapshot = task_snapshot(
            config=config,
            environment_origin_w=environment_origin,
            base_pose_w=(*base_position, *base_quaternion),
        )
        source_target_pose_w = set_source_target_pose(env, snapshot)
        ik = solve_legacy_joint_targets(snapshot, q_start)
        report.update(
            passed=bool(ik["passed"]),
            stage="complete" if ik["passed"] else "legacy_ik",
            config={
                "source_index": config.source_index,
                "destination_index": config.destination_index,
                "control_hz": config.control_hz,
                "seed": config.seed,
                "lift_clearance_m": config.lift_clearance_m,
                "scene_id": config.scene_id,
                "state4_routing": routing_payload(config.state4_routing),
                "source2_routing": vars(config.source2_routing) if config.source2_routing is not None else None,
            },
            measured={
                "environment_origin_w": environment_origin.tolist(),
                "base_pose_w": [*base_position.tolist(), *base_quaternion.tolist()],
                "base_quaternion_raw_norm": base_quaternion_norm,
            },
            task_snapshot=snapshot,
            task_hash=snapshot["task_hash"],
            source_target_pose_w=source_target_pose_w,
            legacy_ik=ik,
            scope="target extraction and Legacy IK only; MotionGen and replay are not run",
        )
        if not report["passed"]:
            raise RuntimeError("one or more Legacy IK targets failed")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["backend"] = args.backend
        report["isaaclab_modules_loaded"] = sorted(
            n for n in sys.modules if n == "isaaclab" or n.startswith("isaaclab.")
        )
        if args.backend == "isaacsim" and report["isaaclab_modules_loaded"]:
            report["passed"] = False
            report["failure_reason"] = "native backend imported Isaac Lab"
        if env is not None:
            env.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


try:
    main()
except Exception:
    traceback.print_exc()
    raise
finally:
    launcher.app.close()
