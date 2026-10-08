"""Freeze the V2 production mesh world and run real CuRobo endpoint queries."""

import argparse
import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from DataPipe.runtime import ensure_curriculumrl_available

ensure_curriculumrl_available()
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--targets-report", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--collision-model",
    type=Path,
    default=Path("configs/curobo/assets/aubo_e5_project/collision_model_v2.json"),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(f"Refusing to overwrite audit directory: {args.output}")
launcher = AppLauncher(args)

import numpy as np
from CurriculumRL.tasks.tcp_docking.trajectory_env_cfg import TcpDockingTrajectoryEnv, TcpDockingTrajectoryEnvCfg

from DataPipe.pipelines.curobo_bootstrap.planner import CuroboMvpPlanner
from DataPipe.pipelines.curobo_bootstrap.production_world import extract_production_world
from DataPipe.pipelines.curobo_bootstrap.source_target import set_source_target_pose


def main():
    args.output.mkdir(parents=True)
    report_path = args.output / "report.json"
    report = {"passed": False, "stage": "startup", "python": sys.executable}
    env = None
    try:
        targets = json.loads(args.targets_report.read_text(encoding="utf-8"))
        if not targets.get("passed"):
            raise ValueError("targets report did not pass")
        task = targets["task_snapshot"]
        report.update(
            source_index=task["source_index"],
            destination_index=task["destination_index"],
            source_name=task["source"]["name"],
            destination_name=task["destination"]["name"],
            task_hash=task["task_hash"],
        )
        cfg = TcpDockingTrajectoryEnvCfg()
        cfg.scene.num_envs = 1
        cfg.sim.device = args.device
        cfg.seed = 7
        cfg.actions.joint_trajectory.open_position = (0.0115, -0.0115)
        cfg.actions.joint_trajectory.closed_position = (0.0, 0.0)
        env = TcpDockingTrajectoryEnv(cfg)
        env.reset()
        report["source_target_pose_w"] = set_source_target_pose(env, task)
        snapshot, evidence = extract_production_world(env, args.output / "world_assets")
        q = np.asarray(
            [targets["legacy_ik"]["q_start"]]
            + [item["joint_position"] for item in targets["legacy_ik"]["targets"].values()],
            dtype=np.float64,
        )
        planner = CuroboMvpPlanner(args.collision_model, world=snapshot)
        full_scores = planner.collision.query(q, world_ignore_links=planner.world_collision_ignore_links)
        planning_world = snapshot.permit_target_contact()
        planner.collision.update_world(planning_world)
        scores = planner.collision.query(q, world_ignore_links=planner.world_collision_ignore_links)
        world_scores = np.asarray(scores["world"], dtype=np.float64)
        report.update(
            passed=bool(world_scores.shape == (6,) and np.isfinite(world_scores).all() and np.max(world_scores) <= 0),
            stage="complete",
            world=asdict(snapshot.config),
            world_hash=snapshot.world_hash,
            world_geometry_valid=True,
            collision_model=str(args.collision_model.resolve()),
            world_evidence=evidence,
            target_labels=["q_start", *targets["legacy_ik"]["targets"]],
            world_collision_score=world_scores.tolist(),
            world_collision_score_with_target=np.asarray(full_scores["world"], dtype=np.float64).tolist(),
            collision_policy={
                "allowed_world_contact": [snapshot.config.target_name],
                "protected_world_meshes": [mesh.name for mesh in planning_world.active_meshes],
                "self_collision_hard_gate": True,
            },
            self_collision_score=np.asarray(scores["self"], dtype=np.float64).tolist(),
            world_collision_ignore_links=list(planner.world_collision_ignore_links),
            scope=(
                "world collision gate excludes the selected sample bottle and the immobile Base_Link; "
                "support trays and all other meshes remain protected; self-collision branch selection remains separate"
            ),
        )
        if not report["passed"]:
            raise RuntimeError(f"production world endpoint collision: {world_scores.tolist()}")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if env is not None:
            env.close()
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


try:
    main()
except Exception:
    traceback.print_exc()
    raise
finally:
    launcher.app.close()
