"""List enabled production collision prim bounds for CuRobo V2 world refinement."""

import argparse
import json
import sys
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

import numpy as np
from CurriculumRL.tasks.tcp_docking.trajectory_env_cfg import TcpDockingTrajectoryEnv, TcpDockingTrajectoryEnvCfg

from pxr import Usd, UsdGeom, UsdPhysics


def main():
    cfg = TcpDockingTrajectoryEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.seed = 7
    cfg.actions.joint_trajectory.open_position = (0.0115, -0.0115)
    cfg.actions.joint_trajectory.closed_position = (0.0, 0.0)
    env = TcpDockingTrajectoryEnv(cfg)
    try:
        env.reset()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        env_root = str(env.scene.env_prim_paths[0])
        records = []
        for prim in Usd.PrimRange.Stage(env.scene.stage, Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            if not path.startswith(env_root) or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            api = UsdPhysics.CollisionAPI(prim)
            if api.GetCollisionEnabledAttr().Get() is False:
                continue
            value = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            low = np.asarray(value.GetMin(), dtype=np.float64)
            high = np.asarray(value.GetMax(), dtype=np.float64)
            records.append(
                {
                    "path": path,
                    "type": prim.GetTypeName(),
                    "minimum_w": low.tolist(),
                    "maximum_w": high.tolist(),
                    "dimensions_m": (high - low).tolist(),
                }
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"env_root": env_root, "collision_prims": records}, indent=2) + "\n")
        print(json.dumps({"count": len(records), "paths": [item["path"] for item in records]}))
    finally:
        env.close()


try:
    main()
finally:
    launcher.app.close()
