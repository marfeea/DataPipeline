"""Select the runtime before importing any Kit-dependent modules."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace


def selected_backend():
    for index, item in enumerate(sys.argv):
        if item == "--backend":
            return sys.argv[index + 1]
        if item.startswith("--backend="):
            return item.split("=", 1)[1]
    return "isaaclab"


def add_launcher_args(parser):
    parser.add_argument("--backend", choices=("isaaclab", "isaacsim"), default="isaaclab")
    if selected_backend() == "isaaclab":
        from DataPipe.runtime import ensure_curriculumrl_available

        ensure_curriculumrl_available()
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    else:
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument("--kit-cache-dir", type=Path, default=Path("data/isaacsim_kit_cache"))


def launch(args):
    if args.backend == "isaaclab":
        from isaaclab.app import AppLauncher

        return AppLauncher(args)
    cache = args.kit_cache_dir.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["OPTIX_CACHE_PATH"] = str(cache / "optix")
    os.environ["WARP_CACHE_PATH"] = str(cache / "warp")
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": args.headless,
            "renderer": "RayTracedLighting",
            "extra_args": [
                f"--/app/tokens/cache={cache}",
                f"--/app/userConfigPath={cache / 'user.config.json'}",
                "--/metricsAssembler/changeListenerEnabled=false",
                "--/app/useFabricSceneDelegate=true",
                "--/rtx/hydra/readTransformsFromFabricInRenderDelegate=true",
                "--/omni/replicator/asyncRendering=false",
                "--/app/renderer/waitIdle=true",
                "--/app/hydraEngine/waitIdle=true",
                "--/app/updateOrder/checkForHydraRenderComplete=1000",
                "--/rtx/post/dlss/execMode=0",
                "--/rtx/ambientOcclusion/enabled=false",
                "--/rtx/raytracing/cached/enabled=false",
                "--/rtx/translucency/enabled=false",
                "--/rtx/reflections/enabled=false",
                "--/rtx/indirectDiffuse/enabled=false",
                "--/rtx/directLighting/sampledLighting/enabled=true",
                "--/rtx/directLighting/sampledLighting/samplesPerPixel=1",
                "--/rtx/sceneDb/ambientLightIntensity=1.0",
                "--/exts/omni.replicator.core/Orchestrator/enabled=false",
            ],
        }
    )
    return SimpleNamespace(app=app)


def trajectory_classes():
    if selected_backend() == "isaaclab":
        from CurriculumRL.tasks.tcp_docking.trajectory_env_cfg import (
            TcpDockingTrajectoryEnv,
            TcpDockingTrajectoryEnvCfg,
        )

        return TcpDockingTrajectoryEnv, TcpDockingTrajectoryEnvCfg
    from DataPipe.native.scene import NativeTrajectoryEnv, NativeTrajectoryEnvCfg

    return NativeTrajectoryEnv, NativeTrajectoryEnvCfg


def ik_class():
    if selected_backend() == "isaaclab":
        from CurriculumRL.tasks.tcp_docking.aubo_ik_action import AuboIKAction
    else:
        from DataPipe.native.ik_controller import AuboIKAction
    return AuboIKAction


def configure_cameras(cfg, width, height):
    from DataPipe.camera_config import project_diagnostic_camera_sensor_specs

    specs = project_diagnostic_camera_sensor_specs(resolution=(width, height), data_types=("rgb",))
    if selected_backend() == "isaacsim":
        cfg.camera_resolution = (width, height)
        return cfg, specs
    from DataPipe.collection_cameras import configure_collection_cameras
    from DataPipe.isaac_scene import _camera_cfg

    for spec in specs[1:]:
        setattr(cfg.scene, spec.name, _camera_cfg(spec))
    return configure_collection_cameras(cfg, specs)
