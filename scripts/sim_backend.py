"""Select the runtime before importing any Kit-dependent modules."""

import json
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
        from scripts.runtime import ensure_curriculumrl_available

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
    verify_gui = os.name == "nt" and not args.headless
    if verify_gui:
        from scripts.native.gui_desktop import desktop_evidence

        if desktop_evidence()["desktop"] != "Default":
            raise RuntimeError("GUI simulation must run on the user's Default desktop, outside the sandbox desktop")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["OPTIX_CACHE_PATH"] = str(cache / "optix")
    os.environ["WARP_CACHE_PATH"] = str(cache / "warp")
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": args.headless,
            "hide_ui": args.headless,
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
                "--/app/renderer/skipWhileMinimized=false",
                "--/app/renderer/sleepMsOnFocus=0",
                "--/app/renderer/sleepMsOutOfFocus=0",
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
    evidence = desktop_evidence() if verify_gui else None
    if verify_gui:
        if not any(w["visible"] and not w["minimized"] and "Isaac" in w["title"] for w in evidence["windows"]):
            app.close()
            raise RuntimeError("Isaac GUI window is not visible on the user's desktop")
        print("DESKTOP_GUI_VERIFIED " + json.dumps(evidence), flush=True)
    return SimpleNamespace(app=app, desktop_gui=evidence)


def trajectory_classes():
    if selected_backend() == "isaaclab":
        from CurriculumRL.tasks.tcp_docking.trajectory_env_cfg import (
            TcpDockingTrajectoryEnv,
            TcpDockingTrajectoryEnvCfg,
        )

        return TcpDockingTrajectoryEnv, TcpDockingTrajectoryEnvCfg
    from scripts.native.scene import NativeTrajectoryEnv, NativeTrajectoryEnvCfg

    return NativeTrajectoryEnv, NativeTrajectoryEnvCfg


def ik_class():
    if selected_backend() == "isaaclab":
        from CurriculumRL.tasks.tcp_docking.aubo_ik_action import AuboIKAction
    else:
        from scripts.native.ik_controller import AuboIKAction
    return AuboIKAction


def configure_cameras(cfg, width, height):
    from configs.camera import project_diagnostic_camera_sensor_specs

    specs = project_diagnostic_camera_sensor_specs(resolution=(width, height), data_types=("rgb",))
    if selected_backend() == "isaacsim":
        cfg.camera_resolution = (width, height)
        return cfg, specs
    from scripts.collection_cameras import configure_collection_cameras
    from scripts.isaac_scene import _camera_cfg

    for spec in specs[1:]:
        setattr(cfg.scene, spec.name, _camera_cfg(spec))
    return configure_collection_cameras(cfg, specs)
