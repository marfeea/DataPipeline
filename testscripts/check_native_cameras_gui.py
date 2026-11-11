"""Verify all three native cameras on the user's desktop before any robot motion."""

import argparse
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.sim_backend import launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    launcher = launch(SimpleNamespace(backend="isaacsim", headless=False,
                                     kit_cache_dir=Path("data/isaacsim_kit_cache")))
    env = None
    report = dict(passed=False, scope="Native camera initialization only; no robot trajectory", headless=False,
                  desktop_gui=launcher.desktop_gui)
    try:
        import numpy as np

        import carb

        from scripts.native.scene import NativeTrajectoryEnv, NativeTrajectoryEnvCfg
        from scripts.pipelines.curobo_bootstrap.replay_visualization import ReplayFrameRecorder

        settings = carb.settings.get_settings()
        settings.set_bool("/app/asyncRendering", False)
        settings.set_bool("/rtx/ecoMode/enabled", False)
        report["render_settings"] = {k: settings.get(k) for k in (
            "/app/asyncRendering", "/rtx/ecoMode/enabled", "/app/renderer/enabled",
        )}
        cfg = NativeTrajectoryEnvCfg()
        cfg.camera_resolution = None
        cfg.camera_warmup_timeout_s = 10
        cfg.camera_diagnostics_directory = args.output / "warmup"
        env = NativeTrajectoryEnv(cfg)
        before = dict(time=env.world.current_time,
                      q=env.robot.data.joint_pos.cpu().numpy().copy(),
                      object=env.target.data.root_pos_w.cpu().numpy().copy())
        try:
            env._create_cameras((256, 256))
        except RuntimeError as error:
            if "warmup did not complete" not in str(error):
                raise
            import omni.replicator.core as rep

            report["ordinary_warmup_error"] = str(error)
            settings.set_bool("/exts/omni.replicator.core/Orchestrator/enabled", True)
            rep.orchestrator.set_capture_on_play(False)
            for _ in range(3):
                rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=False)
            env.scene.update(cfg.sim.dt)
            report["capture_recovery"] = "Replicator synchronous capture at delta_time=0"
        report["initialization_state_delta"] = dict(
            time=env.world.current_time - before["time"],
            max_joint=float(np.max(np.abs(env.robot.data.joint_pos.cpu().numpy() - before["q"]))),
            max_object=float(np.max(np.abs(env.target.data.root_pos_w.cpu().numpy() - before["object"]))),
        )
        for camera in env.cameras.values():
            camera.refresh()
        if any(float(np.std(c.camera.get_rgb())) <= 5 for c in env.cameras.values()):
            from PIL import Image

            for name, camera in env.cameras.items():
                Image.fromarray(camera.camera.get_rgb()).save(args.output / f"recovery_{name}.png")
            raise RuntimeError("Native cameras still blank after synchronous capture")
        env.cameras_ready = True
        recorder = ReplayFrameRecorder(args.output / "execution_visualization", env.camera_specs)
        recorder.capture(env.cameras, render_frame=0, trajectory_index=0, time_s=0, event="initial_scene", force=True)
        report["visualization"] = recorder.finish(source_frames=1, render_calls=1)
        report["rgb_std"] = {name: float(np.std(c.camera.get_rgb())) for name, c in env.cameras.items()}
        report["passed"] = True
    except Exception as error:
        report["failure_reason"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if env is not None:
            env.close()
        launcher.app.close()
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
