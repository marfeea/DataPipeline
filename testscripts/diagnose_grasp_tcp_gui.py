"""Measured GUI grasp/lift probe with an explicit translated task TCP frame."""

import argparse
import ctypes
import json
import os
import sys
import traceback
from ctypes import wintypes
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def desktop_evidence():
    user = ctypes.windll.user32
    user.GetThreadDesktop.restype = wintypes.HANDLE
    handle = user.GetThreadDesktop(ctypes.windll.kernel32.GetCurrentThreadId())
    buffer = ctypes.create_unicode_buffer(256)
    needed = wintypes.DWORD()
    if not user.GetUserObjectInformationW(wintypes.HANDLE(handle), 2, buffer, 512, ctypes.byref(needed)):
        raise RuntimeError("Cannot verify the desktop")
    result = dict(pid=os.getpid(), desktop=buffer.value, windows=[])
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @callback_type
    def collect(hwnd, _):
        pid = wintypes.DWORD()
        user.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value == os.getpid():
            title = ctypes.create_unicode_buffer(512)
            user.GetWindowTextW(hwnd, title, 512)
            rect = wintypes.RECT()
            user.GetWindowRect(hwnd, ctypes.byref(rect))
            result["windows"].append(dict(handle=int(hwnd), title=title.value,
                                          visible=bool(user.IsWindowVisible(hwnd)),
                                          minimized=bool(user.IsIconic(hwnd)),
                                          rect=[rect.left, rect.top, rect.right, rect.bottom]))
        return True

    user.EnumWindows(collect, 0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-output", type=Path, required=True)
    parser.add_argument("--task-offset-t", type=float, nargs=3, default=(0, 0, 0))
    args = parser.parse_args()
    for path in (args.output, args.trajectory_output):
        path.mkdir(parents=True, exist_ok=False)
    report = dict(passed=False, mode="gui", headless=False, task_offset_t=args.task_offset_t,
                  source_index=0, destination_index=1, target_state_writes_after_initialization=0,
                  arm_state_writes_after_initialization=0)
    before = desktop_evidence()
    save(args.output / "desktop_before_launch.json", before)
    if before["desktop"] != "Default":
        raise RuntimeError("Run on the interactive Default desktop; sandbox desktops are not permitted")
    from scripts.sim_backend import launch

    launcher = launch(SimpleNamespace(backend="isaacsim", headless=False,
                                      kit_cache_dir=ROOT / "data/isaacsim_kit_cache"))
    rt = recorder = None
    rows = []
    try:
        visible = desktop_evidence()
        save(args.output / "desktop_window.json", visible)
        if not any(w["visible"] and not w["minimized"] and "Isaac" in w["title"] for w in visible["windows"]):
            raise RuntimeError("Isaac window is not visible on the user's desktop")
        print("DESKTOP_GUI_VERIFIED " + json.dumps(visible), flush=True)
        from scripts.native.aubo_ik import AUBO_E5_TCP_MODEL, solve_ik
        from scripts.pipelines.curobo_bootstrap.physical_failure import GraspEvidence
        from scripts.pipelines.curobo_bootstrap.replay_visualization import ReplayFrameRecorder
        from scripts.pipelines.curobo_bootstrap.source_target import set_source_target_pose
        from scripts.pipelines.curobo_bootstrap.task_targets import PickPlaceV2Config, task_snapshot
        from testscripts.search_stable_placement_isaacsim import SearchRuntime, pose_list

        prior = json.loads((ROOT.parent / "artifacts/0_to_4_gui_retest_20261014_keyframes/0_to_4/trace.json")
                           .read_text(encoding="utf-8"))
        grasp = next(r for r in prior if r["evidence"]["grasp_verified"])
        initial = [r for r in prior if r["stage"] == "free_space_to_pre_pick"][-1]
        rt = SearchRuntime(SimpleNamespace(device="cuda:0", physics_hz=120, capture=False), grasp, [initial["q"]])
        task = task_snapshot(config=PickPlaceV2Config(), environment_origin_w=(0, 0, 0),
                             base_pose_w=pose_list(rt.world_from_base))
        set_source_target_pose(rt.env, task)
        positions = rt.robot.data.joint_pos.clone()
        positions[:, rt.arm_ids] = rt.tensor(initial["q"])
        positions[:, rt.finger_ids] = rt.tensor([0.0115, -0.0115])
        rt.robot.write_joint_state_to_sim(positions, rt.torch.zeros_like(positions))
        rt.robot.set_joint_position_target(rt.tensor(initial["q"]), joint_ids=rt.arm_ids)
        rt.robot.set_joint_position_target(rt.tensor([0.0115, -0.0115]), joint_ids=rt.finger_ids)
        rt.env.sim.forward()
        rt.env.scene.update(rt.cfg.sim.dt)
        rt.initializing = False
        rt.time, rt.rows, rt.capture = 0.0, rows, None
        rt.grasp_verified, rt.carry_reference, rt.carry_loss_s = False, None, 0.0
        rt.goal = [*task["destination"]["position_e"], *task["destination"]["rotation_wxyz"]]
        shift = np.eye(4)
        shift[:3, 3] = args.task_offset_t
        model = replace(AUBO_E5_TCP_MODEL, name="grasp_center_tcp_probe",
                        frame6_to_endpoint=AUBO_E5_TCP_MODEL.frame6_to_endpoint @ shift)
        q = np.array(initial["q"])
        current = rt.tcp_w() @ shift
        report.update(initial_task_tcp_w=pose_list(current), source_object_pose_w=task["source"]["position_e"],
                      task_snapshot=task, physics_contract=rt.contract)
        save(args.output / "report.json", report)
        print("GUI_PROBE_SCENE_READY", flush=True)
        from testscripts.native.gripper_geometry import export_gripper_geometry

        report["gripper_geometry"] = export_gripper_geometry(rt.env, args.output / "gripper_geometry")
        rt.env.cfg.camera_diagnostics_directory = args.output / "camera_warmup"
        rt.env._create_cameras((256, 256))
        recorder = ReplayFrameRecorder(args.output / "execution_visualization", rt.env.camera_specs)
        detector = None

        def step(pose, fingers, stage):
            nonlocal q, detector
            if not launcher.app.is_running():
                raise RuntimeError("User closed the Isaac window")
            target_b = pose_list(rt.base_from_world @ pose)
            solved = solve_ik(target_b[:3], target_b[3:], model=model, seed=q.copy(), max_iterations=120,
                              position_tolerance_m=2e-5, orientation_tolerance_rad=2e-4)
            if not solved.success:
                report["ik_failure"] = dict(stage=stage, reason=solved.reason, position_error_m=solved.position_error_m,
                                            orientation_error_rad=solved.orientation_error_rad,
                                            target_task_tcp_w=pose_list(pose), seed=q.tolist())
                raise RuntimeError("Probe IK failed: " + solved.reason)
            q = solved.q
            row = rt.step(q, fingers, stage, pose)
            row["legacy_tcp_pose_w"] = row["tcp_pose_w"]
            row["tcp_pose_w"] = pose_list(rt.tcp_w() @ shift)
            sample = dict(row, target_pose_w=row["object_pose_w"], target_velocity_w=row["object_velocity_w"])
            if detector is None:
                detector = GraspEvidence(sample)
            row["evidence"] = detector.step(sample)
            rt.grasp_verified = detector.ever_grasped
            rt.env.sim.render()
            recorder.capture(rt.env.cameras, render_frame=len(rows), trajectory_index=len(rows) - 1,
                             time_s=row["time_s"], event=stage,
                             force=len(rows) == 1 or (len(rows) > 1 and rows[-2]["stage"] != stage)
                             or stage == "close_gripper")
            if len(rows) % 120 == 0:
                print(f"GUI_PROBE_PROGRESS frame={len(rows)} stage={stage} "
                      f"object_z={row['object_pose_w'][2]:.5f} grasp={row['evidence']['grasp_verified']}", flush=True)
            return row

        def hold(pose, fingers, stage, seconds):
            for _ in range(round(seconds * 60)):
                step(pose, fingers, stage)

        def move(start, finish, fingers, stage, speed):
            distance = np.linalg.norm(finish[:3, 3] - start[:3, 3])
            count = max(1, int(np.ceil(distance / speed * 60)))
            for index in range(1, count + 1):
                fraction = index / count
                pose = start.copy()
                pose[:3, 3] += (finish[:3, 3] - start[:3, 3]) * fraction
                step(pose, fingers, stage)

        hold(current, [0.0115, -0.0115], "initial_hold", 0.5)
        pick = current.copy()
        pick[:3, 3] = task["source"]["pre_grasp_tcp_position_e"]
        move(current, pick, [0.0115, -0.0115], "approach_pick", 0.03)
        hold(pick, [0.0115, -0.0115], "pre_close_hold", 0.5)
        for index in range(30):
            fraction = 1 - (index + 1) / 30
            step(pick, [0.0115 * fraction, -0.0115 * fraction], "close_gripper")
        hold(pick, [0, 0], "closed_hold", 1.0)
        lift_low = pick.copy()
        lift_low[2, 3] += 0.05
        move(pick, lift_low, [0, 0], "lift_low", 0.02)
        high = lift_low.copy()
        high[2, 3] = 1.38
        move(lift_low, high, [0, 0], "lift_high", 0.05)
        hold(high, [0, 0], "lift_hold", 1.0)
        last = rows[-1]
        report.update(completed=True, final_evidence=last["evidence"],
                      final_object_z=last["object_pose_w"][2],
                      object_lift_z_m=last["object_pose_w"][2] - rows[0]["object_pose_w"][2],
                      passed=bool(last["evidence"]["grasp_verified"] and
                                  last["object_pose_w"][2] - rows[0]["object_pose_w"][2] >= 0.1))
        print("GUI_PROBE_RESULT " + json.dumps({k: report[k] for k in ["passed", "object_lift_z_m"]}), flush=True)
    except Exception as error:
        report["failure_reason"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        if rows:
            save(args.trajectory_output / "trace.json", rows)
            np.savez(args.trajectory_output / "observations.npz", time_s=[r["time_s"] for r in rows],
                     q=[r["q"] for r in rows], object_pose_w=[r["object_pose_w"] for r in rows],
                     tcp_pose_w=[r["tcp_pose_w"] for r in rows])
        if recorder is not None:
            report["visualization"] = recorder.finish(source_frames=len(rows), render_calls=len(rows))
        report["trajectory_directory"] = str(args.trajectory_output.resolve())
        report["samples"] = len(rows)
        save(args.output / "report.json", report)
        if rt is not None:
            rt.env.close()
        launcher.app.close()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
