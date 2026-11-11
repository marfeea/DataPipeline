"""Analyze measured lift speeds, grip stability, command limits, and GUI evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.pipelines.curobo_bootstrap.planner import JOINT_NAMES, TrajectoryData, load_trajectory, save_trajectory
from scripts.pipelines.curobo_bootstrap.robot_adapter import pose_matrix


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def analyze(entry):
    evidence = Path(entry["evidence_directory"])
    trajectory = Path(entry["trajectory_directory"])
    report_file = evidence / "report.json"
    if not report_file.exists():
        return dict(label=entry["label"], running=True)
    report = read(report_file)
    trace = read(trajectory / "trace.json")
    hz = report["execution_contract"]["control_hz"]
    result = dict(label=entry["label"], mode=report["mode"], joint_cap_rad=entry["joint_cap_rad"],
                  execution_passed=report["passed"], failure_reason=report.get("failure_reason"),
                  reference_task_hash=report.get("task_hash"),
                  trajectory_directory=str(trajectory), evidence_directory=str(evidence),
                  physics_hz=report["execution_contract"]["physics_hz"], recording_hz=hz)
    experiment = report.get("lift_experiment", {})
    result["command_hz"] = experiment.get("command_hz", 4.0)
    result["tcp_step_cap_m"] = experiment.get("tcp_step_cap_m", 0.05)
    result["interpolation_fraction"] = experiment.get("interpolation_fraction", 1.0)
    result["interpolation_physics_steps"] = experiment.get("interpolation_physics_steps", 30)
    result["interpolation_duration_s"] = result["interpolation_physics_steps"] / result["physics_hz"]
    result["grasp_settle_s"] = experiment.get("grasp_settle_s", 0.0)
    result["grasp_settle_gripper_position"] = report.get("grasp_settle_gripper_position")
    indices = [i for i, row in enumerate(trace) if row["stage"] == "legacy_lift"]
    updates = [i for i in indices if trace[i]["ik_command_updated"]]
    if indices:
        first = max(0, indices[0] - 1)
        rows = [trace[first], *[trace[i] for i in indices]]
        transform = pose_matrix(report["base_pose_w"][:3], report["base_pose_w"][3:])
        xyz_b = np.asarray([row["tcp_position_b"] for row in rows])
        xyz_w = xyz_b @ transform[:3, :3].T + transform[:3, 3]
        delta = np.diff(xyz_w, axis=0)
        speed = np.linalg.norm(delta, axis=1) * hz
        vertical = delta[:, 2] * hz
        height = xyz_w[-1, 2] - xyz_w[0, 2]
        fraction = (xyz_w[1:, 2] - xyz_w[0, 2]) / max(height, 1e-12)
        steady = vertical[(fraction >= 0.2) & (fraction <= 0.8)]
        command_delta = np.asarray([np.asarray(trace[i]["commanded_tcp_position_b"]) - trace[i]["current_tcp_position_b"] for i in updates])
        spacing = np.diff(updates)
        joint_requested = np.asarray([trace[i]["requested_joint_delta_rad"] for i in updates])
        result.update(
            lift_frames=len(indices), lift_duration_s=len(indices) / hz, tcp_vertical_distance_m=float(height),
            mean_vertical_speed_m_s=float(height / (len(indices) / hz)),
            mean_tcp_path_speed_m_s=float(np.mean(speed)),
            steady_median_vertical_speed_m_s=float(np.median(steady)) if len(steady) else None,
            peak_vertical_speed_m_s=float(np.max(vertical)), peak_tcp_path_speed_m_s=float(np.max(speed)),
            actual_max_joint_speed_rad_s=float(np.max(np.abs([trace[i]["dq"] for i in indices]))),
            ik_commands=len(updates), update_spacing_record_frames=np.unique(spacing).tolist(),
            command_frequency_verified=bool(len(spacing) and np.all(spacing == hz / result["command_hz"])),
            max_command_displacement_m=float(np.max(np.linalg.norm(command_delta, axis=1))),
            tcp_step_cap_verified=bool(np.all(np.linalg.norm(command_delta, axis=1) <= result["tcp_step_cap_m"] + 1e-6)),
            ik_target_matches_command=all(np.allclose(trace[i]["commanded_tcp_position_b"], trace[i]["ik_target_tcp_position_b"], atol=1e-6, rtol=0) for i in updates),
            joint_limited_commands=int(np.count_nonzero(np.max(np.abs(joint_requested), axis=1) > entry["joint_cap_rad"] + 1e-6)),
            median_actual_tcp_displacement_per_record_m=float(np.median(np.linalg.norm(delta, axis=1))),
            max_requested_joint_delta_rad=float(np.max(np.abs(joint_requested))),
            median_max_joint_tracking_error_rad=float(np.median([
                np.max(np.abs(np.asarray(trace[i]["commanded_joint_target"]) - trace[i]["q"])) for i in indices])),
            max_commanded_vs_ik_joint_target_error_rad=float(max(
                np.max(np.abs(np.asarray(trace[i]["commanded_joint_target"]) - trace[i]["ik_solution_joint_position"]))
                for i in indices)),
            maximum_ik_position_error_m=float(max(trace[i]["ik_position_error_m"] for i in indices)),
        )
        carry_rows = [trace[first], *[row for row in trace if row["stage"] in ("legacy_lift", "legacy_lift_hold")]]
        relative = []
        for row in carry_rows:
            tcp = transform @ pose_matrix(row["tcp_position_b"], row["tcp_quaternion_b"])
            obj = pose_matrix(row["target_pose_w"][:3], row["target_pose_w"][3:])
            relative.append(np.linalg.inv(tcp) @ obj)
        translation_drift = [np.linalg.norm(value[:3, 3] - relative[0][:3, 3]) for value in relative]
        rotation_drift = [np.arccos(np.clip((np.trace(relative[0][:3, :3].T @ value[:3, :3]) - 1) / 2, -1, 1)) for value in relative]
        result["max_grasp_translation_drift_m"] = float(max(translation_drift))
        result["max_grasp_rotation_drift_deg"] = float(np.rad2deg(max(rotation_drift)))
        result["object_vertical_lift_m"] = float(carry_rows[-1]["target_pose_w"][2] - carry_rows[0]["target_pose_w"][2])
        result["hold_s"] = sum(row["stage"] == "legacy_lift_hold" for row in trace) / hz
        result["stable_carry"] = bool(result["max_grasp_translation_drift_m"] <= 0.005
                                       and result["max_grasp_rotation_drift_deg"] <= 5
                                       and result["object_vertical_lift_m"] >= 0.35 and result["hold_s"] >= 0.5)
    if len(trace) >= 2:
        q = np.asarray([row["q"] for row in trace])
        dq = np.asarray([row["dq"] for row in trace])
        times = np.arange(len(trace)) / hz
        npz = trajectory / "trajectory.npz"
        save_trajectory(npz, TrajectoryData(times, q, dq, np.gradient(dq, 1 / hz, axis=0), JOINT_NAMES))
        result["npz_roundtrip"] = bool(np.array_equal(load_trajectory(npz).joint_position, q))
    manifest_file = evidence / "execution_visualization/frames.json"
    if manifest_file.exists():
        manifest = read(manifest_file)
        counts, stages, stds = {}, {}, []
        aligned = manifest.get("source_trace_sha256") == hashlib.sha256((trajectory / "trace.json").read_bytes()).hexdigest()
        for frame in manifest["frames"]:
            with Image.open(manifest_file.parent / frame["rgb"]) as image:
                image.load()
                stds.append(float(np.std(np.asarray(image.convert("RGB"), dtype=float))))
            counts[frame["camera"]] = counts.get(frame["camera"], 0) + 1
            stages[frame["execution_stage"]] = stages.get(frame["execution_stage"], 0) + 1
            i = frame["trajectory_index"]
            aligned &= 0 <= i < len(trace) and trace[i]["stage"] == frame["execution_stage"]
            aligned &= abs(frame["time_s"] - i / hz) < 1e-9
        required_stages = ["legacy_approach_pick", "close_gripper"]
        if result["grasp_settle_s"]:
            required_stages.append("grasp_settle")
        required_stages.extend(["legacy_lift", "legacy_lift_hold"])
        complete = bool(stds and min(stds) > 5 and aligned and len(counts) == 3 and all(
            stage in stages for stage in required_stages))
        result["visual_evidence"] = dict(images=len(stds), cameras=counts, stages=stages,
                                          minimum_rgb_std=min(stds) if stds else None, trace_alignment=bool(aligned),
                                          complete=complete, manifest=str(manifest_file))
        sheet = Image.new("RGB", (3 * 256, len(required_stages) * 280), "white")
        draw = ImageDraw.Draw(sheet)
        for row, stage in enumerate(required_stages):
            for column, camera in enumerate(counts):
                choices = [f for f in manifest["frames"] if f["execution_stage"] == stage and f["camera"] == camera]
                if choices:
                    frame = choices[-1]
                    with Image.open(manifest_file.parent / frame["rgb"]) as image:
                        sheet.paste(image.convert("RGB").resize((256, 256)), (column * 256, row * 280))
                    draw.text((column * 256 + 4, row * 280 + 257),
                              f'{stage.replace("legacy_", "")} | {camera} #{frame["trajectory_index"]}', fill="black")
        sheet.save(evidence / "native_keyframes.png")
    result["tray_collision_diagnostic_passed"] = report.get("tray_collision_diagnostic", {}).get("passed")
    result["accepted_speed_trial"] = bool(result["execution_passed"] and result.get("stable_carry")
                                           and result.get("command_frequency_verified") and result.get("tcp_step_cap_verified")
                                           and result.get("ik_target_matches_command") and result.get("npz_roundtrip")
                                           and result.get("visual_evidence", {}).get("complete"))
    (evidence / "speed_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="legacy_lift_speed_20261015_retry01")
    args = parser.parse_args()
    root = PROJECT / "data/test/trajdata" / args.run_id / "0_to_1"
    status = read(root / "status.json")
    results = [analyze(entry) for entry in status["trials"]]
    accepted = [r for r in results if r.get("accepted_speed_trial")]
    best = max(accepted, key=lambda result: result["mean_vertical_speed_m_s"]) if accepted else None
    command_hz = status.get("lift_command_hz", 4.0)
    step_m = status.get("lift_step_m", 0.05)
    summary = dict(scope=f"current calibrated source-0 geometry, {command_hz:g} Hz / {step_m*100:g} cm, isolated lift experiment",
                   nominal_tcp_speed_m_s=command_hz * step_m, results=results, best=best,
                   desktop_gui=read(root / "desktop_gui.json") if (root / "desktop_gui.json").exists() else None)
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [f"# Legacy IK {step_m*100:g} cm / {command_hz:g} Hz 抬升速度实验", "",
             f"固定每条抬升命令的 TCP 总位移上限 {step_m*100:g} cm、命令频率 {command_hz:g} Hz，名义目标供给速度 {step_m*command_hz*100:g} cm/s；物理 120 Hz、记录 60 Hz。",
             "已在隔离实验中取消抬升的固定 2/5 cm/s 阶段速度、1 mm 步长和跟踪超前限制。接近、闭爪保持同一低速策略。",
             "实验未修改主入口或主配置。每次在同一 GUI 场景内复位到相同关节与源物体姿态后重新抓取。", "",
             "| 关节增量上限 | 插值时长 | 抬升用时 | 平均竖直速度 | 稳态中位速度 | TCP 路径峰值 | 抓取漂移 | 图片 | 稳定速度验收 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for result in results:
        if "mean_vertical_speed_m_s" in result:
            lines.append(f'| {result["joint_cap_rad"]:.3f} rad | {result["interpolation_duration_s"]:.4f} s | {result["lift_duration_s"]:.3f} s | '
                         f'{result["mean_vertical_speed_m_s"]*100:.2f} cm/s | {result["steady_median_vertical_speed_m_s"]*100:.2f} cm/s | '
                         f'{result["peak_tcp_path_speed_m_s"]*100:.2f} cm/s | {result["max_grasp_translation_drift_m"]*1000:.2f} mm | '
                         f'{result.get("visual_evidence", {}).get("images", 0)} | {result["accepted_speed_trial"]} |')
        else:
            lines.append(f'| {result.get("joint_cap_rad", "运行中")} | — | — | — | — | — | — | 0 | {result.get("failure_reason", "运行中")} |')
    if best:
        lines += ["", f'本次测得最快的稳定全段平均竖直速度为 {best["mean_vertical_speed_m_s"]*100:.2f} cm/s，'
                  f'对应关节增量上限 {best["joint_cap_rad"]:.3f} rad；仅适用于本次姿态、负载和限制。']
    lines += ["", "速度验收要求：目标到达、物体抬升至少 35 cm、相对 TCP 平移漂移 ≤ 5 mm、旋转漂移 ≤ 5°，并保持 0.5 s。",
              "如设置闭爪后稳定时间，各次在相同的实测 TCP 位姿下保持相同时间，再开始计时抬升。",
              f"{command_hz:g} Hz 限制的是 IK 命令更新频率；步长相对每次实测 TCP 位置限幅。步长乘频率描述目标供给量，实际速度还取决于关节插值、执行器限制和跟踪误差。",
              "本次保留实际物理接触，采用 run003 诊断策略；托盘碰撞诊断单列，稳定速度通过不代表正式路线无碰撞验收通过。", "",
              f'[结构化结果]({(root / "summary.json").as_posix()})', ""]
    for result in results:
        if "evidence_directory" in result:
            evidence = Path(result["evidence_directory"])
            lines += [f'- {result["label"]}：GUI；原生图片 {result.get("visual_evidence", {}).get("images", 0)} 张；'
                      f'[帧与轨迹关联]({(evidence / "execution_visualization/frames.json").as_posix()})；'
                      f'[轨迹数据]({Path(result["trajectory_directory"]).as_posix()})；[详细分析]({(evidence / "speed_analysis.json").as_posix()})。']
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps([dict(label=r["label"], mean_cm_s=r.get("mean_vertical_speed_m_s", 0)*100,
                           accepted=r.get("accepted_speed_trial"), failure=r.get("failure_reason"),
                           images=r.get("visual_evidence", {}).get("images", 0)) for r in results], indent=2))


if __name__ == "__main__":
    main()
