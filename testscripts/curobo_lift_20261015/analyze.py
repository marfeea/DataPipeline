"""Audit measured CuRobo lift, planned motion, grasp stability, and native RGB."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(PROJECT)]
from lift_experiment import line_deviation, normalized_pose
from scripts.pipelines.curobo_bootstrap.planner import JOINT_NAMES, TrajectoryData, save_trajectory, load_trajectory


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def analyze(entry, gui_verified):
    ev, tr = Path(entry["evidence_directory"]), Path(entry["trajectory_directory"])
    if not (ev / "report.json").exists():
        return dict(label=entry["label"], running=True)
    report, trace = read(ev / "report.json"), read(tr / "trace.json")
    result = dict(label=entry["label"], execution_passed=report["passed"], failure_reason=report.get("failure_reason"),
                  planning=report.get("curobo_lift_experiment"), evidence_directory=str(ev), trajectory_directory=str(tr),
                  desktop_gui_verified=gui_verified)
    hz = report["execution_contract"]["control_hz"]
    lift_indices = [i for i, r in enumerate(trace) if r["stage"] == "legacy_lift"]
    if lift_indices:
        first = max(0, lift_indices[0] - 1)
        rows = [trace[first], *[trace[i] for i in lift_indices]]
        base = normalized_pose(report["base_pose_w"][:3], report["base_pose_w"][3:])
        xyz_b = np.asarray([row["tcp_position_b"] for row in rows])
        xyz_w = xyz_b @ base[:3, :3].T + base[:3, 3]
        delta = np.diff(xyz_w, axis=0)
        speed = np.linalg.norm(delta, axis=1) * hz
        vertical = delta[:, 2] * hz
        height = xyz_w[-1, 2] - xyz_w[0, 2]
        progress = (xyz_w[1:, 2] - xyz_w[0, 2]) / height
        steady = vertical[(progress >= 0.2) & (progress <= 0.8)]
        lift_rows = [trace[i] for i in lift_indices]
        reference = np.asarray([row["reference_tcp_position_b"] for row in lift_rows])
        before = np.asarray([row["current_tcp_position_b"] for row in lift_rows])
        goal = result["planning"]["requested_goal_tcp_pose_b"]
        result.update(lift_duration_s=len(lift_indices)/hz, tcp_vertical_distance_m=float(height),
                      mean_vertical_speed_m_s=float(height / (len(lift_indices)/hz)),
                      mean_tcp_path_speed_m_s=float(np.mean(speed)), peak_tcp_path_speed_m_s=float(np.max(speed)),
                      peak_vertical_speed_m_s=float(np.max(vertical)),
                      steady_median_vertical_speed_m_s=float(np.median(steady)) if len(steady) else None,
                      median_actual_displacement_per_record_m=float(np.median(np.linalg.norm(delta, axis=1))),
                      maximum_reference_step_m=float(np.max(np.linalg.norm(np.diff(reference, axis=0), axis=1))),
                      maximum_reference_lead_m=float(np.max(np.linalg.norm(reference - before, axis=1))),
                      median_reference_lead_m=float(np.median(np.linalg.norm(reference - before, axis=1))),
                      max_line_deviation_m=float(max(line_deviation(xyz_b, xyz_b[0], np.asarray(goal[:3])))),
                      actual_max_joint_speed_rad_s=float(np.max(np.abs([r["dq"] for r in lift_rows]))),
                      physical_rate_limited_frames=sum(r["physical_rate_limited"] for r in lift_rows),
                      maximum_joint_tracking_error_rad=float(max(np.max(np.abs(np.asarray(r["commanded_joint_target"]) - r["q"])) for r in lift_rows)),
                      command_frequency_verified=all(r["curobo_physical"] and r["command_hz"] == hz == 60 for r in lift_rows))
        carry = [trace[first], *[r for r in trace if r["stage"] in ("legacy_lift", "legacy_lift_hold")]]
        relative, rotations = [], []
        first_tcp = normalized_pose(carry[0]["tcp_position_b"], carry[0]["tcp_quaternion_b"])
        for row in carry:
            tcp = normalized_pose(row["tcp_position_b"], row["tcp_quaternion_b"])
            obj = normalized_pose(row["target_pose_w"][:3], row["target_pose_w"][3:])
            relative.append(np.linalg.inv(base @ tcp) @ obj)
            rotations.append(np.rad2deg(np.arccos(np.clip((np.trace(first_tcp[:3, :3].T @ tcp[:3, :3])-1)/2, -1, 1))))
        result["max_tcp_orientation_deviation_deg"] = float(max(rotations))
        result["max_grasp_translation_drift_m"] = float(max(np.linalg.norm(x[:3, 3]-relative[0][:3, 3]) for x in relative))
        result["max_grasp_rotation_drift_deg"] = float(max(np.rad2deg(np.arccos(np.clip((np.trace(relative[0][:3, :3].T @ x[:3, :3])-1)/2,-1,1))) for x in relative))
        result["object_vertical_lift_m"] = carry[-1]["target_pose_w"][2] - carry[0]["target_pose_w"][2]
        result["hold_s"] = sum(r["stage"] == "legacy_lift_hold" for r in trace) / hz
        result["stable_carry"] = bool(result["max_grasp_translation_drift_m"] <= .005 and result["max_grasp_rotation_drift_deg"] <= 5
                                       and result["object_vertical_lift_m"] >= .35 and result["hold_s"] >= .5)
        result["straight_path_within_1cm"] = result["max_line_deviation_m"] <= .01
        result["endpoint"] = report.get("lift")
    if len(trace) >= 2:
        q, dq = (np.asarray([r[key] for r in trace]) for key in ("q", "dq"))
        save_trajectory(tr / "trajectory.npz", TrajectoryData(np.arange(len(trace))/hz, q, dq, np.gradient(dq,1/hz,axis=0), JOINT_NAMES))
        result["trajectory_roundtrip"] = np.array_equal(load_trajectory(tr / "trajectory.npz").joint_position, q)
    manifest_path = ev / "execution_visualization/frames.json"
    if manifest_path.exists():
        manifest = read(manifest_path)
        counts, stages, stds = {}, {}, []
        aligned = manifest.get("source_trace_sha256") == hashlib.sha256((tr / "trace.json").read_bytes()).hexdigest()
        for frame in manifest["frames"]:
            with Image.open(manifest_path.parent / frame["rgb"]) as img:
                img.load()
                stds.append(float(np.std(np.asarray(img.convert("RGB"), dtype=float))))
            camera, stage = frame["camera"], frame["execution_stage"]
            counts[camera] = counts.get(camera, 0) + 1
            stages[stage] = stages.get(stage, 0) + 1
            i = frame["trajectory_index"]
            aligned &= 0 <= i < len(trace) and trace[i]["stage"] == stage and abs(frame["time_s"]-i/hz) < 1e-9
        required = ["legacy_approach_pick", "close_gripper", "grasp_settle", "legacy_lift", "legacy_lift_hold"]
        result["visual_evidence"] = dict(images=len(stds), cameras=counts, stages=stages, trace_alignment=bool(aligned),
                                          minimum_rgb_std=min(stds) if stds else None,
                                          complete=bool(stds and min(stds)>5 and aligned and len(counts)==3 and all(x in stages for x in required)))
        actual_stages = [x for x in required if x in stages]
        sheet = Image.new("RGB", (768, 280*len(actual_stages)), "white")
        draw = ImageDraw.Draw(sheet)
        for row, stage in enumerate(actual_stages):
            for column, camera in enumerate(counts):
                frame = [f for f in manifest["frames"] if f["execution_stage"]==stage and f["camera"]==camera][-1]
                with Image.open(manifest_path.parent / frame["rgb"]) as img:
                    sheet.paste(img.convert("RGB").resize((256,256)), (column*256,row*280))
                draw.text((column*256+3,row*280+257), f'{stage.replace("legacy_", "")} | {camera} #{frame["trajectory_index"]}', fill="black")
        sheet.save(ev / "native_keyframes.png")
    result["full_tray_collision_diagnostic_passed"] = report.get("tray_collision_diagnostic", {}).get("passed")
    result["accepted_physical_speed_trial"] = bool(report["passed"] and result.get("stable_carry") and result.get("trajectory_roundtrip")
                                                  and gui_verified and result.get("visual_evidence",{}).get("complete"))
    result["reference_step_within_1cm"] = bool(result.get("maximum_reference_step_m", float("inf")) <= .01 + 1e-6)
    endpoint = result.get("endpoint") or {}
    result["accepted_same_path_speed_trial"] = bool(result["accepted_physical_speed_trial"] and result.get("straight_path_within_1cm")
                                                    and endpoint.get("final_tcp_error_m",float("inf")) <= .01
                                                    and endpoint.get("final_tcp_orientation_error_rad",float("inf")) <= np.deg2rad(1))
    (ev / "speed_analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="curobo_lift_20261015")
    args = parser.parse_args()
    root = PROJECT / "data/test/trajdata" / args.run_id / "0_to_1"
    status = read(root / "status.json")
    gui = read(root / "desktop_gui.json")
    gui_verified = gui["desktop"] == "Default" and any(w["visible"] and "Isaac" in w["title"] for w in gui["windows"])
    results = [analyze(entry,gui_verified) for entry in status["trials"]]
    summary = dict(desktop_gui=gui, results=results,
                   scope="same endpoint CuRobo planning and physical servo execution; collision diagnostic scope reported separately")
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# CuRobo 抬升规划与物理执行", "",
             "复用相同接近、抓取、闭爪稳定 0.35 s，原 lift 目标位置/姿态。抬升采用实际关节位置伺服，物理 120 Hz、目标更新/记录 60 Hz，无速度前馈、无抬升状态强写。", "",
             "所有速度/抬升时长按物理仿真时间（帧数 / 60）计算，不含规划、启动或渲染等待的墙钟耗时，也不代表实机测试。", "",
             "| 方案 | 规划时长 | 实测抬升时长 | 实测平均竖直速度 | 规划 TCP 峰值 | 实测 TCP 峰值 | 规划 / 实测直线偏差 | 稳定携带 |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in results:
        if "mean_vertical_speed_m_s" in r:
            p = r["planning"]
            lines.append(f'| {r["label"]} | {p["planned_duration_s"]:.3f} s | {r["lift_duration_s"]:.3f} s | '
                         f'{r["mean_vertical_speed_m_s"]*100:.2f} cm/s | {p["planned_tcp_peak_speed_m_s"]*100:.2f} cm/s | '
                         f'{r["peak_tcp_path_speed_m_s"]*100:.2f} cm/s | {p["planned_max_line_deviation_m"]*1000:.2f} / '
                         f'{r["max_line_deviation_m"]*1000:.2f} mm | {r["stable_carry"]} |')
        else:
            lines.append(f'| {r["label"]} | — | — | — | — | — | — | {r.get("failure_reason", "运行中")} |')
    lines += ["", "上一轮 Legacy IK：1 cm / 60 Hz 平均 5.35 cm/s（2 个物理步插值）或 6.96 cm/s（1 个物理步插值）；5 cm / 4 Hz 测得最快稳定平均 17.17 cm/s。",
              "CuRobo 生成含时间、关节位置、速度和加速度的完整轨迹。这里实际只发送位置目标，仍可通过连续推进参考轨迹获得较大跟踪驱动力；计划 dq/ddq 用于分析，未作为物理速度状态写入。",
              "现有旧接触探针的 CuRobo 段使用 write_joint_state_to_sim(q,dq) 及逻辑物体位姿更新，属于状态重放；其速度不可直接当作相同物理伺服下可达到的速度。",
              "标准关节空间规划只固定起终点；pose 增加姿态和 Y 方向代价；line 用同原点虚拟工具坐标系锁定原起终点连线的横向投影，保持原物理 TCP 目标。直线偏差独立测量。",
              "完整世界起终点检查及移除 support_tray_11 代理的诊断范围均见各组详细结果。物理场景不变，诊断速度通过不代表正式无碰撞路线验收通过。", ""]
    for r in results:
        ev, tr = Path(r.get("evidence_directory", root/r["label"])), Path(r.get("trajectory_directory", ""))
        lines += [f'## {r["label"]}', "", f'执行结果：{r.get("execution_passed")}；失败原因：{r.get("failure_reason")}。',
                  f'三路原生图片：{r.get("visual_evidence",{}).get("images",0)} 张；GUI：{r.get("desktop_gui_verified")}。',
                  f'[详细分析]({(ev/"speed_analysis.json").as_posix()}) · [轨迹与规划数据]({tr.as_posix()}) · '
                  f'[图片帧关联]({(ev/"execution_visualization/frames.json").as_posix()})', ""]
        if (ev/"native_keyframes.png").exists():
            lines += [f'![三路原生关键帧]({(ev/"native_keyframes.png").as_posix()})', ""]
    (root/"report.md").write_text("\n".join(lines), encoding="utf-8")
    (root/"experiment_source/analyze.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
