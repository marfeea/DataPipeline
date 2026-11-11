"""Consolidate the physical CuRobo comparison without rewriting historical data."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
EVIDENCE = PROJECT/"data/test/trajdata"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaled-run-id",default="curobo_lift_20261015_line_scaled")
    args = parser.parse_args()
    root = EVIDENCE/args.scaled_run_id/"0_to_1"
    run_ids = ["curobo_lift_20261015_retry02","curobo_lift_20261015_line_retry01",args.scaled_run_id]
    results = []
    for run_id in run_ids:
        results.extend(dict(r, run_id=run_id) for r in read(EVIDENCE/run_id/"0_to_1/summary.json")["results"])
    eligible = [r for r in results if r.get("accepted_same_path_speed_trial") and r.get("reference_step_within_1cm")]
    best = max(eligible,key=lambda r:r["mean_vertical_speed_m_s"]) if eligible else None
    combined = dict(results=results,best_same_path_stable_reference_step_within_1cm=best,
                    note="Tested diagnostic scope only. Existing support_tray_11 proxy conflict remains; physical scene unchanged.")
    (root/"combined_summary.json").write_text(json.dumps(combined,indent=2),encoding="utf-8")
    lines = ["# CuRobo 同路径抬升：规划与物理执行结论", "",
             "所有可比较的试验都复用源 0、候选 01、相同接近/闭爪与 0.35 s 闭爪稳定；抬升原目标位置与姿态保持不变，物理 120 Hz、命令/测量 60 Hz。",
             "速度及抬升用时按物理仿真时间统计（记录帧数 / 60），不包含 GUI 渲染等待、启动和规划的墙钟耗时；不表示 GUI 能实时播放，也不是实机速度测试。",
             "本次 CuRobo 仅在抬升前计算带时间的关节轨迹，执行时发送关节位置目标，通过 PhysX 与真实抓取接触运动。未将规划 dq 直接写入物理关节速度，未在抬升期间写机器人/物体状态。", "",
             "| 模式 | 时间倍率 | 平均竖直速度 | TCP 峰值速度 | 仿真抬升用时 | 直线偏差 | 抓取漂移 / 旋转 | 稳定携带 | 参考增量≤1cm |",
             "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in results:
        if "mean_vertical_speed_m_s" not in r:
            continue
        p = r["planning"]
        lines.append(f'| {p["planning_mode"]} | {p["time_scale"]:g} | {r["mean_vertical_speed_m_s"]*100:.2f} cm/s | '
                     f'{r["peak_tcp_path_speed_m_s"]*100:.2f} cm/s | {r["lift_duration_s"]:.3f} s | '
                     f'{r["max_line_deviation_m"]*1000:.2f} mm | {r["max_grasp_translation_drift_m"]*1000:.2f} mm / '
                     f'{r["max_grasp_rotation_drift_deg"]:.2f}° | {r["stable_carry"]} | {r["reference_step_within_1cm"]} |')
    lines += ["", "Legacy IK 先前最快稳定结果：1 cm / 60 Hz 为平均 6.96 cm/s、用时 6.117 s；5 cm / 4 Hz 为平均 17.17 cm/s、用时 2.483 s。", ""]
    if best:
        p = best["planning"]
        lines += [f'本轮同路径、抓取稳定、参考点间位移 ≤ 1 cm 的最快测试档：line × {p["time_scale"]:g}；'
                  f'平均 {best["mean_vertical_speed_m_s"]*100:.2f} cm/s，TCP 峰值 {best["peak_tcp_path_speed_m_s"]*100:.2f} cm/s，'
                  f'物体抬升 {best["object_vertical_lift_m"]*100:.2f} cm，保持 {best["hold_s"]:.2f} s。',
                  f'规划原始时长 {p["planned_duration_s"]:.3f} s，执行包含时间缩放与到位等待共 {best["lift_duration_s"]:.3f} s；'
                  f'原始规划 TCP 速度中位值 {p["planned_tcp_median_speed_m_s"]*100:.2f} cm/s。',
                  f'相邻参考点最大位移 {best["maximum_reference_step_m"]*1000:.2f} mm；参考目标相对实测 TCP 最大超前 '
                  f'{best["maximum_reference_lead_m"]*100:.2f} cm；实际每个 60 Hz 周期位移中位值 '
                  f'{best["median_actual_displacement_per_record_m"]*1000:.2f} mm。',
                  f'终点位置误差 {best["endpoint"]["final_tcp_error_m"]*1000:.2f} mm，终点姿态误差 '
                  f'{best["endpoint"]["final_tcp_orientation_error_rad"]*180/3.141592653589793:.3f}°；'
                  f'全段相对初始姿态最大偏差 {best["max_tcp_orientation_deviation_deg"]:.3f}°。'
                  '验收中的 1° 姿态容差用于终点，全段姿态偏差独立报告；并非严格零误差的笛卡尔路径。', ""]
    lines += ["## 规划方式的区别", "",
              "CuRobo 的 MotionGen 同时优化关节轨迹与时间步长，带关节速度、加速度和 jerk 限制。本工程规划器加载的关节速度限值为前三轴 2.618 rad/s、后三轴 3.14 rad/s，加速度 5 rad/s²、jerk 50 rad/s³。TCP 速度由关节轨迹与 FK/Jacobian 决定，没有直接设成统一的 TCP 恒速。",
              "当前 Legacy IK 的目标为每次实测位置 + 最多 1 cm；伺服滞后时未走完的距离被下一次目标重建吸收。CuRobo 执行参考为 q_ref(t)，沿预先规划的时间表持续推进，允许参考领先实测位置，因此即使只发位置目标，也能建立更大的位置误差来驱动运动。",
              "需要分别定义：相邻参考点增量、参考与实际的跟踪误差、关节速度/加速度限制。把相邻参考点限制为 1 cm，与把参考限定在实测 TCP 前方 1 cm，是不同约束。借鉴 CuRobo 时应保留一条持续推进的参考轨迹，并单独设置跟踪误差的处理，不能只把速度中位值换算成 current + v/Hz。", "",
              "## 边界与证据", "",
              "完整世界中抓取起点仍有 support_tray_11 代理碰撞。本轮诊断规划仅移除该代理，物理托盘、接触和场景均保留；完整托盘碰撞诊断仍不通过。因此速度与稳定性结果不等于正式路线无碰撞通过。",
              "最初单球附着代理还与 support_tray_12 有冲突，停止于抬升前；改为四个保守 AABB 网格单元包围球后消除额外冲突，没有扩大托盘豁免。相机黑屏的初始化尝试均保留日志和图片，未执行运动，不计入速度结果。",
              "原接触探针中的 CuRobo 段直接写关节状态及逻辑物体位姿，属于状态重放；本轮物理速度另行测量，不能将两者混为同一执行链。",
              "实验仅位于 testscripts；主采集入口、主配置与历史数据未更改。", ""]
    for r in results:
        ev, tr = Path(r["evidence_directory"]), Path(r["trajectory_directory"])
        v = r.get("visual_evidence",{})
        lines += [f'- {r["run_id"]}/{r["label"]}：Default 桌面可见 GUI={r["desktop_gui_verified"]}；'
                  f'三路原生 RGB 共 {v.get("images",0)} 张，按相机 {v.get("cameras")}，帧关联检查 {v.get("trace_alignment")}；'
                  f'[详细分析]({(ev/"speed_analysis.json").as_posix()})、[原生关键帧]({(ev/"native_keyframes.png").as_posix()})、'
                  f'[图片帧关联]({(ev/"execution_visualization/frames.json").as_posix()})、[轨迹]({tr.as_posix()})。']
    lines += ["",f'![本轮时间缩放对比]({(root/"planned_vs_measured.png").as_posix()})',""]
    (root/"final_report.md").write_text("\n".join(lines),encoding="utf-8")
    for name in ("finalize.py","plot_comparison.py","test_attachment_cover.py"):
        (root/"experiment_source"/name).write_bytes((HERE/name).read_bytes())
    print(json.dumps(dict(best=best,report=str(root/"final_report.md")),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
