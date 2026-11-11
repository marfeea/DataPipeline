"""Compare the 1 cm / 60 Hz GUI experiment with the prior 5 cm / 4 Hz sweep."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ROOT = PROJECT / "data/test/trajdata/legacy_lift_1cm_60hz_20261015/0_to_1"
PRIOR = PROJECT / "data/test/trajdata/legacy_lift_speed_20261015_fast/0_to_1/summary.json"


def main():
    current = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))["best"]
    rows = [("5 cm / 4 Hz（上一轮最快稳定）", prior)]
    rows.extend((f'1 cm / 60 Hz（{r["interpolation_physics_steps"]} 个物理步插值）', r)
                for r in current["results"])
    lines = ["# 1 cm / 60 Hz 抬升速度对比", "",
             "同一源位置 0、候选 01、相同接近与闭爪，闭爪后稳定 0.35 s，抬升后保持 0.5 s。物理 120 Hz、测量 60 Hz。",
             "仅在 testscripts 隔离副本更改抬升/保持，主采集入口和旧数据均未改写。", "",
             "| 配置 | 插值时间 | 抬升时间 | 平均竖直速度 | 稳态竖直中位速度 | TCP 路径峰值 | 相对平移漂移 / 旋转漂移 | 稳定速度验收 |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for label, result in rows:
        lines.append(f'| {label} | {result["interpolation_duration_s"]*1000:.2f} ms | '
                     f'{result["lift_duration_s"]:.3f} s | {result["mean_vertical_speed_m_s"]*100:.2f} cm/s | '
                     f'{result["steady_median_vertical_speed_m_s"]*100:.2f} cm/s | '
                     f'{result["peak_tcp_path_speed_m_s"]*100:.2f} cm/s | '
                     f'{result["max_grasp_translation_drift_m"]*1000:.2f} mm / '
                     f'{result["max_grasp_rotation_drift_deg"]:.2f}° | {result["accepted_speed_trial"]} |')
    lines += ["", "1 cm × 60 Hz = 0.60 m/s 是目标供给量。每次目标都从实测 TCP 向最终目标推进至多 1 cm，"
              "没有按时间累加一个持续前行的参考轨迹；执行器未跟上时，频率提高并不会让目标超前距离累积。",
              "软件关节增量上限沿用 0.785 rad，底层关节目标推进速度限制仍为 3.14 rad/s；实际物理执行还受驱动器与接触影响。",
              "平均速度包含起步、减速及到位停留，以实际世界 Z 位移 / 抬升阶段时间计算；峰值由 60 Hz 实测 TCP 位移计算，未测量更高频的瞬时峰值。",
              "稳定携带标准：到达目标容差、物体升高 ≥ 35 cm、相对 TCP 平移漂移 ≤ 5 mm、旋转漂移 ≤ 5°、保持 ≥ 0.5 s。",
              "试验采用 run003 诊断策略，保留物理接触；托盘代理碰撞诊断不作为本次速度门禁，不能代替完整路线无碰撞验收。", ""]
    for result in current["results"]:
        ev = Path(result["evidence_directory"])
        lines += [f'## {result["label"]}', "",
                  f'实际 GUI：Default 桌面可见窗口；三路原生 RGB 共 {result["visual_evidence"]["images"]} 张，'
                  f'帧与轨迹关联检查 {result["visual_evidence"]["trace_alignment"]}。',
                  f'命令频率检查 {result["command_frequency_verified"]}；更新间隔 {result["update_spacing_record_frames"]} 个记录帧；'
                  f'最大 TCP 目标增量 {result["max_command_displacement_m"]*1000:.4f} mm；'
                  f'IK 目标一致性检查 {result["ik_target_matches_command"]}。', "",
                  f'每个记录帧的实际 TCP 位移中位值 {result["median_actual_tcp_displacement_per_record_m"]*1000:.4f} mm；'
                  f'最大实测关节速度 {result["actual_max_joint_speed_rad_s"]:.4f} rad/s；'
                  f'关节目标与 IK 解的最大差 {result["max_commanded_vs_ik_joint_target_error_rad"]:.6g} rad；'
                  f'关节目标与实测值的最大关节误差中位值 {result["median_max_joint_tracking_error_rad"]:.6g} rad。', "",
                  f'[详细分析]({(ev / "speed_analysis.json").as_posix()}) · '
                  f'[图片与轨迹帧关联]({(ev / "execution_visualization/frames.json").as_posix()}) · '
                  f'[轨迹数据]({Path(result["trajectory_directory"]).as_posix()})', "",
                  f'![本次原生三路关键帧]({(ev / "native_keyframes.png").as_posix()})', ""]
    (ROOT / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(current["results"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
