"""Combine both completed GUI sweeps and build a native-render overview."""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data/test/trajdata"
run_ids = ["legacy_lift_speed_20261015_retry01", "legacy_lift_speed_20261015_fast"]
results = []
for run_id in run_ids:
    summary = json.loads((DATA / run_id / "0_to_1/summary.json").read_text(encoding="utf-8"))
    for result in summary["results"]:
        result["run_id"] = run_id
        results.append(result)
accepted = [result for result in results if result["accepted_speed_trial"]]
best = max(accepted, key=lambda result: result["mean_vertical_speed_m_s"])
peak = max(accepted, key=lambda result: result["peak_tcp_path_speed_m_s"])
output = DATA / "legacy_lift_speed_20261015_fast/0_to_1"
baseline = results[0]
metadata = json.loads((DATA / run_ids[-1] / "0_to_1/status.json").read_text(encoding="utf-8"))
upstream = Path(metadata["generated"]["upstream"])
combined = dict(scope="measured source-0 lift benchmark; not a global kinematic maximum or full pick/place acceptance",
                nominal_reference_speed_m_s=0.2, best_average=best, highest_peak=peak,
                all_trials=results, total_images=sum(r["visual_evidence"]["images"] for r in results),
                source_hash_normalization="UTF-8 source text with universal newline normalization",
                production_probe_sha256_unchanged=hashlib.sha256(upstream.read_text(encoding="utf-8").encode()).hexdigest() == metadata["generated"]["upstream_sha256"],
                recommended_experimental_settings=dict(ik_command_hz=4, tcp_displacement_limit_m=0.05,
                    joint_delta_limit_rad=best["joint_cap_rad"],
                    interpolation_physics_steps=best["interpolation_physics_steps"],
                    interpolation_duration_s=best["interpolation_duration_s"],
                    grasp_settle_s=best["grasp_settle_s"], physics_hz=120, recording_hz=60,
                    stage_speed_cap=False, legacy_tracking_lead_cap=False),
                startup_failure=dict(run_id="legacy_lift_speed_20261015", attempts=4,
                    images=0, trajectory_records=0,
                    reason="experimental reset adapter omitted joint_ids; corrected before motion sweeps"))
(output / "combined_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
sheet = Image.new("RGB", (3 * 256, len(results) * 284), "white")
draw = ImageDraw.Draw(sheet)
overview_frames = []
for row, result in enumerate(results):
    visual = Path(result["visual_evidence"]["manifest"])
    manifest = json.loads(visual.read_text(encoding="utf-8"))
    candidates = [f for f in manifest["frames"] if f["camera"] == "CameraSensor_2"]
    phases = ["grasp_settle" if result["grasp_settle_s"] else "close_gripper", "legacy_lift", "legacy_lift_hold"]
    for column, phase in enumerate(phases):
        selected = [f for f in candidates if f["execution_stage"] == phase]
        if not selected:
            raise RuntimeError(f"Missing native phase image: {result['label']} {phase}")
        frame = selected[len(selected) // 2] if phase == "legacy_lift" else selected[-1]
        image_path = visual.parent / frame["rgb"]
        with Image.open(image_path) as image:
            sheet.paste(image.convert("RGB").resize((256, 256)), (column * 256, row * 284))
        draw.text((column * 256 + 3, row * 284 + 258),
                  f'q={result["joint_cap_rad"]:.3f} t={result["interpolation_duration_s"]:.3f} {phase.replace("legacy_", "")}', fill="black")
        overview_frames.append(dict(run_id=result["run_id"], label=result["label"], phase=phase,
                                    source_image=str(image_path), trajectory_index=frame["trajectory_index"],
                                    time_s=frame["time_s"]))
sheet.save(output / "all_trials_native_overview.png")
(output / "overview_frames.json").write_text(json.dumps(overview_frames, indent=2), encoding="utf-8")
lines = ["# Legacy IK 5 cm / 4 Hz 抬升速度实测", "",
         f'本次在当前标定的源位置 0 测得最快稳定全段平均竖直速度 **{best["mean_vertical_speed_m_s"]*100:.2f} cm/s**，'
         f'抬升 {best["tcp_vertical_distance_m"]*100:.2f} cm 用时 **{best["lift_duration_s"]:.3f} s**。',
         f'本次最高 TCP 瞬时峰值 **{peak["peak_tcp_path_speed_m_s"]*100:.2f} cm/s**，与最快平均速度对应不同插值档位。', "",
         "固定 IK 命令 4 Hz、每条 TCP 目标相对实测位置总增量 ≤ 5 cm；物理 120 Hz，记录 60 Hz。",
         "20 cm/s 是目标位移的名义平均供给上限；关节可在命令周期的一部分时间内完成推进，因此不是 TCP 瞬时速度硬上限。", "",
         "第一轮保留相同接近、闭爪过程，仅放宽关节软件增量上限；第二轮固定 0.785 rad 上限、闭爪后稳定握持 0.35 s，对比关节插值时长。",
         "实验取消了抬升写死的 2/5 cm/s 阶段速度、1 mm 上限和 2 cm 跟踪超前限制；执行器速度与关节位置限制仍有效。", "",
         "| 批次 | 关节增量上限 | 插值时长 | 抬升用时 | 平均竖直速度 | TCP 峰值 | 抓取漂移 | 旋转漂移 | 稳定速度验收 | GUI 图片 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
for result in results:
    lines.append(f'| {"关节限幅" if result["run_id"] == run_ids[0] else "插值/稳定握持"} | {result["joint_cap_rad"]:.3f} rad | '
                 f'{result["interpolation_duration_s"]:.4f} s | {result["lift_duration_s"]:.3f} s | '
                 f'{result["mean_vertical_speed_m_s"]*100:.2f} cm/s | {result["peak_tcp_path_speed_m_s"]*100:.2f} cm/s | '
                 f'{result["max_grasp_translation_drift_m"]*1000:.2f} mm | {result["max_grasp_rotation_drift_deg"]:.2f}° | '
                 f'{result["accepted_speed_trial"]} | {result["visual_evidence"]["images"]} |')
lines += ["", "较高关节增量档位在 0.25 s 插值下出现 11.51 cm/s 平台，关节增量限幅已不触发。",
          "将插值缩短到约 0.067 s 后平均速度达到 17.17 cm/s；继续缩到约 0.017 s 后为 17.07 cm/s，平均速度未继续增长。",
          f'相比保留 0.08 rad 上限的 {baseline["mean_vertical_speed_m_s"]*100:.2f} cm/s，最快平均速度提高约 '
          f'{best["mean_vertical_speed_m_s"]/baseline["mean_vertical_speed_m_s"]:.2f} 倍。', "",
          "## 最快平均速度对应的实验设置", "",
          "- IK 命令频率：4 Hz；每次 TCP 总位移上限：5 cm。",
          f'- 每条抬升命令的关节软件增量上限：{best["joint_cap_rad"]:.3f} rad。',
          f'- 关节目标插值：{best["interpolation_physics_steps"]} 个物理步，即 {best["interpolation_duration_s"]:.6f} s；剩余周期保持目标。',
          f'- 闭爪后保持实测 TCP 位姿 {best["grasp_settle_s"]:.2f} s，再抬升；终点保持 {best["hold_s"]:.2f} s。',
          f'- 该次物体抬升 {best["object_vertical_lift_m"]*100:.2f} cm，最大相对平移漂移 {best["max_grasp_translation_drift_m"]*1000:.2f} mm，'
          f'最大相对旋转漂移 {best["max_grasp_rotation_drift_deg"]:.2f}°。', "",
          "## 验收边界", "",
          "稳定速度验收要求：到达目标容差 1 cm、物体竖直升高 ≥ 35 cm、相对 TCP 平移漂移 ≤ 5 mm、旋转漂移 ≤ 5°、终点保持 0.5 s。",
          "所有八条运动试验均在 Default 桌面的可见 Isaac GUI 内完成，三路 RGB 非灰屏、文件可读、时间与帧索引和轨迹逐条对应。",
          f'运动试验累计原生图片 {combined["total_images"]} 张。最初四个启动尝试因实验复位接口错误未进入运动，无图片；原失败记录保留。',
          "托盘碰撞诊断仍为阳性；本次只验收受约束的 Legacy IK 抬升速度和稳定携带，不代表正式无碰撞路线或完整搬运放置通过。",
          "本结果是当前源位置、负载和限制下的实测值，不是整个工作空间的全局最快速度。", "",
          "## 代码与合入范围", "",
          "所有新增实验代码位于 testscripts/legacy_lift_speed_20261015；未合入主采集入口或主配置。",
          f'原主探针源码 SHA256 与实验开始时一致：{combined["production_probe_sha256_unchanged"]}。',
          "拟合入内容可限于抬升的 4 Hz/5 cm 策略、实验选定的关节增量与插值，以及闭爪后 0.35 s 稳定握持；接近速度和碰撞验收不在拟合入范围。", "",
          "## 轨迹、日志和原生图片", "",
          f'[完整结构化结果]({(output / "combined_summary.json").as_posix()})',
          f'[八次原生渲染概览]({(output / "all_trials_native_overview.png").as_posix()})',
          f'[概览图片对应原始帧]({(output / "overview_frames.json").as_posix()})', ""]
for result in results:
    evidence = Path(result["evidence_directory"])
    lines.append(f'- {result["run_id"]}/{result["label"]}：GUI、{result["visual_evidence"]["images"]} 张图片；'
                 f'[轨迹]({Path(result["trajectory_directory"]).as_posix()})；[日志和图片]({evidence.as_posix()})；'
                 f'[帧关联]({result["visual_evidence"]["manifest"]})。')
lines += ["", f'[启动失败记录]({(DATA / "legacy_lift_speed_20261015/0_to_1/status.json").as_posix()})', ""]
(output / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(dict(best_average_cm_s=best["mean_vertical_speed_m_s"]*100,
                      highest_peak_cm_s=peak["peak_tcp_path_speed_m_s"]*100,
                      total_images=combined["total_images"], production_unchanged=combined["production_probe_sha256_unchanged"],
                      report=str(output / "final_report.md")), indent=2))
