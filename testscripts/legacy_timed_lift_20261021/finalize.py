"""Final change audit and result index; never amend executed-source snapshots."""
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
ROOT=PROJECT/"data/test/trajdata/legacy_timed_lift_20261021_verified/0_to_1"
IDS=["legacy_timed_lift_20261021_pilot","legacy_timed_lift_20261021_validation","legacy_timed_lift_20261021_smooth","legacy_timed_lift_20261021_verified"]

def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    summaries=[read(PROJECT/"data/test/trajdata"/run_id/"0_to_1/summary.json") for run_id in IDS]
    results=[dict(r,run_id=s["run_id"]) for s in summaries for r in s["results"]]
    current=summaries[-1]
    accepted=current["three_run_acceptance"]
    lines=["# 带时间参考 Legacy IK 抬升：实施与留痕", "",
           f'最终三次验收：**{accepted}**；最终批次通过次数 **{current["accepted_count"]}/3**。',
           "本轮只修改 testscripts 隔离实验。主采集入口、训练链和生产配置均未合入；以运行前清单和 GUI 进程退出后的独立 SHA-256 审计核对。", "",
           "## 实施内容", "",
           "- Legacy IK 沿原起终点直线连续求解，使用上个关节解作种子；平滑时间曲线与样条关节参考预先生成。",
           "- 根据 URDF 速度、5 rad/s² 加速度、50 rad/s³ jerk 做预规划时间缩放；1 cm 限制作用于相邻参考点。",
           "- 60 Hz 独立参考时钟、120 Hz 位置伺服；4 cm 超前开始放慢，6 cm 暂停推进；目标保留连续命令状态，不从每次实测位置重建。",
           "- 执行期独立限制关节目标速度/加速度，保存每个物理步的命令；抬升期间不写关节或物体状态。",
           "- 接近/闭爪保持对照策略；闭爪稳定 0.35 s，抬升后保持 0.5 s。",
           "- 规划、执行、图像、独立度量与失败记录全保留。", "",
           "## 参数迭代与实测", "",
           "| 批次/次数 | 参考速度/过渡 | 平均竖直速度 | 抬升仿真时间 | 路径偏差 | 抓取漂移/旋转 | 原生图片 | 速度与携带验收 |",
           "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in results:
        if "planning" not in r or not r["planning"]:continue
        setting=r["planning"]["settings"]
        lines.append(f'| {r["run_id"]}/{r["label"]} | {setting["speed_m_s"]:g} m/s / {setting["ramp_s"]:g} s | '
                     f'{r.get("mean_vertical_speed_m_s",0)*100:.2f} cm/s | {r.get("lift_duration_s",0):.3f} s | '
                     f'{r.get("max_line_deviation_m",0)*1000:.2f} mm | {r.get("max_grasp_translation_drift_m",0)*1000:.2f} mm / '
                     f'{r.get("max_grasp_rotation_drift_deg",0):.2f}° | {r.get("visual_evidence",{}).get("images",0)} | {r.get("accepted")} |')
    lines += ["", "初版 0.25 / 0.25 因关节约束整体时间缩放，抓取稳定但速度不足；0.35 / 0.45 三次速度达标但漂移超标；第三档延长起停过渡。最终版额外将软阈值修正为预测下一参考点，未放宽验收标准。",
              "所有速度、阶段用时按物理仿真时钟计算；规划与三路相机渲染会增加墙钟时间，详细 JSON 中分别记录。", "",
              "## 验收口径与边界", "",
              "目标平均竖直速度 ≥ 20 cm/s，实际直线偏差 ≤ 1 cm，终点位置 ≤ 1 cm/姿态 ≤ 1°；物体升高 ≥ 35 cm；携带平移漂移 ≤ 5 mm、旋转漂移 ≤ 5°，保持 0.5 s。",
              "同时核对参考点增量 ≤ 1 cm、跟踪距离 ≤ 6 cm、60/120 Hz 调度、执行关节目标速度/加速度及预规划 jerk。预规划 jerk 校验与执行期速度/加速度限幅分别记录。",
              "既有托盘代理碰撞诊断仍单列：原物理托盘和接触保留，沿用 run003 诊断策略，未把任何代理豁免写入主逻辑。速度/携带通过不等于正式无碰撞路线验收。", "",
              "## 留痕索引", "",
              f'- [执行方案与修订]({(HERE/"PLAN.md").as_posix()}) · [修改记录]({(HERE/"CHANGELOG.md").as_posix()})。',
              f'- [执行源码统一差异]({(ROOT/"experiment_source/source.patch").as_posix()}) · [执行源码快照]({(ROOT/"experiment_source").as_posix()})。',
              f'- [源码哈希清单]({(ROOT/"source_inventory.json").as_posix()}) · [生产文件前后对比]({(ROOT/"production_unchanged.json").as_posix()})。',
              f'- [独立批后审计器及收尾修正差异]({(ROOT/"audit_source").as_posix()})；执行快照未回填，详见 CHANGELOG v1.3.1。',
              f'- [运行命令与参数]({(ROOT/"status.json").as_posix()}) · [环境]({(ROOT/"environment.json").as_posix()}) · [事件时间线]({(ROOT/"events.jsonl").as_posix()})。',
              f'- [分析代码与适配差异]({(ROOT/"analysis_source").as_posix()}) · [输入及分析哈希]({(ROOT/"analysis_manifest.json").as_posix()})。',
              f'- [单元与真实 IK 数值测试结果]({(ROOT/"unit_test_final.log").as_posix()})。', ""]
    lines += [f'- [全部试验原生选帧及原图哈希]({(ROOT/"visual_audit_index.json").as_posix()}) · '
              f'[最终三次图像总览]({(ROOT/"native_overview_legacy_timed_lift_20261021_verified.png").as_posix()})。', ""]
    for summary in summaries:
        runroot=PROJECT/"data/test/trajdata"/summary["run_id"]/"0_to_1"
        lines += [f'### {summary["run_id"]}',"",
                  f'GUI：Default 桌面可见 Isaac 窗口；源码快照核验 {summary["source_snapshot_verified"]}；生产文件未改动 {summary["production_unchanged"]["unchanged"]}。',
                  f'[该批次完整报告]({(runroot/"report.md").as_posix()})。',""]
        for r in summary["results"]:
            ev,tr=Path(r["evidence_directory"]),Path(r["trajectory_directory"])
            v=r.get("visual_evidence",{})
            lines += [f'- {r["label"]}：三路 RGB {v.get("images",0)} 张，按相机 {v.get("cameras")}，帧关联 {v.get("trace_alignment")}；'
                      f'[图片索引]({(ev/"execution_visualization/frames.json").as_posix()})、[原生关键帧]({(ev/"native_keyframes.png").as_posix()})、'
                      f'[轨迹与逐物理步命令]({tr.as_posix()})、[度量]({(ev/"speed_analysis.json").as_posix()})。']
        lines.append("")
    lines += [f'![速度、跟踪距离与携带漂移]({(ROOT/"comparison.png").as_posix()})',""]
    (ROOT/"final_report.md").write_text("\n".join(lines),encoding="utf-8")
    shutil.copy2(HERE/"unit_test_final.log",ROOT/"unit_test_final.log")
    delivery=ROOT/"delivery_source"
    delivery.mkdir(exist_ok=True)
    for name in ("PLAN.md","CHANGELOG.md","finalize.py","plot_results.py","visual_audit.py","test_timed_reference.py","audit_closed_runs.py","run_experiment.py"):
        shutil.copy2(HERE/name,delivery/name)
    manifest=dict(created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                  source={p.name:sha(p) for p in delivery.iterdir()},report_sha256=sha(ROOT/"final_report.md"),
                  run_ids=IDS,production_unchanged=all(s["production_unchanged"]["unchanged"] for s in summaries),
                  source_snapshot_verified=all(s["source_snapshot_verified"] for s in summaries))
    (ROOT/"delivery_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    (HERE/"README.md").write_text(f'# Legacy IK 带时间参考抬升\n\n[执行方案]({(HERE/"PLAN.md").as_posix()}) · [修改记录]({(HERE/"CHANGELOG.md").as_posix()})\n\n'
                                  f'[本轮完整执行与留痕报告]({(ROOT/"final_report.md").as_posix()})\n\n生产入口未改动。最终三次验收：{accepted}。\n',encoding="utf-8")
    print(json.dumps(dict(three_run_acceptance=accepted,accepted_count=current["accepted_count"],report=str(ROOT/"final_report.md")),ensure_ascii=False))

if __name__=="__main__":main()
