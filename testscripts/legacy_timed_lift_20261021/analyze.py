"""Auditable physical measurement reuse, with independent timed-reference checks."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
MEASUREMENT=HERE.parent/"curobo_lift_20261015/analyze.py"
sys.path[:0]=[str(PROJECT),str(MEASUREMENT.parent)]


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def write(path,value): Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",default="legacy_timed_lift_20261021")
    args=parser.parse_args()
    root=PROJECT/"data/test/trajdata"/args.run_id/"0_to_1"
    status=read(root/"status.json")
    analysis=root/"analysis_source"
    analysis.mkdir(exist_ok=True)
    original=MEASUREMENT.read_text(encoding="utf-8")
    adapted=original.replace('"curobo_lift_experiment"','"timed_lift_experiment"').replace('"curobo_physical"','"timed_lift"')
    (analysis/"measurement_upstream.py").write_text(original,encoding="utf-8")
    (analysis/"measurement_adapter.py").write_text(adapted,encoding="utf-8")
    (analysis/"measurement.patch").write_text("".join(difflib.unified_diff(original.splitlines(True),adapted.splitlines(True),
                                                                fromfile="prior_measurement.py",tofile="timed_measurement.py")),encoding="utf-8")
    (analysis/"analyze.py").write_bytes(Path(__file__).read_bytes())
    scope={"__name__":"timed_measurement_adapter","__file__":str(MEASUREMENT)}
    exec(compile(adapted,str(analysis/"measurement_adapter.py"),"exec"),scope)
    gui=read(root/"desktop_gui.json") if (root/"desktop_gui.json").exists() else {}
    gui_ok=gui.get("desktop")=="Default" and any(w["visible"] and "Isaac" in w["title"] for w in gui.get("windows",[]))
    results=[]
    for entry in status["trials"]:
        ev,tr=Path(entry["evidence_directory"]),Path(entry["trajectory_directory"])
        if not (ev/"report.json").exists():
            results.append(dict(label=entry["label"],running=True));continue
        report=read(ev/"report.json")
        if not report.get("record_count"):
            results.append(dict(label=entry["label"],execution_passed=False,failure_reason=report.get("failure_reason"),
                                evidence_directory=str(ev),trajectory_directory=str(tr),accepted=False,
                                visual_evidence=dict(images=0,complete=False)))
            continue
        result=scope["analyze"](entry,gui_ok)
        trace=read(tr/"trace.json")
        rows=[r for r in trace if r.get("timed_lift")]
        history=read(tr/"physical_commands.json") if (tr/"physical_commands.json").exists() else []
        metadata=report.get("timed_lift_experiment",{})
        if rows and history:
            command=np.asarray([r["commanded_joint_target"] for r in history])
            initial=metadata.get("start_q",trace[next(i for i,r in enumerate(trace) if r.get("timed_lift"))-1]["q"])
            velocity=np.diff(np.vstack((initial,command)),axis=0)*120
            acceleration=np.diff(np.vstack((np.zeros(6),velocity)),axis=0)*120
            vmax=np.asarray(metadata["velocity_limit_rad_s"])
            amax=np.asarray(metadata["acceleration_limit_rad_s2"])
            leads=np.asarray([np.linalg.norm(np.asarray(r["reference_tcp_position_b"])-r["current_tcp_position_b"]) for r in rows])
            result["timed_contract"]=dict(
                physical_rows=len(history),control_rows=len(rows),physics_decimation_verified=len(history)==2*len(rows),
                commanded_velocity_max_rad_s=np.max(np.abs(velocity),axis=0).tolist(),
                commanded_acceleration_max_rad_s2=np.max(np.abs(acceleration),axis=0).tolist(),
                commanded_velocity_verified=bool(np.all(np.abs(velocity)<=vmax+1e-6)),
                commanded_acceleration_verified=bool(np.all(np.abs(acceleration)<=amax+1e-5)),
                reference_clock_monotone=bool(np.all(np.diff([r["reference_time_s"] for r in rows])>=-1e-10)),
                tracking_lead_limit_verified=bool(np.max(leads)<=.06+1e-5),
                maximum_tracking_lead_m=float(np.max(leads)),
                governor_active_frames=sum(r["governor_active"] for r in rows),
                command_limited_physics_steps=sum(r["limited"] for r in history),
                planned_velocity_verified=bool(np.all(np.asarray(metadata["planned_max_velocity_rad_s"])<=vmax+1e-6)),
                planned_acceleration_verified=bool(np.all(np.asarray(metadata["planned_max_acceleration_rad_s2"])<=amax+1e-5)),
                planned_jerk_verified=bool(np.max(metadata["planned_max_jerk_rad_s3"])<=50+1e-4),
                planning_wall_s=metadata.get("planning_wall_s"),execution_wall_s=metadata.get("execution_wall_s"))
            gate=result["timed_contract"]
            keys=["physics_decimation_verified","commanded_velocity_verified","commanded_acceleration_verified",
                  "reference_clock_monotone","tracking_lead_limit_verified","planned_velocity_verified",
                  "planned_acceleration_verified","planned_jerk_verified"]
            result["functional_pass"]=bool(result.get("accepted_same_path_speed_trial") and result.get("reference_step_within_1cm")
                                             and result.get("command_frequency_verified") and all(gate[k] for k in keys))
            result["speed_target_pass"]=result.get("mean_vertical_speed_m_s",0)>=.2
            result["accepted"]=bool(result["functional_pass"] and result["speed_target_pass"])
        else:
            result.update(functional_pass=False,speed_target_pass=False,accepted=False)
        write(ev/"speed_analysis.json",result)
        results.append(result)
    source_inventory=read(root/"source_inventory.json")
    source_ok=all(sha(root/"experiment_source"/p)==h for p,h in source_inventory.items())
    prod=read(root/"production_unchanged.json") if (root/"production_unchanged.json").exists() else {"unchanged":None}
    accepted=[r for r in results if r.get("accepted")]
    summary=dict(run_id=args.run_id,results=results,desktop_gui=gui,source_snapshot_verified=source_ok,
                 production_unchanged=prod,three_run_acceptance=bool(len(results)==3 and len(accepted)==3),
                 accepted_count=len(accepted),best=max(accepted,key=lambda r:r["mean_vertical_speed_m_s"]) if accepted else None)
    write(root/"summary.json",summary)
    write(root/"analysis_manifest.json",dict(source={p.name:sha(p) for p in analysis.iterdir()},
                                           traces={r["label"]:sha(Path(r["trajectory_directory"])/"trace.json") for r in results if r.get("trajectory_directory") and (Path(r["trajectory_directory"])/"trace.json").exists()}))
    lines=["# Legacy IK 带时间参考抬升执行报告","",
           f'批次 `{args.run_id}`；GUI；源码快照校验 {source_ok}；生产文件未改动 {prod["unchanged"]}。',
           "速度与阶段时间按 60 Hz 物理仿真时间统计；规划与渲染的墙钟耗时单列。", "",
           "| 次数 | 平均竖直速度 | TCP 峰值 | 抬升仿真时间 | 路径偏差 | 抓取平移 / 旋转漂移 | 图片 | 功能验收 | ≥20cm/s |",
           "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in results:
        if "mean_vertical_speed_m_s" in r:
            lines.append(f'| {r["label"]} | {r["mean_vertical_speed_m_s"]*100:.2f} cm/s | {r["peak_tcp_path_speed_m_s"]*100:.2f} cm/s | '
                         f'{r["lift_duration_s"]:.3f} s | {r["max_line_deviation_m"]*1000:.2f} mm | '
                         f'{r["max_grasp_translation_drift_m"]*1000:.2f} mm / {r["max_grasp_rotation_drift_deg"]:.2f}° | '
                         f'{r["visual_evidence"]["images"]} | {r["functional_pass"]} | {r["speed_target_pass"]} |')
        else: lines.append(f'| {r["label"]} | — | — | — | — | — | 0 | {r.get("failure_reason","运行中")} | — |')
    lines += ["",f'连续三次验收：{summary["three_run_acceptance"]}；已通过次数：{len(accepted)}。',
              "本轮完全由 Legacy IK 求解抬升参考，没有调用 CuRobo MotionGen。CuRobo 仅用于原有独立托盘碰撞诊断。",
              "复用 run003 接触诊断策略，物理场景/接触保留；完整托盘碰撞门禁单列，速度通过不代表正式采集可接受。", ""]
    for r in results:
        if "evidence_directory" not in r: continue
        ev,tr=Path(r["evidence_directory"]),Path(r["trajectory_directory"])
        lines += [f'## {r["label"]}',"",f'执行：{r.get("execution_passed")}；失败原因：{r.get("failure_reason")}。',
                  f'GUI Default 可见窗口核验：{gui_ok}；原生三路 RGB {r.get("visual_evidence",{}).get("images",0)} 张；'
                  f'相机数量 {r.get("visual_evidence",{}).get("cameras")}；帧关联 {r.get("visual_evidence",{}).get("trace_alignment")}。',
                  f'[详细度量]({(ev/"speed_analysis.json").as_posix()}) · [帧关联]({(ev/"execution_visualization/frames.json").as_posix()}) · '
                  f'[轨迹与参考]({tr.as_posix()})',""]
        if (ev/"native_keyframes.png").exists(): lines += [f'![原生关键帧]({(ev/"native_keyframes.png").as_posix()})',""]
    (root/"report.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps([dict(label=r["label"],mean_cm_s=r.get("mean_vertical_speed_m_s",0)*100,accepted=r.get("accepted"),
                           functional=r.get("functional_pass"),failure=r.get("failure_reason")) for r in results],ensure_ascii=False,indent=2))


if __name__=="__main__": main()
