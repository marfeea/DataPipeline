"""Model and score accepted physical experiments with the current v2 reward.

Original physics, reward sidecars and images are read-only. Outputs are written
to a fresh directory and retain distinct candidate/full-training validity masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_physical_failure import audit
from scripts.pick_place_reward_candidate import RewardTask, rotation_error
from scripts.pick_place_reward_candidate_v2 import VERSION, PickPlaceRewardV2
from scripts.pipelines.curobo_bootstrap.physical_failure import GraspEvidence, reward_observation
from scripts.pipelines.curobo_bootstrap.physical_reward_validity import gate_reward_validity
from scripts.pipelines.curobo_bootstrap.pick_place_artifact import save_json, sha256_file

TERMS = ("reach", "lift", "position", "orientation", "grasp", "success", "time", "speed", "overlap", "failure")
INPUTS = (
    "report.json",
    "scene_snapshot.json",
    "trajectory.npz",
    "observations.npz",
    "trace.json",
    "rewards.json",
    "rewards.npz",
    "execution_visualization/frames.json",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate(run):
    hashes = {name: sha256_file(run / name) for name in INPUTS}
    input_audit = audit(run)
    report, records, saved = read(run / "report.json"), read(run / "trace.json"), read(run / "rewards.json")
    task = RewardTask(**report["reward_task"])
    detector = GraspEvidence(records[0])
    model = None
    rows, diagnostics, events = [], [], []
    terminal = None
    prior_grasp = prior_release = prior_injection = False
    prior_support = False
    for index, original in enumerate(records):
        sample = dict(original)
        current_support = bool(sample["contact_available"] and sample["contact_force_n"][2] > 0.01)
        sample["normal_supported_release"] = bool(
            index > 0
            and sample["phase"] == "release"
            and not sample["fault_injection_active"]
            and (prior_support or current_support)
            and np.linalg.norm(np.asarray(sample["target_pose_w"][:3]) - task.goal_pose_w[:3]) < 0.02
            and rotation_error(sample["target_pose_w"][3:], task.goal_pose_w[3:]) < np.deg2rad(10)
        )
        evidence = detector.step(sample)
        obs = reward_observation(sample, evidence, timeout=index == len(records) - 1 and not (model and model.done))
        if model is None:
            model = PickPlaceRewardV2(task, obs)
        elif not model.done:
            score = model.step(obs)
            score.update(transition_valid=True, candidate_reward_valid=score["reward_valid"])
            if score["terminated"]:
                terminal = index
                events.append(dict(event=score["failure"] or "success", observation=index, time_s=obs.time_s))
        else:
            score = dict(
                reward=0.0,
                reward_valid=False,
                candidate_reward_valid=False,
                transition_valid=False,
                terminated=False,
                failure=None,
                terms={name: 0.0 for name in TERMS},
            )
        if index > 0:
            rows.append(
                {
                    "transition_index": index - 1,
                    "from_observation": index - 1,
                    "to_observation": index,
                    "time_s": obs.time_s,
                    "execution_stage": sample["stage"],
                    "phase": sample["phase"],
                    **score,
                }
            )
        c = model.config
        gates = dict(
            grasp_previously_paid=bool(model.grasp_paid),
            release_phase=obs.phase == "release",
            support_verified=obs.support_verified,
            evidence_valid=obs.evidence_valid,
            gripper_open=min(abs(q) for q in obs.gripper_position) >= c.open_finger_m,
            position=float(np.linalg.norm(np.asarray(obs.object_pose_w[:3]) - task.goal_pose_w[:3]))
            <= c.position_tolerance_m,
            orientation=rotation_error(obs.object_pose_w[3:], task.goal_pose_w[3:]) <= c.rotation_tolerance_rad,
            linear_speed=obs.object_linear_speed <= c.max_object_speed_m_s,
            angular_speed=obs.object_angular_speed <= c.max_object_angular_speed_rad_s,
        )
        gates = {name: bool(value) for name, value in gates.items()}
        diagnostics.append(
            dict(
                observation=index,
                time_s=obs.time_s,
                phase=obs.phase,
                gates=gates,
                stable=bool(all(gates.values())),
                reward_settled_s=float(model.settled_s),
                object_height_m=obs.object_pose_w[2],
                object_vertical_speed_m_s=sample["target_velocity_w"][2],
                position_error_m=float(np.linalg.norm(np.asarray(obs.object_pose_w[:3]) - task.goal_pose_w[:3])),
                rotation_error_deg=float(np.degrees(rotation_error(obs.object_pose_w[3:], task.goal_pose_w[3:]))),
                object_linear_speed_m_s=obs.object_linear_speed,
                object_angular_speed_rad_s=obs.object_angular_speed,
                grasp_verified=evidence["grasp_verified"],
                support_verified=evidence["support_verified"],
                finger_contact_force_n=sample["contact_force_n"][:2],
                support_force_n=sample["contact_force_n"][2],
                normal_supported_release=detector.supported_release,
            )
        )
        for flag, previous, name in (
            (evidence["ever_grasp_verified"], prior_grasp, "grasp_verified"),
            (detector.supported_release, prior_release, "supported_release"),
            (sample["fault_injection_active"], prior_injection, "gripper_open_injection"),
        ):
            if flag and not previous:
                events.append(dict(event=name, observation=index, time_s=obs.time_s))
        prior_grasp, prior_release = evidence["ever_grasp_verified"], detector.supported_release
        prior_injection, prior_support = sample["fault_injection_active"], current_support
    if terminal is None:
        raise ValueError("no reward terminal event reconstructed")
    coverage_report = {"contact_positive_controls": detector.seen_channels.tolist()}
    gate_reward_validity(rows, coverage_report)
    totals = {name: float(sum(row["terms"][name] for row in rows)) for name in TERMS}
    phases = defaultdict(lambda: dict(transitions=0, return_value=0.0, terms=dict.fromkeys(TERMS, 0.0)))
    for row in rows:
        if not row["transition_valid"]:
            continue
        phase = phases[row["phase"]]
        phase["transitions"] += 1
        phase["return_value"] += row["reward"]
        for name in TERMS:
            phase["terms"][name] += row["terms"][name]
    release = [d for d in diagnostics if d["phase"] == "release" and d["observation"] <= terminal]
    summary = dict(
        case_id=run.name,
        source_run=str(run),
        model_version=VERSION,
        reward_config=asdict(model.config),
        reward_task=asdict(task),
        detector_config=asdict(detector.config),
        observations=len(records),
        transitions=len(rows),
        active_transitions=sum(row["transition_valid"] for row in rows),
        post_terminal_transitions=sum(not row["transition_valid"] for row in rows),
        recorded_duration_s=records[-1]["time_s"],
        terminal_time_s=records[terminal]["time_s"],
        terminal_observation=terminal,
        failure=rows[terminal - 1]["failure"],
        outcome=report["outcome"],
        diagnostic_return_undiscounted=float(sum(row["reward"] for row in rows)),
        component_returns=totals,
        phase_returns=dict(phases),
        events=events,
        release_observations=len(release),
        release_gate_failed_frames={key: sum(not d["gates"][key] for d in release) for key in diagnostics[0]["gates"]},
        max_settled_s=max(d["reward_settled_s"] for d in diagnostics),
        final_state=diagnostics[-1],
        saved_rewards_match=bool(
            np.allclose([r["reward"] for r in rows], [r["reward"] for r in saved], atol=1e-12, rtol=0)
        ),
        input_audit=input_audit,
        input_sha256=hashes,
        input_unchanged=all(sha256_file(run / name) == digest for name, digest in hashes.items()),
        **coverage_report,
    )
    if not summary["input_unchanged"] or not summary["saved_rewards_match"]:
        raise ValueError("original evidence changed or current reward recomputation differs")
    np.testing.assert_allclose(sum(totals.values()), summary["diagnostic_return_undiscounted"], atol=1e-12)
    return summary, rows, diagnostics


def plot_case(output, summary, rows, diagnostics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = [d["time_s"] for d in diagnostics]
    rt = [r["time_s"] for r in rows]
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True, layout="constrained")
    axes[0].plot(rt, np.cumsum([r["reward"] for r in rows]), label="Cumulative reward", color="tab:blue")
    axes[0].set_ylabel("Return")
    axes[1].plot(times, [d["object_height_m"] for d in diagnostics], color="tab:green", label="Object Z")
    axes[1].set_ylabel("World Z (m)")
    axes[2].plot(times, [d["object_linear_speed_m_s"] for d in diagnostics], label="Linear speed (m/s)")
    axes[2].axhline(summary["reward_config"]["max_object_speed_m_s"], color="gray", ls=":", label="Settle speed limit")
    axes[2].set_ylabel("Speed (m/s)")
    axes[3].step(times, [d["grasp_verified"] for d in diagnostics], where="post", label="Verified grasp")
    axes[3].step(times, [d["support_verified"] for d in diagnostics], where="post", label="Object support")
    axes[3].step(times, [d["stable"] for d in diagnostics], where="post", label="All settle gates")
    axes[3].set_ylabel("Evidence")
    axes[3].set_xlabel("Simulation time (s)")
    for ax in axes:
        ax.axvline(summary["terminal_time_s"], color="tab:red", ls="--", lw=1, label="Terminal")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        f"{summary['case_id']} | {summary['failure']} | return={summary['diagnostic_return_undiscounted']:.6f}"
    )
    fig.savefig(output / f"{summary['case_id']}_reward_evidence.png", dpi=150)
    plt.close(fig)


def write_report(output, summaries, excluded):
    labels = dict(
        reach="接近",
        lift="抬升",
        position="目标位置",
        orientation="目标朝向",
        grasp="抓持",
        success="成功",
        time="时间",
        speed="关节速度代价",
        overlap="重叠",
        failure="终止惩罚",
    )
    text = [
        "# 新增物理轨迹奖励建模与评估",
        "",
        "评估日期：2026-10-12。使用当前 `pick-place-candidate/2`，未修改模型参数与原实验数据。",
        "",
        "## 轨迹模型与奖励",
        "",
        "每条轨迹按 `o₀ → a₀ → o₁ … aₜ₋₁ → oₜ` 建模：观测来自实测机械臂 q/dq、TCP 与物体世界位姿、"
        "物体线/角速度、两指状态及接触；动作是实际发送的六轴关节目标和两指目标。N 个观测对应 N−1 条转移。",
        "",
        "抓持证据要求双指接触、脱离支撑、抬升至少 20 mm、相对 TCP 位姿稳定 0.25 s；"
        "掉落要求之前确认抓持、持续失持 0.05 s、相对漂移、下降至少 10 mm及下降速度。"
        "目标承托后的正常释放保持证据，不因沉降改标掉落。",
        "",
        "令 d 为物体到目标的距离、θ 为 SO(3) 朝向误差、d₀ 为源到目标距离、h 为相对源高度：",
        "",
        "- 接近：2 × 接近完成度历史最大值的增量；接近完成度为 "
        "clip(1 − TCP 到抓点距离 / 初始距离, 0, 1)，仅 reach/close 阶段。",
        "- 抬升：2 × clip(h / 0.05 m, 0, 1) 的历史最大值增量；在 lift 阶段且满足模型抓持/闭爪门禁时计算。",
        "- 目标位置：4 × Δclip(1 − d / d₀, 0, 1)；"
        "目标朝向：2 × Δ[clip(1 − d / 0.1 m, 0, 1) × (1 − θ/π)]；"
        "建立抬升后在 transport/place/release 阶段计算，离开目标可扣回进展。",
        "- 已验证抓持奖励 +2，仅一次；稳定放置成功 +10，仅一次。",
        "- 时间代价 −0.1Δt；速度代价 −0.01Δt × (‖dqₜ‖² + ‖dqₜ₋₁‖²)/2，是关节速度平方的积分，不是电能。",
        "- 已确认掉落 −10；超时 −15；IK 中止 −15 是独立控制器事件，"
        "本实验只执行冻结关节目标，没有在线 IK 中止，故本轮不加该项。",
        "- 模型另含非法碰撞 −25、关节越限 −10、几何重叠 −5。"
        "掉落/碰撞/越限终止转移上的正贡献会被清零；本轮缺少完整碰撞/重叠观测，"
        "不能把这些项的数值 0 当作已证实无碰撞。",
        "",
        "稳定成功需在 release 阶段已验证抓持、物体被承托、双指打开、位置误差≤20 mm、"
        "角度误差≤10°、线速度≤0.03 m/s、角速度≤0.174533 rad/s，并连续稳定≥0.25 s。",
        "",
        "## 总回报与分项",
        "",
        "| 奖励分项 | 正常无携物对照 | 验证抓持后开爪 |",
        "|---|---:|---:|",
    ]
    for key in TERMS:
        text.append(
            f"| {labels[key]} | {summaries[0]['component_returns'][key]:.9f} | "
            f"{summaries[1]['component_returns'][key]:.9f} |"
        )
    text.append(
        "| **总回报** | " + " | ".join(f"**{s['diagnostic_return_undiscounted']:.9f}**" for s in summaries) + " |"
    )
    for s in summaries:
        text += [
            "",
            f"## {s['case_id']}",
            "",
            f"{s['observations']} 个观测、{s['transitions']} 条转移；终止时间 {s['terminal_time_s']:.6f} s，"
            f"原因 `{s['failure']}`；终止后 {s['post_terminal_transitions']} 条尾随转移不计奖。",
            "",
            "| 阶段 | 有效控制转移数 | 阶段回报 |",
            "|---|---:|---:|",
        ]
        for phase, values in s["phase_returns"].items():
            text.append(f"| {phase} | {values['transitions']} | {values['return_value']:.9f} |")
        text += [
            "",
            "事件："
            + "；".join(f"`{e['event']}` @ {e['time_s']:.6f} s（观测 {e['observation']}）" for e in s["events"])
            + "。",
            "",
            f"稳定窗口最长为 {s['max_settled_s']:.6f} s，要求 0.25 s。",
        ]
        if s["release_observations"]:
            text += [
                f"release 阶段共 {s['release_observations']} 个观测，各成功门禁未满足的帧数如下；"
                "多个门禁可在同一帧同时失败。",
                "",
                "| 门禁 | 未满足帧数 |",
                "|---|---:|",
            ]
            text += [f"| {k} | {v} |" for k, v in s["release_gate_failed_frames"].items()]
        else:
            text.append("该轨迹在 lift 阶段开爪并掉落，没有进入正常 release 放置阶段。")
        final = s["final_state"]
        image_path = (output / (s["case_id"] + "_reward_evidence.png")).resolve().as_posix()
        text += [
            "",
            f"尾段末帧：位置误差 {final['position_error_m'] * 1000:.3f} mm，"
            f"角度误差 {final['rotation_error_deg']:.3f}°，"
            f"物体线速度 {final['object_linear_speed_m_s']:.6f} m/s、"
            f"角速度 {final['object_angular_speed_rad_s']:.6f} rad/s。"
            + (
                "末帧满足瞬时门禁，但不等于已持续满足完整稳定窗口。"
                if final["stable"]
                else "尾随末帧不满足正常放置门禁，且不再计入已终止回合的奖励。"
            ),
            "",
            f"![reward and measured evidence]({image_path})",
        ]
    delta = summaries[1]["diagnostic_return_undiscounted"] - summaries[0]["diagnostic_return_undiscounted"]
    text += [
        "",
        "## 评价与证据边界",
        "",
        f"受控掉落总分比正常对照高 {delta:.9f}。两条都未成功放置；当前模型对超时罚 −15、"
        "对掉落罚 −10，且持续时间、进展与速度积分不同，所以该排序不能解释成掉落行为更优。"
        "此处仅评估当前函数，没有调整罚分。",
        "",
        "受控试验是开爪注入导致的真实物理掉落，不是保持闭爪时的自然滑落。"
        "正常对照确认了抓持，但没有满足完整稳定放置窗口，不应当作成功 demonstration。",
        "",
        "完整 `reward_valid` 两条均为 false：逐帧非法碰撞与几何重叠的观测尚未覆盖。"
        "物体/抓持/支撑的 `candidate_reward_valid` 分别为 "
        + "、".join(str(s["candidate_valid_reward_transitions"]) for s in summaries)
        + " 条，不能替代完整训练 mask。",
        "",
        "逐步奖励与原已存数值一致至 1e−12；所有输入哈希评估前后一致。"
        "逐步 CSV/JSON、实测门禁 diagnostics、参数、事件和汇总均保存在本目录。",
        "",
        f"排除 {len(excluded)} 个初始化失败、记录前停止或已撤回试跑，详情见 summary.json。",
    ]
    (output / "奖励建模与评估.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or ROOT not in output.parents:
        raise ValueError("use a fresh workspace output directory")
    selected, excluded = [], []
    for path in sorted(args.dataset_root.resolve().glob("*/report.json")):
        r = read(path)
        if r.get("passed") is True and r.get("accepted_for_use", True):
            selected.append(path.parent)
        else:
            excluded.append(
                {"run": path.parent.name, "reason": r.get("superseded_reason", r.get("failure_reason", r["stage"]))}
            )
    if len(selected) != 2 or [read(p / "report.json")["scenario"] for p in selected] != ["natural", "open_after_grasp"]:
        raise ValueError("this comparative report expects the accepted natural/open-after-grasp pair")
    evaluated = [evaluate(run) for run in selected]
    output.mkdir(parents=True)
    for summary, rows, diagnostics in evaluated:
        key = summary["case_id"]
        save_json(output / f"{key}_summary.json", summary)
        save_json(output / f"{key}_transitions.json", rows)
        save_json(output / f"{key}_diagnostics.json", diagnostics)
        with (output / f"{key}_rewards.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            fields = [
                "transition_index",
                "time_s",
                "execution_stage",
                "phase",
                "reward",
                "candidate_reward_valid",
                "reward_valid",
                "transition_valid",
                "terminated",
                "failure",
                *TERMS,
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({**{k: row[k] for k in fields if k not in TERMS}, **row["terms"]})
        plot_case(output, summary, rows, diagnostics)
    summary = dict(
        model_version=VERSION,
        cases=[s for s, _, _ in evaluated],
        excluded=excluded,
        model_source_sha256={
            p: sha256_file(ROOT / p)
            for p in ("scripts/pick_place_reward_candidate.py", "scripts/pick_place_reward_candidate_v2.py")
        },
        evaluation_source_sha256={
            p: sha256_file(ROOT / p)
            for p in (
                "testscripts/evaluate_physical_failure_rewards.py",
                "scripts/pipelines/curobo_bootstrap/physical_failure.py",
                "scripts/pipelines/curobo_bootstrap/physical_reward_validity.py",
            )
        },
    )
    save_json(output / "summary.json", summary)
    write_report(output, summary["cases"], excluded)
    print(
        json.dumps(
            [
                {
                    k: s[k]
                    for k in (
                        "case_id",
                        "diagnostic_return_undiscounted",
                        "failure",
                        "max_settled_s",
                        "release_gate_failed_frames",
                    )
                }
                for s in summary["cases"]
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
