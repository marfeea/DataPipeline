"""Summarize the five-state directed CuRobo V2 route evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def classify(root: Path, name: str, run: dict) -> str:
    stage = run.get("stage")
    evidence = root / f"{name}_evidence"
    if stage == "targets":
        targets = read(evidence / "targets.json") or {}
        failures = [
            f"{key}: {value.get('reason', 'IK failed')}"
            for key, value in targets.get("legacy_ik", {}).get("targets", {}).items()
            if not value.get("passed")
        ]
        return "目标 IK；" + "; ".join(failures) if failures else "目标提取失败"
    if stage == "pre_pick_branches":
        branches = read(evidence / "pre_pick_branches.json") or {}
        self_scores = [item["self_collision_score"] for item in branches.get("candidates", [])]
        reason = (
            f"预抓取自碰撞，最小分数 {min(self_scores):.4f}"
            if self_scores and min(self_scores) > 0
            else "预抓取碰撞"
        )
        return (
            f"{reason}；{branches.get('collision_free_count', 0)}/"
            f"{branches.get('candidate_count', 0)} 个 IK 候选无碰撞"
        )
    failures = run.get("artifact_failures") or []
    if failures:
        tray_contacts = set()
        for item in failures:
            candidate = item.get("candidate")
            artifact = read(evidence / f"artifact_{candidate}" / "plan_report.json") if candidate else None
            for segment in (artifact or {}).get("legacy_collision_diagnostic", {}).get("segments", {}).values():
                tray_contacts.update(segment.get("colliding_trays", {}))
        if tray_contacts and run.get("contact_policy", "strict") == "strict":
            return (
                f"机器人—托盘碰撞；{len(failures)} 个集成候选；"
                + ", ".join(sorted(tray_contacts))
            )
        reasons = "; ".join(str(item.get("reason")) for item in failures)
        return "混合轨迹验收；" + reasons
    if stage and stage.startswith("contact_probe"):
        contacts = [
            read(path) or {}
            for path in sorted(evidence.glob("contact_probe_*/report.json"))
        ]
        trays = sorted({
            tray for item in contacts
            for segment in item.get("tray_collision_diagnostic", {}).get("segments", {}).values()
            for tray in segment.get("colliding_trays", {})
        })
        if trays and run.get("contact_policy", "strict") == "strict":
            return (
                f"实测抓取/抬升机器人—托盘碰撞；"
                f"{len(contacts)} 个抓取候选；" + ", ".join(trays)
            )
    if stage and stage.startswith("transfer_probe"):
        probe = read(evidence / stage / "report.json") or {}
        return "转运规划；" + str(probe.get("failure_reason") or run.get("failure_reason"))
    return str(run.get("failure_reason") or "运行失败")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    routes = {}
    rows = []
    strict_rows = []
    contact_policies = set()
    for source in range(5):
        for destination in range(5):
            if source == destination:
                continue
            name = f"{source}_to_{destination}"
            candidates = [root / name, root / f"{name}_retry"]
            selected = next(
                (path for path in reversed(candidates)
                 if (read(path.with_name(path.name + "_evidence") / "run_report.json") or {}).get("passed")),
                next((path for path in reversed(candidates)
                      if path.with_name(path.name + "_evidence").exists()), candidates[0]),
            )
            evidence = selected.with_name(selected.name + "_evidence")
            run_path = evidence / "run_report.json"
            run = read(run_path)
            if run is None:
                routes[name] = {"tested": False, "passed": False}
                rows.append(f"| {source}→{destination} | 未测试 | 无报告 | |")
                continue
            if not run.get("passed") and not run.get("failure_reason"):
                routes[name] = {"tested": False, "passed": False, "stage": run.get("stage"), "run_report": str(run_path)}
                rows.append(
                    f"| {source}→{destination} | 运行中 | {run.get('stage')} | "
                    f"[运行报告]({evidence.name}/run_report.json) |"
                )
                continue
            replay_path = selected / "replay_report.json"
            contact_policies.add(run.get("contact_policy", "strict"))
            replay = read(replay_path)
            passed = bool(run.get("passed") and replay and replay.get("passed"))
            entry = {
                "tested": True,
                "contact_policy": run.get("contact_policy", "strict"),
                "source_index": source,
                "destination_index": destination,
                "source_name": run.get("source_name"),
                "destination_name": run.get("destination_name"),
                "task_hash": run.get("task_hash"),
                "world_hash": run.get("world_hash"),
                "passed": passed,
                "plan_passed": bool(run.get("passed")),
                "replay_passed": bool(replay and replay.get("passed")),
                "run_report": str(run_path),
                "stage": run.get("stage"),
                "artifact_failures": run.get("artifact_failures", []),
            }
            if passed:
                plan = read(selected / "plan_report.json") or {}
                entry["new_tray_policy_passed"] = bool(
                    plan.get("collision_policy", {}).get("tray_collision_hard_gate")
                    and plan.get("legacy_collision_diagnostic", {}).get("tray_collision_free")
                )
                entry.update(
                    samples=run.get("samples"),
                    trajectory=str(selected / "trajectory.npz"),
                    replay_report=str(replay_path),
                    visualization_dir=replay.get("visualization_dir"),
                    visualization_images=replay.get("visualization", {}).get("image_count"),
                )
                rows.append(
                    f"| {source}→{destination} | "
                    f"{'run_003 模式通过' if entry['contact_policy'] == 'run003' else ('新托盘规则通过' if entry['new_tray_policy_passed'] else '旧规则可采集')}，重播通过 | "
                    f"{entry['samples']} 点；{entry['visualization_images']} 张相机图 | "
                    f"[轨迹]({selected.name}/trajectory.npz) · "
                    f"[重播]({selected.name}/replay_report.json) · "
                    f"[可视化]({selected.name}/replay_visualization/frames.json) |"
                )
            else:
                diagnosis = classify(root, selected.name, run)
                entry["diagnosis"] = diagnosis
                rows.append(
                    f"| {source}→{destination} | 当前预算内未采集 | {diagnosis} | "
                    f"[运行报告]({evidence.name}/run_report.json) |"
                )
            if passed:
                strict_path = root / "tray_policy_reaudit_robot_only" / f"{selected.name}.json"
                strict = read(strict_path)
                if strict is not None:
                    segments = strict["legacy_collision_diagnostic"]["segments"]
                    contacts = sorted({
                        tray for segment in segments.values()
                        for tray in segment.get("colliding_trays", {})
                    })
                    entry["strict_tray_policy"] = {
                        "passed": bool(strict["passed"]),
                        "audit_report": str(strict_path),
                        "colliding_trays": contacts,
                        "max_tray_collision_score": max(
                            segment["tray_max_world_score"] for segment in segments.values()
                        ),
                    }
                    strict_rows.append(
                        f"| {source}→{destination} | {'通过' if strict['passed'] else '不通过'} | "
                        f"{', '.join(contacts) or '无'} | "
                        f"{entry['strict_tray_policy']['max_tray_collision_score']:.4f} | "
                        f"[复核报告](tray_policy_reaudit_robot_only/{selected.name}.json) |"
                    )
            if name == "1_to_3":
                extended_path = evidence / "transfer_probe_04_01_extended" / "report.json"
                extended = read(extended_path)
                if extended is not None:
                    entry["extended_transfer_probe"] = {
                        "report": str(extended_path),
                        "passed": bool(extended.get("passed")),
                        "max_attempts": extended.get("max_attempts"),
                        "timeout_s": extended.get("timeout_s"),
                        "failure_reason": extended.get("failure_reason"),
                    }
            routes[name] = entry
    report = {
        "schema": "curobo_v2_directed_graph/2",
        "state_count": 5,
        "contact_policies": sorted(contact_policies),
        "directed_route_count": 20,
        "tested_count": sum(item["tested"] for item in routes.values()),
        "collected_and_replayed_count": sum(item["passed"] for item in routes.values()),
        "search": {
            "pre_pick_ik_random_seeds": 200,
            "max_pick_and_place_candidates": 8,
            "motion_gen_max_attempts": 4,
            "motion_gen_timeout_s": 30,
        },
        "routes": routes,
        "strict_tray_policy": {
            "historical_successes_audited": len(strict_rows),
            "historical_successes_compliant": sum(
                item.get("strict_tray_policy", {}).get("passed", False) for item in routes.values()
            ),
            "other_routes_not_retested_end_to_end": 20 - len(strict_rows) if strict_rows else 0,
        },
    }
    (root / "graph_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    historical = bool(strict_rows)
    introduction = (
        f"已测试 {report['tested_count']}/20 条有向路线；"
        f"{report['collected_and_replayed_count']} 条完成轨迹采集与 Isaac 重播。"
    )
    if historical:
        introduction += "这些是旧版碰撞规则下的结果，须参照下方托盘硬门禁复核。"
    markdown = [
        "# CuRobo 五点有向抓放图测试",
        "",
        introduction,
        "",
        "测试使用 Isaac Lab cuda:0，目标来自 configs/task.py 的 TARGET_STATES。"
        "常规路线使用 200 个随机 IK 种子、至多 8 个抓取和放置候选、"
        "每段 CuRobo MotionGen 至多 4 次尝试和 30 秒超时。",
        "未采集成功仅表示此配置及预算内未通过，不等于证明物理不可达。",
        "",
        "| 路线 | 结果 | 诊断或产物 | 证据 |",
        "|---|---|---|---|",
        *rows,
        "",
    ]
    if contact_policies == {"run003"}:
        markdown.extend([
            "规划模式：run_003 混合规划与状态重播。自由空间段继续实施世界碰撞和自碰撞验收；"
            "遗留接触段的托盘碰撞记录为诊断，不据此拒绝采集。成功表示本模式下采集与状态重播通过，"
            "不能据此宣称满足严格托盘规则或完成独立物理执行验证。",
            "",
        ])
    elif not historical:
        markdown.extend([
            "碰撞规则：机器人允许接触本次目标采样瓶 target；support_tray_* "
            "与其他非目标网格仍参与碰撞检查。实测抓取/抬升和最终混合轨迹均对"
            "机器人—托盘碰撞实施硬门禁；瓶子放置时与托盘的接触单独记录。",
            "",
        ])
    if historical:
        markdown.extend([
            "## 托盘硬门禁复核（2026-09-27）",
            "",
            "机器人允许接触采样瓶 target；机器人与 support_tray_* 及其他场景障碍的碰撞仍被检查。"
            "瓶子放上托盘的接触单独记录；机器人自碰撞继续作为硬门禁。"
            "旧版通过轨迹逐帧复核后均有机器人—托盘正碰撞分数，不能直接宣称满足新规则。",
            "",
            "| 旧规则通过路线 | 新托盘规则 | 机器人碰撞托盘 | 最大机器人—托盘分数 | 证据 |",
            "|---|---|---|---:|---|",
            *strict_rows,
            "",
            "状态 2 作为抓取源的预抓取复核：允许接触采样瓶后，"
            "2→0、2→1、2→3 各自的 8 个 IK 候选仍全部自碰撞，0 个通过；"
            "托盘及其他非目标网格仍在保护列表。"
            "证据见 [2→0](2_to_0_evidence/pre_pick_branches_target_contact.json)、"
            "[2→1](2_to_1_evidence/pre_pick_branches_target_contact.json)、"
            "[2→3](2_to_3_evidence/pre_pick_branches_target_contact.json)。",
            "",
        ])
    extended = routes.get("1_to_3", {}).get("extended_transfer_probe")
    if extended is not None:
        markdown.extend([
            "1→3 另用同一附着物与无碰撞端点进行 12 次尝试、120 秒的转运探针，"
            "仍返回 CuRobo TRAJOPT_FAIL；详见 "
            "[扩展探针](1_to_3_evidence/transfer_probe_04_01_extended/report.json)。",
            "",
        ])
    (root / "RESULTS.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps({"tested": report["tested_count"], "collected_and_replayed": report["collected_and_replayed_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

