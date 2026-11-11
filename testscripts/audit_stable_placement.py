"""Recompute placement and path gates from saved physical observations."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pick_place_reward_candidate import RewardConfig, rotation_error
from scripts.pipelines.curobo_bootstrap.physical_failure import relative_pose
from scripts.pipelines.curobo_bootstrap.pick_place_artifact import save_json, sha256_file
from scripts.pipelines.curobo_bootstrap.placement_search import (
    carrying_metrics,
    continuous_stable_s,
    placement_metrics,
    pose_list,
)


def audit_trial(directory):
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    trace_path = directory / "trace.json"
    rows = json.loads(trace_path.read_text(encoding="utf-8"))
    config = RewardConfig(**report["thresholds"])
    initial = [r for r in rows if r["stage"] == "initial_grasp"][-16:]
    grasp = False
    if len(initial) == 16:
        reference = relative_pose(initial[0]["tcp_pose_w"], initial[0]["object_pose_w"])
        relative = [relative_pose(r["tcp_pose_w"], r["object_pose_w"]) for r in initial]
        grasp = bool(
            all(
                r["contact_available"] and min(r["contact_force_n"][:2]) > 0.01 and r["contact_force_n"][2] <= 0.01
                for r in initial
            )
            and max(np.linalg.norm(r[:3, 3] - reference[:3, 3]) for r in relative) < 0.005
            and max(rotation_error(pose_list(r)[3:], pose_list(reference)[3:]) for r in relative) < np.deg2rad(5)
        )
    for row in rows:
        row["metrics"] = placement_metrics(row, report["goal_object_pose_w"], grasp_verified=grasp, config=config)
    settling = [r for r in rows if r["stage"] in {"release", "settle"}]
    after = [r for r in rows if r["stage"] == "after_withdraw"]
    path = [r for r in rows if r["stage"] in {"descent", "pre_release_hold", "release"}]
    carry_safe = True
    loss_s = 0.0
    if initial:
        carry_reference = relative_pose(initial[-1]["tcp_pose_w"], initial[-1]["object_pose_w"])
        for row in rows:
            if row["stage"] not in {"descent", "pre_release_hold"}:
                continue
            carrying = carrying_metrics(row, report["goal_object_pose_w"], carry_reference)
            loss_s = loss_s + 1 / 60 if carrying["unsupported_loss"] else 0.0
            carry_safe = carry_safe and not carrying["premature_support"] and loss_s + 1e-12 < 0.05
    endpoint = [r for r in rows if r["stage"] == "pre_release_hold"]
    before_s, after_s = continuous_stable_s(settling), continuous_stable_s(after)
    xy = max(
        (np.linalg.norm(np.asarray(r["tcp_pose_w"][:2]) - r["requested_tcp_pose_w"][:2]) for r in path), default=np.inf
    )
    rotation = max((rotation_error(r["tcp_pose_w"][3:], r["requested_tcp_pose_w"][3:]) for r in path), default=np.inf)
    constant_xy = bool(
        path
        and np.allclose(
            [r["requested_tcp_pose_w"][:2] for r in path], path[0]["requested_tcp_pose_w"][:2], atol=1e-12, rtol=0
        )
    )
    constant_rotation = bool(
        path
        and all(rotation_error(r["requested_tcp_pose_w"][3:], path[0]["requested_tcp_pose_w"][3:]) < 1e-6 for r in path)
    )
    endpoint_error = (
        np.linalg.norm(np.asarray(endpoint[-1]["tcp_pose_w"][:3]) - endpoint[-1]["requested_tcp_pose_w"][:3])
        if endpoint
        else np.inf
    )
    withdrawal_error = (
        np.linalg.norm(np.asarray(after[-1]["tcp_pose_w"][:3]) - after[-1]["requested_tcp_pose_w"][:3])
        if after
        else np.inf
    )
    passed = bool(
        grasp
        and carry_safe
        and before_s >= config.settle_s
        and after_s >= 0.5
        and xy <= 0.003
        and rotation <= np.deg2rad(2)
        and endpoint_error <= 0.003
        and withdrawal_error <= 0.003
        and constant_xy
        and constant_rotation
        and not any(r["illegal_contact"] or r["joint_limit"] for r in rows)
    )
    return dict(
        directory=str(directory.resolve()),
        trace_sha256=sha256_file(trace_path),
        candidate=report["candidate"],
        passed=passed,
        reported_passed=report["passed"],
        agrees_with_report=passed == report["passed"],
        grasp_verified=grasp,
        carry_safe=carry_safe,
        stable_before_withdraw_s=before_s,
        stable_after_withdraw_s=after_s,
        constant_descent_xy=constant_xy,
        constant_descent_rotation=constant_rotation,
        max_descent_xy_error_m=float(xy) if np.isfinite(xy) else None,
        max_descent_orientation_error_rad=float(rotation) if np.isfinite(rotation) else None,
        endpoint_error_m=float(endpoint_error) if np.isfinite(endpoint_error) else None,
        withdrawal_error_m=float(withdrawal_error) if np.isfinite(withdrawal_error) else None,
        final_metrics=rows[-1]["metrics"] if rows else None,
        scope="physical placement gate; full-route collision/reward evidence not certified",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directories = []
    for directory in args.directories:
        directories.extend([directory] if (directory / "trace.json").exists() else sorted(directory.glob("trial_*")))
    results = [audit_trial(d) for d in directories if (d / "trace.json").exists()]
    save_json(
        args.output,
        dict(
            schema="stable_placement_independent_audit/1",
            trials=len(results),
            successes=sum(r["passed"] for r in results),
            all_reports_agree=all(r["agrees_with_report"] for r in results),
            results=results,
        ),
    )
    print(
        json.dumps(
            dict(
                trials=len(results),
                successes=sum(r["passed"] for r in results),
                all_reports_agree=all(r["agrees_with_report"] for r in results),
            )
        )
    )
    return 0 if results and all(r["agrees_with_report"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
