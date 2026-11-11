"""Measure completed placement attempts without changing their trace or targets."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    task = read(evidence / "targets.json")["task_snapshot"]
    target = np.asarray(task["waypoint_position_b"]["place"])
    descent = np.asarray(task["contact_waypoint_position_b"]["place_descent"])
    object_destination_w = (np.asarray(task["frames"]["T_WE"]) @ [*task["destination"]["position_e"], 1])[:3]
    rows = []
    insertion_traces = []
    for run in sorted(evidence.glob("integrated_*")):
        if not (run / "report.json").exists():
            continue
        report = read(run / "report.json")
        records = read(run / "trace.json")
        placement = [r for r in records if r.get("contact_waypoint") == "place_insert_xy"]
        if placement:
            insertion_traces.append((run.name, placement))
        lowering = [r for r in records if r.get("contact_waypoint") == "place_descent_outside_roof"]
        if not (placement or lowering):
            continue
        stage_records = placement or lowering
        final = stage_records[-1]
        expected = target if placement else descent
        error = np.asarray(final["tcp_position_b"]) - expected
        first_quat = np.asarray(stage_records[0]["target_pose_w"][3:])
        final_quat = np.asarray(final["target_pose_w"][3:])
        rotation = 2 * np.arccos(
            np.clip(
                abs(np.dot(first_quat, final_quat)) / (np.linalg.norm(first_quat) * np.linalg.norm(final_quat)), 0, 1
            )
        )
        rows.append(
            dict(
                run=str(run),
                passed=report["passed"],
                failure_reason=report.get("failure_reason"),
                final_contact_waypoint=final["contact_waypoint"],
                descent_steps=len(lowering),
                insert_steps=len(placement),
                final_tcp_error_xyz_m=error.tolist(),
                final_tcp_error_m=float(np.linalg.norm(error)),
                object_rotation_change_deg=float(np.degrees(rotation)),
                object_z_change_m=final["target_pose_w"][2] - stage_records[0]["target_pose_w"][2],
                object_final_destination_error_m=float(
                    np.linalg.norm(np.asarray(final["target_pose_w"][:3]) - object_destination_w)
                ),
                first_object_pose_w=stage_records[0]["target_pose_w"],
                final_object_pose_w=final["target_pose_w"],
                trace_sha256=hashlib.sha256((run / "trace.json").read_bytes()).hexdigest(),
            )
        )
    result = dict(
        task_hash=read(evidence / "targets.json")["task_hash"],
        state4_routing=task["state4_routing"],
        scope="measured TCP tracking and object pose changes; no contact-pair attribution",
        attempts=rows,
    )
    (evidence / "insertion_diagnostics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if insertion_traces:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        figure, axes = plt.subplots(len(insertion_traces), 1, figsize=(10, 3.2 * len(insertion_traces)), squeeze=False)
        for axis, (name, samples) in zip(axes.flat, insertion_traces, strict=True):
            errors = np.asarray([r["tcp_position_b"] for r in samples]) - target
            time_s = np.arange(len(samples)) / task["control_hz"]
            axis.plot(time_s, np.linalg.norm(errors[:, :2], axis=1) * 1000, label="XY error")
            axis.plot(time_s, abs(errors[:, 2]) * 1000, label="Z error")
            axis.plot(time_s, np.linalg.norm(errors, axis=1) * 1000, color="black", alpha=0.5, label="Total TCP error")
            axis.axhline(10, color="gray", linestyle="--", label="Original 10 mm arrival tolerance")
            axis.set(title=name, xlabel="Time during horizontal insertion (s)", ylabel="Tracking error (mm)")
            axis.grid(alpha=0.25)
            axis.legend()
        figure.suptitle("State4 insertion: commanded final Z and orientation unchanged")
        figure.tight_layout()
        figure.savefig(evidence / "insertion_tracking.png", dpi=150)
        plt.close(figure)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
