"""Plot measured standalone state2 contact dynamics without editing any trace."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(args.output)
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    summaries = []
    for run, label, color in (
        (args.baseline, "Original wrist drive", "#bd3b3b"),
        (args.candidate, "Wrist drive cap 0.7 rad/s", "#247da4"),
    ):
        run = run.resolve()
        report = json.loads((run / "report.json").read_text(encoding="utf-8"))
        trace_path = run / "trace.json"
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        contract = report["execution_contract"]
        assert contract["physics_hz"] == 120 and contract["control_hz"] == 60
        assert all(not row["stage"].startswith("free_space") for row in trace)
        velocity = np.asarray([row["dq"] for row in trace])
        acceleration = np.vstack((np.zeros((1, 6)), np.diff(velocity, axis=0) * 60))
        object_position = np.asarray([row["target_position_w"] for row in trace])
        displacement = np.linalg.norm(object_position - report["target_pose_w_at_reset"][:3], axis=1)
        time = np.arange(len(trace)) / 60
        axes[0].plot(time, np.max(np.abs(velocity[:, 3:]), axis=1), color=color, label=label, linewidth=1)
        axes[1].plot(time, np.max(np.abs(acceleration[:, 3:]), axis=1), color=color, linewidth=1)
        axes[2].plot(time, displacement * 1000, color=color, linewidth=1)
        summaries.append(dict(
            run=str(run), trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            execution_contract=contract, pick_lift_probe_passed=report["passed"],
            peak_wrist_velocity_rad_s=float(np.max(np.abs(velocity[:, 3:]))),
            peak_wrist_acceleration_rad_s2=float(np.max(np.abs(acceleration[:, 3:]))),
            final_object_displacement_mm=float(displacement[-1] * 1000),
        ))
    axes[0].axhline(3.14, color="#555555", linestyle="--", linewidth=1, label="Original wrist acceptance limit")
    axes[1].axhline(50, color="#555555", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Max wrist |velocity| (rad/s)")
    axes[1].set_ylabel("Max wrist |acceleration| (rad/s²)")
    axes[2].set_ylabel("Object displacement (mm)")
    axes[2].set_xlabel("Control time (s)")
    axes[0].legend(loc="upper right", fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle("State2 native pick/lift: measured dynamics at 120 Hz physics / 60 Hz control")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    metadata = dict(
        comparisons=summaries,
        scope="Standalone native contact traces, not full-route acceptance or proof of contact pairs",
        acceleration_definition="Backward difference of measured 60 Hz joint velocities, matching artifact assembly",
    )
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
