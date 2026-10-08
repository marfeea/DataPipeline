"""Plot measured insertion traces; leave source evidence unchanged."""

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

root = Path(__file__).resolve().parents[1]
trials = root / "data/curobo_pick_place_graph_run003_fix_20261004/state4_placement_20261005"
fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
summaries = []
for name, label, color in (
    ("measured_entry004_standoff240", "4 mm entry: failed", "#b34432"),
    ("measured_entry012_standoff240_contacts_r2", "12 mm entry: passed", "#176a98"),
):
    run = trials / name / "0_to_4_evidence/integrated_01_01"
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    trace = json.loads((run / "trace.json").read_text(encoding="utf-8"))
    rows = [row for row in trace if row.get("contact_waypoint") == "place_insert_xy"]
    goal = report["placement_calibration"]["desired_object_pose_w"]
    assert rows and np.allclose(np.abs(goal[3:]), (1, 0, 0, 0), atol=1e-12)
    time_s = np.arange(len(rows)) / report["execution_contract"]["control_hz"]
    height = 1000 * (np.array([r["target_pose_w"][2] for r in rows]) - goal[2])
    q = np.array([r["target_pose_w"][3:] for r in rows])
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1)))
    z_error = 1000 * np.array([r["tcp_position_b"][2] - r["requested_tcp_pose_b"][2] for r in rows])
    for axis, values in zip(axes, (height, tilt, z_error), strict=True):
        axis.plot(time_s, values, label=label, color=color, linewidth=1.8)
        axis.grid(alpha=0.25)
    summaries.append(
        dict(
            run=str(run),
            passed=report["passed"],
            insertion_samples=len(rows),
            final_insertion_tilt_deg=float(tilt[-1]),
            trace_sha256=hashlib.sha256((run / "trace.json").read_bytes()).hexdigest(),
        )
    )
axes[0].axhline(0, color="grey", linestyle="--", linewidth=1)
axes[0].set_ylabel("Object root above goal (mm)")
axes[1].axhline(1, color="grey", linestyle="--", linewidth=1)
axes[1].set_ylabel("Object vertical tilt (deg)")
axes[2].axhline(0, color="grey", linestyle="--", linewidth=1)
axes[2].set_ylabel("Actual TCP Z - request (mm)")
axes[2].set_xlabel("Time since horizontal insertion began (s)")
axes[0].legend()
fig.suptitle("0 to 4: measured-grasp insertion comparison")
output = root / "docs/run003_failure_analysis_20261004/state4_entry_height_comparison_20261005.png"
fig.savefig(output, dpi=160)
output.with_suffix(".json").write_text(
    json.dumps(
        dict(trials=summaries, scope="Measured object/TCP trace; not contact-pair or mesh clearance proof"), indent=2
    )
    + "\n",
    encoding="utf-8",
)
print(output)
