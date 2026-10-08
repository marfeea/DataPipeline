"""Check all five IK targets after a diagnostic height/seed change (no collisions)."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from analyze_run003_failures_offline import ROOT, ik, pack, read

sys.path.insert(0, str(ROOT))
from DataPipe.pipelines.curobo_bootstrap.task_targets import legacy_tcp_poses_b

run = ROOT / "data/curobo_pick_place_graph_run003_20261003_r2"
output = ROOT / "docs/run003_failure_analysis_20261004/full_target_diagnostics.json"
if output.exists():
    raise FileExistsError(output)
results = []
for file in sorted((run / "evidence").glob("*_evidence/targets.json")):
    report = read(file)
    if report["passed"]:
        continue
    for clearance in (.4, .35, .3):
        task = deepcopy(report["task_snapshot"])
        for waypoint in ("lift", "place_high"):
            task["waypoint_position_b"][waypoint][2] += clearance - .4
        poses = legacy_tcp_poses_b(task)
        seed = np.asarray(report["legacy_ik"]["q_start"])
        stages = {}
        for name, pose in poses.items():
            seeds = [seed, np.asarray(report["legacy_ik"]["q_start"]), np.zeros(6)]
            stage = None
            for i, start in enumerate(seeds):
                trial = pack(ik.solve_tcp_ik(pose[:3], pose[3:], seed=start), np.asarray(pose))
                trial.update(seed_index=i)
                stage = trial
                if trial["success"]:
                    seed = np.asarray(trial["q"])
                    break
            stages[name] = stage
            if not stage["success"]:
                break
        entry = dict(route=file.parent.name.removesuffix("_evidence"), clearance_m=clearance,
                     passed=len(stages) == 5 and all(s["success"] for s in stages.values()),
                     stages=stages, source_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        results.append(entry)
        print(json.dumps({key: entry[key] for key in ("route", "clearance_m", "passed")}), flush=True)
output.write_text(json.dumps(dict(scope="full five-waypoint IK feasibility only; alternative seeds do not prove a continuous collision-free path",
                                 results=results), indent=2, allow_nan=False) + "\n", encoding="utf-8")
