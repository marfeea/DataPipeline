"""Re-audit an existing CuRobo V2 artifact under the strict support-tray policy."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from DataPipe.pipelines.curobo_bootstrap.pick_place_artifact import verify_hybrid_artifact, sha256_file
from scripts.build_curobo_pick_place_v2_artifact import _legacy_collision_diagnostic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    artifact = verify_hybrid_artifact(args.run)
    report = artifact["report"]
    snapshot = artifact["snapshot"]
    diagnostic = _legacy_collision_diagnostic(
        artifact["trajectory"],
        artifact["acceptance"]["segment_ranges"],
        snapshot["world"],
        snapshot["collision_model"],
        report["attachment"],
    )
    result = {
        "schema": "curobo_v2_saved_tray_policy_audit/1",
        "passed": diagnostic["tray_collision_free"],
        "source_index": report["source_index"],
        "destination_index": report["destination_index"],
        "task_hash": report["task_hash"],
        "world_hash": report["world_hash"],
        "source_run": str(args.run.resolve()),
        "trajectory_sha256": sha256_file(args.run / "trajectory.npz"),
        "plan_report_sha256": sha256_file(args.run / "plan_report.json"),
        "collision_policy": {
            "allowed_world_contact": [snapshot["world"]["target_name"]],
            "protected_trays": diagnostic["protected_trays"],
            "self_collision_hard_gate": True,
            "tray_collision_hard_gate": True,
        },
        "legacy_collision_diagnostic": diagnostic,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "run": str(args.run), "output": str(args.output)}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

