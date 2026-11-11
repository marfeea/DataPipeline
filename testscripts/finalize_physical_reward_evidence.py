"""Finalize reward coverage without altering any measured physics or images."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pipelines.curobo_bootstrap.physical_reward_validity import finalize_reward_files
from scripts.pipelines.curobo_bootstrap.pick_place_artifact import save_json, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / "report.json").read_text(encoding="utf-8"))
    rows = json.loads((args.run / "rewards.json").read_text(encoding="utf-8"))
    if not report["passed"]:
        raise ValueError("cannot finalize a failed or superseded experiment")
    immutable = {
        name: sha256_file(args.run / name)
        for name in ("trajectory.npz", "trace.json", "observations.npz", "scene_snapshot.json")
    }
    finalize_reward_files(args.run, rows, report)
    report["reward_validity_source_sha256"] = sha256_file(
        Path(__file__).parent / "pipelines/curobo_bootstrap/physical_reward_validity.py"
    )
    assert all(sha256_file(args.run / name) == digest for name, digest in immutable.items())
    report["physics_unchanged_after_reward_finalization"] = True
    save_json(args.run / "report.json", report)
    save_json(args.run / "plan_report.json", report)
    print(
        json.dumps(
            {
                "run": str(args.run),
                "reward_evidence_coverage": report["reward_evidence_coverage"],
                "valid_reward_transitions": report["valid_reward_transitions"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
