"""独立校验 AUBO IK 数据集。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from DataPipe.validator import validate_hdf5_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = validate_hdf5_dataset(args.output_dir)
    payload = {
        "passed": report.passed,
        "episode_count": report.episode_count,
        "shard_count": report.shard_count,
        "errors": report.errors,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        status = "PASS" if report.passed else "FAIL"
        print(f"DATASET_VALIDATION={status} episodes={report.episode_count} shards={report.shard_count}")
        for error in report.errors:
            print(f"ERROR {error}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
