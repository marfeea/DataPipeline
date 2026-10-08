"""Create a separate project TCP URDF using measured P3 evidence."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.coordinate_frames import FLANGE_TO_TCP_ROTATION_F_WXYZ, FLANGE_TO_TCP_TRANSLATION_F
from DataPipe.pipelines.curobo_bootstrap.robot_asset import derive_robot_asset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = derive_robot_asset(
        args.source,
        args.output_dir,
        json.loads(args.audit_report.read_text(encoding="utf-8")),
        tcp_position=FLANGE_TO_TCP_TRANSLATION_F,
        tcp_quaternion_wxyz=FLANGE_TO_TCP_ROTATION_F_WXYZ,
    )
    print(json.dumps({key: manifest[key] for key in ("source_sha256", "derived_sha256", "velocity_changes")}))


if __name__ == "__main__":
    main()
