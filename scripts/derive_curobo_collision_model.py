"""Derive deterministic conservative collision spheres from the project AUBO URDF."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from DataPipe.pipelines.curobo_bootstrap.collision_model import write_collision_model

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--urdf", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--max-cell-m", type=float, default=0.08)
parser.add_argument("--padding-m", type=float, default=0.002)
parser.add_argument("--attachment-slots", type=int, default=4)
parser.add_argument("--gripper-mode", choices=("range_envelope", "fixed_open_components"), default="range_envelope")
parser.add_argument("--refined-flange-cell-m", type=float, default=None)
parser.add_argument("--gripper-cell-m", type=float, default=None)
args = parser.parse_args()

model = write_collision_model(
    args.urdf,
    args.output,
    max_cell_m=args.max_cell_m,
    padding_m=args.padding_m,
    attachment_slots=args.attachment_slots,
    gripper_mode=args.gripper_mode,
    refined_flange_cell_m=args.refined_flange_cell_m,
    gripper_cell_m=args.gripper_cell_m,
)
print(
    json.dumps(
        {
            "output": str(args.output.resolve()),
            "sphere_counts": {name: len(spheres) for name, spheres in model["collision_spheres"].items()},
        },
        indent=2,
    )
)
