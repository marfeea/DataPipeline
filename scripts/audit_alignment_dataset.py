"""Produce reproducible v4 acceptance evidence and an optional legacy shape comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from DataPipe.config import ALIGNED_SCHEMA_VERSION
from DataPipe.mamba_sequence_buffer import MambaSequenceBuffer
from DataPipe.reward import compute_reward_prediction_loss
from DataPipe.validator import validate_hdf5_dataset

FILE_KEYS = ("task_id", "robot_model", "control_frequency_hz", "physics_dt_s")
EPISODE_KEYS = (
    "episode_id", "target_object_id", "tcp_frame_name", "grasp_definition_source",
    "selected_grasp_id", "grasp_reference_frame", "action_frame", "action_position_scale_m",
    "action_rotation_scale_rad", "goal_position_tolerance_m", "goal_rotation_tolerance_rad",
    "object_reference_frame", "object_state_source",
)


def field_shapes(group):
    result = {}

    def visit(parent, prefix="", ancestors=()):
        if parent.id in ancestors:
            return
        for key, node in parent.items():
            name = prefix + key
            if isinstance(node, h5py.Group):
                visit(node, name + "/", (*ancestors, parent.id))
                continue
            shape = list(node.shape)
            if name == "transforms/T_w_b" and node.ndim == 2:
                pass
            elif shape and shape[0] == group["actions"].shape[0] + 1:
                shape[0] = "T+1"
            elif shape and shape[0] == group["actions"].shape[0]:
                shape[0] = "T"
            result[name] = shape

    # Keep every public path, including aliases backed by HDF5 hard links.
    visit(group)
    return result


def audit(root):
    validation = validate_hdf5_dataset(root)
    errors = list(validation.errors)
    if errors:
        return {"passed": False, "errors": errors, "episodes": []}
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != ALIGNED_SCHEMA_VERSION:
        return {"passed": False, "errors": ["acceptance audit requires aligned v4"], "episodes": []}
    episodes = []
    for shard in manifest["shards"]:
        with h5py.File(root / shard["path"]) as handle:
            data = handle["data"]
            for key in FILE_KEYS:
                if key not in data.attrs:
                    errors.append(f"{shard['path']}: missing file metadata {key}")
            for name, demo in data.items():
                for key in EPISODE_KEYS:
                    if key not in demo.attrs:
                        errors.append(f"{name}: missing episode metadata {key}")
                episodes.append({
                    "episode_id": str(demo.attrs["episode_id"]),
                    "termination_reason": str(demo.attrs["termination_reason"]),
                    "transitions": demo["actions"].shape[0],
                    "observations": demo["obs/q"].shape[0],
                    "valid_rewards": int(np.count_nonzero(demo["reward_valid"])),
                    "commanded_applied_max_abs_difference": float(np.max(np.abs(
                        demo["action_commanded"][...] - demo["action_applied"][...]
                    ))),
                    "field_shapes": field_shapes(demo),
                })
    if not episodes:
        errors.append("acceptance audit requires at least one episode")
    if episodes and not errors:
        horizon = min(16, min(row["transitions"] for row in episodes))
        with MambaSequenceBuffer(root, horizon=horizon, state_keys=("q", "tcp_pose_b", "gripper_force")) as buffer:
            batch = buffer.sample(2)
            loss = compute_reward_prediction_loss(
                np.ones_like(batch["rewards"]), batch["rewards"], reward_valid=batch["reward_valid"],
            )
            if loss != 0 or np.any(batch["reward_valid"]):
                errors.append("placeholder reward reached the training target")
    return {"passed": not errors, "errors": errors, "episodes": episodes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--legacy-dataset", type=Path)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.dataset)
    if args.legacy_dataset is not None and result["episodes"]:
        legacy_validation = validate_hdf5_dataset(args.legacy_dataset)
        manifest = json.loads((args.legacy_dataset / "manifest.json").read_text(encoding="utf-8"))
        with h5py.File(args.legacy_dataset / manifest["shards"][0]["path"]) as handle:
            legacy = field_shapes(next(iter(handle["data"].values())))
        aligned = result["episodes"][0]["field_shapes"]
        result["legacy_comparison"] = {
            "schema_version": manifest["schema_version"], "passed": legacy_validation.passed,
            "legacy_fields": legacy, "aligned_fields": aligned,
            "added_paths": sorted(set(aligned) - set(legacy)),
            "removed_paths": sorted(set(legacy) - set(aligned)),
            "changed_shapes": {key: [legacy[key], aligned[key]] for key in legacy.keys() & aligned.keys()
                               if legacy[key] != aligned[key]},
        }
        if not legacy_validation.passed:
            result["passed"] = False
            result["errors"].extend(legacy_validation.errors)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "episodes": len(result["episodes"]), "errors": result["errors"]}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
