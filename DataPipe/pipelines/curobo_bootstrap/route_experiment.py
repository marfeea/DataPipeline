"""Freeze route heights and reject mixed experimental data on resume."""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path

from DataPipe.source2_routing import Source2RoutingConfig
from DataPipe.state4_routing import State4RoutingConfig

from .task_targets import LIFT_CLEARANCE_M, PickPlaceV2Config


def route_config(
    source, destination, *, lift_clearance_m=LIFT_CLEARANCE_M, state4_lift_clearance_m=None,
    state4_routing=None, source2_routing=None
):
    clearance = (
        state4_lift_clearance_m
        if state4_lift_clearance_m is not None and 4 in (source, destination)
        else lift_clearance_m
    )
    return PickPlaceV2Config(
        source_index=source,
        destination_index=destination,
        lift_clearance_m=clearance,
        state4_routing=state4_routing if 4 in (source, destination) else None,
        source2_routing=source2_routing if source == 2 else None,
    )


def freeze_experiment(root: Path, experiment: dict) -> dict:
    """Keep the first manifest; never silently mix heights, models or GUI modes."""
    path = root / "experiment_manifest.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != experiment:
            raise ValueError("cannot resume with a different frozen experiment")
        return saved
    if (root / "graph_summary.json").exists() or any(root.glob("*_to_*")):
        raise ValueError("existing route data lacks a frozen experiment; select a new output root")
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(experiment, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return experiment


def require_height(task: dict, expected: float) -> None:
    """Audit the configured height and both horizontal-transfer waypoints."""
    config = PickPlaceV2Config(
        source_index=task["source_index"], destination_index=task["destination_index"], lift_clearance_m=expected
    )
    if task["lift_clearance_m"] != config.lift_clearance_m:
        raise ValueError("task height differs from frozen experiment")
    if "state4_routing" in task or "source2_routing" in task:
        import numpy as np

        from .task_targets import _rotation_wxyz_from_matrix, task_snapshot

        transform = np.asarray(task["frames"]["T_WB"])
        origin = np.asarray(task["frames"]["T_WE"])[:3, 3]
        config = PickPlaceV2Config(
            source_index=task["source_index"],
            destination_index=task["destination_index"],
            control_hz=task["control_hz"],
            seed=task["seed"],
            lift_clearance_m=expected,
            scene_id=task["scene_id"],
            state4_routing=State4RoutingConfig(**task["state4_routing"]) if "state4_routing" in task else None,
            source2_routing=Source2RoutingConfig(**task["source2_routing"]) if "source2_routing" in task else None,
        )
        expected_task = task_snapshot(
            config=config,
            environment_origin_w=origin,
            base_pose_w=(*transform[:3, 3], *_rotation_wxyz_from_matrix(transform)),
        )
        for field in ("waypoint_position_b", "contact_waypoint_position_b"):
            if field not in expected_task:
                continue
            if set(task[field]) != set(expected_task[field]) or any(
                not np.allclose(task[field][name], value, atol=1e-12, rtol=0)
                for name, value in expected_task[field].items()
            ):
                raise ValueError("state4 routing geometry differs from frozen experiment")
        return
    positions = task["waypoint_position_b"]
    safe_z = max(positions["pick"][2], positions["place"][2]) + config.lift_clearance_m
    if not isfinite(safe_z):
        raise ValueError("transfer height must be finite")
    for name in ("lift", "place_high"):
        if not isfinite(positions[name][2]) or abs(positions[name][2] - safe_z) > 1e-12:
            raise ValueError(f"{name} height differs from frozen experiment geometry")
