"""Validate that every CuRobo V2 evidence file belongs to one selected route."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from configs.task import TARGET_STATES

from .contracts import config_hash
from .task_targets import PickPlaceV2Config


def task_identity(task: dict[str, Any]) -> dict[str, Any]:
    config = PickPlaceV2Config(
        source_index=task["source_index"],
        destination_index=task["destination_index"],
        control_hz=task["control_hz"],
        seed=task["seed"],
        lift_clearance_m=task["lift_clearance_m"],
        scene_id=task["scene_id"],
    )
    normalized = lambda value: json.loads(json.dumps(value))  # noqa: E731
    # Historical schemas remain readable for audits, but collection separately
    # rejects references that do not contain the current direct TCP targets.
    old_fields = {"name", "position_e", "rotation_wxyz", "preposition_e", "grasp_tcp_position_e"}
    old_placement_fields = old_fields | {
        "place_object_position_e", "place_tracking_compensation", "place_entry_lift_m", "place_standoff_m",
    }
    for role, index in (("source", config.source_index), ("destination", config.destination_index)):
        expected = asdict(TARGET_STATES[index])
        fields = set(task[role])
        if fields == old_placement_fields:
            expected["place_object_position_e"] = expected["pre_place_tcp_position_e"]
        elif fields not in (set(expected), old_fields):
            raise ValueError(f"task {role} differs from TARGET_STATES")
        if normalized(task[role]) != normalized({k: expected[k] for k in fields}):
            raise ValueError(f"task {role} differs from TARGET_STATES")
    payload = {key: value for key, value in task.items() if key != "task_hash"}
    if task.get("task_hash") != config_hash(payload):
        raise ValueError("task hash mismatch")
    return {
        "source_index": config.source_index,
        "destination_index": config.destination_index,
        "source_name": task["source"]["name"],
        "destination_name": task["destination"]["name"],
        "task_hash": task["task_hash"],
    }


def require_report_identity(
    report: dict[str, Any],
    task: dict[str, Any],
    *,
    label: str,
    require_passed: bool = True,
    world_hash: str | None = None,
) -> None:
    identity = task_identity(task)
    if require_passed and report.get("passed") is not True:
        raise ValueError(f"{label} did not pass")
    for key, expected in identity.items():
        if report.get(key) != expected:
            raise ValueError(f"{label} {key} differs from selected task")
    if world_hash is not None and report.get("world_hash") != world_hash:
        raise ValueError(f"{label} world_hash differs from selected world")
