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
    if normalized(task["source"]) != normalized(asdict(TARGET_STATES[config.source_index])):
        raise ValueError("task source differs from TARGET_STATES")
    if normalized(task["destination"]) != normalized(asdict(TARGET_STATES[config.destination_index])):
        raise ValueError("task destination differs from TARGET_STATES")
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
