"""Fixed CuRobo/Legacy IK ownership for the V2 0-to-1 MVP route."""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class RouteStage(str, Enum):
    MOVE_TO_PICK = "MOVE_TO_PICK"
    PICK = "PICK"
    MOVE_TO_PLACE = "MOVE_TO_PLACE"
    PLACE = "PLACE"


class RouteSource(str, Enum):
    CUROBO = "curobo"
    LEGACY_IK = "legacy_ik"
    GRIPPER_EVENT = "gripper_event"


@dataclass(frozen=True)
class RouteStep:
    name: str
    stage: RouteStage
    source: RouteSource
    start: str
    goal: str


HYBRID_ROUTE = (
    RouteStep("free_space_to_pre_pick", RouteStage.MOVE_TO_PICK, RouteSource.CUROBO, "q_start", "pre_pick"),
    RouteStep("legacy_approach_pick", RouteStage.MOVE_TO_PICK, RouteSource.LEGACY_IK, "pre_pick", "pick"),
    RouteStep("close_gripper", RouteStage.PICK, RouteSource.GRIPPER_EVENT, "pick", "pick"),
    RouteStep("legacy_lift", RouteStage.MOVE_TO_PLACE, RouteSource.LEGACY_IK, "pick", "lift"),
    RouteStep("free_space_transfer", RouteStage.MOVE_TO_PLACE, RouteSource.CUROBO, "lift", "place_high"),
    RouteStep("legacy_place_approach", RouteStage.MOVE_TO_PLACE, RouteSource.LEGACY_IK, "place_high", "place"),
    RouteStep("open_gripper", RouteStage.PLACE, RouteSource.GRIPPER_EVENT, "place", "place"),
)


def curobo_goal_specs(joint_targets):
    """Return two MotionGen goals; execution must use the measured segment start."""
    expected = {name for step in HYBRID_ROUTE for name in (step.start, step.goal)}
    if set(joint_targets) != expected:
        raise ValueError(f"joint target fields differ: {sorted(set(joint_targets) ^ expected)}")
    values = {name: np.asarray(value, dtype=np.float64) for name, value in joint_targets.items()}
    if any(value.shape != (6,) or not np.isfinite(value).all() for value in values.values()):
        raise ValueError("all joint targets must contain six finite values")
    return tuple(
        (step.name, step.start, values[step.goal].copy()) for step in HYBRID_ROUTE if step.source is RouteSource.CUROBO
    )


def validate_hybrid_route(steps):
    """Reject alternate ownership or discontinuous waypoint order."""
    steps = tuple(steps)
    if steps != HYBRID_ROUTE:
        raise ValueError("V2 MVP hybrid route differs from the frozen Legacy/CuRobo ownership")
    if any(left.goal != right.start for left, right in zip(steps, steps[1:], strict=False)):
        raise ValueError("hybrid route waypoint seam is discontinuous")
    return steps
