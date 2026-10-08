"""Route the source2 approach beside and above its unchanged grasp target."""

from dataclasses import dataclass
from math import hypot, isfinite


@dataclass(frozen=True)
class Source2RoutingConfig:
    pre_pick_side_m: float = 0.06
    pre_pick_lift_m: float = 0.0

    def __post_init__(self):
        if not isfinite(self.pre_pick_side_m) or self.pre_pick_side_m == 0:
            raise ValueError("pre_pick_side_m must be finite and nonzero")
        if not isfinite(self.pre_pick_lift_m) or self.pre_pick_lift_m < 0:
            raise ValueError("pre_pick_lift_m must be finite and nonnegative")


def add_source2_routing_args(parser):
    parser.add_argument("--state2-pre-pick-side-m", type=float, default=0.0)
    parser.add_argument("--state2-pre-pick-lift-m", type=float, default=0.0)


def source2_routing_from_args(args):
    side = args.state2_pre_pick_side_m
    lift = args.state2_pre_pick_lift_m
    if lift and not side:
        raise ValueError("state2 lift requires a nonzero side offset")
    return Source2RoutingConfig(side, lift) if side else None


def source2_routing_arguments(config):
    if config is None:
        return []
    arguments = ["--state2-pre-pick-side-m", str(config.pre_pick_side_m)]
    if config.pre_pick_lift_m:
        arguments += ["--state2-pre-pick-lift-m", str(config.pre_pick_lift_m)]
    return arguments


def source2_routing_payload(config):
    """Preserve historical zero-lift task hashes."""
    if config is None:
        return None
    payload = dict(pre_pick_side_m=config.pre_pick_side_m)
    if config.pre_pick_lift_m:
        payload["pre_pick_lift_m"] = config.pre_pick_lift_m
    return payload


def source2_advance_waypoints(*, pick, config):
    """Approach above the target before the final vertical grasp descent."""
    pick = tuple(pick)
    if config is None or not config.pre_pick_lift_m:
        return (pick,)
    if len(pick) != 3 or not all(isfinite(v) for v in pick):
        raise ValueError("source2 grasp must be a finite xyz position")
    return ((*pick[:2], pick[2] + config.pre_pick_lift_m), pick)


def source2_pre_pick(*, source_index, pre_pick, pick, config):
    """Positive offsets follow the horizontal perpendicular to the outward approach."""
    if config is None:
        return tuple(pre_pick)
    if source_index != 2:
        raise ValueError("source2 routing requires source_index=2")
    if len(pre_pick) != 3 or len(pick) != 3 or not all(isfinite(v) for v in (*pre_pick, *pick)):
        raise ValueError("source2 approach positions must be finite xyz positions")
    dx, dy = pre_pick[0] - pick[0], pre_pick[1] - pick[1]
    length = hypot(dx, dy)
    if length <= 1e-9:
        raise ValueError("source2 routing requires a horizontal approach direction")
    return (
        pre_pick[0] - dy / length * config.pre_pick_side_m,
        pre_pick[1] + dx / length * config.pre_pick_side_m,
        pre_pick[2] + config.pre_pick_lift_m,
    )
