"""State4 contact waypoints: lift below its roof, withdraw, and insert horizontally."""

from dataclasses import dataclass
from math import hypot, isfinite


def add_state4_routing_args(parser):
    import argparse

    parser.add_argument("--state4-routing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--state4-pick-lift-m", type=float, default=0.015)
    parser.add_argument("--state4-pick-exit-m", type=float, default=0.18)
    parser.add_argument("--state4-place-standoff-m", type=float, default=0.18)
    parser.add_argument("--state4-place-entry-lift-m", type=float, default=0.0)
    parser.add_argument("--state4-place-measured-grasp", action=argparse.BooleanOptionalAction, default=False)


def routing_from_args(args):
    return (
        State4RoutingConfig(
            pick_lift_m=args.state4_pick_lift_m,
            pick_exit_m=args.state4_pick_exit_m,
            place_standoff_m=args.state4_place_standoff_m,
            place_entry_lift_m=args.state4_place_entry_lift_m,
            place_measured_grasp=args.state4_place_measured_grasp,
        )
        if args.state4_routing
        else None
    )


def routing_arguments(config):
    if config is None:
        return []
    arguments = [
        "--state4-routing",
        "--state4-pick-lift-m",
        str(config.pick_lift_m),
        "--state4-pick-exit-m",
        str(config.pick_exit_m),
        "--state4-place-standoff-m",
        str(config.place_standoff_m),
    ]
    if config.place_entry_lift_m:
        arguments += ["--state4-place-entry-lift-m", str(config.place_entry_lift_m)]
    if config.place_measured_grasp:
        arguments += ["--state4-place-measured-grasp"]
    return arguments


def routing_payload(config):
    """Keep historical identities byte equivalent when the optional entry lift is zero."""
    if config is None:
        return None
    payload = dict(vars(config))
    if not config.place_entry_lift_m:
        payload.pop("place_entry_lift_m")
    if not config.place_measured_grasp:
        payload.pop("place_measured_grasp")
    return payload


@dataclass(frozen=True)
class State4RoutingConfig:
    pick_lift_m: float = 0.015
    pick_exit_m: float = 0.18
    place_standoff_m: float = 0.18
    place_entry_lift_m: float = 0.0
    place_measured_grasp: bool = False

    def __post_init__(self):
        for name in ("pick_lift_m", "pick_exit_m", "place_standoff_m"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isfinite(self.place_entry_lift_m) or self.place_entry_lift_m < 0:
            raise ValueError("place_entry_lift_m must be finite and nonnegative")
        if type(self.place_measured_grasp) is not bool:
            raise ValueError("place_measured_grasp must be boolean")


def offset_away_xy(position, preposition, distance):
    dx, dy = preposition[0] - position[0], preposition[1] - position[1]
    length = hypot(dx, dy)
    if length <= 1e-9:
        raise ValueError("state4 requires a nonzero horizontal approach direction")
    return (position[0] + distance * dx / length, position[1] + distance * dy / length, position[2])


def state4_waypoints(
    *, source_index, destination_index, pre_pick, pick, lift, place_high, place, place_preposition, config
):
    positions = dict(lift=tuple(lift), place_high=tuple(place_high), place=tuple(place))
    contact = {}
    if source_index == 4:
        low_lift = (pick[0], pick[1], pick[2] + config.pick_lift_m)
        contact["pick_low_lift"] = low_lift
        positions["lift"] = offset_away_xy(low_lift, pre_pick, config.pick_exit_m)
    if destination_index == 4:
        if place_preposition is None:
            raise ValueError("state4 placement requires place_preposition")
        lowered = offset_away_xy(place, place_preposition, config.place_standoff_m)
        lowered = (*lowered[:2], lowered[2] + config.place_entry_lift_m)
        positions["place_high"] = (*lowered[:2], place_high[2])
        contact["place_descent"] = lowered
        if config.place_entry_lift_m or config.place_measured_grasp:
            contact["place_entry"] = (*place[:2], place[2] + config.place_entry_lift_m)
    return positions, contact
