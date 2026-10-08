"""Acceptance thresholds are explicit experiment inputs, not inferred defaults."""

from dataclasses import dataclass, fields
from math import isfinite


@dataclass(frozen=True)
class AcceptanceConfig:
    joint_tracking_max_rad: float
    tcp_tracking_max_m: float
    place_position_max_m: float
    place_rotation_max_rad: float
    collision_margin_m: float
    fk_position_max_m: float
    fk_rotation_max_rad: float

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{field.name} must be finite and nonnegative")
