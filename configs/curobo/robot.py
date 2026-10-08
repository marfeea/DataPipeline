"""Measured robot limits must be supplied explicitly; never invent asset limits."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class RobotConfig:
    joint_names: tuple[str, ...]
    position_lower: tuple[float, ...]
    position_upper: tuple[float, ...]
    velocity_limit: tuple[float, ...]
    acceleration_limit: tuple[float, ...]
    jerk_limit: tuple[float, ...]
    base_frame: str
    flange_frame: str
    tcp_frame: str
    flange_to_tcp: tuple[float, ...]  # xyz + wxyz
    quaternion_order: str = "wxyz"
    position_unit: str = "m"
    joint_unit: str = "rad"

    def __post_init__(self):
        names = tuple(self.joint_names)
        if len(names) != 6 or len(set(names)) != 6 or not all(isinstance(x, str) and x.strip() for x in names):
            raise ValueError("six distinct joint names are required")
        object.__setattr__(self, "joint_names", names)
        for field in ("position_lower", "position_upper", "velocity_limit", "acceleration_limit", "jerk_limit"):
            values = tuple(float(x) for x in getattr(self, field))
            if len(values) != 6 or not all(isfinite(x) for x in values):
                raise ValueError(f"{field} must have six finite values")
            if field.endswith("limit") and any(x <= 0 for x in values):
                raise ValueError(f"{field} must be positive")
            object.__setattr__(self, field, values)
        if any(lo >= hi for lo, hi in zip(self.position_lower, self.position_upper)):
            raise ValueError("joint lower limits must be below upper limits")
        if not all(isinstance(x, str) and x.strip() for x in (self.base_frame, self.flange_frame, self.tcp_frame)):
            raise ValueError("explicit frame names are required")
        if len({self.base_frame, self.flange_frame, self.tcp_frame}) != 3:
            raise ValueError("base/flange/TCP frame names must be distinct")
        pose = tuple(float(x) for x in self.flange_to_tcp)
        if len(pose) != 7 or not all(isfinite(x) for x in pose) or abs(sum(x * x for x in pose[3:]) - 1) > 1e-6:
            raise ValueError("flange_to_tcp must be xyz plus a unit wxyz quaternion")
        object.__setattr__(self, "flange_to_tcp", pose)
        if (self.quaternion_order, self.position_unit, self.joint_unit) != ("wxyz", "m", "rad"):
            raise ValueError("only wxyz, metres and radians are supported")
