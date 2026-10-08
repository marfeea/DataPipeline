"""No backend imports or automatic installation at configuration time."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class PlannerConfig:
    # Required: deployment must supply the tested release/commit, not a floating latest.
    version: str
    control_hz: float = 60.0
    max_attempts: int = 4
    timeout_s: float = 10.0
    device: str = "cuda:0"
    online_replanning: bool = False
    fallback_used: bool = False
    cuda_graph_enabled: bool = True
    graph_first: bool = False

    def __post_init__(self):
        if not isinstance(self.version, str) or not self.version.strip() or self.version.lower() == "latest":
            raise ValueError("an explicit tested CuRobo version is required")
        if any(not isfinite(x) or x <= 0 for x in (self.control_hz, self.timeout_s)):
            raise ValueError("control_hz and timeout_s must be finite and positive")
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if not self.device.startswith("cuda:") or not self.device[5:].isdigit():
            raise ValueError("an explicit CUDA device is required")
        if self.online_replanning is not False or self.fallback_used is not False:
            raise ValueError("online replanning and legacy fallback are forbidden")
        if type(self.cuda_graph_enabled) is not bool:
            raise ValueError("cuda_graph_enabled must be boolean")
        if type(self.graph_first) is not bool:
            raise ValueError("graph_first must be boolean")
