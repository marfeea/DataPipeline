"""Output root is isolated from the Legacy IK collector."""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BootstrapConfig:
    run_id: str
    seed: int = 0
    episode_count: int = 1
    output_root: str = "data/curobo_bootstrap"

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", self.run_id):
            raise ValueError("run_id must be a safe single path component")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be a nonnegative 63-bit integer")
        if type(self.episode_count) is not int or self.episode_count < 1:
            raise ValueError("episode_count must be positive")
        if Path(self.output_root).name != "curobo_bootstrap":
            raise ValueError("output root must be a dedicated curobo_bootstrap directory")

    @property
    def run_directory(self) -> Path:
        return Path(self.output_root) / self.run_id
