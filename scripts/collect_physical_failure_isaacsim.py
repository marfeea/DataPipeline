"""Compatibility entry for the shared physical execution used by collect_isaacsim."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pipelines.curobo_bootstrap.physical_execution import parse_args, run_physical_collection


def main():
    return run_physical_collection(parse_args())


if __name__ == "__main__":
    os._exit(main())
