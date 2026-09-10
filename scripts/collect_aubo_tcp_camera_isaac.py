"""Run the dataset collector with the verified three-camera visual pipeline.

HandCamera, CameraSensor_2 and CameraSensor_3 are configured by the shared
DataPipe.collection_cameras module used by the dataset entry point.
"""

from __future__ import annotations


def main() -> int:
    from collect_aubo_ik_dataset_isaac import main as collect

    return int(collect())


if __name__ == "__main__":
    raise SystemExit(main())
