"""P4 GPU primitive-world and attach/detach smoke without Isaac application startup."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.curobo.world import CuboidConfig, WorldConfig
from DataPipe.pipelines.curobo_bootstrap.world_adapter import AttachedSphere, WorldSnapshot, curobo_world_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch

    from curobo.geom.sdf.world import CollisionQueryBuffer, WorldCollisionConfig, WorldPrimitiveCollision
    from curobo.geom.types import WorldConfig as CuroboWorldConfig
    from curobo.types.base import TensorDeviceType

    world = WorldConfig(
        (
            CuboidConfig("table", (0, 0, -0.1, 1, 0, 0, 0), (2, 2, 0.2)),
            CuboidConfig("target", (0.3, 0, 0.05, 1, 0, 0, 0), (0.04, 0.04, 0.1)),
        ),
        "target",
    )
    before = WorldSnapshot(world)
    attached = before.attach((AttachedSphere((0, 0, 0.03), 0.02),))
    tensor_args = TensorDeviceType(device=torch.device("cuda:0"))

    def query(snapshot, points):
        model = CuroboWorldConfig.from_dict(curobo_world_dict(snapshot))
        checker = WorldPrimitiveCollision(WorldCollisionConfig(tensor_args=tensor_args, world_model=model))
        spheres = tensor_args.to_device(points).view(1, 1, -1, 4)
        buffer = CollisionQueryBuffer.initialize_from_shape(spheres.shape, tensor_args, checker.collision_types)
        return (
            checker.get_sphere_distance(spheres, buffer, tensor_args.to_device([1]), tensor_args.to_device([0]))
            .view(-1)
            .cpu()
            .tolist()
        )

    points = [[0.3, 0, 0.05, 0.01], [0, 0, 0.2, 0.01]]
    before_distance = query(before, points)
    attached_distance = query(attached, points)
    detached_distance = query(attached.detach(), points)
    passed = before_distance[0] > 0 and attached_distance[0] == 0 and detached_distance == before_distance
    report = {
        "passed": passed,
        "world_hash": before.world_hash,
        "before": before_distance,
        "attached": attached_distance,
        "detached": detached_distance,
        "target_disabled_while_attached": True,
        "cuda_device": torch.cuda.get_device_name(0),
        "scope": "synthetic cuboid world; production scene geometry and robot spheres not validated",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    if not passed:
        raise RuntimeError(report)


if __name__ == "__main__":
    main()
