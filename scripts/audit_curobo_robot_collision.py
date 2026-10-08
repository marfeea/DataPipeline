"""Run the P4 CuRobo GPU robot/world/self/attachment collision gate."""

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.curobo.world import CuboidConfig, WorldConfig
from DataPipe.pipelines.curobo_bootstrap.collision_model import load_collision_model
from DataPipe.pipelines.curobo_bootstrap.curobo_collision import CuroboCollision
from DataPipe.pipelines.curobo_bootstrap.robot_adapter import pose_matrix
from DataPipe.pipelines.curobo_bootstrap.world_adapter import AttachedSphere, WorldSnapshot

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--collision-model", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(f"Refusing to overwrite audit: {args.output}")


def score(values):
    return float(np.max(values))


def snapshot(target_pose=(3.0, 3.0, 3.0), probe_pose=(-3.0, -3.0, -3.0)):
    identity = (1.0, 0.0, 0.0, 0.0)
    return WorldSnapshot(
        WorldConfig(
            (
                CuboidConfig("target", (*target_pose, *identity), (0.02, 0.02, 0.02)),
                CuboidConfig("probe", (*probe_pose, *identity), (0.02, 0.02, 0.02)),
            ),
            "target",
        )
    )


def main():
    asset = load_collision_model(args.collision_model)
    far_world = snapshot()
    checker = CuroboCollision(asset, far_world, device=args.device)
    safe_q = np.array([0.0, -0.52359879, 1.22173047, 0.78539819, 1.57079637, 0.0])
    collision_q = np.array([-2.37419581, -2.98899174, 2.95020127, 0.51214123, -2.12580705, -0.51233101])

    safe = checker.query(safe_q)
    colliding = checker.query(collision_q)

    robot_probe_center = safe["spheres"][0, 0, :3]
    robot_probe_world = snapshot(probe_pose=robot_probe_center)
    checker.update_world(robot_probe_world)
    robot_world_collision = checker.query(safe_q)
    checker.update_world(far_world)

    attached_sphere = AttachedSphere((0.25, 0.0, 0.0), 0.04)
    tcp_pose_b = pose_matrix(safe["ee_position"][0], safe["ee_quaternion"][0])
    attached_world = far_world.attach((attached_sphere,))
    attached_center_b = attached_world.attached_spheres_b(tcp_pose_b)[0, :3]
    detached_probe_world = snapshot(probe_pose=attached_center_b)
    before_attach = checker.query(safe_q)
    attached_probe_world = detached_probe_world.attach((attached_sphere,))
    checker.attach(attached_probe_world)
    during_attach = checker.query(safe_q)
    checker.detach(detached_probe_world)
    after_detach = checker.query(safe_q)

    checks = {
        "joint_order": checker.joint_names == ("Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Flange"),
        "sphere_count": checker.total_spheres
        == sum(len(items) for items in asset["collision_spheres"].values()) + asset["attachment_slots"],
        "safe_self_zero": score(safe["self"]) == 0.0,
        "safe_world_zero": score(safe["world"]) == 0.0,
        "known_self_collision_positive": score(colliding["self"]) > 0.0,
        "robot_world_collision_positive": score(robot_world_collision["world"]) > 0.0,
        "before_attach_clear": score(before_attach["world"]) == 0.0,
        "attached_probe_collision_positive": score(during_attach["world"]) > 0.0,
        "after_detach_clear": score(after_detach["world"]) == 0.0,
        "target_removed_atomically": [item.name for item in attached_probe_world.active_cuboids] == ["probe"],
    }
    report = {
        "passed": all(checks.values()),
        "python": sys.executable,
        "curobo_version": checker.version,
        "device": args.device,
        "collision_model": str(args.collision_model.resolve()),
        "collision_model_urdf_sha256": asset["urdf_sha256"],
        "sphere_counts": {name: len(items) for name, items in asset["collision_spheres"].items()},
        "attachment_slots": asset["attachment_slots"],
        "checks": checks,
        "scores": {
            "safe_self": score(safe["self"]),
            "safe_world": score(safe["world"]),
            "known_self_collision": score(colliding["self"]),
            "robot_world_collision": score(robot_world_collision["world"]),
            "before_attach_world": score(before_attach["world"]),
            "during_attach_world": score(during_attach["world"]),
            "after_detach_world": score(after_detach["world"]),
        },
        "safe_q": safe_q.tolist(),
        "known_collision_q": collision_q.tolist(),
        "robot_probe_center_b": robot_probe_center.tolist(),
        "attached_probe": {
            "sphere_tcp": [*attached_sphere.center_tcp, attached_sphere.radius_m],
            "center_b": attached_center_b.tolist(),
        },
        "scope": (
            "synthetic deterministic GPU gate for robot spheres, self collision, primitive world collision, "
            "and attachment transitions; not production-scene clearance or MotionGen validation"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError("P4 collision audit failed")


try:
    main()
except Exception:
    traceback.print_exc()
    raise
