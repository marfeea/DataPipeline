"""Search deterministic collision-free Legacy IK branches for a V2 free-space goal."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from DataPipe.runtime import ensure_curriculumrl_available

ensure_curriculumrl_available()
from CurriculumRL.logic.aubo_ik import JOINT_LIMITS_RAD, solve_tcp_ik

from configs.curobo.world import MeshConfig, WorldConfig
from DataPipe.pipelines.curobo_bootstrap.attachment_geometry import attached_sphere_from_measured_target
from DataPipe.pipelines.curobo_bootstrap.curobo_collision import CuroboCollision, self_collision_pairs
from DataPipe.pipelines.curobo_bootstrap.route_identity import require_report_identity, task_identity
from DataPipe.pipelines.curobo_bootstrap.world_adapter import WorldSnapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-report", type=Path, required=True)
    parser.add_argument("--targets-report", type=Path, required=True)
    parser.add_argument("--waypoint", choices=("pre_pick", "place_high"), default="pre_pick")
    parser.add_argument("--legacy-contact-report", type=Path)
    parser.add_argument("--seed-count", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.seed_count < 1:
        raise ValueError("seed-count must be positive")
    world_report = json.loads(args.world_report.read_text(encoding="utf-8"))
    target_report = json.loads(args.targets_report.read_text(encoding="utf-8"))
    task = target_report["task_snapshot"]
    identity = task_identity(task)
    require_report_identity(target_report, task, label="targets")
    require_report_identity(world_report, task, label="world", require_passed=False)
    if not world_report.get("world_geometry_valid"):
        raise ValueError("selected world lacks validated geometry evidence")
    world_data = world_report["world"]
    meshes = tuple(MeshConfig(**item) for item in world_data["meshes"])
    for mesh in meshes:
        if hashlib.sha256(Path(mesh.file_path).read_bytes()).hexdigest() != mesh.sha256:
            raise ValueError(f"world mesh evidence mismatch: {mesh.name}")
    world = WorldSnapshot(WorldConfig((), world_data["target_name"], meshes))
    if world.world_hash != world_report["world_hash"]:
        raise ValueError("world hash differs from selected world report")
    attached_world = None
    if args.waypoint == "place_high":
        if args.legacy_contact_report is None:
            raise ValueError("place_high search requires --legacy-contact-report")
        contact = json.loads(args.legacy_contact_report.read_text(encoding="utf-8"))
        require_report_identity(contact, task, label="Legacy contact", world_hash=world.world_hash)
        target_mesh = next(mesh for mesh in meshes if mesh.name == world.config.target_name)
        vertices_b = np.asarray(trimesh.load_mesh(target_mesh.file_path, process=False).vertices)
        sphere = attached_sphere_from_measured_target(
            vertices_b,
            base_pose_w=contact["base_pose_w"],
            object_pose_w_at_reset=contact["target_pose_w_at_reset"],
            object_pose_w_current=contact["target_pose_w_after_lift"],
            tcp_pose_b_current=contact["lift_tcp_pose_b"],
        )
        attached_world = world.attach((sphere,))
        q_start = np.asarray(contact["lift_q"], dtype=np.float64)
    else:
        q_start = np.asarray(target_report["legacy_ik"]["q_start"], dtype=np.float64)
    pose = target_report["legacy_ik"]["targets"][args.waypoint]["tcp_pose_b"]
    reference_q = np.asarray(target_report["legacy_ik"]["targets"][args.waypoint]["joint_position"], dtype=np.float64)
    rng = np.random.default_rng(target_report["config"]["seed"])
    seeds = [q_start, reference_q]
    seeds.extend(rng.uniform(JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1]) for _ in range(args.seed_count))
    candidates = []
    for seed in seeds:
        solution = solve_tcp_ik(pose[:3], pose[3:], seed=seed)
        if not solution.success or solution.q is None:
            continue
        q = np.asarray(solution.q, dtype=np.float64)
        if solution.position_error_m > 1e-5 or solution.orientation_error_rad > 1e-4:
            continue
        if any(np.max(np.abs(q - prior)) < 0.01 for prior in candidates):
            continue
        candidates.append(q)
    if not candidates:
        raise RuntimeError("no Legacy IK branch converged")
    checker = CuroboCollision(world_report["collision_model"], world)
    if attached_world is not None:
        checker.attach(attached_world)
        full_start_scores = None
        full_scores = None
        collision_world = attached_world
    else:
        full_start_scores = checker.query(np.asarray([q_start]), world_ignore_links=("Base_Link",))
        full_scores = checker.query(np.asarray(candidates), world_ignore_links=("Base_Link",))
        collision_world = world.permit_target_contact()
        checker.update_world(collision_world)
    start_scores = checker.query(np.asarray([q_start]), world_ignore_links=("Base_Link",))
    start_world_score = float(start_scores["world"][0])
    start_self_score = float(start_scores["self"][0])
    scores = checker.query(np.asarray(candidates), world_ignore_links=("Base_Link",))
    records = []
    for index, q in enumerate(candidates):
        world_score = float(scores["world"][index])
        self_score = float(scores["self"][index])
        records.append(
            {
                "joint_position": q.tolist(),
                "max_abs_displacement_from_start_rad": float(np.max(np.abs(q - q_start))),
                "l2_displacement_from_start_rad": float(np.linalg.norm(q - q_start)),
                "world_collision_score": world_score,
                "world_collision_score_with_target": (
                    float(full_scores["world"][index]) if full_scores is not None else world_score
                ),
                "self_collision_score": self_score,
                "self_collision_pairs": (
                    self_collision_pairs(checker.asset, scores["spheres"][index]) if self_score > 0 else []
                ),
                "collision_free": world_score <= 0 and self_score <= 0,
            }
        )
    records.sort(key=lambda item: (not item["collision_free"], item["l2_displacement_from_start_rad"]))
    result = {
        "schema": "curobo_v2_goal_branches/1",
        **identity,
        "waypoint": args.waypoint,
        "passed": (
            start_world_score <= 0
            and start_self_score <= 0
            and any(item["collision_free"] for item in records)
        ),
        "start_world_collision_score": start_world_score,
        "start_world_collision_score_with_target": (
            float(full_start_scores["world"][0]) if full_start_scores is not None else start_world_score
        ),
        "start_self_collision_score": start_self_score,
        "collision_policy": {
            "allowed_world_contact": [world.config.target_name] if args.waypoint == "pre_pick" else [],
            "protected_world_meshes": [mesh.name for mesh in collision_world.active_meshes],
            "self_collision_hard_gate": True,
        },
        "world_hash": world_report["world_hash"],
        "planning_world_hash": collision_world.world_hash,
        "seed": target_report["config"]["seed"],
        "random_seed_count": args.seed_count,
        "candidate_count": len(records),
        "collision_free_count": sum(item["collision_free"] for item in records),
        "candidates": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("candidate_count", "collision_free_count")}, indent=2))
    print(json.dumps(records[:5], indent=2))


if __name__ == "__main__":
    main()

