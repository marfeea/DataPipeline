"""Reconstruct state4 placement with the measured grasp transform from saved evidence."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from DataPipe.pipelines.curobo_bootstrap.replay import matrix_pose_wxyz
from DataPipe.pipelines.curobo_bootstrap.robot_adapter import pose_matrix


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def transform(pose):
    pose = np.asarray(pose, dtype=float)
    return pose_matrix(pose[:3], pose[3:] / np.linalg.norm(pose[3:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    evidence = run.parent
    report = read(run / "report.json")
    targets = read(evidence / "targets.json")
    world = read(evidence / "world/report.json")
    task = targets["task_snapshot"]
    assert task["destination_index"] == 4 and report["task_hash"] == task["task_hash"]
    world_from_base = transform(report["base_pose_w"])
    tcp_from_object = np.linalg.inv(world_from_base @ transform(report["lift_tcp_pose_b"])) @ transform(
        report["target_pose_w_after_lift"]
    )
    nominal_tcp = transform(targets["legacy_ik"]["targets"]["place"]["tcp_pose_b"])
    nominal_object = world_from_base @ nominal_tcp @ tcp_from_object
    destination = task["destination"]
    desired_object = np.asarray(task["frames"]["T_WE"]) @ pose_matrix(
        destination["position_e"], destination["rotation_wxyz"]
    )
    compensated_tcp = np.linalg.inv(world_from_base) @ desired_object @ np.linalg.inv(tcp_from_object)
    assert np.allclose(world_from_base @ compensated_tcp @ tcp_from_object, desired_object, atol=1e-10)
    mesh = next(m for m in world["world"]["meshes"] if m["name"] == "target")
    vertices_b = np.asarray(trimesh.load_mesh(mesh["file_path"], process=False).vertices)
    homogeneous = np.column_stack((vertices_b, np.ones(len(vertices_b))))
    vertices_object = (np.linalg.inv(transform(report["target_pose_w_at_reset"])) @ world_from_base @ homogeneous.T).T

    def bounds(pose):
        vertices = (pose @ vertices_object.T).T[:, :3]
        return dict(min_w=vertices.min(axis=0).tolist(), max_w=vertices.max(axis=0).tolist())

    def tilt(pose):
        return float(np.degrees(np.arccos(np.clip(pose[2, 2], -1, 1))))

    nominal_bounds, desired_bounds = bounds(nominal_object), bounds(desired_object)
    records = read(run / "trace.json")
    descent = [r for r in records if r.get("contact_waypoint") == "place_descent_outside_roof"]
    insertion = [r for r in records if r.get("contact_waypoint") == "place_insert_xy"]
    stages = {}
    for name, values in (("descent_start", descent[:1]), ("insertion_start", insertion[:1]), ("final", records[-1:])):
        if values:
            row = values[0]
            stages[name] = dict(
                tcp_position_b=row["tcp_position_b"],
                object_pose_w=row["target_pose_w"],
                object_bounds=bounds(transform(row["target_pose_w"])),
                tilt_deg=tilt(transform(row["target_pose_w"])),
            )
    dependencies = [run / "report.json", run / "trace.json", evidence / "targets.json", evidence / "world/report.json"]
    result = dict(
        schema="state4_measured_placement_analysis/1",
        source_run=str(run),
        source_index=task["source_index"],
        tcp_from_object=tcp_from_object.tolist(),
        nominal_tcp_pose_b=list(matrix_pose_wxyz(nominal_tcp)),
        compensated_tcp_pose_b=list(matrix_pose_wxyz(compensated_tcp)),
        tcp_compensation_position_b_m=(compensated_tcp[:3, 3] - nominal_tcp[:3, 3]).tolist(),
        nominal_predicted_object_pose_w=list(matrix_pose_wxyz(nominal_object)),
        nominal_predicted_object_tilt_deg=tilt(nominal_object),
        desired_object_pose_w=list(matrix_pose_wxyz(desired_object)),
        nominal_predicted_object_position_error_m=float(np.linalg.norm(nominal_object[:3, 3] - desired_object[:3, 3])),
        nominal_predicted_bounds=nominal_bounds,
        desired_bounds=desired_bounds,
        nominal_roof_clearance_m=1.144 - nominal_bounds["max_w"][2],
        desired_roof_clearance_m=1.144 - desired_bounds["max_w"][2],
        nominal_bottom_relative_to_desired_m=nominal_bounds["min_w"][2] - desired_bounds["min_w"][2],
        stages=stages,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies},
        scope=(
            "Rigid grasp reconstruction and mesh AABB against saved roof height; "
            "not contact-pair proof or execution acceptance"
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("source_sha256", "tcp_from_object", "stages")}))


if __name__ == "__main__":
    main()
