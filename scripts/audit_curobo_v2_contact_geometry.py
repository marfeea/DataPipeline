"""Prove pick-pose contact against URDF gripper boxes and frozen production meshes."""

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from DataPipe.pipelines.curobo_bootstrap.collision_model import (
    _collision_bounds,
    _origin_matrix,
    load_collision_model,
)
from DataPipe.pipelines.curobo_bootstrap.robot_adapter import pose_matrix


def _joint(root, name):
    return next(item for item in root.findall("joint") if item.attrib["name"] == name)


def _finger_bounds(root, urdf_dir, link_name):
    link = next(item for item in root.findall("link") if item.attrib["name"] == link_name)
    low, high = _collision_bounds(link, urdf_dir, [])
    if link_name == "gripper_base":
        return low, high
    joint = _joint(root, link_name)
    limit = joint.find("limit")
    open_value = float(limit.attrib["upper" if link_name == "UpperFinger" else "lower"])
    axis = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
    transform = _origin_matrix(joint.find("origin"))
    transform[:3, 3] += axis * open_value
    corners = np.array([[x, y, z, 1] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    points = (transform @ corners.T).T[:, :3]
    return points.min(axis=0), points.max(axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-report", type=Path, required=True)
    parser.add_argument("--targets-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    world_report = json.loads(args.world_report.read_text(encoding="utf-8"))
    targets_report = json.loads(args.targets_report.read_text(encoding="utf-8"))
    if not targets_report["passed"] or not targets_report["legacy_ik"]["targets"]["pick"]["passed"]:
        raise ValueError("pick target must be a passed Legacy IK solution")
    model = load_collision_model(world_report["collision_model"])
    urdf = Path(model["urdf"])
    if hashlib.sha256(urdf.read_bytes()).hexdigest() != model["urdf_sha256"]:
        raise ValueError("URDF hash mismatch")
    root = ET.fromstring(urdf.read_bytes())
    flange_to_gripper = _origin_matrix(_joint(root, "flange_to_gripper").find("origin"))
    flange_to_tcp = _origin_matrix(_joint(root, "flange_to_project_tcp").find("origin"))
    tcp_pose = targets_report["legacy_ik"]["targets"]["pick"]["tcp_pose_b"]
    base_to_tcp = pose_matrix(tcp_pose[:3], tcp_pose[3:])
    gripper_from_base = np.linalg.inv(base_to_tcp @ np.linalg.inv(flange_to_tcp) @ flange_to_gripper)
    boxes = {name: _finger_bounds(root, urdf.parent, name) for name in ("gripper_base", "UpperFinger", "DownFinger")}
    result = {
        "schema": "curobo_v2_pick_contact_geometry/1",
        "world_report": str(args.world_report.resolve()),
        "targets_report": str(args.targets_report.resolve()),
        "urdf_sha256": model["urdf_sha256"],
        "tcp_pose_b": tcp_pose,
        "finger_policy": "URDF collision boxes at commanded open finger limits",
        "mesh_contacts": {},
    }
    for mesh in world_report["world"]["meshes"]:
        if mesh["name"] not in {"target", "support_tray_11"}:
            continue
        if not np.allclose(mesh["pose_b"], (0, 0, 0, 1, 0, 0, 0), rtol=0, atol=1e-12):
            raise ValueError("contact audit expects Base-frame OBJ vertices with identity mesh pose")
        path = Path(mesh["file_path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != mesh["sha256"]:
            raise ValueError(f"mesh hash mismatch: {mesh['name']}")
        vertices_b = np.asarray(trimesh.load_mesh(path, process=False).vertices, dtype=np.float64)
        vertices_g = (gripper_from_base @ np.column_stack((vertices_b, np.ones(len(vertices_b)))).T).T[:, :3]
        contacts = {}
        for name, (low, high) in boxes.items():
            inside = vertices_g[np.all((vertices_g >= low) & (vertices_g <= high), axis=1)]
            contacts[name] = {
                "mesh_vertices_inside_urdf_collision_box": int(len(inside)),
                "sample_vertices_gripper": inside[:8].tolist(),
                "box_minimum_gripper": low.tolist(),
                "box_maximum_gripper": high.tolist(),
            }
        result["mesh_contacts"][mesh["name"]] = contacts
    if set(result["mesh_contacts"]) != {"target", "support_tray_11"}:
        raise ValueError("frozen target and support_tray_11 meshes required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "mesh_contacts": result["mesh_contacts"]}, indent=2))


if __name__ == "__main__":
    main()
