"""Offline per-link attribution of a recorded robot/tray sphere overlap."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.pipelines.curobo_bootstrap.collision_model import load_collision_model
from scripts.pipelines.curobo_bootstrap.robot_adapter import UrdfChain, pose_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--world-report", type=Path, required=True)
    parser.add_argument("--tray", default="support_tray_11")
    parser.add_argument("--collision-model", type=Path)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("stride must be positive")
    trace = json.loads(args.trace.read_text())
    world = json.loads(args.world_report.read_text())
    model = load_collision_model(args.collision_model or world["collision_model"])
    tray = next(m for m in world["world"]["meshes"] if m["name"] == args.tray)
    mesh = trimesh.load_mesh(tray["file_path"], process=False)
    mesh.apply_transform(pose_matrix(tray["pose_b"][:3], tray["pose_b"][3:]))
    result = {"scope": "Offline proxy attribution; does not certify physical collision freedom", "links": {}}
    # Include every close frame and every tenth approach/lift frame.
    close_indices = [i for i, row in enumerate(trace) if row["stage"] == "close_gripper"]
    indices = set(range(0, len(trace), args.stride))
    if close_indices:
        indices.update((close_indices[0], close_indices[len(close_indices) // 2], close_indices[-1]))
    indices = sorted(indices)
    for name, spheres in model["collision_spheres"].items():
        chain = UrdfChain(model["urdf"], "AUBO_E5", name)
        local = np.array([s["center"] for s in spheres])
        radii = np.array([s["radius"] for s in spheres])
        centers = []
        for i in indices:
            transform = chain.fk(np.asarray(trace[i]["q"])[:len(chain.joint_names)])
            centers.append(local @ transform[:3, :3].T + transform[:3, 3])
        centers = np.asarray(centers)
        _, distance, _ = trimesh.proximity.closest_point(mesh, centers.reshape(-1, 3))
        inside = mesh.contains(centers.reshape(-1, 3))
        distance[inside] *= -1
        margins = distance.reshape(centers.shape[:2]) - radii
        frame, sphere = np.unravel_index(np.argmin(margins), margins.shape)
        result["links"][name] = {
            "minimum_sphere_surface_margin_m": float(margins[frame, sphere]),
            "trace_index": indices[frame], "stage": trace[indices[frame]]["stage"],
            "sphere_index": int(sphere), "sphere_center_b": centers[frame, sphere].tolist(),
            "sphere_radius_m": float(radii[sphere]), "tray_bounds_b": mesh.bounds.tolist(),
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
