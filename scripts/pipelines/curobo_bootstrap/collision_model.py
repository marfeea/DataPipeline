"""Traceable conservative collision spheres for the CuRobo robot model."""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CollisionSphere:
    center: tuple[float, float, float]
    radius: float

    def __post_init__(self):
        center = tuple(float(value) for value in self.center)
        if len(center) != 3 or not np.isfinite(center).all():
            raise ValueError("collision sphere center must contain three finite values")
        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("collision sphere radius must be finite and positive")
        object.__setattr__(self, "center", center)


def aabb_cover_spheres(minimum, maximum, *, max_cell_m=0.08, padding_m=0.002):
    """Cover an AABB with deterministic cell-circumscribing spheres."""
    low = np.asarray(minimum, dtype=np.float64)
    high = np.asarray(maximum, dtype=np.float64)
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite((low, high)).all():
        raise ValueError("finite 3D AABB bounds required")
    extent = high - low
    if np.any(extent <= 0) or not math.isfinite(max_cell_m) or max_cell_m <= 0:
        raise ValueError("positive AABB extents and max_cell_m required")
    if not math.isfinite(padding_m) or padding_m < 0:
        raise ValueError("nonnegative finite padding_m required")
    counts = np.maximum(1, np.ceil(extent / max_cell_m).astype(int))
    step = extent / counts
    radius = float(0.5 * np.linalg.norm(step) + padding_m)
    axes = [low[index] + (np.arange(counts[index]) + 0.5) * step[index] for index in range(3)]
    return tuple(CollisionSphere(tuple(float(value) for value in center), radius) for center in product(*axes))


def _rpy_matrix(values):
    roll, pitch, yaw = values
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _origin_matrix(node):
    result = np.eye(4, dtype=np.float64)
    if node is None:
        return result
    result[:3, 3] = [float(value) for value in node.attrib.get("xyz", "0 0 0").split()]
    result[:3, :3] = _rpy_matrix([float(value) for value in node.attrib.get("rpy", "0 0 0").split()])
    return result


def _transformed_bounds(minimum, maximum, transform):
    corners = np.array([(*corner, 1.0) for corner in product(*zip(minimum, maximum, strict=True))], dtype=np.float64)
    points = (transform @ corners.T).T[:, :3]
    return points.min(axis=0), points.max(axis=0)


def _collision_bounds(link, urdf_dir, mesh_evidence):
    collision = link.find("collision")
    if collision is None:
        return None
    geometry = collision.find("geometry")
    if geometry is None:
        raise ValueError(f"collision geometry missing for {link.attrib['name']}")
    transform = _origin_matrix(collision.find("origin"))
    mesh_node = geometry.find("mesh")
    box_node = geometry.find("box")
    if mesh_node is not None:
        import trimesh

        path = (urdf_dir / mesh_node.attrib["filename"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh = trimesh.load_mesh(path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if mesh.bounds.shape != (2, 3):
            raise ValueError(f"invalid mesh bounds for {path}")
        mesh_evidence.append(
            {"link": link.attrib["name"], "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
        return _transformed_bounds(mesh.bounds[0], mesh.bounds[1], transform)
    if box_node is not None:
        size = np.asarray([float(value) for value in box_node.attrib["size"].split()], dtype=np.float64)
        if size.shape != (3,) or np.any(size <= 0):
            raise ValueError(f"invalid box size for {link.attrib['name']}")
        return _transformed_bounds(-0.5 * size, 0.5 * size, transform)
    raise ValueError(f"unsupported collision geometry for {link.attrib['name']}")


def derive_collision_model(
    urdf,
    *,
    max_cell_m=0.08,
    padding_m=0.002,
    attachment_slots=4,
    gripper_mode="range_envelope",
    refined_flange_cell_m=None,
    gripper_cell_m=None,
):
    """Derive a JSON-serializable conservative model from the project URDF."""
    path = Path(urdf).resolve()
    content = path.read_bytes()
    root = ET.fromstring(content)
    links = {link.attrib["name"]: link for link in root.findall("link")}
    arm_links = ("Base_Link", "Link_01", "Link_02", "Link_03", "Link_04", "Link_05", "AUBO_E5_Flange")
    if any(name not in links for name in arm_links):
        raise ValueError("expected AUBO E5 collision links are missing")
    mesh_evidence = []
    bounds = {name: _collision_bounds(links[name], path.parent, mesh_evidence) for name in arm_links}

    if gripper_mode not in {"range_envelope", "fixed_open_components"}:
        raise ValueError("unsupported gripper_mode")
    if gripper_cell_m is not None and (not math.isfinite(gripper_cell_m) or gripper_cell_m <= 0):
        raise ValueError("gripper_cell_m must be finite and positive")
    gripper_base_bounds = _collision_bounds(links["gripper_base"], path.parent, mesh_evidence)
    gripper_bounds = [gripper_base_bounds]
    open_component_bounds = [gripper_base_bounds]
    for joint_name, link_name in (("UpperFinger", "UpperFinger"), ("DownFinger", "DownFinger")):
        joint = next((item for item in root.findall("joint") if item.attrib["name"] == joint_name), None)
        if joint is None or joint.find("parent").attrib["link"] != "gripper_base":
            raise ValueError(f"unexpected {joint_name} kinematic attachment")
        axis = np.asarray([float(value) for value in joint.find("axis").attrib["xyz"].split()])
        limit = joint.find("limit")
        local_low, local_high = _collision_bounds(links[link_name], path.parent, mesh_evidence)
        joint_transform = _origin_matrix(joint.find("origin"))
        lower, upper = float(limit.attrib["lower"]), float(limit.attrib["upper"])
        open_value = upper if joint_name == "UpperFinger" else lower
        for value in (lower, upper):
            transform = joint_transform.copy()
            transform[:3, 3] += axis * value
            gripper_bounds.append(_transformed_bounds(local_low, local_high, transform))
        open_transform = joint_transform.copy()
        open_transform[:3, 3] += axis * open_value
        open_component_bounds.append(_transformed_bounds(local_low, local_high, open_transform))
    bounds["gripper_base"] = (
        np.min([item[0] for item in gripper_bounds], axis=0),
        np.max([item[1] for item in gripper_bounds], axis=0),
    )

    spheres = {
        name: [
            {"center": list(sphere.center), "radius": sphere.radius}
            for sphere in aabb_cover_spheres(*bounds[name], max_cell_m=max_cell_m, padding_m=padding_m)
        ]
        for name in (*arm_links, "gripper_base")
    }
    if refined_flange_cell_m is not None:
        if not math.isfinite(refined_flange_cell_m) or refined_flange_cell_m <= 0:
            raise ValueError("refined_flange_cell_m must be finite and positive")
        spheres["AUBO_E5_Flange"] = [
            {"center": list(sphere.center), "radius": sphere.radius}
            for sphere in aabb_cover_spheres(
                *bounds["AUBO_E5_Flange"], max_cell_m=refined_flange_cell_m, padding_m=padding_m
            )
        ]
    if gripper_mode == "fixed_open_components":
        component_cell = gripper_cell_m or refined_flange_cell_m or max_cell_m
        component_spheres = []
        for low, high in open_component_bounds:
            component_spheres.extend(aabb_cover_spheres(low, high, max_cell_m=component_cell, padding_m=padding_m))
        spheres["gripper_base"] = [
            {"center": list(sphere.center), "radius": sphere.radius} for sphere in component_spheres
        ]
    collision_links = list(spheres)
    # AABB-cover spheres deliberately over-approximate each casting. Ignore only the local
    # kinematic neighbourhood whose conservative boxes overlap at normal joint assemblies.
    ignore = {
        name: collision_links[index + 1 : index + 3]
        for index, name in enumerate(collision_links)
        if index + 1 < len(collision_links)
    }
    ignore["Link_03"].append("AUBO_E5_Flange")
    ignore["attached_object"] = ["Link_05", "AUBO_E5_Flange", "gripper_base"]
    geometry_bounds = {
        name: {"minimum": low.tolist(), "maximum": high.tolist()} for name, (low, high) in bounds.items()
    }
    if attachment_slots < 1:
        raise ValueError("at least one attachment sphere slot is required")
    return {
        "format_version": 1,
        "method": "urdf_collision_component_aabb_grid_circumscribed_spheres",
        "coverage": "each local AABB cell is fully covered by its circumscribing sphere",
        "urdf": str(path),
        "urdf_sha256": hashlib.sha256(content).hexdigest(),
        "max_cell_m": float(max_cell_m),
        "padding_m": float(padding_m),
        "attachment_link": "attached_object",
        "attachment_parent": "project_tcp",
        "attachment_slots": int(attachment_slots),
        "collision_link_names": collision_links,
        "collision_spheres": spheres,
        "self_collision_ignore": ignore,
        "self_collision_ignore_policy": ("chain distance <= 2, plus audited Link_03/AUBO_E5_Flange assembly overlap"),
        "geometry_bounds": geometry_bounds,
        "mesh_evidence": mesh_evidence,
        "gripper_policy": (
            "fixed envelope over complete UpperFinger and DownFinger joint travel"
            if gripper_mode == "range_envelope"
            else "fixed open-position base and separate finger component AABBs; attachment covers carried object"
        ),
        "gripper_mode": gripper_mode,
        "refined_flange_cell_m": refined_flange_cell_m,
        "gripper_cell_m": gripper_cell_m,
    }


def write_collision_model(urdf, output, **kwargs):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    payload = derive_collision_model(urdf, **kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_collision_model(path, *, verify_sources=True):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1 or not payload.get("collision_link_names"):
        raise ValueError("unsupported or empty collision model")
    if verify_sources:
        urdf = Path(payload["urdf"])
        if not urdf.is_file() or hashlib.sha256(urdf.read_bytes()).hexdigest() != payload["urdf_sha256"]:
            raise ValueError("collision model URDF evidence no longer matches")
        for item in payload["mesh_evidence"]:
            source = Path(item["path"])
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"collision mesh evidence no longer matches: {source}")
    for name in payload["collision_link_names"]:
        if not payload["collision_spheres"].get(name):
            raise ValueError(f"collision link has no spheres: {name}")
        for sphere in payload["collision_spheres"][name]:
            CollisionSphere(tuple(sphere["center"]), sphere["radius"])
    return payload


def curobo_robot_dict(model, *, urdf=None):
    """Build the tested CuRobo 0.7.8 collision/FK input without importing CuRobo."""
    if isinstance(model, str | Path):
        model = load_collision_model(model)
    robot_urdf = Path(urdf or model["urdf"]).resolve()
    if hashlib.sha256(robot_urdf.read_bytes()).hexdigest() != model["urdf_sha256"]:
        raise ValueError("requested URDF does not match collision model")
    spheres = {
        name: [{"center": item["center"], "radius": item["radius"]} for item in values]
        for name, values in model["collision_spheres"].items()
    }
    return {
        "kinematics": {
            "urdf_path": str(robot_urdf),
            "asset_root_path": str(robot_urdf.parent),
            "base_link": "AUBO_E5",
            "ee_link": "project_tcp",
            "link_names": ["project_tcp"],
            "collision_link_names": [*model["collision_link_names"], model["attachment_link"]],
            "collision_spheres": spheres,
            "collision_sphere_buffer": 0.0,
            "extra_collision_spheres": {model["attachment_link"]: model["attachment_slots"]},
            "extra_links": {
                model["attachment_link"]: {
                    "parent_link_name": model["attachment_parent"],
                    "link_name": model["attachment_link"],
                    "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
                    "joint_type": "FIXED",
                    "joint_name": "attach_joint",
                }
            },
            "self_collision_ignore": model["self_collision_ignore"],
            "self_collision_buffer": {name: 0.0 for name in model["collision_link_names"]},
            "use_global_cumul": True,
        }
    }
