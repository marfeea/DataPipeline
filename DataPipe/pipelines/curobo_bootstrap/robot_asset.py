"""Traceable planning asset derivation; never edit the original URDF."""

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .robot_adapter import UrdfChain, pose_matrix


def derive_robot_asset(source, destination, audit_report, *, tcp_position, tcp_quaternion_wxyz):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    content = source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != audit_report["urdf_sha256"]:
        raise ValueError("audit evidence does not match source URDF bytes")
    if destination.exists():
        raise FileExistsError(destination)
    chain = UrdfChain(source, "AUBO_E5", "AUBO_E5_Flange")
    actual = audit_report["isaac_contract"]
    if tuple(actual["joint_names"]) != chain.joint_names or len(chain.joint_names) != 6:
        raise ValueError("joint names/order must agree before derivation")
    measured = np.asarray(actual["velocity"], dtype=float)
    if measured.shape != (6,):
        raise ValueError("six measured velocity limits required")
    velocity = np.minimum(chain.limits["velocity"], measured)
    if not np.isfinite(velocity).all() or np.any(velocity <= 0):
        raise ValueError("positive finite source and measured velocity limits required")
    expected = pose_matrix(tcp_position, tcp_quaternion_wxyz)
    if not np.allclose(expected[:3, :3], [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], atol=1e-10, rtol=0):
        raise ValueError("recipe requires the explicit project +pi/2 F-y TCP rotation")
    root = ET.fromstring(content)
    if any(link.attrib["name"] == "project_tcp" for link in root.findall("link")):
        raise ValueError("project_tcp already exists")
    changes = []
    for joint in root.findall("joint"):
        if joint.attrib["name"] in chain.joint_names:
            index = chain.joint_names.index(joint.attrib["name"])
            limit = joint.find("limit")
            value = float(velocity[index])
            changes.append(
                {
                    "joint": joint.attrib["name"],
                    "old_velocity": float(limit.attrib["velocity"]),
                    "isaac_velocity": float(measured[index]),
                    "effective_velocity": value,
                }
            )
            limit.set("velocity", format(value, ".17g"))
    meshes = []
    for mesh in root.findall(".//mesh"):
        original = mesh.attrib["filename"]
        path = (source.parent / original).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set("filename", Path(os.path.relpath(path, destination)).as_posix())
        meshes.append(
            {"source_reference": original, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    ET.SubElement(root, "link", name="project_tcp")
    joint = ET.SubElement(root, "joint", name="flange_to_project_tcp", type="fixed")
    ET.SubElement(joint, "parent", link="AUBO_E5_Flange")
    ET.SubElement(joint, "child", link="project_tcp")
    ET.SubElement(joint, "origin", xyz=" ".join(format(v, ".17g") for v in tcp_position), rpy=f"0 {np.pi / 2:.17g} 0")
    ET.indent(root)
    result = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    manifest = {
        "recipe_version": 1,
        "source": str(source),
        "source_sha256": digest,
        "derived_sha256": hashlib.sha256(result).hexdigest(),
        "ee_link": "project_tcp",
        "velocity_policy": "min(source_urdf, measured_isaac)",
        "velocity_changes": changes,
        "flange_to_project_tcp": expected.tolist(),
        "original_tool_center_point_preserved": True,
        "meshes": meshes,
        "scope": "kinematics only; collision spheres not validated",
    }
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "aubo_e5_project.urdf").write_bytes(result)
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
