"""Independent URDF FK and strict alignment audit; no legacy IK imports."""

import hashlib
import xml.etree.ElementTree as ET
from itertools import combinations
from pathlib import Path

import numpy as np


def rigid_transform(value):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("expected finite 4x4 rigid transform")
    rotation = matrix[:3, :3]
    if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-7, rtol=0):
        raise ValueError("invalid homogeneous bottom row")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError("rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0):
        raise ValueError("rotation must have determinant +1")
    return matrix.copy()


def pose_matrix(position, quaternion_wxyz):
    p = np.asarray(position, dtype=np.float64)
    q = np.asarray(quaternion_wxyz, dtype=np.float64)
    if p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("invalid xyz/wxyz pose")
    if not np.isclose(q @ q, 1, atol=1e-6, rtol=0):
        raise ValueError("unit quaternion required; no implicit normalization")
    w, x, y, z = q
    result = np.eye(4)
    result[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    result[:3, 3] = p
    return rigid_transform(result)


def pose_error(reference, actual):
    reference, actual = rigid_transform(reference), rigid_transform(actual)
    relative = reference[:3, :3].T @ actual[:3, :3]
    angle = np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))
    return float(np.linalg.norm(reference[:3, 3] - actual[:3, 3])), float(angle)


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    if axis.shape != (3,) or not np.isfinite(axis).all() or not np.isclose(axis @ axis, 1, atol=1e-7):
        raise ValueError("URDF axis must be unit length")
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


class UrdfChain:
    """Read-only serial-chain FK for fixed/revolute URDF joints in metres/radians.

    No joint renaming, limit override, meshes, collisions or planning capability.
    """

    def __init__(self, path, base_link, end_link):
        self.path = Path(path).resolve()
        content = self.path.read_bytes()
        self.sha256 = hashlib.sha256(content).hexdigest()
        root = ET.fromstring(content)
        links = {link.attrib["name"] for link in root.findall("link")}
        if base_link not in links or end_link not in links:
            raise ValueError("requested URDF frame missing")
        by_child = {}
        for joint in root.findall("joint"):
            child = joint.find("child").attrib["link"]
            if child in by_child:
                raise ValueError("URDF child has multiple parent joints")
            by_child[child] = joint
        chain, visited = [], set()
        link = end_link
        while link != base_link:
            if link in visited or link not in by_child:
                raise ValueError("URDF chain is disconnected or cyclic")
            visited.add(link)
            joint = by_child[link]
            if joint.attrib["type"] not in {"fixed", "revolute"} or joint.find("mimic") is not None:
                raise ValueError("only independent fixed/revolute chains are supported")
            chain.append(joint)
            link = joint.find("parent").attrib["link"]
        self.chain = tuple(reversed(chain))
        active = [j for j in self.chain if j.attrib["type"] == "revolute"]
        self.joint_names = tuple(j.attrib["name"] for j in active)
        if len(set(self.joint_names)) != len(active):
            raise ValueError("duplicate active joint names")
        self.limits = {
            key: tuple(float(j.find("limit").attrib[key]) for j in active) for key in ("lower", "upper", "velocity")
        }
        if not all(np.isfinite(v).all() for v in self.limits.values()):
            raise ValueError("nonfinite URDF limits")
        if any(a >= b for a, b in zip(self.limits["lower"], self.limits["upper"])):
            raise ValueError("invalid URDF position limits")

    def fk(self, q):
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (len(self.joint_names),) or not np.isfinite(q).all():
            raise ValueError("FK joint vector shape/value mismatch")
        result, index = np.eye(4), 0
        for joint in self.chain:
            origin = joint.find("origin")
            attrs = {} if origin is None else origin.attrib
            xyz = np.fromstring(attrs.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(attrs.get("rpy", "0 0 0"), sep=" ")
            if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite(rpy).all():
                raise ValueError("invalid URDF origin")
            transform = np.eye(4)
            transform[:3, 3] = xyz
            transform[:3, :3] = (
                _rotation((0, 0, 1), rpy[2]) @ _rotation((0, 1, 0), rpy[1]) @ _rotation((1, 0, 0), rpy[0])
            )
            result = result @ transform
            if joint.attrib["type"] == "revolute":
                axis = joint.find("axis")
                values = np.fromstring("1 0 0" if axis is None else axis.attrib["xyz"], sep=" ")
                motion = np.eye(4)
                motion[:3, :3] = _rotation(values, q[index])
                result = result @ motion
                index += 1
        return rigid_transform(result)


def audit_fk(samples, backends, *, position_tolerance_m, rotation_tolerance_rad):
    """Evaluate named FK callbacks on identical inputs; missing CuRobo fails gate.

    All callbacks must return Base-to-project-TCP matrices, never flange poses.
    Calibration can include URDF evidence, but URDF never substitutes for curobo.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 6 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("nonempty [N,6] FK sample set required")
    for tolerance in (position_tolerance_m, rotation_tolerance_rad):
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("finite nonnegative FK tolerance required")
    missing = sorted({"current", "isaac", "curobo"} - backends.keys())
    poses = {name: [rigid_transform(fn(q.copy())) for q in samples] for name, fn in backends.items()}
    comparisons = {}
    for left, right in combinations(poses, 2):
        errors = [pose_error(a, b) for a, b in zip(poses[left], poses[right])]
        position, rotation = np.max(errors, axis=0)
        comparisons[f"{left}:{right}"] = {
            "position_max_m": float(position),
            "rotation_max_rad": float(rotation),
            "passed": bool(position <= position_tolerance_m and rotation <= rotation_tolerance_rad),
        }
    return {
        "passed": not missing and all(row["passed"] for row in comparisons.values()),
        "missing_backends": missing,
        "sample_count": len(samples),
        "comparisons": comparisons,
    }


def compare_joint_contract(joint_names, lower, upper, velocity, other, *, atol=1e-6):
    """Return explicit mismatches; never silently reorder or override limits."""
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("invalid comparison tolerance")
    issues = []
    if tuple(joint_names) != tuple(other["joint_names"]):
        issues.append("joint_names_or_order")
    for name, values in (("lower", lower), ("upper", upper), ("velocity", velocity)):
        a, b = np.asarray(values, dtype=float), np.asarray(other[name], dtype=float)
        if a.shape != (6,) or b.shape != (6,) or not np.isfinite(a).all() or not np.isfinite(b).all():
            issues.append(f"{name}_shape_or_nonfinite")
        elif not np.allclose(a, b, rtol=0, atol=atol):
            issues.append(f"{name}_mismatch")
        if name == "velocity" and (np.any(a <= 0) or np.any(b <= 0)):
            issues.append("velocity_not_positive")
    return issues
