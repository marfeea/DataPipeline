"""Freeze enabled production gripper collision meshes in the planning link frame."""

import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

from pxr import Usd, UsdGeom, UsdPhysics

from configs.assets import ROBOT_PRIM_CONTRACT
from configs.curobo.robot import DEFAULT_COLLISION_MODEL
from scripts.pipelines.curobo_bootstrap.collision_model import load_collision_model
from scripts.pipelines.curobo_bootstrap.production_world import _triangles
from scripts.pipelines.curobo_bootstrap.robot_adapter import UrdfChain, pose_matrix


def export_gripper_geometry(env, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    robot = env.robot
    model = load_collision_model(DEFAULT_COLLISION_MODEL)
    arm_ids = robot.find_joints(ROBOT_PRIM_CONTRACT.arm_joints)[0]
    q = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy()
    root = pose_matrix(robot.data.root_pos_w[0].cpu().numpy(), robot.data.root_quat_w[0].cpu().numpy())
    chain = UrdfChain(model["urdf"], "AUBO_E5", "gripper_base")
    link_from_world = np.linalg.inv(root @ chain.fk(q))
    cache = UsdGeom.XformCache()
    records = []
    for prim in Usd.PrimRange.Stage(env.world.stage, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if "/AUBObot/" not in path or not any(n in path for n in ("ClawTool", "UpperFinger", "DownFinger")):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        mesh = UsdGeom.Mesh(prim)
        if not mesh:
            raise ValueError(f"Unsupported gripper collision primitive: {path}")
        body = prim
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        body_id = robot.find_bodies([body.GetName()])[0][0]
        measured_body = pose_matrix(robot.data.body_pos_w[0, body_id].cpu().numpy(),
                                    robot.data.body_quat_w[0, body_id].cpu().numpy())
        authored_body = np.asarray(cache.GetLocalToWorldTransform(body)).T
        transform = (link_from_world @ measured_body @ np.linalg.inv(authored_body)
                     @ np.asarray(cache.GetLocalToWorldTransform(prim)).T)
        vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
        geometry = trimesh.Trimesh(vertices, _triangles(mesh), process=False)
        destination = output / f"component_{len(records):02d}.obj"
        geometry.export(destination)
        records.append(dict(prim_path=path, mesh=str(destination.resolve()),
                            sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                            bounds=geometry.bounds.tolist(), vertices=len(vertices), faces=len(geometry.faces)))
    report = dict(frame="gripper_base", q=q.tolist(), components=records,
                  scope="Enabled production collision mesh geometry at measured initialized open-finger state")
    if not records:
        raise ValueError("No enabled production gripper collision meshes found")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
