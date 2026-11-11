"""Offline collision diagnosis of the saved, GUI-measured lift endpoints."""
import json
from pathlib import Path
import os
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))
runtime = Path(sys.executable).parent
os.environ["PATH"] = os.pathsep.join([str(runtime),str(runtime/"Library/bin"),os.environ.get("PATH","")])

import numpy as np
from configs.curobo.world import MeshConfig, WorldConfig
from scripts.pipelines.curobo_bootstrap.curobo_collision import CuroboCollision
from scripts.pipelines.curobo_bootstrap.world_adapter import WorldSnapshot, AttachedSphere


def main():
    root = PROJECT / "data/test/trajdata/curobo_lift_20261015/0_to_1"
    report = json.loads((root/"joint_time_1/report.json").read_text())
    frozen = json.loads((PROJECT/"data/0_to_1_1014_tcp/0_to_1_evidence/world/report.json").read_text())
    world = WorldSnapshot(WorldConfig((), frozen["world"]["target_name"], tuple(MeshConfig(**m) for m in frozen["world"]["meshes"])))
    meta = report["curobo_lift_experiment"]
    q = np.asarray([meta["start_q"], meta["endpoint_q"]])
    checker = CuroboCollision(frozen["collision_model"], world.permit_target_contact())
    result = dict(unattached=checker.query(q,world_ignore_links=("Base_Link",))["world"].tolist(), per_mesh={})
    sphere = AttachedSphere(**meta["attachment"])
    checker.attach(world.attach((sphere,)))
    for mesh in world.config.meshes:
        if mesh.name == world.config.target_name:
            continue
        subset = WorldSnapshot(WorldConfig((),world.config.target_name,tuple(m for m in world.config.meshes if m.name in (mesh.name,world.config.target_name)))).attach((sphere,))
        checker.update_world(subset)
        scores = checker.query(q,world_ignore_links=("Base_Link",))
        if np.max(scores["world"]) > 0:
            result["per_mesh"][mesh.name] = dict(world=scores["world"].tolist(),self=scores["self"].tolist())
    import trimesh
    from lift_experiment import attached_cover
    target = next(m for m in world.config.meshes if m.name==world.config.target_name)
    spheres = attached_cover(np.asarray(trimesh.load_mesh(target.file_path,process=False).vertices),
                             base_pose_w=report["base_pose_w"],object_pose_reset=report["target_pose_w_at_reset"],
                             object_pose_current=report["target_pose_w_at_pick"],tcp_pose_b=meta["start_tcp_pose_b"])
    refined = CuroboCollision(frozen["collision_model"],world.permit_target_contact())
    reduced = WorldSnapshot(WorldConfig((),world.config.target_name,tuple(m for m in world.config.meshes if m.name!="support_tray_11"))).attach(spheres)
    refined.attach(reduced)
    scores = refined.query(q,world_ignore_links=("Base_Link",))
    result["four_sphere_cover"] = dict(spheres=[dict(center_tcp=s.center_tcp,radius_m=s.radius_m) for s in spheres],
                                       excluded_proxy="support_tray_11",world=scores["world"].tolist(),self=scores["self"].tolist())
    (root/"endpoint_collision_diagnosis.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
