"""Retired trial: PhysX contact/query positive controls were inconclusive; use USD geometry audit."""
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from DataPipe.runtime import ensure_curriculumrl_available
ensure_curriculumrl_available()
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--branches', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
launcher = AppLauncher(args)

import numpy as np
import torch
from pxr import UsdPhysics, PhysxSchema, PhysicsSchemaTools, Sdf, Gf
import carb
from omni.physx import get_physx_simulation_interface, get_physx_scene_query_interface
from isaaclab import sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim import SimulationContext
from CurriculumRL.configs.assets import ROBOT_PRIM_CONTRACT
from CurriculumRL.tasks.tcp_docking.trajectory_env_cfg import TcpDockingTrajectoryEnvCfg
from DataPipe.config import GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POSITION


def main():
    raise RuntimeError("Retired trial: PhysX contact/query positive controls failed; use export_curobo_v2_rigid_geometry_isaac.py")
    branch = json.loads(args.branches.read_text(encoding='utf-8'))
    source = TcpDockingTrajectoryEnvCfg().scene
    cfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=False)
    robot_usd = deepcopy(source.AUBObot_usd)
    robot = deepcopy(source.AUBObot)
    robot_usd.prim_path = '{ENV_REGEX_NS}/AUBObot'
    robot.prim_path = f'{{ENV_REGEX_NS}}/AUBObot/{ROBOT_PRIM_CONTRACT.articulation_prim}'
    robot_usd.init_state.pos = (0.0, 0.0, 0.0)
    robot_usd.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    robot.init_state.pos = (0.0, 0.0, 0.0)
    robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    robot_usd.spawn.articulation_props.enabled_self_collisions = True
    robot_usd.spawn.activate_contact_sensors = True
    first_q = branch['candidates'][0]['joint_position']
    joint_pos = dict(robot.init_state.joint_pos)
    joint_pos.update(zip(ROBOT_PRIM_CONTRACT.arm_joints, first_q, strict=True))
    joint_pos.update(zip(GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POSITION, strict=True))
    robot.init_state.joint_pos = joint_pos
    setattr(cfg, 'AUBObot_usd', robot_usd)
    setattr(cfg, 'AUBObot', robot)
    setattr(cfg, 'contact_control', AssetBaseCfg(
        prim_path='{ENV_REGEX_NS}/contact_control',
        spawn=sim_utils.CuboidCfg(size=(0.6,0.6,0.6),collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0,0.0,0.15)),
    ))
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0/120.0, device=args.device))
    scene = InteractiveScene(cfg)
    print("audit: sim.reset", flush=True)
    sim.reset()
    scene.reset()
    scene.update(sim.get_physics_dt())
    body = scene.articulations['AUBObot']
    stage = scene.stage
    prefix = f'{scene.env_prim_paths[0]}/AUBObot/AUBO_E5'
    rigids = [str(p.GetPath()) for p in stage.Traverse() if str(p.GetPath()).startswith(prefix+'/') and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    contacts = [str(p.GetPath()) for p in stage.Traverse() if str(p.GetPath()).startswith(prefix+'/') and p.HasAPI(PhysxSchema.PhysxContactReportAPI)]
    collision_prims = [str(p.GetPath()) for p in stage.Traverse() if str(p.GetPath()).startswith(prefix+'/') and p.HasAPI(UsdPhysics.CollisionAPI)]
    root_prim = stage.GetPrimAtPath(prefix)
    enabled = PhysxSchema.PhysxArticulationAPI(root_prim).GetEnabledSelfCollisionsAttr().Get()
    report = {'schema':'curobo_v2_rigid_self_collision_audit/1', 'branches':str(args.branches.resolve()),
        'self_collisions_enabled':enabled, 'rigid_bodies':rigids, 'contact_report_bodies':contacts,
        'collision_prims':collision_prims, 'candidates':[]}
    if not enabled:
        raise RuntimeError('USD articulation self collisions did not enable')
    link_path = prefix + '/Link_02'
    gripper_paths = sorted(path for path in rigids if '/ClawTool_C/' in path)
    if link_path not in rigids or len(gripper_paths) != 3 or not set(gripper_paths).issubset(contacts):
        raise RuntimeError('Link_02 or gripper rigid-body contact reporters are missing')
    report['contact_filter_bodies'] = gripper_paths
    events = []
    def on_contact(headers, data):
        for header in headers:
            actor0 = str(PhysicsSchemaTools.intToSdfPath(header.actor0))
            actor1 = str(PhysicsSchemaTools.intToSdfPath(header.actor1))
            points = []
            for i in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data):
                point = data[i]
                points.append({'separation_m':float(point.separation), 'impulse_Ns':float(point.impulse)})
            events.append({'actor0':actor0, 'actor1':actor1, 'type':int(header.type), 'points':points})
    subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    print('audit: contact callback ready', flush=True)
    arm_ids, arm_names = body.find_joints(list(ROBOT_PRIM_CONTRACT.arm_joints), preserve_order=True)
    finger_ids, finger_names = body.find_joints(list(GRIPPER_JOINT_NAMES), preserve_order=True)
    if tuple(arm_names) != ROBOT_PRIM_CONTRACT.arm_joints or tuple(finger_names) != GRIPPER_JOINT_NAMES:
        raise RuntimeError('Isaac joint names/order differs from saved poses')
    link_shape = link_path + '/Collisions_02'
    def overlap_hits(shape_path):
        encoded = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(shape_path))
        hits = []
        def on_hit(hit):
            hits.append({'rigid_body':str(hit.rigid_body),'collider':str(hit.collider)})
            return True
        get_physx_scene_query_interface().overlap_shape(encoded[0],encoded[1],on_hit,False)
        return hits
    broad_hits=[]
    def on_broad_hit(hit):
        broad_hits.append(str(hit.rigid_body))
        return True
    get_physx_scene_query_interface().overlap_box(Gf.Vec3f(2.0,2.0,2.0), Gf.Vec3f(0.0,0.0,0.15), carb.Float4(0.0,0.0,0.0,1.0), on_broad_hit, False)
    print('audit: broad control hits', broad_hits, flush=True)
    report['broad_control_hits']=broad_hits
    for index, candidate in enumerate(branch['candidates']):
        positions = body.data.joint_pos.clone()
        velocities = torch.zeros_like(body.data.joint_vel)
        positions[:, arm_ids] = torch.as_tensor(candidate['joint_position'], device=positions.device, dtype=positions.dtype)
        positions[:, finger_ids] = torch.as_tensor(GRIPPER_OPEN_POSITION, device=positions.device, dtype=positions.dtype)
        body.write_joint_state_to_sim(positions, velocities)
        body.set_joint_position_target(positions[:, arm_ids], joint_ids=arm_ids)
        body.set_joint_position_target(positions[:, finger_ids], joint_ids=finger_ids)
        scene.write_data_to_sim()
        sim.forward()
        direct_hits = overlap_hits(link_shape)
        events.clear()
        print("audit: candidate", index, flush=True)
        for step in range(4):
            sim.step(render=False)
            scene.update(sim.get_physics_dt())
        after_step_hits = overlap_hits(link_shape)
        pair_events = [event for event in events if link_path in (event['actor0'], event['actor1'])
            and (event['actor0'] in gripper_paths or event['actor1'] in gripper_paths)]
        actual = body.data.joint_pos[0, arm_ids].detach().cpu().numpy()
        result = {'candidate':index, 'curobo_self_score':candidate['self_collision_score'],
            'joint_position':candidate['joint_position'], 'actual_joint_position':actual.tolist(),
            'max_abs_joint_error_rad':float(np.max(np.abs(actual-np.asarray(candidate['joint_position'])))),
            'all_robot_contact_events':events.copy(),
            'link02_gripper_contact_events':pair_events,
            'link02_shape_overlap_hits':direct_hits, 'link02_after_step_hits':after_step_hits}
        report['candidates'].append(result)
        print(json.dumps({'candidate':index,'pair_events':len(pair_events),'all_events':len(events),
            'overlap_hits':[x['rigid_body'] for x in direct_hits],
            'joint_error_rad':result['max_abs_joint_error_rad']}),flush=True)
    link_id, _ = body.find_bodies(['Link_02'], preserve_order=True)
    control_position = body.data.body_pos_w[0, link_id[0]].detach().clone().view(1,3)
    scene.extras['contact_control'].set_world_poses(positions=control_position)
    scene.write_data_to_sim()
    events.clear()
    for step in range(4):
        sim.step(render=False)
        scene.update(sim.get_physics_dt())
    control_path = f'{scene.env_prim_paths[0]}/contact_control'
    report['positive_control'] = {'cube_path':control_path, 'cube_center_w':control_position[0].cpu().tolist(),
        'events':events.copy(), 'link02_cube_events':[event for event in events
            if link_path in (event['actor0'],event['actor1'])
            and control_path in (event['actor0'],event['actor1'])]}
    report['passed'] = len(report['positive_control']['link02_cube_events']) > 0
    print('audit: positive control', len(report['positive_control']['link02_cube_events']), flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')


try:
    main()
finally:
    launcher.app.close()











