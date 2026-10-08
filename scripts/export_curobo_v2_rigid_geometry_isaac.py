"""Export actual USD collision meshes and exact Isaac body poses for saved IK branches."""
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from DataPipe.runtime import ensure_curriculumrl_available
ensure_curriculumrl_available()
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--branches',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--focused',action='store_true',help='export only Link_02 and the three gripper bodies')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if args.output.exists() or args.output.with_suffix('.npz').exists():
    raise FileExistsError(args.output)
launcher=AppLauncher(args)
import numpy as np
import torch
from pxr import Gf,UsdGeom,UsdPhysics
from isaaclab import sim as sim_utils
from isaaclab.scene import InteractiveScene,InteractiveSceneCfg
from CurriculumRL.configs.assets import ROBOT_PRIM_CONTRACT
from CurriculumRL.tasks.tcp_docking.trajectory_env_cfg import TcpDockingTrajectoryEnvCfg
from DataPipe.config import GRIPPER_JOINT_NAMES,GRIPPER_OPEN_POSITION

def main():
    branches=json.loads(args.branches.read_text(encoding='utf-8'))
    source=TcpDockingTrajectoryEnvCfg().scene
    cfg=InteractiveSceneCfg(num_envs=1,env_spacing=4.0,replicate_physics=False)
    usd=deepcopy(source.AUBObot_usd)
    robot_cfg=deepcopy(source.AUBObot)
    usd.prim_path='{ENV_REGEX_NS}/AUBObot'
    robot_cfg.prim_path=f'{{ENV_REGEX_NS}}/AUBObot/{ROBOT_PRIM_CONTRACT.articulation_prim}'
    usd.init_state.pos=(0.0,0.0,0.0)
    usd.init_state.rot=(1.0,0.0,0.0,0.0)
    robot_cfg.init_state.pos=(0.0,0.0,0.0)
    robot_cfg.init_state.rot=(1.0,0.0,0.0,0.0)
    initial=dict(robot_cfg.init_state.joint_pos)
    initial.update(zip(ROBOT_PRIM_CONTRACT.arm_joints,branches['candidates'][0]['joint_position'],strict=True))
    initial.update(zip(GRIPPER_JOINT_NAMES,GRIPPER_OPEN_POSITION,strict=True))
    robot_cfg.init_state.joint_pos=initial
    setattr(cfg,'AUBObot_usd',usd)
    setattr(cfg,'AUBObot',robot_cfg)
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/120,device=args.device))
    scene=InteractiveScene(cfg)
    sim.reset()
    scene.reset()
    body=scene.articulations['AUBObot']
    arm_ids,arm_names=body.find_joints(list(ROBOT_PRIM_CONTRACT.arm_joints),preserve_order=True)
    finger_ids,finger_names=body.find_joints(list(GRIPPER_JOINT_NAMES),preserve_order=True)
    names=(('Link_02','M_ClawTool_C1','M_ClawTool_C2','M_ClawTool_C3') if args.focused else
           ('Base_Link','Link_01','Link_02','Link_03','Link_04','Link_05','Flange',
            'M_ClawTool_C1','M_ClawTool_C2','M_ClawTool_C3'))
    body_ids,actual_names=body.find_bodies(list(names),preserve_order=True)
    if tuple(arm_names)!=ROBOT_PRIM_CONTRACT.arm_joints or tuple(finger_names)!=GRIPPER_JOINT_NAMES or tuple(actual_names)!=names:
        raise RuntimeError('Isaac joint/body ordering mismatch')
    prefix=f'{scene.env_prim_paths[0]}/AUBObot/AUBO_E5'
    stage=scene.stage
    cache=UsdGeom.XformCache()
    arrays={}
    geometry={}
    for name in names:
        rigid_path=(prefix+'/ClawTool_C/ClawTool_C/'+name if name.startswith('M_') else prefix+'/'+name)
        rigid=stage.GetPrimAtPath(rigid_path)
        shapes=[p for p in stage.Traverse() if str(p.GetPath()).startswith(rigid_path+'/') and p.HasAPI(UsdPhysics.CollisionAPI)]
        if len(shapes)!=1:
            raise RuntimeError(f'{name}: expected one collision mesh, found {len(shapes)}')
        prim=shapes[0]
        mesh=UsdGeom.Mesh(prim)
        if not mesh:
            raise RuntimeError(f'{name}: collision shape is not a USD mesh')
        relative,_=cache.ComputeRelativeTransform(prim,rigid)
        vertices=np.asarray([tuple(relative.Transform(Gf.Vec3d(*point))) for point in mesh.GetPointsAttr().Get()],dtype=np.float64)
        counts=list(mesh.GetFaceVertexCountsAttr().Get())
        indices=list(mesh.GetFaceVertexIndicesAttr().Get())
        faces=[]
        offset=0
        for count in counts:
            polygon=indices[offset:offset+count]
            faces.extend((polygon[0],polygon[i],polygon[i+1]) for i in range(1,count-1))
            offset+=count
        arrays[name+'_vertices']=vertices
        arrays[name+'_faces']=np.asarray(faces,dtype=np.int32)
        mesh_api=UsdPhysics.MeshCollisionAPI(prim)
        geometry[name]={'body_path':rigid_path,'shape_path':str(prim.GetPath()),
            'approximation':mesh_api.GetApproximationAttr().Get() if mesh_api else None,
            'collision_enabled':UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
            'vertices':len(vertices),'triangles':len(faces)}
    poses=[]
    for index,candidate in enumerate(branches['candidates']):
        q=torch.as_tensor(candidate['joint_position'],dtype=body.data.joint_pos.dtype,device=body.data.joint_pos.device)
        positions=body.data.joint_pos.clone()
        positions[:,arm_ids]=q
        positions[:,finger_ids]=torch.as_tensor(GRIPPER_OPEN_POSITION,dtype=positions.dtype,device=positions.device)
        body.write_joint_state_to_sim(positions,torch.zeros_like(body.data.joint_vel))
        sim.forward()
        scene.update(sim.get_physics_dt())
        actual=body.data.joint_pos[0,arm_ids].detach().cpu().numpy()
        pos=body.data.body_pos_w[0,body_ids].detach().cpu().numpy()
        quat=body.data.body_quat_w[0,body_ids].detach().cpu().numpy()
        poses.append({'candidate':index,'max_abs_joint_error_rad':float(np.max(np.abs(actual-np.asarray(candidate['joint_position'])))),
            'body_positions_w':pos.tolist(),'body_quaternions_wxyz':quat.tolist()})
    report={'schema':'curobo_v2_usd_rigid_geometry/1','branches':str(args.branches.resolve()),
        'geometry':geometry,'body_names':names,'poses':poses,'npz':str(args.output.with_suffix('.npz').resolve())}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'),**arrays)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'geometry':geometry,'joint_errors':[x['max_abs_joint_error_rad'] for x in poses]},indent=2),flush=True)
try:
    main()
finally:
    launcher.app.close()


