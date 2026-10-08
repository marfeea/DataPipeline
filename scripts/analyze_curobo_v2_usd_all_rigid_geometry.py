"""Audit all CuRobo nonignored robot pairs against production USD collision meshes."""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from analyze_curobo_v2_usd_rigid_geometry import convex_hulls_intersect, world_vertices


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry',type=Path,required=True)
    parser.add_argument('--branches',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=Path('configs/curobo/assets/aubo_e5_project/collision_model_v2.json'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report=json.loads(args.geometry.read_text(encoding='utf-8'))
    branches=json.loads(args.branches.read_text(encoding='utf-8'))
    model=json.loads(args.model.read_text(encoding='utf-8'))
    if Path(report['branches']).resolve()!=args.branches.resolve():
        raise ValueError('geometry and branch evidence mismatch')
    names=tuple(report['body_names'])
    expected=('Base_Link','Link_01','Link_02','Link_03','Link_04','Link_05','Flange',
              'M_ClawTool_C1','M_ClawTool_C2','M_ClawTool_C3')
    if names!=expected or not all(report['geometry'][name]['collision_enabled'] for name in names):
        raise ValueError('unexpected USD collision geometry')
    curobo_names={name:('AUBO_E5_Flange' if name=='Flange' else 'gripper_base' if name.startswith('M_') else name)
                  for name in names}
    ignored={frozenset((name,other)) for name,others in model['self_collision_ignore'].items() for other in others}
    pairs=[(i,j) for i,j in combinations(range(len(names)),2)
           if curobo_names[names[i]]!=curobo_names[names[j]]
           and frozenset((curobo_names[names[i]],curobo_names[names[j]])) not in ignored]
    arrays=np.load(report['npz'],allow_pickle=False)
    candidates=[]
    for pose,branch in zip(report['poses'],branches['candidates'],strict=True):
        if pose['max_abs_joint_error_rad']>1e-5:
            raise ValueError('Isaac pose does not match saved IK branch')
        vertices=[world_vertices(arrays[name+'_vertices'],pose['body_positions_w'][i],
                                 pose['body_quaternions_wxyz'][i]) for i,name in enumerate(names)]
        problem=[]
        for i,j in pairs:
            left,right=vertices[i],vertices[j]
            if np.any(left.max(axis=0)<right.min(axis=0)) or np.any(right.max(axis=0)<left.min(axis=0)):
                continue
            if not convex_hulls_intersect(left,right):
                continue
            inside_count=0
            if 'M_ClawTool_C1' in (names[i],names[j]):
                other=right if names[i]=='M_ClawTool_C1' else left
                claw=left if names[i]=='M_ClawTool_C1' else right
                hull=ConvexHull(claw)
                signed=other@hull.equations[:,:3].T+hull.equations[:,3]
                inside_count=int(np.count_nonzero(np.max(signed,axis=1)<-1e-6))
            problem.append({'bodies':[names[i],names[j]],
                            'classification':'collision' if inside_count>0 else 'indeterminate',
                            'surface_vertices_inside_c1_hull':inside_count})
        state=('collision' if any(item['classification']=='collision' for item in problem) else
               'separated' if not problem else 'indeterminate')
        candidates.append({'candidate':pose['candidate'],'classification':state,
                           'curobo_self_score':branch['self_collision_score'],
                           'curobo_world_score':branch['world_collision_score'],
                           'unresolved_or_colliding_pairs':problem})
    result={'schema':'curobo_v2_usd_all_rigid_geometry/1',
            'scope':'all CuRobo nonignored pairs among ten production USD robot rigid bodies at saved IK endpoints',
            'limits':'convex hull separation proves no mesh contact; C1 convexHull vertex penetration proves contact; other hull overlaps remain indeterminate',
            'geometry':str(args.geometry.resolve()),'model':str(args.model.resolve()),
            'checked_body_pairs':[[names[i],names[j]] for i,j in pairs],
            'candidates':candidates}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps([{'candidate':x['candidate'],'classification':x['classification'],
                       'pairs':x['unresolved_or_colliding_pairs']} for x in candidates],indent=2))


if __name__=='__main__':
    main()
