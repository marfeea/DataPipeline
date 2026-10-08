"""Classify Link_02/gripper overlaps from exported production USD collision meshes."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def world_vertices(local, position, quaternion_wxyz):
    w, x, y, z = quaternion_wxyz
    return Rotation.from_quat([x, y, z, w]).apply(local) + np.asarray(position)


def convex_hulls_intersect(a, b):
    va = a[ConvexHull(a).vertices]
    vb = b[ConvexHull(b).vertices]
    center = 0.5 * (va.mean(axis=0) + vb.mean(axis=0))
    va = (va - center) * 1000.0
    vb = (vb - center) * 1000.0
    eq = np.zeros((5, len(va) + len(vb)))
    eq[:3, :len(va)] = va.T
    eq[:3, len(va):] = -vb.T
    eq[3, :len(va)] = 1.0
    eq[4, len(va):] = 1.0
    result = linprog(np.zeros(eq.shape[1]), A_eq=eq, b_eq=(0, 0, 0, 1, 1), bounds=(0, None), method='highs')
    if result.status not in (0, 2):
        raise RuntimeError(f'convex separation solver failed: {result.message}')
    return result.success


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--branches', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.geometry.read_text(encoding='utf-8'))
    branches = json.loads(args.branches.read_text(encoding='utf-8'))
    if Path(report['branches']).resolve() != args.branches.resolve():
        raise ValueError('geometry and branches evidence mismatch')
    names = tuple(report['body_names'])
    if names != ('Link_02', 'M_ClawTool_C1', 'M_ClawTool_C2', 'M_ClawTool_C3'):
        raise ValueError('unexpected collision body ordering')
    if report['geometry']['M_ClawTool_C1']['approximation'] != 'convexHull' or not all(
        report['geometry'][name]['collision_enabled'] for name in names
    ):
        raise ValueError('USD collision geometry contract changed')
    arrays = np.load(report['npz'], allow_pickle=False)
    results = []
    for pose, branch in zip(report['poses'], branches['candidates'], strict=True):
        if pose['max_abs_joint_error_rad'] > 1e-5:
            raise ValueError('Isaac body pose differs from saved IK joint state')
        points = [world_vertices(arrays[name + '_vertices'], pose['body_positions_w'][i],
                                 pose['body_quaternions_wxyz'][i]) for i, name in enumerate(names)]
        link_points = points[0]
        pairs = []
        for name, gripper_points in zip(names[1:], points[1:], strict=True):
            hull_intersect = convex_hulls_intersect(link_points, gripper_points)
            inside_count = 0
            if name == 'M_ClawTool_C1' and hull_intersect:
                hull = ConvexHull(gripper_points)
                signed = link_points @ hull.equations[:, :3].T + hull.equations[:, 3]
                inside_count = int(np.count_nonzero(np.max(signed, axis=1) < -1e-6))
            classification = ('separated' if not hull_intersect else
                              'collision' if name == 'M_ClawTool_C1' and inside_count else 'indeterminate')
            pairs.append({'body': name, 'classification': classification,
                          'convex_hulls_intersect': hull_intersect,
                          'link02_vertices_inside_gripper_hull': inside_count})
        overall = ('collision' if any(item['classification'] == 'collision' for item in pairs) else
                   'separated' if all(item['classification'] == 'separated' for item in pairs) else 'indeterminate')
        results.append({'candidate': pose['candidate'], 'curobo_self_score': branch['self_collision_score'],
                        'curobo_world_score': branch['world_collision_score'],
                        'classification': overall, 'pairs': pairs})
    result = {'schema':'curobo_v2_usd_rigid_collision_analysis/1',
              'scope':'Link_02 against three gripper rigid collision bodies at saved pre-pick endpoints',
              'method':'convex hull disjointness proves separation; Link_02 surface vertices inside C1 PhysX convexHull prove penetration',
              'geometry':str(args.geometry.resolve()), 'candidates':results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps([{'candidate':x['candidate'],'classification':x['classification'],
                       'world_score':x['curobo_world_score']} for x in results],indent=2))


if __name__ == '__main__':
    main()

