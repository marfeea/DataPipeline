"""Print the CuRobo sphere pairs behind saved V2 branch self-collision scores."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.curobo.world import MeshConfig, WorldConfig
from DataPipe.pipelines.curobo_bootstrap.curobo_collision import CuroboCollision, self_collision_pairs
from DataPipe.pipelines.curobo_bootstrap.world_adapter import WorldSnapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world-report', type=Path, required=True)
    parser.add_argument('--branches', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=3)
    args = parser.parse_args()
    report = json.loads(args.world_report.read_text(encoding='utf-8'))
    branches = json.loads(args.branches.read_text(encoding='utf-8'))
    data = report['world']
    world = WorldSnapshot(WorldConfig((), data['target_name'], tuple(MeshConfig(**item) for item in data['meshes'])))
    if world.world_hash != report['world_hash'] or branches['world_hash'] != world.world_hash:
        raise ValueError('world hash mismatch')
    checker = CuroboCollision(report['collision_model'], world)
    candidates = branches['candidates']
    scores = checker.query(np.asarray([item['joint_position'] for item in candidates]))
    for index, item in enumerate(candidates):
        expected = item['self_collision_score']
        actual = float(scores['self'][index])
        if abs(expected - actual) > 1e-5:
            raise ValueError(f'candidate {index} score mismatch: {expected} vs {actual}')
        if actual > 0:
            print(json.dumps({'candidate': index, 'q_rad': item['joint_position'],
                'curobo_score': actual,
                'sphere_pairs': self_collision_pairs(checker.asset, scores['spheres'][index], limit=args.limit)},
                ensure_ascii=False))


if __name__ == '__main__':
    main()



