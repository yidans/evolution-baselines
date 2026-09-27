"""Reference CLI shell for an external score_candidates(adj, candidates) module."""
import argparse
import json
from pathlib import Path

import numpy as np


def solve(score_candidates):
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    instance = json.loads(args.instance.read_text())
    adj = [set() for _ in range(instance['node_count'])]
    for u, v in instance['train_edges']:
        adj[u].add(v)
        adj[v].add(u)
    candidates = np.asarray(instance['candidates'], dtype=int).reshape((-1, 2))
    scores = np.asarray(score_candidates(adj, candidates), dtype=float)
    if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
        raise ValueError('score_candidates must return one finite number per candidate, in order')
    args.output.write_text(json.dumps({
        'schema_version': 'local-link-prediction-solution.v1', 'scores': scores.tolist(),
    }, allow_nan=False) + '\n')
