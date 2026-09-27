from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


CandidateArray = np.ndarray


def _common_neighbors(adj: Sequence[Set[int]], u: int, v: int) -> Set[int]:
    return adj[u] & adj[v]


def adamic_adar(adj: Sequence[Set[int]], candidates: CandidateArray) -> np.ndarray:
    scores = []
    for u, v in candidates:
        score = 0.0
        for w in _common_neighbors(adj, u, v):
            deg = len(adj[w])
            if deg > 1:
                score += 1.0 / math.log(deg)
        scores.append(score)
    return np.asarray(scores, dtype=float)

score_candidates = adamic_adar
