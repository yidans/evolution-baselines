from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


CandidateArray = np.ndarray


def _common_neighbors(adj: Sequence[Set[int]], u: int, v: int) -> Set[int]:
    return adj[u] & adj[v]


def common_neighbors(adj: Sequence[Set[int]], candidates: CandidateArray) -> np.ndarray:
    return np.asarray([len(_common_neighbors(adj, u, v)) for u, v in candidates], dtype=float)


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


def resource_allocation(adj: Sequence[Set[int]], candidates: CandidateArray) -> np.ndarray:
    scores = []
    for u, v in candidates:
        score = 0.0
        for w in _common_neighbors(adj, u, v):
            deg = len(adj[w])
            if deg > 0:
                score += 1.0 / deg
        scores.append(score)
    return np.asarray(scores, dtype=float)


def jaccard(adj: Sequence[Set[int]], candidates: CandidateArray) -> np.ndarray:
    scores = []
    for u, v in candidates:
        inter = len(adj[u] & adj[v])
        union = len(adj[u] | adj[v])
        scores.append(0.0 if union == 0 else inter / union)
    return np.asarray(scores, dtype=float)


def preferential_attachment(adj: Sequence[Set[int]], candidates: CandidateArray) -> np.ndarray:
    return np.asarray([len(adj[u]) * len(adj[v]) for u, v in candidates], dtype=float)


def car(adj: Sequence[Set[int]], candidates: CandidateArray) -> np.ndarray:
    scores = []
    for u, v in candidates:
        cn = _common_neighbors(adj, u, v)
        cn_count = len(cn)
        if cn_count == 0:
            scores.append(0.0)
            continue
        cn_list = list(cn)
        lcl = 0
        for i in range(len(cn_list)):
            a = cn_list[i]
            for j in range(i + 1, len(cn_list)):
                b = cn_list[j]
                if b in adj[a]:
                    lcl += 1
        scores.append(cn_count * lcl)
    return np.asarray(scores, dtype=float)


BASELINES: Dict[str, Callable[[Sequence[Set[int]], CandidateArray], np.ndarray]] = {
    "cn": common_neighbors,
    "aa": adamic_adar,
    "ra": resource_allocation,
    "jaccard": jaccard,
    "pa": preferential_attachment,
    "car": car,
}
