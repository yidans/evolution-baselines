from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np


CandidateArray = np.ndarray


def _common_neighbors(adj: Sequence[Set[int]], u: int, v: int) -> Set[int]:
    return adj[u] & adj[v]


def common_neighbors(adj: Sequence[Set[int]], candidates: CandidateArray) -> np.ndarray:
    return np.asarray([len(_common_neighbors(adj, u, v)) for u, v in candidates], dtype=float)

score_candidates = common_neighbors
