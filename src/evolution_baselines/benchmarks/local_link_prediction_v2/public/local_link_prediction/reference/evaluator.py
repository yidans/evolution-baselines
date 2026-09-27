from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import networkx as nx
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


Edge = Tuple[int, int]


@dataclass
class SplitResult:
    graph: str
    split_id: int
    num_nodes: int
    num_edges_original: int
    num_edges_train: int
    num_hidden_edges: int
    num_candidates: int
    average_precision: float
    roc_auc: float
    hits_at_l: float
    seconds_scoring: float


@dataclass
class PreparedSplit:
    graph_name: str
    split_id: int
    num_nodes: int
    num_edges_original: int
    num_edges_train: int
    adj: List[Set[int]]
    candidates: np.ndarray
    labels: np.ndarray


def _canonical_edge(u: int, v: int) -> Edge:
    return (u, v) if u < v else (v, u)


def load_edgelist(path: Path) -> nx.Graph:
    graph = nx.Graph()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid edge list line: {raw_line!r}")
            u, v = parts[0], parts[1]
            graph.add_edge(u, v)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    if graph.number_of_edges() == 0:
        raise ValueError(f"Graph {path} has no valid edges.")
    graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
    return graph


def load_graph(path: Path) -> nx.Graph:
    if path.suffix == ".gml":
        graph = nx.read_gml(path)
        graph = nx.Graph(graph)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        graph = nx.convert_node_labels_to_integers(graph, ordering="sorted")
        return graph
    return load_edgelist(path)


def adjacency_from_graph(graph: nx.Graph) -> List[Set[int]]:
    return [set(graph.neighbors(u)) for u in range(graph.number_of_nodes())]


def candidate_pairs_distance2(adj: Sequence[Set[int]]) -> np.ndarray:
    candidates = set()
    for w, nbrs in enumerate(adj):
        nbrs = sorted(nbrs)
        for i in range(len(nbrs)):
            u = nbrs[i]
            for j in range(i + 1, len(nbrs)):
                v = nbrs[j]
                if v not in adj[u]:
                    candidates.add((u, v) if u < v else (v, u))
    if not candidates:
        return np.zeros((0, 2), dtype=int)
    return np.asarray(sorted(candidates), dtype=int)


def eligible_removals(graph: nx.Graph) -> List[Edge]:
    bridges = {frozenset(edge) for edge in nx.bridges(graph)}
    eligible = []
    for u, v in graph.edges():
        if frozenset((u, v)) in bridges:
            continue
        if graph.degree(u) <= 1 or graph.degree(v) <= 1:
            continue
        if len(set(nx.common_neighbors(graph, u, v))) == 0:
            continue
        eligible.append(_canonical_edge(u, v))
    return eligible


def sample_hidden_edges(
    graph: nx.Graph,
    test_fraction: float,
    rng: np.random.Generator,
    min_hidden_edges: int = 1,
    max_attempts: int = 50,
) -> Tuple[nx.Graph, List[Edge]]:
    target = max(min_hidden_edges, int(round(test_fraction * graph.number_of_edges())))
    best_train = None
    best_hidden: List[Edge] = []

    for _ in range(max_attempts):
        train = graph.copy()
        hidden: List[Edge] = []

        while len(hidden) < target:
            eligible = eligible_removals(train)
            if not eligible:
                break
            idx = int(rng.integers(0, len(eligible)))
            u, v = eligible[idx]
            train.remove_edge(u, v)
            hidden.append((u, v))

        final_hidden = []
        for u, v in hidden:
            if train.has_edge(u, v):
                continue
            if len(set(nx.common_neighbors(train, u, v))) > 0:
                final_hidden.append((u, v))

        if len(final_hidden) > len(best_hidden):
            best_train = train.copy()
            best_hidden = final_hidden

        if len(final_hidden) >= min_hidden_edges:
            return train, final_hidden

    if best_train is None or len(best_hidden) < min_hidden_edges:
        raise RuntimeError(
            "Could not construct a valid split with hidden distance-2 positives. "
            "Use a graph with more triangle-rich structure or reduce test_fraction."
        )
    return best_train, best_hidden


def load_solution(path: Path):
    spec = importlib.util.spec_from_file_location("rsi_solution", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import solution from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "score_candidates"):
        raise AttributeError(f"Solution file {path} does not define score_candidates(adj, candidates)")
    return module.score_candidates


def compute_hits_at_l(labels: np.ndarray, scores: np.ndarray, l_value: int) -> float:
    if l_value <= 0:
        return float("nan")
    order = np.argsort(-scores)
    top = order[:l_value]
    return float(np.mean(labels[top]))


def prepare_split(
    graph_name: str,
    graph: nx.Graph,
    split_id: int,
    test_fraction: float,
    rng: np.random.Generator,
    candidate_subsetter=None,
) -> PreparedSplit:
    train_graph, hidden_edges = sample_hidden_edges(graph, test_fraction=test_fraction, rng=rng)
    adj = adjacency_from_graph(train_graph)
    candidates = candidate_pairs_distance2(adj)
    if candidate_subsetter is not None:
        candidates = np.asarray(candidate_subsetter(adj, candidates), dtype=int)
    hidden_set = set(hidden_edges)
    labels = np.asarray([1 if tuple(pair) in hidden_set else 0 for pair in candidates], dtype=int)

    if labels.sum() == 0:
        raise RuntimeError("No positive candidates remained after split construction.")

    return PreparedSplit(
        graph_name=graph_name,
        split_id=split_id,
        num_nodes=graph.number_of_nodes(),
        num_edges_original=graph.number_of_edges(),
        num_edges_train=train_graph.number_of_edges(),
        adj=adj,
        candidates=candidates,
        labels=labels,
    )


def evaluate_prepared_split(prepared: PreparedSplit, score_fn) -> SplitResult:
    start = time.perf_counter()
    raw_scores = score_fn(prepared.adj, prepared.candidates)
    elapsed = time.perf_counter() - start
    scores = np.asarray(raw_scores, dtype=float)

    if scores.shape != (len(prepared.candidates),):
        raise ValueError(
            f"score_candidates returned shape {scores.shape}; expected {(len(prepared.candidates),)}"
        )
    if not np.isfinite(scores).all():
        raise ValueError("score_candidates returned NaN or infinite values.")

    ap = float(average_precision_score(prepared.labels, scores))
    auc = float(roc_auc_score(prepared.labels, scores)) if len(np.unique(prepared.labels)) == 2 else float("nan")
    hits_at_l = compute_hits_at_l(prepared.labels, scores, int(prepared.labels.sum()))

    return SplitResult(
        graph=prepared.graph_name,
        split_id=prepared.split_id,
        num_nodes=prepared.num_nodes,
        num_edges_original=prepared.num_edges_original,
        num_edges_train=prepared.num_edges_train,
        num_hidden_edges=int(prepared.labels.sum()),
        num_candidates=len(prepared.candidates),
        average_precision=ap,
        roc_auc=auc,
        hits_at_l=hits_at_l,
        seconds_scoring=elapsed,
    )


def evaluate_single_split(
    graph_name: str,
    graph: nx.Graph,
    score_fn,
    split_id: int,
    test_fraction: float,
    rng: np.random.Generator,
    candidate_subsetter=None,
) -> SplitResult:
    prepared = prepare_split(
        graph_name=graph_name,
        graph=graph,
        split_id=split_id,
        test_fraction=test_fraction,
        rng=rng,
        candidate_subsetter=candidate_subsetter,
    )
    return evaluate_prepared_split(prepared, score_fn)


def summarize(results: List[SplitResult]) -> Dict[str, object]:
    return {
        "num_splits": len(results),
        "mean_average_precision": float(np.mean([r.average_precision for r in results])),
        "mean_roc_auc": float(np.nanmean([r.roc_auc for r in results])),
        "mean_hits_at_l": float(np.nanmean([r.hits_at_l for r in results])),
        "mean_seconds_scoring": float(np.mean([r.seconds_scoring for r in results])),
        "results": [asdict(r) for r in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an RSI-discovered local link prediction algorithm.")
    parser.add_argument("--solution", type=Path, required=True, help="Python file defining score_candidates(adj, candidates)")
    parser.add_argument("--edgelist", type=Path, nargs="+", required=True, help="One or more edge-list files")
    parser.add_argument("--splits", type=int, default=5, help="Number of random splits per graph")
    parser.add_argument("--test-fraction", type=float, default=0.1, help="Target fraction of edges to hide")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    score_fn = load_solution(args.solution)
    rng = np.random.default_rng(args.seed)

    results: List[SplitResult] = []
    for edgelist_path in args.edgelist:
        graph = load_graph(edgelist_path)
        for split_id in range(args.splits):
            split_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
            result = evaluate_single_split(
                graph_name=edgelist_path.name,
                graph=graph,
                score_fn=score_fn,
                split_id=split_id,
                test_fraction=args.test_fraction,
                rng=split_rng,
            )
            results.append(result)

    print(json.dumps(summarize(results), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
