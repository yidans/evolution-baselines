# Problem specification: local topological link prediction

## Informal statement

You are given a simple, undirected, unweighted graph that is missing some edges. Your job is to invent a deterministic scoring rule that ranks candidate node pairs by how likely they are to be true but currently unobserved edges.

This benchmark only considers **candidate pairs at shortest-path distance 2** in the observed graph, i.e. pairs of nodes that are not adjacent but share at least one common neighbor. Intuitively, this is a **triangle-completion** benchmark.

## Exact input to your algorithm

Your algorithm must be a Python module exposing exactly this function:

```python
def score_candidates(adj, candidates):
    ...
```

### Input types

- `adj`: a Python list of Python sets.
  - `adj[u]` is the set of neighbors of node `u` in the observed graph.
  - Nodes are relabeled to consecutive integers `0, 1, ..., n-1`.
- `candidates`: a NumPy array of shape `(m, 2)` and integer dtype.
  - Each row is a candidate non-edge `(u, v)` with `u < v`.
  - Every candidate pair is guaranteed to have at least one common neighbor in the observed graph.

### Required output

- Return a 1D NumPy array, Python list, or other array-like object of length `m`.
- Entry `i` must be a finite numeric score for candidate pair `candidates[i]`.
- Higher score means "more likely to be a missing edge."

## Rules

Your algorithm:

- may use only the observed graph topology provided through `adj`;
- must be deterministic;
- must not use external training data, node attributes, text, labels, or internet access;
- should be reasonably efficient on medium-sized sparse graphs.

This benchmark is intended for **interpretable local-topology algorithms**. In spirit, you should rely on information in or near the neighborhoods of the candidate pair, though the evaluator does not enforce a formal locality constraint.

## Success criterion

The evaluator removes a subset of true edges from a graph, builds the candidate set from the remaining graph, and asks your algorithm to rank all candidate pairs.

Your algorithm is judged primarily by:

1. **Average Precision (AP)** on the candidate set.
2. **Hits@L**, where `L` is the number of hidden positive edges in that split.

Runtime is reported as a secondary metric.

## What counts as a good algorithm?

A good algorithm should:

- rank withheld true edges above non-edges;
- generalize across multiple graphs and random train/test splits;
- outperform a simple common-neighbors baseline;
- ideally outperform stronger public heuristics such as Adamic-Adar, Resource Allocation, Jaccard, Preferential Attachment, or CAR.

## Minimal baseline

A valid baseline is:

```python
def score_candidates(adj, candidates):
    return [len(adj[u] & adj[v]) for u, v in candidates]
```

This is the **Common Neighbors (CN)** heuristic.

## Notes on the hidden evaluation protocol

- The evaluator only uses withheld edges that remain in the distance-2 candidate set after edge removal.
- If multiple graphs are supplied, the final score is the mean of per-split metrics across graphs.
- You should optimize the ranking quality, not classification calibration.

## Notes on the leaderboard protocol

The bundled `leaderboard.py` script may optionally cap the candidate set size for very dense graphs by keeping only the highest-CN candidate pairs before evaluation. This is only to keep the full baseline sweep computationally manageable. The core evaluator itself does not require such capping unless explicitly requested.
