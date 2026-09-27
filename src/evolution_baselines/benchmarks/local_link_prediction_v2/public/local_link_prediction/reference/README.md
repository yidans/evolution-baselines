# RSI Benchmark: Local Link Prediction by Triangle Completion

This benchmark is designed for recursive self-improvement (RSI) systems that search for new algorithms.

## One-sentence task

Given a partially observed undirected network, invent a scoring function that ranks currently non-adjacent node pairs with at least one common neighbor by how likely they are to be a missing edge.

## Why this benchmark is a good RSI target

- It is easy to explain.
- It is native to network science.
- Candidate algorithms are compact and interpretable.
- Evaluation is fully empirical and automatable.
- There are clear baselines and strong hidden references.

## Public files

- `problem_spec.md` — exact task definition to give the RSI system.
- `evaluator.py` — evaluator and CLI.
- `baselines.py` — reference baselines for human benchmarking.
- `baseline_cn.py` — a minimal baseline you can expose to the RSI system.
- `leaderboard.py` — runs all bundled baselines across all bundled real datasets.
- `prepare_datasets.py` — converts downloaded raw datasets into benchmark-ready edge lists.
- `rsi_context_and_setup.md` — benchmark rationale and handoff notes for the student.
- `data/processed/dataset_manifest.json` — metadata for the bundled real datasets.
- `toy_graph.edgelist` — tiny sanity-check graph.

## Human-only file

- `human_notes.md` — benchmarking advice, suggested hidden references, and state-of-the-art notes. Do not expose this file to the RSI system if you want a fair hidden-target evaluation.

## Real datasets

Processed benchmark datasets live in `data/processed/`:

- `dolphins.edgelist`
- `ca_GrQc_core.edgelist`
- `facebook_ego.edgelist`
- `dataset_manifest.json`

## Quick start

Run the exposed CN baseline on the toy graph:

```bash
python /workspace/rsi_benchmarks/local_link_prediction/evaluator.py \
  --solution /workspace/rsi_benchmarks/local_link_prediction/baseline_cn.py \
  --edgelist /workspace/rsi_benchmarks/local_link_prediction/toy_graph.edgelist \
  --splits 3 \
  --seed 0
```

Prepare the real datasets after download:

```bash
python3 /workspace/rsi_benchmarks/local_link_prediction/prepare_datasets.py
```

Run the bundled baseline leaderboard:

```bash
python3 /workspace/rsi_benchmarks/local_link_prediction/leaderboard.py \
  --data-dir /workspace/rsi_benchmarks/local_link_prediction/data/processed \
  --splits 3 \
  --max-candidates 5000 \
  --seed 0
```

## Intended use

1. Give the RSI system `problem_spec.md` and optionally `baseline_cn.py`.
2. Ask it to output a Python file implementing the required function signature.
3. Evaluate with `evaluator.py` on one or more graphs.
4. Compare the discovered algorithm against the leaderboard baselines and against hidden references stored only for the human evaluator.
