"""Subprocess worker: run one candidate ``solution.py`` over projected instances.

Standalone on purpose (standard library + NumPy only); it never imports the
controller package and never sees hidden labels.  Each instance gets its own
wall-clock alarm so a runaway candidate on one split does not consume the
whole evaluation budget.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import signal
import sys
import time
import traceback
from pathlib import Path

SOLUTION_SCHEMA = "local-link-prediction-solution.v1"


class _InstanceTimeout(Exception):
    pass


def _alarm(_signum, _frame):
    raise _InstanceTimeout("per-instance wall-clock limit exceeded")


def _load_score_candidates(path: Path):
    spec = importlib.util.spec_from_file_location("candidate_solution", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate_solution"] = module
    spec.loader.exec_module(module)
    function = getattr(module, "score_candidates", None)
    if not callable(function):
        raise AttributeError("solution.py must define a callable score_candidates(adj, candidates)")
    return function


def _run_instance(function, instance, timeout):
    import numpy as np

    adj = [set() for _ in range(int(instance["node_count"]))]
    for u, v in instance["train_edges"]:
        adj[u].add(v)
        adj[v].add(u)
    candidates = np.asarray(instance["candidates"], dtype=int).reshape((-1, 2))
    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, float(timeout))
    start = time.perf_counter()
    try:
        raw = function(adj, candidates)
        scores = np.asarray(raw, dtype=float)
        seconds = time.perf_counter() - start
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
    if scores.shape != (len(candidates),):
        raise ValueError(
            f"score_candidates returned shape {tuple(scores.shape)} for {len(candidates)} candidates"
        )
    values = [float(x) if math.isfinite(float(x)) else None for x in scores.tolist()]
    return values, seconds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    args = parser.parse_args()
    instances = json.loads(args.instances.read_text())
    results = []
    try:
        function = _load_score_candidates(args.solution)
    except BaseException as exc:  # noqa: BLE001 - report any import failure to the parent
        error = f"{type(exc).__name__}: {exc}"
        detail = traceback.format_exc()[-1500:]
        for instance in instances:
            results.append(
                {
                    "instance_id": instance["instance_id"],
                    "status": "failed",
                    "seconds": 0.0,
                    "error": error,
                    "traceback": detail,
                    "solution": None,
                }
            )
        args.output.write_text(json.dumps(results))
        return 0
    for instance in instances:
        start = time.perf_counter()
        try:
            values, seconds = _run_instance(function, instance, args.timeout)
            results.append(
                {
                    "instance_id": instance["instance_id"],
                    "status": "passed",
                    "seconds": seconds,
                    "error": None,
                    "solution": {"schema_version": SOLUTION_SCHEMA, "scores": values},
                }
            )
        except BaseException as exc:  # noqa: BLE001 - candidate code may raise anything
            results.append(
                {
                    "instance_id": instance["instance_id"],
                    "status": "failed",
                    "seconds": time.perf_counter() - start,
                    "error": f"{type(exc).__name__}: {exc}"[:600],
                    "traceback": traceback.format_exc()[-1500:],
                    "solution": None,
                }
            )
    args.output.write_text(json.dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
