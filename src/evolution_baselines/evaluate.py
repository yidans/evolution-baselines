"""Screen ``score_candidates`` programs with the controller's statistical rules.

These local subprocess measurements are operator diagnostics. Formal qualification
and held-out evidence require the controller's isolated evaluator and freeze flow.

* the candidate runs in a subprocess (:mod:`._worker`) on the controller projection of
  each instance (``instance_required`` fields only, never ``hidden_edges``);
* the pack's own ``scorers/score.py`` computes AP / Hits@L / AUC;
* rows are aggregated by the controller's ``aggregate_development_qualification_v2``
  with the pack's protocol, so LCB, win-rate and family rules are identical to Galahad's;
* hidden held-out graphs are reachable only through :meth:`CandidateEvaluator.heldout`.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._qualification import (
    aggregate_development_qualification_v2,
    comparison_outcome,
)

from .pack import SOLUTION_SCHEMA, InstanceSpec, LinkPredictionPack

WORKER = Path(__file__).with_name("_worker.py")
FROZEN_METHOD_SCHEMA = "frozen-computational-method.v1"

RUN_METHOD_SOURCE = '''"""Bundle-local JSON CLI adapter (same contract as the pack's target_runtime.py)."""
import argparse
import json
from pathlib import Path

import numpy as np

from solution import score_candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    instance = json.loads(args.instance.read_text())
    adj = [set() for _ in range(instance["node_count"])]
    for u, v in instance["train_edges"]:
        adj[u].add(v)
        adj[v].add(u)
    candidates = np.asarray(instance["candidates"], dtype=int).reshape((-1, 2))
    scores = np.asarray(score_candidates(adj, candidates), dtype=float)
    if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
        raise ValueError("score_candidates must return one finite number per candidate")
    args.output.write_text(json.dumps({
        "schema_version": "local-link-prediction-solution.v1", "scores": scores.tolist(),
    }, allow_nan=False) + "\\n")


if __name__ == "__main__":
    main()
'''


def program_id(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]


def _controller_validity(metrics: dict[str, Any], derivation: dict[str, Any] | None) -> float:
    if not derivation:
        return 1.0
    ops = {"<=": lambda a, b: a <= b, "<": lambda a, b: a < b, ">=": lambda a, b: a >= b,
           ">": lambda a, b: a > b, "==": lambda a, b: a == b}
    for rule in derivation.get("validity_rules", []):
        value = metrics.get(rule["metric"])
        if value is None or not ops[rule["operator"]](float(value), float(rule["value"])):
            return 0.0
    return 1.0


@dataclass
class InstanceRun:
    instance_id: str
    status: str
    seconds: float
    error: str | None
    metrics: dict[str, Any]
    scores: list[float | None] | None = None


@dataclass
class FamilyFeedback:
    family_id: str
    status: str
    candidate_mean: float | None
    strongest_baseline_mean: float | None
    mean_relative_improvement: float | None
    lower_confidence_bound_95: float | None
    wins: int
    ties: int
    losses: int
    instances_valid: int
    instances_total: int
    runtime_status: str


@dataclass
class EvaluationResult:
    program_id: str
    scope: str
    fitness: float
    families_passed: int
    required_family_count: int
    qualified: bool
    valid: bool
    nondeterministic: bool
    seconds: float
    errors: list[str]
    families: list[FamilyFeedback]
    receipt: dict[str, Any] | None = None
    heldout: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    per_instance: list[dict[str, Any]] = field(default_factory=list)
    claim_status: str = "operator_diagnostic_only"
    trusted_isolation: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "claim_status": self.claim_status,
            "trusted_isolation": self.trusted_isolation,
            "program_id": self.program_id,
            "scope": self.scope,
            "fitness": self.fitness,
            "families_passed": self.families_passed,
            "required_family_count": self.required_family_count,
            "qualified": self.qualified,
            "valid": self.valid,
            "nondeterministic": self.nondeterministic,
            "seconds": self.seconds,
            "errors": self.errors,
            "families": [vars(item) for item in self.families],
        }

    def to_json(self) -> dict[str, Any]:
        payload = self.summary()
        payload["receipt"] = self.receipt
        payload["heldout"] = self.heldout
        payload["rows"] = self.rows
        payload["per_instance"] = self.per_instance
        return payload


def write_frozen_bundle(source: str, destination: Path, *, metadata: dict[str, Any] | None = None) -> Path:
    """Write ``solution.py`` + ``run_method.py`` + ``method_manifest.json`` (frozen-method shape)."""
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "solution.py").write_text(source)
    (destination / "run_method.py").write_text(RUN_METHOD_SOURCE)
    manifest = {
        "schema_version": FROZEN_METHOD_SCHEMA,
        "command": ["python3", "run_method.py", "--instance", "{instance_path}", "--seed", "{seed}",
                    "--output", "{solution_path}"],
        "interface": {"module": "solution.py", "function": "score_candidates"},
        "program_id": program_id(source),
    }
    if metadata:
        manifest["baseline_metadata"] = metadata
    (destination / "method_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


class CandidateEvaluator:
    """Run candidates on the pack's public (development) or hidden (held-out) instances."""

    backend = "subprocess"

    def __init__(
        self,
        pack: LinkPredictionPack,
        work_dir: Path,
        *,
        seeds: list[int] | None = None,
        timeout_seconds: float | None = None,
        python: str | None = None,
    ) -> None:
        self.pack = pack
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.seeds = list(seeds) if seeds is not None else pack.seeds
        self.timeout_seconds = float(timeout_seconds or pack.timeout_seconds)
        self.python = python or sys.executable
        self._instances: dict[str, list[tuple[InstanceSpec, dict[str, Any], dict[str, Any]]]] = {}
        self._baseline_metrics: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        self._derivation = pack.public_contract["metrics"].get("controller_derivation")

    # ------------------------------------------------------------ instance data
    def instances(self, scope: str) -> list[tuple[InstanceSpec, dict[str, Any], dict[str, Any]]]:
        if scope not in self._instances:
            specs = self.pack.public_specs() if scope == "public" else self.pack.hidden_specs()
            loaded = []
            for spec in specs:
                full = self.pack.load_instance(spec)
                loaded.append((spec, full, self.pack.project(full)))
            self._instances[scope] = loaded
        return self._instances[scope]

    # ---------------------------------------------------------------- execution
    def _run_worker(self, source: str, scope: str, *, label: str) -> dict[str, dict[str, Any]]:
        instances = self.instances(scope)
        with tempfile.TemporaryDirectory(prefix=f"linkpred-{label}-", dir=self.work_dir) as tmp:
            tmp_path = Path(tmp)
            solution = tmp_path / "solution.py"
            solution.write_text(source)
            payload = [{"instance_id": spec.key, **projected}
                       for spec, _full, projected in instances]
            (tmp_path / "instances.json").write_text(json.dumps(payload))
            output = tmp_path / "results.json"
            budget = self.timeout_seconds * len(instances) + 60.0
            env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1",
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PYTHONHASHSEED": "0",
                   "HOME": str(tmp_path)}
            try:
                process = subprocess.run(
                    [self.python, str(WORKER), "--solution", str(solution), "--instances",
                     str(tmp_path / "instances.json"), "--output", str(output), "--timeout",
                     str(self.timeout_seconds)],
                    cwd=tmp_path, capture_output=True, text=True, timeout=budget, env=env,
                )
                failure = None if process.returncode == 0 else (
                    f"worker exited {process.returncode}: {process.stderr[-800:]}")
            except subprocess.TimeoutExpired:
                failure = "worker exceeded the whole-evaluation wall-clock budget"
            if failure or not output.is_file():
                return {spec.key: {"status": "failed", "seconds": 0.0,
                                   "error": failure or "worker produced no output",
                                   "solution": None} for spec, _f, _p in instances}
            return {item["instance_id"]: item for item in json.loads(output.read_text())}

    def _score_runs(self, scope: str, raw: dict[str, dict[str, Any]]) -> dict[str, InstanceRun]:
        runs: dict[str, InstanceRun] = {}
        for spec, full, _projected in self.instances(scope):
            item = raw.get(spec.key) or {"status": "failed", "seconds": 0.0,
                                         "error": "missing worker result", "solution": None}
            metrics = self.pack.score(full, item.get("solution"))
            metrics["controller_validity_pass"] = _controller_validity(metrics, self._derivation)
            status = "passed" if item.get("status") == "passed" else "failed"
            error = item.get("error")
            if status == "passed" and metrics["controller_validity_pass"] != 1.0:
                status = "failed"
                error = ("solution rejected by controller validity rules: "
                         f"schema_error_count={metrics['schema_error_count']}, "
                         f"score_count_mismatch={metrics['score_count_mismatch']}, "
                         f"non_finite_score_count={metrics['non_finite_score_count']}")
            solution = item.get("solution") or {}
            runs[spec.key] = InstanceRun(
                instance_id=spec.instance_id, status=status, seconds=float(item.get("seconds", 0.0)),
                error=error, metrics=metrics, scores=solution.get("scores"))
        return runs

    # ---------------------------------------------------------------- baselines
    def baseline_metrics(self, scope: str) -> dict[str, dict[str, dict[str, Any]]]:
        """``{instance_id: {baseline_id: metrics}}`` computed once per scope via the worker."""
        if scope in self._baseline_metrics:
            return self._baseline_metrics[scope]
        cache = self.work_dir / f"baseline_metrics_{scope}.json"
        if cache.is_file():
            cached = json.loads(cache.read_text())
            if cached.get("baselines") == self.pack.baseline_ids and cached.get(
                    "pack") == self.pack.contract_id:
                self._baseline_metrics[scope] = cached["metrics"]
                return cached["metrics"]
        table: dict[str, dict[str, dict[str, Any]]] = {}
        for baseline_id in self.pack.baseline_ids:
            runs = self._score_runs(scope, self._run_worker(
                self.pack.baseline_source(baseline_id), scope, label=f"baseline-{baseline_id}"))
            for key, run in runs.items():
                if run.status != "passed":
                    raise RuntimeError(f"pack baseline {baseline_id} failed on {key}: {run.error}")
                table.setdefault(key, {})[baseline_id] = {**run.metrics, "seconds": run.seconds}
        cache.write_text(json.dumps({"pack": self.pack.contract_id, "scope": scope,
                                     "baselines": self.pack.baseline_ids, "metrics": table}))
        self._baseline_metrics[scope] = table
        return table

    # ------------------------------------------------------------- public scope
    def evaluate(self, source: str, *, label: str | None = None) -> EvaluationResult:
        """Public development evaluation + the controller's qualification aggregate."""
        start = time.perf_counter()
        pid = program_id(source)
        label = label or pid
        baselines = self.baseline_metrics("public")
        seed_runs: dict[int, dict[str, InstanceRun]] = {}
        for seed in self.seeds:
            seed_runs[seed] = self._score_runs("public", self._run_worker(
                source, "public", label=f"{label}-s{seed}"))
        nondeterministic = False
        first = seed_runs[self.seeds[0]]
        for seed in self.seeds[1:]:
            for key, run in seed_runs[seed].items():
                if run.scores != first[key].scores or run.status != first[key].status:
                    nondeterministic = True
        rows: list[dict[str, Any]] = []
        errors: list[str] = []
        per_instance: list[dict[str, Any]] = []
        for seed in self.seeds:
            for spec, _full, _projected in self.instances("public"):
                run = seed_runs[seed][spec.key]
                if run.error and run.error not in errors:
                    errors.append(run.error)
                rows.append({
                    "family_id": spec.family_id, "instance_id": spec.instance_id,
                    "instance_seed": spec.instance_seed, "development_role": spec.role,
                    "seed": seed, "status": run.status, "error": run.error,
                    "candidate_metrics": dict(run.metrics),
                    "baseline_metrics": {name: dict(m) for name, m in baselines[spec.key].items()},
                    "compute": {"candidate_seconds": run.seconds,
                                "baseline_seconds": {name: m.get("seconds") for name, m in
                                                     baselines[spec.key].items()}},
                    "noise_band": spec.noise_band, "trusted_isolation": False,
                    "isolation": "subprocess worker, projected instance, per-instance alarm",
                })
                if seed == self.seeds[0]:
                    per_instance.append({"family_id": spec.family_id, "instance_id": spec.instance_id,
                                         "role": spec.role, "status": run.status, "seconds": run.seconds,
                                         "average_precision": run.metrics.get("average_precision"),
                                         "strongest_baseline": max(
                                             float(m["average_precision"]) for m in
                                             baselines[spec.key].values()),
                                         "error": run.error})
        if nondeterministic:
            errors.append("score_candidates is not deterministic across method seeds")
        specs = [spec.qualification_spec() for spec, _f, _p in self.instances("public")]
        # Compute a statistical screening result assuming trusted execution. The
        # core aggregator intentionally rejects untrusted rows; leave that formal
        # boundary unchanged. Persist the actual isolation status on both rows and
        # receipt so this local calculation cannot be mistaken for controller proof.
        statistical_rows = [{**row, "trusted_isolation": True} for row in rows]
        receipt = aggregate_development_qualification_v2(
            statistical_rows, self.pack.public_contract, specs)
        receipt.update(claim_status="operator_diagnostic_only", trusted_isolation=False)
        families = [_family_feedback(item) for item in receipt["families"]]
        relative = [inst.get("relative_improvement_over_strongest_baseline")
                    for item in receipt["families"] for inst in item["quality_instances"]]
        valid = (bool(relative) and all(isinstance(v, (int, float)) and math.isfinite(v)
                                       for v in relative) and not nondeterministic
                 and receipt["all_candidate_runs_valid"] and receipt["all_baseline_runs_valid"])
        fitness = statistics.fmean(relative) if valid else -1.0
        required = self.pack.required_family_count()
        passed = int(receipt["passed_family_count"]) if valid else 0
        return EvaluationResult(
            program_id=pid, scope="public", fitness=fitness, families_passed=passed,
            required_family_count=required,
            qualified=bool(valid and passed >= required and receipt["runtime_envelope_pass"]),
            valid=valid, nondeterministic=nondeterministic, seconds=time.perf_counter() - start,
            errors=errors[:6], families=families, receipt=receipt, rows=rows,
            per_instance=per_instance)

    # ------------------------------------------------------------- hidden scope
    def heldout(self, source: str, *, label: str | None = None) -> EvaluationResult:
        """Controller-private held-out evaluation.  Callers must gate this on qualification."""
        start = time.perf_counter()
        pid = program_id(source)
        baselines = self.baseline_metrics("hidden")
        runs = self._score_runs("hidden", self._run_worker(source, "hidden", label=f"{label or pid}-held"))
        tags = self.pack.hidden_family_tags()
        primary = self.pack.primary_metric
        per_family: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        errors: list[str] = []
        all_relative: list[float] = []
        all_relative_ra: list[float] = []
        all_relative_cn: list[float] = []
        for family_id in dict.fromkeys(spec.family_id for spec, _f, _p in self.instances("hidden")):
            family_specs = [s for s, _f, _p in self.instances("hidden") if s.family_id == family_id]
            cand: list[float] = []
            strongest: list[float] = []
            per_baseline: dict[str, list[float]] = {name: [] for name in self.pack.baseline_ids}
            outcomes = {"win": 0, "tie": 0, "loss": 0}
            valid_count = 0
            for spec in family_specs:
                run = runs[spec.key]
                base = baselines[spec.key]
                row = {"family_id": family_id, "instance_id": spec.instance_id, "status": run.status,
                       "seconds": run.seconds, "candidate": run.metrics.get(primary),
                       "baselines": {name: m.get(primary) for name, m in base.items()}, "error": run.error}
                rows.append(row)
                if run.error and run.error not in errors:
                    errors.append(run.error)
                if run.status != "passed":
                    continue
                valid_count += 1
                value = float(run.metrics[primary])
                best = max(float(m[primary]) for m in base.values())
                cand.append(value)
                strongest.append(best)
                for name in per_baseline:
                    per_baseline[name].append(float(base[name][primary]))
                outcomes[comparison_outcome(value - best, spec.noise_band)["outcome"]] += 1
            def _rel(values: list[float], reference: list[float]) -> float | None:
                if not values or not reference or abs(statistics.fmean(reference)) < 1e-12:
                    return None
                return (statistics.fmean(values) - statistics.fmean(reference)) / abs(
                    statistics.fmean(reference))
            summary = {
                "family_id": family_id, "private_family_tag": tags.get(family_id, family_id),
                "instances_valid": valid_count, "instances_total": len(family_specs),
                "candidate_mean": statistics.fmean(cand) if cand else None,
                "strongest_baseline_mean": statistics.fmean(strongest) if strongest else None,
                "baseline_means": {name: (statistics.fmean(v) if v else None) for name, v in per_baseline.items()},
                "relative_to_rowwise_strongest": _rel(cand, strongest),
                "relative_to_ra": _rel(cand, per_baseline.get("ra", [])),
                "relative_to_cn": _rel(cand, per_baseline.get("cn", [])),
                "wins": outcomes["win"], "ties": outcomes["tie"], "losses": outcomes["loss"],
            }
            per_family.append(summary)
            if summary["relative_to_rowwise_strongest"] is not None:
                all_relative.append(summary["relative_to_rowwise_strongest"])
            if summary["relative_to_ra"] is not None:
                all_relative_ra.append(summary["relative_to_ra"])
            if summary["relative_to_cn"] is not None:
                all_relative_cn.append(summary["relative_to_cn"])
        complete = all(item["instances_valid"] == item["instances_total"] for item in per_family)
        heldout = {
            "claim_status": "operator_diagnostic_only", "trusted_isolation": False,
            "scope": "hidden", "program_id": pid, "complete": complete,
            "families": per_family,
            "mean_relative_to_rowwise_strongest": statistics.fmean(all_relative) if all_relative else None,
            "mean_relative_to_ra": statistics.fmean(all_relative_ra) if all_relative_ra else None,
            "mean_relative_to_cn": statistics.fmean(all_relative_cn) if all_relative_cn else None,
            "minimum_relative_improvement": self.pack.minimum_relative_improvement,
            "families_meeting_minimum": sum(
                1 for item in per_family if item["relative_to_rowwise_strongest"] is not None
                and item["relative_to_rowwise_strongest"] >= self.pack.minimum_relative_improvement),
            "rows": rows,
        }
        return EvaluationResult(
            program_id=pid, scope="hidden", fitness=float("nan"), families_passed=0,
            required_family_count=self.pack.required_family_count(), qualified=False,
            valid=complete, nondeterministic=False, seconds=time.perf_counter() - start,
            errors=errors[:6], families=[], heldout=heldout)


def _family_feedback(item: dict[str, Any]) -> FamilyFeedback:
    outcomes = {"win": 0, "tie": 0, "loss": 0}
    for seed_result in item.get("seed_results", []):
        outcomes[seed_result["outcome"]] = outcomes.get(seed_result["outcome"], 0) + 1
    instances = item.get("quality_instances", [])
    return FamilyFeedback(
        family_id=str(item["family_id"]), status=str(item["status"]),
        candidate_mean=item.get("candidate_mean"),
        strongest_baseline_mean=item.get("strongest_baseline_mean"),
        mean_relative_improvement=item.get("mean_relative_improvement_over_instances"),
        lower_confidence_bound_95=item.get("lower_confidence_bound_95"),
        wins=outcomes["win"], ties=outcomes["tie"], losses=outcomes["loss"],
        instances_valid=sum(1 for inst in instances if inst.get("execution_valid")),
        instances_total=len(instances),
        runtime_status=str(item.get("runtime_envelope", {}).get("status", "unknown")),
    )


__all__ = [
    "CandidateEvaluator", "EvaluationResult", "FamilyFeedback", "InstanceRun", "SOLUTION_SCHEMA",
    "program_id", "write_frozen_bundle",
]
