"""Optional public evaluator using Galahad's actual isolated execution path.

No controller code is copied here. Install Galahad in the operator environment.
This adapter collects public evidence; it does not issue a freeze certificate or
open a hidden test set. Those operations retain Galahad's existing requirements.
"""
from __future__ import annotations

import json
import math
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from .evaluate import CandidateEvaluator, EvaluationResult, _family_feedback, program_id, write_frozen_bundle


class GalahadEvaluator(CandidateEvaluator):
    backend = "galahad"

    def __init__(self, pack, work_dir):
        super().__init__(pack, work_dir)
        try:
            from ai_professor.autonomy.computational_project import MethodProject
            from ai_professor.autonomy.isolation import _selected_process_sandbox, _trusted_sandbox_names
            from ai_professor.autonomy.qualification import aggregate_development_qualification_v2
        except ImportError as exc:
            raise RuntimeError(
                "Galahad evaluator requires the Galahad package in this Python environment; "
                "install it with pip install -e /path/to/galahad"
            ) from exc
        sandbox = _selected_process_sandbox()
        if not _trusted_sandbox_names({sandbox}):
            raise RuntimeError(
                "Galahad requires trusted process isolation: configure GALAHAD_EVALUATOR_IMAGE "
                "on Linux with Docker, or use macOS Seatbelt. Host evaluation is not accepted."
            )
        # The row evaluator only needs execution methods. No research stages or
        # synthetic review/correctness artifacts are created to satisfy freeze.
        self._runner = MethodProject(self.work_dir / "projects", "baseline-evaluation")
        self._aggregate = aggregate_development_qualification_v2
        self._profile: dict[str, Any] | None = None

    def baseline_metrics(self, scope):
        if scope != "public":
            raise ValueError("Galahad search evaluator accepts public development inputs only")
        if self._profile is None:
            # Run through the same generator/baseline/scorer path; the profile
            # only feeds public prompts. Every candidate evaluation reruns the
            # baselines, as Galahad qualification does.
            result = self.evaluate(self.pack.baseline_source("cn"), label="public-profile")
            profile = {}
            for row in result.rows:
                if row["status"] != "passed" or not row["trusted_isolation"]:
                    raise RuntimeError(f"Galahad public baseline execution failed: {row.get('error')}")
                key = f"{row['family_id']}/{row['instance_id']}"
                profile.setdefault(key, {
                    name: {**metrics, "seconds": row["compute"]["baseline_seconds"][name]}
                    for name, metrics in row["baseline_metrics"].items()
                })
            self._profile = profile
        return self._profile

    def evaluate(self, source, *, label=None):
        start = time.perf_counter()
        specs = self.pack.public_specs()
        rows = []
        with tempfile.TemporaryDirectory(prefix="galahad-public-", dir=self.work_dir) as directory:
            root = Path(directory)
            bundle = write_frozen_bundle(source, root / "candidate")
            method = json.loads((bundle / "method_manifest.json").read_text())
            for spec in specs:
                for seed in self.seeds:
                    run_dir = root / f"{spec.family_id}-{spec.instance_id}-{seed}"
                    run_dir.mkdir()
                    row = self._runner._evaluate_one_heldout_run(
                        family_id=spec.family_id, instance_id=spec.instance_id,
                        instance_seed=spec.instance_seed, config_path=spec.config_path,
                        seed=seed, run_dir=run_dir, algorithm_dir=bundle,
                        method_manifest=method, benchmark_dir=self.pack.dev_bundle,
                        benchmark_manifest=self.pack.dev_manifest,
                        protocol=self.pack.public_contract, timeout=self.timeout_seconds,
                        required_generated_instance_fields=(
                            self.pack.public_contract["development_qualification_policy"]
                            ["runtime_envelope"].get("required_generated_instance_fields")
                            if spec.role == "runtime_envelope" else None
                        ),
                    )
                    row.update(development_role=spec.role, noise_band=spec.noise_band)
                    rows.append(row)
        receipt = self._aggregate(rows, self.pack.public_contract,
                                  [spec.qualification_spec() for spec in specs])
        trusted = bool(rows) and all(row.get("trusted_isolation") is True for row in rows)
        signatures: dict[tuple[str, str], set] = {}
        for row in rows:
            key = (row["family_id"], row["instance_id"])
            signatures.setdefault(key, set()).add(
                (row.get("candidate_execution") or {}).get("solution_canonical_sha256")
            )
        nondeterministic = any(len(values) > 1 for values in signatures.values())
        relative = [instance.get("relative_improvement_over_strongest_baseline")
                    for family in receipt["families"] for instance in family["quality_instances"]]
        valid = bool(relative) and trusted and not nondeterministic and all(
            isinstance(value, (float, int)) and math.isfinite(value) for value in relative
        ) and receipt["all_candidate_runs_valid"] and receipt["all_baseline_runs_valid"]
        passed = int(receipt["passed_family_count"]) if valid else 0
        required = self.pack.required_family_count()
        qualified = bool(valid and passed >= required and receipt["runtime_envelope_pass"])
        errors = list(dict.fromkeys(row["error"] for row in rows if row.get("error")))
        if nondeterministic:
            errors.append("score_candidates is not deterministic across method seeds")
        if not trusted:
            errors.append("one or more controller executions lacked trusted isolation")
        claim = "controller_public_evaluation" if trusted else "controller_execution_failed"
        receipt.update(claim_status=claim, trusted_isolation=trusted)
        per_instance = [{
            "family_id": row["family_id"], "instance_id": row["instance_id"],
            "role": row["development_role"], "status": row["status"],
            "seconds": row.get("compute", {}).get("candidate_seconds"),
            "average_precision": row.get("candidate_metrics", {}).get("average_precision"),
            "strongest_baseline": max((m["average_precision"] for m in
                row.get("baseline_metrics", {}).values()), default=None),
            "error": row.get("error"),
        } for row in rows if row["seed"] == self.seeds[0]]
        result = EvaluationResult(
            program_id=program_id(source), scope="public", fitness=statistics.fmean(relative) if valid else -1.0,
            families_passed=passed, required_family_count=required, qualified=qualified,
            valid=bool(valid), nondeterministic=nondeterministic, seconds=time.perf_counter() - start,
            errors=errors[:6], families=[_family_feedback(f) for f in receipt["families"]],
            receipt=receipt, rows=rows, per_instance=per_instance,
            claim_status=claim, trusted_isolation=trusted,
        )
        evidence = self.work_dir / f"{label or result.program_id}.controller.json"
        evidence.write_text(json.dumps(result.to_json(), indent=1, allow_nan=False) + "\n")
        return result

    def heldout(self, source, *, label=None):
        raise ValueError(
            "Use the Galahad project's qualification, method review, freeze_algorithm and "
            "evaluate_heldout flow for final evidence. This adapter only evaluates public inputs."
        )


def qualify_run(run: Path, project_path: Path, *, control_root: Path | None = None):
    """Requalify the selected program in an existing Galahad benchmark project.

The controller checks its existing locked benchmark and baseline receipts.
The resulting receipt alone does not register, review or freeze a method.
"""
    from ai_professor.autonomy.computational_project import MethodProject

    project = MethodProject.open(project_path, control_root=control_root)
    protocol = project.benchmark_protocol()
    manifest = json.loads((run / "run_manifest.json").read_text())
    contract = protocol.get("hidden_benchmark_contract_id")
    if contract != manifest["pack"]:
        raise ValueError(f"Galahad project pack {contract!r} differs from run pack {manifest['pack']!r}")
    source = (run / "best" / "solution.py").read_text()
    if program_id(source) != (manifest.get("best") or {}).get("program_id"):
        raise ValueError("selected program differs from run_manifest.json")
    # Retain the exact submitted candidate under stage 3 for controller review.
    destination = project.stage(3) / "outputs" / f"baseline-{program_id(source)}"
    destination.mkdir(exist_ok=False)
    bundle = write_frozen_bundle(source, destination)
    receipt = project._qualify_development_frontier(
        source=bundle, method_manifest=json.loads((bundle / "method_manifest.json").read_text()),
        version_id=f"baseline-{program_id(source)}", protocol=protocol,
    )
    (run / "galahad_qualification.json").write_text(json.dumps(receipt, indent=1) + "\n")
    return receipt
