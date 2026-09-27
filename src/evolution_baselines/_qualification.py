"""Galahad statistical qualification functions; see SOURCE.md for provenance."""
from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any, Mapping, Sequence


def summarize_execution_failures(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compact public-development facts, never inferred research explanations.

    Call only with the public qualification's rows. Do not traverse paths or copy
    raw logs; identical failures share one example while counts cover every row.
    """
    counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    signatures: set[tuple[str, str, str]] = set()
    for row in rows:
        if row.get("status") == "passed":
            continue
        execution = row.get("candidate_execution")
        execution = execution if isinstance(execution, Mapping) else {}
        diagnostic = execution.get("failure_diagnostic")
        diagnostic = diagnostic if isinstance(diagnostic, Mapping) else {}
        category = str(diagnostic.get("category") or row.get("status") or "execution_unknown")
        if category not in {
            "method_protocol",
            "method_execution",
            "method_output",
            "candidate_failed",
            "generator_failed",
            "baseline_failed",
            "scorer_failed",
        }:
            category = "execution_unknown"
        counts[category] += 1
        error = str(diagnostic.get("error") or row.get("error") or "execution failed")[:600]
        phase = str(diagnostic.get("phase") or "execution")[:40]
        signature = (category, error, phase)
        if signature in signatures or len(examples) >= 3:
            continue
        signatures.add(signature)
        example: dict[str, Any] = {
            "category": category,
            "error": error,
            "phase": phase,
            "family_id": str(row.get("family_id") or "")[:120],
            "instance_id": str(row.get("instance_id") or "")[:120],
            "response_excerpt": str(diagnostic.get("response_excerpt") or "")[:480],
        }
        for field in ("event_index", "accepted_response_count", "accepted_empty_response_count"):
            value = diagnostic.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                example[field] = value
        examples.append(example)
    return {
        "failure_category": next(iter(counts))
        if len(counts) == 1
        else "mixed_execution"
        if counts
        else "quality_or_validity",
        "failed_row_count": sum(counts.values()),
        "completed_comparison_count": sum(row.get("status") == "passed" for row in rows),
        "counts": dict(counts),
        "examples": examples,
    }


DEVELOPMENT_QUALIFICATION_POLICY_SCHEMA_VERSION = "development-qualification-policy.v2"
DEVELOPMENT_QUALIFICATION_RECEIPT_SCHEMA_VERSION = "development-frontier-qualification.v2"

_SELECTION_STATISTICS = frozenset({"worst_instance", "lower_confidence_bound"})
_POLICY_FIELDS = frozenset(
    {
        "schema_version",
        "minimum_quality_instances_per_family",
        "selection_statistic",
        "confidence_level",
        "runtime_envelope",
    }
)
_RUNTIME_FIELDS = frozenset(
    {
        "required",
        "minimum_instances_per_family",
        "maximum_candidate_timeout_fraction",
        "required_generated_instance_fields",
    }
)
_REQUIRED_RUNTIME_FIELDS = frozenset(
    {
        "required",
        "minimum_instances_per_family",
        "maximum_candidate_timeout_fraction",
    }
)

# One-sided 95% Student-t critical values.  The v2 contract deliberately fixes
# the confidence level so qualification receipts remain dependency-free and
# numerically reproducible in the minimal controller runtime.
_ONE_SIDED_T_95 = (
    0.0,
    6.313752,
    2.919986,
    2.353363,
    2.131847,
    2.015048,
    1.943180,
    1.894579,
    1.859548,
    1.833113,
    1.812461,
    1.795885,
    1.782288,
    1.770933,
    1.761310,
    1.753050,
    1.745884,
    1.739607,
    1.734064,
    1.729133,
    1.724718,
    1.720743,
    1.717144,
    1.713872,
    1.710882,
    1.708141,
    1.705618,
    1.703288,
    1.701131,
    1.699127,
    1.697261,
)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def comparison_outcome(
    improvement: float, noise_band: float = 0, multiplier: float = 1
) -> dict[str, Any]:
    """Apply an absolute, predeclared per-instance reference noise allowance.

    The raw measurement is retained. Unknown noise defaults to zero rather than
    estimating it from candidate differences, which would reward regressions.
    """
    if (
        not all(_finite_number(v) for v in (improvement, noise_band, multiplier))
        or noise_band < 0
        or multiplier < 0
    ):
        raise ValueError("noise band and multiplier must be finite nonnegative numbers")
    band = noise_band * multiplier
    adjusted = 0.0 if abs(improvement) <= band else improvement
    return {
        "raw_improvement": improvement,
        "noise_band": band,
        "improvement": adjusted,
        "outcome": "win" if adjusted > 0 else "loss" if adjusted < 0 else "tie",
    }


def summarize_public_comparisons(
    rows: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """Point to public counterexamples without guessing why a candidate lost."""
    metric = protocol["metrics"]["primary"]
    maximize = protocol["metrics"]["direction"] == "maximize"
    comparisons = []
    for row in rows:
        if row.get("status") != "passed" or row.get("development_role") in {
            "runtime_envelope",
            "runtime_only",
        }:
            continue
        candidate = row.get("candidate_metrics", {}).get(metric)
        values = {
            name: row.get("baseline_metrics", {}).get(name, {}).get(metric)
            for name in protocol["baselines"]
        }
        if (
            not _finite_number(candidate)
            or not values
            or not all(_finite_number(value) for value in values.values())
        ):
            continue
        best = (max if maximize else min)(values, key=values.get)
        improvement = candidate - values[best] if maximize else values[best] - candidate
        comparisons.append(
            {
                "family_id": row["family_id"],
                "instance_id": row.get("instance_id"),
                "seed": row["seed"],
                "development_role": row.get("development_role"),
                "candidate": candidate,
                "baseline": values[best],
                "baseline_id": best,
                **comparison_outcome(
                    improvement, row.get("noise_band", 0), protocol.get("noise_band_multiplier", 1)
                ),
                "instance_path": row.get("generator_execution", {}).get("public_instance_path"),
                "candidate_solution_path": row.get("candidate_execution", {}).get(
                    "public_solution_path"
                ),
                "baseline_solution_path": row.get("baseline_executions", {})
                .get(best, {})
                .get("public_solution_path"),
            }
        )
    return {
        "comparison_unit": "quality instance and method seed; runtime rows excluded; not independent samples",
        "comparison_count": len(comparisons),
        "win_count": sum(row["improvement"] > 0 for row in comparisons),
        "tie_count": sum(row["improvement"] == 0 for row in comparisons),
        "loss_count": sum(row["improvement"] < 0 for row in comparisons),
        "worst_regressions": sorted(
            (row for row in comparisons if row["improvement"] < 0),
            key=lambda row: row["improvement"],
        )[:5],
    }


def _normal_confidence_interval_95(values: Sequence[float]) -> list[float] | None:
    if not values:
        return None
    mean = statistics.fmean(values)
    if len(values) == 1:
        return [mean, mean]
    half_width = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - half_width, mean + half_width]


def _one_sided_lower_confidence_bound_95(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    degrees_of_freedom = len(values) - 1
    if degrees_of_freedom < len(_ONE_SIDED_T_95):
        critical = _ONE_SIDED_T_95[degrees_of_freedom]
    else:
        # The normal limit is slightly anti-conservative at finite df.  A
        # three-term asymptotic expansion keeps the value on the Student-t
        # scale without adding SciPy to the controller's core dependencies.
        z = 1.6448536269514722
        df = float(degrees_of_freedom)
        critical = (
            z
            + (z**3 + z) / (4.0 * df)
            + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
            + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z) / (384.0 * df**3)
        )
    return statistics.fmean(values) - critical * statistics.stdev(values) / math.sqrt(len(values))


def validate_development_qualification_policy(value: Any) -> dict[str, Any]:
    """Validate and normalize the opt-in v2 qualification policy."""

    if not isinstance(value, Mapping):
        raise ValueError("development_qualification_policy must be an object")
    unexpected = sorted(set(value) - _POLICY_FIELDS)
    missing = sorted(_POLICY_FIELDS - set(value))
    if unexpected or missing:
        raise ValueError(
            "development_qualification_policy fields differ from the v2 contract: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    if value.get("schema_version") != DEVELOPMENT_QUALIFICATION_POLICY_SCHEMA_VERSION:
        raise ValueError(
            "development_qualification_policy must use "
            f"{DEVELOPMENT_QUALIFICATION_POLICY_SCHEMA_VERSION}"
        )
    minimum_quality = value.get("minimum_quality_instances_per_family")
    if (
        isinstance(minimum_quality, bool)
        or not isinstance(minimum_quality, int)
        or not 3 <= minimum_quality <= 32
    ):
        raise ValueError(
            "development qualification v2 requires at least three independent quality "
            "instances per family (maximum 32)"
        )
    selection = value.get("selection_statistic")
    if selection not in _SELECTION_STATISTICS:
        raise ValueError(
            "development qualification selection_statistic must be worst_instance or "
            "lower_confidence_bound"
        )
    confidence = value.get("confidence_level")
    if not _finite_number(confidence) or not math.isclose(
        float(confidence), 0.95, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("development qualification v2 confidence_level must be 0.95")
    runtime = value.get("runtime_envelope")
    if not isinstance(runtime, Mapping):
        raise ValueError("development qualification v2 requires runtime_envelope")
    unexpected_runtime = sorted(set(runtime) - _RUNTIME_FIELDS)
    missing_runtime = sorted(_REQUIRED_RUNTIME_FIELDS - set(runtime))
    if unexpected_runtime or missing_runtime:
        raise ValueError(
            "runtime_envelope fields differ from the v2 contract: "
            f"missing={missing_runtime!r}, unexpected={unexpected_runtime!r}"
        )
    if runtime.get("required") is not True:
        raise ValueError("development qualification v2 runtime_envelope.required must be true")
    minimum_runtime = runtime.get("minimum_instances_per_family")
    if (
        isinstance(minimum_runtime, bool)
        or not isinstance(minimum_runtime, int)
        or not 1 <= minimum_runtime <= 8
    ):
        raise ValueError(
            "runtime_envelope.minimum_instances_per_family must be an integer in [1, 8]"
        )
    timeout_fraction = runtime.get("maximum_candidate_timeout_fraction")
    if not _finite_number(timeout_fraction) or not 0 < float(timeout_fraction) <= 1:
        raise ValueError("runtime_envelope.maximum_candidate_timeout_fraction must be in (0, 1]")
    required_instance_fields = runtime.get("required_generated_instance_fields")
    if required_instance_fields is not None:
        if not isinstance(required_instance_fields, Mapping) or not required_instance_fields:
            raise ValueError(
                "runtime_envelope.required_generated_instance_fields must be a non-empty "
                "object when present"
            )
        for field, expected in required_instance_fields.items():
            if not isinstance(field, str) or not field.strip():
                raise ValueError(
                    "runtime-envelope generated-instance field names must be non-empty strings"
                )
            if isinstance(expected, (Mapping, list)) or not isinstance(
                expected, (str, int, float, bool)
            ):
                raise ValueError(
                    "runtime-envelope generated-instance requirements support scalar JSON "
                    "field values only"
                )
            if isinstance(expected, float) and not math.isfinite(expected):
                raise ValueError("runtime-envelope generated-instance requirements must be finite")
    normalized_runtime = {
        "required": True,
        "minimum_instances_per_family": int(minimum_runtime),
        "maximum_candidate_timeout_fraction": float(timeout_fraction),
    }
    if required_instance_fields is not None:
        normalized_runtime["required_generated_instance_fields"] = dict(required_instance_fields)
    return {
        "schema_version": DEVELOPMENT_QUALIFICATION_POLICY_SCHEMA_VERSION,
        "minimum_quality_instances_per_family": int(minimum_quality),
        "selection_statistic": str(selection),
        "confidence_level": 0.95,
        "runtime_envelope": normalized_runtime,
    }


def _run_validity(
    row: Mapping[str, Any],
    *,
    baselines: Sequence[str],
    controller_derivation: bool,
) -> tuple[bool, bool]:
    if row.get("status") != "passed" or row.get("trusted_isolation") is not True:
        return False, False
    candidate_metrics = row.get("candidate_metrics")
    baseline_metrics = row.get("baseline_metrics")
    candidate_valid = isinstance(candidate_metrics, Mapping)
    baselines_valid = isinstance(baseline_metrics, Mapping)
    if controller_derivation:
        candidate_valid = bool(
            candidate_valid and candidate_metrics.get("controller_validity_pass") == 1.0
        )
        baselines_valid = bool(
            baselines_valid
            and all(
                isinstance(baseline_metrics.get(name), Mapping)
                and baseline_metrics[name].get("controller_validity_pass") == 1.0
                for name in baselines
            )
        )
    return candidate_valid, baselines_valid


def _instance_quality_summary(
    instance_rows: Sequence[Mapping[str, Any]],
    *,
    spec: Mapping[str, Any],
    seeds: Sequence[int],
    baselines: Sequence[str],
    primary: str,
    direction: str,
    minimum_win_rate: float,
    controller_derivation: bool,
) -> dict[str, Any]:
    expected_seeds = sorted(int(seed) for seed in seeds)
    observed_seeds = sorted(
        int(row["seed"])
        for row in instance_rows
        if isinstance(row.get("seed"), int) and not isinstance(row.get("seed"), bool)
    )
    candidate_validity = []
    baseline_validity = []
    for row in instance_rows:
        candidate_valid, baselines_valid = _run_validity(
            row,
            baselines=baselines,
            controller_derivation=controller_derivation,
        )
        candidate_validity.append(candidate_valid)
        baseline_validity.append(baselines_valid)
    execution_valid = bool(
        observed_seeds == expected_seeds
        and len(instance_rows) == len(expected_seeds)
        and all(candidate_validity)
        and all(baseline_validity)
    )
    candidate_values: list[float] = []
    strongest_values: list[float] = []
    improvements: list[float] = []
    seed_results: list[dict[str, Any]] = []
    if execution_valid:
        for row in sorted(instance_rows, key=lambda item: int(item["seed"])):
            candidate = row["candidate_metrics"].get(primary)
            baseline_values = [row["baseline_metrics"][name].get(primary) for name in baselines]
            if not _finite_number(candidate) or any(
                not _finite_number(value) for value in baseline_values
            ):
                execution_valid = False
                break
            candidate_value = float(candidate)
            numeric_baselines = [float(value) for value in baseline_values]
            strongest = (
                max(numeric_baselines) if direction == "maximize" else min(numeric_baselines)
            )
            improvement = (
                candidate_value - strongest
                if direction == "maximize"
                else strongest - candidate_value
            )
            candidate_values.append(candidate_value)
            strongest_values.append(strongest)
            outcome = comparison_outcome(
                improvement, row.get("noise_band", spec.get("noise_band", 0))
            )
            improvements.append(outcome["improvement"])
            seed_results.append(
                {
                    "seed": int(row["seed"]),
                    "candidate": candidate_value,
                    "strongest_baseline": strongest,
                    **outcome,
                }
            )
    mean_improvement = _mean(improvements)
    strongest_mean = _mean(strongest_values)
    relative = (
        mean_improvement / abs(strongest_mean)
        if mean_improvement is not None
        and strongest_mean is not None
        and abs(strongest_mean) > 1e-12
        else None
    )
    win_rate = (
        sum(improvement > 0 for improvement in improvements) / len(improvements)
        if improvements
        else 0.0
    )
    return {
        "instance_id": str(spec["instance_id"]),
        "instance_seed": int(spec["instance_seed"]),
        "method_seed_count": len(instance_rows),
        "execution_valid": execution_valid,
        "candidate_runs_valid": bool(candidate_validity and all(candidate_validity)),
        "baseline_runs_valid": bool(baseline_validity and all(baseline_validity)),
        "candidate_mean": _mean(candidate_values),
        "strongest_baseline_mean": strongest_mean,
        "mean_improvement_over_strongest_baseline": mean_improvement,
        "relative_improvement_over_strongest_baseline": relative,
        "win_rate_over_strongest_baseline": win_rate,
        "win_rate_pass": bool(execution_valid and win_rate >= minimum_win_rate),
        "seed_results": seed_results,
    }


def _runtime_summary(
    family_rows: Sequence[Mapping[str, Any]],
    *,
    specs: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    baselines: Sequence[str],
    controller_derivation: bool,
    allowed_seconds: float,
) -> dict[str, Any]:
    instances: list[dict[str, Any]] = []
    for spec in specs:
        instance_id = str(spec["instance_id"])
        rows = [row for row in family_rows if row.get("instance_id") == instance_id]
        expected_seeds = sorted(int(seed) for seed in seeds)
        observed_seeds = sorted(
            int(row["seed"])
            for row in rows
            if isinstance(row.get("seed"), int) and not isinstance(row.get("seed"), bool)
        )
        candidate_validity: list[bool] = []
        baseline_validity: list[bool] = []
        durations: list[float] = []
        for row in rows:
            candidate_valid, baselines_valid = _run_validity(
                row,
                baselines=baselines,
                controller_derivation=controller_derivation,
            )
            candidate_validity.append(candidate_valid)
            baseline_validity.append(baselines_valid)
            seconds = row.get("compute", {}).get("candidate_seconds")
            if _finite_number(seconds):
                durations.append(float(seconds))
        execution_valid = bool(
            observed_seeds == expected_seeds
            and len(rows) == len(expected_seeds)
            and all(candidate_validity)
            and all(baseline_validity)
            and len(durations) == len(expected_seeds)
        )
        maximum_seconds = max(durations) if durations else None
        passed = bool(
            execution_valid and maximum_seconds is not None and maximum_seconds <= allowed_seconds
        )
        instances.append(
            {
                "instance_id": instance_id,
                "instance_seed": int(spec["instance_seed"]),
                "scale": str(spec.get("scale") or ""),
                "method_seed_count": len(rows),
                "execution_valid": execution_valid,
                "maximum_candidate_seconds": maximum_seconds,
                "status": "pass" if passed else "fail",
            }
        )
    maximum_seconds = max(
        (
            float(row["maximum_candidate_seconds"])
            for row in instances
            if _finite_number(row.get("maximum_candidate_seconds"))
        ),
        default=None,
    )
    passed = bool(instances and all(row["status"] == "pass" for row in instances))
    return {
        "status": "pass" if passed else "fail",
        "allowed_candidate_seconds": allowed_seconds,
        "maximum_candidate_seconds": maximum_seconds,
        "instances": instances,
    }


def aggregate_development_qualification_v2(
    rows: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    instance_specs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate v2 qualification with graph instances as the repeat unit."""

    policy = validate_development_qualification_policy(
        protocol.get("development_qualification_policy")
    )
    metrics = protocol.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("development qualification v2 requires metrics")
    primary = str(metrics.get("primary") or "")
    direction = str(metrics.get("direction") or "")
    minimum_relative = metrics.get("minimum_relative_improvement")
    minimum_win_rate = metrics.get("minimum_win_rate", 0.5)
    if (
        not primary
        or direction not in {"maximize", "minimize"}
        or not _finite_number(minimum_relative)
        or float(minimum_relative) < 0
        or not _finite_number(minimum_win_rate)
        or not 0 <= float(minimum_win_rate) <= 1
    ):
        raise ValueError(
            "development qualification v2 requires a valid primary metric, direction, "
            "minimum_relative_improvement, and minimum_win_rate"
        )
    seeds = protocol.get("seed_policy", {}).get("seeds")
    baselines = protocol.get("baselines")
    timeout = protocol.get("compute_budget", {}).get("timeout_seconds_per_run")
    if (
        not isinstance(seeds, list)
        or not seeds
        or not isinstance(baselines, list)
        or not baselines
        or not _finite_number(timeout)
        or float(timeout) <= 0
    ):
        raise ValueError("development qualification v2 protocol is incomplete")
    family_ids = list(dict.fromkeys(str(spec.get("family_id") or "") for spec in instance_specs))
    if not family_ids or any(not family_id for family_id in family_ids):
        raise ValueError("development qualification v2 instance specs require family_id")
    controller_derivation = isinstance(metrics.get("controller_derivation"), Mapping)
    allowed_seconds = float(timeout) * float(
        policy["runtime_envelope"]["maximum_candidate_timeout_fraction"]
    )
    families: list[dict[str, Any]] = []
    for family_id in family_ids:
        family_specs = [spec for spec in instance_specs if spec.get("family_id") == family_id]
        quality_specs = [spec for spec in family_specs if spec.get("role") == "quality"]
        runtime_specs = [spec for spec in family_specs if spec.get("role") == "runtime_envelope"]
        if len(quality_specs) < policy["minimum_quality_instances_per_family"]:
            raise ValueError(f"development family {family_id} has too few quality instances")
        if len(runtime_specs) < policy["runtime_envelope"]["minimum_instances_per_family"]:
            raise ValueError(
                f"development family {family_id} has too few runtime-envelope instances"
            )
        family_rows = [row for row in rows if row.get("family_id") == family_id]
        quality_instances = [
            _instance_quality_summary(
                [row for row in family_rows if row.get("instance_id") == spec["instance_id"]],
                spec=spec,
                seeds=[int(seed) for seed in seeds],
                baselines=[str(name) for name in baselines],
                primary=primary,
                direction=direction,
                minimum_win_rate=float(minimum_win_rate),
                controller_derivation=controller_derivation,
            )
            for spec in quality_specs
        ]
        relative_values = [
            float(instance["relative_improvement_over_strongest_baseline"])
            for instance in quality_instances
            if _finite_number(instance.get("relative_improvement_over_strongest_baseline"))
        ]
        lower_confidence_bound = _one_sided_lower_confidence_bound_95(relative_values)
        if policy["selection_statistic"] == "worst_instance":
            selection_value = min(relative_values) if relative_values else None
        else:
            selection_value = lower_confidence_bound
        selection_pass = bool(
            len(relative_values) == len(quality_specs)
            and selection_value is not None
            and selection_value >= float(minimum_relative)
            and all(instance["execution_valid"] for instance in quality_instances)
            and all(instance["win_rate_pass"] for instance in quality_instances)
        )
        runtime = _runtime_summary(
            family_rows,
            specs=runtime_specs,
            seeds=[int(seed) for seed in seeds],
            baselines=[str(name) for name in baselines],
            controller_derivation=controller_derivation,
            allowed_seconds=allowed_seconds,
        )
        quality_rows = [
            row
            for row in family_rows
            if row.get("instance_id") in {str(spec["instance_id"]) for spec in quality_specs}
            and row.get("status") == "passed"
        ]
        candidate_values = [
            float(row["candidate_metrics"][primary])
            for row in quality_rows
            if _finite_number(row.get("candidate_metrics", {}).get(primary))
        ]
        strongest_values: list[float] = []
        improvements: list[float] = []
        seed_results: list[dict[str, Any]] = []
        for row in quality_rows:
            values = [
                row.get("baseline_metrics", {}).get(name, {}).get(primary) for name in baselines
            ]
            candidate = row.get("candidate_metrics", {}).get(primary)
            if not _finite_number(candidate) or any(not _finite_number(value) for value in values):
                continue
            numeric = [float(value) for value in values]
            strongest = max(numeric) if direction == "maximize" else min(numeric)
            improvement = (
                float(candidate) - strongest
                if direction == "maximize"
                else strongest - float(candidate)
            )
            strongest_values.append(strongest)
            outcome = comparison_outcome(improvement, row.get("noise_band", 0))
            improvements.append(outcome["improvement"])
            seed_results.append(
                {
                    "instance_id": str(row.get("instance_id") or ""),
                    "instance_seed": int(row.get("instance_seed") or 0),
                    "seed": int(row["seed"]),
                    "candidate": float(candidate),
                    "strongest_baseline": strongest,
                    **outcome,
                }
            )
        mean_improvement = _mean(improvements)
        strongest_mean = _mean(strongest_values)
        aggregate_relative = (
            mean_improvement / abs(strongest_mean)
            if mean_improvement is not None
            and strongest_mean is not None
            and abs(strongest_mean) > 1e-12
            else None
        )
        valid_runs = sum(row.get("status") == "passed" for row in family_rows)
        failed_runs = len(family_rows) - valid_runs
        family_pass = bool(selection_pass and runtime["status"] == "pass")
        families.append(
            {
                "family_id": family_id,
                "status": "pass" if family_pass else "fail",
                "valid_runs": valid_runs,
                "failed_runs": failed_runs,
                "candidate_mean": _mean(candidate_values),
                "candidate_confidence_interval_95": _normal_confidence_interval_95(
                    [
                        instance["candidate_mean"]
                        for instance in quality_instances
                        if _finite_number(instance.get("candidate_mean"))
                    ]
                ),
                "candidate_mean_seconds": _mean(
                    [
                        float(row.get("compute", {}).get("candidate_seconds"))
                        for row in quality_rows
                        if _finite_number(row.get("compute", {}).get("candidate_seconds"))
                    ]
                ),
                "mean_improvement_over_strongest_baseline": mean_improvement,
                "strongest_baseline_mean": strongest_mean,
                "relative_improvement_over_strongest_baseline": aggregate_relative,
                "improvement_confidence_interval_95": _normal_confidence_interval_95(
                    [
                        instance["mean_improvement_over_strongest_baseline"]
                        for instance in quality_instances
                        if _finite_number(instance.get("mean_improvement_over_strongest_baseline"))
                    ]
                ),
                "confidence_interval_repeat_unit": "quality_instance_mean_over_method_seeds",
                "seed_results": seed_results,
                "required_minimum_relative_improvement": float(minimum_relative),
                "required_minimum_win_rate": float(minimum_win_rate),
                "improvement_requirement": "relative",
                "selection_statistic": policy["selection_statistic"],
                "selection_value": selection_value,
                "selection_pass": selection_pass,
                "mean_relative_improvement_over_instances": _mean(relative_values),
                "lower_confidence_bound_95": lower_confidence_bound,
                "relative_improvement_confidence_interval_95": (
                    _normal_confidence_interval_95(relative_values)
                ),
                "quality_instances": quality_instances,
                "runtime_envelope": runtime,
            }
        )
    candidate_validity = [
        _run_validity(
            row,
            baselines=[str(name) for name in baselines],
            controller_derivation=controller_derivation,
        )[0]
        for row in rows
    ]
    baseline_validity = [
        _run_validity(
            row,
            baselines=[str(name) for name in baselines],
            controller_derivation=controller_derivation,
        )[1]
        for row in rows
    ]
    return {
        "schema_version": DEVELOPMENT_QUALIFICATION_RECEIPT_SCHEMA_VERSION,
        "development_qualification_policy": policy,
        "primary_metric": primary,
        "direction": direction,
        "families": families,
        "passed_family_count": sum(row["status"] == "pass" for row in families),
        "runtime_envelope_pass": bool(
            families and all(row["runtime_envelope"]["status"] == "pass" for row in families)
        ),
        "all_candidate_runs_valid": bool(candidate_validity and all(candidate_validity)),
        "all_baseline_runs_valid": bool(baseline_validity and all(baseline_validity)),
        "expected_rows": len(instance_specs) * len(seeds),
        "observed_rows": len(rows),
    }
