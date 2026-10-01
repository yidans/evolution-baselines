"""The baseline arms: classical heuristics, best-of-N sampling, OpenEvolve-style evolution.

All arms share :class:`RunRecorder` so their outputs have one shape:

``programs/<id>.py`` every evaluated program · ``evaluations.jsonl`` one line per evaluation ·
``iterations.jsonl`` one line per search step · ``best/`` frozen bundle of the selected program
with ``public_qualification.json`` (a local screening receipt) · ``run_manifest.json`` config,
budget usage and outcome · ``summary.md`` a human table.
"""
from __future__ import annotations

import json
import random
import re
import statistics
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from importlib.metadata import PackageNotFoundError, version

from ._io import utc_now

from .evaluate import CandidateEvaluator, EvaluationResult, program_id, write_frozen_bundle
from .llm import BudgetExhausted, LedgerClient
from .pack import LinkPredictionPack
from .prompts import SYSTEM_PROMPT, PromptContext, initial_prompt, mutation_prompt, parse_program

CN_SEED_PROGRAM = '''"""Common Neighbors: the minimal exposed baseline (seed program)."""
import numpy as np


def score_candidates(adj, candidates):
    return np.asarray([len(adj[u] & adj[v]) for u, v in candidates], dtype=float)
'''


@dataclass
class Program:
    id: str
    source: str
    origin: str
    iteration: int
    island: int
    parent_id: str | None
    fitness: float
    families_passed: int
    qualified: bool
    valid: bool
    llm_call: int | None = None
    parse_mode: str | None = None
    rationale: str = ""
    feedback: dict[str, Any] = field(default_factory=dict)
    per_instance: list[dict[str, Any]] = field(default_factory=list)

    def rank_key(self) -> tuple[int, int, float]:
        return (int(self.qualified), self.families_passed, self.fitness)


def _sort_key(program: Program) -> tuple[int, int, float, str]:
    return (*program.rank_key(), program.id)


def _environment(backend: str) -> dict[str, Any]:
    """Record normal package/Git versions alongside the measured run."""
    result: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for package in ("numpy", "evolution-baselines", "hermes-ai-professor"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    roots = {"baseline": Path(__file__).resolve().parents[2]}
    if backend == "galahad":
        from ai_professor.autonomy import qualification
        from ai_professor.autonomy.isolation import _configured_evaluator_image, _selected_process_sandbox
        roots["galahad"] = Path(qualification.__file__).resolve().parents[3]
        result.update(sandbox=_selected_process_sandbox(), evaluator_image=_configured_evaluator_image())
    for name, root in roots.items():
        if not (root / ".git").exists():
            continue
        try:
            revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                      capture_output=True, text=True, check=True, timeout=5)
            changes = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                     capture_output=True, text=True, check=True, timeout=5)
            result[name + "_git"] = {"revision": revision.stdout.strip(), "dirty": bool(changes.stdout)}
        except (OSError, subprocess.SubprocessError):
            result[name + "_git"] = None
    return result


class RunRecorder:
    """Uniform on-disk record for every arm."""

    def __init__(self, out_dir: Path, *, arm: str, pack: LinkPredictionPack, config: dict[str, Any]) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if any((self.out_dir / name).exists() for name in
               ("run_manifest.json", "evaluations.jsonl", "iterations.jsonl", "best")):
            raise FileExistsError(f"run already has results: {self.out_dir}; use a new output directory")
        (self.out_dir / "programs").mkdir(exist_ok=True)
        self.arm = arm
        self.pack = pack
        self.config = config
        self.environment = _environment(config.get("evaluator", "subprocess"))
        self.started = time.perf_counter()
        self.started_at = utc_now()
        self.evaluations = 0
        self.results: dict[str, EvaluationResult] = {}
        self._manifest_extra: dict[str, Any] = {}

    def record_program(self, program: Program, result: EvaluationResult | None) -> None:
        (self.out_dir / "programs" / f"{program.id}.py").write_text(program.source)
        if result is not None:
            self.results[program.id] = result
            self.evaluations += 1
            (self.out_dir / "programs" / f"{program.id}.evaluation.json").write_text(
                json.dumps(result.to_json(), indent=1, allow_nan=False, default=_json_default) + "\n")
        entry = {k: v for k, v in asdict(program).items() if k not in {"source", "per_instance", "feedback"}}
        entry["families"] = program.feedback.get("families") if program.feedback else None
        entry["errors"] = program.feedback.get("errors") if program.feedback else None
        with (self.out_dir / "evaluations.jsonl").open("a") as handle:
            handle.write(json.dumps(entry, allow_nan=False, default=_json_default) + "\n")

    def record_iteration(self, payload: dict[str, Any]) -> None:
        with (self.out_dir / "iterations.jsonl").open("a") as handle:
            handle.write(json.dumps(payload, allow_nan=False, default=_json_default) + "\n")

    def extra(self, **values: Any) -> None:
        self._manifest_extra.update(values)

    def finish(self, *, best: Program | None, programs: list[Program], client: LedgerClient | None,
               outcome: dict[str, Any]) -> dict[str, Any]:
        selected = self.results.get(best.id) if best is not None else None
        claim_status = selected.claim_status if selected else "operator_diagnostic_only"
        trusted_isolation = selected.trusted_isolation if selected else False
        if best is not None:
            bundle = write_frozen_bundle(best.source, self.out_dir / "best",
                                         metadata={"arm": self.arm, "origin": best.origin,
                                                   "iteration": best.iteration, "parent_id": best.parent_id})
            result = self.results.get(best.id)
            if result is not None:
                (bundle / "public_qualification.json").write_text(
                    json.dumps({"program_id": best.id, "pack": self.pack.contract_id,
                                "claim_status": claim_status,
                                "trusted_isolation": trusted_isolation, "qualified": result.qualified,
                                "families_passed": result.families_passed,
                                "required_family_count": result.required_family_count,
                                "fitness": result.fitness, "summary": result.summary(),
                                "receipt": result.receipt}, indent=1, allow_nan=False,
                               default=_json_default) + "\n")
        manifest = {
            "schema_version": "linkpred-baseline-run.v1", "arm": self.arm, "pack": self.pack.contract_id,
            "claim_status": claim_status, "trusted_isolation": trusted_isolation,
            "pack_root": str(self.pack.root), "started_at": self.started_at, "finished_at": utc_now(),
            "seconds": time.perf_counter() - self.started, "config": self.config,
            "environment": self.environment,
            "evaluations": self.evaluations, "programs": len(programs),
            "client": client.usage() if client is not None else None,
            "best": None if best is None else {
                "program_id": best.id, "origin": best.origin, "iteration": best.iteration,
                "fitness": best.fitness, "families_passed": best.families_passed, "qualified": best.qualified},
            "outcome": outcome, **self._manifest_extra,
        }
        (self.out_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=1, allow_nan=False, default=_json_default) + "\n")
        (self.out_dir / "summary.md").write_text(_summary_markdown(manifest, programs))
        return manifest


def _json_default(value: Any) -> Any:
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _summary_markdown(manifest: dict[str, Any], programs: list[Program]) -> str:
    lines = [f"# {manifest['arm']} on {manifest['pack']}", "",
             f"Started {manifest['started_at']}, finished {manifest['finished_at']} "
             f"({manifest['seconds']:.0f} s), {manifest['evaluations']} evaluations, "
             f"{manifest['programs']} programs."]
    client = manifest.get("client")
    if client:
        lines.append(f"Model calls: {client['calls_used']}/{client['max_calls']} "
                     f"({client['physical_calls']} physical, {client['cached_calls']} cached); "
                     f"tokens in/out {client['prompt_tokens']}/{client['completion_tokens']}.")
    lines.append("")
    lines.append(f"Outcome: `{json.dumps(manifest['outcome'], default=_json_default)}`")
    lines.append("")
    lines.append("| rank | program | origin | iter | island | fitness | families | qualified | families detail |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    ordered = sorted(programs, key=_sort_key, reverse=True)
    for rank, program in enumerate(ordered[:25], 1):
        detail = "; ".join(
            f"{fam['family_id']}:{fam['status']} lcb={fam['lower_confidence_bound_95']:+.2%} "
            f"{fam['wins']}/{fam['ties']}/{fam['losses']}"
            if isinstance(fam.get("lower_confidence_bound_95"), (int, float)) else
            f"{fam['family_id']}:{fam['status']}"
            for fam in (program.feedback.get("families") or []))
        lines.append(f"| {rank} | {program.id} | {program.origin} | {program.iteration} | {program.island} | "
                     f"{program.fitness:+.4f} | {program.families_passed} | {program.qualified} | {detail} |")
    return "\n".join(lines) + "\n"


def _program_from_result(source: str, result: EvaluationResult, *, origin: str, iteration: int, island: int,
                         parent_id: str | None, llm_call: int | None, parse_mode: str | None,
                         rationale: str) -> Program:
    return Program(id=result.program_id, source=source, origin=origin, iteration=iteration, island=island,
                   parent_id=parent_id, fitness=result.fitness, families_passed=result.families_passed,
                   qualified=result.qualified, valid=result.valid, llm_call=llm_call, parse_mode=parse_mode,
                   rationale=rationale, feedback=result.summary(), per_instance=result.per_instance)


# =========================================================================== classical
CLASSICAL_HEURISTICS = ("cn", "aa", "ra", "jaccard", "pa", "car")


def classical_sources(pack: LinkPredictionPack, names: tuple[str, ...] = CLASSICAL_HEURISTICS) -> dict[str, str]:
    """Each sender heuristic as a standalone solution.py (operator file + ``score_candidates = f``)."""
    operator = pack.operator_heuristics_source()
    functions = {"cn": "common_neighbors", "aa": "adamic_adar", "ra": "resource_allocation",
                 "jaccard": "jaccard", "pa": "preferential_attachment", "car": "car"}
    sources = {}
    for name in names:
        function = functions.get(name)
        if function is None or not re.search(rf"^def {function}\(", operator, re.MULTILINE):
            raise KeyError(f"heuristic {name!r} is not defined in operator_sources/baselines.py")
        sources[name] = operator.rstrip("\n") + f"\n\nscore_candidates = {function}\n"
    return sources


def run_classical(pack: LinkPredictionPack, evaluator: CandidateEvaluator, out_dir: Path, *,
                  heuristics: tuple[str, ...] = CLASSICAL_HEURISTICS,
                  extra_solutions: dict[str, str] | None = None,
                  heldout: bool = False) -> dict[str, Any]:
    """Zero-model-call floor: every classical heuristic through the same qualification."""
    recorder = RunRecorder(out_dir, arm="classical", pack=pack,
                           config={"heuristics": list(heuristics), "extra": sorted(extra_solutions or {}),
                                   "heldout": heldout, "evaluator": evaluator.backend})
    sources = classical_sources(pack, heuristics)
    sources.update(extra_solutions or {})
    programs: list[Program] = []
    table: dict[str, Any] = {}
    for name, source in sources.items():
        result = evaluator.evaluate(source, label=name)
        program = _program_from_result(source, result, origin=f"classical:{name}", iteration=0, island=0,
                                       parent_id=None, llm_call=None, parse_mode=None, rationale="")
        recorder.record_program(program, result)
        programs.append(program)
        entry: dict[str, Any] = {"program_id": program.id, "public": result.summary()}
        if heldout:
            held = evaluator.heldout(source, label=name)
            entry["heldout"] = {k: v for k, v in (held.heldout or {}).items() if k != "rows"}
            (recorder.out_dir / "programs" / f"{program.id}.heldout.json").write_text(
                json.dumps(held.heldout, indent=1, allow_nan=False, default=_json_default) + "\n")
        table[name] = entry
        recorder.record_iteration({"heuristic": name, "program_id": program.id, "fitness": program.fitness,
                                   "families_passed": program.families_passed, "qualified": program.qualified})
    best = max(programs, key=_sort_key)
    (recorder.out_dir / "classical_table.json").write_text(
        json.dumps(table, indent=1, allow_nan=False, default=_json_default) + "\n")
    return recorder.finish(best=best, programs=programs, client=None,
                           outcome={"any_qualified": any(p.qualified for p in programs),
                                    "qualified": [p.origin for p in programs if p.qualified],
                                    "heldout_diagnostic": heldout,
                                    "note": "classical arm; hidden rows are an operator-side diagnostic, "
                                            "not a held-out claim"})


# ========================================================================== best of N
def run_best_of_n(pack: LinkPredictionPack, evaluator: CandidateEvaluator, client: LedgerClient,
                  out_dir: Path, *, samples: int, seed: int = 0, stop_on_qualify: bool = True) -> dict[str, Any]:
    """Independent samples from the initial prompt; no feedback, no evolution."""
    if samples < 0:
        raise ValueError("samples must be nonnegative")
    config = {"samples": samples, "seed": seed, "stop_on_qualify": stop_on_qualify,
              "evaluator": evaluator.backend,
              "max_llm_calls": client.max_calls}
    recorder = RunRecorder(out_dir, arm="best_of_n", pack=pack, config=config)
    context = PromptContext.from_pack(pack, evaluator)
    system = SYSTEM_PROMPT
    programs: list[Program] = []
    seen: dict[str, Program] = {}
    first_qualified: int | None = None
    parse_failures = 0
    stop_reason = "samples_exhausted"
    for index in range(samples):
        # A per-sample nonce keeps the ledger cache from collapsing N samples into one call.
        user = initial_prompt(context) + f"\n<!-- sample {index} seed {seed} -->\n"
        try:
            response = client.complete(system=system, user=user, tag=f"best_of_n:{index}")
        except BudgetExhausted as exc:
            recorder.record_iteration({"sample": index, "event": "budget_exhausted", "detail": str(exc)})
            stop_reason = "budget_exhausted"
            break
        except Exception as exc:
            recorder.record_iteration({"sample": index, "event": "client_error", "error_type": type(exc).__name__})
            stop_reason = "client_error"
            break
        parsed = parse_program(response.text, parent_source=None, diff_based=False)
        if parsed.source is None:
            parse_failures += 1
            recorder.record_iteration({"sample": index, "event": "parse_failure", "error": parsed.error,
                                       "llm_call": client.calls_used})
            continue
        pid = program_id(parsed.source)
        if pid in seen:
            recorder.record_iteration({"sample": index, "event": "duplicate", "program_id": pid,
                                       "llm_call": client.calls_used})
            continue
        result = evaluator.evaluate(parsed.source, label=f"bon{index}")
        program = _program_from_result(parsed.source, result, origin="best_of_n", iteration=index, island=0,
                                       parent_id=None, llm_call=client.calls_used, parse_mode=parsed.mode,
                                       rationale=parsed.rationale)
        seen[pid] = program
        programs.append(program)
        recorder.record_program(program, result)
        recorder.record_iteration({"sample": index, "event": "evaluated", "program_id": pid,
                                   "fitness": program.fitness, "families_passed": program.families_passed,
                                   "qualified": program.qualified, "llm_call": client.calls_used,
                                   "errors": result.errors[:2]})
        if program.qualified and first_qualified is None:
            first_qualified = index
            if stop_on_qualify:
                stop_reason = "qualified"
                break
    best = max(programs, key=_sort_key) if programs else None
    return recorder.finish(best=best, programs=programs, client=client,
                           outcome={"stop_reason": stop_reason, "first_qualified_sample": first_qualified,
                                    "qualified": best is not None and best.qualified,
                                    "parse_failures": parse_failures, "samples_evaluated": len(programs)})


# ============================================================================ evolve
@dataclass
class EvolveConfig:
    max_llm_calls: int = 600
    max_iterations: int | None = None
    islands: int = 3
    migration_interval: int = 10
    num_top_programs: int = 3
    num_diverse_programs: int = 2
    exploitation_ratio: float = 0.7
    exploration_ratio: float = 0.2
    diff_based: bool = True
    seed: int = 0
    stop_on_qualify: bool = True
    complexity_bins: tuple[int, ...] = (15, 30, 60)
    seed_program: str = CN_SEED_PROGRAM

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["seed_program_id"] = program_id(self.seed_program)
        del payload["seed_program"]
        return payload


class Island:
    """A MAP-Elites grid keyed by (families_passed, complexity bin); one program per cell."""

    def __init__(self, index: int, bins: tuple[int, ...]) -> None:
        self.index = index
        self.bins = bins
        self.cells: dict[tuple[int, int], Program] = {}

    def cell(self, program: Program) -> tuple[int, int]:
        lines = sum(1 for line in program.source.splitlines() if line.strip() and not line.strip().startswith("#"))
        complexity = sum(1 for edge in self.bins if lines > edge)
        return (program.families_passed if program.valid else -1, complexity)

    def add(self, program: Program) -> bool:
        key = self.cell(program)
        incumbent = self.cells.get(key)
        if incumbent is None or program.rank_key() > incumbent.rank_key():
            self.cells[key] = program
            return True
        return False

    def members(self) -> list[Program]:
        return sorted(self.cells.values(), key=_sort_key, reverse=True)


def select_parent(island: Island, rng: random.Random, config: EvolveConfig) -> Program:
    members = island.members()
    if not members:
        raise RuntimeError("island has no programs")
    draw = rng.random()
    if draw < config.exploitation_ratio:
        weights = [1.0 / (rank + 1) for rank in range(len(members))]
        return rng.choices(members, weights=weights, k=1)[0]
    if draw < config.exploitation_ratio + config.exploration_ratio:
        return rng.choice(members)
    return members[0]


def run_evolve(pack: LinkPredictionPack, evaluator: CandidateEvaluator, client: LedgerClient, out_dir: Path,
               config: EvolveConfig) -> dict[str, Any]:
    """OpenEvolve-style loop: islands, MAP-Elites archive, diff mutations, migration, feedback."""
    if config.islands < 1 or config.migration_interval < 0:
        raise ValueError("islands must be positive and migration_interval nonnegative")
    recorder = RunRecorder(out_dir, arm="evolve", pack=pack,
                           config={**config.as_dict(), "max_llm_calls": client.max_calls,
                                   "evaluator": evaluator.backend})
    context = PromptContext.from_pack(pack, evaluator)
    rng = random.Random(config.seed)
    islands = [Island(i, config.complexity_bins) for i in range(config.islands)]
    seed_result = evaluator.evaluate(config.seed_program, label="seed")
    seed_program = _program_from_result(config.seed_program, seed_result, origin="seed", iteration=0, island=-1,
                                        parent_id=None, llm_call=None, parse_mode=None, rationale="")
    recorder.record_program(seed_program, seed_result)
    for island in islands:
        island.add(seed_program)
    programs: list[Program] = [seed_program]
    by_id: dict[str, Program] = {seed_program.id: seed_program}
    first_qualified: int | None = None
    parse_failures = 0
    no_change = 0
    duplicates = 0
    iteration = 0
    stop_reason = "max_iterations"
    while config.max_iterations is None or iteration < config.max_iterations:
        if client.remaining <= 0:
            stop_reason = "budget_exhausted"
            break
        iteration += 1
        island = islands[(iteration - 1) % len(islands)]
        parent = select_parent(island, rng, config)
        ranked = sorted(by_id.values(), key=_sort_key, reverse=True)
        top = [p for p in ranked if p.id != parent.id][: config.num_top_programs]
        pool = [p for p in ranked if p.id != parent.id and p not in top]
        inspirations = rng.sample(pool, k=min(config.num_diverse_programs, len(pool))) if pool else []
        user = mutation_prompt(context, parent_source=parent.source, parent_feedback=parent.feedback,
                               parent_per_instance=parent.per_instance,
                               top_programs=[(p.source, p.feedback) for p in top],
                               inspirations=[(p.source, p.feedback) for p in inspirations],
                               diff_based=config.diff_based)
        user += f"\n<!-- iteration {iteration} island {island.index} parent {parent.id} -->\n"
        try:
            response = client.complete(system=SYSTEM_PROMPT, user=user,
                                       tag=f"evolve:{iteration}:island{island.index}")
        except BudgetExhausted:
            stop_reason = "budget_exhausted"
            break
        except Exception as exc:
            recorder.record_iteration({"iteration": iteration, "event": "client_error",
                                       "error_type": type(exc).__name__})
            stop_reason = "client_error"
            break
        parsed = parse_program(response.text, parent_source=parent.source, diff_based=config.diff_based)
        event: dict[str, Any] = {"iteration": iteration, "island": island.index, "parent_id": parent.id,
                                 "parent_fitness": parent.fitness, "llm_call": client.calls_used,
                                 "parse_mode": parsed.mode}
        if parsed.source is None:
            parse_failures += 1
            event.update({"event": "parse_failure", "error": parsed.error})
            recorder.record_iteration(event)
            continue
        if parsed.source.strip() == parent.source.strip():
            no_change += 1
            event.update({"event": "no_change", "program_id": parent.id})
            recorder.record_iteration(event)
            continue
        pid = program_id(parsed.source)
        if pid in by_id:
            duplicates += 1
            child = by_id[pid]
            event.update({"event": "duplicate", "program_id": pid})
            island.add(child)
            recorder.record_iteration(event)
            continue
        result = evaluator.evaluate(parsed.source, label=f"it{iteration}")
        child = _program_from_result(parsed.source, result, origin="mutation", iteration=iteration,
                                     island=island.index, parent_id=parent.id, llm_call=client.calls_used,
                                     parse_mode=parsed.mode, rationale=parsed.rationale)
        by_id[pid] = child
        programs.append(child)
        recorder.record_program(child, result)
        inserted = island.add(child)
        event.update({"event": "evaluated", "program_id": pid, "fitness": child.fitness,
                      "families_passed": child.families_passed, "qualified": child.qualified,
                      "improved_parent": child.fitness > parent.fitness, "archived": inserted,
                      "cell": list(island.cell(child)), "errors": result.errors[:2],
                      "rationale": parsed.rationale[:300]})
        recorder.record_iteration(event)
        if child.qualified and first_qualified is None:
            first_qualified = iteration
            if config.stop_on_qualify:
                stop_reason = "qualified"
                break
        if config.migration_interval and iteration % config.migration_interval == 0 and len(islands) > 1:
            migrants = []
            for source_island in islands:
                best_local = source_island.members()[0]
                target = islands[(source_island.index + 1) % len(islands)]
                if target.add(best_local):
                    migrants.append({"from": source_island.index, "to": target.index, "program_id": best_local.id})
            recorder.record_iteration({"iteration": iteration, "event": "migration", "migrants": migrants})
    best = max(programs, key=_sort_key)
    archive = [{"island": island.index, "cell": list(key), "program_id": program.id, "fitness": program.fitness,
                "families_passed": program.families_passed, "qualified": program.qualified}
               for island in islands for key, program in sorted(island.cells.items())]
    (recorder.out_dir / "archive.json").write_text(json.dumps(archive, indent=1, default=_json_default) + "\n")
    fitness_trace = [p.fitness for p in programs if p.valid]
    return recorder.finish(best=best, programs=programs, client=client,
                           outcome={"stop_reason": stop_reason, "iterations": iteration,
                                    "first_qualified_iteration": first_qualified,
                                    "qualified": best.qualified, "parse_failures": parse_failures,
                                    "no_change": no_change, "duplicates": duplicates,
                                    "best_fitness": best.fitness,
                                    "seed_fitness": seed_program.fitness,
                                    "mean_valid_fitness": statistics.fmean(fitness_trace) if fitness_trace else None})


__all__ = ["CLASSICAL_HEURISTICS", "CN_SEED_PROGRAM", "EvolveConfig", "Island", "Program", "RunRecorder",
           "classical_sources", "run_best_of_n", "run_classical", "run_evolve", "select_parent"]
