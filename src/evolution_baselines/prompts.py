"""Prompts for the LLM arms.  Only public pack material is ever included.

The model sees: the sender's public problem specification, the interface contract, the
three exposed comparator implementations (CN/RA/AA, verbatim from the pack bundle), the
public development families with their baseline profile, and the exact qualification
rule.  It never sees hidden graphs, hidden results, or the operator's full heuristic file.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Any

from .evaluate import CandidateEvaluator, EvaluationResult
from .pack import LinkPredictionPack

DIFF_PATTERN = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
FENCE_PATTERN = r"```(?:python|py)?\s*\n(.*?)```"

SYSTEM_PROMPT = """You are an algorithm designer working on local topological link prediction.
You write one deterministic Python module that defines score_candidates(adj, candidates).
Constraints you must respect exactly:
- Python 3.11 standard library and NumPy only; no other imports, no files, no network, no randomness.
- Use only the observed topology in `adj` (list of neighbor sets) for the candidate pairs.
- Return one finite float per candidate, in candidate order (a NumPy array or list).
- Deterministic: identical input must give identical output on every run.
- Every split must finish well within 60 seconds; graphs have up to a few hundred nodes and
  up to ~11,000 edges, candidate sets up to a few tens of thousands of pairs.
You are judged only by measured Average Precision against classical comparators, so reason about
what structural signal the comparators miss and test that idea, rather than restating known heuristics.
"""


@dataclass
class PromptContext:
    spec_text: str
    interface_text: str
    reference_code: dict[str, str]
    family_profile: list[dict[str, Any]]
    minimum_relative_improvement: float
    required_family_count: int
    family_count: int
    quality_splits_per_family: int

    @classmethod
    def from_pack(cls, pack: LinkPredictionPack, evaluator: CandidateEvaluator) -> "PromptContext":
        reference = pack.public_reference_dir()
        spec = (reference / "problem_spec.md").read_text()
        reference_code = {name: pack.baseline_source(name) for name in pack.baseline_ids}
        baselines = evaluator.baseline_metrics("public")
        profile: list[dict[str, Any]] = []
        primary = pack.primary_metric
        for family_id in dict.fromkeys(spec_.family_id for spec_, _f, _p in evaluator.instances("public")):
            quality = [(s, full) for s, full, _p in evaluator.instances("public")
                       if s.family_id == family_id and s.role == "quality"]
            nodes = sorted({int(full["node_count"]) for _s, full in quality})
            edges = sorted({len(full["train_edges"]) for _s, full in quality})
            candidates = sorted({len(full["candidates"]) for _s, full in quality})
            means = {name: statistics.fmean(float(baselines[s.key][name][primary]) for s, _f in quality)
                     for name in pack.baseline_ids}
            strongest_per_split = [max(pack.baseline_ids, key=lambda n: float(baselines[s.key][n][primary]))
                                   for s, _f in quality]
            profile.append({
                "family_id": family_id, "splits": len(quality), "nodes": nodes[0] if nodes else None,
                "observed_edges": f"{edges[0]}-{edges[-1]}" if edges else None,
                "candidates": f"{candidates[0]}-{candidates[-1]}" if candidates else None,
                "baseline_mean_ap": {k: round(v, 4) for k, v in means.items()},
                "strongest_baseline_by_split": {name: strongest_per_split.count(name)
                                                for name in pack.baseline_ids},
            })
        interface = pack.public_contract["interface"]["description"]
        return cls(spec_text=spec, interface_text=interface, reference_code=reference_code,
                   family_profile=profile,
                   minimum_relative_improvement=pack.minimum_relative_improvement,
                   required_family_count=pack.required_family_count(), family_count=len(profile),
                   quality_splits_per_family=max((p["splits"] for p in profile), default=8))

    # --------------------------------------------------------------- sections
    def task_section(self) -> str:
        rule = (
            f"Qualification rule (measured by the controller, not by you): for each public family, on "
            f"every one of its {self.quality_splits_per_family} splits your AP must be strictly higher "
            f"than the best of CN, RA and AA on that same split, and the one-sided 95% lower confidence "
            f"bound of the relative AP improvement over that per-split strongest comparator must be at least "
            f"{self.minimum_relative_improvement:.0%}. At least {self.required_family_count} of "
            f"{self.family_count} families must pass. Hidden graphs of different kinds are evaluated "
            f"afterwards, so prefer mechanisms that generalize over per-graph tuning."
        )
        profile_lines = ["| family | splits | nodes | observed edges | candidates | mean AP cn / ra / aa | strongest per split |",
                         "| --- | --- | --- | --- | --- | --- | --- |"]
        for item in self.family_profile:
            means = item["baseline_mean_ap"]
            profile_lines.append(
                f"| {item['family_id']} | {item['splits']} | {item['nodes']} | {item['observed_edges']} | "
                f"{item['candidates']} | {means.get('cn')} / {means.get('ra')} / {means.get('aa')} | "
                f"{item['strongest_baseline_by_split']} |")
        code = "\n\n".join(f"### {name}\n```python\n{source.strip()}\n```"
                           for name, source in self.reference_code.items())
        return (
            "## Problem specification (verbatim)\n\n" + self.spec_text.strip() + "\n\n"
            "## Interface contract\n\n" + self.interface_text.strip() + "\n\n"
            "## Public development families and comparator profile\n\n" + "\n".join(profile_lines)
            + "\n\n" + rule + "\n\n## Exposed comparator implementations (verbatim)\n\n" + code + "\n"
        )

    def output_section(self, *, diff_based: bool) -> str:
        if diff_based:
            return (
                "## Output format\n\nFirst give a 1-3 sentence rationale for ONE focused change. Then give the change "
                "as one or more SEARCH/REPLACE blocks against the current program, exactly in this form:\n\n"
                "<<<<<<< SEARCH\n<exact lines copied from the current program>\n=======\n<replacement lines>\n"
                ">>>>>>> REPLACE\n\nEach SEARCH text must appear exactly once in the current program. If the change is "
                "large, you may instead return the complete new program in a single ```python fenced block that "
                "defines score_candidates.\n"
            )
        return (
            "## Output format\n\nGive a 1-3 sentence rationale, then the complete program in a single ```python "
            "fenced block. The block must define score_candidates(adj, candidates) and may import only the "
            "standard library and numpy.\n"
        )


def initial_prompt(context: PromptContext) -> str:
    return (
        context.task_section()
        + "\n## Your task\n\nWrite a new score_candidates that beats the per-split strongest comparator by the "
        "required margin on at least the required number of families. Start from a concrete structural "
        "hypothesis about which candidate pairs the comparators mis-rank.\n\n"
        + context.output_section(diff_based=False)
    )


def _feedback_block(title: str, result_summary: dict[str, Any], per_instance: list[dict[str, Any]] | None) -> str:
    lines = [f"### {title}",
             f"fitness (mean relative AP improvement over per-split strongest comparator): "
             f"{result_summary['fitness']:+.4f}; families passed: {result_summary['families_passed']}"
             f"/{result_summary['required_family_count']} required; valid: {result_summary['valid']}"]
    if result_summary.get("errors"):
        lines.append("errors: " + " | ".join(str(e)[:300] for e in result_summary["errors"][:3]))
    lines.append("| family | status | mean AP | strongest mean AP | mean rel. improvement | 95% LCB | W/T/L |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for fam in result_summary.get("families", []):
        def fmt(value: Any, spec: str = ".4f") -> str:
            return format(value, spec) if isinstance(value, (int, float)) else "n/a"
        lines.append(
            f"| {fam['family_id']} | {fam['status']} | {fmt(fam['candidate_mean'])} | "
            f"{fmt(fam['strongest_baseline_mean'])} | {fmt(fam['mean_relative_improvement'], '+.2%')} | "
            f"{fmt(fam['lower_confidence_bound_95'], '+.2%')} | {fam['wins']}/{fam['ties']}/{fam['losses']} |")
    if per_instance:
        lines.append("")
        lines.append("Per split (candidate AP vs strongest comparator AP):")
        for row in per_instance:
            if row.get("role") != "quality":
                continue
            cand = row.get("average_precision")
            best = row.get("strongest_baseline")
            if isinstance(cand, (int, float)) and isinstance(best, (int, float)):
                delta = cand - best
                lines.append(f"- {row['instance_id']}: {cand:.4f} vs {best:.4f} ({delta:+.4f})")
            else:
                lines.append(f"- {row['instance_id']}: failed ({str(row.get('error'))[:120]})")
    return "\n".join(lines)


def mutation_prompt(
    context: PromptContext,
    *,
    parent_source: str,
    parent_feedback: dict[str, Any],
    parent_per_instance: list[dict[str, Any]] | None,
    top_programs: list[tuple[str, dict[str, Any]]],
    inspirations: list[tuple[str, dict[str, Any]]],
    diff_based: bool,
    max_program_chars: int = 6000,
) -> str:
    def clip(source: str) -> str:
        return source if len(source) <= max_program_chars else source[:max_program_chars] + "\n# ... truncated ..."

    parts = [context.task_section(), "\n## Current program\n\n```python\n" + parent_source.strip() + "\n```\n",
             _feedback_block("Evaluation of the current program", parent_feedback, parent_per_instance)]
    if top_programs:
        parts.append("\n## Best programs found so far (for reference; do not copy blindly)\n")
        for source, summary in top_programs:
            parts.append(f"fitness {summary['fitness']:+.4f}, families passed {summary['families_passed']}:\n"
                         f"```python\n{clip(source).strip()}\n```\n")
    if inspirations:
        parts.append("\n## Other explored programs (different regions of the search space)\n")
        for source, summary in inspirations:
            parts.append(f"fitness {summary['fitness']:+.4f}, families passed {summary['families_passed']}:\n"
                         f"```python\n{clip(source).strip()}\n```\n")
    parts.append(
        "\n## Your task\n\nImprove the current program. Use the per-split feedback to decide which mechanism "
        "to change; a family that fails only on the confidence bound needs a more consistent gain, a family "
        "that loses splits needs a different signal. Keep the module deterministic and within the import "
        "constraints.\n\n" + context.output_section(diff_based=diff_based))
    return "\n".join(parts)


@dataclass
class ParsedProgram:
    source: str | None
    mode: str
    error: str | None
    rationale: str


def _rationale(text: str) -> str:
    head = re.split(r"<<<<<<< SEARCH|```", text, maxsplit=1)[0].strip()
    return head[:600]


def parse_program(text: str, *, parent_source: str | None, diff_based: bool) -> ParsedProgram:
    """Apply SEARCH/REPLACE blocks (OpenEvolve's format) or extract a full program."""
    rationale = _rationale(text)
    blocks = re.findall(DIFF_PATTERN, text, re.DOTALL) if diff_based and parent_source else []
    if blocks:
        source = parent_source or ""
        for search, replace in blocks:
            occurrences = source.count(search)
            if occurrences != 1:
                # OpenEvolve also tries the rstripped form before failing.
                stripped = search.rstrip("\n")
                if stripped and source.count(stripped) == 1:
                    source = source.replace(stripped, replace.rstrip("\n"), 1)
                    continue
                return ParsedProgram(None, "diff",
                                     f"SEARCH block matched {occurrences} times (must be exactly 1)", rationale)
            source = source.replace(search, replace, 1)
        if "def score_candidates" not in source:
            return ParsedProgram(None, "diff", "patched program lost score_candidates", rationale)
        return ParsedProgram(source, "diff", None, rationale)
    fenced = [block for block in re.findall(FENCE_PATTERN, text, re.DOTALL) if "def score_candidates" in block]
    if fenced:
        return ParsedProgram(fenced[-1].strip("\n") + "\n", "full", None, rationale)
    if "def score_candidates" in text and "```" not in text:
        return ParsedProgram(text.strip("\n") + "\n", "full", None, rationale)
    return ParsedProgram(None, "none", "no SEARCH/REPLACE block or fenced program defining score_candidates",
                         rationale)


def feedback_payload(result: EvaluationResult) -> dict[str, Any]:
    """The subset of an evaluation that is fed back to the model (public rows only)."""
    return result.summary()


__all__ = ["DIFF_PATTERN", "ParsedProgram", "PromptContext", "SYSTEM_PROMPT", "feedback_payload",
           "initial_prompt", "mutation_prompt", "parse_program"]
