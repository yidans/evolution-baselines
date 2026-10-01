"""Run comparison arms, public controller evaluation and historical diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .arms import CLASSICAL_HEURISTICS, EvolveConfig, run_best_of_n, run_classical, run_evolve
from .evaluate import CandidateEvaluator, program_id
from .llm import build_client
from .openevolve_export import export_openevolve_project
from .pack import LinkPredictionPack

DEFAULT_PACK = Path(__file__).resolve().parent / "benchmarks" / "local_link_prediction_v2"


def _pack(args: argparse.Namespace) -> LinkPredictionPack:
    return LinkPredictionPack(Path(args.pack))


def _evaluator(pack: LinkPredictionPack, out: Path, backend: str = "subprocess") -> CandidateEvaluator:
    if backend == "galahad":
        from .galahad import GalahadEvaluator
        return GalahadEvaluator(pack, out / "evaluator_work")
    return CandidateEvaluator(pack, out / "evaluator_work")


def _client(args: argparse.Namespace, out: Path, max_calls: int):
    scripted = None
    if args.client == "scripted":
        if not args.scripted_file:
            raise SystemExit("--scripted-file is required with --client scripted")
        payload = json.loads(Path(args.scripted_file).read_text())
        scripted = payload if isinstance(payload, list) else payload["responses"]
    return build_client(args.client, out_dir=out, max_calls=max_calls, model=args.model,
                        reasoning_effort=args.reasoning_effort, scripted=scripted,
                        max_output_tokens=args.max_output_tokens,
                        cache_path=Path(args.cache) if args.cache else None)


def _add_client_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--client", choices=["azure", "scripted", "replay"], default="azure")
    parser.add_argument("--model", default=None, help="Azure deployment name (default: AZURE_OPENAI_MODEL)")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--max-output-tokens", type=int, default=16000)
    parser.add_argument("--scripted-file", default=None, help="JSON list of canned responses")
    parser.add_argument("--cache", default=None, help="replay cache path (default: <out>/llm_cache.jsonl)")


def cmd_classical(args: argparse.Namespace) -> int:
    pack = _pack(args)
    out = Path(args.out)
    extra = {}
    for item in args.solution or []:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit("--solution expects name=path")
        extra[name] = Path(path).read_text()
    heuristics = tuple(args.heuristics.split(",")) if args.heuristics else CLASSICAL_HEURISTICS
    manifest = run_classical(pack, _evaluator(pack, out, args.evaluator), out, heuristics=heuristics, extra_solutions=extra,
                             heldout=args.heldout)
    print(json.dumps({k: manifest[k] for k in ("arm", "evaluations", "best", "outcome")}, indent=1))
    return 0


def cmd_best_of_n(args: argparse.Namespace) -> int:
    pack = _pack(args)
    out = Path(args.out)
    evaluator = _evaluator(pack, out, args.evaluator)
    client = _client(args, out, args.samples if args.max_calls is None else args.max_calls)
    manifest = run_best_of_n(pack, evaluator, client, out, samples=args.samples, seed=args.seed,
                             stop_on_qualify=not args.no_stop_on_qualify)
    print(json.dumps({k: manifest[k] for k in ("arm", "evaluations", "client", "best", "outcome")}, indent=1))
    return 1 if manifest["outcome"].get("stop_reason") == "client_error" else 0


def cmd_evolve(args: argparse.Namespace) -> int:
    pack = _pack(args)
    out = Path(args.out)
    evaluator = _evaluator(pack, out, args.evaluator)
    client = _client(args, out, args.max_calls)
    config = EvolveConfig(max_llm_calls=args.max_calls, max_iterations=args.max_iterations, islands=args.islands,
                          migration_interval=args.migration_interval, diff_based=not args.full_rewrite,
                          seed=args.seed, stop_on_qualify=not args.no_stop_on_qualify,
                          num_top_programs=args.top_programs, num_diverse_programs=args.diverse_programs)
    if args.seed_program:
        config.seed_program = Path(args.seed_program).read_text()
    manifest = run_evolve(pack, evaluator, client, out, config)
    print(json.dumps({k: manifest[k] for k in ("arm", "evaluations", "client", "best", "outcome")}, indent=1))
    return 1 if manifest["outcome"].get("stop_reason") == "client_error" else 0


def cmd_heldout(args: argparse.Namespace) -> int:
    pack = _pack(args)
    if args.run:
        run = Path(args.run)
        solution = run / "best" / "solution.py"
        receipt_path = run / "best" / "public_qualification.json"
        out = Path(args.out) if args.out else run
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
    else:
        solution = Path(args.solution)
        out = Path(args.out) if args.out else solution.parent
        receipt = None
    if not solution.is_file():
        raise SystemExit(f"no solution.py at {solution}")
    source = solution.read_text()
    receipt_matches = bool(receipt and receipt.get("program_id") == program_id(source)
                           and receipt.get("pack") == pack.contract_id)
    qualified = bool(receipt_matches and receipt.get("qualified") is True)
    if not qualified and not args.diagnostic:
        print(json.dumps({"status": "refused", "reason": "candidate has no passing public qualification receipt for this program and pack",
                          "receipt": None if receipt is None else {k: receipt.get(k) for k in
                                                                     ("program_id", "qualified", "families_passed",
                                                                      "required_family_count")},
                          "hint": "pass --diagnostic to run an explicitly labelled operator diagnostic"},
                         indent=1))
        return 2
    evaluator = _evaluator(pack, out, args.evaluator)
    result = evaluator.heldout(source, label="heldout")
    payload = {"schema_version": "linkpred-baseline-heldout.v1",
               "gate": {"qualified": qualified, "diagnostic": bool(args.diagnostic),
                        "receipt_matches": receipt_matches,
                        "receipt_program_id": None if receipt is None else receipt.get("program_id")},
               "claim_status": "operator_diagnostic_only",
               **(result.heldout or {})}
    target = out / "heldout"
    target.mkdir(parents=True, exist_ok=True)
    (target / "heldout.json").write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    (target / "heldout.md").write_text(_heldout_markdown(payload))
    print(json.dumps({k: v for k, v in payload.items() if k not in ("rows",)}, indent=1))
    return 0


def _heldout_markdown(payload: dict[str, Any]) -> str:
    lines = [f"# Held-out evaluation ({payload['claim_status']})", "",
             f"program {payload.get('program_id')}; complete={payload.get('complete')}; "
             f"gate={payload['gate']}", "",
             "| family (private tag) | valid | candidate mean AP | strongest mean AP | vs strongest | vs RA | vs CN | W/T/L |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for fam in payload.get("families", []):
        def pct(v: Any) -> str:
            return f"{v:+.2%}" if isinstance(v, (int, float)) else "n/a"
        def num(v: Any) -> str:
            return f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"
        lines.append(f"| {fam['family_id']} ({fam['private_family_tag']}) | {fam['instances_valid']}/"
                     f"{fam['instances_total']} | {num(fam['candidate_mean'])} | {num(fam['strongest_baseline_mean'])} "
                     f"| {pct(fam['relative_to_rowwise_strongest'])} | {pct(fam['relative_to_ra'])} | "
                     f"{pct(fam['relative_to_cn'])} | {fam['wins']}/{fam['ties']}/{fam['losses']} |")
    lines.append("")
    lines.append(f"Mean vs per-split strongest: {payload.get('mean_relative_to_rowwise_strongest')}; "
                 f"vs RA: {payload.get('mean_relative_to_ra')}; vs CN: {payload.get('mean_relative_to_cn')}; "
                 f"families meeting the {payload.get('minimum_relative_improvement')} minimum: "
                 f"{payload.get('families_meeting_minimum')}.")
    return "\n".join(lines) + "\n"


def cmd_export_openevolve(args: argparse.Namespace) -> int:
    manifest = export_openevolve_project(_pack(args), Path(args.out), model=args.model, api_base=args.api_base,
                                         max_iterations=args.iterations, islands=args.islands,
                                         population_size=args.population, reasoning_effort=args.reasoning_effort,
                                         evaluator_backend=args.evaluator)
    print(json.dumps(manifest, indent=1))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from .comparison import run_comparison
    if args.client == "replay" or args.cache:
        raise SystemExit("comparison runs use independent caches; replay each recorded arm separately")
    result = run_comparison(
        _pack(args), Path(args.out), seeds=args.seeds, max_calls=args.max_calls,
        client_factory=lambda out, limit: _client(args, out, limit),
        evaluator_factory=lambda pack, out: _evaluator(pack, out, args.evaluator),
        islands=args.islands, stop_on_qualify=args.stop_on_qualify,
    )
    report_args = argparse.Namespace(
        runs=[str(Path(args.out) / run["path"]) for run in result["runs"]],
        out=str(Path(args.out) / "comparison.md"), json_out=str(Path(args.out) / "comparison_rows.json"),
    )
    cmd_report(report_args)
    return 0 if result["complete"] else 1


def cmd_qualify_galahad(args: argparse.Namespace) -> int:
    from .galahad import qualify_run
    receipt = qualify_run(Path(args.run), Path(args.project),
                          control_root=Path(args.control_root) if args.control_root else None)
    print(json.dumps({key: receipt.get(key) for key in
                      ("status", "passed_family_count", "required_family_count", "problems")}, indent=2))
    return 0 if receipt.get("status") == "pass" else 2


def cmd_report(args: argparse.Namespace) -> int:
    rows = []
    for run in args.runs:
        run_path = Path(run)
        manifest_path = run_path / "run_manifest.json"
        if not manifest_path.is_file():
            rows.append({"run": run, "error": "no run_manifest.json"})
            continue
        manifest = json.loads(manifest_path.read_text())
        held_path = run_path / "heldout" / "heldout.json"
        held = json.loads(held_path.read_text()) if held_path.is_file() else None
        client = manifest.get("client") or {}
        best = manifest.get("best") or {}
        rows.append({"run": run_path.name, "arm": manifest.get("arm"), "calls": client.get("calls_used"),
                     "seed": manifest.get("config", {}).get("seed"), "pack": manifest.get("pack"),
                     "model": client.get("model"), "reasoning_effort": client.get("reasoning_effort"),
                     "client_kind": client.get("client_kind"),
                     "max_calls": client.get("max_calls"), "physical_calls": client.get("physical_calls"),
                     "prompt_tokens": client.get("prompt_tokens"), "completion_tokens": client.get("completion_tokens"),
                     "failed_calls": client.get("failed_calls"), "unknown_usage_calls": client.get("unknown_usage_calls"),
                     "seconds": manifest.get("seconds"), "claim_status": manifest.get("claim_status"),
                     "environment": manifest.get("environment"),
                     "evaluator": manifest.get("config", {}).get("evaluator"),
                     "stop_reason": manifest.get("outcome", {}).get("stop_reason"),
                     "stop_on_qualify": manifest.get("config", {}).get("stop_on_qualify"),
                     "evaluations": manifest.get("evaluations"), "best_fitness": best.get("fitness"),
                     "families_passed": best.get("families_passed"), "qualified": best.get("qualified"),
                     "first_qualified": (manifest.get("outcome") or {}).get("first_qualified_iteration",
                                                                             (manifest.get("outcome") or {}).get(
                                                                                 "first_qualified_sample")),
                     "heldout_vs_strongest": None if held is None else held.get("mean_relative_to_rowwise_strongest"),
                     "heldout_vs_ra": None if held is None else held.get("mean_relative_to_ra"),
                     "heldout_claim": None if held is None else held.get("claim_status")})
    lines = ["| run | arm | seed | model | calls / cap | physical | tokens in / out | seconds | evals | fitness | families | qualified | evidence | stop |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        if "error" in row:
            lines.append(f"| {row['run']} | {row['error']} |")
            continue
        def pct(v: Any) -> str:
            return f"{v:+.2%}" if isinstance(v, (int, float)) else "-"
        lines.append(f"| {row['run']} | {row['arm']} | {row['seed']} | {row['model']} | "
                     f"{row['calls']} / {row['max_calls']} | {row['physical_calls']} | "
                     f"{row['prompt_tokens']} / {row['completion_tokens']} | {row['seconds']} | "
                     f"{row['evaluations']} | {pct(row['best_fitness'])} | {row['families_passed']} | "
                     f"{row['qualified']} | {row['claim_status']} | {row['stop_reason']} |")
    if getattr(args, "json_out", None):
        Path(args.json_out).write_text(json.dumps(rows, indent=2) + "\n")
    held_rows = [row for row in rows if row.get("heldout_claim")]
    if held_rows:
        lines.extend(["", "| run | held-out vs strongest | held-out vs RA | held-out status |",
                      "| --- | --- | --- | --- |"])
        for row in held_rows:
            lines.append(f"| {row['run']} | {pct(row['heldout_vs_strongest'])} | "
                         f"{pct(row['heldout_vs_ra'])} | {row['heldout_claim']} |")
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evolution_baselines", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--pack", default=str(DEFAULT_PACK))
        p.add_argument("--evaluator", choices=["subprocess", "galahad"], default="subprocess",
                       help="local diagnostic worker or Galahad's isolated public evaluator")

    p = sub.add_parser("classical", help="zero-call floor: CN/AA/RA/Jaccard/PA/CAR through qualification")
    common(p)
    p.add_argument("--out", required=True)
    p.add_argument("--heuristics", default=None, help="comma list, default all six")
    p.add_argument("--solution", action="append", help="extra name=path solution.py to include")
    p.add_argument("--heldout", action="store_true", help="also run the operator-side hidden diagnostic")
    p.set_defaults(func=cmd_classical)

    p = sub.add_parser("best-of-n", help="independent samples from the initial prompt, no evolution")
    common(p)
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=60)
    p.add_argument("--max-calls", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-stop-on-qualify", action="store_true")
    _add_client_args(p)
    p.set_defaults(func=cmd_best_of_n)

    p = sub.add_parser("evolve", help="OpenEvolve-style islands + MAP-Elites + diff mutations")
    common(p)
    p.add_argument("--out", required=True)
    p.add_argument("--max-calls", type=int, default=600)
    p.add_argument("--max-iterations", type=int, default=None)
    p.add_argument("--islands", type=int, default=3)
    p.add_argument("--migration-interval", type=int, default=10)
    p.add_argument("--top-programs", type=int, default=3)
    p.add_argument("--diverse-programs", type=int, default=2)
    p.add_argument("--full-rewrite", action="store_true", help="ask for whole programs instead of diffs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seed-program", default=None, help="solution.py to seed the archive (default: CN)")
    p.add_argument("--no-stop-on-qualify", action="store_true")
    _add_client_args(p)
    p.set_defaults(func=cmd_evolve)

    p = sub.add_parser("heldout", help="historical test diagnostic using a matching public qualification receipt")
    common(p)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", help="run directory with best/ (from any arm)")
    group.add_argument("--solution", help="a bare solution.py (diagnostic only)")
    p.add_argument("--out", default=None)
    p.add_argument("--diagnostic", action="store_true", help="run even without a passing receipt")
    p.set_defaults(func=cmd_heldout)

    p = sub.add_parser("export-openevolve", help="write a stock-OpenEvolve project using this evaluator")
    common(p)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--api-base", default=None)
    p.add_argument("--iterations", type=int, default=600)
    p.add_argument("--islands", type=int, default=3)
    p.add_argument("--population", type=int, default=60)
    p.add_argument("--reasoning-effort", default=None)
    p.set_defaults(func=cmd_export_openevolve)

    p = sub.add_parser("report", help="one comparison table over several run directories")
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default=None)
    p.add_argument("--json-out", default=None)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("compare", help="matched-budget best-of-N vs program evolution across replicate seeds")
    common(p)
    _add_client_args(p)
    p.add_argument("--out", required=True)
    p.add_argument("--max-calls", required=True, type=int, help="same logical-call cap for each arm and seed")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--islands", type=int, default=3)
    p.add_argument("--stop-on-qualify", action="store_true", help="default is to use the full call budget")
    p.set_defaults(func=cmd_compare, evaluator="galahad")

    p = sub.add_parser("qualify-galahad", help="submit a selected program to an existing Galahad project's public qualification")
    p.add_argument("--run", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--control-root", default=None)
    p.set_defaults(func=cmd_qualify_galahad)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
