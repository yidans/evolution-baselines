"""Run existing search arms with one explicit budget and replicate schedule."""
from __future__ import annotations

import json
from pathlib import Path

from .arms import EvolveConfig, run_best_of_n, run_evolve


def run_comparison(pack, out: Path, *, seeds: list[int], max_calls: int,
                   client_factory, evaluator_factory, islands: int = 3,
                   stop_on_qualify: bool = False):
    if max_calls < 1 or not seeds or len(set(seeds)) != len(seeds) or islands < 1:
        raise ValueError("comparison needs a positive call budget/island count and unique replicate seeds")
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("comparison output must be empty; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "pack": pack.contract_id, "max_calls_per_run": max_calls,
        "replicate_seeds": seeds,
        "stopping_rule": "first_public_qualification" if stop_on_qualify else "fixed_call_budget",
        "budget_basis": "logical model requests; tokens and evaluation costs reported separately",
        "runs": [], "complete": False,
    }

    def save():
        (out / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")

    save()
    for index, seed in enumerate(seeds):
        # Alternate order to reduce a consistent first-arm timing advantage.
        order = ["best-of-n", "evolve"] if index % 2 == 0 else ["evolve", "best-of-n"]
        for arm in order:
            path = out / f"seed-{seed}" / arm
            evaluator = evaluator_factory(pack, path)
            client = client_factory(path, max_calls)
            if arm == "best-of-n":
                result = run_best_of_n(pack, evaluator, client, path, samples=max_calls,
                                       seed=seed, stop_on_qualify=stop_on_qualify)
            else:
                result = run_evolve(pack, evaluator, client, path, EvolveConfig(
                    max_llm_calls=max_calls, seed=seed, islands=islands,
                    stop_on_qualify=stop_on_qualify,
                ))
            summary["runs"].append({
                "path": str(path.relative_to(out)), "seed": seed, "arm": arm,
                "claim_status": result["claim_status"], "client": result["client"],
                "evaluations": result["evaluations"], "seconds": result["seconds"],
                "best": result["best"], "outcome": result["outcome"],
            })
            save()
            if result["outcome"].get("stop_reason") == "client_error":
                summary["error"] = "model request failed; partial run and budget usage preserved"
                save()
                return summary
    summary["complete"] = True
    save()
    return summary
