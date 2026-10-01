"""Export a ready-to-run OpenEvolve project that uses this package's evaluator.

The exported evaluator scores programs with :class:`CandidateEvaluator` on the public
families only and reports ``combined_score`` = mean relative improvement over the per-split
strongest comparator (the same fitness the in-house arm uses), plus ``families_passed`` and
``qualified`` for local statistical screening. Hidden graphs are not used by this
evaluator; subprocess execution does not enforce a filesystem security boundary.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .arms import CN_SEED_PROGRAM
from .pack import LinkPredictionPack

EVALUATOR_TEMPLATE = '''"""OpenEvolve evaluator: Galahad v2 pack, public families, controller qualification rule."""
import sys
from pathlib import Path

sys.path.insert(0, {src!r})

from {evaluator_module} import {evaluator_class} as CandidateEvaluator  # noqa: E402
from evolution_baselines.pack import LinkPredictionPack  # noqa: E402

PACK = Path({pack!r})
WORK = Path(__file__).resolve().parent / "evaluator_work"
_EVALUATOR = None


def _evaluator():
    global _EVALUATOR
    if _EVALUATOR is None:
        _EVALUATOR = CandidateEvaluator(LinkPredictionPack(PACK), WORK)
    return _EVALUATOR


def evaluate(program_path):
    source = Path(program_path).read_text()
    result = _evaluator().evaluate(source)
    families = {{f.family_id: f for f in result.families}}
    metrics = {{
        # OpenEvolve maximizes combined_score; failed programs get the same -1 as the in-house arm.
        "combined_score": float(result.fitness),
        "families_passed": float(result.families_passed),
        "qualified": 1.0 if result.qualified else 0.0,
        "valid": 1.0 if result.valid else 0.0,
    }}
    for family_id, item in families.items():
        lcb = item.lower_confidence_bound_95
        metrics[f"lcb_{{family_id}}"] = float(lcb) if isinstance(lcb, (int, float)) else -1.0
        metrics[f"wins_{{family_id}}"] = float(item.wins)
    if result.errors:
        metrics["error"] = " | ".join(str(e)[:200] for e in result.errors[:2])
    return metrics
'''

RUNNER_TEMPLATE = '''#!/usr/bin/env python3
"""Run OpenEvolve on the exported project with the Azure gpt-5.x parameter shim.

OpenEvolve decides reasoning-model request parameters from the model-name prefix
("gpt-5", "o3", ...).  Azure deployments are named freely (for example {model!r}), so this
wrapper rewrites max_tokens -> max_completion_tokens and drops temperature/top_p before the
request leaves the OpenAI client.  Everything else is stock OpenEvolve.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def install_azure_shim():
    from openai.resources.chat import completions as chat_completions

    original = chat_completions.AsyncCompletions.create

    async def create(self, *args, **kwargs):
        if "max_tokens" in kwargs and "max_completion_tokens" not in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        kwargs.pop("temperature", None)
        kwargs.pop("top_p", None)
        return await original(self, *args, **kwargs)

    chat_completions.AsyncCompletions.create = create


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--output", type=Path, default=HERE / "openevolve_output")
    parser.add_argument("--no-shim", action="store_true", help="send stock parameters")
    args = parser.parse_args()
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")
    if not api_key:
        sys.exit("set OPENAI_API_KEY (or AZURE_OPENAI_API_KEY) before running")
    os.environ["OPENAI_API_KEY"] = api_key
    if not args.no_shim:
        install_azure_shim()
    from openevolve.api import run_evolution

    result = run_evolution(
        initial_program=str(HERE / "initial_program.py"),
        evaluator=str(HERE / "evaluator.py"),
        config=str(HERE / "config.yaml"),
        iterations=args.iterations,
        output_dir=str(args.output),
        cleanup=False,
    )
    print("best score:", result.best_score)
    print("best program written under:", args.output)


if __name__ == "__main__":
    main()
'''


def _initial_program() -> str:
    lines = CN_SEED_PROGRAM.splitlines()
    out = []
    for line in lines:
        if line.startswith("def score_candidates"):
            out.append("# EVOLVE-BLOCK-START")
        out.append(line)
    out.append("# EVOLVE-BLOCK-END")
    return "\n".join(out) + "\n"


def export_openevolve_project(pack: LinkPredictionPack, destination: Path, *, model: str | None = None,
                              api_base: str | None = None, max_iterations: int = 600,
                              islands: int = 3, population_size: int = 60,
                              reasoning_effort: str | None = None,
                              evaluator_backend: str = "subprocess") -> dict[str, Any]:
    if evaluator_backend not in {"subprocess", "galahad"}:
        raise ValueError("unknown evaluator backend")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    src = str(Path(__file__).resolve().parents[1])
    model = model or os.environ.get("AZURE_OPENAI_MODEL") or "professor-gpt55"
    api_base = api_base or os.environ.get("AZURE_OPENAI_ENDPOINT") or "https://<resource>.openai.azure.com/openai/v1/"
    (destination / "initial_program.py").write_text(_initial_program())
    module, cls = (("evolution_baselines.galahad", "GalahadEvaluator") if evaluator_backend == "galahad"
                   else ("evolution_baselines.evaluate", "CandidateEvaluator"))
    (destination / "evaluator.py").write_text(EVALUATOR_TEMPLATE.format(
        src=src, pack=str(pack.root), evaluator_module=module, evaluator_class=cls))
    (destination / "run_openevolve.py").write_text(RUNNER_TEMPLATE.format(model=model))
    llm: dict[str, Any] = {"api_base": api_base, "models": [{"name": model, "weight": 1.0}],
                           "max_tokens": 16000, "timeout": 900, "retries": 2}
    if reasoning_effort:
        llm["reasoning_effort"] = reasoning_effort
    config = {
        "max_iterations": max_iterations, "checkpoint_interval": 10, "random_seed": 0, "language": "python",
        "diff_based_evolution": True, "max_code_length": 20000, "llm": llm,
        "prompt": {"system_message": _system_message(pack), "num_top_programs": 3, "num_diverse_programs": 2},
        "database": {"population_size": population_size, "archive_size": 20, "num_islands": islands,
                     "elite_selection_ratio": 0.1, "exploration_ratio": 0.2, "exploitation_ratio": 0.7,
                     "feature_dimensions": ["families_passed", "complexity"], "feature_bins": 4,
                     "migration_interval": 10, "migration_rate": 0.1, "random_seed": 0},
        "evaluator": {"timeout": 60 * 27 * 2 + 120, "max_retries": 1, "cascade_evaluation": False,
                      "parallel_evaluations": 1, "use_llm_feedback": False},
    }
    _write_yaml(destination / "config.yaml", config)
    readme = f"""# OpenEvolve baseline project for `{pack.contract_id}`

Generated by `python -m evolution_baselines export-openevolve`.

* `initial_program.py`: the exposed CN baseline with EVOLVE-BLOCK markers.
* `evaluator.py`: public families only; `combined_score` is the mean relative AP improvement over the
  per-split strongest comparator (CN/RA/AA), computed by the controller's own qualification code.
  `families_passed` and `qualified` use the identical 2%-LCB / all-splits-win / 2-of-3 rule.
* `config.yaml`: {islands} islands, population {population_size}, diff-based evolution, {max_iterations} iterations,
  model `{model}` at `{api_base}`.
* `run_openevolve.py`: launches stock OpenEvolve with an Azure parameter shim (see its docstring).

Run:

```bash
pip install openevolve
export OPENAI_API_KEY="$AZURE_OPENAI_API_KEY"
python run_openevolve.py --iterations {max_iterations}
```

Evaluator backend: `{evaluator_backend}`. The subprocess backend provides operator diagnostics;
the Galahad backend uses its isolated generator/candidate/baseline/scorer execution path and
requires Galahad installed in the same environment. Both use public instances only.
Formal held-out claims retain the controller's qualification, review and freeze requirements.
To screen the exported best program and create its local qualification receipt, run from the
repository root (use a new output directory):

```bash
python -m evolution_baselines classical --pack {str(pack.root)!r} \\
  --out runs/openevolve-screen --heuristics cn \\
  --solution openevolve=/absolute/path/to/openevolve_output/best/best_program.py
python -m evolution_baselines heldout --pack {str(pack.root)!r} \\
  --run runs/openevolve-screen
```

The second command requires a matching passing screening receipt. Its output remains an
operator diagnostic and must never be fed back to program search.
"""
    (destination / "README.md").write_text(readme)
    manifest = {"pack": pack.contract_id, "model": model, "api_base": api_base, "max_iterations": max_iterations,
                "evaluator": evaluator_backend,
                "islands": islands, "population_size": population_size, "files": sorted(
                    p.name for p in destination.iterdir())}
    (destination / "export_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def _system_message(pack: LinkPredictionPack) -> str:
    from .prompts import SYSTEM_PROMPT

    spec = (pack.public_reference_dir() / "problem_spec.md").read_text()
    return SYSTEM_PROMPT + "\n\n" + spec


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError:  # pragma: no cover - pyyaml is a core dependency
        path.write_text(json.dumps(payload, indent=2))
        return
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


__all__ = ["export_openevolve_project"]
