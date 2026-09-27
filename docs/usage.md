# Link-prediction baseline suite

`src/evolution_baselines/` runs external-style search systems on the same
`local_link_prediction_v2` pack Galahad uses, with the same statistical qualification
code. It supports comparisons between strategy evolution and direct program search.
The local evaluator produces operator diagnostics; formal comparisons require the
controller's isolated qualification and held-out flow.

## What is compared

| arm | model calls | what it is | why it is here |
| --- | --- | --- | --- |
| `classical` | 0 | CN, AA, RA, Jaccard, PA, CAR (sender file, verbatim) through public screening, optional hidden diagnostic | the classical reference floor |
| `best-of-n` | N | N independent programs from one prompt, no feedback, best kept | the sampling control: does evolution add anything beyond drawing N programs? |
| `evolve` | ≤ budget | OpenEvolve-style loop: islands, MAP-Elites archive keyed by (families passed, code size), rank-weighted parent selection, SEARCH/REPLACE diff mutations with per-split feedback, ring migration | the direct-program-search baseline for Galahad |
| `export-openevolve` | iteration limit | a stock OpenEvolve project whose evaluator is this suite's evaluator | the external implementation; record its installed version and actual usage |

The in-house arms' model prompts use public material: the sender's `problem_spec.md`, the interface contract, the
three exposed comparator implementations, per-family baseline means on the public splits, and
the exact qualification rule.  Nothing from `hidden_registry.json`, the hidden splits, or
`operator_sources/baselines.py` reaches a prompt (tests assert this on the recorded prompts).
The exported OpenEvolve configuration uses the public problem specification; it does not
replicate the in-house prompt and inspiration selection exactly.

## One evaluator for everything

`evaluate.CandidateEvaluator`

* runs the candidate in a subprocess worker on the controller projection of each instance
  (`instance_required` only; no labels), one wall-clock alarm per instance (60 s, the pack
  timeout), both method seeds, and flags non-determinism between seeds as invalid;
* scores with the pack's own `scorers/score.py`;
* computes a statistical screening result with `aggregate_development_qualification_v2` and the pack's
  `public_contract.json`, so the 95% one-sided LCB, the per-instance win requirement, the runtime
  envelope and the `max(2, families-1)` family rule are Galahad's, not a reimplementation;
  every candidate/baseline execution must be valid and every runtime envelope must pass,
  including envelopes in families that do not meet the quality-improvement threshold;
* fitness for the search arms = mean over the 24 public quality splits of the relative AP
  improvement over the per-split strongest of CN/RA/AA; invalid programs get −1.

The subprocess has a minimal environment but can still access host files and the network.
Use it only for reviewed candidate code on a disposable operator machine. Timing covers the
function call in a shared worker, whereas the controller runs isolated commands; local timing
is not controller timing evidence. The statistical calculation assumes trusted execution
only while calling the aggregator. Persisted rows and receipts report
`trusted_isolation: false` and `claim_status: operator_diagnostic_only`.
`qualified` means that the local statistical screening passed.

## The hidden gate

`heldout` requires a passing `best/public_qualification.json` whose existing `program_id`
matches `best/solution.py` and whose `pack` matches the selected pack. Old receipts without
pack identity must be regenerated; `--diagnostic` explicitly bypasses the screening requirement.
Every local hidden result remains `operator_diagnostic_only`, including qualified runs.
Hidden results are written under `<run>/heldout/` and are never read by any arm.
The diagnostic numbers report relative AP versus the per-split
strongest comparator, versus RA, and versus CN, per private family and pooled.
Freeze and final held-out claims belong to the controller protocol described in
[HIDDEN_DATA_RULES.md](HIDDEN_DATA_RULES.md).

## Commands

```bash
# From this repository root, in a Python 3.11+ virtual environment:
python -m pip install -e .

# floor (about 1 minute, no model calls)
python -m evolution_baselines classical --out runs/classical --heldout

# direct program search, a 600 logical-call limit
python -m evolution_baselines evolve --out runs/evolve --max-calls 600 --islands 3
python -m evolution_baselines best-of-n --out runs/bon --samples 60

# operator-only hidden diagnostic for a locally qualified best
python -m evolution_baselines heldout --run runs/evolve

# stock OpenEvolve
python -m evolution_baselines export-openevolve --out runs/openevolve_project
pip install openevolve && export OPENAI_API_KEY="$AZURE_OPENAI_API_KEY"
python runs/openevolve_project/run_openevolve.py --iterations 600

# one table
python -m evolution_baselines report runs/classical runs/bon runs/evolve

# Replay with the original search settings and a NEW output directory:
python -m evolution_baselines evolve --out runs/evolve-replay \
  --client replay --cache runs/evolve/llm_cache.jsonl --max-calls 600 --islands 3
```

Clients: `--client azure` (default; `.env` `AZURE_OPENAI_ENDPOINT` / `_API_KEY` / `_MODEL`),
`--client scripted --scripted-file responses.json` (tests), `--client replay` (re-walk a run from
its `llm_cache.jsonl` with zero physical calls). Replay infers the original request model from
new caches; for older caches pass `--model <original-deployment-name>`. The ledger counts
cache hits against the budget. Use the same pack, search settings and environment to reproduce
the search path; timings are measured again. Output directories with calls or results cannot
be reused because these arms do not implement budget-preserving resume.

## Run directory

```
programs/<id>.py, <id>.evaluation.json   every evaluated program and its full receipt
evaluations.jsonl                        one line per evaluation
iterations.jsonl                         one line per search step (parse failures, duplicates, migrations)
calls.jsonl, llm_cache.jsonl             every prompt/response, replay cache
archive.json                             MAP-Elites cells (evolve)
best/solution.py, run_method.py, method_manifest.json, public_qualification.json
run_manifest.json, summary.md            config, budget usage, outcome, ranked table
heldout/heldout.json, heldout.md         only after the gate
```

## OpenEvolve specifics

OpenEvolve chooses reasoning-model request parameters from the model-name prefix.  Azure
deployment names (for example `professor-gpt55`) do not match, so the exported
`run_openevolve.py` installs a shim that rewrites `max_tokens` to `max_completion_tokens` and
drops `temperature`/`top_p`.  Everything else is stock.  The exported evaluator returns
`combined_score` (the fitness above), `families_passed`, `qualified` and per-family LCBs, and
`config.yaml` uses `families_passed` and `complexity` as MAP-Elites dimensions.

## Comparing results

Report the same task version, model, reasoning setting, stopping rule, replicate seeds and
evaluation environment for each arm. Record logical calls, physical calls, input/output
tokens, wall time and candidate evaluations. Equal call limits alone do not equalize compute:
Galahad's research sessions and direct code-generation prompts have different costs; OpenEvolve
iterations and API retries can also differ from logical calls.

Keep the in-house `evolve` arm labelled **OpenEvolve-style**. A named OpenEvolve result requires
executing the exported project with a recorded OpenEvolve version. The current export tests
exercise generated code import and argument handling without making model calls.

`first_qualified_iteration` / `first_qualified_sample` record when local screening first passed;
`calls_used` and `evaluations` record its search cost. Report multiple independent runs before
claiming one search system is better. Keep public screening, operator hidden diagnostics and
formal controller held-out results distinct in tables.
