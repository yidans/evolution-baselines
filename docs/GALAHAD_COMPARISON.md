# Using this suite as Galahad's baseline

The experimental contrast is **direct program search versus strategy evolution**.
`best-of-n` samples independent programs, `evolve` mutates programs with feedback,
Galahad Lite runs a fixed research strategy, and Galahad evolves that strategy.
Keep these names in result tables. The in-house `evolve` arm is OpenEvolve-style;
a named OpenEvolve result requires running the external implementation.

## Matched runs

Install both packages in the same operator Python environment. Hermes is needed
to run Galahad itself, but is not used by this suite's evaluator.

```bash
python -m pip install -e . -e /absolute/path/to/galahad
python -m evolution_baselines compare \
  --pack /absolute/path/to/galahad/controller_benchmarks/local_link_prediction_v2 \
  --out runs/comparison --max-calls 600 --seeds 0 1 2 \
  --model <same-Azure-deployment> --reasoning-effort <same-setting> \
  --max-output-tokens 16000
```

`compare` defaults to the Galahad evaluator and fixed call budget. Both search
arms receive the same call cap, model, reasoning effort, output-token limit,
public pack and replicate schedule. Arm order alternates between replicates.
Outputs: `comparison.json`, `comparison.md`, `comparison_rows.json`, and each
arm's full run directory. For time-to-qualification experiments use
`--stop-on-qualify` on the entire comparison. Individual legacy commands keep
their original defaults.

Replicate seeds control search selection and sampling prompt nonces. They do
not guarantee deterministic model output or replace the pack's method seeds.
SDK retries are disabled: each API attempt, including a failure, consumes one
call. A model request failure saves the partial run and stops with a nonzero
exit code. Unknown token usage is counted explicitly. Restart in a new directory.

Offline workflow check:

```bash
python -m evolution_baselines compare --out runs/comparison-smoke \
  --max-calls 1 --seeds 0 1 --evaluator subprocess \
  --client scripted --scripted-file examples/responses.json
```

Scripted tokens are estimates; replay tokens describe original calls. Neither
is model-performance evidence. Replay each arm separately with its own cache;
do not share caches across independent replicates.

## Same evaluator as Galahad

The optional backend calls Galahad's `_evaluate_one_heldout_run` public execution
path and `aggregate_development_qualification_v2`. It uses the pack's generator,
projected inputs, both method seeds, baseline commands, scorer, runtime envelopes
and statistical rules. Each command runs in Galahad's isolation. Baselines rerun
for every candidate. Import, preprocessing and serialization are included in the
measured command time. Unsafe host evaluation is refused.

Use the same Galahad revision, evaluator image, host resources and pack directory
for every arm. On Linux, configure Docker and the same immutable
`GALAHAD_EVALUATOR_IMAGE` used by Galahad. macOS uses Galahad's Seatbelt backend.
Controller execution rows are retained in `evaluator_work/*.controller.json`.
`controller_public_evaluation` means public evidence; it is not a freeze
certificate or final held-out claim.

Total run time includes the public prompt-profile preparation. `evaluations`
counts candidate evaluations, including the initial CN program in `evolve`;
comparator-profile setup is additional computation. Report both time and counts.

Stock OpenEvolve can use this backend too:

```bash
python -m evolution_baselines export-openevolve --evaluator galahad \
  --pack /absolute/path/to/galahad/controller_benchmarks/local_link_prediction_v2 \
  --out runs/openevolve --model <same-Azure-deployment>
```

Its iterations are not the in-house ledger's model calls. Record its installed
version, actual requests/retries and tokens separately. The export maximizes
mean relative AP; the in-house arm prioritizes qualification and families passed
before AP. They are different search methods.

## Against Galahad and Galahad Lite

Run the existing Galahad launchers on the same pack and evaluator environment.
Count RSI mutation, parent, child and champion model usage together; count a
reused parent's original evaluation cost once. A per-session Galahad cap cannot
be compared to this suite's whole-run cap. Equal calls also do not imply equal
tokens, time or cost. This command runs the two baseline arms; it does not launch
Galahad or translate its session budgets automatically.

Report qualification rate over replicates, cost to first qualification, best
public fitness, calls, tokens, wall time and candidate evaluations. Choose
fixed-budget quality or time-to-qualification before interpreting results.
JSON reports expose model settings, seed, stopping rule, budget and evidence
status so mismatched runs remain visible.

## Existing controller qualification and final evaluation

Submit a selected program to a Galahad project that already has its benchmark locked:

```bash
python -m evolution_baselines qualify-galahad \
  --run runs/comparison/seed-0/evolve \
  --project /absolute/path/to/research_projects/comparison-project \
  --control-root /absolute/path/to/controller-root
```

This checks task and program identity, writes the executable bundle under stage
3, invokes the existing project qualification, and saves
`galahad_qualification.json` in the baseline run. Failed qualification returns
exit code 2. Existing candidate directories are not overwritten.

Complete candidate-specific correctness evidence and method review before final
evaluation. Register the completed method bundle (which requalifies those exact
files), then use Galahad's existing `freeze_algorithm` and `evaluate_heldout` flow.
This package does not invent proofs, review verdicts or freeze certificates.

## Disclosed historical test data

The bundled historical hidden splits are present in this public repository.
They support reproduction and diagnostics, not an undisclosed final test.
Use fresh controller-only held-out data shared across the comparison arms for
future generalization claims. Public searches need only `public_contract.json`,
`public/` and `development_benchmark_bundle/`; private inputs are opened only by
an explicit historical diagnostic. Keep final results out of subsequent search.
