# Working on the comparison suite

Use Python 3.11+ on Linux or macOS. Each collaborator keeps their own checkout
and output directories:

```bash
uv sync --locked
uv run pytest -q
uv run ruff check src tests
uv run evolution-baselines compare --out runs/first-scripted-comparison \
  --client scripted --scripted-file examples/responses.json \
  --max-calls 1 --seeds 0 --evaluator subprocess
```

These commands need no model credentials. The scripted comparison checks the
workflow; its measurements are not model-performance results. Use a fresh output
directory when repeating it. Galahad integration tests skip when Galahad is not
installed; the remaining suite exercises the standalone package.

For Galahad comparisons, read [GALAHAD_COMPARISON.md](docs/GALAHAD_COMPARISON.md),
install the private Galahad checkout in the same operator environment, and run
the full tests there. Linux integration needs Docker and the configured Galahad
evaluator image. The public CI intentionally needs no private-repository token,
model credentials or Azure access.

Use a focused branch and a PR against `main`. Describe behavior, actual
validation and any experimental consequence. Keep the in-house OpenEvolve-style
arm distinct from the external OpenEvolve implementation. Compare runs only
with their recorded pack, code revision, model settings, seed schedule, budget
and stopping rule visible.

Keep each experiment's source, settings, model-call ledger, evaluator receipts
and summary together in its existing run directory. Link an operator-managed
archive in the handoff instead of committing raw transcripts or large results.
Retain failures that explain later runs; do not overwrite the previous output
to make a rerun appear continuous. See [VALIDATION.md](docs/VALIDATION.md) for
the implementation checks already performed.

This repository is public. The bundled historical test splits have already
been disclosed. New private packs, provider secrets, VM configuration and
future held-out evidence stay in controller-owned storage. Before editing
data handling, read [HIDDEN_DATA_RULES.md](docs/HIDDEN_DATA_RULES.md).
