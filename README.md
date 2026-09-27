# Evolution baselines

A standalone comparison suite extracted from Galahad. It includes the baseline
code, statistical evaluator, link-prediction benchmark snapshot and tests; a
Galahad checkout, Hermes runtime and organization-repository access are not needed.

| Command | Method |
| --- | --- |
| `classical` | CN, AA, RA, Jaccard, PA and CAR; no model calls |
| `best-of-n` | Independent generated programs, without evaluation feedback |
| `evolve` | OpenEvolve-style islands, MAP-Elites archive and feedback-driven mutations |
| `export-openevolve` | Generate a project for the separately installed OpenEvolve implementation |
| `heldout` | Operator-only hidden diagnostic using a matching local screening receipt |
| `report` | Compare recorded runs |

The in-house `evolve` implementation is labelled **OpenEvolve-style**. The
OpenEvolve export is an integration with the external package. Other evolution
systems are not included in this snapshot.

## Install

Python 3.11+ on Linux or macOS (the worker uses POSIX alarms):

```bash
git clone https://github.com/yidans/evolution-baselines.git
cd evolution-baselines
uv sync --locked
uv run evolution-baselines --help
```

Or install with pip in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m evolution_baselines --help
```

`uv.lock` records the tested dependency resolution. Benchmark data is included
in the Python package and wheel; commands can run from another working directory.

## Try it without credentials

```bash
uv run evolution-baselines classical --out runs/classical
uv run evolution-baselines best-of-n --out runs/scripted --samples 1 \
  --client scripted --scripted-file examples/responses.json
uv run evolution-baselines best-of-n --out runs/replay --samples 1 \
  --client replay --cache runs/scripted/llm_cache.jsonl
uv run evolution-baselines report runs/classical runs/scripted runs/replay
uv run pytest -q
```

The scripted example is an integration smoke using the existing CN reference,
not evidence that a search system improves on it. Use a fresh output directory
for each run. Replay reads the previous cache and records its own results.

## Run model-backed search

Copy `.env.example` to `.env` if you do not already have one, then fill in the
endpoint, deployment name and key. The following commands make model calls:

```bash
uv run evolution-baselines best-of-n --out runs/best-of-n --samples 60
uv run evolution-baselines evolve --out runs/evolve --max-calls 600 --islands 3
uv run evolution-baselines export-openevolve --out runs/openevolve \
  --model <deployment-name> --api-base https://<resource>.openai.azure.com/openai/v1/
```

See [the usage guide](docs/usage.md) for OpenEvolve setup, replay, reports,
output files and comparison budgets. OpenEvolve is an optional external
installation and is not installed by the default dependency set.

## Evaluation and sharing

The included `local_link_prediction_v2` snapshot has 24 public quality splits,
three runtime envelopes and 16 operator-only hidden splits. Local screening
uses Galahad's numerical qualification rules, including global runtime validity.
All outputs are labelled `operator_diagnostic_only` and `trusted_isolation: false`.

Candidate code runs in ordinary subprocesses with access to the host filesystem
and network. Use reviewed code on a disposable operator machine. Formal claims
require the Galahad controller's isolated qualification/freeze/held-out flow.
Never feed hidden measurements back into program search; see
[data handling](docs/HIDDEN_DATA_RULES.md).

Keep this repository private: it includes operator-only benchmark data and
third-party material whose redistribution terms are not uniformly specified.
See [source provenance](SOURCE.md) and the benchmark's preserved
[upstream README](src/evolution_baselines/benchmarks/local_link_prediction_v2/UPSTREAM_README.md).
