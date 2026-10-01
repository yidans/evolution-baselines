# Galahad comparison verification — 2026-09-28

Environment: macOS, Python 3.11.5; local Galahad source imported through
`PYTHONPATH=../ai-professor-computational-methods/src`. No paid model calls.

- 40 tests passed in three nonoverlapping batches: 38 existing/comparison tests
  (183.39 s), the full existing-project controller qualification integration
  (112.20 s), and the model-failure/partial-run test added during review (7.82 s).
- The complete CN public evaluation executed 27 instances at two method seeds:
  54 controller rows, all trusted and valid. No family qualified, as expected.
  Evidence: `runs/verification-20260928/controller-public/cn-full.controller.json`.
- A real isolated candidate was denied access to a file outside its readable
  inputs. The controller-backed CN scores equalled the pack's CN comparator.
- An actual Galahad project was created and its benchmark locked in a temporary
  test directory; `qualify_run` submitted the selected candidate through the
  existing qualification. It correctly returned a failed quality qualification
  with valid executions and did not create a freeze certificate or held-out result.
- Matched-budget orchestration, separate caches, replay request settings,
  request-failure accounting and partial comparison records were exercised.
- Ruff and `git diff --check` passed. Type checking passed for all eight changed
  or added implementation modules. A broader type check reports six existing
  diagnostics in `_qualification.py`; all six were reproduced on that file from
  the original Git commit. Its bytes remain unchanged.
- The packaged benchmark matches the current local Galahad pack in all 180
  shared non-README files. No split, scorer or comparator was changed.
- Source distribution and wheel built. Installing the wheel into a temporary
  directory exposed the new commands/modules and included benchmark without
  requiring Galahad for ordinary imports.

These checks establish implementation and execution integration, not comparative
model performance. Azure/Docker execution and a real stock OpenEvolve search
were not run. Historical bundled hidden data is publicly disclosed; see
[the comparison guide](GALAHAD_COMPARISON.md) before a formal experiment.

Full regression command (includes the optional Galahad integration tests):

```bash
PYTHONPATH=/absolute/path/to/galahad/src python -m pytest -q
```

# Original package verification — 2026-09-27

Environment: macOS, Python 3.11.5; dependencies recorded in `uv.lock`.

- All 27 regression tests passed (164.64 seconds), including evaluation parity,
  scripted evolution, replay, malformed outputs and hidden-screening receipt checks.
- Ruff passed with the source project's E4/E7/E9/F rule set.
- Source distribution and wheel built successfully using `uv build`.
- All 182 benchmark files were present in the wheel.
- Installed the wheel into a fresh virtual environment and ran from a separate
  temporary directory, with no source checkout on `PYTHONPATH`.
- The installed console command loaded the packaged benchmark and completed CN
  classical evaluation, one scripted best-of-N sample, replay of that sample and
  a combined report.
- OpenEvolve export completed. The exported runner's help and evaluator import
  worked; the exported evaluator evaluated its initial CN program successfully.
- No paid model requests were made. The external OpenEvolve search loop itself
  was not run; these checks establish packaging and evaluator integration.

Run the regression suite with `uv run pytest -q`.
