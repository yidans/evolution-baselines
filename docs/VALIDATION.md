# Package verification — 2026-09-27

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
