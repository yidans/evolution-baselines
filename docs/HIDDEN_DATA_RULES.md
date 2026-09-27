# Benchmark data handling

The package includes public development data and operator-only hidden data.
The latter lives in `src/evolution_baselines/benchmarks/local_link_prediction_v2/`:
`hidden_registry.json`, hidden configurations/splits, `baseline_validation.json`
and `operator_sources/` are not model prompt inputs.

The search prompts include only public problem specifications, exposed
comparators and public evaluation feedback. Keep hidden results out of prompts,
candidate code, strategy notes and future search decisions. Do not give a
research agent unrestricted access to this repository as its workspace.

`heldout` verifies that a passing local screening receipt names the same program
and pack. `--diagnostic` permits an explicitly labelled operator diagnostic
without that receipt. Both paths remain diagnostics: the evaluator uses ordinary
subprocesses and does not enforce filesystem or network isolation.

Formal qualification and final held-out claims require the isolated controller
in the Galahad repository. This standalone package does not replace its freeze
or final-evaluation procedure.
