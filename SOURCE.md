# Source provenance

Extracted from [GRAIL-innovationai/galahad](https://github.com/GRAIL-innovationai/galahad),
commit `53059b1b3e46f3c6b7291aeb2d30b24e10aa3561`, on 2026-09-27.

| This repository | Source |
| --- | --- |
| `src/evolution_baselines/*.py` | `src/ai_professor/baselines/linkpred/` |
| `_qualification.py` | Statistical functions and constants before `ProjectQualification` in `src/ai_professor/autonomy/qualification.py` |
| `_io.py` | `utc_now`, `load_project_env` and `_find_env_file` from `src/ai_professor/io_utils.py` |
| Package benchmark directory | Tracked files in `controller_benchmarks/local_link_prediction_v2/`, excluding top-level rebuild/download scripts |
| `tests/` | The link-prediction suite and collaborator regression tests |

Packaging changes rename the Python namespace, make benchmark lookup relative
to the installed package, adjust generated evaluator imports, and remove the
statistical module's unused controller imports. The search algorithms, scorer,
data snapshot and statistical function bodies are preserved. The environment
helper retains `AI_PROFESSOR_DISABLE_DOTENV=1` for compatibility with the tests.

The full Galahad controller, Hermes deployment, RSI strategy layer and other
task packs are outside this repository. Rebuilding this benchmark belongs in
the source repository; this package runs the included snapshot.

The OpenEvolve exporter creates integration files but does not vendor the
external OpenEvolve source. Record the installed OpenEvolve version for an actual
comparison. Preserve the OpenEvolve-style label for the in-house search arm.

Dataset attribution, raw-source records and the original pack documentation
are retained in the benchmark directory. This extraction does not assign a new
license to the source code, sender-provided heuristics or graph data.
