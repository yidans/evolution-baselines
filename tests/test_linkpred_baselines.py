"""Baseline suite for the link-prediction pack: evaluator parity, arms, gate, export."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evolution_baselines import arms, cli
from evolution_baselines.evaluate import CandidateEvaluator, program_id, write_frozen_bundle
from evolution_baselines.llm import BudgetExhausted, LedgerClient, ScriptedClient
from evolution_baselines.openevolve_export import export_openevolve_project
from evolution_baselines.pack import LinkPredictionPack
from evolution_baselines.prompts import PromptContext, initial_prompt, mutation_prompt, parse_program

PACK_ROOT = Path(__file__).resolve().parents[1] / "src/evolution_baselines/benchmarks/local_link_prediction_v2"

CN = arms.CN_SEED_PROGRAM
DIFF_TO_RA = """Use resource allocation instead of a raw count.
""" "<<<<<<< SEARCH\n" """    return np.asarray([len(adj[u] & adj[v]) for u, v in candidates], dtype=float)
""" "=======\n" """    out = []
    for u, v in candidates:
        out.append(sum(1.0 / len(adj[w]) for w in adj[u] & adj[v]))
    return np.asarray(out, dtype=float)
""" ">>>>>>> REPLACE\n" """"""
FULL_HYBRID = """CN plus a small RA term and a degree-difference tiebreak.
```python
import numpy as np


def score_candidates(adj, candidates):
    out = []
    for u, v in candidates:
        cn = adj[u] & adj[v]
        ra = sum(1.0 / len(adj[w]) for w in cn)
        out.append(len(cn) + 0.02 * ra + 0.001 / (1 + abs(len(adj[u]) - len(adj[v]))))
    return np.asarray(out, dtype=float)
```
"""
BROKEN = "```python\ndef score_candidates(adj, candidates):\n    return [1 / 0 for _ in candidates]\n```\n"
GARBAGE = "I cannot help with that."
NONDETERMINISTIC = """import random
import numpy as np


def score_candidates(adj, candidates):
    return np.asarray([len(adj[u] & adj[v]) + random.random() for u, v in candidates], dtype=float)
"""
HANGS = """def score_candidates(adj, candidates):
    while True:
        pass
"""


@pytest.fixture(scope="session")
def pack() -> LinkPredictionPack:
    return LinkPredictionPack(PACK_ROOT)


@pytest.fixture(scope="session")
def evaluator(pack: LinkPredictionPack, tmp_path_factory: pytest.TempPathFactory) -> CandidateEvaluator:
    return CandidateEvaluator(pack, tmp_path_factory.mktemp("linkpred-eval"))


# ------------------------------------------------------------------------------ pack
def test_pack_specs_projection_and_family_keys(pack: LinkPredictionPack) -> None:
    public = pack.public_specs()
    hidden = pack.hidden_specs()
    assert len(public) == 27 and sum(s.role == "quality" for s in public) == 24
    assert len(hidden) == 16 and pack.required_family_count() == 2
    # hidden families reuse instance ids, so evaluation keys must be per family
    assert len({s.instance_id for s in hidden}) == 8 and len({s.key for s in hidden}) == 16
    full = pack.load_instance(public[0])
    projected = pack.project(full)
    assert set(projected) == {"schema_version", "node_count", "train_edges", "candidates"}
    assert "hidden_edges" in full and "hidden_edges" not in projected


# --------------------------------------------------------------------------- parsing
def test_parse_program_diff_full_and_failures() -> None:
    parsed = parse_program(DIFF_TO_RA, parent_source=CN, diff_based=True)
    assert parsed.mode == "diff" and parsed.source and "1.0 / len(adj[w])" in parsed.source
    assert parsed.rationale.startswith("Use resource allocation")
    ambiguous = "<<<<<<< SEARCH\nimport numpy as np\n=======\nimport numpy\n>>>>>>> REPLACE\n"
    assert parse_program(ambiguous, parent_source=CN + CN, diff_based=True).source is None
    full = parse_program(FULL_HYBRID, parent_source=CN, diff_based=True)
    assert full.mode == "full" and full.source and full.source.startswith("import numpy")
    assert parse_program(GARBAGE, parent_source=CN, diff_based=True).source is None
    assert parse_program(DIFF_TO_RA, parent_source=None, diff_based=False).source is None
    bare = parse_program("def score_candidates(adj, candidates):\n    return [0.0] * len(candidates)\n",
                         parent_source=None, diff_based=False)
    assert bare.mode == "full" and bare.source


# ------------------------------------------------------------------------------ llm
def test_ledger_client_budget_cache_and_ledger(tmp_path: Path) -> None:
    client = LedgerClient(ScriptedClient(["a", "b"]), tmp_path, max_calls=3)
    first = client.complete(system="s", user="u", tag="t1")
    again = client.complete(system="s", user="u", tag="t2")
    assert first.text == "a" and again.text == "a" and again.cached
    assert client.calls_used == 2 and client.physical_calls == 1
    client.complete(system="s", user="v", tag="t3")
    with pytest.raises(BudgetExhausted):
        client.complete(system="s", user="w", tag="t4")
    lines = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert [line["cached"] for line in lines] == [False, True, False]
    replay = LedgerClient(ScriptedClient(["never"]), tmp_path / "replay", max_calls=5,
                          cache_path=tmp_path / "llm_cache.jsonl")
    assert replay.complete(system="s", user="v", tag="r").text == "b" and replay.physical_calls == 0


# ------------------------------------------------------------------------- archive
def _program(pid: str, source: str, fitness: float, families: int, valid: bool = True) -> arms.Program:
    return arms.Program(id=pid, source=source, origin="t", iteration=0, island=0, parent_id=None,
                        fitness=fitness, families_passed=families, qualified=families >= 2, valid=valid)


def test_island_cells_replacement_and_parent_selection() -> None:
    island = arms.Island(0, bins=(15, 30, 60))
    weak = _program("w", CN, -0.1, 0)
    strong = _program("s", CN + "\n# comment\n", 0.05, 1)
    assert island.add(weak) and island.cell(weak) == (0, 0)
    assert island.add(strong) and island.cell(strong) == (1, 0)
    weaker_same_cell = _program("x", CN, -0.2, 0)
    assert not island.add(weaker_same_cell) and island.cells[(0, 0)].id == "w"
    invalid = _program("i", CN, -1.0, 0, valid=False)
    assert island.add(invalid) and island.cell(invalid) == (-1, 0)
    assert [p.id for p in island.members()] == ["s", "w", "i"]
    import random

    config = arms.EvolveConfig(seed=3)
    picks = [arms.select_parent(island, random.Random(7), config).id for _ in range(5)]
    assert picks == [arms.select_parent(island, random.Random(7), config).id for _ in range(5)]


# --------------------------------------------------------------------- evaluator
@pytest.mark.slow
def test_evaluator_reproduces_pack_baseline_validation(pack: LinkPredictionPack, evaluator: CandidateEvaluator) -> None:
    validation = json.loads((PACK_ROOT / "baseline_validation.json").read_text())
    rows = {(r["scope"], r["family_id"], r["instance_id"], r["baseline"]): r for r in validation["rows"]}
    result = evaluator.evaluate(pack.baseline_source("ra"), label="ra")
    assert result.valid and not result.nondeterministic and result.families_passed == 0
    by_family = {f.family_id: f for f in result.families}
    assert by_family["ca_hepth"].ties == 16 and by_family["ca_hepth"].wins == 0
    dolphins_01 = next(r for r in result.per_instance if r["instance_id"] == "dolphins_01")
    assert dolphins_01["average_precision"] == pytest.approx(
        rows[("public", "dolphins", "dolphins_01", "ra")]["average_precision"])
    held = evaluator.heldout(pack.baseline_source("ra"), label="ra").heldout
    assert held and held["complete"]
    families = {f["family_id"]: f for f in held["families"]}
    for family_id, item in families.items():
        expected = [rows[("hidden", family_id, f"inst_{i:03d}", "ra")]["average_precision"] for i in range(1, 9)]
        assert item["candidate_mean"] == pytest.approx(sum(expected) / 8)
        assert item["relative_to_ra"] == pytest.approx(0.0)
    assert families["heldout_slot_01"]["candidate_mean"] != families["heldout_slot_02"]["candidate_mean"]


@pytest.mark.slow
def test_evaluator_flags_invalid_nondeterministic_and_hanging(pack: LinkPredictionPack, evaluator: CandidateEvaluator) -> None:
    broken = evaluator.evaluate(parse_program(BROKEN, parent_source=None, diff_based=False).source or "", label="broken")
    assert not broken.valid and broken.fitness == -1.0 and any("ZeroDivisionError" in e for e in broken.errors)
    assert broken.families_passed == 0 and not broken.qualified
    nondet = evaluator.evaluate(NONDETERMINISTIC, label="nondet")
    assert nondet.nondeterministic and not nondet.valid and nondet.fitness == -1.0
    quick = CandidateEvaluator(pack, evaluator.work_dir, timeout_seconds=1.0)
    hanging = quick.evaluate(HANGS, label="hang")
    assert not hanging.valid and any("wall-clock" in e for e in hanging.errors)


@pytest.mark.slow
def test_frozen_bundle_runs_under_pack_scorer(pack: LinkPredictionPack, tmp_path: Path) -> None:
    bundle = write_frozen_bundle(pack.baseline_source("ra"), tmp_path / "best")
    manifest = json.loads((bundle / "method_manifest.json").read_text())
    assert manifest["schema_version"] == "frozen-computational-method.v1"
    assert manifest["command"][:2] == ["python3", "run_method.py"]
    spec = pack.public_specs()[0]
    full = pack.load_instance(spec)
    (tmp_path / "instance.json").write_text(json.dumps(pack.project(full)))
    command = [part.format(instance_path=str(tmp_path / "instance.json"), seed=pack.seeds[0],
                           solution_path=str(tmp_path / "solution.json")) for part in manifest["command"]]
    command[0] = sys.executable
    subprocess.run(command, cwd=bundle, check=True, capture_output=True, text=True)
    metrics = pack.score(full, json.loads((tmp_path / "solution.json").read_text()))
    validation = json.loads((PACK_ROOT / "baseline_validation.json").read_text())
    expected = next(r for r in validation["rows"] if r["instance_id"] == spec.instance_id and r["baseline"] == "ra")
    assert metrics["average_precision"] == pytest.approx(expected["average_precision"])


# ------------------------------------------------------------------------ prompts
@pytest.mark.slow
def test_prompts_use_public_material_only(pack: LinkPredictionPack, evaluator: CandidateEvaluator) -> None:
    context = PromptContext.from_pack(pack, evaluator)
    assert {p["family_id"] for p in context.family_profile} == {"dolphins", "ca_hepth", "email_eu"}
    text = initial_prompt(context)
    for forbidden in ("GrQc", "facebook", "heldout_slot", "hidden_edges", "def jaccard", "def car("):
        assert forbidden not in text
    assert "strictly higher" in text and "2%" in text and "2 of 3" in text
    seed = evaluator.evaluate(CN, label="cn")
    prompt = mutation_prompt(context, parent_source=CN, parent_feedback=seed.summary(),
                             parent_per_instance=seed.per_instance, top_programs=[], inspirations=[],
                             diff_based=True)
    assert "<<<<<<< SEARCH" in prompt and "dolphins_01:" in prompt and "GrQc" not in prompt


# --------------------------------------------------------------------------- arms
@pytest.mark.slow
def test_evolve_scripted_end_to_end_and_heldout_gate(pack: LinkPredictionPack, evaluator: CandidateEvaluator,
                                                      tmp_path: Path) -> None:
    out = tmp_path / "evolve"
    client = LedgerClient(ScriptedClient([DIFF_TO_RA, FULL_HYBRID, BROKEN, GARBAGE, FULL_HYBRID]), out, max_calls=5)
    config = arms.EvolveConfig(max_llm_calls=5, islands=2, migration_interval=2, stop_on_qualify=False, seed=1)
    manifest = arms.run_evolve(pack, evaluator, client, out, config)
    outcome = manifest["outcome"]
    assert manifest["evaluations"] == 4 and outcome["stop_reason"] == "budget_exhausted"
    # the fifth response re-sends the hybrid: a duplicate, or a no-change if its parent was the hybrid
    assert outcome["parse_failures"] == 1 and outcome["duplicates"] + outcome["no_change"] == 1
    assert manifest["best"]["families_passed"] == 1 and not manifest["best"]["qualified"]
    events = [json.loads(line) for line in (out / "iterations.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events[:5]] == ["evaluated", "evaluated", "migration", "evaluated", "parse_failure"]
    assert events[5]["event"] in {"duplicate", "no_change"} and events[5]["program_id"] == manifest["best"]["program_id"]
    assert (out / "best" / "public_qualification.json").is_file() and (out / "archive.json").is_file()
    receipt = json.loads((out / "best" / "public_qualification.json").read_text())
    assert receipt["qualified"] is False and receipt["receipt"]["passed_family_count"] == 1
    # the hidden gate refuses an unqualified best, and labels an explicit diagnostic
    assert cli.main(["heldout", "--pack", str(PACK_ROOT), "--run", str(out)]) == 2
    assert not (out / "heldout").exists()
    assert cli.main(["heldout", "--pack", str(PACK_ROOT), "--run", str(out), "--diagnostic"]) == 0
    held = json.loads((out / "heldout" / "heldout.json").read_text())
    assert held["claim_status"] == "operator_diagnostic_only" and held["gate"]["diagnostic"] is True
    assert held["families_meeting_minimum"] == 0
    # no prompt ever carried hidden material
    for line in (out / "calls.jsonl").read_text().splitlines():
        item = json.loads(line)
        assert "GrQc" not in item["user"] and "heldout_slot" not in item["user"]


@pytest.mark.slow
def test_evolve_replay_from_cache_walks_the_same_path(pack: LinkPredictionPack, evaluator: CandidateEvaluator,
                                                      tmp_path: Path) -> None:
    first = tmp_path / "first"
    client = LedgerClient(ScriptedClient([DIFF_TO_RA, FULL_HYBRID]), first, max_calls=2)
    arms.run_evolve(pack, evaluator, client, first, arms.EvolveConfig(islands=1, stop_on_qualify=False, seed=5))
    second = tmp_path / "second"
    replay = LedgerClient(ScriptedClient(["unused"]), second, max_calls=2, cache_path=first / "llm_cache.jsonl")
    arms.run_evolve(pack, evaluator, replay, second, arms.EvolveConfig(islands=1, stop_on_qualify=False, seed=5))
    assert replay.physical_calls == 0 and replay.calls_used == 2
    def ids(path: Path) -> list[str | None]:
        return [json.loads(line).get("program_id")
                for line in (path / "iterations.jsonl").read_text().splitlines()]

    assert ids(first) == ids(second)


@pytest.mark.slow
def test_best_of_n_scripted(pack: LinkPredictionPack, evaluator: CandidateEvaluator, tmp_path: Path) -> None:
    out = tmp_path / "bon"
    client = LedgerClient(ScriptedClient([DIFF_TO_RA, FULL_HYBRID, BROKEN, GARBAGE]), out, max_calls=4)
    manifest = arms.run_best_of_n(pack, evaluator, client, out, samples=4, seed=0)
    assert manifest["evaluations"] == 2 and manifest["outcome"]["parse_failures"] == 2
    assert manifest["best"]["families_passed"] == 1
    hashes = {json.loads(line)["prompt_sha256"] for line in (out / "calls.jsonl").read_text().splitlines()}
    assert len(hashes) == 4


@pytest.mark.slow
def test_classical_arm(pack: LinkPredictionPack, evaluator: CandidateEvaluator, tmp_path: Path) -> None:
    sources = arms.classical_sources(pack)
    assert set(sources) == set(arms.CLASSICAL_HEURISTICS)
    functions = {"cn": "common_neighbors", "car": "car"}
    assert all(sources[name].rstrip().endswith(f"score_candidates = {function}")
               for name, function in functions.items())
    manifest = arms.run_classical(pack, evaluator, tmp_path / "classical", heuristics=("cn",))
    assert manifest["evaluations"] == 1 and manifest["best"]["origin"] == "classical:cn"
    assert manifest["outcome"]["heldout_diagnostic"] is False
    table = json.loads((tmp_path / "classical" / "classical_table.json").read_text())
    assert table["cn"]["public"]["families_passed"] == 0 and "heldout" not in table["cn"]


# ------------------------------------------------------------------------- export
def test_export_openevolve_project(pack: LinkPredictionPack, tmp_path: Path) -> None:
    manifest = export_openevolve_project(pack, tmp_path / "oe", model="deployment-x",
                                         api_base="https://example.openai.azure.com/openai/v1/", max_iterations=7)
    assert set(manifest["files"]) >= {"config.yaml", "evaluator.py", "initial_program.py", "run_openevolve.py"}
    import yaml

    config = yaml.safe_load((tmp_path / "oe" / "config.yaml").read_text())
    assert config["max_iterations"] == 7 and config["llm"]["models"][0]["name"] == "deployment-x"
    assert config["database"]["feature_dimensions"] == ["families_passed", "complexity"]
    initial = (tmp_path / "oe" / "initial_program.py").read_text()
    assert "# EVOLVE-BLOCK-START" in initial and "# EVOLVE-BLOCK-END" in initial
    compile((tmp_path / "oe" / "evaluator.py").read_text(), "evaluator.py", "exec")
    compile((tmp_path / "oe" / "run_openevolve.py").read_text(), "run_openevolve.py", "exec")
    assert str(pack.root) in (tmp_path / "oe" / "evaluator.py").read_text()


def test_program_id_is_content_hash() -> None:
    assert program_id(CN) == program_id(CN) and program_id(CN) != program_id(CN + "\n#")
