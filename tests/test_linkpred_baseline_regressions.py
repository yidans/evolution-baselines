"""Exercise collaborator entry points, receipt identity and diagnostic provenance."""
from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from evolution_baselines import arms, cli
from evolution_baselines.evaluate import CandidateEvaluator, program_id
from evolution_baselines.llm import LedgerClient, ScriptedClient, build_client
from evolution_baselines.openevolve_export import export_openevolve_project
from evolution_baselines.pack import LinkPredictionPack

PACK_ROOT = Path(__file__).resolve().parents[1] / "src/evolution_baselines/benchmarks/local_link_prediction_v2"


@pytest.fixture
def pack() -> LinkPredictionPack:
    return LinkPredictionPack(PACK_ROOT)


def single_instance_evaluator(pack, work_dir):
    evaluator = CandidateEvaluator(pack, work_dir)
    spec = pack.public_specs()[0]
    full = pack.load_instance(spec)
    evaluator._instances["public"] = [(spec, full, pack.project(full))]
    return evaluator, spec


def test_worker_accepts_relative_output_directory(pack, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    evaluator, spec = single_instance_evaluator(pack, Path("runs/test/evaluator_work"))
    raw = evaluator._run_worker(arms.CN_SEED_PROGRAM, "public", label="cn")
    assert raw[spec.key]["status"] == "passed", raw[spec.key].get("error")
    assert pack.score(pack.load_instance(spec), raw[spec.key]["solution"])["schema_error_count"] == 0


def test_worker_rejects_matrix_scores_like_the_frozen_adapter(pack, tmp_path):
    evaluator, spec = single_instance_evaluator(pack, tmp_path)
    source = "def score_candidates(adj, candidates):\n    return [[1.0] for _ in candidates]\n"
    raw = evaluator._run_worker(source, "public", label="matrix")
    assert raw[spec.key]["status"] == "failed"
    assert "shape" in raw[spec.key]["error"]


def test_exported_evaluator_can_be_imported(pack, tmp_path):
    export_openevolve_project(pack, tmp_path, model="example", api_base="https://example.com/")
    module = runpy.run_path(str(tmp_path / "evaluator.py"))
    assert callable(module["evaluate"])


def test_exported_runner_reports_missing_credentials(pack, tmp_path):
    export_openevolve_project(pack, tmp_path, model="example", api_base="https://example.com/")
    env = {k: v for k, v in os.environ.items() if k not in {"OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"}}
    result = subprocess.run([sys.executable, str(tmp_path / "run_openevolve.py")],
                            env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert "set OPENAI_API_KEY" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("mismatch", ["program", "pack"])
def test_heldout_refuses_receipt_from_another_program_or_pack(pack, tmp_path, monkeypatch, mismatch):
    best = tmp_path / "best"
    best.mkdir()
    best.joinpath("solution.py").write_text(arms.CN_SEED_PROGRAM)
    best.joinpath("public_qualification.json").write_text(json.dumps({
        "qualified": True,
        "program_id": "other" if mismatch == "program" else program_id(arms.CN_SEED_PROGRAM),
        "pack": "other" if mismatch == "pack" else pack.contract_id,
    }))

    def unexpected_evaluation(*args, **kwargs):
        pytest.fail("mismatched receipt reached hidden evaluation")

    monkeypatch.setattr(cli, "_evaluator", unexpected_evaluation)
    assert cli.main(["heldout", "--pack", str(PACK_ROOT), "--run", str(tmp_path)]) == 2
    assert not (tmp_path / "heldout").exists()


def test_replay_uses_recorded_request_model_without_live_client(tmp_path):
    original = LedgerClient(ScriptedClient(["answer"]), tmp_path / "first", max_calls=1)
    original.complete(system="system", user="question", tag="first")
    replay = build_client("replay", out_dir=tmp_path / "replay", max_calls=1,
                          cache_path=tmp_path / "first/llm_cache.jsonl")
    result = replay.complete(system="system", user="question", tag="replay")
    assert result.text == "answer" and result.cached
    assert replay.calls_used == 1 and replay.physical_calls == 0


def test_ledger_rejects_existing_run_without_changing_its_calls(tmp_path):
    client = LedgerClient(ScriptedClient(["answer"]), tmp_path, max_calls=1)
    client.complete(system="s", user="u", tag="t")
    before = (tmp_path / "calls.jsonl").read_bytes()
    with pytest.raises(FileExistsError, match="new.*directory"):
        LedgerClient(ScriptedClient(["other"]), tmp_path, max_calls=1)
    assert (tmp_path / "calls.jsonl").read_bytes() == before


def test_recorder_rejects_existing_results(pack, tmp_path):
    recorder = arms.RunRecorder(tmp_path, arm="classical", pack=pack, config={})
    recorder.finish(best=None, programs=[], client=None, outcome={})
    before = (tmp_path / "run_manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="new.*directory"):
        arms.RunRecorder(tmp_path, arm="classical", pack=pack, config={})
    assert (tmp_path / "run_manifest.json").read_bytes() == before


def test_public_results_do_not_claim_container_isolation(pack, tmp_path):
    result = CandidateEvaluator(pack, tmp_path).evaluate(arms.CN_SEED_PROGRAM)
    assert result.valid and not result.qualified
    assert result.summary()["claim_status"] == "operator_diagnostic_only"
    assert result.receipt["trusted_isolation"] is False
    assert all(row["trusted_isolation"] is False for row in result.rows)


def test_qualified_local_hidden_run_is_still_a_diagnostic(pack, tmp_path):
    best = tmp_path / "best"
    best.mkdir()
    best.joinpath("solution.py").write_text(arms.CN_SEED_PROGRAM)
    best.joinpath("public_qualification.json").write_text(json.dumps({
        "qualified": True, "program_id": program_id(arms.CN_SEED_PROGRAM), "pack": pack.contract_id,
    }))
    assert cli.main(["heldout", "--pack", str(PACK_ROOT), "--run", str(tmp_path), "--diagnostic"]) == 0
    result = json.loads((tmp_path / "heldout/heldout.json").read_text())
    assert result["claim_status"] == "operator_diagnostic_only"
    assert result["gate"]["qualified"] is True and result["complete"] is True


@pytest.mark.parametrize("runtime_failure", ["execution", "headroom"])
def test_screening_requires_every_runtime_envelope(pack, tmp_path, monkeypatch, runtime_failure):
    evaluator = CandidateEvaluator(pack, tmp_path)
    raw = {}
    baselines = {}
    failed_key = None
    # Simulate worker results at the subprocess boundary. Public fixture labels
    # create perfect scoring outputs; no oracle code is written into a candidate.
    for spec, full, projected in evaluator.instances("public"):
        positives = {tuple(sorted(edge)) for edge in full["hidden_edges"]}
        solution = {"schema_version": "local-link-prediction-solution.v1", "scores": [
            float(tuple(sorted(pair)) in positives) for pair in projected["candidates"]]}
        metrics = {**pack.score(full, solution), "average_precision": 0.1,
                   "controller_validity_pass": 1.0, "seconds": 0.01}
        baselines[spec.key] = {name: dict(metrics) for name in pack.baseline_ids}
        raw[spec.key] = {"status": "passed", "seconds": 0.01, "error": None, "solution": solution}
        if failed_key is None and spec.role == "runtime_envelope":
            failed_key = spec.key
    assert failed_key is not None
    if runtime_failure == "execution":
        raw[failed_key].update(status="failed", error="worker failed", solution=None)
    else:
        raw[failed_key]["seconds"] = pack.timeout_seconds + 1.0
    monkeypatch.setattr(evaluator, "baseline_metrics", lambda scope: baselines)
    monkeypatch.setattr(evaluator, "_run_worker", lambda *args, **kwargs: raw)
    result = evaluator.evaluate(arms.CN_SEED_PROGRAM)
    assert result.receipt["passed_family_count"] == 2
    assert result.receipt["runtime_envelope_pass"] is False
    assert result.valid is (runtime_failure == "headroom")
    assert not result.qualified
