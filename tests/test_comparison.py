"""Fair comparison behavior: public inputs, cost accounting and matched runs."""
import json
import shutil

import pytest

from evolution_baselines.cli import DEFAULT_PACK
from evolution_baselines.llm import BudgetExhausted, LedgerClient, ScriptedClient
from evolution_baselines.pack import LinkPredictionPack


def test_failed_request_consumes_budget_and_is_recorded(tmp_path):
    class FailingClient:
        model = "test"

        def complete(self, **kwargs):
            raise ConnectionError("transport unavailable")

    client = LedgerClient(FailingClient(), tmp_path, max_calls=1)
    with pytest.raises(ConnectionError, match="transport unavailable"):
        client.complete(system="s", user="u", tag="attempt")
    assert client.calls_used == 1
    assert client.physical_calls == 1
    row = json.loads((tmp_path / "calls.jsonl").read_text())
    assert row["status"] == "failed" and row["error_type"] == "ConnectionError"
    with pytest.raises(BudgetExhausted):
        client.complete(system="s", user="u", tag="retry")


def test_public_pack_does_not_require_private_files(tmp_path):
    root = tmp_path / "public-pack"
    root.mkdir()
    shutil.copy2(DEFAULT_PACK / "public_contract.json", root)
    shutil.copytree(DEFAULT_PACK / "development_benchmark_bundle", root / "development_benchmark_bundle")
    shutil.copytree(DEFAULT_PACK / "public", root / "public")
    pack = LinkPredictionPack(root)
    spec = pack.public_specs()[0]
    assert pack.score(pack.load_instance(spec), None)["schema_error_count"] > 0
    assert "score_candidates" in pack.baseline_source("cn")
    with pytest.raises(FileNotFoundError):
        pack.hidden_specs()


def test_zero_budget_is_not_replaced_by_default(tmp_path):
    client = LedgerClient(ScriptedClient(["unused"]), tmp_path, max_calls=0)
    with pytest.raises(BudgetExhausted):
        client.complete(system="s", user="u", tag="zero")
    assert client.physical_calls == 0


def test_cache_separates_reasoning_settings(tmp_path):
    original = ScriptedClient(["original"])
    original.reasoning_effort = "low"
    first = LedgerClient(original, tmp_path / "first", max_calls=1)
    first.complete(system="s", user="u", tag="first")
    changed = ScriptedClient(["changed"])
    changed.reasoning_effort = "high"
    second = LedgerClient(changed, tmp_path / "second", max_calls=1,
                          cache_path=tmp_path / "first/llm_cache.jsonl")
    assert second.complete(system="s", user="u", tag="second").text == "changed"


def test_comparison_runs_both_arms_at_same_budget_without_hidden_data(tmp_path):
    from evolution_baselines import cli

    response_file = tmp_path / "responses.json"
    response_file.write_text(json.dumps(["def score_candidates(adj, candidates):\n    return [0.0] * len(candidates)\n"]))
    out = tmp_path / "comparison"
    assert cli.main(["compare", "--out", str(out), "--max-calls", "1", "--seeds", "7",
                     "--client", "scripted", "--scripted-file", str(response_file),
                     "--evaluator", "subprocess"]) == 0
    summary = json.loads((out / "comparison.json").read_text())
    assert summary["complete"] and summary["stopping_rule"] == "fixed_call_budget"
    assert {run["arm"] for run in summary["runs"]} == {"evolve", "best-of-n"}
    for run in summary["runs"]:
        assert run["seed"] == 7
        assert run["client"]["calls_used"] == run["client"]["max_calls"] == 1
        assert not (out / run["path"] / "heldout").exists()
        assert run["claim_status"] == "operator_diagnostic_only"
    rows = json.loads((out / "comparison_rows.json").read_text())
    assert {row["client_kind"] for row in rows} == {"ScriptedClient"}
    assert all(row["physical_calls"] == 1 and row["prompt_tokens"] > 0 for row in rows)


def test_compare_defaults_to_controller_evaluation():
    from evolution_baselines.cli import build_parser

    args = build_parser().parse_args(["compare", "--out", "example", "--max-calls", "10"])
    assert args.evaluator == "galahad"
    assert not args.stop_on_qualify


def test_comparison_rejects_a_shared_cache_before_starting(tmp_path):
    from evolution_baselines.cli import main
    with pytest.raises(SystemExit, match="independent caches"):
        main(["compare", "--out", str(tmp_path / "run"), "--max-calls", "1", "--cache", "shared.jsonl"])
    assert not (tmp_path / "run").exists()


def test_replay_preserves_recorded_reasoning_settings(tmp_path):
    from evolution_baselines.llm import build_client
    original = ScriptedClient(["answer"])
    original.reasoning_effort = "high"
    original.max_output_tokens = 1234
    client = LedgerClient(original, tmp_path / "first", max_calls=1)
    client.complete(system="s", user="u", tag="original")
    replay = build_client("replay", out_dir=tmp_path / "replay", max_calls=1,
                          cache_path=tmp_path / "first/llm_cache.jsonl")
    assert replay.complete(system="s", user="u", tag="replay").text == "answer"
    assert replay.usage()["reasoning_effort"] == "high"
    assert replay.usage()["max_output_tokens"] == 1234
    assert replay.physical_calls == 0


def test_controller_backend_rejects_unsafe_host_evaluation(tmp_path, monkeypatch):
    isolation = pytest.importorskip("ai_professor.autonomy.isolation")
    from evolution_baselines.galahad import GalahadEvaluator

    monkeypatch.setattr(isolation, "_selected_process_sandbox", lambda: "unsafe-host-opt-in")
    with pytest.raises(RuntimeError, match="trusted process isolation"):
        GalahadEvaluator(LinkPredictionPack(DEFAULT_PACK), tmp_path)


def test_controller_executes_candidate_and_denies_private_file_access(tmp_path):
    pytest.importorskip("ai_professor.autonomy.computational_project")
    from evolution_baselines.evaluate import write_frozen_bundle
    from evolution_baselines.galahad import GalahadEvaluator

    pack = LinkPredictionPack(DEFAULT_PACK)
    evaluator = GalahadEvaluator(pack, tmp_path / "work")
    spec = pack.public_specs()[0]
    secret = tmp_path / "private-labels.json"
    secret.write_text("must not reach candidate")
    probe = (
        "from pathlib import Path\n"
        "def score_candidates(adj, candidates):\n"
        f"    Path({str(secret)!r}).read_text()\n"
        "    return [0.0] * len(candidates)\n"
    )
    for name, source in [("cn", pack.baseline_source("cn")), ("file-probe", probe)]:
        bundle = write_frozen_bundle(source, tmp_path / name)
        run_dir = tmp_path / f"run-{name}"
        run_dir.mkdir()
        row = evaluator._runner._evaluate_one_heldout_run(
            family_id=spec.family_id, instance_id=spec.instance_id, instance_seed=spec.instance_seed,
            config_path=spec.config_path, seed=pack.seeds[0], run_dir=run_dir, algorithm_dir=bundle,
            method_manifest=json.loads((bundle / "method_manifest.json").read_text()),
            benchmark_dir=pack.dev_bundle, benchmark_manifest=pack.dev_manifest,
            protocol=pack.public_contract, timeout=pack.timeout_seconds,
        )
        assert row["trusted_isolation"] is True
        if name == "cn":
            assert row["status"] == "passed", row["error"]
            assert row["candidate_metrics"]["average_precision"] == pytest.approx(0.10689692982456139)
            assert row["candidate_metrics"]["average_precision"] == row["baseline_metrics"]["cn"]["average_precision"]
        else:
            assert row["status"] == "candidate_failed"
            assert "PermissionError" in row["candidate_execution"]["stderr"]


@pytest.mark.slow
def test_selected_program_reaches_existing_controller_qualification(tmp_path):
    module = pytest.importorskip("ai_professor.autonomy.computational_project")
    from ai_professor.autonomy.project_types import tree_hash
    from evolution_baselines.evaluate import program_id
    from evolution_baselines.galahad import qualify_run

    pack = LinkPredictionPack(DEFAULT_PACK)
    project = module.MethodProject.create(tmp_path / "projects", "baseline-check")
    dev = project.stage(2) / "outputs" / "development_benchmark_bundle"
    shutil.copytree(pack.dev_bundle, dev)
    protocol = {
        **pack.public_contract,
        "hidden_benchmark_contract_id": pack.contract_id,
        "development_families": [{"id": family["id"], "description": family["description"],
                                  "generator": pack.dev_manifest["generator_command"][1]}
                                 for family in pack.dev_manifest["development_families"]],
        "heldout_family_ids": pack.public_contract["heldout_slots"],
        "heldout_isolation_level": "family_hidden",
        "hidden_family_details_visible_before_freeze": False,
    }
    project.lock_benchmark(
        protocol, hidden_registry_path=pack.root / "hidden_registry.json",
        benchmark_bundle_path=pack.bundle, development_benchmark_bundle_path=dev,
        bundle_origin="controller_pack", public_contract_path=pack.root / "public_contract.json",
        benchmark_pack_sha256=tree_hash(pack.root),
    )
    run = tmp_path / "baseline-run"
    (run / "best").mkdir(parents=True)
    source = pack.baseline_source("cn")
    (run / "best" / "solution.py").write_text(source)
    (run / "run_manifest.json").write_text(json.dumps({
        "pack": pack.contract_id, "best": {"program_id": program_id(source)},
    }))
    receipt = qualify_run(run, project.path)
    assert receipt["status"] == "fail"  # CN cannot strictly beat itself or the stronger references.
    assert receipt["passed_family_count"] == 0
    assert receipt["all_candidate_runs_valid"] is True
    assert receipt["all_baseline_runs_valid"] is True
    assert (run / "galahad_qualification.json").is_file()
    assert not (project.stage(4) / "freeze_certificate.json").exists()
    assert not list((project.stage(5) / "outputs").iterdir())


def test_openevolve_export_uses_selected_controller_backend(tmp_path):
    import runpy
    pytest.importorskip("ai_professor.autonomy.computational_project")
    from evolution_baselines.galahad import GalahadEvaluator
    from evolution_baselines.openevolve_export import export_openevolve_project

    export_openevolve_project(LinkPredictionPack(DEFAULT_PACK), tmp_path,
                              evaluator_backend="galahad")
    module = runpy.run_path(str(tmp_path / "evaluator.py"))
    assert isinstance(module["_evaluator"](), GalahadEvaluator)


def test_comparison_preserves_failed_model_attempt_and_stops(tmp_path):
    from evolution_baselines.comparison import run_comparison
    from evolution_baselines.evaluate import CandidateEvaluator

    class OfflineFailure:
        model = "offline-test"

        def complete(self, **kwargs):
            raise ConnectionError("simulated unavailable transport")

    result = run_comparison(
        LinkPredictionPack(DEFAULT_PACK), tmp_path / "comparison", seeds=[1, 2], max_calls=5,
        client_factory=lambda out, limit: LedgerClient(OfflineFailure(), out, max_calls=limit),
        evaluator_factory=lambda pack, out: CandidateEvaluator(pack, out / "work"),
    )
    assert result["complete"] is False
    assert len(result["runs"]) == 1
    run = result["runs"][0]
    assert run["outcome"]["stop_reason"] == "client_error"
    assert run["client"]["calls_used"] == run["client"]["physical_calls"] == 1
    assert run["client"]["failed_calls"] == run["client"]["unknown_usage_calls"] == 1
    manifest = tmp_path / "comparison" / run["path"] / "run_manifest.json"
    assert json.loads(manifest.read_text())["outcome"]["stop_reason"] == "client_error"
