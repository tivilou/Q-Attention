from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "experiments" / "run_retacred_qceasc_formal_single_seed.py"
TOY_PATH = ROOT / "experiments" / "run_q_ceasc_toy.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adaptive_policy_prefers_micro_batch_before_chunking():
    runner = _load("qceasc_formal_runner_contract", RUNNER_PATH)
    names = [profile["name"] for profile in runner.ADAPTIVE_HARDWARE_PROFILES]
    assert names[:4] == [
        "adaptive_full_batch",
        "adaptive_micro_128",
        "adaptive_micro_64",
        "adaptive_micro_32",
    ]
    assert all("chunk_" not in name for name in names[:4])
    assert names[4].startswith("adaptive_chunk_2x_micro_32")
    assert runner.ADAPTIVE_MEMORY_STATE_SCHEMA.endswith(".v2")


def test_rating_policy_keeps_l1_and_relative_qi_gates_separate():
    runner = _load("qceasc_formal_rating_contract", RUNNER_PATH)
    config = __import__("json").loads(
        (ROOT / "configs" / "retacred_qceasc_formal_single_seed.json").read_text(encoding="utf-8")
    )
    assert config["gates"]["minimum_candidate_minus_disabled_macro_f1"] == 0.001
    assert runner.RATING_POLICY["id"] == "q-attention-utility-and-qi-v1"
    assert runner.relative_metric_gain(0.0001, 0.2) == 0.0005
    assert runner.relative_metric_gain(0.1, 0.0) is None


def test_selector_worker_invocation_uses_checked_in_worker_path():
    runner = _load("qceasc_formal_runner_worker_path_contract", RUNNER_PATH)
    assert runner.SELECTOR_WORKER_PATH.name == "run_qceasc_selector_worker.py"
    assert runner.SELECTOR_WORKER_PATH.is_file()
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "run_q_ceasc_selector_worker.py" not in source
    assert "str(SELECTOR_WORKER_PATH)" in source


def test_shuffled_generation_preserves_query_and_swaps_context():
    toy = _load("qceasc_toy_contract", TOY_PATH)
    query = torch.tensor([[1.0, 2.0]])
    first_key = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    second_key = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    entity = torch.zeros_like(valid)
    manifest = {
        "generation": {
            "corrupted_primary": {
                "query": query,
                "key": first_key,
                "valid": valid,
                "entity": entity,
            },
            "out_of_bank": {
                "query": query + 10.0,
                "key": second_key,
                "valid": valid,
                "entity": entity,
            },
        }
    }
    shuffled, source = toy._shuffled_generation_case(manifest, "corrupted_primary")
    assert source == "out_of_bank"
    assert torch.equal(shuffled["query"], query)
    assert torch.equal(shuffled["key"], second_key)
    assert not torch.equal(shuffled["key"], first_key)


def test_sample_trace_validator_rejects_missing_observed_stage():
    validator = _load("sample_trace_validator", ROOT / "scripts" / "validate_sample_trace.py")
    trace = {
        "schema_version": "sample-trace.v1",
        "trace_id": "fixture",
        "experiment": {"run_id": "run", "dataset": "valid", "code_revision": "abc", "config_sha256": "hash", "model_identity": "model"},
        "sample_selection": {"rule": "fixed", "population_scope": "valid", "selected_count": 1, "selected_sample_ids": ["s0"]},
        "coverage": {stage: ("not_applicable" if stage in {"retrieval", "generation"} else "observed") for stage in ("data", "preprocess", "training", "retrieval", "scoring", "selection", "context", "generation", "evaluation", "diagnosis")},
        "samples": [{"sample_id": "s0", "stages": [{"stage": "data", "status": "observed", "observed_fields": {"tokens": ["x"]}}]}],
    }
    errors = validator.validate(trace)
    assert any("observed coverage stage" in error for error in errors)


def test_sample_trace_validator_accepts_explicit_not_applicable_stages():
    validator = _load("sample_trace_validator_valid", ROOT / "scripts" / "validate_sample_trace.py")
    stages = []
    for name in ("data", "preprocess", "training", "scoring", "selection", "context", "evaluation", "diagnosis"):
        stages.append({"stage": name, "status": "observed", "observed_fields": {"value": name}})
    stages.extend([
        {"stage": "retrieval", "status": "not_applicable"},
        {"stage": "generation", "status": "not_applicable"},
    ])
    trace = {
        "schema_version": "sample-trace.v1",
        "trace_id": "fixture",
        "experiment": {"run_id": "run", "dataset": "valid", "code_revision": "abc", "config_sha256": "hash", "model_identity": "model"},
        "sample_selection": {"rule": "fixed", "population_scope": "valid", "selected_count": 1, "selected_sample_ids": ["s0"]},
        "coverage": {stage: ("not_applicable" if stage in {"retrieval", "generation"} else "observed") for stage in ("data", "preprocess", "training", "retrieval", "scoring", "selection", "context", "generation", "evaluation", "diagnosis")},
        "samples": [{"sample_id": "s0", "stages": stages}],
    }
    assert validator.validate(trace) == []
    assert validator.main(["trace.json", "--expected-config-sha256", "wrong"]) == 2
