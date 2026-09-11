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
