from __future__ import annotations

import torch
import pytest
import sys
from types import SimpleNamespace

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import run_qtriad_relation_transfer as formal_runner

from q_attention.plugins.q_triad import QTriadAttentionScoreKernel


def _inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(41)
    batch, heads, tokens, dim = 2, 2, 5, 4
    attention_mask = torch.ones(batch, tokens, dtype=torch.bool)
    attention_mask[1, -1] = False
    subject_mask = torch.zeros_like(attention_mask)
    object_mask = torch.zeros_like(attention_mask)
    subject_mask[:, 1] = True
    object_mask[:, 3] = True
    return {
        "query": torch.randn(batch, heads, tokens, dim),
        "key": torch.randn(batch, heads, tokens, dim),
        "scores": torch.randn(batch, heads, tokens, tokens),
        "attention_mask": attention_mask,
        "subject_mask": subject_mask,
        "object_mask": object_mask,
        "layer_index": torch.tensor(0),
    }


def _kernel(pair_chunk_size: int) -> QTriadAttentionScoreKernel:
    return QTriadAttentionScoreKernel(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        num_qubits=2,
        circuit_depth=1,
        seed=29,
        pair_chunk_size=pair_chunk_size,
    )


def _call(kernel: QTriadAttentionScoreKernel, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return kernel(
        inputs["query"],
        inputs["key"],
        scores=inputs["scores"],
        layer_index=int(inputs["layer_index"].item()),
        attention_mask=inputs["attention_mask"],
        subject_mask=inputs["subject_mask"],
        object_mask=inputs["object_mask"],
    )


def test_chunked_score_matches_single_chunk_and_gradients() -> None:
    inputs = _inputs()
    reference = _kernel(10_000)
    chunked = _kernel(3)
    chunked.load_state_dict(reference.state_dict())
    reference_residual = _call(reference, inputs)
    chunked_residual = _call(chunked, inputs)
    assert torch.allclose(chunked_residual, reference_residual, atol=2e-6, rtol=2e-6)

    reference.zero_grad(set_to_none=True)
    chunked.zero_grad(set_to_none=True)
    _call(reference, inputs).square().mean().backward()
    _call(chunked, inputs).square().mean().backward()
    for reference_parameter, chunked_parameter in zip(reference.parameters(), chunked.parameters()):
        assert reference_parameter.grad is not None
        assert chunked_parameter.grad is not None
        assert torch.allclose(chunked_parameter.grad, reference_parameter.grad, atol=2e-5, rtol=2e-5)


def test_pair_chunk_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="pair_chunk_size"):
        _kernel(0)


def test_selector_scheduler_reuses_first_free_gpu(tmp_path, monkeypatch) -> None:
    class FakeProcess:
        _next_pid = 4100

        def __init__(self, command, **_kwargs):
            self.pid = FakeProcess._next_pid
            FakeProcess._next_pid += 1
            self.return_code = 0
            selector = command[command.index("--selector") + 1]
            output_dir = __import__("pathlib").Path(
                command[command.index("--output-dir") + 1]
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "metrics.json").write_text("{}", encoding="utf-8")

        def poll(self):
            return self.return_code

        def wait(self, **_kwargs):
            return self.return_code

    monkeypatch.setattr(formal_runner.subprocess, "Popen", FakeProcess)
    args = SimpleNamespace(python_bin=sys.executable, log_every_batches=1)
    statuses = formal_runner.run_selector_workers(
        selectors=["q_triad", "classical_density_tensor", "quantum_product"],
        gpu_ids=[2, 5],
        args=args,
        config_path=tmp_path / "config.json",
        baseline_dir=tmp_path / "baseline",
        data_dir=tmp_path / "data",
        run_dir=tmp_path / "run",
        seed=13,
    )
    assert [statuses[name]["status"] for name in statuses] == ["complete"] * 3
    assert statuses["q_triad"]["gpu"] == 2
    assert statuses["classical_density_tensor"]["gpu"] == 5
    assert statuses["quantum_product"]["gpu"] == 2


def test_gpu_ids_follow_physical_ids_exposed_by_cuda_visible_devices(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert formal_runner.resolve_gpu_ids("2,5", "cuda") == [2, 5]


def test_auto_gpu_selects_all_sufficient_physical_devices(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    inventory = [
        {"index": 0, "name": "A100", "memory_total_mib": 40 * 1024, "memory_free_mib": 39 * 1024, "memory_used_mib": 1024},
        {"index": 1, "name": "A100", "memory_total_mib": 40 * 1024, "memory_free_mib": 38 * 1024, "memory_used_mib": 2 * 1024},
    ]
    assert formal_runner.resolve_gpu_ids("auto", "cuda", inventory) == [0, 1]


def test_auto_gpu_resolution_does_not_initialize_torch_cuda(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(AssertionError("CUDA initialized")))
    inventory = [
        {"index": 0, "name": "A100", "memory_total_mib": 40 * 1024, "memory_free_mib": 39 * 1024, "memory_used_mib": 1024},
    ]
    assert formal_runner.resolve_gpu_ids("auto", "cuda", inventory) == [0]


def test_auto_profile_is_conservative_on_small_or_busy_gpu() -> None:
    config = {"kernel": {"pair_chunk_size": 256}}
    low_inventory = [
        {"index": 0, "name": "3080 Ti", "memory_total_mib": 12 * 1024, "memory_free_mib": 11 * 1024, "memory_used_mib": 1024},
    ]
    low = formal_runner.choose_hardware_profile("auto", config, [0], low_inventory)
    assert low["name"] == "low_memory"
    assert low["pair_chunk_size"] == 64
    assert low["activation_checkpointing"] is True

    high_inventory = [
        {"index": 0, "name": "A100", "memory_total_mib": 80 * 1024, "memory_free_mib": 78 * 1024, "memory_used_mib": 2 * 1024},
    ]
    high = formal_runner.choose_hardware_profile("auto", config, [0], high_inventory)
    assert high["name"] == "high_memory"
    assert high["pair_chunk_size"] == 1024
    assert high["activation_checkpointing"] is False


def test_auto_profile_uses_the_weakest_selected_gpu() -> None:
    config = {"kernel": {"pair_chunk_size": 256}}
    inventory = [
        {"index": 0, "name": "A100", "memory_total_mib": 80 * 1024, "memory_free_mib": 70 * 1024, "memory_used_mib": 10 * 1024},
        {"index": 1, "name": "A100", "memory_total_mib": 40 * 1024, "memory_free_mib": 20 * 1024, "memory_used_mib": 20 * 1024},
    ]
    profile = formal_runner.choose_hardware_profile("auto", config, [0, 1], inventory)
    assert profile["name"] == "balanced"
    assert profile["minimum_memory_total_mib"] == 40 * 1024
    assert profile["minimum_memory_free_mib"] == 20 * 1024
