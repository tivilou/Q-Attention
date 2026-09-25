from __future__ import annotations

import pytest
import torch

from q_attention.adapters.attention_scores import AttentionScoreHookConfig, AttentionScoreKernelAdapter
from q_attention.models import RelationExtractionModel, RelationTransformerConfig
from q_attention.plugins.q_triad import QTriadAttentionScoreKernel


def _inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(7)
    input_ids = torch.randint(0, 20, (2, 6))
    attention_mask = torch.ones(2, 6, dtype=torch.bool)
    subject_mask = torch.zeros(2, 6, dtype=torch.bool)
    object_mask = torch.zeros(2, 6, dtype=torch.bool)
    subject_mask[:, 1] = True
    object_mask[:, 4] = True
    return input_ids, attention_mask, subject_mask, object_mask


def test_layer_sharded_cpu_canary_matches_serial_model() -> None:
    config = RelationTransformerConfig(
        vocab_size=20,
        num_labels=3,
        dim=16,
        num_layers=2,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
    )
    torch.manual_seed(11)
    serial = RelationExtractionModel(config).eval()
    sharded = RelationExtractionModel(config).eval()
    sharded.load_state_dict(serial.state_dict())
    sharded.configure_model_parallel((torch.device("cpu"), torch.device("cpu")))

    assert sharded.model_parallel_enabled
    assert sharded.model_parallel_layer_devices == (torch.device("cpu"), torch.device("cpu"))
    assert sharded.model_parallel_metadata()["module_devices"]["classifier"] == "cpu"
    inputs = _inputs()
    assert torch.equal(serial(*inputs), sharded(*inputs))


def test_model_parallel_requires_at_least_two_devices() -> None:
    config = RelationTransformerConfig(vocab_size=10, num_labels=2, num_layers=2)
    with pytest.raises(ValueError, match="at least two"):
        RelationExtractionModel(config).configure_model_parallel((torch.device("cpu"),))


def test_model_parallel_does_not_accept_more_devices_than_layers() -> None:
    config = RelationTransformerConfig(vocab_size=10, num_labels=2, num_layers=1)
    with pytest.raises(ValueError, match="cannot exceed"):
        RelationExtractionModel(config).configure_model_parallel(
            (torch.device("cpu"), torch.device("cpu"))
        )


def test_qtriad_kernel_records_layer_device_mapping() -> None:
    kernel = QTriadAttentionScoreKernel(
        num_layers=2,
        num_heads=2,
        head_dim=4,
        num_qubits=2,
        circuit_depth=1,
    )
    kernel.configure_model_parallel((torch.device("cpu"), torch.device("cpu")))
    metadata = kernel.metadata()
    assert metadata["model_parallel"] is True
    assert metadata["model_parallel_layer_devices"] == ["cpu", "cpu"]


def test_model_parallel_qtriad_hook_forward_is_finite() -> None:
    config = RelationTransformerConfig(
        vocab_size=20,
        num_labels=3,
        dim=16,
        num_layers=2,
        num_heads=4,
        ff_dim=32,
        dropout=0.0,
    )
    model = RelationExtractionModel(config).eval()
    model.configure_model_parallel((torch.device("cpu"), torch.device("cpu")))
    kernel = QTriadAttentionScoreKernel(
        num_layers=2,
        num_heads=4,
        head_dim=4,
        num_qubits=2,
        circuit_depth=1,
        pair_chunk_size=4,
    )
    kernel.configure_model_parallel((torch.device("cpu"), torch.device("cpu")))
    kernel.train()
    input_ids, attention_mask, subject_mask, object_mask = _inputs()
    adapter = AttentionScoreKernelAdapter(model, model.score_module_paths, kernel)
    adapter.attach(
        AttentionScoreHookConfig(
            attention_mask=attention_mask,
            subject_mask=subject_mask,
            object_mask=object_mask,
        )
    )
    try:
        logits = model(input_ids, attention_mask, subject_mask, object_mask)
        logits.square().mean().backward()
    finally:
        adapter.remove()
    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    assert all(parameter.grad is not None for parameter in kernel.parameters())
