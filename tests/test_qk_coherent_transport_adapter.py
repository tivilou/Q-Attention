from __future__ import annotations

import copy

import torch

from q_attention.adapters import AttentionScoreHookConfig, AttentionScoreKernelAdapter
from q_attention.models import RelationExtractionModel, RelationTransformerConfig
from q_attention.plugins import (
    QueryKeyCoherentTransportConfig,
    QuantumQueryKeyCoherentTransportKernel,
)


def test_zero_gain_adapter_is_exactly_baseline_equivalent() -> None:
    torch.manual_seed(37)
    model_config = RelationTransformerConfig(
        vocab_size=23,
        num_labels=3,
        dim=8,
        num_layers=1,
        num_heads=2,
        ff_dim=16,
        dropout=0.0,
        max_length=8,
    )
    baseline_model = RelationExtractionModel(model_config)
    steered_model = copy.deepcopy(baseline_model)
    kernel = QuantumQueryKeyCoherentTransportKernel(
        QueryKeyCoherentTransportConfig(
            num_layers=1,
            num_heads=2,
            head_dim=4,
            depth=2,
            initial_transport=0.0,
            pair_chunk_size=8,
            seed=37,
        )
    )
    adapter = AttentionScoreKernelAdapter(
        steered_model,
        steered_model.score_module_paths,
        kernel,
    )
    attention_mask = torch.ones(2, 5, dtype=torch.bool)
    subject_mask = torch.zeros(2, 5, dtype=torch.bool)
    object_mask = torch.zeros(2, 5, dtype=torch.bool)
    subject_mask[:, 0] = True
    object_mask[:, 2] = True
    input_ids = torch.tensor([[1, 4, 7, 9, 2], [3, 5, 6, 8, 2]])

    baseline = baseline_model(input_ids, attention_mask, subject_mask, object_mask)
    with adapter.steering(
        AttentionScoreHookConfig(
            attention_mask=attention_mask,
            subject_mask=subject_mask,
            object_mask=object_mask,
        )
    ):
        steered = steered_model(input_ids, attention_mask, subject_mask, object_mask)

    assert torch.equal(steered, baseline)
    assert not adapter.attached
