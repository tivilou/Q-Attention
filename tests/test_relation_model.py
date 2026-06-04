from __future__ import annotations

import torch

from q_attention.adapters import resolve_module
from q_attention.models import RelationExtractionModel, RelationTransformerConfig


def test_relation_model_forward_and_key_paths() -> None:
    config = RelationTransformerConfig(vocab_size=20, num_labels=3, dim=16, num_layers=2, num_heads=4, ff_dim=32)
    model = RelationExtractionModel(config)
    input_ids = torch.randint(0, 20, (2, 5))
    attention_mask = torch.ones(2, 5, dtype=torch.bool)
    subject_mask = torch.tensor([[True, False, False, False, False], [False, True, False, False, False]])
    object_mask = torch.tensor([[False, False, True, False, False], [False, False, False, True, False]])

    logits = model(input_ids, attention_mask, subject_mask, object_mask)

    assert logits.shape == (2, 3)
    assert model.key_module_paths == ("encoder.layers.0.attn.key_proj", "encoder.layers.1.attn.key_proj")
    assert resolve_module(model, model.key_module_paths[0]) is model.encoder.layers[0].attn.key_proj
