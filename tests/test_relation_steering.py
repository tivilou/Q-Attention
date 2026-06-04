from __future__ import annotations

import torch

from q_attention.experiments import (
    anchor_mask_from_batch,
    build_anchor_projector,
    collect_anchor_key_vectors,
    evaluate_relation_model,
    make_relation_loader,
)
from q_attention.models import RelationExtractionModel, RelationTransformerConfig
from q_attention.projectors import SpectralProjectorConfig
from q_attention.tasks.relation import RelationRecord, build_label_map, build_vocab


def _records() -> list[RelationRecord]:
    return [
        RelationRecord(tokens=("Alice", "founded", "Acme"), subject=(0, 1), object=(2, 3), label="founded_by"),
        RelationRecord(tokens=("Paris", "is", "in", "France"), subject=(0, 1), object=(3, 4), label="located_in"),
    ]


def test_anchor_mask_from_batch_supports_entity_and_context_masks() -> None:
    batch = {
        "subject_mask": torch.tensor([[True, False, False], [False, True, False]]),
        "object_mask": torch.tensor([[False, False, True], [True, False, False]]),
        "attention_mask": torch.tensor([[True, True, True], [True, True, False]]),
    }

    assert anchor_mask_from_batch(batch, "subject").sum().item() == 2
    assert anchor_mask_from_batch(batch, "object").sum().item() == 2
    assert anchor_mask_from_batch(batch, "subject_object").sum().item() == 4
    assert anchor_mask_from_batch(batch, "all_tokens").sum().item() == 5


def test_collect_anchor_key_vectors_counts_each_steerable_layer() -> None:
    torch.manual_seed(31)
    records = _records()
    vocab = build_vocab(records)
    label_to_id = build_label_map(records)
    config = RelationTransformerConfig(vocab_size=len(vocab), num_labels=len(label_to_id), dim=12, num_layers=2, num_heads=3, ff_dim=24, dropout=0.0)
    model = RelationExtractionModel(config)
    loader = make_relation_loader(records, vocab, label_to_id, batch_size=2)

    collection = collect_anchor_key_vectors(model, loader, torch.device("cpu"), model.key_module_paths, anchor="subject_object")

    expected_per_layer = 4
    assert collection.keys.shape == (expected_per_layer * config.num_layers, config.dim)
    assert collection.layer_counts == {path: expected_per_layer for path in model.key_module_paths}
    projector = build_anchor_projector(collection.keys, SpectralProjectorConfig(rank=2))
    assert projector.shape == (config.dim, config.dim)


def test_evaluate_relation_model_runs_with_key_steering_adapter() -> None:
    torch.manual_seed(37)
    records = _records()
    vocab = build_vocab(records)
    label_to_id = build_label_map(records)
    config = RelationTransformerConfig(vocab_size=len(vocab), num_labels=len(label_to_id), dim=8, num_layers=1, num_heads=2, ff_dim=16, dropout=0.0)
    model = RelationExtractionModel(config)
    loader = make_relation_loader(records, vocab, label_to_id, batch_size=2)

    result = evaluate_relation_model(
        model,
        loader,
        torch.device("cpu"),
        len(label_to_id),
        projector=torch.eye(config.dim),
        key_module_paths=model.key_module_paths,
        gain=0.0,
    )

    assert len(result.predictions) == len(records)
    assert len(result.labels) == len(records)
    assert "macro_f1" in result.metrics