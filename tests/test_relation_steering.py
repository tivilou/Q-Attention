from __future__ import annotations

import torch

from q_attention.experiments import (
    anchor_mask_from_batch,
    build_anchor_projector,
    collect_anchor_key_vectors,
    collect_relation_key_samples,
    evaluate_relation_model,
    make_relation_loader,
    relation_pair_features,
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


def test_relation_pair_features_preserve_key_dimension_and_pair_structure() -> None:
    keys = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    subject_mask = torch.tensor([[True, False, False]])
    object_mask = torch.tensor([[False, False, True]])

    relation_keys, features = relation_pair_features(keys, subject_mask, object_mask)

    assert relation_keys.shape == (1, 2)
    assert features.shape == (1, 8)
    assert torch.allclose(relation_keys, torch.tensor([[3.0, 4.0]]))
    assert torch.allclose(features[0, :2], torch.tensor([1.0, 2.0]))
    assert torch.allclose(features[0, 2:4], torch.tensor([5.0, 6.0]))


def test_collect_relation_key_samples_aligns_labels_across_layers() -> None:
    torch.manual_seed(73)
    records = _records()
    vocab = build_vocab(records)
    label_to_id = build_label_map(records)
    config = RelationTransformerConfig(vocab_size=len(vocab), num_labels=len(label_to_id), dim=8, num_layers=2, num_heads=2, ff_dim=16, dropout=0.0)
    model = RelationExtractionModel(config)
    loader = make_relation_loader(records, vocab, label_to_id, batch_size=2)

    collection = collect_relation_key_samples(model, loader, torch.device("cpu"), model.key_module_paths)

    assert collection.keys.shape == (4, 8)
    assert collection.relation_features.shape == (4, 32)
    assert collection.labels.shape == (4,)
    assert collection.layer_counts == {path: 2 for path in model.key_module_paths}
