from __future__ import annotations

from collections import Counter
from pathlib import Path

from q_attention.tasks.relation import (
    RelationDataset,
    RelationRecord,
    build_label_map,
    build_vocab,
    collate_relation_batch,
    load_relation_jsonl,
    sample_relation_records_proportional,
)


def test_load_relation_jsonl_and_collate() -> None:
    records = load_relation_jsonl(Path("examples/relation_toy_train.jsonl"))
    vocab = build_vocab(records)
    labels = build_label_map(records)
    dataset = RelationDataset(records[:2], vocab, labels)
    batch = collate_relation_batch([dataset[0], dataset[1]], pad_id=vocab["<pad>"])

    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].dtype is __import__("torch").bool
    assert batch["subject_mask"].sum().item() > 0
    assert batch["object_mask"].sum().item() > 0
    assert len(labels) == 4


def test_proportional_sampling_preserves_label_ratios_and_is_deterministic() -> None:
    records = [
        RelationRecord(tokens=(f"common-{index}",), subject=(0, 1), object=(0, 1), label="common")
        for index in range(80)
    ] + [
        RelationRecord(tokens=(f"rare-{index}",), subject=(0, 1), object=(0, 1), label="rare")
        for index in range(20)
    ]

    first = sample_relation_records_proportional(records, 50, seed=17)
    second = sample_relation_records_proportional(records, 50, seed=17)

    assert first == second
    assert Counter(record.label for record in first) == {"common": 40, "rare": 10}
