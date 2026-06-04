from __future__ import annotations

from pathlib import Path

from q_attention.tasks.relation import RelationDataset, build_label_map, build_vocab, collate_relation_batch, load_relation_jsonl


def test_load_relation_jsonl_and_collate() -> None:
    records = load_relation_jsonl(Path("examples/relation_toy_train.jsonl"))
    vocab = build_vocab(records)
    labels = build_label_map(records)
    dataset = RelationDataset(records[:2], vocab, labels)
    batch = collate_relation_batch([dataset[0], dataset[1]], pad_id=vocab["<pad>"])

    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].dtype.name == "bool"
    assert batch["subject_mask"].sum().item() > 0
    assert batch["object_mask"].sum().item() > 0
    assert len(labels) == 4
