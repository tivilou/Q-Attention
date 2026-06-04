"""Relation extraction JSONL loading and batching."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Mapping

import torch
from torch.utils.data import Dataset

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass(frozen=True)
class RelationRecord:
    """A tokenized relation extraction example.

    The JSONL format is:

    ```json
    {"tokens": ["Steve", "Jobs", "founded", "Apple"],
     "subject": [0, 2], "object": [3, 4], "label": "founded_by"}
    ```
    """

    tokens: tuple[str, ...]
    subject: tuple[int, int]
    object: tuple[int, int]
    label: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        length = len(self.tokens)
        for name, span in {"subject": self.subject, "object": self.object}.items():
            start, end = span
            if start < 0 or end <= start or end > length:
                raise ValueError(f"invalid {name} span {span} for {length} tokens")


def load_relation_jsonl(path: str | Path) -> list[RelationRecord]:
    """Load relation examples from JSONL."""
    records: list[RelationRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            record = RelationRecord(
                tokens=tuple(obj["tokens"]),
                subject=tuple(obj["subject"]),  # type: ignore[arg-type]
                object=tuple(obj["object"]),  # type: ignore[arg-type]
                label=str(obj["label"]),
                metadata=obj.get("metadata", {}),
            )
            try:
                record.validate()
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            records.append(record)
    if not records:
        raise ValueError(f"no relation records found in {path}")
    return records


def build_vocab(records: Iterable[RelationRecord], min_freq: int = 1) -> dict[str, int]:
    """Build a token vocabulary."""
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(token.lower() for token in record.tokens)
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, count in sorted(counter.items()):
        if count >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def build_label_map(records: Iterable[RelationRecord]) -> dict[str, int]:
    """Build a deterministic label map."""
    labels = sorted({record.label for record in records})
    return {label: idx for idx, label in enumerate(labels)}


def _span_to_mask(length: int, span: tuple[int, int]) -> torch.Tensor:
    mask = torch.zeros(length, dtype=torch.bool)
    start, end = span
    mask[start:end] = True
    return mask


class RelationDataset(Dataset):
    """Torch dataset for relation extraction records."""

    def __init__(self, records: list[RelationRecord], vocab: Mapping[str, int], label_to_id: Mapping[str, int]) -> None:
        self.records = records
        self.vocab = vocab
        self.label_to_id = label_to_id

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        input_ids = [self.vocab.get(token.lower(), self.vocab[UNK_TOKEN]) for token in record.tokens]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "subject_mask": _span_to_mask(len(record.tokens), record.subject),
            "object_mask": _span_to_mask(len(record.tokens), record.object),
            "label": torch.tensor(self.label_to_id[record.label], dtype=torch.long),
            "length": len(record.tokens),
        }


def collate_relation_batch(batch: list[dict[str, object]], pad_id: int = 0) -> dict[str, torch.Tensor]:
    """Pad relation examples into a batch."""
    max_len = max(int(item["length"]) for item in batch)
    batch_size = len(batch)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    subject_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    object_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    labels = torch.empty(batch_size, dtype=torch.long)

    for row, item in enumerate(batch):
        ids = item["input_ids"]
        subj = item["subject_mask"]
        obj = item["object_mask"]
        length = int(item["length"])
        input_ids[row, :length] = ids  # type: ignore[index]
        attention_mask[row, :length] = True
        subject_mask[row, :length] = subj  # type: ignore[index]
        object_mask[row, :length] = obj  # type: ignore[index]
        labels[row] = item["label"]  # type: ignore[assignment]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "subject_mask": subject_mask,
        "object_mask": object_mask,
        "labels": labels,
    }
