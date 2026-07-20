"""Relation extraction JSONL loading and batching."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
from typing import Iterable, Mapping, Sequence

import torch
from torch.utils.data import Dataset

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


@dataclass(frozen=True)
class RelationRecord:
    """A tokenized relation extraction example.

    The canonical JSONL format is:

    ```json
    {"tokens": ["Steve", "Jobs", "founded", "Apple"],
     "subject": [0, 2], "object": [3, 4], "label": "founded_by"}
    ```

    Subject/object spans are zero-based and end-exclusive.
    """

    tokens: tuple[str, ...]
    subject: tuple[int, int]
    object: tuple[int, int]
    label: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        length = len(self.tokens)
        if length == 0:
            raise ValueError("tokens must not be empty")
        for name, span in {"subject": self.subject, "object": self.object}.items():
            start, end = span
            if start < 0 or end <= start or end > length:
                raise ValueError(f"invalid {name} span {span} for {length} tokens")


def _as_span(value: object, *, field_name: str) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{field_name} must be a two-item span")
    return int(value[0]), int(value[1])


def load_relation_jsonl(path: str | Path) -> list[RelationRecord]:
    """Load canonical relation examples from JSONL."""
    records: list[RelationRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            record = RelationRecord(
                tokens=tuple(str(token) for token in obj["tokens"]),
                subject=_as_span(obj["subject"], field_name="subject"),
                object=_as_span(obj["object"], field_name="object"),
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


def write_relation_jsonl(records: Iterable[RelationRecord], path: str | Path) -> int:
    """Write canonical relation examples to JSONL and return the count."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            record.validate()
            item = {
                "tokens": list(record.tokens),
                "subject": list(record.subject),
                "object": list(record.object),
                "label": record.label,
            }
            if record.metadata:
                item["metadata"] = dict(record.metadata)
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    if count == 0:
        raise ValueError(f"no relation records were written to {path}")
    return count


def sample_relation_records(
    records: Sequence[RelationRecord],
    limit: int | None,
    *,
    seed: int = 13,
    stratified: bool = True,
) -> list[RelationRecord]:
    """Return a deterministic small subset for smoke/debug runs.

    Stratified sampling keeps rare labels visible in tiny real-data smoke runs.
    """
    source = list(records)
    if limit is None or limit <= 0 or len(source) <= limit:
        return source

    rng = random.Random(seed)
    if not stratified:
        indices = sorted(rng.sample(range(len(source)), limit))
        return [source[index] for index in indices]

    by_label: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(source):
        by_label[record.label].append(index)
    for indices in by_label.values():
        rng.shuffle(indices)

    selected: list[int] = []
    label_order = sorted(by_label)
    while len(selected) < limit and label_order:
        next_labels: list[str] = []
        for label in label_order:
            if by_label[label] and len(selected) < limit:
                selected.append(by_label[label].pop())
            if by_label[label]:
                next_labels.append(label)
        label_order = next_labels

    return [source[index] for index in sorted(selected)]


def sample_relation_records_proportional(
    records: Sequence[RelationRecord],
    limit: int | None,
    *,
    seed: int = 13,
) -> list[RelationRecord]:
    """Return a deterministic subset that preserves label proportions.

    Allocation uses the largest-remainder method, so each label receives a
    count close to its share of the source data. This is appropriate for
    validation and test subsets; use ``sample_relation_records`` when a
    balanced training smoke subset is intentional.
    """
    source = list(records)
    if limit is None or limit <= 0 or len(source) <= limit:
        return source

    by_label: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(source):
        by_label[record.label].append(index)

    total = len(source)
    allocations: dict[str, int] = {}
    remainders: list[tuple[int, str]] = []
    allocated = 0
    for label in sorted(by_label):
        numerator = limit * len(by_label[label])
        count, remainder = divmod(numerator, total)
        allocations[label] = count
        allocated += count
        remainders.append((remainder, label))

    for _, label in sorted(remainders, key=lambda item: (-item[0], item[1]))[: limit - allocated]:
        allocations[label] += 1

    rng = random.Random(seed)
    selected: list[int] = []
    for label in sorted(by_label):
        indices = list(by_label[label])
        rng.shuffle(indices)
        selected.extend(indices[: allocations[label]])
    return [source[index] for index in sorted(selected)]


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
