"""Span-centric NLP task structures."""

from .relation import (
    PAD_TOKEN,
    UNK_TOKEN,
    RelationDataset,
    RelationRecord,
    build_label_map,
    build_vocab,
    collate_relation_batch,
    load_relation_jsonl,
    sample_relation_records,
    write_relation_jsonl,
)
from .relation_formats import RELATION_DATA_FORMATS, load_relation_records, relation_record_summary
from .span_examples import LabeledSpan, RelationExtractionExample, SpanCentricExample

__all__ = [
    "LabeledSpan",
    "PAD_TOKEN",
    "RELATION_DATA_FORMATS",
    "RelationDataset",
    "RelationExtractionExample",
    "RelationRecord",
    "SpanCentricExample",
    "UNK_TOKEN",
    "build_label_map",
    "build_vocab",
    "collate_relation_batch",
    "load_relation_jsonl",
    "load_relation_records",
    "relation_record_summary",
    "sample_relation_records",
    "write_relation_jsonl",
]