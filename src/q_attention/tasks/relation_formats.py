"""Adapters for real relation extraction dataset formats."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .relation import RelationRecord, load_relation_jsonl

RELATION_DATA_FORMATS = ("project_jsonl", "tacred_json", "tacred_jsonl", "semeval2010_task8")

_TOKEN_RE = re.compile(r"</?e[12]>|\w+(?:[-']\w+)*|[^\w\s]", re.UNICODE)


def load_relation_records(path: str | Path, data_format: str = "project_jsonl") -> list[RelationRecord]:
    """Load relation records from a supported raw or canonical format."""
    if data_format == "project_jsonl":
        return load_relation_jsonl(path)
    if data_format == "tacred_json":
        return load_tacred_json(path)
    if data_format == "tacred_jsonl":
        return load_tacred_jsonl(path)
    if data_format == "semeval2010_task8":
        return load_semeval2010_task8(path)
    raise ValueError(f"unknown relation data format '{data_format}', expected one of {RELATION_DATA_FORMATS}")


def relation_record_summary(records: Sequence[RelationRecord]) -> dict[str, Any]:
    """Return compact dataset statistics for configs and smoke logs."""
    if not records:
        raise ValueError("cannot summarize an empty relation dataset")
    label_counts = Counter(record.label for record in records)
    lengths = [len(record.tokens) for record in records]
    return {
        "num_records": len(records),
        "num_labels": len(label_counts),
        "label_counts": dict(sorted(label_counts.items())),
        "max_length": max(lengths),
        "mean_length": sum(lengths) / len(lengths),
    }


def load_tacred_json(path: str | Path) -> list[RelationRecord]:
    """Load TACRED/Re-TACRED style JSON data.

    Expected fields are ``token`` or ``tokens``, ``subj_start``, ``subj_end``,
    ``obj_start``, ``obj_end``, and ``relation``. TACRED end offsets are
    inclusive; they are converted to this project's end-exclusive spans.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        objects = payload
    elif isinstance(payload, Mapping):
        objects = _records_from_mapping_payload(payload)
    else:
        raise ValueError(f"unsupported TACRED JSON payload in {path}")
    return [_tacred_object_to_record(obj, source_path=path, row=index) for index, obj in enumerate(objects)]


def load_tacred_jsonl(path: str | Path) -> list[RelationRecord]:
    """Load TACRED/Re-TACRED style JSONL data."""
    records: list[RelationRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            records.append(_tacred_object_to_record(json.loads(line), source_path=path, row=line_no))
    if not records:
        raise ValueError(f"no TACRED records found in {path}")
    return records


def _records_from_mapping_payload(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("data", "records", "examples"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    if "token" in payload or "tokens" in payload:
        return [payload]
    raise ValueError("JSON object must contain a data/records/examples list or a single TACRED record")


def _tacred_object_to_record(obj: Mapping[str, Any], *, source_path: str | Path, row: int) -> RelationRecord:
    tokens_value = obj.get("token", obj.get("tokens"))
    if not isinstance(tokens_value, Sequence) or isinstance(tokens_value, (str, bytes)):
        raise ValueError(f"{source_path}:{row}: TACRED record must contain token/tokens list")
    tokens = tuple(str(token) for token in tokens_value)

    if all(key in obj for key in ("subj_start", "subj_end", "obj_start", "obj_end")):
        subject = (int(obj["subj_start"]), int(obj["subj_end"]) + 1)
        object_span = (int(obj["obj_start"]), int(obj["obj_end"]) + 1)
    elif "subject" in obj and "object" in obj:
        subject = _span_from_sequence(obj["subject"], source_path=source_path, row=row, field_name="subject")
        object_span = _span_from_sequence(obj["object"], source_path=source_path, row=row, field_name="object")
    else:
        raise ValueError(f"{source_path}:{row}: missing TACRED subject/object offsets")

    label = str(obj.get("relation", obj.get("label", "")))
    if not label:
        raise ValueError(f"{source_path}:{row}: missing relation/label field")

    metadata = _compact_metadata(
        obj,
        keep=("id", "docid", "sent_id", "subj_type", "obj_type", "stanford_ner", "source"),
    )
    metadata.setdefault("source_format", "tacred")
    record = RelationRecord(tokens=tokens, subject=subject, object=object_span, label=label, metadata=metadata)
    try:
        record.validate()
    except ValueError as exc:
        raise ValueError(f"{source_path}:{row}: {exc}") from exc
    return record


def _span_from_sequence(value: object, *, source_path: str | Path, row: int, field_name: str) -> tuple[int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{source_path}:{row}: {field_name} must be a two-item span")
    return int(value[0]), int(value[1])


def _compact_metadata(obj: Mapping[str, Any], *, keep: Iterable[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in keep:
        if key in obj:
            value = obj[key]
            if isinstance(value, (str, int, float, bool)) or value is None:
                metadata[key] = value
    return metadata


def load_semeval2010_task8(path: str | Path) -> list[RelationRecord]:
    """Load the original SemEval-2010 Task 8 four-line text format."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    records: list[RelationRecord] = []
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        if index + 1 >= len(lines):
            raise ValueError(f"{path}:{index + 1}: incomplete SemEval block")
        raw_id, sentence = _parse_semeval_sentence_line(lines[index], source_path=path, line_no=index + 1)
        label = lines[index + 1].strip()
        comment = lines[index + 2].strip() if index + 2 < len(lines) else ""
        tokens, subject, object_span = _tokenize_entity_marked_sentence(sentence, source_path=path, line_no=index + 1)
        record = RelationRecord(
            tokens=tuple(tokens),
            subject=subject,
            object=object_span,
            label=label,
            metadata={"source_format": "semeval2010_task8", "id": raw_id, "comment": comment},
        )
        try:
            record.validate()
        except ValueError as exc:
            raise ValueError(f"{path}:{index + 1}: {exc}") from exc
        records.append(record)
        index += 4
    if not records:
        raise ValueError(f"no SemEval records found in {path}")
    return records


def _parse_semeval_sentence_line(line: str, *, source_path: str | Path, line_no: int) -> tuple[str, str]:
    if "\t" in line:
        raw_id, sentence = line.split("\t", 1)
    else:
        match = re.match(r"^(\S+)\s+(.+)$", line.strip())
        if match is None:
            raise ValueError(f"{source_path}:{line_no}: cannot parse SemEval sentence line")
        raw_id, sentence = match.group(1), match.group(2)
    sentence = sentence.strip()
    if len(sentence) >= 2 and sentence[0] == '"' and sentence[-1] == '"':
        sentence = sentence[1:-1]
    return raw_id.strip(), sentence


def _tokenize_entity_marked_sentence(
    sentence: str,
    *,
    source_path: str | Path,
    line_no: int,
) -> tuple[list[str], tuple[int, int], tuple[int, int]]:
    tokens: list[str] = []
    subject_indices: list[int] = []
    object_indices: list[int] = []
    active: str | None = None

    for match in _TOKEN_RE.finditer(sentence):
        piece = match.group(0)
        lowered = piece.lower()
        if lowered == "<e1>":
            active = "subject"
            continue
        if lowered == "</e1>":
            active = None
            continue
        if lowered == "<e2>":
            active = "object"
            continue
        if lowered == "</e2>":
            active = None
            continue

        token_index = len(tokens)
        tokens.append(piece)
        if active == "subject":
            subject_indices.append(token_index)
        elif active == "object":
            object_indices.append(token_index)

    subject = _indices_to_span(subject_indices, source_path=source_path, line_no=line_no, name="e1")
    object_span = _indices_to_span(object_indices, source_path=source_path, line_no=line_no, name="e2")
    return tokens, subject, object_span


def _indices_to_span(indices: Sequence[int], *, source_path: str | Path, line_no: int, name: str) -> tuple[int, int]:
    if not indices:
        raise ValueError(f"{source_path}:{line_no}: missing {name} entity markers")
    start, end = min(indices), max(indices) + 1
    if list(range(start, end)) != sorted(indices):
        raise ValueError(f"{source_path}:{line_no}: non-contiguous {name} entity span")
    return start, end