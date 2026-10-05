#!/usr/bin/env python3
"""Convert project Re-TACRED JSONL into the upstream TACRED JSON contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


UPSTREAM_LABELS = (
    "no_relation",
    "org:founded_by",
    "per:identity",
    "org:alternate_names",
    "per:children",
    "per:origin",
    "per:countries_of_residence",
    "per:employee_of",
    "per:title",
    "org:city_of_branch",
    "per:religion",
    "per:age",
    "per:date_of_death",
    "org:website",
    "per:stateorprovinces_of_residence",
    "org:top_members/employees",
    "org:number_of_employees/members",
    "org:members",
    "org:country_of_branch",
    "per:spouse",
    "org:stateorprovince_of_branch",
    "org:political/religious_affiliation",
    "org:member_of",
    "per:siblings",
    "per:stateorprovince_of_birth",
    "org:dissolved",
    "per:other_family",
    "org:shareholders",
    "per:parents",
    "per:charges",
    "per:schools_attended",
    "per:cause_of_death",
    "per:city_of_death",
    "per:stateorprovince_of_death",
    "org:founded",
    "per:country_of_death",
    "per:country_of_birth",
    "per:date_of_birth",
    "per:cities_of_residence",
    "per:city_of_birth",
)
UPSTREAM_LABEL_TO_ID = {label: index for index, label in enumerate(UPSTREAM_LABELS)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _span(value: object, *, field: str, path: Path, line_number: int) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2 or not all(isinstance(item, int) for item in value):
        raise ValueError(f"{path}:{line_number}: {field} must be a two-item integer list")
    start, end = value
    if start < 0 or end <= start:
        raise ValueError(f"{path}:{line_number}: invalid {field} span {value}")
    return start, end


def _load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            tokens = item.get("tokens")
            if not isinstance(tokens, list) or not tokens or not all(isinstance(token, str) for token in tokens):
                raise ValueError(f"{path}:{line_number}: tokens must be a non-empty string list")
            subject = _span(item.get("subject"), field="subject", path=path, line_number=line_number)
            object_span = _span(item.get("object"), field="object", path=path, line_number=line_number)
            for field_name, span in (("subject", subject), ("object", object_span)):
                if span[1] > len(tokens):
                    raise ValueError(f"{path}:{line_number}: {field_name} span exceeds token count")
            label = item.get("label")
            if not isinstance(label, str) or label not in UPSTREAM_LABEL_TO_ID:
                raise ValueError(f"{path}:{line_number}: unsupported Re-TACRED label {label!r}")
            metadata = item.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError(f"{path}:{line_number}: metadata must be an object")
            subject_type = metadata.get("subj_type")
            object_type = metadata.get("obj_type")
            if not isinstance(subject_type, str) or not subject_type:
                raise ValueError(f"{path}:{line_number}: metadata.subj_type is required")
            if not isinstance(object_type, str) or not object_type:
                raise ValueError(f"{path}:{line_number}: metadata.obj_type is required")
            records.append(
                {
                    "tokens": tokens,
                    "subject": subject,
                    "object": object_span,
                    "label": label,
                    "metadata": metadata,
                }
            )
    if not records:
        raise ValueError(f"no records found in {path}")
    return records


def _upstream_record(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record["metadata"]
    subject_start, subject_end = record["subject"]
    object_start, object_end = record["object"]
    output: dict[str, Any] = {
        "token": list(record["tokens"]),
        "subj_start": subject_start,
        "subj_end": subject_end - 1,
        "obj_start": object_start,
        "obj_end": object_end - 1,
        "subj_type": metadata["subj_type"],
        "obj_type": metadata["obj_type"],
        "relation": record["label"],
    }
    for source_key, target_key in (("id", "id"), ("docid", "docid"), ("sent_id", "sent_id")):
        if source_key in metadata:
            output[target_key] = metadata[source_key]
    return output


def convert_split(source_path: Path, destination_path: Path) -> dict[str, Any]:
    records = _load_records(source_path)
    converted = [_upstream_record(record) for record in records]
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(json.dumps(converted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    labels = sorted({record["label"] for record in records})
    return {
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "output_path": str(destination_path),
        "output_sha256": sha256_file(destination_path),
        "records": len(records),
        "labels": labels,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--valid-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_paths = {
        "train": Path(args.train_path).resolve(),
        "valid": Path(args.valid_path).resolve(),
        "test": Path(args.test_path).resolve(),
    }
    manifest = {
        "schema": "q-attention.retacred-reference-improved-baseline-data.v1",
        "dataset": "Re-TACRED",
        "format": "upstream_tacred_json",
        "source_contract": "project canonical JSONL with end-exclusive spans",
        "target_contract": "upstream JSON array with inclusive spans",
        "upstream_label_to_id": UPSTREAM_LABEL_TO_ID,
        "splits": {},
    }
    for split, source_path in split_paths.items():
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        manifest["splits"][split] = convert_split(source_path, output_dir / f"{split}.json")
    (output_dir / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "splits": manifest["splits"]}, sort_keys=True))


if __name__ == "__main__":
    main()
