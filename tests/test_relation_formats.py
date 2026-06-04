from __future__ import annotations

import json

from q_attention.tasks.relation import RelationRecord, load_relation_jsonl, sample_relation_records, write_relation_jsonl
from q_attention.tasks.relation_formats import load_relation_records, relation_record_summary


def test_load_tacred_json_converts_inclusive_offsets(tmp_path) -> None:
    raw_path = tmp_path / "tacred.json"
    raw_path.write_text(
        json.dumps(
            [
                {
                    "id": "ex-1",
                    "token": ["The", "company", "acquired", "startup", "."],
                    "subj_start": 1,
                    "subj_end": 1,
                    "obj_start": 3,
                    "obj_end": 3,
                    "subj_type": "ORG",
                    "obj_type": "ORG",
                    "relation": "org:acquired",
                }
            ]
        ),
        encoding="utf-8",
    )

    records = load_relation_records(raw_path, "tacred_json")

    assert len(records) == 1
    assert records[0].tokens == ("The", "company", "acquired", "startup", ".")
    assert records[0].subject == (1, 2)
    assert records[0].object == (3, 4)
    assert records[0].label == "org:acquired"
    assert records[0].metadata["source_format"] == "tacred"


def test_load_tacred_jsonl_and_summary(tmp_path) -> None:
    raw_path = tmp_path / "tacred.jsonl"
    rows = [
        {
            "tokens": ["Alice", "works", "at", "Acme"],
            "subject": [0, 1],
            "object": [3, 4],
            "label": "per:employee_of",
        },
        {
            "tokens": ["Bob", "lives", "in", "Paris"],
            "subject": [0, 1],
            "object": [3, 4],
            "label": "per:cities_of_residence",
        },
    ]
    raw_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    records = load_relation_records(raw_path, "tacred_jsonl")
    summary = relation_record_summary(records)

    assert len(records) == 2
    assert summary["num_labels"] == 2
    assert summary["max_length"] == 4


def test_load_semeval2010_task8_text(tmp_path) -> None:
    raw_path = tmp_path / "semeval.txt"
    raw_path.write_text(
        '1\t"The <e1>company</e1> acquired the <e2>startup</e2>."\n'
        "Cause-Effect(e1,e2)\n"
        "Comment: example\n"
        "\n",
        encoding="utf-8",
    )

    records = load_relation_records(raw_path, "semeval2010_task8")

    assert len(records) == 1
    assert records[0].tokens == ("The", "company", "acquired", "the", "startup", ".")
    assert records[0].subject == (1, 2)
    assert records[0].object == (4, 5)
    assert records[0].label == "Cause-Effect(e1,e2)"
    assert records[0].metadata["id"] == "1"


def test_write_and_stratified_sample_relation_jsonl(tmp_path) -> None:
    records = [
        RelationRecord(tokens=("a", "x"), subject=(0, 1), object=(1, 2), label="a"),
        RelationRecord(tokens=("b", "x"), subject=(0, 1), object=(1, 2), label="b"),
        RelationRecord(tokens=("a", "y"), subject=(0, 1), object=(1, 2), label="a"),
        RelationRecord(tokens=("b", "y"), subject=(0, 1), object=(1, 2), label="b"),
    ]

    sampled = sample_relation_records(records, 2, seed=7, stratified=True)
    output_path = tmp_path / "sample.jsonl"
    count = write_relation_jsonl(sampled, output_path)
    loaded = load_relation_jsonl(output_path)

    assert count == 2
    assert {record.label for record in loaded} == {"a", "b"}