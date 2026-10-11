from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CONVERTER = ROOT / "experiments" / "prepare_retacred_reference_improved_baseline.py"


def write_split(path: Path, label: str = "org:founded_by") -> None:
    record = {
        "tokens": ["Acme", "founded", "Bob"],
        "subject": [0, 1],
        "object": [2, 3],
        "label": label,
        "metadata": {"subj_type": "ORGANIZATION", "obj_type": "PERSON", "id": "x1"},
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def run_converter(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    source = tmp_path / "source"
    output = tmp_path / "prepared"
    source.mkdir()
    for split in ("train", "valid", "test"):
        write_split(source / f"{split}.jsonl")
    return subprocess.run(
        [
            sys.executable,
            str(CONVERTER),
            "--train-path",
            str(source / "train.jsonl"),
            "--valid-path",
            str(source / "valid.jsonl"),
            "--test-path",
            str(source / "test.jsonl"),
            "--output-dir",
            str(output),
            *extra,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_converter_preserves_entities_and_changes_only_span_convention(tmp_path: Path) -> None:
    result = run_converter(tmp_path)
    assert result.returncode == 0, result.stderr
    prepared = json.loads((tmp_path / "prepared" / "train.json").read_text(encoding="utf-8"))
    assert prepared == [
        {
            "token": ["Acme", "founded", "Bob"],
            "subj_start": 0,
            "subj_end": 0,
            "obj_start": 2,
            "obj_end": 2,
            "subj_type": "ORGANIZATION",
            "obj_type": "PERSON",
            "relation": "org:founded_by",
            "id": "x1",
        }
    ]
    manifest = json.loads((tmp_path / "prepared" / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["target_contract"] == "upstream JSON array with inclusive spans"
    assert manifest["splits"]["train"]["records"] == 1
    assert manifest["upstream_label_to_id"]["org:founded_by"] == 1


def test_converter_rejects_missing_entity_type(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    record = {
        "tokens": ["Acme", "founded", "Bob"],
        "subject": [0, 1],
        "object": [2, 3],
        "label": "org:founded_by",
        "metadata": {"obj_type": "PERSON"},
    }
    for split in ("train", "valid", "test"):
        (source / f"{split}.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(CONVERTER),
            "--train-path",
            str(source / "train.jsonl"),
            "--valid-path",
            str(source / "valid.jsonl"),
            "--test-path",
            str(source / "test.jsonl"),
            "--output-dir",
            str(tmp_path / "prepared"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "metadata.subj_type is required" in result.stderr


def test_runner_contract_mentions_pinned_source_and_private_artifact_boundaries() -> None:
    runner = (ROOT / "scripts" / "run_retacred_reference_improved_baseline.sh").read_text(encoding="utf-8")
    exporter = (ROOT / "scripts" / "export_retacred_reference_improved_baseline_report.sh").read_text(encoding="utf-8")
    assert "fe00781116ab4c34cbcd9a2ae5b02513afae87ee" in runner
    assert "case_study_tensors.pt" in runner
    assert "case_study_tensors.pt" in exporter
    assert "Forbidden private artifact" in exporter
    assert "origin/1.1" in runner
