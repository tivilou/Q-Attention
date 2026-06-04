from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_builder_module():
    path = Path("experiments/build_retacred_from_tacred.py")
    spec = importlib.util.spec_from_file_location("build_retacred_from_tacred", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_iter_json_array_handles_chunk_boundaries(tmp_path) -> None:
    module = load_builder_module()
    path = tmp_path / "items.json"
    path.write_text(json.dumps([{"id": "a", "text": "hello"}, {"id": "b", "text": "world"}]), encoding="utf-8")

    rows = list(module.iter_json_array(path, chunk_size=7))

    assert rows == [{"id": "a", "text": "hello"}, {"id": "b", "text": "world"}]


def test_apply_split_writes_patched_jsonl(tmp_path) -> None:
    module = load_builder_module()
    tacred_dir = tmp_path / "tacred"
    patch_dir = tmp_path / "patch"
    output_dir = tmp_path / "out"
    tacred_dir.mkdir()
    patch_dir.mkdir()
    (tacred_dir / "train.json").write_text(
        json.dumps(
            [
                {"id": "keep", "token": ["A", "works", "B"], "relation": "old"},
                {"id": "drop", "token": ["C", "knows", "D"], "relation": "old"},
            ]
        ),
        encoding="utf-8",
    )
    (patch_dir / "train_id2label.json").write_text(json.dumps({"keep": "per:employee_of"}), encoding="utf-8")

    summary = module.apply_split(split="train", tacred_dir=tacred_dir, patch_dir=patch_dir, output_dir=output_dir)
    rows = [(json.loads(line)) for line in (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()]

    assert summary["original_records"] == 2
    assert summary["patched_records"] == 1
    assert rows[0]["id"] == "keep"
    assert rows[0]["relation"] == "per:employee_of"