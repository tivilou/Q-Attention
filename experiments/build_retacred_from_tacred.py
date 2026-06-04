from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]


def iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[dict[str, Any]]:
    """Yield objects from a top-level JSON array without loading the full file."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        eof = False

        def fill() -> None:
            nonlocal buffer, eof
            if eof:
                return
            chunk = handle.read(chunk_size)
            if chunk:
                buffer += chunk
            else:
                eof = True

        def ensure_buffer() -> None:
            while not buffer and not eof:
                fill()

        ensure_buffer()
        buffer = buffer.lstrip()
        while not buffer and not eof:
            fill()
            buffer = buffer.lstrip()
        if not buffer or buffer[0] != "[":
            raise ValueError(f"{path} is not a top-level JSON array")
        buffer = buffer[1:]

        while True:
            while True:
                ensure_buffer()
                buffer = buffer.lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:]
                    continue
                break
            ensure_buffer()
            buffer = buffer.lstrip()
            if not buffer:
                if eof:
                    raise ValueError(f"unexpected EOF while reading {path}")
                continue
            if buffer.startswith("]"):
                return

            while True:
                try:
                    obj, end = decoder.raw_decode(buffer)
                    if not isinstance(obj, dict):
                        raise ValueError(f"expected object inside {path}")
                    buffer = buffer[end:]
                    yield obj
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    fill()


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def apply_split(*, split: str, tacred_dir: Path, patch_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Apply one Re-TACRED split patch and write TACRED-style JSONL."""
    input_path = tacred_dir / f"{split}.json"
    patch_path = patch_dir / f"{split}_id2label.json"
    output_path = output_dir / f"{split}.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"missing TACRED split: {input_path}")
    if not patch_path.exists():
        raise FileNotFoundError(f"missing Re-TACRED patch: {patch_path}")

    id_to_label = json.loads(patch_path.read_text(encoding="utf-8"))
    labels: Counter[str] = Counter()
    original_count = 0
    patched_count = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for record in iter_json_array(input_path):
            original_count += 1
            record_id = record.get("id")
            label = id_to_label.get(record_id, id_to_label.get(str(record_id)))
            if label is None:
                continue
            record["relation"] = label
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            labels[label] += 1
            patched_count += 1

    return {
        "split": split,
        "input_path": str(input_path),
        "patch_path": str(patch_path),
        "output_path": str(output_path),
        "original_records": original_count,
        "patched_records": patched_count,
        "num_labels": len(labels),
        "label_counts": dict(sorted(labels.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Re-TACRED TACRED-style JSONL from licensed TACRED plus public patches.")
    parser.add_argument("--tacred_dir", default="data/raw/tacred/data/json", help="Directory containing TACRED train/dev/test JSON")
    parser.add_argument("--patch_dir", default="data/raw/Re-TACRED-source/Re-TACRED", help="Directory containing *_id2label.json patches")
    parser.add_argument("--output_dir", default="data/raw/Re-TACRED-patched-jsonl", help="Output directory for patched JSONL")
    parser.add_argument("--splits", default="train,dev,test", help="Comma-separated splits to patch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tacred_dir = resolve_project_path(args.tacred_dir)
    patch_dir = resolve_project_path(args.patch_dir)
    output_dir = resolve_project_path(args.output_dir)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    summaries = [apply_split(split=split, tacred_dir=tacred_dir, patch_dir=patch_dir, output_dir=output_dir) for split in splits]
    summary = {"tacred_dir": str(tacred_dir), "patch_dir": str(patch_dir), "output_dir": str(output_dir), "splits": summaries}
    summary_path = output_dir / "retacred_build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), "splits": summaries}, sort_keys=True))


if __name__ == "__main__":
    main()