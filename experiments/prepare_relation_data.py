from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.tasks.relation import sample_relation_records, write_relation_jsonl  # noqa: E402
from q_attention.tasks.relation_formats import (  # noqa: E402
    RELATION_DATA_FORMATS,
    load_relation_records,
    relation_record_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert real relation extraction data to Q-Attention JSONL.")
    parser.add_argument("--format", required=True, choices=RELATION_DATA_FORMATS, help="Input data format")
    parser.add_argument("--dataset_name", default=None, help="Short dataset name written to data_config.json")
    parser.add_argument("--train_path", required=True, help="Raw/canonical train split path")
    parser.add_argument("--valid_path", required=True, help="Raw/canonical validation split path")
    parser.add_argument("--test_path", default=None, help="Optional raw/canonical test split path")
    parser.add_argument("--output_dir", required=True, help="Directory for canonical train/valid/test JSONL")
    parser.add_argument("--train_limit", type=int, default=None, help="Optional train subset size for smoke runs")
    parser.add_argument("--valid_limit", type=int, default=None, help="Optional validation subset size for smoke runs")
    parser.add_argument("--test_limit", type=int, default=None, help="Optional test subset size for smoke runs")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no_stratified", action="store_true", help="Use plain random sampling instead of label-stratified sampling")
    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def prepare_split(
    *,
    name: str,
    source_path: Path,
    output_dir: Path,
    data_format: str,
    limit: int | None,
    seed: int,
    stratified: bool,
) -> dict[str, Any]:
    records = load_relation_records(source_path, data_format)
    original_summary = relation_record_summary(records)
    selected = sample_relation_records(records, limit, seed=seed, stratified=stratified)
    output_path = output_dir / f"{name}.jsonl"
    write_relation_jsonl(selected, output_path)
    return {
        "source_path": str(source_path),
        "path": str(output_path),
        "limit": limit,
        "original": original_summary,
        "prepared": relation_record_summary(selected),
    }


def label_gap(train_labels: set[str], other_labels: set[str]) -> list[str]:
    return sorted(other_labels - train_labels)


def main() -> None:
    args = parse_args()
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_args = {
        "train": (args.train_path, args.train_limit),
        "valid": (args.valid_path, args.valid_limit),
    }
    if args.test_path is not None:
        split_args["test"] = (args.test_path, args.test_limit)

    splits: dict[str, dict[str, Any]] = {}
    for split_name, (path_value, limit) in split_args.items():
        splits[split_name] = prepare_split(
            name=split_name,
            source_path=resolve_project_path(path_value),
            output_dir=output_dir,
            data_format=args.format,
            limit=limit,
            seed=args.seed,
            stratified=not args.no_stratified,
        )

    train_labels = set(splits["train"]["prepared"]["label_counts"])
    label_warnings = {
        name: label_gap(train_labels, set(info["prepared"]["label_counts"]))
        for name, info in splits.items()
        if name != "train"
    }
    label_warnings = {name: labels for name, labels in label_warnings.items() if labels}

    config: dict[str, Any] = {
        "task": "relation_extraction",
        "dataset_name": args.dataset_name or args.format,
        "source_format": args.format,
        "seed": args.seed,
        "stratified_sampling": not args.no_stratified,
        "splits": splits,
        "train_path": splits["train"]["path"],
        "valid_path": splits["valid"]["path"],
        "label_warnings": label_warnings,
    }
    if "test" in splits:
        config["test_path"] = splits["test"]["path"]

    config_path = output_dir / "data_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "config_path": str(config_path),
                "dataset_name": config["dataset_name"],
                "source_format": args.format,
                "splits": {name: info["prepared"] for name, info in splits.items()},
                "label_warnings": label_warnings,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()