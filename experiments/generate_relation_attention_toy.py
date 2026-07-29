from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any


RELATION_TEMPLATES = {
    "born_in": ("was", "born", "in"),
    "capital_of": ("is", "the", "capital", "of"),
    "founded_by": ("founded",),
    "works_with": ("worked", "with"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a compositional relation-attention Toy dataset."
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_per_label", type=int, default=64)
    parser.add_argument("--valid_per_label", type=int, default=32)
    parser.add_argument("--entity_pool_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def relation_record(label: str, subject: str, object_: str) -> dict[str, Any]:
    template = RELATION_TEMPLATES[label]
    tokens = [subject, *template, object_]
    return {
        "tokens": tokens,
        "subject": [0, 1],
        "object": [len(tokens) - 1, len(tokens)],
        "label": label,
    }


def generate_records(
    *,
    train_per_label: int,
    valid_per_label: int,
    entity_pool_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if train_per_label <= 0 or valid_per_label <= 0 or entity_pool_size <= 1:
        raise ValueError("dataset sizes must be positive and entity_pool_size must exceed one")
    available_pairs = entity_pool_size * entity_pool_size
    if train_per_label + valid_per_label > available_pairs:
        raise ValueError("requested examples exceed the available subject-object pairs")

    subjects = [f"entity_{index}" for index in range(entity_pool_size)]
    objects = [f"target_{index}" for index in range(entity_pool_size)]
    train: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for label_index, label in enumerate(sorted(RELATION_TEMPLATES)):
        pairs = [(subject, object_) for subject in subjects for object_ in objects]
        random.Random(seed + 1009 * label_index).shuffle(pairs)
        selected_train = pairs[:train_per_label]
        selected_valid = pairs[train_per_label : train_per_label + valid_per_label]
        train.extend(relation_record(label, *pair) for pair in selected_train)
        valid.extend(relation_record(label, *pair) for pair in selected_valid)

    random.Random(seed + 7919).shuffle(train)
    random.Random(seed + 15401).shuffle(valid)
    return train, valid


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def validate_compositional_split(
    train: list[dict[str, Any]],
    valid: list[dict[str, Any]],
) -> dict[str, int]:
    train_vocab = {token for record in train for token in record["tokens"]}
    valid_vocab = {token for record in valid for token in record["tokens"]}
    train_pairs = {
        (record["label"], record["tokens"][0], record["tokens"][-1])
        for record in train
    }
    valid_pairs = {
        (record["label"], record["tokens"][0], record["tokens"][-1])
        for record in valid
    }
    valid_oov = valid_vocab - train_vocab
    pair_overlap = train_pairs & valid_pairs
    if valid_oov:
        raise RuntimeError(f"validation split contains OOV tokens: {sorted(valid_oov)}")
    if pair_overlap:
        raise RuntimeError("training and validation relation-entity pairs overlap")
    return {"valid_oov_tokens": 0, "pair_overlap": 0}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, valid = generate_records(
        train_per_label=args.train_per_label,
        valid_per_label=args.valid_per_label,
        entity_pool_size=args.entity_pool_size,
        seed=args.seed,
    )
    validation = validate_compositional_split(train, valid)
    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(valid_path, valid)
    metadata = {
        "args": vars(args),
        "labels": sorted(RELATION_TEMPLATES),
        "train_examples": len(train),
        "valid_examples": len(valid),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "split_validation": validation,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
