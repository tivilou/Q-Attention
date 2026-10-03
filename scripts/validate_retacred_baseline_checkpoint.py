#!/usr/bin/env python3
"""Validate a reusable completed baseline checkpoint before creating a run."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

EXPECTED_ARGS = {
    "train_path": "data/relation/retacred/train.jsonl",
    "valid_path": "data/relation/retacred/valid.jsonl",
    "epochs": 12, "batch_size": 128, "lr": 0.0005, "dim": 128,
    "num_layers": 4, "num_heads": 8, "ff_dim": 256, "dropout": 0.1,
    "max_length": 128, "selection_metric": "macro_f1_then_loss", "device": "cuda",
}
REQUIRED_FILES = ("model.pt", "metrics.json", "vocab.json", "labels.json")

def fail(message: str) -> None:
    raise SystemExit(f"Invalid baseline checkpoint: {message}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--expected-seed", type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    model_dir = args.model_dir.resolve()
    try:
        model_dir.relative_to((root / "runs").resolve())
    except ValueError:
        fail(f"model directory must be inside {root / 'runs'}")
    if not model_dir.is_dir():
        fail(f"model directory does not exist: {model_dir}")
    if (model_dir / "RUN_PAUSED").exists():
        fail("checkpoint has a RUN_PAUSED marker")
    for name in REQUIRED_FILES:
        path = model_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            fail(f"required non-empty file is missing: {path}")
    try:
        metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
        vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
        labels = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"checkpoint metadata is not valid JSON: {exc}")
    checkpoint_args = metrics.get("args")
    if not isinstance(checkpoint_args, dict):
        fail("metrics.json has no recorded training args")
    if checkpoint_args.get("seed") != args.expected_seed:
        fail(f"seed mismatch: requested {args.expected_seed}, checkpoint records {checkpoint_args.get('seed')!r}")
    for name, expected in EXPECTED_ARGS.items():
        actual = checkpoint_args.get(name)
        matches = (
            isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            if isinstance(expected, float) else actual == expected
        )
        if not matches:
            fail(f"training arg {name} mismatch: expected {expected!r}, got {actual!r}")
    if metrics.get("selection_metric") != EXPECTED_ARGS["selection_metric"]:
        fail("metrics.json selection_metric does not match the frozen baseline contract")
    if not isinstance(metrics.get("best_epoch"), int) or not 1 <= metrics["best_epoch"] <= 12:
        fail("metrics.json has an invalid best_epoch")
    if not isinstance(metrics.get("best_valid"), dict):
        fail("metrics.json has no selected validation metrics")
    if not isinstance(vocab, dict) or not vocab:
        fail("vocab.json must contain a non-empty token mapping")
    if not isinstance(labels, dict) or not labels:
        fail("labels.json must contain a non-empty label mapping")
    print(f"checkpoint_seed={checkpoint_args['seed']}")
    print("baseline checkpoint contract=OK")

if __name__ == "__main__":
    main()
