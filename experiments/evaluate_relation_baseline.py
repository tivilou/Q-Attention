#!/usr/bin/env python3
"""Evaluate one frozen relation baseline on every declared split.

The script is intentionally training-free. It can therefore repair an
incomplete historical baseline report when the original checkpoint and data
are still available, while keeping the test split outside model selection.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.experiments import choose_device, load_relation_run, make_relation_loader, move_batch  # noqa: E402
from q_attention.metrics import classification_report  # noqa: E402
from q_attention.tasks.relation import load_relation_jsonl  # noqa: E402


EXPECTED_SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--train-path", default="data/relation/retacred/train.jsonl", type=Path)
    parser.add_argument("--valid-path", default="data/relation/retacred/valid.jsonl", type=Path)
    parser.add_argument("--test-path", default="data/relation/retacred/test.jsonl", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--write-predictions", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def evaluate_split(
    model: torch.nn.Module,
    records: list[Any],
    vocab: dict[str, int],
    label_to_id: dict[str, int],
    id_to_label: dict[int, str],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    unknown = sorted({record.label for record in records if record.label not in label_to_id})
    if unknown:
        raise ValueError(f"split contains labels absent from baseline label map: {unknown}")
    loader = make_relation_loader(records, vocab, label_to_id, batch_size=batch_size, shuffle=False)
    predictions: list[int] = []
    labels: list[int] = []
    total_loss = 0.0
    total_items = 0
    model.eval()
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            logits = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["subject_mask"],
                batch["object_mask"],
            )
            if not torch.isfinite(logits).all():
                raise FloatingPointError("baseline produced non-finite logits")
            loss = F.cross_entropy(logits, batch["labels"])
            if not torch.isfinite(loss):
                raise FloatingPointError("baseline produced non-finite loss")
            total_loss += float(loss.item()) * int(batch["labels"].shape[0])
            total_items += int(batch["labels"].shape[0])
            predictions.extend(logits.argmax(dim=-1).detach().cpu().tolist())
            labels.extend(batch["labels"].detach().cpu().tolist())

    report = classification_report(
        predictions,
        labels,
        len(label_to_id),
        label_names=id_to_label,
    )
    return {
        "items": total_items,
        "batches": len(loader),
        "metrics": {
            **report["metrics"],
            "loss": total_loss / max(total_items, 1),
        },
        "per_class": report["per_class"],
        "confusion_matrix": report["confusion_matrix"],
        "predictions": predictions,
        "labels": labels,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Re-TACRED Baseline 完整评估",
        "",
        f"- schema: `{summary['schema']}`",
        f"- seed: `{summary['seed']}`",
        f"- checkpoint: `{summary['checkpoint_sha256']}`",
        "- checkpoint selection: `valid macro-F1 then loss`",
        "- test isolation: `test was not used for training or checkpoint selection`",
        "",
        "| split | micro-P | micro-R | micro-F1 | accuracy | macro-P | macro-R | macro-F1 | loss | items |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in EXPECTED_SPLITS:
        metrics = summary["splits"][split]["metrics"]
        lines.append(
            f"| {split} | {metrics['micro_precision']:.6f} | {metrics['micro_recall']:.6f} | "
            f"{metrics['micro_f1']:.6f} | {metrics['accuracy']:.6f} | "
            f"{metrics['macro_precision']:.6f} | {metrics['macro_recall']:.6f} | "
            f"{metrics['macro_f1']:.6f} | {metrics['loss']:.6f} | {summary['splits'][split]['items']} |"
        )
    lines.extend(["", "## Per-class test metrics", "", "| id | relation | precision | recall | F1 | support |", "| ---: | --- | ---: | ---: | ---: | ---: |"])
    for label_id, row in summary["splits"]["test"]["per_class"].items():
        lines.append(
            f"| {label_id} | {row['label']} | {row['precision']:.6f} | {row['recall']:.6f} | "
            f"{row['f1']:.6f} | {row['support']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_summary_data(summary: dict[str, Any], path: Path) -> None:
    """Write a compact, line-oriented identity record for report exporters."""
    lines = [
        "schema=q-attention.retacred-baseline-run-summary-data.v1",
        f"status={summary['status']}",
        f"seed={summary['seed']}",
        f"checkpoint_sha256={summary['checkpoint_sha256']}",
        f"selection_metric={summary['selection_metric']}",
        "test_used_for_training_or_selection=false",
    ]
    for split in EXPECTED_SPLITS:
        data = summary["data"][split]
        result = summary["splits"][split]
        lines.extend(
            [
                f"split.{split}.records={data['records']}",
                f"split.{split}.sha256={data['sha256']}",
                f"split.{split}.evaluated_items={result['items']}",
                f"split.{split}.batches={result['batches']}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    model_dir = resolve_path(args.model_dir).resolve()
    output_dir = resolve_path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    artifacts = load_relation_run(model_dir, device)
    split_paths = {
        "train": resolve_path(args.train_path).resolve(),
        "valid": resolve_path(args.valid_path).resolve(),
        "test": resolve_path(args.test_path).resolve(),
    }
    records = {split: load_relation_jsonl(path) for split, path in split_paths.items()}
    split_results = {
        split: evaluate_split(
            artifacts.model,
            records[split],
            artifacts.vocab,
            artifacts.label_to_id,
            artifacts.id_to_label,
            device,
            args.batch_size,
        )
        for split in EXPECTED_SPLITS
    }
    prediction_dir = output_dir / "predictions"
    for split in EXPECTED_SPLITS:
        result = split_results[split]
        if args.write_predictions:
            prediction_dir.mkdir(parents=True, exist_ok=True)
            prediction_path = prediction_dir / f"{split}.json"
            prediction_path.write_text(
                json.dumps({"labels": result["labels"], "predictions": result["predictions"]}, separators=(",", ":")),
                encoding="utf-8",
            )
            result["predictions_sha256"] = sha256_file(prediction_path)
        result.pop("predictions", None)
        result.pop("labels", None)
    metrics_path = model_dir / "metrics.json"
    model_path = model_dir / "model.pt"
    summary: dict[str, Any] = {
        "schema": "q-attention.retacred-baseline-evaluation.v1",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(artifacts.args.get("seed", 13)),
        "device": str(device),
        "batch_size": args.batch_size,
        "model_dir": str(model_dir),
        "checkpoint_sha256": sha256_file(model_path),
        "baseline_metrics_sha256": sha256_file(metrics_path),
        "label_to_id": artifacts.label_to_id,
        "id_to_label": {str(key): value for key, value in artifacts.id_to_label.items()},
        "selection_metric": artifacts.metrics.get("selection_metric", "macro_f1_then_loss"),
        "best_epoch": artifacts.metrics.get("best_epoch"),
        "test_used_for_training_or_selection": False,
        "splits": split_results,
        "data": {
            split: {
                "path": str(path),
                "records": len(records[split]),
                "sha256": sha256_file(path),
            }
            for split, path in split_paths.items()
        },
        "provenance": {
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_branch": git_output("branch", "--show-current"),
            "git_dirty": bool(git_output("status", "--porcelain")),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str, sort_keys=True), encoding="utf-8")
    write_run_summary_data(summary, output_dir / "run_summary.data")
    write_markdown(summary, output_dir / "run_summary.md")
    print(json.dumps({"event": "baseline_evaluation_complete", "output_dir": str(output_dir), "status": summary["status"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
