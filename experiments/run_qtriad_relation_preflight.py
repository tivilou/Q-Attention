#!/usr/bin/env python3
"""Run the bounded Q-TRIAD/Re-TACRED integration preflight.

This command checks data loading, the score-hook action, finite metrics, and
gradient flow on a small sampled split.  It is not a formal task experiment.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.adapters import AttentionScoreHookConfig, AttentionScoreKernelAdapter  # noqa: E402
from q_attention.experiments.relation_steering import (  # noqa: E402
    choose_device,
    load_relation_run,
    make_relation_loader,
    move_batch,
)
from q_attention.metrics import classification_metrics  # noqa: E402
from q_attention.plugins.q_triad import QTriadAttentionScoreKernel  # noqa: E402
from q_attention.tasks.relation import (  # noqa: E402
    load_relation_jsonl,
    sample_relation_records,
    sample_relation_records_proportional,
    write_relation_jsonl,
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_split(source: Path, destination: Path, limit: int, seed: int, split: str, required_labels: set[str] | None = None) -> dict[str, Any]:
    records = load_relation_jsonl(source)
    if limit <= 0 or len(records) <= limit:
        selected = records
        sampling = "source"
    elif split == "train":
        by_label: dict[str, list[Any]] = {}
        for record in records:
            by_label.setdefault(record.label, []).append(record)
        selected = [by_label[label][0] for label in sorted(required_labels or ())]
        selected.extend(sample_relation_records([r for r in records if r not in selected], limit - len(selected), seed=seed))
        sampling = "label_coverage_plus_random"
    else:
        selected = sample_relation_records_proportional(records, limit, seed=seed)
        sampling = "proportional_stratified"
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_relation_jsonl(selected, destination)
    return {
        "source_path": str(source),
        "source_sha256": sha256(source),
        "path": str(destination),
        "records": len(selected),
        "source_records": len(records),
        "sampling": sampling,
        "seed": seed,
    }


def evaluate(model: torch.nn.Module, loader: Any, device: torch.device, labels: int, adapter: AttentionScoreKernelAdapter | None = None) -> dict[str, Any]:
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []
    total_loss = 0.0
    total = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        if adapter is None:
            logits = model(batch["input_ids"], batch["attention_mask"], batch["subject_mask"], batch["object_mask"])
        else:
            config = AttentionScoreHookConfig(
                attention_mask=batch["attention_mask"],
                subject_mask=batch["subject_mask"],
                object_mask=batch["object_mask"],
            )
            with adapter.steering(config):
                logits = model(batch["input_ids"], batch["attention_mask"], batch["subject_mask"], batch["object_mask"])
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite logits in Q-TRIAD preflight")
        total_loss += float(F.cross_entropy(logits, batch["labels"]).item()) * int(batch["labels"].shape[0])
        total += int(batch["labels"].shape[0])
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        targets.extend(batch["labels"].cpu().tolist())
    metrics = classification_metrics(predictions, targets, labels)
    metrics["loss"] = total_loss / max(total, 1)
    return {"metrics": metrics, "items": total}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/retacred_qtriad_natural_preflight.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--python-bin", default=sys.executable, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "q-attention.qtriad-natural-preflight.v1":
        raise ValueError("unsupported Q-TRIAD natural preflight schema")
    seed = int(config["optimization"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    device = choose_device(args.device or "cpu")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = resolve_path(args.output_dir) if args.output_dir else ROOT / "runs" / config["experiment_id"] / stamp
    output.mkdir(parents=True, exist_ok=False)
    data_dir = output / "preflight_data"
    train_source = resolve_path(config["train_path"])
    valid_source = resolve_path(config["valid_path"])
    valid_info = sample_split(valid_source, data_dir / "valid.jsonl", int(config["max_valid_records"]), seed + 101, "valid")
    valid_records = load_relation_jsonl(data_dir / "valid.jsonl")
    train_info = sample_split(
        train_source,
        data_dir / "train.jsonl",
        int(config["max_train_records"]),
        seed,
        "train",
        required_labels={record.label for record in valid_records},
    )
    train_records = load_relation_jsonl(data_dir / "train.jsonl")
    max_length = max(len(record.tokens) for record in train_records + valid_records) + 4
    baseline_dir = output / "baseline"
    command = [
        args.python_bin,
        str(ROOT / "experiments" / "train_relation_baseline.py"),
        "--train_path", str(data_dir / "train.jsonl"),
        "--valid_path", str(data_dir / "valid.jsonl"),
        "--output_dir", str(baseline_dir),
        "--device", str(device),
        "--epochs", str(config["optimization"]["baseline_epochs"]),
        "--batch_size", str(config["optimization"]["baseline_batch_size"]),
        "--lr", str(config["optimization"]["baseline_lr"]),
        "--dim", str(config["model"]["dim"]),
        "--num_layers", str(config["model"]["num_layers"]),
        "--num_heads", str(config["model"]["num_heads"]),
        "--ff_dim", str(config["model"]["ff_dim"]),
        "--dropout", str(config["model"]["dropout"]),
        "--max_length", str(max_length),
        "--seed", str(seed),
    ]
    import subprocess

    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    (output / "baseline_train.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"baseline preflight failed with exit code {result.returncode}")
    artifacts = load_relation_run(baseline_dir, device)
    loader = make_relation_loader(valid_records, artifacts.vocab, artifacts.label_to_id, batch_size=int(config["optimization"]["batch_size"]))
    for parameter in artifacts.model.parameters():
        parameter.requires_grad_(False)
    baseline = evaluate(artifacts.model, loader, device, len(artifacts.label_to_id))
    kernel = QTriadAttentionScoreKernel(
        num_layers=artifacts.model.config.num_layers,
        num_heads=artifacts.model.config.num_heads,
        head_dim=artifacts.model.config.dim // artifacts.model.config.num_heads,
        num_qubits=int(config["qtriad"]["num_qubits"]),
        circuit_depth=int(config["qtriad"]["circuit_depth"]),
        angle_scale=float(config["qtriad"]["angle_scale"]),
        max_gain=float(config["qtriad"]["max_gain"]),
        initial_gain=float(config["qtriad"]["initial_gain"]),
        seed=int(config["qtriad"]["seed"]),
    ).to(device)
    adapter = AttentionScoreKernelAdapter(artifacts.model, artifacts.model.score_module_paths, kernel)  # type: ignore[arg-type]
    candidate = evaluate(artifacts.model, loader, device, len(artifacts.label_to_id), adapter)
    batch = move_batch(next(iter(loader)), device)
    artifacts.model.zero_grad(set_to_none=True)
    kernel.zero_grad(set_to_none=True)
    with adapter.steering(AttentionScoreHookConfig(attention_mask=batch["attention_mask"], subject_mask=batch["subject_mask"], object_mask=batch["object_mask"])):
        loss = F.cross_entropy(artifacts.model(batch["input_ids"], batch["attention_mask"], batch["subject_mask"], batch["object_mask"]), batch["labels"])
    loss.backward()
    gradients = [parameter.grad for parameter in kernel.parameters() if parameter.requires_grad]
    summary = {
        "schema_version": "q-attention.qtriad-natural-preflight.run.v1",
        "status": "pass" if gradients and all(g is not None and torch.isfinite(g).all() for g in gradients) else "fail",
        "stage": "preflight",
        "formal_experiment": False,
        "device": str(device),
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "data": {"train": train_info, "valid": valid_info},
        "baseline": baseline,
        "q_triad": {**candidate, "metadata": kernel.metadata()},
        "gradient_probe": {"finite": bool(gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)), "nonzero_tensors": sum(int(g is not None and bool(torch.any(g.abs() > 0).item())) for g in gradients)},
        "contract": config["contract"],
        "claim_limits": {"task_utility": "not_established_by_preflight", "natural_transfer": "not_established", "quantum_advantage": "not_established"},
        "provenance": {"git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip(), "git_branch": subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip(), "torch": torch.__version__},
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "run_summary.md").write_text("# Q-TRIAD Natural Task Preflight\n\nThis is an integration preflight, not a task-utility result.\n\n" + f"- status: `{summary['status']}`\n- device: `{device}`\n- baseline macro-F1: `{baseline['metrics']['macro_f1']:.6f}`\n- Q-TRIAD macro-F1: `{candidate['metrics']['macro_f1']:.6f}`\n- finite gradients: `{summary['gradient_probe']['finite']}`\n", encoding="utf-8")
    (output / "PREFLIGHT_COMPLETE").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "status": summary["status"]}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
