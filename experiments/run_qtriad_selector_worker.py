#!/usr/bin/env python3
"""Run one Q-TRIAD selector worker for the formal single-seed scheduler."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXPERIMENTS = ROOT / "experiments"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from q_attention.experiments.relation_steering import (  # noqa: E402
    choose_device,
    load_relation_run,
    make_relation_loader,
)
from q_attention.tasks.relation import load_relation_jsonl  # noqa: E402
from run_q_causal_value_evidence_relation_transfer import train_kernel  # noqa: E402
from run_qtriad_relation_transfer import build_kernel, evaluate_selector  # noqa: E402


MIN_WORKER_FREE_MIB = 8 * 1024


def check_worker_gpu_capacity(device_name: str) -> None:
    """Reject a worker start when its assigned physical GPU is already busy."""
    if device_name != "cuda":
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.isdigit():
        return
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"worker GPU capacity check failed: {result.stderr.strip()}")
    rows = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3:
            continue
        try:
            rows[int(fields[0])] = (int(fields[1]), int(fields[2]))
        except ValueError:
            continue
    physical_id = int(visible)
    if physical_id not in rows:
        raise RuntimeError(f"worker GPU capacity check could not find physical GPU {physical_id}")
    free_mib, total_mib = rows[physical_id]
    if free_mib < MIN_WORKER_FREE_MIB:
        raise RuntimeError(
            f"worker GPU {physical_id} has only {free_mib} MiB free of {total_mib} MiB; "
            f"at least {MIN_WORKER_FREE_MIB} MiB is required. "
            "A competing CUDA process is using the assigned GPU."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--log-every-batches", type=int, default=50)
    parser.add_argument("--pair-chunk-size", type=int, default=None)
    parser.add_argument("--activation-checkpointing", type=int, choices=(0, 1), default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    check_worker_gpu_capacity(args.device)
    device = choose_device(args.device)
    artifacts = load_relation_run(args.baseline_dir, device)
    data_dir = args.data_dir
    train_records = load_relation_jsonl(data_dir / "train.jsonl")
    valid_records = load_relation_jsonl(data_dir / "valid.jsonl")
    test_records = load_relation_jsonl(data_dir / "test.jsonl")
    kernel_config = config["kernel"]
    valid_loader = make_relation_loader(
        valid_records,
        artifacts.vocab,
        artifacts.label_to_id,
        batch_size=int(kernel_config["batch_size"]),
    )
    test_loader = make_relation_loader(
        test_records,
        artifacts.vocab,
        artifacts.label_to_id,
        batch_size=int(kernel_config["batch_size"]),
    )
    for parameter in artifacts.model.parameters():
        parameter.requires_grad_(False)
    baseline_test = json.loads(
        (args.data_dir.parent / "baseline_eval.json").read_text(encoding="utf-8")
    )["test"]
    kernel = build_kernel(
        args.selector,
        artifacts.model,
        args.seed,
        config,
        pair_chunk_size=args.pair_chunk_size,
        activation_checkpointing=(
            None
            if args.activation_checkpointing is None
            else bool(args.activation_checkpointing)
        ),
    ).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_args = argparse.Namespace(
        batch_size=int(kernel_config["batch_size"]),
        epochs=int(kernel_config["epochs"]),
        kernel_lr=float(kernel_config["lr"]),
        log_every_batches=args.log_every_batches,
    )
    train_result = train_kernel(
        artifacts.model,
        kernel,
        train_records,
        valid_loader,
        artifacts,
        device,
        args.selector,
        args.seed,
        train_args,
        args.output_dir,
    )
    valid_result = evaluate_selector(
        artifacts.model,
        valid_loader,
        device,
        len(artifacts.label_to_id),
        kernel,
        f"{args.selector}_valid_final",
    )
    test_result = evaluate_selector(
        artifacts.model,
        test_loader,
        device,
        len(artifacts.label_to_id),
        kernel,
        f"{args.selector}_test",
    )
    from run_qtriad_relation_transfer import metric_delta

    metadata: dict[str, Any] = kernel.metadata()
    trainable_parameters = sum(parameter.numel() for parameter in kernel.parameters())
    torch.save(
        {"state_dict": kernel.state_dict(), "metadata": metadata},
        args.output_dir / "best_kernel_with_metadata.pt",
    )
    row = {
        "selector": args.selector,
        "seed": args.seed,
        "valid": valid_result,
        "test": {
            **test_result,
            "delta_vs_baseline": metric_delta(test_result["metrics"], baseline_test["metrics"]),
        },
        "train": train_result,
        "metadata": metadata,
        "trainable_parameters": trainable_parameters,
        "finite": all(
            torch.isfinite(torch.tensor(value))
            for value in list(valid_result["metrics"].values())
            + list(test_result["metrics"].values())
        ),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(row, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "event": "selector_complete",
                "selector": args.selector,
                "device": str(device),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
