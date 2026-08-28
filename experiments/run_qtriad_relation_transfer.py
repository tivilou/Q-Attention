#!/usr/bin/env python3
"""Run one complete Re-TACRED Q-TRIAD seed under the frozen contract."""

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
from q_attention.plugins.q_triad import QTriadAttentionScoreKernel  # noqa: E402
from run_q_causal_value_evidence_relation_smoke import (  # noqa: E402
    materialize_subset,
    resolve_path,
    run_logged_command,
)
from run_q_causal_value_evidence_relation_transfer import (  # noqa: E402
    evaluate,
    metric_delta,
    train_kernel,
)
from q_attention.tasks.relation import load_relation_jsonl  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "retacred_qtriad_formal_single_seed.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every-batches", type=int, default=50)
    parser.add_argument("--started-at-utc", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--python-bin", default=sys.executable, help=argparse.SUPPRESS)
    return parser.parse_args()


def build_kernel(mode: str, model: torch.nn.Module, seed: int, config: dict[str, Any]) -> QTriadAttentionScoreKernel:
    kernel_config = config["kernel"]
    return QTriadAttentionScoreKernel(
        num_layers=model.config.num_layers,
        num_heads=model.config.num_heads,
        head_dim=model.config.dim // model.config.num_heads,
        num_qubits=int(kernel_config["num_qubits"]),
        circuit_depth=int(kernel_config["circuit_depth"]),
        angle_scale=float(kernel_config["angle_scale"]),
        max_gain=float(kernel_config["max_gain"]),
        initial_gain=float(kernel_config["initial_gain"]),
        seed=seed + 307,
        control_mode=mode,
    )


def evaluate_selector(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    label_count: int,
    kernel: QTriadAttentionScoreKernel | None,
    stage: str,
) -> dict[str, Any]:
    return evaluate(
        model,
        loader,
        device,
        label_count,
        kernel=kernel,
        stage=stage,
        log_every_batches=50,
        collect_geometry=True,
    )


def main() -> int:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "q-attention.qtriad-formal-single-seed.v1":
        raise ValueError("unsupported Q-TRIAD formal config")
    seed = int(config["seed"] if args.seed is None else args.seed)
    if seed != 13:
        raise ValueError("the formal handoff contract is frozen to seed 13")
    selectors = list(config["selectors"])
    if selectors[0] != "disabled" or config["candidate"] not in selectors:
        raise ValueError("config must include disabled and the candidate selector")
    if config["matched_control"] not in selectors:
        raise ValueError("config must include the matched classical control")
    if args.log_every_batches <= 0:
        raise ValueError("--log-every-batches must be positive")
    device = choose_device(args.device)
    stamp = args.started_at_utc or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if len(stamp) != 16 or not stamp.endswith("Z"):
        raise ValueError("--started-at-utc must use UTC format YYYYMMDDTHHMMSSZ")
    run_dir = args.output_dir or ROOT / "runs" / "retacred_qtriad_formal_single_seed" / f"{stamp}_seed13"
    run_dir = resolve_path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_source = resolve_path(config["train_path"])
    valid_source = resolve_path(config["valid_path"])
    test_source = resolve_path(config["test_path"])
    valid_info = materialize_subset(valid_source, data_dir / "valid.jsonl", 0, seed=seed + 101, split="valid")
    valid_records = load_relation_jsonl(data_dir / "valid.jsonl")
    required_labels = {record.label for record in valid_records}
    train_info = materialize_subset(train_source, data_dir / "train.jsonl", 0, seed=seed, split="train", required_labels=required_labels)
    test_info = materialize_subset(test_source, data_dir / "test.jsonl", 0, seed=seed + 211, split="test")
    train_records = load_relation_jsonl(data_dir / "train.jsonl")
    test_records = load_relation_jsonl(data_dir / "test.jsonl")
    expected = config["expected_records"]
    for name, info in (("train", train_info), ("valid", valid_info), ("test", test_info)):
        if int(info["records"]) != int(expected[name]):
            raise ValueError(f"{name} record count differs from frozen contract")
    max_length = max(len(record.tokens) for record in train_records + valid_records + test_records) + 4
    model_config = config["model"]
    baseline_dir = run_dir / "baseline"
    baseline_command = [
        args.python_bin,
        str(ROOT / "experiments" / "train_relation_baseline.py"),
        "--train_path", str(data_dir / "train.jsonl"),
        "--valid_path", str(data_dir / "valid.jsonl"),
        "--output_dir", str(baseline_dir),
        "--device", str(device),
        "--epochs", str(config["baseline"]["epochs"]),
        "--batch_size", str(config["baseline"]["batch_size"]),
        "--lr", str(config["baseline"]["lr"]),
        "--dim", str(model_config["dim"]),
        "--num_layers", str(model_config["num_layers"]),
        "--num_heads", str(model_config["num_heads"]),
        "--ff_dim", str(model_config["ff_dim"]),
        "--dropout", str(model_config["dropout"]),
        "--max_length", str(max_length),
        "--seed", str(seed),
    ]
    run_logged_command(baseline_command, run_dir / "baseline_train.log")
    artifacts = load_relation_run(baseline_dir, device)
    valid_loader = make_relation_loader(valid_records, artifacts.vocab, artifacts.label_to_id, batch_size=int(config["kernel"]["batch_size"]))
    test_loader = make_relation_loader(test_records, artifacts.vocab, artifacts.label_to_id, batch_size=int(config["kernel"]["batch_size"]))
    for parameter in artifacts.model.parameters():
        parameter.requires_grad_(False)
    baseline_valid = evaluate_selector(artifacts.model, valid_loader, device, len(artifacts.label_to_id), None, "baseline_valid")
    baseline_test = evaluate_selector(artifacts.model, test_loader, device, len(artifacts.label_to_id), None, "baseline_test")
    train_args = argparse.Namespace(
        batch_size=int(config["kernel"]["batch_size"]),
        epochs=int(config["kernel"]["epochs"]),
        kernel_lr=float(config["kernel"]["lr"]),
        log_every_batches=args.log_every_batches,
    )
    rows: list[dict[str, Any]] = []
    for selector in selectors:
        selector_dir = run_dir / "selectors" / selector
        selector_dir.mkdir(parents=True, exist_ok=True)
        if selector == "disabled":
            valid_result = baseline_valid
            test_result = baseline_test
            train_result = {"history": [], "best_epoch": 0, "runtime_seconds": 0.0}
            kernel = None
            metadata: dict[str, Any] = {"type": "disabled"}
            trainable_parameters = 0
        else:
            kernel = build_kernel(selector, artifacts.model, seed, config).to(device)
            train_result = train_kernel(
                artifacts.model,
                kernel,
                train_records,
                valid_loader,
                artifacts,
                device,
                selector,
                seed,
                train_args,
                selector_dir,
            )
            valid_result = evaluate_selector(artifacts.model, valid_loader, device, len(artifacts.label_to_id), kernel, f"{selector}_valid_final")
            test_result = evaluate_selector(artifacts.model, test_loader, device, len(artifacts.label_to_id), kernel, f"{selector}_test")
            metadata = kernel.metadata()
            trainable_parameters = sum(parameter.numel() for parameter in kernel.parameters())
            torch.save({"state_dict": kernel.state_dict(), "metadata": metadata}, selector_dir / "best_kernel_with_metadata.pt")
        row = {
            "selector": selector,
            "seed": seed,
            "valid": valid_result,
            "test": {**test_result, "delta_vs_baseline": metric_delta(test_result["metrics"], baseline_test["metrics"])},
            "train": train_result,
            "metadata": metadata,
            "trainable_parameters": trainable_parameters,
            "finite": all(torch.isfinite(torch.tensor(value)) for value in list(valid_result["metrics"].values()) + list(test_result["metrics"].values())),
        }
        (selector_dir / "metrics.json").write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
        rows.append(row)
        print(json.dumps({"event": "selector_complete", "selector": selector}, sort_keys=True), flush=True)
    by_name = {row["selector"]: row for row in rows}
    candidate = by_name[config["candidate"]]
    matched = by_name[config["matched_control"]]
    disabled = by_name["disabled"]
    candidate_minus_disabled = metric_delta(candidate["test"]["metrics"], disabled["test"]["metrics"])
    candidate_minus_matched = metric_delta(candidate["test"]["metrics"], matched["test"]["metrics"])
    gates = {
        "candidate_minus_disabled_macro_f1": candidate_minus_disabled["delta_macro_f1"],
        "candidate_minus_matched_macro_f1": candidate_minus_matched["delta_macro_f1"],
        "practical_gain_gate": candidate_minus_disabled["delta_macro_f1"] >= float(config["gates"]["minimum_candidate_minus_disabled_macro_f1"]),
        "matched_comparator_gate": candidate_minus_matched["delta_macro_f1"] >= float(config["gates"]["minimum_candidate_minus_matched_macro_f1"]),
        "finite_metrics": all(row["finite"] for row in rows),
        "test_used_for_training_or_selection": False,
    }
    summary = {
        "schema_version": "q-attention.qtriad-formal-single-seed.run.v1",
        "name": config["name"],
        "formal_experiment": True,
        "stage": "formal_single_seed",
        "seed": seed,
        "run_dir": str(run_dir),
        "device": str(device),
        "selectors": selectors,
        "candidate": config["candidate"],
        "matched_control": config["matched_control"],
        "data": {"train": train_info, "valid": valid_info, "test": test_info},
        "baseline": {"valid": baseline_valid, "test": baseline_test, "command": baseline_command},
        "rows": rows,
        "candidate_minus_disabled": candidate_minus_disabled,
        "candidate_minus_matched": candidate_minus_matched,
        "gates": gates,
        "test_used_for_training_or_selection": False,
        "claim_limits": config["claim_limits"],
        "provenance": {
            "config_path": str(config_path),
            "config_sha256": sha256(config_path),
            "git_revision": git_output("rev-parse", "HEAD"),
            "git_branch": git_output("branch", "--show-current"),
            "git_dirty": bool(git_output("status", "--porcelain")),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "started_at_utc": stamp,
        },
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Q-TRIAD Re-TACRED Formal Single Seed",
        "",
        "This is one complete seed-13 run under the frozen natural-task contract.",
        "",
        f"- candidate: `{config['candidate']}`",
        f"- matched control: `{config['matched_control']}`",
        f"- candidate minus disabled test macro-F1: `{candidate_minus_disabled['delta_macro_f1']:.6f}`",
        f"- candidate minus matched test macro-F1: `{candidate_minus_matched['delta_macro_f1']:.6f}`",
        f"- practical gain gate: `{str(gates['practical_gain_gate']).lower()}`",
        f"- matched comparator gate: `{str(gates['matched_comparator_gate']).lower()}`",
        "",
        "The test split is evaluated only after training and validation selection. This single seed does not authorize multi-seed replication.",
    ]
    (run_dir / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run_dir / "RUN_COMPLETE").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
