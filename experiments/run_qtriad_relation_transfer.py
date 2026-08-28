#!/usr/bin/env python3
"""Run one complete Re-TACRED Q-TRIAD seed under the frozen contract."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
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

AUTO_MIN_FREE_MIB = 8 * 1024
HARDWARE_PROFILES: dict[str, dict[str, Any]] = {
    "low_memory": {
        "pair_chunk_size": 64,
        "activation_checkpointing": True,
        "reason": "at least one selected GPU has less than 16 GiB total or 12 GiB free",
    },
    "balanced": {
        "pair_chunk_size": 256,
        "activation_checkpointing": True,
        "reason": "at least one selected GPU has less than 40 GiB total or 28 GiB free",
    },
    "high_memory": {
        "pair_chunk_size": 256,
        "activation_checkpointing": True,
        "reason": "all selected GPUs have at least 40 GiB total and 28 GiB free; retain streamed backward safety",
    },
}


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
    parser.add_argument(
        "--gpus",
        default=None,
        help="comma-separated physical GPU IDs or auto; selectors run as independent workers",
    )
    parser.add_argument(
        "--hardware-profile",
        choices=("config", "auto", "low_memory", "balanced", "high_memory"),
        default="config",
        help="execution-memory profile; auto derives it from selected GPU VRAM",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every-batches", type=int, default=50)
    parser.add_argument("--started-at-utc", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--python-bin", default=sys.executable, help=argparse.SUPPRESS)
    return parser.parse_args()


def query_gpu_inventory() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed while discovering GPUs: {result.stderr.strip()}")
    inventory: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 4)]
        if len(fields) != 5:
            raise RuntimeError(f"unexpected nvidia-smi GPU row: {line!r}")
        index, name, total, free, used = fields
        inventory.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(total),
                "memory_free_mib": int(free),
                "memory_used_mib": int(used),
            }
        )
    if not inventory:
        raise RuntimeError("nvidia-smi reported no GPUs")
    return inventory


def query_compute_apps() -> list[dict[str, Any]]:
    """Return a compact list of processes currently using CUDA compute."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    apps: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3:
            continue
        pid, process_name, used_memory = fields
        try:
            apps.append(
                {
                    "pid": int(pid),
                    "process_name": process_name,
                    "used_memory_mib": int(used_memory),
                }
            )
        except ValueError:
            continue
    return apps


def validate_gpu_capacity(
    gpu_ids: list[int],
    inventory: list[dict[str, Any]],
    *,
    phase: str,
) -> None:
    """Stop before a run if any selected GPU is already too busy."""
    selected = {
        int(item["index"]): item
        for item in inventory
        if int(item["index"]) in set(gpu_ids)
    }
    missing = [gpu_id for gpu_id in gpu_ids if gpu_id not in selected]
    if missing:
        raise RuntimeError(f"{phase}: nvidia-smi did not report selected GPU IDs {missing}")
    insufficient = [
        item
        for item in selected.values()
        if int(item["memory_free_mib"]) < AUTO_MIN_FREE_MIB
    ]
    if not insufficient:
        return
    apps = query_compute_apps()
    app_text = ", ".join(
        f"pid={app['pid']} {app['process_name']} ({app['used_memory_mib']} MiB)"
        for app in apps
    ) or "no process details available"
    details = "; ".join(
        f"GPU {item['index']} free={item['memory_free_mib']} MiB/{item['memory_total_mib']} MiB"
        for item in insufficient
    )
    raise RuntimeError(
        f"{phase}: selected GPU capacity is unsafe ({details}; required at least "
        f"{AUTO_MIN_FREE_MIB} MiB free). Compute apps: {app_text}. "
        "Stop or wait for the competing process, then rerun the unchanged contract."
    )


def resolve_gpu_ids(
    spec: str | None,
    device_name: str,
    inventory: list[dict[str, Any]] | None = None,
) -> list[int]:
    """Validate the requested visible GPUs before creating a run directory."""
    if device_name == "cpu":
        if spec:
            raise ValueError("--gpus cannot be used with --device cpu")
        return []
    auto_mode = bool(spec and spec.strip().lower() == "auto")
    if not auto_mode and not torch.cuda.is_available():
        if device_name == "auto" and spec is None:
            return []
        raise RuntimeError("CUDA requested but no CUDA device is available")
    if spec is None:
        ids = [0]
    elif spec.strip().lower() == "auto":
        inventory = inventory or query_gpu_inventory()
        visible_spec = os.environ.get("CUDA_VISIBLE_DEVICES")
        allowed = None
        if visible_spec:
            visible_fields = [field.strip() for field in visible_spec.split(",")]
            if all(field.isdigit() for field in visible_fields):
                allowed = {int(field) for field in visible_fields}
        candidates = [
            item
            for item in inventory
            if allowed is None or int(item["index"]) in allowed
        ]
        ids = [
            int(item["index"])
            for item in candidates
            if int(item["memory_free_mib"]) >= AUTO_MIN_FREE_MIB
        ]
        if not ids:
            raise RuntimeError(
                f"--gpus auto found no GPU with at least {AUTO_MIN_FREE_MIB // 1024} GiB free"
            )
    else:
        fields = [field.strip() for field in spec.split(",")]
        if not fields or any(not field.isdigit() for field in fields):
            raise ValueError("--gpus must be a comma-separated list of non-negative integers")
        ids = [int(field) for field in fields]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("--gpus must contain at least one unique GPU ID")
    if auto_mode:
        return ids
    available = torch.cuda.device_count()
    visible_spec = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible_ids = None
    if visible_spec:
        visible_fields = [field.strip() for field in visible_spec.split(",")]
        if all(field.isdigit() for field in visible_fields):
            visible_ids = {int(field) for field in visible_fields}
    if visible_ids is not None and not set(ids).issubset(visible_ids):
        raise ValueError(f"requested GPU IDs {ids} are not in CUDA_VISIBLE_DEVICES={visible_spec}")
    if visible_ids is None and any(index >= available for index in ids):
        raise ValueError(f"requested GPU IDs {ids} exceed visible device count {available}")
    return ids


def choose_hardware_profile(
    requested: str,
    config: dict[str, Any],
    gpu_ids: list[int],
    inventory: list[dict[str, Any]],
) -> dict[str, Any]:
    if requested == "config":
        return {
            "name": "config",
            "pair_chunk_size": int(config["kernel"].get("pair_chunk_size", 256)),
            "activation_checkpointing": True,
            "selection_reason": "frozen config profile",
        }
    if requested == "auto":
        selected = [item for item in inventory if int(item["index"]) in set(gpu_ids)]
        if not selected:
            raise RuntimeError("no GPU inventory entries match the selected GPU IDs")
        min_total = min(int(item["memory_total_mib"]) for item in selected)
        min_free = min(int(item["memory_free_mib"]) for item in selected)
        if min_total < 16 * 1024 or min_free < 12 * 1024:
            requested = "low_memory"
        elif min_total < 40 * 1024 or min_free < 28 * 1024:
            requested = "balanced"
        else:
            requested = "high_memory"
        profile = dict(HARDWARE_PROFILES[requested])
        profile.update(
            {
                "name": requested,
                "selection_reason": profile.pop("reason"),
                "minimum_memory_total_mib": min_total,
                "minimum_memory_free_mib": min_free,
            }
        )
        return profile
    profile = dict(HARDWARE_PROFILES[requested])
    profile.update({"name": requested, "selection_reason": "explicit profile override"})
    profile.pop("reason", None)
    return profile


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _terminate_worker(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)


def run_selector_workers(
    *,
    selectors: list[str],
    gpu_ids: list[int],
    args: argparse.Namespace,
    config_path: Path,
    baseline_dir: Path,
    data_dir: Path,
    run_dir: Path,
    seed: int,
    hardware_profile: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run independent selectors with one dynamically scheduled worker per GPU."""
    hardware_profile = hardware_profile or {
        "name": "config",
        "pair_chunk_size": 256,
        "activation_checkpointing": True,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    pending = list(selectors)
    available = list(gpu_ids)
    statuses: dict[str, dict[str, Any]] = {
        selector: {"selector": selector, "status": "pending", "gpu": None}
        for selector in selectors
    }
    active: dict[str, dict[str, Any]] = {}
    assignments_path = run_dir / "gpu_assignments.json"
    _write_json_atomic(
        assignments_path,
        {"requested_gpu_ids": gpu_ids, "hardware_profile": hardware_profile, "workers": statuses},
    )
    dashboard_at = 0.0

    def fail_run(reason: str) -> None:
        for entry in active.values():
            _terminate_worker(entry["process"])
            entry["handle"].close()
        now = datetime.now(timezone.utc).isoformat()
        for selector in pending:
            statuses[selector].update({"status": "not_started", "finished_at": now})
        _write_json_atomic(
            assignments_path,
            {"requested_gpu_ids": gpu_ids, "hardware_profile": hardware_profile, "workers": statuses},
        )
        (run_dir / "RUN_FAILED").write_text(
            json.dumps(
                {
                    "failed_at_utc": now,
                    "reason": reason,
                    "workers": statuses,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    try:
        while pending or active:
            while pending and available:
                selector = pending.pop(0)
                gpu_id = available.pop(0)
                selector_dir = run_dir / "selectors" / selector
                selector_dir.mkdir(parents=True, exist_ok=True)
                log_handle = (selector_dir / "worker.log").open("w", encoding="utf-8")
                command = [
                    args.python_bin,
                    str(ROOT / "experiments" / "run_qtriad_selector_worker.py"),
                    "--config", str(config_path),
                    "--baseline-dir", str(baseline_dir),
                    "--data-dir", str(data_dir),
                    "--output-dir", str(selector_dir),
                    "--selector", selector,
                    "--device", "cuda" if gpu_id >= 0 else "cpu",
                    "--seed", str(seed),
                    "--log-every-batches", str(args.log_every_batches),
                    "--pair-chunk-size", str(hardware_profile["pair_chunk_size"]),
                    "--activation-checkpointing", str(int(hardware_profile["activation_checkpointing"])),
                ]
                environment = os.environ.copy()
                if gpu_id >= 0:
                    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                started_at = datetime.now(timezone.utc).isoformat()
                active[selector] = {
                    "process": process,
                    "handle": log_handle,
                    "gpu": gpu_id,
                    "started_at": started_at,
                    "started_monotonic": time.monotonic(),
                }
                statuses[selector].update(
                    {"status": "running", "gpu": gpu_id, "pid": process.pid, "started_at": started_at}
                )
                _write_json_atomic(
                    assignments_path,
                    {"requested_gpu_ids": gpu_ids, "hardware_profile": hardware_profile, "workers": statuses},
                )
                print(json.dumps({"event": "selector_started", "selector": selector, "gpu": gpu_id}, sort_keys=True), flush=True)

            for selector, entry in list(active.items()):
                process = entry["process"]
                return_code = process.poll()
                if return_code is None:
                    continue
                entry["handle"].close()
                finished_at = datetime.now(timezone.utc).isoformat()
                duration = round(time.monotonic() - entry["started_monotonic"], 3)
                if return_code != 0:
                    statuses[selector].update(
                        {"status": "failed", "return_code": return_code, "finished_at": finished_at, "duration_seconds": duration}
                    )
                    _write_json_atomic(assignments_path, {"requested_gpu_ids": gpu_ids, "hardware_profile": hardware_profile, "workers": statuses})
                    fail_run(f"selector worker {selector} failed with exit code {return_code}")
                    raise RuntimeError(f"selector worker {selector} failed; inspect {run_dir / 'selectors' / selector / 'worker.log'}")
                metrics_path = run_dir / "selectors" / selector / "metrics.json"
                if not metrics_path.exists():
                    statuses[selector].update(
                        {"status": "failed", "return_code": return_code, "finished_at": finished_at, "duration_seconds": duration}
                    )
                    fail_run(f"selector worker {selector} exited without metrics.json")
                    raise RuntimeError(f"selector worker {selector} produced no metrics.json")
                statuses[selector].update(
                    {"status": "complete", "return_code": 0, "finished_at": finished_at, "duration_seconds": duration}
                )
                available.append(int(entry["gpu"]))
                del active[selector]
                _write_json_atomic(assignments_path, {"requested_gpu_ids": gpu_ids, "hardware_profile": hardware_profile, "workers": statuses})
                print(json.dumps({"event": "selector_complete", "selector": selector, "gpu": entry["gpu"], "duration_seconds": duration}, sort_keys=True), flush=True)

            now = time.monotonic()
            if now - dashboard_at >= 30.0:
                dashboard_at = now
                counts = {state: sum(item["status"] == state for item in statuses.values()) for state in ("pending", "running", "complete", "failed", "not_started")}
                print(json.dumps({"event": "selector_dashboard", **counts, "active_gpus": {selector: item["gpu"] for selector, item in statuses.items() if item["status"] == "running"}}, sort_keys=True), flush=True)
            if active:
                time.sleep(0.5)
    except BaseException as exc:
        if not (run_dir / "RUN_FAILED").exists():
            fail_run(f"selector scheduler aborted: {type(exc).__name__}: {exc}")
        raise
    return statuses


def build_kernel(
    mode: str,
    model: torch.nn.Module,
    seed: int,
    config: dict[str, Any],
    *,
    pair_chunk_size: int | None = None,
    activation_checkpointing: bool | None = None,
) -> QTriadAttentionScoreKernel:
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
        pair_chunk_size=int(pair_chunk_size if pair_chunk_size is not None else kernel_config.get("pair_chunk_size", 256)),
        activation_checkpointing=(
            bool(activation_checkpointing)
            if activation_checkpointing is not None
            else True
        ),
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
    inventory = query_gpu_inventory() if args.gpus and args.gpus.strip().lower() == "auto" else []
    if args.gpus and args.gpus.strip().lower() != "auto" and os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus.strip()
    gpu_ids = resolve_gpu_ids(args.gpus, args.device, inventory)
    if gpu_ids and not inventory:
        inventory = query_gpu_inventory()
    if gpu_ids and os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    if gpu_ids:
        # Re-read after visibility and before creating the run so an explicit
        # GPU request gets the same minimum-capacity guard as --gpus auto.
        inventory = query_gpu_inventory()
        validate_gpu_capacity(gpu_ids, inventory, phase="before baseline")
    profile_request = "auto" if args.gpus and args.gpus.strip().lower() == "auto" and args.hardware_profile == "config" else args.hardware_profile
    hardware_profile = choose_hardware_profile(profile_request, config, gpu_ids, inventory)
    hardware_profile.update(
        {
            "requested_gpu_spec": args.gpus or "default",
            "selected_gpu_ids": gpu_ids,
            "gpu_inventory": inventory,
        }
    )
    device = torch.device("cuda:0") if gpu_ids else choose_device(args.device)
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
        "--device", "cuda" if gpu_ids else str(device),
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
    (run_dir / "baseline_eval.json").write_text(
        json.dumps({"valid": baseline_valid, "test": baseline_test}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    disabled_dir = run_dir / "selectors" / "disabled"
    disabled_dir.mkdir(parents=True, exist_ok=True)
    disabled_row = {
        "selector": "disabled",
        "seed": seed,
        "valid": baseline_valid,
        "test": {**baseline_test, "delta_vs_baseline": metric_delta(baseline_test["metrics"], baseline_test["metrics"])},
        "train": {"history": [], "best_epoch": 0, "runtime_seconds": 0.0},
        "metadata": {"type": "disabled"},
        "trainable_parameters": 0,
        "finite": True,
    }
    (disabled_dir / "metrics.json").write_text(
        json.dumps(disabled_row, indent=2, sort_keys=True), encoding="utf-8"
    )
    # Release the parent's model copy before selector workers load the shared
    # checkpoint. This is important when the first worker uses GPU 0.
    label_count = len(artifacts.label_to_id)
    artifacts.model.to("cpu")
    del valid_loader, test_loader
    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    del artifacts
    del train_records
    if gpu_ids:
        # The baseline and its CUDA allocator must be gone before a worker is
        # allowed to claim the first physical GPU.
        gc.collect()
        torch.cuda.empty_cache()
        inventory = query_gpu_inventory()
        validate_gpu_capacity(gpu_ids, inventory, phase="before selector workers")
    worker_selectors = [selector for selector in selectors if selector != "disabled"]
    worker_statuses = run_selector_workers(
        selectors=worker_selectors,
        gpu_ids=gpu_ids or [-1],
        args=args,
        config_path=config_path,
        baseline_dir=baseline_dir,
        data_dir=data_dir,
        run_dir=run_dir,
        seed=seed,
        hardware_profile=hardware_profile,
    )
    rows: list[dict[str, Any]] = []
    for selector in selectors:
        metrics_path = run_dir / "selectors" / selector / "metrics.json"
        if not metrics_path.exists():
            raise RuntimeError(f"missing selector metrics: {metrics_path}")
        rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
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
        "multi_gpu": {
            "requested_gpu_ids": gpu_ids,
            "selector_parallelism": len(gpu_ids),
            "worker_statuses": worker_statuses,
        },
        "hardware_profile": hardware_profile,
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
            "visible_cuda_devices": gpu_ids,
            "pair_chunk_size": int(hardware_profile["pair_chunk_size"]),
            "activation_checkpointing": bool(hardware_profile["activation_checkpointing"]),
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
        f"- hardware profile: `{hardware_profile['name']}` (pair_chunk_size={hardware_profile['pair_chunk_size']}, activation_checkpointing={str(hardware_profile['activation_checkpointing']).lower()})",
        f"- selected physical GPUs: `{gpu_ids}`",
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
