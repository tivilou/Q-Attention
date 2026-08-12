#!/usr/bin/env python3
"""Run the train/valid-only proportional gate for the Q-NESS selector."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import threading
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from q_attention.tasks.relation import (  # noqa: E402
    load_relation_jsonl,
    sample_relation_records,
    sample_relation_records_proportional,
    write_relation_jsonl,
)


CONTROL_SELECTORS = (
    "qness_commuting",
    "qness_separable",
    "qness_phase_scrambled",
    "qness_dephased",
)
_CONSOLE_BROKEN = threading.Event()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_path", default="data/relation/retacred/train.jsonl")
    parser.add_argument("--valid_path", default="data/relation/retacred/valid.jsonl")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--baseline_train_limit", type=int, default=8192)
    parser.add_argument("--train_limit", type=int, default=8192)
    parser.add_argument("--valid_limit", type=int, default=2048)
    parser.add_argument("--baseline_epochs", type=int, default=6)
    parser.add_argument("--core_epochs", type=int, default=4)
    parser.add_argument("--selector_epochs", type=int, default=5)
    parser.add_argument("--baseline_batch_size", type=int, default=128)
    parser.add_argument("--plugin_batch_size", type=int, default=64)
    parser.add_argument("--diagnostic_batches", type=int, default=16)
    parser.add_argument("--quantum_diagnostic_limit", type=int, default=64)
    parser.add_argument("--random_repeats", type=int, default=1)
    parser.add_argument("--log_every_batches", type=int, default=25)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--gpus",
        default="0",
        help="Physical GPU indexes separated by commas, or auto to query nvidia-smi.",
    )
    parser.add_argument("--min_baseline_macro_f1", type=float, default=0.10)
    parser.add_argument("--min_loss_gain", type=float, default=1e-4)
    parser.add_argument("--max_f1_drop", type=float, default=0.005)
    parser.add_argument(
        "--run_controls",
        choices=("always", "never"),
        default="always",
        help="Run the four Q-NESS mechanism controls or only the primary pair.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--summarize_only",
        action="store_true",
        help="Regenerate summaries from an existing completed stage directory.",
    )
    parser.add_argument("--fail_on_gate", action="store_true")
    return parser.parse_args()


def _resolve_gpus(spec: str) -> list[int]:
    value = str(spec).strip()
    if not value:
        raise ValueError("--gpus must contain at least one GPU index")
    if value.lower() == "auto":
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise RuntimeError("--gpus auto requires nvidia-smi") from exc
        if result.returncode != 0:
            raise RuntimeError(
                "--gpus auto requires nvidia-smi: "
                + (result.stderr.strip() or "command failed")
            )
        tokens = result.stdout.splitlines()
    else:
        tokens = value.split(",")
    gpus: list[int] = []
    for token in tokens:
        token = token.strip()
        if not token or not token.isdigit():
            raise ValueError(f"GPU indexes must be non-negative integers: {token!r}")
        gpu_id = int(token)
        if gpu_id in gpus:
            raise ValueError(f"GPU index appears more than once: {gpu_id}")
        gpus.append(gpu_id)
    if not gpus:
        raise ValueError("--gpus resolved to no GPU")
    return gpus


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _default_output_dir(seed: int) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "runs" / f"retacred_qness_proportional_{stamp}_seed{seed}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(path: Path, fields: dict[str, Any]) -> None:
    lines = []
    for key, value in fields.items():
        text = str(value).replace("\n", " ")
        lines.append(f"{key}={text}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _selector_names(run_controls: str) -> list[str]:
    names = ["qness", "qness_classical"]
    if run_controls == "always":
        names.extend(CONTROL_SELECTORS)
    return names


def _gpu_assignment_manifest(
    requested: str, gpus: list[int], selector_names: list[str]
) -> dict[str, Any]:
    stage_names = ["baseline", "core_quantum"] + [
        f"selector_{selector}" for selector in selector_names
    ]
    stages = {
        name: {
            "status": "pending",
            "gpu_id": None,
            "started_at": None,
            "completed_at": None,
            "duration_seconds": None,
            "exit_code": None,
        }
        for name in stage_names
    }
    stages["baseline"]["gpu_id"] = gpus[0]
    stages["core_quantum"]["gpu_id"] = gpus[0]
    return {
        "schema_version": "retacred-qness-gpu-assignments.v1",
        "scheduler": "dynamic_selector_workers",
        "ddp": False,
        "requested_gpus": requested,
        "resolved_gpus": gpus,
        "selector_stages": [f"selector_{selector}" for selector in selector_names],
        "stages": stages,
    }


def _assignment_update(
    payload: dict[str, Any],
    path: Path,
    lock: threading.Lock,
    stage: str,
    **fields: Any,
) -> None:
    with lock:
        payload["stages"][stage].update(fields)
        _write_json(path, payload)


def _console_print(value: str, lock: threading.Lock | None) -> None:
    if _CONSOLE_BROKEN.is_set():
        return
    try:
        if lock is None:
            print(value, end="", flush=True)
            return
        with lock:
            print(value, end="", flush=True)
    except BrokenPipeError:
        _CONSOLE_BROKEN.set()


def _run_stage(
    name: str,
    command: list[str],
    output_dir: Path,
    heartbeat: Path | None,
    gpu_id: int,
    assignments: dict[str, Any] | None = None,
    assignments_path: Path | None = None,
    assignments_lock: threading.Lock | None = None,
    console_lock: threading.Lock | None = None,
    required_paths: tuple[Path, ...] = (),
) -> None:
    log_path = output_dir / "logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    status_dir = output_dir / "status"
    status_path = status_dir / f"{name}.env"
    stage_heartbeat = status_dir / f"{name}.heartbeat"
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    status_dir.mkdir(parents=True, exist_ok=True)
    stage_heartbeat.touch()
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = environment.get("PYTHONHASHSEED", "0")
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    environment["Q_ATTENTION_HEARTBEAT_FILE"] = str(stage_heartbeat)
    if heartbeat is not None:
        heartbeat.parent.mkdir(parents=True, exist_ok=True)
        heartbeat.touch()
    _write_status(
        status_path,
        {
            "STATUS": "running",
            "GPU_ID": gpu_id,
            "STARTED_AT": started_at,
            "HEARTBEAT_FILE": stage_heartbeat,
            "LOG_FILE": log_path,
        },
    )
    if assignments is not None and assignments_path is not None and assignments_lock is not None:
        _assignment_update(
            assignments,
            assignments_path,
            assignments_lock,
            name,
            status="running",
            gpu_id=gpu_id,
            started_at=started_at,
        )
    _write_json(
        output_dir / "logs" / f"{name}.command.json",
        {
            "command": command,
            "cwd": str(ROOT),
            "gpu_id": gpu_id,
            "cuda_visible_devices": str(gpu_id),
        },
    )
    _console_print(
        json.dumps(
            {
                "event": "stage_started",
                "stage": name,
                "gpu_id": gpu_id,
                "log": str(log_path),
            }
        )
        + "\n",
        console_lock,
    )
    status = None
    process_error: BaseException | None = None
    with log_path.open("w", encoding="utf-8") as handle:
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            _write_status(
                status_path,
                {
                    "STATUS": "running",
                    "GPU_ID": gpu_id,
                    "PID": process.pid,
                    "STARTED_AT": started_at,
                    "HEARTBEAT_FILE": stage_heartbeat,
                    "LOG_FILE": log_path,
                },
            )
            assert process.stdout is not None
            for line in process.stdout:
                _console_print(f"[{name}][gpu={gpu_id}] {line}", console_lock)
                handle.write(line)
                handle.flush()
                if heartbeat is not None:
                    heartbeat.touch()
            status = process.wait()
        except BaseException as exc:
            process_error = exc
    duration = round(max(time.monotonic() - started_monotonic, 0.0), 3)
    completed_at = _utc_now()
    if process_error is not None:
        error_text = str(process_error)
        _write_status(
            status_path,
            {
                "STATUS": "failed",
                "GPU_ID": gpu_id,
                "EXIT_CODE": -1,
                "STARTED_AT": started_at,
                "FAILED_AT": completed_at,
                "DURATION_SECONDS": duration,
                "HEARTBEAT_FILE": stage_heartbeat,
                "LOG_FILE": log_path,
                "ERROR": error_text,
            },
        )
        if assignments is not None and assignments_path is not None and assignments_lock is not None:
            _assignment_update(
                assignments,
                assignments_path,
                assignments_lock,
                name,
                status="failed",
                completed_at=completed_at,
                duration_seconds=duration,
                exit_code=-1,
                error=error_text,
            )
        raise process_error
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    output_error = ""
    if status == 0 and missing_paths:
        status = 1
        output_error = "missing expected outputs: " + ", ".join(missing_paths)
    if status != 0:
        tail = log_path.read_text(encoding="utf-8").splitlines()[-40:]
        error_text = output_error or (f"exit code {status}: " + "\n".join(tail))
        _write_status(
            status_path,
            {
                "STATUS": "failed",
                "GPU_ID": gpu_id,
                "EXIT_CODE": status,
                "STARTED_AT": started_at,
                "FAILED_AT": completed_at,
                "DURATION_SECONDS": duration,
                "HEARTBEAT_FILE": stage_heartbeat,
                "LOG_FILE": log_path,
                "ERROR": error_text,
            },
        )
        if assignments is not None and assignments_path is not None and assignments_lock is not None:
            _assignment_update(
                assignments,
                assignments_path,
                assignments_lock,
                name,
                status="failed",
                completed_at=completed_at,
                duration_seconds=duration,
                exit_code=status,
                error=error_text,
            )
        raise RuntimeError(
            f"stage {name} failed with exit code {status}:\n{error_text}"
        )
    _write_status(
        status_path,
        {
            "STATUS": "complete",
            "GPU_ID": gpu_id,
            "EXIT_CODE": 0,
            "STARTED_AT": started_at,
            "COMPLETED_AT": completed_at,
            "DURATION_SECONDS": duration,
            "HEARTBEAT_FILE": stage_heartbeat,
            "LOG_FILE": log_path,
        },
    )
    if assignments is not None and assignments_path is not None and assignments_lock is not None:
        _assignment_update(
            assignments,
            assignments_path,
            assignments_lock,
            name,
            status="complete",
            completed_at=completed_at,
            duration_seconds=duration,
            exit_code=0,
        )
    _console_print(
        json.dumps(
            {
                "event": "stage_completed",
                "stage": name,
                "gpu_id": gpu_id,
                "duration_seconds": duration,
            }
        )
        + "\n",
        console_lock,
    )


def _run_selector_workers(
    selector_names: list[str],
    gpu_ids: list[int],
    run_selector: Callable[[str, int], None],
) -> list[dict[str, Any]]:
    """Run selector jobs with one queue-consuming worker per physical GPU."""
    jobs: Queue[str] = Queue()
    for selector in selector_names:
        jobs.put(selector)
    outcomes: list[dict[str, Any]] = []
    outcomes_lock = threading.Lock()

    def worker(gpu_id: int) -> None:
        while True:
            try:
                selector = jobs.get_nowait()
            except Empty:
                return
            stage = f"selector_{selector}"
            try:
                run_selector(selector, gpu_id)
            except Exception as exc:
                outcome = {
                    "stage": stage,
                    "selector": selector,
                    "gpu_id": gpu_id,
                    "status": "failed",
                    "error": str(exc),
                }
            else:
                outcome = {
                    "stage": stage,
                    "selector": selector,
                    "gpu_id": gpu_id,
                    "status": "complete",
                }
            finally:
                jobs.task_done()
            with outcomes_lock:
                outcomes.append(outcome)

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        list(executor.map(worker, gpu_ids))
    return outcomes


def _subset_manifest(records: list[Any], source_path: Path, sampler: str) -> dict[str, Any]:
    return {
        "source_sha256": _sha256(source_path),
        "source_record_count": sum(1 for _ in source_path.open(encoding="utf-8")),
        "selected_record_count": len(records),
        "selected_label_counts": dict(sorted(Counter(r.label for r in records).items())),
        "sampler": sampler,
    }


def _prepare_subsets(
    args: argparse.Namespace, output_dir: Path
) -> tuple[dict[str, Path], dict[str, Any]]:
    train_path = _resolve(args.train_path)
    valid_path = _resolve(args.valid_path)
    train_records = load_relation_jsonl(train_path)
    valid_records = load_relation_jsonl(valid_path)
    selected = {
        "baseline_train": sample_relation_records(
            train_records, args.baseline_train_limit, seed=args.seed, stratified=True
        ),
        "train": sample_relation_records(
            train_records, args.train_limit, seed=args.seed + 1, stratified=True
        ),
        "valid": sample_relation_records_proportional(
            valid_records, args.valid_limit, seed=args.seed + 101
        ),
    }
    evaluation_labels = {record.label for record in selected["valid"]}
    for split in ("baseline_train", "train"):
        missing = evaluation_labels - {record.label for record in selected[split]}
        if missing:
            raise ValueError(f"{split} does not cover validation labels: {sorted(missing)}")
    paths: dict[str, Path] = {}
    manifest: dict[str, Any] = {
        "seed": args.seed,
        "source": {
            "train": str(train_path),
            "valid": str(valid_path),
            "train_sha256": _sha256(train_path),
            "valid_sha256": _sha256(valid_path),
            "train_record_count": len(train_records),
            "valid_record_count": len(valid_records),
        },
        "splits": {},
    }
    for split, records in selected.items():
        path = output_dir / f"{split}.jsonl"
        write_relation_jsonl(records, path)
        paths[split] = path
        source = train_path if split != "valid" else valid_path
        sampler = (
            "balanced_round_robin"
            if split != "valid"
            else "proportional_largest_remainder"
        )
        manifest["splits"][split] = _subset_manifest(records, source, sampler)
    _write_json(output_dir / "subset_manifest.json", manifest)
    return paths, manifest


def _baseline_command(
    args: argparse.Namespace, paths: dict[str, Path], output_dir: Path
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "experiments" / "train_relation_baseline.py"),
        "--train_path",
        str(paths["baseline_train"]),
        "--valid_path",
        str(paths["valid"]),
        "--output_dir",
        str(output_dir),
        "--epochs",
        str(args.baseline_epochs),
        "--batch_size",
        str(args.baseline_batch_size),
        "--lr",
        "0.0005",
        "--dim",
        "128",
        "--num_layers",
        "4",
        "--num_heads",
        "8",
        "--ff_dim",
        "256",
        "--dropout",
        "0.1",
        "--max_length",
        "128",
        "--selection_metric",
        "valid_loss",
        "--log_every_batches",
        str(args.log_every_batches),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]


def _core_command(
    args: argparse.Namespace, paths: dict[str, Path], output_dir: Path
) -> list[str]:
    run_dir = output_dir.parent.parent
    return [
        sys.executable,
        str(ROOT / "experiments" / "train_relation_attention_score_kernel.py"),
        "--model_dir",
        str(run_dir / "baseline"),
        "--train_path",
        str(paths["train"]),
        "--valid_path",
        str(paths["valid"]),
        "--output_dir",
        str(output_dir),
        "--kernel_type",
        "quantum",
        "--num_qubits",
        "4",
        "--depth",
        "2",
        "--angle_scale",
        "1.0",
        "--score_readout",
        "observable",
        "--input_encoding",
        "factorized_shared",
        "--query_scope",
        "all",
        "--epochs",
        str(args.core_epochs),
        "--batch_size",
        str(args.plugin_batch_size),
        "--lr",
        "0.001",
        "--selection_metric",
        "valid_loss",
        "--diagnostic_batches",
        str(args.diagnostic_batches),
        "--log_every_batches",
        str(args.log_every_batches),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]


def _selector_command(
    args: argparse.Namespace,
    selector: str,
    paths: dict[str, Path],
    output_dir: Path,
) -> list[str]:
    run_dir = output_dir.parent.parent
    return [
        sys.executable,
        str(ROOT / "experiments" / "train_relation_counterfactual_evidence.py"),
        "--model_dir",
        str(run_dir / "baseline"),
        "--core_checkpoint",
        str(run_dir / "core" / "quantum" / "attention_score_kernel.pt"),
        "--train_path",
        str(paths["train"]),
        "--valid_path",
        str(paths["valid"]),
        "--output_dir",
        str(output_dir),
        "--evidence_type",
        selector,
        "--num_qubits",
        "4",
        "--depth",
        "2",
        "--angle_scale",
        "1.0",
        "--evidence_gate_calibration",
        "context_budget",
        "--evidence_view_score_mode",
        "positive",
        "--evidence_task_readout",
        "dual",
        "--evidence_readout",
        "connected_relation_token",
        "--evidence_correlation_mode",
        "connected",
        "--evidence_weight_mode",
        "signed_centered_l1",
        "--evidence_measurement_mode",
        "fixed",
        "--intervention_mode",
        "direct_bias",
        "--direct_bias_mode",
        "centered",
        "--evidence_budget",
        "0.35",
        "--random_repeats",
        str(args.random_repeats),
        "--diagnostic_batches",
        str(args.diagnostic_batches),
        "--quantum_diagnostic_limit",
        str(args.quantum_diagnostic_limit),
        "--epochs",
        str(args.selector_epochs),
        "--batch_size",
        str(args.plugin_batch_size),
        "--lr",
        "0.01",
        "--log_every_batches",
        str(args.log_every_batches),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _below(value: float | None, threshold: float) -> bool:
    return value is not None and value < threshold


def _metrics(payload: dict[str, Any], prefer_task: bool = False) -> dict[str, float]:
    candidate = payload.get("best_task_valid") if prefer_task else None
    if not isinstance(candidate, dict):
        candidate = payload.get("best_valid")
    if not isinstance(candidate, dict):
        raise ValueError("metrics payload has no usable validation metrics")
    result = {
        name: float(candidate[name])
        for name in ("loss", "macro_f1", "correct_label_margin")
    }
    if not all(_finite_number(value) for value in result.values()):
        raise ValueError("validation metrics contain a non-finite value")
    return result


def _diagnostic_mean(payload: dict[str, Any], name: str) -> float | None:
    diagnostics = payload.get("best_selectivity")
    if not isinstance(diagnostics, dict):
        return None
    metrics = diagnostics.get("metrics")
    if not isinstance(metrics, dict) or not isinstance(metrics.get(name), dict):
        return None
    value = metrics[name].get("mean")
    return float(value) if _finite_number(value) else None


def _stage_summary(path: Path, *, prefer_task: bool = False) -> dict[str, Any]:
    metrics = _read_json(path / "metrics.json")
    diagnostics_path = path / "diagnostics.json"
    diagnostics = _read_json(diagnostics_path) if diagnostics_path.is_file() else {}
    return {
        "metrics_path": str(path / "metrics.json"),
        "diagnostics_path": str(diagnostics_path) if diagnostics_path.is_file() else None,
        "validation_metrics": _metrics(metrics, prefer_task=prefer_task),
        "best_epoch": metrics.get("best_epoch"),
        "best_task_epoch": metrics.get("best_task_epoch"),
        "selectivity_pass": bool(
            isinstance(metrics.get("best_selectivity"), dict)
            and metrics["best_selectivity"].get("selectivity_pass", False)
        ),
        "diagnostic_means": {
            name: _diagnostic_mean(metrics, name)
            for name in (
                "complement_error",
                "off_diagonal_density_norm",
                "mutual_information",
                "observable_commutator_norm",
                "keep_advantage",
                "drop_advantage",
            )
        },
        "health": metrics.get("health"),
        "gradients": metrics.get("gradients"),
        "diagnostics_schema": diagnostics.get("schema_version"),
    }


def proportional_gate_decision(
    stages: dict[str, dict[str, Any]],
    *,
    min_loss_gain: float = 1e-4,
    max_f1_drop: float = 0.005,
    min_baseline_macro_f1: float = 0.10,
    controls_requested: bool = True,
) -> dict[str, Any]:
    baseline = stages["baseline"]["validation_metrics"]
    core = stages["core_quantum"]["validation_metrics"]
    qness = stages["selector_qness"]["validation_metrics"]
    classical = stages["selector_qness_classical"]["validation_metrics"]
    qness_task = stages["selector_qness"].get("best_task_epoch") is not None
    qness_resource = stages["selector_qness"]["diagnostic_means"]
    resource_checks = {
        "qness_commutator_nonzero": (
            qness_resource["observable_commutator_norm"] or 0.0
        )
        > 1e-4,
        "qness_off_diagonal_nonzero": (
            qness_resource["off_diagonal_density_norm"] or 0.0
        )
        > 1e-6,
        "qness_mutual_information_nonzero": (
            qness_resource["mutual_information"] or 0.0
        )
        > 1e-6,
    }
    if controls_requested:
        resource_checks.update(
            {
                "commuting_commutator_removed": (
                    _below(
                        stages["selector_qness_commuting"]["diagnostic_means"][
                            "observable_commutator_norm"
                        ],
                        1e-7,
                    )
                ),
                "separable_mutual_information_removed": _below(
                    stages["selector_qness_separable"]["diagnostic_means"][
                        "mutual_information"
                    ],
                    1e-6,
                ),
                "dephased_off_diagonal_removed": _below(
                    stages["selector_qness_dephased"]["diagnostic_means"][
                        "off_diagonal_density_norm"
                    ],
                    1e-7,
                ),
            }
        )
    task_checks = {
        "baseline_usable": baseline["macro_f1"] >= min_baseline_macro_f1,
        "qness_has_task_checkpoint": qness_task,
        "qness_improves_over_quantum_core": core["loss"] - qness["loss"] >= min_loss_gain,
        "qness_improves_over_classical_control": classical["loss"] - qness["loss"] >= min_loss_gain,
        "qness_f1_guardrail": qness["macro_f1"]
        >= min(core["macro_f1"], classical["macro_f1"]) - max_f1_drop,
    }
    technical = all(
        _finite_number(value)
        for stage in stages.values()
        for value in stage["validation_metrics"].values()
    )
    controls_pass = all(resource_checks.values()) if controls_requested else True
    task_pass = all(task_checks.values())
    return {
        "gate_pass": bool(technical and task_pass and controls_pass),
        "technical_finite": technical,
        "task_checks": task_checks,
        "resource_checks": resource_checks,
        "controls_requested": controls_requested,
        "gains": {
            "qness_over_quantum_core_loss": core["loss"] - qness["loss"],
            "qness_over_classical_control_loss": classical["loss"] - qness["loss"],
            "qness_over_quantum_core_macro_f1": qness["macro_f1"] - core["macro_f1"],
            "qness_over_classical_control_macro_f1": qness["macro_f1"] - classical["macro_f1"],
        },
        "thresholds": {
            "min_loss_gain": min_loss_gain,
            "max_f1_drop": max_f1_drop,
            "min_baseline_macro_f1": min_baseline_macro_f1,
        },
    }


def _render_summary(summary: dict[str, Any]) -> str:
    decision = summary["decision"]
    lines = [
        "# Re-TACRED Q-NESS Proportional Gate",
        "",
        f"- revision: `{summary['revision']}`",
        f"- seed: `{summary['config']['seed']}`",
        f"- overall gate: `{str(decision['gate_pass']).lower()}`",
        "- evaluation: validation only; blind test was not read.",
        "",
        "| Stage | Loss | Macro-F1 | Correct-label margin | Selectivity |",
        "| --- | ---: | ---: | ---: | :---: |",
    ]
    for name, stage in summary["stages"].items():
        metrics = stage["validation_metrics"]
        lines.append(
            f"| {name} | {metrics['loss']:.6f} | {metrics['macro_f1']:.6f} | "
            f"{metrics['correct_label_margin']:.6f} | "
            f"{str(stage.get('selectivity_pass', False)).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- task checks: `{json.dumps(decision['task_checks'], sort_keys=True)}`",
            f"- resource checks: `{json.dumps(decision['resource_checks'], sort_keys=True)}`",
            f"- gains: `{json.dumps(decision['gains'], sort_keys=True)}`",
            "",
            "This is a screening gate, not evidence of statistical significance or quantum advantage.",
        ]
    )
    return "\n".join(lines) + "\n"


def summarize_existing_run(output_dir: Path) -> dict[str, Any]:
    manifest = _read_json(output_dir / "run_manifest.json")
    config = manifest["config"]
    controls_requested = config.get("run_controls", "always") == "always"
    stage_paths = {
        "baseline": output_dir / "baseline",
        "core_quantum": output_dir / "core" / "quantum",
        "selector_qness": output_dir / "selector" / "qness",
        "selector_qness_classical": output_dir / "selector" / "qness_classical",
    }
    if controls_requested:
        stage_paths.update(
            {
                f"selector_{selector}": output_dir / "selector" / selector
                for selector in CONTROL_SELECTORS
            }
        )
    stage_summaries = {
        "baseline": _stage_summary(stage_paths["baseline"]),
        "core_quantum": _stage_summary(stage_paths["core_quantum"]),
        "selector_qness": _stage_summary(
            stage_paths["selector_qness"], prefer_task=True
        ),
        "selector_qness_classical": _stage_summary(
            stage_paths["selector_qness_classical"]
        ),
    }
    for selector in CONTROL_SELECTORS:
        name = f"selector_{selector}"
        if name in stage_paths:
            stage_summaries[name] = _stage_summary(stage_paths[name])
    decision = proportional_gate_decision(
        stage_summaries,
        min_loss_gain=float(config.get("min_loss_gain", 1e-4)),
        max_f1_drop=float(config.get("max_f1_drop", 0.005)),
        min_baseline_macro_f1=float(config.get("min_baseline_macro_f1", 0.10)),
        controls_requested=controls_requested,
    )
    summary = {
        "schema_version": "retacred-qness-proportional-gate.v1",
        "revision": config.get("revision", _git_revision()),
        "config": config,
        "subset_manifest": manifest["subset_manifest"],
        "gpu_assignments": (
            _read_json(output_dir / "gpu_assignments.json")
            if (output_dir / "gpu_assignments.json").is_file()
            else None
        ),
        "stages": stage_summaries,
        "decision": decision,
    }
    _write_json(output_dir / "run_summary.json", summary)
    (output_dir / "run_summary.md").write_text(
        _render_summary(summary), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    positive = (
        args.baseline_train_limit,
        args.train_limit,
        args.valid_limit,
        args.baseline_epochs,
        args.core_epochs,
        args.selector_epochs,
        args.baseline_batch_size,
        args.plugin_batch_size,
        args.diagnostic_batches,
        args.quantum_diagnostic_limit,
        args.random_repeats,
        args.log_every_batches,
    )
    if min(positive) <= 0:
        raise ValueError("limits, epochs, batch sizes, diagnostics, and repeats must be positive")
    if args.min_loss_gain < 0.0 or args.max_f1_drop < 0.0:
        raise ValueError("gate thresholds must be non-negative")
    if args.dry_run and args.summarize_only:
        raise ValueError("dry_run and summarize_only cannot be combined")
    output_dir = _resolve(args.output_dir) if args.output_dir else _default_output_dir(args.seed)
    if args.summarize_only:
        if not output_dir.is_dir():
            raise FileNotFoundError(f"existing run directory not found: {output_dir}")
        summary = summarize_existing_run(output_dir)
        print(
            json.dumps(
                {
                    "event": "summary_completed",
                    "output_dir": str(output_dir),
                    "gate_pass": summary["decision"]["gate_pass"],
                }
            ),
            flush=True,
        )
        if args.fail_on_gate and not summary["decision"]["gate_pass"]:
            raise SystemExit(2)
        return
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse proportional output directory: {output_dir}")
    resolved_gpus = _resolve_gpus(args.gpus)
    selector_names = _selector_names(args.run_controls)
    if args.dry_run:
        paths = {
            "baseline_train": output_dir / "private_subsets" / "baseline_train.jsonl",
            "train": output_dir / "private_subsets" / "train.jsonl",
            "valid": output_dir / "private_subsets" / "valid.jsonl",
        }
        commands = {
            "baseline": _baseline_command(args, paths, output_dir / "baseline"),
            "core_quantum": _core_command(args, paths, output_dir / "core" / "quantum"),
        }
        for selector in selector_names:
            commands[f"selector_{selector}"] = _selector_command(
                args, selector, paths, output_dir / "selector" / selector
            )
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "output_dir": str(output_dir),
                    "requested_gpus": args.gpus,
                    "resolved_gpus": resolved_gpus,
                    "baseline_gpu": resolved_gpus[0],
                    "core_quantum_gpu": resolved_gpus[0],
                    "selector_scheduler": "dynamic_selector_workers",
                    "selector_stages": [
                        f"selector_{selector}" for selector in selector_names
                    ],
                    "commands": commands,
                },
                indent=2,
            )
        )
        return

    output_dir.mkdir(parents=True, exist_ok=False)
    heartbeat_value = os.environ.get("Q_ATTENTION_HEARTBEAT_FILE")
    heartbeat = Path(heartbeat_value) if heartbeat_value else None
    assignments_path = output_dir / "gpu_assignments.json"
    assignments_lock = threading.Lock()
    console_lock = threading.Lock()
    assignments = _gpu_assignment_manifest(args.gpus, resolved_gpus, selector_names)
    current_stage = "subset_preparation"
    try:
        paths, subset_manifest = _prepare_subsets(args, output_dir / "private_subsets")
        config = {
            **vars(args),
            "output_dir": str(output_dir),
            "revision": _git_revision(),
            "test_used": False,
            "resolved_gpus": resolved_gpus,
            "baseline_gpu": resolved_gpus[0],
            "core_quantum_gpu": resolved_gpus[0],
            "selector_scheduler": "dynamic_selector_workers",
            "ddp": False,
        }
        _write_json(output_dir / "run_config.json", config)
        _write_json(assignments_path, assignments)
        _write_json(
            output_dir / "run_manifest.json",
            {
                "config": config,
                "subset_manifest": subset_manifest,
                "gpu_assignments_file": "gpu_assignments.json",
            },
        )

        current_stage = "baseline"
        _run_stage(
            "baseline",
            _baseline_command(args, paths, output_dir / "baseline"),
            output_dir,
            heartbeat,
            resolved_gpus[0],
            assignments,
            assignments_path,
            assignments_lock,
            console_lock,
            required_paths=(
                output_dir / "baseline" / "metrics.json",
                output_dir / "baseline" / "model.pt",
            ),
        )
        current_stage = "core_quantum"
        _run_stage(
            "core_quantum",
            _core_command(args, paths, output_dir / "core" / "quantum"),
            output_dir,
            heartbeat,
            resolved_gpus[0],
            assignments,
            assignments_path,
            assignments_lock,
            console_lock,
            required_paths=(
                output_dir / "core" / "quantum" / "metrics.json",
                output_dir / "core" / "quantum" / "diagnostics.json",
                output_dir / "core" / "quantum" / "attention_score_kernel.pt",
            ),
        )

        current_stage = "selector_workers"

        def run_selector(selector: str, gpu_id: int) -> None:
            stage_name = f"selector_{selector}"
            _run_stage(
                stage_name,
                _selector_command(args, selector, paths, output_dir / "selector" / selector),
                output_dir,
                heartbeat,
                gpu_id,
                assignments,
                assignments_path,
                assignments_lock,
                console_lock,
                required_paths=(
                    output_dir / "selector" / selector / "metrics.json",
                    output_dir / "selector" / selector / "diagnostics.json",
                ),
            )

        selector_outcomes = _run_selector_workers(
            selector_names, resolved_gpus, run_selector
        )
        failed_selectors = [
            outcome for outcome in selector_outcomes if outcome["status"] == "failed"
        ]
        if failed_selectors:
            details = "; ".join(
                f"{item['stage']} gpu={item['gpu_id']}: {item['error']}"
                for item in failed_selectors
            )
            raise RuntimeError(f"selector worker failures: {details}")

        summary = summarize_existing_run(output_dir)
        decision = summary["decision"]
        (output_dir / "RUN_COMPLETE").write_text(
            datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "event": "gate_completed",
                    "output_dir": str(output_dir),
                    "gate_pass": decision["gate_pass"],
                }
            ),
            flush=True,
        )
        if args.fail_on_gate and not decision["gate_pass"]:
            (output_dir / "GATE_FAILED").write_text(
                "The run completed but the proportional gate did not pass.\n",
                encoding="utf-8",
            )
            raise SystemExit(2)
    except SystemExit:
        raise
    except BaseException as exc:
        (output_dir / "RUN_FAILED").write_text(
            f"FAILED_AT={datetime.now(timezone.utc).isoformat()}\n"
            f"STAGE={current_stage}\nERROR={exc}\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
