
from __future__ import annotations

"""Run the Q-CEASC L2 replication as a two-phase task graph.

The scheduler separates baseline(seed) tasks from selector(seed, name) tasks.
Baselines form a barrier, then selectors enter one global ready queue. A
selected physical GPU runs at most one heavy child by default.
"""

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

DEFAULT_SEEDS = (13, 29, 53)
MIN_FREE_MIB = 8 * 1024
SINGLE_RUNNER = EXPERIMENTS / "run_retacred_qceasc_formal_single_seed.py"
SELECTOR_WORKER = EXPERIMENTS / "run_qceasc_selector_worker.py"
CONFIG_PATH = ROOT / "configs" / "retacred_qceasc_formal_single_seed.json"
GROUP_ROOT = ROOT / "runs" / "retacred_qceasc_formal_multi_seed"
SELECTORS = ("disabled", "q_ceasc", "classical_ceasc")
SELECTOR_TASKS = ("q_ceasc", "classical_ceasc")
PROTOCOL = "qceasc"
BASELINE_ARTIFACTS = ("model.pt", "vocab.json", "labels.json", "metrics.json")
PAUSED_EXIT_CODE = 75
CUDA_OOM_EXIT_CODE = 86
MEMORY_PRESSURE_EXIT_CODE = 87
SAFE_PAUSE_TIMEOUT_SECONDS = 15 * 60
MANIFEST_SCHEMA = "q-attention.q-ceasc.formal-task-graph.v2"
STATUS_SCHEMA = "q-attention.q-ceasc.formal-task-status.v2"
RUN_SCHEMA = "q-attention.q-ceasc.formal-task-graph-run.v1"
ADAPTIVE_STATE_SCHEMA = "q-attention.q-ceasc.task-adaptive-memory.v1"
OOM_MARKERS = (
    "cuda out of memory",
    "cuda error: out of memory",
    "cuda_error_out_of_memory",
    "cublas_status_alloc_failed",
    "cudaerrormemoryallocation",
)

COUNTERFACTUAL_CONFIG_SCHEMA = "q-attention.q-ceasc-counterfactual-formal-single-seed.v1"
COUNTERFACTUAL_GROUP_ROOT = ROOT / "runs" / "retacred_qceasc_counterfactual_formal_multi_seed"
COUNTERFACTUAL_MANIFEST_SCHEMA = "q-attention.q-ceasc-counterfactual.formal-task-graph.v1"
COUNTERFACTUAL_STATUS_SCHEMA = "q-attention.q-ceasc-counterfactual.formal-task-status.v1"
COUNTERFACTUAL_RUN_SCHEMA = "q-attention.q-ceasc-counterfactual.formal-task-graph-run.v1"
COUNTERFACTUAL_ADAPTIVE_STATE_SCHEMA = "q-attention.q-ceasc-counterfactual.task-adaptive-memory.v1"
COUNTERFACTUAL_SOURCE_FILES = {
    "attention_adapter": "src/q_attention/adapters/attention_scores.py",
    "baseline_trainer": "experiments/train_relation_baseline.py",
    "batch_resume": "src/q_attention/experiments/batch_resume.py",
    "kernel_trainer": "experiments/run_q_causal_value_evidence_relation_transfer.py",
    "q_ceasc": "src/q_attention/plugins/q_ceasc_score.py",
    "q_ceasc_core": "src/q_attention/plugins/q_ceasc.py",
    "q_ceasc_counterfactual": "src/q_attention/plugins/q_ceasc_counterfactual.py",
    "relation_model": "src/q_attention/models/relation_transformer.py",
    "relation_steering": "src/q_attention/experiments/relation_steering.py",
    "relation_task": "src/q_attention/tasks/relation.py",
    "runner": "experiments/run_retacred_qceasc_formal_single_seed.py",
    "worker": "experiments/run_qceasc_selector_worker.py",
}


def configure_protocol(config_path: Path, config: dict[str, Any]) -> None:
    """Select the task-graph contract from the frozen experiment config."""
    global CONFIG_PATH, GROUP_ROOT, SELECTORS, SELECTOR_TASKS, PROTOCOL
    global MANIFEST_SCHEMA, STATUS_SCHEMA, RUN_SCHEMA, ADAPTIVE_STATE_SCHEMA
    CONFIG_PATH = config_path.resolve()
    if config.get("counterfactual") is True:
        PROTOCOL = "qceasc_counterfactual"
        GROUP_ROOT = COUNTERFACTUAL_GROUP_ROOT
        SELECTORS = tuple(str(item) for item in config.get("selectors", ()))
        SELECTOR_TASKS = tuple(item for item in SELECTORS if item != "disabled")
        MANIFEST_SCHEMA = COUNTERFACTUAL_MANIFEST_SCHEMA
        STATUS_SCHEMA = COUNTERFACTUAL_STATUS_SCHEMA
        RUN_SCHEMA = COUNTERFACTUAL_RUN_SCHEMA
        ADAPTIVE_STATE_SCHEMA = COUNTERFACTUAL_ADAPTIVE_STATE_SCHEMA
    else:
        PROTOCOL = "qceasc"
        GROUP_ROOT = ROOT / "runs" / "retacred_qceasc_formal_multi_seed"
        SELECTORS = ("disabled", "q_ceasc", "classical_ceasc")
        SELECTOR_TASKS = ("q_ceasc", "classical_ceasc")
        MANIFEST_SCHEMA = "q-attention.q-ceasc.formal-task-graph.v2"
        STATUS_SCHEMA = "q-attention.q-ceasc.formal-task-status.v2"
        RUN_SCHEMA = "q-attention.q-ceasc.formal-task-graph-run.v1"
        ADAPTIVE_STATE_SCHEMA = "q-attention.q-ceasc.task-adaptive-memory.v1"
    if not SELECTORS or "disabled" not in SELECTORS or not SELECTOR_TASKS:
        raise ValueError("formal config must declare disabled plus at least one selector")


def _canonical_config(config: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(config))
    value["seed"] = 0
    value.pop("replication", None)
    return value


def _source_contract_hashes(repo_root: Path) -> dict[str, dict[str, Any]]:
    return {
        name: {"sha256": sha256(repo_root / relative), "path": relative}
        for name, relative in COUNTERFACTUAL_SOURCE_FILES.items()
    }


def _parse_data_hashes(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValueError(f"invalid data.sha256 row: {line!r}")
        result[Path(fields[1]).name] = fields[0]
    return result


def validate_seed13_report(
    source_dir: Path,
    *,
    config_path: Path,
    config: dict[str, Any],
    current_commit: str,
) -> dict[str, Any]:
    """Validate an audited seed-13 report before allowing report reuse."""
    source_dir = source_dir.resolve()
    if not source_dir.is_dir():
        raise ValueError(f"seed-13 report directory does not exist: {source_dir}")
    if not config.get("counterfactual"):
        raise ValueError("seed-13 report reuse is available only for counterfactual Q-CEASC")
    required = [
        "RUN_COMPLETE",
        "run_config.json",
        "run_summary.json",
        "run_summary.md",
        "data.sha256",
        "data_counts.txt",
        "provenance.json",
        "metrics/baseline.json",
    ]
    for selector in SELECTOR_TASKS:
        required.extend(
            [
                f"metrics/{selector}.json",
                f"case_study/{selector}.json",
                f"case_study/{selector}.sample-trace.json",
            ]
        )
    missing = [relative for relative in required if not (source_dir / relative).is_file()]
    if missing:
        raise ValueError("seed-13 report is incomplete: " + ", ".join(missing))

    report_config = load_json(source_dir / "run_config.json")
    if _canonical_config(report_config) != _canonical_config(config):
        raise ValueError("seed-13 report config differs from the frozen counterfactual config")
    if int(report_config.get("seed", -1)) != 13:
        raise ValueError("seed-13 report run_config.json does not declare seed 13")

    summary = load_json(source_dir / "run_summary.json")
    expected_selectors = list(SELECTORS)
    if (
        summary.get("formal_experiment") is not True
        or summary.get("stage") != "formal_single_seed"
        or int(summary.get("seed", -1)) != 13
        or summary.get("selectors") != expected_selectors
        or summary.get("test_used_for_training_or_selection") is not False
    ):
        raise ValueError("seed-13 report summary violates the frozen formal contract")

    provenance = load_json(source_dir / "provenance.json")
    if provenance.get("git_dirty") is not False:
        raise ValueError("seed-13 report provenance is dirty")
    if provenance.get("config_sha256") != sha256(config_path):
        raise ValueError("seed-13 report config hash differs from checked-out config")
    recorded_files = provenance.get("source_contract", {}).get("files")
    if not isinstance(recorded_files, dict):
        raise ValueError("seed-13 report lacks source_contract file hashes")
    current_files = _source_contract_hashes(ROOT)
    for name, current in current_files.items():
        recorded = recorded_files.get(name)
        if not isinstance(recorded, dict) or recorded.get("sha256") != current["sha256"]:
            raise ValueError(f"seed-13 report source hash differs for {name}")
    source_revision = provenance.get("git_revision")
    if not isinstance(source_revision, str) or not source_revision:
        raise ValueError("seed-13 report lacks source git revision")

    counts = (source_dir / "data_counts.txt").read_text(encoding="utf-8")
    for split, expected in config["expected_records"].items():
        marker = f"{int(expected)} data/relation/retacred/{split}.jsonl"
        if marker not in counts:
            raise ValueError(f"seed-13 report data count mismatch for {split}")
    report_hashes = _parse_data_hashes(source_dir / "data.sha256")
    for split in ("train", "valid", "test"):
        data_path = (ROOT / config[f"{split}_path"]).resolve()
        if not data_path.is_file():
            raise ValueError(f"checked-out data file is missing: {data_path}")
        if report_hashes.get(data_path.name) != sha256(data_path):
            raise ValueError(f"seed-13 report data hash differs for {split}")

    for selector in SELECTOR_TASKS:
        metrics = load_json(source_dir / f"metrics/{selector}.json")
        if metrics.get("selector") != selector or metrics.get("finite") is not True:
            raise ValueError(f"seed-13 report metrics are invalid for {selector}")
        if not isinstance(metrics.get("test", {}).get("metrics"), dict):
            raise ValueError(f"seed-13 report test metrics are missing for {selector}")
        case = load_json(source_dir / f"case_study/{selector}.json")
        if case.get("schema_version") != "q-attention.q-ceasc-case-study.v2":
            raise ValueError(f"seed-13 Case Study schema is invalid for {selector}")
        if set(case.get("required_splits", [])) != {"train", "valid", "test"}:
            raise ValueError(f"seed-13 Case Study split coverage is incomplete for {selector}")
        if len(case.get("cases", [])) != 27:
            raise ValueError(f"seed-13 Case Study must contain 27 cases for {selector}")
        trace = load_json(source_dir / f"case_study/{selector}.sample-trace.json")
        if trace.get("schema_version") != "sample-trace.v1":
            raise ValueError(f"seed-13 sample trace schema is invalid for {selector}")
        if trace.get("experiment", {}).get("config_sha256") != sha256(source_dir / "run_config.json"):
            raise ValueError(f"seed-13 sample trace config hash is invalid for {selector}")

    return {
        "schema_version": "q-attention.q-ceasc-counterfactual.seed13-import.v1",
        "seed": 13,
        "source_report_dir": str(source_dir),
        "source_report_commit": (
            (source_dir / "reporting_commit.txt").read_text(encoding="utf-8").strip()
            if (source_dir / "reporting_commit.txt").is_file()
            else None
        ),
        "source_git_revision": source_revision,
        "validated_against_git_commit": current_commit,
        "config_sha256": sha256(config_path),
        "data_hashes": report_hashes,
        "mode": "audited_report_reuse",
    }


def import_seed13_report(
    source_dir: Path,
    group_dir: Path,
    *,
    config_path: Path,
    config: dict[str, Any],
    current_commit: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = metadata or validate_seed13_report(
        source_dir,
        config_path=config_path,
        config=config,
        current_commit=current_commit,
    )
    seed_dir = group_dir / "seed_13"
    if seed_dir.exists() and any(seed_dir.iterdir()):
        raise ValueError("cannot import seed-13 report into a non-empty seed directory")
    (seed_dir / "baseline").mkdir(parents=True, exist_ok=True)
    for selector in SELECTOR_TASKS:
        (seed_dir / "selectors" / selector).mkdir(parents=True, exist_ok=True)
    copy_map = {
        "RUN_COMPLETE": "RUN_COMPLETE",
        "run_config.json": "run_config.json",
        "run_summary.json": "run_summary.json",
        "run_summary.md": "run_summary.md",
        "data.sha256": "data.sha256",
        "data_counts.txt": "data_counts.txt",
        "provenance.json": "provenance.json",
        "metrics/baseline.json": "baseline/metrics.json",
    }
    for selector in SELECTOR_TASKS:
        copy_map.update(
            {
                f"metrics/{selector}.json": f"selectors/{selector}/metrics.json",
                f"case_study/{selector}.json": f"selectors/{selector}/case_study.json",
                f"case_study/{selector}.sample-trace.json": f"selectors/{selector}/sample_trace.json",
            }
        )
    for source_relative, target_relative in copy_map.items():
        target = seed_dir / target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_dir / source_relative, target)
    write_json(seed_dir / "imported_report.json", metadata)
    return metadata


def parse_int_list(value: str, *, label: str) -> list[int]:
    items = [part.strip() for part in value.split(",")]
    if not items or any(not item.isdigit() for item in items):
        raise ValueError(f"{label} must be a comma-separated list of non-negative integers")
    values = [int(item) for item in items]
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{label} must contain at least one unique integer")
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def query_gpu_inventory() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    inventory: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",", 4)]
        if len(fields) != 5:
            raise RuntimeError(f"unexpected nvidia-smi row: {line!r}")
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


def resolve_gpu_ids(spec: str, inventory: list[dict[str, Any]]) -> list[int]:
    if spec.strip().lower() == "auto":
        ids = [
            int(item["index"])
            for item in inventory
            if int(item["memory_free_mib"]) >= MIN_FREE_MIB
        ]
    else:
        ids = parse_int_list(spec, label="gpus")
    known = {int(item["index"]): item for item in inventory}
    missing = [gpu_id for gpu_id in ids if gpu_id not in known]
    if missing:
        raise ValueError(f"requested GPU IDs are unavailable: {missing}")
    insufficient = [
        gpu_id for gpu_id in ids if int(known[gpu_id]["memory_free_mib"]) < MIN_FREE_MIB
    ]
    if insufficient:
        detail = ", ".join(
            f"GPU {gpu_id} free={known[gpu_id]['memory_free_mib']} MiB"
            for gpu_id in insufficient
        )
        raise RuntimeError(f"selected GPU capacity is below 8 GiB: {detail}")
    if not ids:
        raise RuntimeError("no GPU with at least 8 GiB free was selected")
    return ids


def canonical_config(config: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(config))
    value["seed"] = 0
    value.pop("replication", None)
    return value


def protocol_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(
        canonical_config(config), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_seed_config(
    base_config: dict[str, Any], seed: int, destination: Path
) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    config["seed"] = seed
    config["replication"] = {
        "schema_version": "q-attention.q-ceasc-replication-child.v1",
        "seed_set": list(DEFAULT_SEEDS),
        "child_seed": seed,
        "parent_contract": protocol_fingerprint(base_config),
    }
    write_json(destination, config)
    return config


def build_child_command(
    *,
    seed: int,
    gpu_id: int,
    config_path: Path,
    run_dir: Path,
    args: argparse.Namespace,
    resume: bool = False,
    baseline_only: bool = False,
) -> list[str]:
    command = [
        args.python_bin,
        str(SINGLE_RUNNER),
        "--config",
        str(config_path),
        "--resume" if resume else "--output-dir",
        str(run_dir),
        "--device",
        "cuda",
        "--gpus",
        str(gpu_id),
        "--seed",
        str(seed),
        "--hardware-profile",
        args.hardware_profile,
        "--log-every-batches",
        str(args.log_every_batches),
        "--checkpoint-every-batches",
        str(args.checkpoint_every_batches),
        "--replication-child",
    ]
    if baseline_only:
        command.append("--baseline-only")
    if resume and getattr(args, "allow_gpu_topology_change", False):
        command.append("--allow-gpu-topology-change")
    return command


def build_selector_command(
    *,
    seed: int,
    selector: str,
    gpu_id: int,
    config_path: Path,
    seed_dir: Path,
    selector_dir: Path,
    args: argparse.Namespace,
    profile: dict[str, Any],
    adaptive: bool,
    resume: bool,
) -> list[str]:
    pair_chunk = profile.get("pair_chunk_size")
    command = [
        args.python_bin,
        str(SELECTOR_WORKER),
        "--config",
        str(config_path),
        "--baseline-dir",
        str(seed_dir / "baseline"),
        "--data-dir",
        str(seed_dir / "data"),
        "--output-dir",
        str(selector_dir),
        "--selector",
        selector,
        "--device",
        "cuda",
        "--seed",
        str(seed),
        "--log-every-batches",
        str(args.log_every_batches),
        "--checkpoint-every-batches",
        str(args.checkpoint_every_batches),
        "--pair-chunk-size",
        "all" if pair_chunk is None else str(int(pair_chunk)),
        "--pair-chunk-divisor",
        str(int(profile.get("pair_chunk_divisor", 1))),
        "--micro-batch-size",
        str(int(profile.get("micro_batch_size", 256))),
        "--gradient-accumulation-steps",
        str(int(profile.get("gradient_accumulation_steps", 1))),
        "--activation-checkpointing",
        str(int(bool(profile.get("activation_checkpointing", False)))),
    ]
    if adaptive:
        command.append("--adaptive-memory")
    if resume:
        command.append("--resume")
        # A resumed worker may restart from a durable checkpoint after an
        # execution-only memory-tier change (or a legacy fixed-tier run).
        # Keep the scientific contract strict while allowing that scheduler-
        # controlled execution migration explicitly.
        if adaptive:
            command.append("--elastic-resume")
    return command


def task_id(kind: str, seed: int, selector: str | None = None) -> str:
    if kind == "baseline":
        return f"baseline:{seed}"
    if kind == "selector" and selector:
        return f"selector:{seed}:{selector}"
    raise ValueError(f"invalid task identity: {kind}, {seed}, {selector}")


def make_tasks(seeds: list[int]) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        key = task_id("baseline", seed)
        tasks[key] = {
            "task_id": key,
            "kind": "baseline",
            "seed": seed,
            "selector": None,
            "status": "pending",
            "gpu_id": None,
        }
        for selector in SELECTOR_TASKS:
            key = task_id("selector", seed, selector)
            tasks[key] = {
                "task_id": key,
                "kind": "selector",
                "seed": seed,
                "selector": selector,
                "status": "blocked",
                "gpu_id": None,
            }
    return tasks


def _seed_dir(group_dir: Path, seed: int) -> Path:
    return group_dir / f"seed_{seed}"


def _selector_dir(group_dir: Path, seed: int, selector: str) -> Path:
    return _seed_dir(group_dir, seed) / "selectors" / selector


def _valid_selector_metrics(path: Path, selector: str) -> bool:
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return (
        payload.get("selector") == selector
        and isinstance(payload.get("valid"), dict)
        and isinstance(payload.get("test"), dict)
    )


def _baseline_complete(seed_dir: Path) -> bool:
    return (
        (
            (seed_dir / "BASELINE_COMPLETE").is_file()
            # Legacy multi-seed children predate BASELINE_COMPLETE. A full
            # RUN_COMPLETE plus complete baseline/evaluation artifacts is
            # sufficient evidence that the baseline barrier is durable.
            or (seed_dir / "RUN_COMPLETE").is_file()
        )
        and (seed_dir / "data" / "data_manifest.json").is_file()
        and all((seed_dir / "baseline" / name).is_file() for name in BASELINE_ARTIFACTS)
        and (seed_dir / "baseline_eval.json").is_file()
        and _valid_selector_metrics(
            seed_dir / "selectors" / "disabled" / "metrics.json", "disabled"
        )
    )


def _selector_complete(group_dir: Path, seed: int, selector: str) -> bool:
    directory = _selector_dir(group_dir, seed, selector)
    return (
        _valid_selector_metrics(directory / "metrics.json", selector)
        and (directory / "case_study.json").is_file()
        and (directory / "sample_trace.json").is_file()
    )


def _task_paths(
    group_dir: Path, task: dict[str, Any]
) -> tuple[Path, Path, Path]:
    seed = int(task["seed"])
    seed_dir = _seed_dir(group_dir, seed)
    if task["kind"] == "baseline":
        log_path = group_dir / "task_logs" / f"baseline_seed_{seed}.log"
        command_path = group_dir / "task_commands" / f"baseline_seed_{seed}.json"
        return seed_dir, log_path, command_path
    selector = str(task["selector"])
    selector_dir = _selector_dir(group_dir, seed, selector)
    log_path = group_dir / "task_logs" / f"selector_seed_{seed}_{selector}.log"
    command_path = group_dir / "task_commands" / f"selector_seed_{seed}_{selector}.json"
    return selector_dir, log_path, command_path


def _write_command_artifact(
    path: Path,
    *,
    task: dict[str, Any],
    gpu_id: int,
    command: list[str],
    commit: str,
) -> None:
    target = SINGLE_RUNNER if task["kind"] == "baseline" else SELECTOR_WORKER
    write_json(
        path,
        {
            "schema_version": "q-attention.task-command.v1",
            "task_id": task["task_id"],
            "kind": task["kind"],
            "seed": int(task["seed"]),
            "selector": task.get("selector"),
            "physical_gpu_id": gpu_id,
            "cwd": str(ROOT),
            "target": str(target.relative_to(ROOT)),
            "argv": command,
            "git_revision": commit,
        },
    )


def _latest_heartbeat(group_dir: Path, task: dict[str, Any]) -> dict[str, Any] | None:
    seed_dir = _seed_dir(group_dir, int(task["seed"]))
    if task["kind"] == "baseline":
        path = seed_dir / "baseline" / "heartbeat.json"
    else:
        path = _selector_dir(group_dir, int(task["seed"]), str(task["selector"])) / "heartbeat.json"
    if not path.is_file():
        return None
    try:
        value = load_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    value["heartbeat_file"] = str(path)
    return value


def _format_duration(seconds: Any) -> str:
    try:
        value = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        return "--"
    hours, value = divmod(value, 3600)
    minutes, seconds = divmod(value, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def render_dashboard(
    tasks: dict[str, dict[str, Any]], phase: str, group_dir: Path
) -> str:
    counts: dict[str, int] = {}
    lines: list[str] = []
    for item in tasks.values():
        state = str(item["status"])
        counts[state] = counts.get(state, 0) + 1
        if state != "running":
            continue
        progress = item.get("progress") or {}
        label = (
            f"baseline(seed={item['seed']})"
            if item["kind"] == "baseline"
            else f"selector(seed={item['seed']}, {item['selector']})"
        )
        batch = progress.get("batch", progress.get("completed_batches", "?"))
        total = progress.get("batches", progress.get("total_batches", "?"))
        rate = progress.get("batches_per_second")
        rate_text = f" | {float(rate):.2f} batch/s" if isinstance(rate, (int, float)) else ""
        tier = (
            f" | tier={item.get('adaptive_tier')} {item.get('memory_profile')}"
            if item["kind"] == "selector" and item.get("memory_profile")
            else ""
        )
        lines.append(
            f"  GPU {item.get('gpu_id', '?')} | {label} | "
            f"batch {batch}/{total}{rate_text} | ETA "
            f"{_format_duration(progress.get('eta_seconds'))}{tier}"
        )
    total = len(tasks)
    complete = sum(item["status"] == "complete" for item in tasks.values())
    return "\n".join(
        [
            f"[Q-CEASC task-graph] {group_dir.name} | phase={phase} | "
            f"complete {complete}/{total} | running {counts.get('running', 0)} | "
            f"pending {counts.get('pending', 0)} | blocked {counts.get('blocked', 0)} | "
            f"failed {counts.get('failed', 0)}",
            *lines,
        ]
    )


def _terminate(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            process.wait(timeout=30)


def _request_pause(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        if entry["process"].poll() is None:
            try:
                os.killpg(entry["process"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + SAFE_PAUSE_TIMEOUT_SECONDS
    remaining = list(entries)
    while remaining and time.monotonic() < deadline:
        remaining = [entry for entry in remaining if entry["process"].poll() is None]
        if remaining:
            time.sleep(0.2)
    for entry in remaining:
        _terminate(entry["process"])


def _oom_in_log(path: Path, return_code: int) -> bool:
    if return_code == CUDA_OOM_EXIT_CODE:
        return True
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 1024 * 1024))
            tail = handle.read().decode("utf-8", errors="replace").lower()
    except OSError:
        return False
    return any(marker in tail for marker in OOM_MARKERS)


def _runtime_module() -> Any:
    import run_retacred_qceasc_formal_single_seed as runtime

    return runtime


def _profile_at(hardware_profile: dict[str, Any], tier: int) -> dict[str, Any]:
    if not hardware_profile.get("adaptive"):
        return dict(hardware_profile)
    return _runtime_module()._adaptive_profile_at(hardware_profile, tier)


def _make_adaptive_state(
    group_dir: Path,
    tasks: dict[str, dict[str, Any]],
    hardware_profile: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    path = group_dir / "adaptive_memory_state.json"
    if resume and path.is_file():
        state = load_json(path)
        if state.get("schema_version") != ADAPTIVE_STATE_SCHEMA:
            raise ValueError("unsupported task-level adaptive memory state schema")
        return state
    if resume and not path.is_file():
        raise ValueError("resume group is missing adaptive_memory_state.json")
    tiers = hardware_profile.get("tiers", [])
    state = {
        "schema_version": ADAPTIVE_STATE_SCHEMA,
        "strategy": "per_task",
        "tiers": tiers,
        "tasks": {
            key: {
                "current_tier": 0,
                "current_profile": (
                    tiers[0].get("name")
                    if hardware_profile.get("adaptive") and tiers
                    else hardware_profile.get("name")
                ),
                "oom_retries": 0,
                "memory_pressure_retries": 0,
                "events": [],
            }
            for key, item in tasks.items()
            if item["kind"] == "selector"
        },
    }
    write_json(path, state)
    return state


def _task_memory_state(
    adaptive_state: dict[str, Any], task_key: str
) -> dict[str, Any]:
    return adaptive_state.setdefault("tasks", {}).setdefault(
        task_key,
        {
            "current_tier": 0,
            "current_profile": "adaptive_full_batch",
            "oom_retries": 0,
            "memory_pressure_retries": 0,
            "events": [],
        },
    )


def _metric_delta(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    keys = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "loss")
    return {
        f"delta_{key}": float(current[key]) - float(baseline[key]) for key in keys
    }


def _relative_gain(delta: float, baseline: float) -> float | None:
    return None if baseline == 0.0 else float(delta) / abs(float(baseline))


def _write_seed_summary(
    *,
    group_dir: Path,
    seed: int,
    config_path: Path,
    commit: str,
    gpu_ids: list[int],
    hardware_profile: dict[str, Any],
    adaptive_state: dict[str, Any],
) -> None:
    seed_dir = _seed_dir(group_dir, seed)
    config = load_json(config_path)
    baseline_eval = load_json(seed_dir / "baseline_eval.json")
    rows = [
        load_json(seed_dir / "selectors" / selector / "metrics.json")
        for selector in SELECTORS
    ]
    by_name = {str(row["selector"]): row for row in rows}
    baseline_test = by_name["disabled"]["test"]["metrics"]
    candidate_test = by_name[config["candidate"]]["test"]["metrics"]
    matched_test = by_name[config["matched_control"]]["test"]["metrics"]
    candidate_minus_disabled = _metric_delta(candidate_test, baseline_test)
    candidate_minus_matched = _metric_delta(candidate_test, matched_test)
    candidate_relative = _relative_gain(
        candidate_minus_disabled["delta_macro_f1"], float(baseline_test["macro_f1"])
    )
    classical_relative = _relative_gain(
        float(matched_test["macro_f1"]) - float(baseline_test["macro_f1"]),
        float(baseline_test["macro_f1"]),
    )
    gates_config = config.get("gates", {})
    gates = {
        "candidate_minus_disabled_macro_f1": candidate_minus_disabled["delta_macro_f1"],
        "candidate_minus_matched_macro_f1": candidate_minus_matched["delta_macro_f1"],
        "candidate_relative_gain_macro_f1": candidate_relative,
        "classical_relative_gain_macro_f1": classical_relative,
        "l1_utility_gate": candidate_minus_disabled["delta_macro_f1"] > 0.0,
        "practical_gain_gate": candidate_minus_disabled["delta_macro_f1"]
        >= float(gates_config.get("minimum_candidate_minus_disabled_macro_f1", 0.0)),
        "quantum_inspired_relative_gain_gate": classical_relative is not None
        and classical_relative
        > float(gates_config.get("minimum_classical_relative_gain", 0.01)),
        "matched_comparator_gate": candidate_minus_matched["delta_macro_f1"]
        >= float(gates_config.get("minimum_candidate_minus_matched_macro_f1", 0.0)),
        "finite_metrics": all(bool(row.get("finite")) for row in rows),
        "test_used_for_training_or_selection": False,
    }
    summary = {
        "schema_version": "q-attention.q-ceasc-formal-single-seed.run.v1",
        "name": config["name"],
        "formal_experiment": True,
        "stage": "formal_single_seed",
        "seed": seed,
        "run_dir": str(seed_dir),
        "device": "cuda",
        "parallel_mode": "task_graph",
        "model_parallel": {"enabled": False, "physical_gpu_ids": []},
        "multi_gpu": {
            "requested_gpu_ids": gpu_ids,
            "selector_parallelism": len(gpu_ids),
            "scheduler": "global_selector_queue",
        },
        "hardware_profile": hardware_profile,
        "adaptive_memory": adaptive_state,
        "selectors": list(SELECTORS),
        "candidate": config["candidate"],
        "matched_control": config["matched_control"],
        "data": {},
        "baseline": {
            "valid": baseline_eval["valid"],
            "test": baseline_eval["test"],
            "import": None,
        },
        "baseline_import": None,
        "rows": rows,
        "candidate_minus_disabled": candidate_minus_disabled,
        "candidate_minus_matched": candidate_minus_matched,
        "relative_deltas": {
            "candidate_macro_f1_vs_disabled": candidate_relative,
            "classical_macro_f1_vs_disabled": classical_relative,
            "classical_quantum_inspired_threshold": float(
                gates_config.get("minimum_classical_relative_gain", 0.01)
            ),
        },
        "rating_policy": {
            "id": "q-attention-utility-and-qi-v1",
            "version": "2026-09-13",
        },
        "gates": gates,
        "test_used_for_training_or_selection": False,
        "claim_limits": config.get("claim_limits", {}),
        "provenance": {
            "config_path": str(config_path),
            "config_sha256": sha256(config_path),
            "git_revision": commit,
            "git_dirty": False,
            "visible_cuda_devices": gpu_ids,
            "memory_strategy": "adaptive"
            if hardware_profile.get("adaptive")
            else "fixed",
            "started_at_utc": load_json(group_dir / "multi_seed_manifest.json")[
                "started_at_utc"
            ],
        },
    }
    write_json(seed_dir / "run_config.json", config)
    write_json(seed_dir / "run_summary.json", summary)
    lines = [
        "# Q-CEASC Re-TACRED Formal Single Seed",
        "",
        f"- seed: {seed}",
        "- parallel mode: task_graph",
        f"- selected physical GPUs: {gpu_ids}",
        f"- candidate minus disabled test macro-F1: {candidate_minus_disabled['delta_macro_f1']:.6f}",
        f"- candidate minus matched test macro-F1: {candidate_minus_matched['delta_macro_f1']:.6f}",
        f"- L1 utility gate: {str(gates['l1_utility_gate']).lower()}",
        f"- quantum-inspired relative-gain gate: {str(gates['quantum_inspired_relative_gain_gate']).lower()}",
        "",
        "The test split is evaluated only after training and validation selection.",
    ]
    (seed_dir / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (seed_dir / "RUN_PAUSED").unlink(missing_ok=True)
    (seed_dir / "RUN_FAILED").unlink(missing_ok=True)
    (seed_dir / "RUN_COMPLETE").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )


def run_group(
    args: argparse.Namespace,
    *,
    group_dir: Path,
    seeds: list[int],
    gpu_ids: list[int],
    commit: str,
    base_config: dict[str, Any],
    resume_group: bool = False,
    import_seed13_report_path: Path | None = None,
    import_seed13_metadata: dict[str, Any] | None = None,
) -> int:
    configs_dir = group_dir / "configs"
    if resume_group:
        if not group_dir.is_dir() or not (group_dir / "multi_seed_manifest.json").is_file():
            raise ValueError("--resume-group requires an existing task-graph manifest")
        manifest = load_json(group_dir / "multi_seed_manifest.json")
        if manifest.get("schema_version") not in {
            MANIFEST_SCHEMA,
            "q-attention.q-ceasc.formal-multiseed-manifest.v1",
        }:
            raise ValueError("unsupported multi-seed manifest schema")
        if manifest.get("protocol", PROTOCOL) != PROTOCOL:
            raise ValueError("resume group protocol differs from the checked-out formal config")
        if [int(seed) for seed in manifest.get("seeds", [])] != seeds:
            raise ValueError("resume group seed set differs from frozen 13,29,53")
        if manifest.get("git_commit") != commit:
            raise ValueError("resume group was created by a different code revision")
        if manifest.get("base_config_sha256") != sha256(CONFIG_PATH):
            raise ValueError("resume group was created from a different formal config")
        previous_gpus = manifest.get("gpus")
        if isinstance(previous_gpus, list):
            try:
                previous_gpu_ids = sorted({int(gpu_id) for gpu_id in previous_gpus})
            except (TypeError, ValueError) as exc:
                raise ValueError("resume group manifest contains invalid GPU IDs") from exc
            if previous_gpu_ids != sorted(gpu_ids) and not args.allow_gpu_topology_change:
                raise ValueError(
                    "resume GPU topology differs from the original group; "
                    "rerun with --allow-gpu-topology-change to authorize reassignment"
                )
        if (group_dir / "MULTI_SEED_COMPLETE").is_file():
            raise ValueError("multi-seed group is already complete")
        for seed in seeds:
            if not (configs_dir / f"seed_{seed}.json").is_file():
                raise ValueError(f"resume group is missing configs/seed_{seed}.json")
        (group_dir / "MULTI_SEED_FAILED").unlink(missing_ok=True)
        (group_dir / "MULTI_SEED_PAUSED").unlink(missing_ok=True)
    else:
        group_dir.mkdir(parents=True, exist_ok=False)
        for seed in seeds:
            build_seed_config(base_config, seed, configs_dir / f"seed_{seed}.json")
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "protocol": PROTOCOL,
            "scheduler": "two_phase_task_graph",
            "task_granularity": ["baseline(seed)", "selector(seed,name)"],
            "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": commit,
            "base_config": str(CONFIG_PATH),
            "base_config_sha256": sha256(CONFIG_PATH),
            "protocol_fingerprint": protocol_fingerprint(base_config),
            "seeds": seeds,
            "selectors": list(SELECTORS),
            "gpus": gpu_ids,
            "gpu_inventory": args.gpu_inventory,
            "worker_count": min(len(seeds), len(gpu_ids)),
            "max_workers_per_gpu": 1,
            "hardware_profile": args.hardware_profile,
            "checkpoint_every_batches": args.checkpoint_every_batches,
            "imported_seed_reports": {},
        }
        write_json(group_dir / "multi_seed_manifest.json", manifest)

    imported_seed_reports = manifest.get("imported_seed_reports", {})
    if not isinstance(imported_seed_reports, dict):
        raise ValueError("multi-seed manifest has invalid imported_seed_reports")
    if import_seed13_report_path is not None:
        if resume_group:
            raise ValueError("--import-seed13-report cannot be combined with --resume-group")
        if 13 not in seeds:
            raise ValueError("seed-13 report reuse requires seed 13 in the replication set")
        metadata = import_seed13_report(
            import_seed13_report_path,
            group_dir,
            config_path=CONFIG_PATH,
            config=base_config,
            current_commit=commit,
            metadata=import_seed13_metadata,
        )
        imported_seed_reports["13"] = metadata
        manifest["imported_seed_reports"] = imported_seed_reports
        write_json(group_dir / "multi_seed_manifest.json", manifest)

    imported_seed_set = {
        int(seed)
        for seed in imported_seed_reports
        if str(seed).isdigit()
    }
    tasks = make_tasks(seeds)
    for key, item in tasks.items():
        seed = int(item["seed"])
        seed_dir = _seed_dir(group_dir, seed)
        if seed in imported_seed_set:
            item.update({"status": "complete", "resumed_skip": True, "imported_report": True})
            continue
        if item["kind"] == "baseline":
            if resume_group and _baseline_complete(seed_dir):
                item.update({"status": "complete", "resumed_skip": True})
            elif resume_group and seed_dir.exists() and not (seed_dir / "run_manifest.json").is_file():
                if any(seed_dir.iterdir()):
                    raise ValueError(
                        f"{key} has artifacts but no run_manifest.json; refusing unsafe restart"
                    )
                raise ValueError(
                    f"{key} output directory exists without run_manifest.json; "
                    "start a new group or repair the directory explicitly"
                )
        else:
            selector = str(item["selector"])
            if resume_group and _selector_complete(group_dir, seed, selector):
                item.update({"status": "complete", "resumed_skip": True})
            elif resume_group:
                selector_dir = _selector_dir(group_dir, seed, selector)
                if selector_dir.exists() and any(selector_dir.iterdir()) and not (
                    selector_dir / "checkpoints" / "latest.pt"
                ).is_file():
                    raise ValueError(
                        f"{key} has partial artifacts but no batch checkpoint; refusing unsafe restart"
                    )

    runtime = _runtime_module()
    hardware_profile = runtime.choose_hardware_profile(
        args.hardware_profile, base_config, gpu_ids, args.gpu_inventory
    )
    adaptive = bool(hardware_profile.get("adaptive"))
    adaptive_state = _make_adaptive_state(
        group_dir, tasks, hardware_profile, resume=resume_group
    )

    phase = "baseline"
    pending: deque[str] = deque(
        key
        for key, item in tasks.items()
        if item["kind"] == "baseline" and item["status"] != "complete"
    )
    if not pending:
        phase = "selectors"
        for key, item in tasks.items():
            if item["kind"] == "selector" and item["status"] != "complete":
                item["status"] = "pending"
                pending.append(key)
    available: deque[int] = deque(gpu_ids)
    active: dict[str, dict[str, Any]] = {}
    failed = False
    paused = False
    stop_requested = False
    previous_handlers: dict[int, Any] = {}

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(
            f"[task-graph] received signal {signum}; requesting safe child pause",
            flush=True,
        )

    for signal_name in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if signal_name is not None:
            previous_handlers[signal_name] = signal.getsignal(signal_name)
            signal.signal(signal_name, request_stop)

    def save_state() -> None:
        write_json(
            group_dir / "multi_seed_status.json",
            {
                "schema_version": STATUS_SCHEMA,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "group_dir": str(group_dir),
                "phase": phase,
                "assignments": list(tasks.values()),
            },
        )
        counts: dict[str, int] = {}
        for item in tasks.values():
            counts[item["status"]] = counts.get(item["status"], 0) + 1
        write_json(
            group_dir / "multi_seed_heartbeat.json",
            {
                "schema_version": "q-attention.task-graph-heartbeat.v1",
                "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "phase": phase,
                "counts": counts,
                "active_gpus": {
                    key: item.get("gpu_id")
                    for key, item in tasks.items()
                    if item["status"] == "running"
                },
            },
        )

    def abort_active(reason: str) -> None:
        for key, entry in list(active.items()):
            _terminate(entry["process"])
            entry["handle"].close()
            tasks[key].update(
                {
                    "status": "failed",
                    "return_code": entry["process"].poll(),
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "reason": reason,
                }
            )
            available.append(int(entry["gpu_id"]))
            del active[key]

    def pause_active(reason: str) -> None:
        nonlocal paused
        entries = list(active.values())
        _request_pause(entries)
        now = datetime.now(timezone.utc).isoformat()
        for key, entry in list(active.items()):
            code = entry["process"].poll()
            entry["handle"].close()
            tasks[key].update(
                {
                    "status": "paused" if code == PAUSED_EXIT_CODE else "failed",
                    "return_code": code,
                    "finished_at_utc": now,
                    "reason": reason,
                }
            )
            available.append(int(entry["gpu_id"]))
            del active[key]
        for key in list(pending):
            tasks[key].update({"status": "paused", "reason": reason})
        pending.clear()
        paused = True

    def launch(key: str, gpu_id: int) -> None:
        item = tasks[key]
        seed = int(item["seed"])
        config_path = configs_dir / f"seed_{seed}.json"
        if item["kind"] == "baseline":
            seed_dir = _seed_dir(group_dir, seed)
            resume = resume_group and (seed_dir / "run_manifest.json").is_file()
            command = build_child_command(
                seed=seed,
                gpu_id=gpu_id,
                config_path=config_path,
                run_dir=seed_dir,
                args=args,
                resume=resume,
                baseline_only=True,
            )
            output_path, log_path, command_path = _task_paths(group_dir, item)
        else:
            selector = str(item["selector"])
            selector_dir = _selector_dir(group_dir, seed, selector)
            selector_dir.mkdir(parents=True, exist_ok=True)
            memory = _task_memory_state(adaptive_state, key)
            tier = int(memory.get("current_tier", 0))
            profile = _profile_at(hardware_profile, tier)
            checkpoint = selector_dir / "checkpoints" / "latest.pt"
            resume = checkpoint.is_file()
            command = build_selector_command(
                seed=seed,
                selector=selector,
                gpu_id=gpu_id,
                config_path=config_path,
                seed_dir=_seed_dir(group_dir, seed),
                selector_dir=selector_dir,
                args=args,
                profile=profile,
                adaptive=adaptive,
                resume=resume,
            )
            output_path, log_path, command_path = _task_paths(group_dir, item)
            item.update(
                {
                    "adaptive_tier": tier if adaptive else None,
                    "memory_profile": profile.get("name"),
                    "pair_chunk_size": "all"
                    if profile.get("pair_chunk_size") is None
                    else int(profile["pair_chunk_size"]),
                    "pair_chunk_divisor": int(profile.get("pair_chunk_divisor", 1)),
                    "micro_batch_size": int(profile.get("micro_batch_size", 256)),
                    "gradient_accumulation_steps": int(
                        profile.get("gradient_accumulation_steps", 1)
                    ),
                    "activation_checkpointing": bool(
                        profile.get("activation_checkpointing", False)
                    ),
                }
            )
        _write_command_artifact(
            command_path, task=item, gpu_id=gpu_id, command=command, commit=commit
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a" if resume else "w", encoding="utf-8")
        heartbeat_path = (
            _seed_dir(group_dir, seed) / "baseline" / "heartbeat.json"
            if item["kind"] == "baseline"
            else _selector_dir(group_dir, seed, str(item["selector"])) / "heartbeat.json"
        )
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONUNBUFFERED": "1",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": str(gpu_id),
                "Q_ATTENTION_PROGRESS_FORMAT": "json",
                "Q_ATTENTION_HEARTBEAT_FILE": str(heartbeat_path),
            }
        )
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        now = datetime.now(timezone.utc).isoformat()
        item.update(
            {
                "status": "running",
                "gpu_id": gpu_id,
                "pid": process.pid,
                "started_at_utc": now,
                "log_file": str(log_path),
                "command_file": str(command_path),
            }
        )
        active[key] = {
            "process": process,
            "handle": handle,
            "gpu_id": gpu_id,
            "started_monotonic": time.monotonic(),
            "log_path": log_path,
            "output_path": output_path,
        }
        print(f"[task-graph] started {key} on GPU {gpu_id}", flush=True)

    save_state()
    dashboard_at = 0.0
    try:
        while pending or active:
            if stop_requested:
                pause_active("parent signal")
                break
            while pending and available:
                key = pending.popleft()
                if tasks[key]["status"] in {"complete", "paused", "failed"}:
                    continue
                gpu_id = available.popleft()
                launch(key, gpu_id)
                save_state()

            for key, entry in list(active.items()):
                progress = _latest_heartbeat(group_dir, tasks[key])
                if progress:
                    tasks[key]["progress"] = progress
                return_code = entry["process"].poll()
                if return_code is None:
                    continue
                entry["handle"].close()
                elapsed = round(time.monotonic() - entry["started_monotonic"], 3)
                item = tasks[key]
                seed = int(item["seed"])
                success = return_code == 0 and (
                    _baseline_complete(_seed_dir(group_dir, seed))
                    if item["kind"] == "baseline"
                    else _selector_complete(group_dir, seed, str(item["selector"]))
                )
                if success:
                    item.update(
                        {
                            "status": "complete",
                            "return_code": 0,
                            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                            "duration_seconds": elapsed,
                        }
                    )
                    available.append(int(entry["gpu_id"]))
                    del active[key]
                    print(
                        f"[task-graph] complete {key} on GPU {entry['gpu_id']} "
                        f"in {_format_duration(elapsed)}",
                        flush=True,
                    )
                    continue

                if item["kind"] == "selector" and adaptive:
                    memory = _task_memory_state(adaptive_state, key)
                    tier = int(memory.get("current_tier", 0))
                    tiers = hardware_profile.get("tiers", [])
                    checkpoint = (
                        _selector_dir(group_dir, seed, str(item["selector"]))
                        / "checkpoints"
                        / "latest.pt"
                    )
                    oom = _oom_in_log(entry["log_path"], int(return_code))
                    pressure = int(return_code) == MEMORY_PRESSURE_EXIT_CODE
                    if (oom or pressure) and checkpoint.is_file() and tier + 1 < len(tiers):
                        next_tier = tier + 1
                        memory["current_tier"] = next_tier
                        memory["current_profile"] = tiers[next_tier]["name"]
                        field = "memory_pressure_retries" if pressure else "oom_retries"
                        memory[field] = int(memory.get(field, 0)) + 1
                        event = {
                            "event": "memory_tier_retry",
                            "task_id": key,
                            "seed": seed,
                            "selector": item["selector"],
                            "from_tier": tier,
                            "to_tier": next_tier,
                            "trigger": "memory_pressure" if pressure else "cuda_oom",
                            "checkpoint": str(checkpoint),
                            "restart_count": int(memory[field]),
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        }
                        memory.setdefault("events", []).append(event)
                        write_json(group_dir / "adaptive_memory_state.json", adaptive_state)
                        item.update(
                            {
                                "status": "pending",
                                "gpu_id": None,
                                "last_return_code": int(return_code),
                                field: int(memory[field]),
                                "last_memory_tier": tier,
                            }
                        )
                        available.append(int(entry["gpu_id"]))
                        pending.insert(0, key)
                        del active[key]
                        print(
                            f"[task-graph] retry {key} after "
                            f"{'memory pressure' if pressure else 'CUDA OOM'} -> "
                            f"{tiers[next_tier]['name']}",
                            flush=True,
                        )
                        continue

                item.update(
                    {
                        "status": "failed",
                        "return_code": int(return_code),
                        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                        "duration_seconds": elapsed,
                        "reason": (
                            "missing required task artifacts"
                            if return_code == 0
                            else "child process failed"
                        ),
                    }
                )
                available.append(int(entry["gpu_id"]))
                del active[key]
                failed = True
                while pending:
                    pending_key = pending.popleft()
                    tasks[pending_key].update(
                        {"status": "not_started", "reason": f"sibling task {key} failed"}
                    )
                abort_active(f"sibling task {key} failed")
                print(f"[task-graph] failed {key}", file=sys.stderr, flush=True)
                break

            if failed:
                break
            if phase == "baseline" and all(
                item["status"] == "complete"
                for item in tasks.values()
                if item["kind"] == "baseline"
            ):
                phase = "selectors"
                for key, item in tasks.items():
                    if item["kind"] == "selector" and item["status"] not in {
                        "complete",
                        "failed",
                    }:
                        item["status"] = "pending"
                        pending.append(key)
                print(
                    "[task-graph] baseline barrier passed; six selectors are ready",
                    flush=True,
                )
            save_state()
            now = time.monotonic()
            if now - dashboard_at >= args.dashboard_interval:
                dashboard_at = now
                print(render_dashboard(tasks, phase, group_dir), flush=True)
            if active:
                time.sleep(0.2)
    except BaseException:
        if active:
            abort_active("scheduler exception")
        failed = True
        raise
    finally:
        for entry in active.values():
            entry["handle"].close()
        for signal_name, handler in previous_handlers.items():
            signal.signal(signal_name, handler)

    if paused:
        for item in tasks.values():
            if item["status"] in {"pending", "blocked", "running"}:
                item.update({"status": "paused", "reason": "parent paused"})
        save_state()
        write_json(
            group_dir / "multi_seed_run_summary.json",
            {
                "schema_version": RUN_SCHEMA,
                "group_dir": str(group_dir),
                "phase": phase,
                "success": False,
                "paused": True,
                "tasks": list(tasks.values()),
            },
        )
        (group_dir / "MULTI_SEED_PAUSED").write_text(
            datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
        )
        return PAUSED_EXIT_CODE

    completed = all(item["status"] == "complete" for item in tasks.values())
    if completed:
        for seed in seeds:
            if seed in imported_seed_set:
                continue
            _write_seed_summary(
                group_dir=group_dir,
                seed=seed,
                config_path=configs_dir / f"seed_{seed}.json",
                commit=commit,
                gpu_ids=gpu_ids,
                hardware_profile=hardware_profile,
                adaptive_state=adaptive_state,
            )
        write_json(
            group_dir / "multi_seed_run_summary.json",
            {
                "schema_version": RUN_SCHEMA,
                "group_dir": str(group_dir),
                "phase": phase,
                "success": True,
                "paused": False,
                "tasks": list(tasks.values()),
            },
        )
        summary_command = [
            args.python_bin,
            str(ROOT / "scripts" / "summarize_retacred_qceasc_formal_multi_seed.py"),
            "--group-dir",
            str(group_dir),
            "--output-json",
            str(group_dir / "multi_seed_summary.json"),
            "--output-md",
            str(group_dir / "multi_seed_summary.md"),
        ]
        subprocess.run(summary_command, cwd=ROOT, check=True)
        (group_dir / "MULTI_SEED_COMPLETE").write_text(
            datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
        )
        print(f"[task-graph] complete group_dir={group_dir}", flush=True)
        if not args.no_export:
            export_command = [
                "bash",
                "scripts/export_retacred_qceasc_formal_multi_seed_report.sh",
                "--group-dir",
                str(group_dir),
            ]
            if args.report_dir:
                export_command.extend(["--report-dir", str(args.report_dir)])
            result = subprocess.run(export_command, cwd=ROOT, check=False)
            if result.returncode != 0:
                print(
                    "[task-graph] experiment complete but audited report export failed",
                    file=sys.stderr,
                    flush=True,
                )
                return result.returncode
        return 0

    write_json(
        group_dir / "multi_seed_run_summary.json",
        {
            "schema_version": RUN_SCHEMA,
            "group_dir": str(group_dir),
            "phase": phase,
            "success": False,
            "paused": False,
            "tasks": list(tasks.values()),
        },
    )
    (group_dir / "MULTI_SEED_FAILED").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )
    print(f"[task-graph] failed group_dir={group_dir}", file=sys.stderr, flush=True)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "retacred_qceasc_formal_single_seed.json",
        help="frozen formal config; counterfactual config enables the counterfactual protocol",
    )
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--gpus", "--gpu", dest="gpus", default="auto")
    parser.add_argument(
        "--hardware-profile",
        choices=("adaptive", "auto", "low_memory", "balanced", "high_memory"),
        default="adaptive",
    )
    parser.add_argument("--log-every-batches", type=int, default=50)
    parser.add_argument("--checkpoint-every-batches", type=int, default=50)
    parser.add_argument("--dashboard-interval", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume-group", type=Path, default=None)
    parser.add_argument(
        "--import-seed13-report",
        type=Path,
        default=None,
        metavar="REPORT_DIR",
        help=(
            "reuse an audited counterfactual seed-13 report and run only fresh seeds 29 and 53; "
            "the report is rejected unless config, data and source hashes match"
        ),
    )
    parser.add_argument("--allow-gpu-topology-change", action="store_true")
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="reserved for canary-approved overcommit; current maximum is 1",
    )
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--no-export", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python-bin", default=sys.executable, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        seeds = parse_int_list(args.seeds, label="seeds")
        if seeds != list(DEFAULT_SEEDS):
            raise ValueError("L2 replication seed set is frozen to 13,29,53")
        if args.log_every_batches <= 0 or args.checkpoint_every_batches <= 0:
            raise ValueError("batch intervals must be positive")
        if args.dashboard_interval <= 0:
            raise ValueError("--dashboard-interval must be positive")
        if args.workers_per_gpu != 1:
            raise ValueError(
                "overcommit mode is disabled until a hardware-specific canary approves two workers per GPU"
            )
        config_path = args.config if args.config.is_absolute() else ROOT / args.config
        config_path = config_path.resolve()
        if not config_path.is_file() or not SINGLE_RUNNER.is_file() or not SELECTOR_WORKER.is_file():
            raise ValueError("Q-CEASC formal runner or selector worker is missing")
        base_config = load_json(config_path)
        configure_protocol(config_path, base_config)
        if base_config.get("schema_version") not in {
            "q-attention.q-ceasc-formal-single-seed.v1",
            COUNTERFACTUAL_CONFIG_SCHEMA,
        }:
            raise ValueError("unsupported Q-CEASC formal config")
        if args.import_seed13_report is not None and not base_config.get("counterfactual"):
            raise ValueError("--import-seed13-report requires the counterfactual formal config")
        if args.import_seed13_report is not None and args.resume_group:
            raise ValueError("--import-seed13-report cannot be combined with --resume-group")
        inventory = query_gpu_inventory()
        gpu_ids = resolve_gpu_ids(args.gpus, inventory)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
    args.gpu_inventory = inventory
    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    commit = commit_result.stdout.strip() if commit_result.returncode == 0 else "unknown"
    import_seed13_metadata = None
    import_seed13_report_path = None
    if args.import_seed13_report is not None:
        import_seed13_report_path = (
            args.import_seed13_report
            if args.import_seed13_report.is_absolute()
            else ROOT / args.import_seed13_report
        ).resolve()
        import_seed13_metadata = validate_seed13_report(
            import_seed13_report_path,
            config_path=CONFIG_PATH,
            config=base_config,
            current_commit=commit,
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.output_dir and args.resume_group:
        raise SystemExit("--output-dir and --resume-group are mutually exclusive")
    group_dir = args.resume_group or args.output_dir or GROUP_ROOT / stamp
    if not group_dir.is_absolute():
        group_dir = ROOT / group_dir
    group_dir = group_dir.resolve()
    if not group_dir.is_relative_to(GROUP_ROOT.resolve()):
        raise SystemExit(f"output directory must be under {GROUP_ROOT}/")
    if group_dir.exists() and not args.dry_run and not args.resume_group:
        raise SystemExit(f"refusing to reuse output directory: {group_dir}")
    print(
        f"[task-graph] seeds={','.join(map(str, seeds))} | "
        f"gpus={','.join(map(str, gpu_ids))} | max_workers_per_gpu=1 | "
        "baseline barrier -> global selector queue",
        flush=True,
    )
    if args.dry_run:
        for seed in seeds:
            command = build_child_command(
                seed=seed,
                gpu_id=gpu_ids[(seed - seeds[0]) % len(gpu_ids)],
                config_path=group_dir / "configs" / f"seed_{seed}.json",
                run_dir=group_dir / f"seed_{seed}",
                args=args,
                baseline_only=True,
            )
            print(f"[dry-run] baseline:{seed} {' '.join(command)}", flush=True)
        for seed in seeds:
            for selector in SELECTOR_TASKS:
                command = build_selector_command(
                    seed=seed,
                    selector=selector,
                    gpu_id=gpu_ids[0],
                    config_path=group_dir / "configs" / f"seed_{seed}.json",
                    seed_dir=group_dir / f"seed_{seed}",
                    selector_dir=group_dir / f"seed_{seed}" / "selectors" / selector,
                    args=args,
                    profile={
                        "pair_chunk_size": None,
                        "pair_chunk_divisor": 1,
                        "micro_batch_size": 256,
                        "gradient_accumulation_steps": 1,
                        "activation_checkpointing": False,
                    },
                    adaptive=args.hardware_profile == "adaptive",
                    resume=False,
                )
                print(f"[dry-run] selector:{seed}:{selector} {' '.join(command)}", flush=True)
        return 0
    if not args.skip_preflight:
        preflight = [
            args.python_bin if PROTOCOL == "qceasc_counterfactual" else "bash",
            (
                "scripts/check_retacred_qceasc_counterfactual_formal_single_seed.py"
                if PROTOCOL == "qceasc_counterfactual"
                else "scripts/check_retacred_qceasc_formal_single_seed.sh"
            ),
            "--fresh",
            "--gpus",
            ",".join(map(str, gpu_ids)),
            "--device",
            "cuda",
            "--hardware-profile",
            args.hardware_profile,
        ]
        subprocess.run(preflight, cwd=ROOT, check=True)
    return run_group(
        args,
        group_dir=group_dir,
        seeds=seeds,
        gpu_ids=gpu_ids,
        commit=commit,
        base_config=base_config,
        resume_group=bool(args.resume_group),
        import_seed13_report_path=import_seed13_report_path,
        import_seed13_metadata=import_seed13_metadata,
    )


if __name__ == "__main__":
    raise SystemExit(main())
