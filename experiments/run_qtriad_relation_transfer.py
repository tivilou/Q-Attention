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
import threading
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
from q_attention.experiments.progress import format_gpu_memory  # noqa: E402
from q_attention.experiments.batch_resume import (  # noqa: E402
    PAUSED_EXIT_CODE,
    PauseController,
    ResumeCompatibilityError,
    TrainingPaused,
    fingerprint,
    file_contract,
)
from q_attention.plugins.q_triad import QTriadAttentionScoreKernel  # noqa: E402
from run_q_causal_value_evidence_relation_smoke import (  # noqa: E402
    materialize_subset,
    resolve_path,
)
from run_q_causal_value_evidence_relation_transfer import (  # noqa: E402
    evaluate,
    metric_delta,
    train_kernel,
)
from q_attention.tasks.relation import load_relation_jsonl  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs" / "retacred_qtriad_formal_single_seed.json"
RUN_MANIFEST_SCHEMA = "q-attention.qtriad-batch-resume-run.v1"
DATA_MANIFEST_SCHEMA = "q-attention.qtriad-materialized-data.v1"
SAFE_PAUSE_TIMEOUT_SECONDS = 15 * 60

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

# The adaptive path deliberately starts with the largest safe streamed-backward
# chunk.  The streamed backward implementation remains enabled internally: the
# old graph-retaining implementation is not a valid performance tier because it
# can grow until a late, unrecoverable OOM.  Each lower tier trades throughput
# for a smaller per-chunk peak and is selected only after a confirmed CUDA OOM.
ADAPTIVE_HARDWARE_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "name": "adaptive_fast",
        "pair_chunk_size": 16384,
        "activation_checkpointing": False,
    },
    {
        "name": "adaptive_large",
        "pair_chunk_size": 4096,
        "activation_checkpointing": False,
    },
    {
        "name": "adaptive_medium",
        "pair_chunk_size": 1024,
        "activation_checkpointing": True,
    },
    {
        "name": "adaptive_conservative",
        "pair_chunk_size": 256,
        "activation_checkpointing": True,
    },
    {
        "name": "adaptive_low_memory",
        "pair_chunk_size": 64,
        "activation_checkpointing": True,
    },
)
CUDA_OOM_MARKERS = (
    "cuda out of memory",
    "cuda error: out of memory",
    "cuda_error_out_of_memory",
    "cublas_status_alloc_failed",
    "cudaerrormemoryallocation",
)
CUDA_OOM_EXIT_CODE = 86
ADAPTIVE_MEMORY_STATE_SCHEMA = "q-attention.qtriad-adaptive-memory.v1"


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
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help="resume an existing compatible run directory in place",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--gpus",
        default=None,
        help="comma-separated physical GPU IDs or auto; selectors run as independent workers",
    )
    parser.add_argument(
        "--model-parallel-gpus",
        default=None,
        help="comma-separated physical GPU IDs for layer-sharded model parallelism",
    )
    parser.add_argument(
        "--hardware-profile",
        choices=("config", "auto", "adaptive", "low_memory", "balanced", "high_memory"),
        default="adaptive",
        help="execution-memory profile; adaptive starts fast and falls back after CUDA OOM",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every-batches", type=int, default=50)
    parser.add_argument("--checkpoint-every-batches", type=int, default=50)
    parser.add_argument("--started-at-utc", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--python-bin", default=sys.executable, help=argparse.SUPPRESS)
    return parser.parse_args()


class RunPaused(RuntimeError):
    """The run stopped at a durable post-update checkpoint."""


def _source_contract() -> dict[str, Any]:
    paths = {
        "runner": ROOT / "experiments" / "run_qtriad_relation_transfer.py",
        "worker": ROOT / "experiments" / "run_qtriad_selector_worker.py",
        "baseline_trainer": ROOT / "experiments" / "train_relation_baseline.py",
        "kernel_trainer": ROOT
        / "experiments"
        / "run_q_causal_value_evidence_relation_transfer.py",
        "batch_resume": ROOT
        / "src"
        / "q_attention"
        / "experiments"
        / "batch_resume.py",
        "relation_steering": ROOT
        / "src"
        / "q_attention"
        / "experiments"
        / "relation_steering.py",
        "relation_model": ROOT
        / "src"
        / "q_attention"
        / "models"
        / "relation_transformer.py",
        "relation_task": ROOT
        / "src"
        / "q_attention"
        / "tasks"
        / "relation.py",
        "attention_adapter": ROOT
        / "src"
        / "q_attention"
        / "adapters"
        / "attention_scores.py",
        "q_triad": ROOT
        / "src"
        / "q_attention"
        / "plugins"
        / "q_triad.py",
    }
    return {
        "git_revision": git_output("rev-parse", "HEAD"),
        "files": {name: file_contract(path) for name, path in paths.items()},
    }


def selector_resume_contract(
    *,
    config_path: Path,
    baseline_dir: Path,
    data_dir: Path,
    selector: str,
    seed: int,
    pair_chunk_size: int | None,
    activation_checkpointing: int | bool | None,
    model_parallel_gpu_ids: list[int] | None = None,
    adaptive_memory: bool = False,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    kernel = config["kernel"]
    return {
        "stage": "qtriad_selector",
        "training_semantics": {
            "selector": selector,
            "seed": int(seed),
            "batch_size": int(kernel["batch_size"]),
            "epochs": int(kernel["epochs"]),
            "kernel_lr": float(kernel["lr"]),
            # Adaptive retries intentionally keep these controls out of the
            # immutable checkpoint fingerprint.  The scientific optimizer
            # contract stays fixed while only the execution strategy changes.
            **(
                {"memory_strategy": "adaptive"}
                if adaptive_memory
                else {
                    "pair_chunk_size": int(
                        pair_chunk_size
                        if pair_chunk_size is not None
                        else kernel.get("pair_chunk_size", 256)
                    ),
                    "activation_checkpointing": (
                        True
                        if activation_checkpointing is None
                        else bool(activation_checkpointing)
                    ),
                }
            ),
            "model_parallel_gpu_ids": list(model_parallel_gpu_ids or []),
        },
        "config": file_contract(config_path),
        "baseline": {
            name: file_contract(baseline_dir / name)
            for name in ("model.pt", "vocab.json", "labels.json", "metrics.json")
        },
        "data": {
            split: file_contract(data_dir / f"{split}.jsonl")
            for split in ("train", "valid", "test")
        },
        "materialization": file_contract(data_dir / "data_manifest.json"),
        "source": _source_contract(),
    }


def _valid_selector_metrics(path: Path, selector: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("selector") == selector
        and isinstance(payload.get("valid"), dict)
        and isinstance(payload.get("test"), dict)
    )


def _record_resume_event(run_dir: Path, *, event: str, **fields: Any) -> None:
    path = run_dir / "resume_state.json"
    state = _read_json(path)
    if state.get("schema_version") != RUN_MANIFEST_SCHEMA:
        state = {
            "schema_version": RUN_MANIFEST_SCHEMA,
            "resume_count": 0,
            "events": [],
        }
    if event == "resume_started":
        state["resume_count"] = int(state.get("resume_count", 0)) + 1
    events = list(state.get("events", []))
    events.append(
        {
            "event": event,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
    )
    state["events"] = events[-100:]
    _write_json_atomic(path, state)


def _write_root_marker(run_dir: Path, name: str, **payload: Any) -> None:
    _write_json_atomic(
        run_dir / name,
        {
            "schema_version": RUN_MANIFEST_SCHEMA,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **payload,
        },
    )


def is_cuda_oom_error(error: BaseException | str) -> bool:
    """Return whether an exception or message is a CUDA allocation failure."""
    # torch.cuda.OutOfMemoryError was added after some PyTorch releases still
    # used by collaborators.  Text matching remains the compatibility path.
    oom_exception = getattr(torch.cuda, "OutOfMemoryError", ())
    if isinstance(error, oom_exception):
        return True
    message = str(error).lower()
    return any(marker in message for marker in CUDA_OOM_MARKERS)


def _worker_reported_cuda_oom(log_path: Path, return_code: int) -> bool:
    if return_code == CUDA_OOM_EXIT_CODE:
        return True
    try:
        with log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 1024 * 1024))
            tail = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return False
    return is_cuda_oom_error(tail)


def _adaptive_profile_at(
    hardware_profile: dict[str, Any], tier: int
) -> dict[str, Any]:
    tiers = hardware_profile.get("tiers")
    if not hardware_profile.get("adaptive") or not isinstance(tiers, list):
        return dict(hardware_profile)
    if tier < 0 or tier >= len(tiers):
        raise ValueError(f"adaptive memory tier {tier} is out of range")
    return dict(tiers[tier])


def _adaptive_state_path(run_dir: Path) -> Path:
    return run_dir / "adaptive_memory_state.json"


def _load_or_create_adaptive_memory_state(
    run_dir: Path,
    hardware_profile: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any] | None:
    if not hardware_profile.get("adaptive"):
        return None
    path = _adaptive_state_path(run_dir)
    if path.is_file():
        state = _read_json(path)
        tier = state.get("current_tier")
        tiers = hardware_profile.get("tiers", [])
        if (
            state.get("schema_version") != ADAPTIVE_MEMORY_STATE_SCHEMA
            or not isinstance(tier, int)
            or tier < 0
            or tier >= len(tiers)
        ):
            raise ResumeCompatibilityError("invalid adaptive_memory_state.json")
        selectors = state.get("selectors")
        if selectors is not None:
            if not isinstance(selectors, dict):
                raise ResumeCompatibilityError("invalid adaptive selector state")
            for selector, selector_state in selectors.items():
                if not isinstance(selector_state, dict):
                    raise ResumeCompatibilityError(
                        f"invalid adaptive selector state for {selector}"
                    )
                selector_tier = selector_state.get("current_tier")
                if (
                    not isinstance(selector_tier, int)
                    or selector_tier < 0
                    or selector_tier >= len(tiers)
                ):
                    raise ResumeCompatibilityError(
                        f"invalid adaptive memory tier for selector {selector}"
                    )
        return state
    if resume:
        raise ResumeCompatibilityError(
            "adaptive resume requires the original adaptive_memory_state.json"
        )
    initial = _adaptive_profile_at(hardware_profile, 0)
    state = {
        "schema_version": ADAPTIVE_MEMORY_STATE_SCHEMA,
        "current_tier": 0,
        "current_profile": initial["name"],
        "pair_chunk_size": int(initial["pair_chunk_size"]),
        "activation_checkpointing": bool(initial["activation_checkpointing"]),
        "oom_retries": 0,
        "events": [],
    }
    _write_json_atomic(path, state)
    return state


def _adaptive_selector_state(
    run_dir: Path,
    state: dict[str, Any],
    hardware_profile: dict[str, Any],
    selector: str,
) -> dict[str, Any]:
    """Return one selector's adaptive tier, preserving legacy global state."""
    tiers = hardware_profile.get("tiers")
    if not hardware_profile.get("adaptive") or not isinstance(tiers, list):
        return {
            "current_tier": 0,
            "current_profile": hardware_profile.get("name", "config"),
            "oom_retries": 0,
        }
    selectors = state.setdefault("selectors", {})
    if not isinstance(selectors, dict):
        raise ResumeCompatibilityError("invalid adaptive selector state")
    existing = selectors.get(selector)
    if existing is not None:
        if not isinstance(existing, dict):
            raise ResumeCompatibilityError(
                f"invalid adaptive selector state for {selector}"
            )
        tier = existing.get("current_tier")
        if not isinstance(tier, int) or tier < 0 or tier >= len(tiers):
            raise ResumeCompatibilityError(
                f"invalid adaptive memory tier for selector {selector}"
            )
        return existing

    # A pre-selector-state run used one global tier. Seed newly observed selectors
    # from that tier so resuming an old run never silently upgrades memory usage.
    tier = state.get("current_tier", 0)
    if not isinstance(tier, int) or tier < 0 or tier >= len(tiers):
        raise ResumeCompatibilityError("invalid adaptive_memory_state.json")
    profile = _adaptive_profile_at(hardware_profile, tier)
    retries = sum(
        1
        for event in state.get("events", [])
        if isinstance(event, dict) and event.get("selector") == selector
    )
    created = {
        "current_tier": tier,
        "current_profile": profile["name"],
        "pair_chunk_size": int(profile["pair_chunk_size"]),
        "activation_checkpointing": bool(profile["activation_checkpointing"]),
        "oom_retries": retries,
    }
    selectors[selector] = created
    _write_json_atomic(_adaptive_state_path(run_dir), state)
    return created


def _record_adaptive_oom_retry(
    run_dir: Path,
    state: dict[str, Any],
    *,
    selector: str,
    gpu: int,
    from_tier: int,
    to_tier: int,
    hardware_profile: dict[str, Any],
) -> dict[str, Any]:
    previous = _adaptive_profile_at(hardware_profile, from_tier)
    selected = _adaptive_profile_at(hardware_profile, to_tier)
    selector_state = _adaptive_selector_state(
        run_dir, state, hardware_profile, selector
    )
    event = {
        "event": "oom_retry",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selector": selector,
        "gpu": gpu,
        "from_tier": from_tier,
        "from_profile": previous["name"],
        "to_tier": to_tier,
        "to_profile": selected["name"],
        "pair_chunk_size": int(selected["pair_chunk_size"]),
        "activation_checkpointing": bool(selected["activation_checkpointing"]),
    }
    events = list(state.get("events", []))
    events.append(event)
    selector_state.update(
        {
            "current_tier": to_tier,
            "current_profile": selected["name"],
            "pair_chunk_size": int(selected["pair_chunk_size"]),
            "activation_checkpointing": bool(selected["activation_checkpointing"]),
            "oom_retries": int(selector_state.get("oom_retries", 0)) + 1,
        }
    )
    selector_states = state.setdefault("selectors", {})
    selector_states[selector] = selector_state
    max_tier = max(
        int(item.get("current_tier", 0))
        for item in selector_states.values()
        if isinstance(item, dict)
    )
    max_profile = _adaptive_profile_at(hardware_profile, max_tier)
    state.update(
        {
            "current_tier": max_tier,
            "current_profile": max_profile["name"],
            "pair_chunk_size": int(max_profile["pair_chunk_size"]),
            "activation_checkpointing": bool(max_profile["activation_checkpointing"]),
            "oom_retries": int(state.get("oom_retries", 0)) + 1,
            "events": events[-100:],
        }
    )
    hardware_profile.update(
        {
            "current_tier": max_tier,
            "current_profile": max_profile["name"],
            "pair_chunk_size": int(max_profile["pair_chunk_size"]),
            "activation_checkpointing": bool(max_profile["activation_checkpointing"]),
        }
    )
    _write_json_atomic(_adaptive_state_path(run_dir), state)
    return event


def _run_resume_contract(
    *,
    config_path: Path,
    config: dict[str, Any],
    seed: int,
    data_dir: Path,
    hardware_profile: dict[str, Any],
    model_parallel_gpu_ids: list[int],
) -> dict[str, Any]:
    return {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "config": file_contract(config_path),
        "source": _source_contract(),
        "training_semantics": {
            "seed": seed,
            "selectors": list(config["selectors"]),
            "candidate": config["candidate"],
            "matched_control": config["matched_control"],
            "baseline": config["baseline"],
            "kernel": config["kernel"],
            "model": config["model"],
            **(
                {"memory_strategy": "adaptive"}
                if hardware_profile.get("adaptive")
                else {
                    "pair_chunk_size": int(hardware_profile["pair_chunk_size"]),
                    "activation_checkpointing": bool(
                        hardware_profile["activation_checkpointing"]
                    ),
                }
            ),
            "parallel_mode": "model_parallel" if model_parallel_gpu_ids else "selector_or_serial",
            "model_parallel_gpu_ids": list(model_parallel_gpu_ids),
            "selector_gpu_ids": list(hardware_profile["selected_gpu_ids"]),
        },
        "data": {
            split: file_contract(data_dir / f"{split}.jsonl")
            for split in ("train", "valid", "test")
        },
        "materialization": file_contract(data_dir / "data_manifest.json"),
    }


def _validate_or_create_run_manifest(
    run_dir: Path,
    contract: dict[str, Any],
    *,
    resume: bool,
    started_at_utc: str,
) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    contract_fingerprint = fingerprint(contract)
    if resume:
        persisted = _read_json(path)
        if persisted.get("schema_version") != RUN_MANIFEST_SCHEMA:
            raise ResumeCompatibilityError("unsupported run manifest schema")
        if persisted.get("contract_fingerprint") != contract_fingerprint:
            raise ResumeCompatibilityError(
                "resume contract differs: code, config, data, selector or training settings changed"
            )
        if persisted.get("started_at_utc") != started_at_utc:
            raise ResumeCompatibilityError(
                "resume timestamp differs from the original run manifest"
            )
        return persisted
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "contract_fingerprint": contract_fingerprint,
        "contract": contract,
        "started_at_utc": started_at_utc,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(
        path,
        manifest,
    )
    return manifest


def _write_data_manifest(
    data_dir: Path, split_info: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    manifest = {
        "schema_version": DATA_MANIFEST_SCHEMA,
        "splits": {
            split: {
                **info,
                "materialized": file_contract(data_dir / f"{split}.jsonl"),
            }
            for split, info in split_info.items()
        },
    }
    _write_json_atomic(data_dir / "data_manifest.json", manifest)
    return manifest


def _load_data_manifest(data_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = data_dir / "data_manifest.json"
    manifest = _read_json(path)
    splits = manifest.get("splits")
    if manifest.get("schema_version") != DATA_MANIFEST_SCHEMA or not isinstance(
        splits, dict
    ):
        raise ResumeCompatibilityError(
            "resume requires the original data_manifest.json; legacy directories can only restart from baseline"
        )
    restored: dict[str, dict[str, Any]] = {}
    for split in ("train", "valid", "test"):
        info = splits.get(split)
        if not isinstance(info, dict) or not isinstance(info.get("materialized"), dict):
            raise ResumeCompatibilityError(
                f"data manifest is missing materialized provenance for {split}"
            )
        actual = file_contract(data_dir / f"{split}.jsonl")
        if actual != info["materialized"]:
            raise ResumeCompatibilityError(
                f"materialized {split} data differs from its immutable data manifest"
            )
        restored[split] = dict(info)
    return manifest, restored


def parse_model_parallel_gpu_ids(spec: str | None) -> list[int]:
    if spec is None:
        return []
    fields = [field.strip() for field in spec.split(",")]
    if len(fields) < 2 or any(not field.isdigit() for field in fields):
        raise ValueError("--model-parallel-gpus must contain at least two GPU IDs")
    ids = [int(field) for field in fields]
    if len(set(ids)) != len(ids):
        raise ValueError("--model-parallel-gpus must not contain duplicate GPU IDs")
    return ids


def local_model_parallel_devices(physical_gpu_ids: list[int]) -> tuple[torch.device, ...]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        fields = [field.strip() for field in visible.split(",")]
        if all(field.isdigit() for field in fields):
            mapping = {int(field): index for index, field in enumerate(fields)}
            missing = [gpu_id for gpu_id in physical_gpu_ids if gpu_id not in mapping]
            if missing:
                raise ValueError(
                    f"model-parallel GPUs {missing} are not visible in CUDA_VISIBLE_DEVICES={visible}"
                )
            return tuple(torch.device(f"cuda:{mapping[gpu_id]}") for gpu_id in physical_gpu_ids)
    available = torch.cuda.device_count()
    if any(gpu_id >= available for gpu_id in physical_gpu_ids):
        raise ValueError(
            f"model-parallel GPU IDs {physical_gpu_ids} exceed visible device count {available}"
        )
    return tuple(torch.device(f"cuda:{gpu_id}") for gpu_id in physical_gpu_ids)


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
    if requested == "adaptive":
        tiers = [dict(profile) for profile in ADAPTIVE_HARDWARE_PROFILES]
        return {
            **tiers[0],
            "name": "adaptive",
            "adaptive": True,
            "tiers": tiers,
            "selection_reason": (
                "start with the largest streamed-backward pair chunk and "
                "fall back only after a confirmed CUDA OOM"
            ),
        }
    if requested == "config":
        return {
            "name": "config",
            "pair_chunk_size": int(config["kernel"].get("pair_chunk_size", 256)),
            "activation_checkpointing": True,
            "adaptive": False,
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
                "adaptive": False,
                "selection_reason": profile.pop("reason"),
                "minimum_memory_total_mib": min_total,
                "minimum_memory_free_mib": min_free,
            }
        )
        return profile
    profile = dict(HARDWARE_PROFILES[requested])
    profile.update(
        {
            "name": requested,
            "adaptive": False,
            "selection_reason": "explicit profile override",
        }
    )
    profile.pop("reason", None)
    return profile


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _append_scheduler_event(run_dir: Path, payload: dict[str, Any]) -> None:
    """Keep machine-readable scheduler events separate from the human console UI."""
    event_path = run_dir / "scheduler_events.jsonl"
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()


def _read_json(path: Path) -> dict[str, Any]:
    """Read a heartbeat snapshot, tolerating startup races and stale files."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _format_duration(seconds: Any) -> str:
    if seconds is None:
        return "--:--"
    try:
        total = max(int(round(float(seconds))), 0)
    except (TypeError, ValueError):
        return "--:--"
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _render_selector_dashboard(
    statuses: dict[str, dict[str, Any]],
    active: dict[str, dict[str, Any]],
) -> str:
    """Render a readable, append-only snapshot for serial and multi-GPU runs."""
    total = len(statuses)
    counts = {
        state: sum(item.get("status") == state for item in statuses.values())
        for state in ("complete", "running", "pending", "failed", "not_started")
    }
    lines = [
        (
            f"Q-TRIAD selectors: {counts['complete']}/{total} complete | "
            f"{counts['running']} running | {counts['pending']} queued | "
            f"{counts['not_started']} not started | {counts['failed']} failed"
        )
    ]
    for selector, item in sorted(statuses.items(), key=lambda pair: (str(pair[1].get("gpu", "?")), pair[0])):
        status = str(item.get("status", "pending")).upper()
        gpu = item.get("gpu") if item.get("gpu") is not None else "-"
        if status != "RUNNING":
            lines.append(f"GPU {gpu} | {selector:<28} | {status.lower()}")
            continue
        heartbeat_value = item.get("heartbeat_file")
        heartbeat = _read_json(Path(str(heartbeat_value))) if heartbeat_value else {}
        progress: list[str] = []
        phase = heartbeat.get("phase") or heartbeat.get("stage")
        if phase:
            progress.append(str(phase))
        epoch, epochs = heartbeat.get("epoch"), heartbeat.get("epochs")
        if epoch is not None and epochs is not None:
            progress.append(f"epoch {epoch}/{epochs}")
        batch, batches = heartbeat.get("batch"), heartbeat.get("batches")
        percent = heartbeat.get("percent")
        if batch is not None and batches is not None:
            try:
                suffix = f" {float(percent):.1f}%" if percent is not None else ""
            except (TypeError, ValueError):
                suffix = ""
            progress.append(f"batch {batch}/{batches}{suffix}")
        if not progress:
            progress.append("starting")
        eta = _format_duration(heartbeat.get("eta_seconds"))
        rate = heartbeat.get("batches_per_second")
        try:
            rate_text = f" | {float(rate):.2f} batch/s" if rate is not None else ""
        except (TypeError, ValueError):
            rate_text = ""
        progress.append(f"ETA {eta}{rate_text}")
        memory_text = format_gpu_memory(heartbeat.get("gpu_memory"))
        if memory_text:
            progress.append(memory_text)
        lines.append(f"GPU {gpu} | {selector:<28} | " + " | ".join(progress))

    for label, state in (
        ("Completed", "complete"),
        ("Queued", "pending"),
        ("Not started", "not_started"),
        ("Failed", "failed"),
    ):
        names = [name for name, item in statuses.items() if item.get("status") == state]
        lines.append(f"{label}: {', '.join(names) if names else 'none'}")
    if active:
        updated = []
        for selector, entry in active.items():
            elapsed = time.monotonic() - float(entry["started_monotonic"])
            updated.append(f"{selector} {_format_duration(elapsed)}")
        lines.append("Active time: " + ", ".join(sorted(updated)))
    return "\n".join(lines)


def _render_baseline_line(line: str, *, epochs: int) -> str | None:
    """Render one baseline JSON event without losing the original log line."""
    stripped = line.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return f"[baseline] {stripped}"
    if not isinstance(payload, dict):
        return f"[baseline] {stripped}"
    event = payload.get("event")
    phase = payload.get("phase")
    epoch = payload.get("epoch")
    epoch_text = f" epoch {epoch}/{epochs}" if epoch is not None else ""
    label = f"[baseline]" + (f"[{phase}]" if phase else "")
    if event == "phase_start":
        return f"{label}{epoch_text} started | batches={payload.get('batches', '?')}"
    if event == "batch_progress":
        try:
            percent = float(payload.get("percent", 0.0))
        except (TypeError, ValueError):
            percent = 0.0
        width = 24
        filled = min(max(int(round(width * percent / 100.0)), 0), width)
        bar = "#" * filled + "-" * (width - filled)
        try:
            rate = f" | {float(payload['batches_per_second']):.2f} batch/s"
        except (KeyError, TypeError, ValueError):
            rate = ""
        memory_text = format_gpu_memory(payload.get("gpu_memory"))
        if memory_text:
            memory_text = f" | {memory_text}"
        return (
            f"{label}{epoch_text} [{bar}] {percent:5.1f}% "
            f"batch {payload.get('batch', '?')}/{payload.get('batches', '?')} | "
            f"elapsed {_format_duration(payload.get('elapsed_seconds'))} | "
            f"ETA {_format_duration(payload.get('eta_seconds'))}{rate}{memory_text}"
        )
    if event == "phase_complete":
        return (
            f"{label}{epoch_text} complete | "
            f"batches={payload.get('completed_batches', '?')} | "
            f"elapsed {_format_duration(payload.get('elapsed_seconds'))}"
        )
    if event == "health_warning":
        warning = payload.get("warning") or payload.get("message") or "health warning"
        return f"[baseline] warning | {warning}"
    if "epoch" in payload and isinstance(payload.get("valid"), dict):
        valid = payload["valid"]
        return (
            f"[baseline] epoch {payload['epoch']}/{epochs} complete | "
            f"train_loss={float(payload.get('train_loss', 0.0)):.4f} | "
            f"valid_loss={float(valid.get('loss', 0.0)):.4f} | "
            f"valid_macro_f1={float(valid.get('macro_f1', 0.0)):.4f}"
        )
    if "output_dir" in payload and "best_valid" in payload:
        best_valid = payload.get("best_valid") or {}
        return (
            "[baseline] complete | "
            f"best_valid_macro_f1={float(best_valid.get('macro_f1', 0.0)):.4f}"
        )
    return f"[baseline] event={event or 'json'}"


def _run_baseline_logged_command(
    command: list[str],
    log_path: Path,
    heartbeat_path: Path,
    *,
    epochs: int,
    pause: PauseController | None = None,
    append_log: bool = False,
) -> dict[str, Any]:
    """Run baseline with a readable console while retaining its raw JSON log."""
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "Q_ATTENTION_PROGRESS_FORMAT": "json",
            "Q_ATTENTION_HEARTBEAT_FILE": str(heartbeat_path),
        }
    )
    heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[baseline] starting | epochs={epochs}", flush=True)
    started = time.perf_counter()
    with log_path.open("a" if append_log else "w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None
        reader_error: list[BaseException] = []

        def read_output() -> None:
            try:
                for line in process.stdout:
                    log_handle.write(line)
                    log_handle.flush()
                    rendered = _render_baseline_line(line, epochs=epochs)
                    if rendered is not None:
                        print(rendered, flush=True)
            except BaseException as exc:  # surfaced after the child is joined
                reader_error.append(exc)

        reader = threading.Thread(target=read_output, name="baseline-output", daemon=True)
        reader.start()
        pause_forwarded = False
        pause_deadline: float | None = None
        while process.poll() is None:
            if pause is not None and pause.requested and not pause_forwarded:
                pause_forwarded = True
                pause_deadline = time.monotonic() + SAFE_PAUSE_TIMEOUT_SECONDS
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            if pause_deadline is not None and time.monotonic() >= pause_deadline:
                _terminate_worker(process)
                reader.join(timeout=30)
                raise RuntimeError(
                    "baseline did not reach a post-update checkpoint before the safe-pause timeout"
                )
            time.sleep(0.1)
        return_code = process.wait()
        reader.join(timeout=30)
        if reader.is_alive():
            _terminate_worker(process)
            raise RuntimeError("baseline output reader did not finish")
        if reader_error:
            raise RuntimeError(f"baseline output reader failed: {reader_error[0]}")
    result = {
        "command": command,
        "returncode": return_code,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "log_path": str(log_path),
    }
    if return_code == PAUSED_EXIT_CODE:
        raise RunPaused("baseline paused at a post-update checkpoint")
    if return_code != 0:
        print(f"[baseline] failed | exit_code={return_code}", file=sys.stderr, flush=True)
        raise RuntimeError(f"command failed with return code {return_code}: {command}")
    print(f"[baseline] process complete | elapsed {_format_duration(result['duration_seconds'])}", flush=True)
    return result


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
    resume: bool = False,
    pause: PauseController | None = None,
    adaptive_memory_state: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run independent selectors with one dynamically scheduled worker per GPU."""
    hardware_profile = hardware_profile or {
        "name": "config",
        "pair_chunk_size": 256,
        "activation_checkpointing": True,
        "adaptive": False,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    adaptive = bool(hardware_profile.get("adaptive"))
    adaptive_memory_state = adaptive_memory_state or _load_or_create_adaptive_memory_state(
        run_dir, hardware_profile, resume=resume
    )
    if adaptive_memory_state is not None:
        # Initialize all selectors before scheduling. This prevents one worker's
        # OOM from making an unstarted independent worker inherit its lower tier.
        for selector in selectors:
            _adaptive_selector_state(
                run_dir, adaptive_memory_state, hardware_profile, selector
            )
    pending = list(selectors)
    available = list(gpu_ids)
    statuses: dict[str, dict[str, Any]] = {
        selector: {"selector": selector, "status": "pending", "gpu": None}
        for selector in selectors
    }
    if resume:
        for selector in list(pending):
            metrics_path = run_dir / "selectors" / selector / "metrics.json"
            if _valid_selector_metrics(metrics_path, selector):
                statuses[selector].update(
                    {"status": "complete", "gpu": None, "resumed_skip": True}
                )
                pending.remove(selector)
    active: dict[str, dict[str, Any]] = {}
    assignments_path = run_dir / "gpu_assignments.json"
    _write_json_atomic(
        assignments_path,
        {"requested_gpu_ids": gpu_ids, "hardware_profile": hardware_profile, "workers": statuses},
    )
    dashboard_at = 0.0

    def write_assignments() -> None:
        _write_json_atomic(
            assignments_path,
            {"requested_gpu_ids": gpu_ids, "hardware_profile": hardware_profile, "workers": statuses},
        )

    def fail_run(reason: str) -> None:
        for entry in active.values():
            _terminate_worker(entry["process"])
            entry["handle"].close()
        now = datetime.now(timezone.utc).isoformat()
        for selector in pending:
            statuses[selector].update({"status": "not_started", "finished_at": now})
        write_assignments()
        _write_root_marker(run_dir, "RUN_FAILED", reason=reason, workers=statuses)

    def pause_run(reason: str) -> None:
        """Ask every active worker to save a post-update checkpoint before exit."""
        for entry in active.values():
            process = entry["process"]
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

        deadline = time.monotonic() + SAFE_PAUSE_TIMEOUT_SECONDS
        while active:
            for selector, entry in list(active.items()):
                process = entry["process"]
                return_code = process.poll()
                if return_code is None:
                    continue
                entry["handle"].close()
                finished_at = datetime.now(timezone.utc).isoformat()
                duration = round(time.monotonic() - entry["started_monotonic"], 3)
                if return_code not in (0, PAUSED_EXIT_CODE):
                    statuses[selector].update(
                        {
                            "status": "failed",
                            "return_code": return_code,
                            "finished_at": finished_at,
                            "duration_seconds": duration,
                        }
                    )
                    del active[selector]
                    for sibling in active.values():
                        _terminate_worker(sibling["process"])
                        sibling["handle"].close()
                    raise RuntimeError(
                        f"selector worker {selector} failed while pausing with exit code {return_code}"
                    )
                statuses[selector].update(
                    {
                        "status": "complete" if return_code == 0 else "paused",
                        "return_code": return_code,
                        "finished_at": finished_at,
                        "duration_seconds": duration,
                    }
                )
                del active[selector]
            if active:
                if time.monotonic() >= deadline:
                    for entry in active.values():
                        _terminate_worker(entry["process"])
                        entry["handle"].close()
                    raise RuntimeError(
                        "selector workers did not reach a post-update checkpoint before the safe-pause timeout"
                    )
                time.sleep(0.1)

        now = datetime.now(timezone.utc).isoformat()
        for selector in pending:
            statuses[selector].update({"status": "paused", "finished_at": now})
        write_assignments()
        _write_root_marker(run_dir, "RUN_PAUSED", reason=reason, workers=statuses)

    try:
        while pending or active:
            if pause is not None and pause.requested:
                pause_run(pause.reason or "requested")
                raise RunPaused("selector scheduler pause requested")
            while pending and available and not (pause is not None and pause.requested):
                selector = pending.pop(0)
                gpu_id = available.pop(0)
                selector_dir = run_dir / "selectors" / selector
                selector_dir.mkdir(parents=True, exist_ok=True)
                has_checkpoint = (selector_dir / "checkpoints" / "latest.pt").is_file()
                if resume and not has_checkpoint and any(selector_dir.iterdir()):
                    raise ResumeCompatibilityError(
                        f"selector {selector} has partial artifacts but no batch checkpoint; refusing unsafe restart"
                    )
                worker_resume = has_checkpoint
                tier = (
                    int(
                        _adaptive_selector_state(
                            run_dir, adaptive_memory_state, hardware_profile, selector
                        )["current_tier"]
                    )
                    if adaptive
                    else 0
                )
                selected_profile = _adaptive_profile_at(hardware_profile, tier)
                log_handle = (selector_dir / "worker.log").open(
                    "a" if worker_resume else "w", encoding="utf-8"
                )
                heartbeat_path = selector_dir / "heartbeat.json"
                heartbeat_path.touch()
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
                    "--checkpoint-every-batches", str(getattr(args, "checkpoint_every_batches", 50)),
                    "--pair-chunk-size", str(selected_profile["pair_chunk_size"]),
                    "--activation-checkpointing", str(int(selected_profile["activation_checkpointing"])),
                ]
                if adaptive:
                    command.append("--adaptive-memory")
                if worker_resume:
                    command.append("--resume")
                environment = os.environ.copy()
                if gpu_id >= 0:
                    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                environment["PYTHONUNBUFFERED"] = "1"
                environment["Q_ATTENTION_PROGRESS_FORMAT"] = "json"
                environment["Q_ATTENTION_HEARTBEAT_FILE"] = str(heartbeat_path)
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
                    "heartbeat_file": heartbeat_path,
                    "adaptive_tier": tier,
                    "memory_profile": selected_profile["name"],
                }
                statuses[selector].update(
                    {
                        "status": "running",
                        "gpu": gpu_id,
                        "pid": process.pid,
                        "started_at": started_at,
                        "heartbeat_file": str(heartbeat_path),
                        "log_file": str(selector_dir / "worker.log"),
                        "adaptive_tier": tier if adaptive else None,
                        "memory_profile": selected_profile["name"],
                        "pair_chunk_size": int(selected_profile["pair_chunk_size"]),
                        "activation_checkpointing": bool(selected_profile["activation_checkpointing"]),
                        "oom_retries": (
                            int(
                                _adaptive_selector_state(
                                    run_dir,
                                    adaptive_memory_state,
                                    hardware_profile,
                                    selector,
                                ).get("oom_retries", 0)
                            )
                            if adaptive
                            else int(statuses[selector].get("oom_retries", 0))
                        ),
                    }
                )
                write_assignments()
                _append_scheduler_event(
                    run_dir,
                    {
                        "event": "selector_started",
                        "selector": selector,
                        "gpu": gpu_id,
                        "pid": process.pid,
                        "adaptive_tier": tier if adaptive else None,
                        "memory_profile": selected_profile["name"],
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    },
                )
                print(f"[selector-scheduler] started {selector} on GPU {gpu_id}", flush=True)

            for selector, entry in list(active.items()):
                process = entry["process"]
                return_code = process.poll()
                if return_code is None:
                    continue
                entry["handle"].close()
                finished_at = datetime.now(timezone.utc).isoformat()
                duration = round(time.monotonic() - entry["started_monotonic"], 3)
                if return_code == PAUSED_EXIT_CODE:
                    statuses[selector].update(
                        {"status": "paused", "return_code": return_code, "finished_at": finished_at, "duration_seconds": duration}
                    )
                    del active[selector]
                    pause_run(f"selector worker {selector} paused")
                    raise RunPaused(f"selector worker {selector} paused")
                if return_code != 0:
                    log_path = run_dir / "selectors" / selector / "worker.log"
                    oom = adaptive and _worker_reported_cuda_oom(log_path, return_code)
                    checkpoint_path = (
                        run_dir / "selectors" / selector / "checkpoints" / "latest.pt"
                    )
                    tier_used = int(entry["adaptive_tier"])
                    next_tier = tier_used + 1
                    if oom and checkpoint_path.is_file() and next_tier < len(hardware_profile["tiers"]):
                        event = _record_adaptive_oom_retry(
                            run_dir,
                            adaptive_memory_state,
                            selector=selector,
                            gpu=int(entry["gpu"]),
                            from_tier=tier_used,
                            to_tier=next_tier,
                            hardware_profile=hardware_profile,
                        )
                        retries = int(statuses[selector].get("oom_retries", 0)) + 1
                        statuses[selector].update(
                            {
                                "status": "pending",
                                "gpu": None,
                                "last_return_code": return_code,
                                "oom_retries": retries,
                                "last_oom_at": finished_at,
                                "last_memory_profile": entry["memory_profile"],
                            }
                        )
                        available.append(int(entry["gpu"]))
                        del active[selector]
                        pending.insert(0, selector)
                        write_assignments()
                        _append_scheduler_event(run_dir, event)
                        selected = _adaptive_profile_at(hardware_profile, next_tier)
                        print(
                            f"[selector-scheduler] CUDA OOM on {selector} | retry {retries} | "
                            f"profile {selected['name']} | pair_chunk_size={selected['pair_chunk_size']}",
                            flush=True,
                        )
                        continue
                    statuses[selector].update(
                        {
                            "status": "failed",
                            "return_code": return_code,
                            "last_return_code": return_code,
                            "finished_at": finished_at,
                            "duration_seconds": duration,
                            "oom_retries": int(statuses[selector].get("oom_retries", 0)),
                        }
                    )
                    write_assignments()
                    if oom and not checkpoint_path.is_file():
                        reason = (
                            f"selector worker {selector} hit CUDA OOM without a batch checkpoint; "
                            "unsafe adaptive restart refused"
                        )
                    elif oom:
                        reason = (
                            f"selector worker {selector} exhausted all adaptive memory tiers "
                            f"after CUDA OOM (exit code {return_code})"
                        )
                    else:
                        reason = f"selector worker {selector} failed with exit code {return_code}"
                    fail_run(reason)
                    raise RuntimeError(
                        f"{reason}; inspect {run_dir / 'selectors' / selector / 'worker.log'}"
                    )
                metrics_path = run_dir / "selectors" / selector / "metrics.json"
                if not _valid_selector_metrics(metrics_path, selector):
                    statuses[selector].update(
                        {"status": "failed", "return_code": return_code, "finished_at": finished_at, "duration_seconds": duration}
                    )
                    fail_run(f"selector worker {selector} exited without valid metrics.json")
                    raise RuntimeError(f"selector worker {selector} produced no valid metrics.json")
                statuses[selector].update(
                    {"status": "complete", "return_code": 0, "finished_at": finished_at, "duration_seconds": duration}
                )
                available.append(int(entry["gpu"]))
                del active[selector]
                write_assignments()
                _append_scheduler_event(
                    run_dir,
                    {
                        "event": "selector_complete",
                        "selector": selector,
                        "gpu": entry["gpu"],
                        "duration_seconds": duration,
                        "timestamp": finished_at,
                    },
                )
                print(
                    f"[selector-scheduler] complete {selector} on GPU {entry['gpu']} "
                    f"in {_format_duration(duration)}",
                    flush=True,
                )

            now = time.monotonic()
            if now - dashboard_at >= 30.0:
                dashboard_at = now
                counts = {state: sum(item["status"] == state for item in statuses.values()) for state in ("pending", "running", "complete", "failed", "not_started")}
                _append_scheduler_event(
                    run_dir,
                    {
                        "event": "selector_dashboard",
                        **counts,
                        "active_gpus": {
                            selector: item["gpu"]
                            for selector, item in statuses.items()
                            if item["status"] == "running"
                        },
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    },
                )
                print(_render_selector_dashboard(statuses, active), flush=True)
            if active:
                time.sleep(0.5)
    except RunPaused:
        raise
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
    model_parallel_devices: tuple[torch.device, ...] = (),
) -> QTriadAttentionScoreKernel:
    kernel_config = config["kernel"]
    kernel = QTriadAttentionScoreKernel(
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
    if model_parallel_devices:
        kernel.configure_model_parallel(model_parallel_devices)
    return kernel


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


def _run(args: argparse.Namespace, pause: PauseController) -> int:
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
    if args.checkpoint_every_batches <= 0:
        raise ValueError("--checkpoint-every-batches must be positive")
    model_parallel_gpu_ids = parse_model_parallel_gpu_ids(args.model_parallel_gpus)
    if model_parallel_gpu_ids and args.gpus:
        raise ValueError("--gpus/selector-parallel cannot be combined with --model-parallel-gpus")
    if model_parallel_gpu_ids:
        if args.device == "cpu":
            raise ValueError("model parallelism requires CUDA")
        inventory = query_gpu_inventory()
        validate_gpu_capacity(model_parallel_gpu_ids, inventory, phase="before model-parallel baseline")
        model_parallel_devices = local_model_parallel_devices(model_parallel_gpu_ids)
        gpu_ids: list[int] = []
        profile_gpu_ids = model_parallel_gpu_ids
    else:
        model_parallel_devices = ()
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
            inventory = query_gpu_inventory()
            validate_gpu_capacity(gpu_ids, inventory, phase="before baseline")
        profile_gpu_ids = gpu_ids
    profile_request = "auto" if args.gpus and args.gpus.strip().lower() == "auto" and args.hardware_profile == "config" else args.hardware_profile
    hardware_profile = choose_hardware_profile(profile_request, config, profile_gpu_ids, inventory)
    hardware_profile.update(
        {
            "requested_gpu_spec": args.model_parallel_gpus or args.gpus or "default",
            "selected_gpu_ids": profile_gpu_ids,
            "gpu_inventory": inventory,
        }
    )
    device = model_parallel_devices[0] if model_parallel_devices else (torch.device("cuda:0") if gpu_ids else choose_device(args.device))
    if args.resume is not None and args.output_dir is not None:
        raise ValueError("--resume cannot be combined with --output-dir")
    provisional_stamp = args.started_at_utc or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.resume or args.output_dir or ROOT / "runs" / "retacred_qtriad_formal_single_seed" / f"{provisional_stamp}_seed13"
    run_dir = resolve_path(run_dir)
    resuming = args.resume is not None
    if resuming:
        if not run_dir.is_dir() or not (run_dir / "run_manifest.json").is_file():
            raise ResumeCompatibilityError(
                "--resume requires an existing run_manifest.json; legacy directories can only restart from baseline"
            )
        if (run_dir / "RUN_COMPLETE").exists():
            raise ResumeCompatibilityError("run is already complete")
        original_manifest = _read_json(run_dir / "run_manifest.json")
        stamp = original_manifest.get("started_at_utc")
        if not isinstance(stamp, str) or len(stamp) != 16 or not stamp.endswith("Z"):
            raise ResumeCompatibilityError("run manifest is missing the original startup timestamp")
        if args.started_at_utc is not None and args.started_at_utc != stamp:
            raise ResumeCompatibilityError("--started-at-utc must match the original run when resuming")
    else:
        stamp = provisional_stamp
        if len(stamp) != 16 or not stamp.endswith("Z"):
            raise ValueError("--started-at-utc must use UTC format YYYYMMDDTHHMMSSZ")
        run_dir.mkdir(parents=True, exist_ok=False)
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_source = resolve_path(config["train_path"])
    valid_source = resolve_path(config["valid_path"])
    test_source = resolve_path(config["test_path"])
    if resuming:
        if not all((data_dir / f"{split}.jsonl").is_file() for split in ("train", "valid", "test")):
            raise ResumeCompatibilityError("resume requires original data/train.jsonl, valid.jsonl and test.jsonl")
        _, data_info = _load_data_manifest(data_dir)
        train_info = data_info["train"]
        valid_info = data_info["valid"]
        test_info = data_info["test"]
    else:
        valid_info = materialize_subset(valid_source, data_dir / "valid.jsonl", 0, seed=seed + 101, split="valid")
        valid_records = load_relation_jsonl(data_dir / "valid.jsonl")
        required_labels = {record.label for record in valid_records}
        train_info = materialize_subset(train_source, data_dir / "train.jsonl", 0, seed=seed, split="train", required_labels=required_labels)
        test_info = materialize_subset(test_source, data_dir / "test.jsonl", 0, seed=seed + 211, split="test")
        _write_data_manifest(
            data_dir,
            {"train": train_info, "valid": valid_info, "test": test_info},
        )
    valid_records = load_relation_jsonl(data_dir / "valid.jsonl")
    train_records = load_relation_jsonl(data_dir / "train.jsonl")
    test_records = load_relation_jsonl(data_dir / "test.jsonl")
    expected = config["expected_records"]
    for name, info in (("train", train_info), ("valid", valid_info), ("test", test_info)):
        if int(info["records"]) != int(expected[name]):
            raise ValueError(f"{name} record count differs from frozen contract")
    run_contract = _run_resume_contract(
        config_path=config_path,
        config=config,
        seed=seed,
        data_dir=data_dir,
        hardware_profile=hardware_profile,
        model_parallel_gpu_ids=model_parallel_gpu_ids,
    )
    if model_parallel_gpu_ids and hardware_profile.get("adaptive"):
        raise ValueError(
            "adaptive hardware profile is currently supported for selector workers only; "
            "use a fixed profile for --model-parallel-gpus"
        )
    if hardware_profile.get("adaptive"):
        hardware_profile.setdefault("current_tier", 0)
        hardware_profile.setdefault("current_profile", hardware_profile["name"])
    _validate_or_create_run_manifest(
        run_dir, run_contract, resume=resuming, started_at_utc=stamp
    )
    adaptive_memory_state = _load_or_create_adaptive_memory_state(
        run_dir, hardware_profile, resume=resuming
    )
    if adaptive_memory_state is not None:
        current_profile = _adaptive_profile_at(
            hardware_profile, int(adaptive_memory_state["current_tier"])
        )
        hardware_profile.update(
            {
                "current_tier": int(adaptive_memory_state["current_tier"]),
                "current_profile": current_profile["name"],
                "pair_chunk_size": int(current_profile["pair_chunk_size"]),
                "activation_checkpointing": bool(
                    current_profile["activation_checkpointing"]
                ),
            }
        )
    args._run_dir = run_dir
    if resuming:
        _record_resume_event(run_dir, event="resume_started", contract_fingerprint=fingerprint(run_contract))
        (run_dir / "RUN_PAUSED").unlink(missing_ok=True)
        (run_dir / "RUN_FAILED").unlink(missing_ok=True)
    max_length = max(len(record.tokens) for record in train_records + valid_records + test_records) + 4
    model_config = config["model"]
    baseline_dir = run_dir / "baseline"
    baseline_command = [
        args.python_bin,
        str(ROOT / "experiments" / "train_relation_baseline.py"),
        "--train_path", str(data_dir / "train.jsonl"),
        "--valid_path", str(data_dir / "valid.jsonl"),
        "--output_dir", str(baseline_dir),
        "--device", "cuda" if (gpu_ids or model_parallel_gpu_ids) else str(device),
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
    if model_parallel_gpu_ids:
        baseline_command.extend(
            ["--model-parallel-gpus", ",".join(str(gpu_id) for gpu_id in model_parallel_gpu_ids)]
        )
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_complete = all(
        (baseline_dir / name).is_file()
        for name in ("metrics.json", "model.pt", "vocab.json", "labels.json")
    )
    if not baseline_complete:
        if resuming and any(baseline_dir.iterdir()) and not (baseline_dir / "checkpoints" / "latest.pt").is_file():
            raise ResumeCompatibilityError(
                "baseline has partial artifacts but no batch checkpoint; refusing unsafe restart"
            )
        if (baseline_dir / "checkpoints" / "latest.pt").is_file():
            baseline_command.append("--resume")
        baseline_command.extend(["--checkpoint-every-batches", str(args.checkpoint_every_batches)])
        _run_baseline_logged_command(
            baseline_command,
            run_dir / "baseline_train.log",
            baseline_dir / "heartbeat.json",
            epochs=int(config["baseline"]["epochs"]),
            pause=pause,
            append_log=resuming,
        )
    artifacts = load_relation_run(
        baseline_dir,
        device,
        model_parallel_devices=model_parallel_devices,
    )
    valid_loader = make_relation_loader(valid_records, artifacts.vocab, artifacts.label_to_id, batch_size=int(config["kernel"]["batch_size"]))
    test_loader = make_relation_loader(test_records, artifacts.vocab, artifacts.label_to_id, batch_size=int(config["kernel"]["batch_size"]))
    for parameter in artifacts.model.parameters():
        parameter.requires_grad_(False)
    baseline_eval_path = run_dir / "baseline_eval.json"
    if baseline_eval_path.is_file():
        baseline_eval = json.loads(baseline_eval_path.read_text(encoding="utf-8"))
        baseline_valid, baseline_test = baseline_eval["valid"], baseline_eval["test"]
    else:
        baseline_valid = evaluate_selector(artifacts.model, valid_loader, device, len(artifacts.label_to_id), None, "baseline_valid")
        baseline_test = evaluate_selector(artifacts.model, test_loader, device, len(artifacts.label_to_id), None, "baseline_test")
        baseline_eval_path.write_text(
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
    if not (disabled_dir / "metrics.json").is_file():
        (disabled_dir / "metrics.json").write_text(
            json.dumps(disabled_row, indent=2, sort_keys=True), encoding="utf-8"
        )
    if model_parallel_devices:
        # Model-parallel mode keeps one sharded model alive and runs selectors
        # serially; independent selector workers would each duplicate the
        # sharded model and defeat the memory purpose of this option.
        worker_statuses = {}
        train_args = argparse.Namespace(
            batch_size=int(config["kernel"]["batch_size"]),
            epochs=int(config["kernel"]["epochs"]),
            kernel_lr=float(config["kernel"]["lr"]),
            log_every_batches=args.log_every_batches,
            batch_resume=True,
            checkpoint_every_batches=args.checkpoint_every_batches,
            pause=pause,
        )
        pending_selectors = [selector for selector in selectors if selector != "disabled"]
        for selector in pending_selectors:
            worker_statuses[selector] = {
                "selector": selector,
                "status": "pending",
                "physical_gpu_ids": model_parallel_gpu_ids,
            }
        _write_json_atomic(
            run_dir / "gpu_assignments.json",
            {
                "parallel_mode": "model_parallel",
                "requested_gpu_ids": model_parallel_gpu_ids,
                "hardware_profile": hardware_profile,
                "workers": worker_statuses,
            },
        )
        current_selector: str | None = None
        try:
            for selector in pending_selectors:
                selector_dir = run_dir / "selectors" / selector
                if resuming and _valid_selector_metrics(selector_dir / "metrics.json", selector):
                    worker_statuses[selector].update({"status": "complete", "resumed_skip": True})
                    continue
                if (
                    resuming
                    and selector_dir.exists()
                    and any(selector_dir.iterdir())
                    and not (selector_dir / "checkpoints" / "latest.pt").is_file()
                ):
                    raise ResumeCompatibilityError(
                        f"selector {selector} has partial artifacts but no batch checkpoint; refusing unsafe restart"
                    )
                current_selector = selector
                selector_dir.mkdir(parents=True, exist_ok=True)
                started_at = datetime.now(timezone.utc).isoformat()
                started = time.perf_counter()
                worker_statuses[selector].update({"status": "running", "started_at": started_at})
                _write_json_atomic(
                    run_dir / "gpu_assignments.json",
                    {
                        "parallel_mode": "model_parallel",
                        "requested_gpu_ids": model_parallel_gpu_ids,
                        "hardware_profile": hardware_profile,
                        "workers": worker_statuses,
                    },
                )
                kernel = build_kernel(
                    selector,
                    artifacts.model,
                    seed,
                    config,
                    pair_chunk_size=int(hardware_profile["pair_chunk_size"]),
                    activation_checkpointing=bool(hardware_profile["activation_checkpointing"]),
                    model_parallel_devices=model_parallel_devices,
                )
                train_args.resume = resuming and (selector_dir / "checkpoints" / "latest.pt").is_file()
                train_args.resume_contract = selector_resume_contract(
                    config_path=config_path,
                    baseline_dir=baseline_dir,
                    data_dir=data_dir,
                    selector=selector,
                    seed=seed,
                    pair_chunk_size=int(hardware_profile["pair_chunk_size"]),
                    activation_checkpointing=bool(hardware_profile["activation_checkpointing"]),
                    model_parallel_gpu_ids=model_parallel_gpu_ids,
                )
                try:
                    train_result = train_kernel(
                        artifacts.model, kernel, train_records, valid_loader, artifacts,
                        device, selector, seed, train_args, selector_dir,
                    )
                except TrainingPaused as exc:
                    worker_statuses[selector].update(
                        {"status": "paused", "finished_at": datetime.now(timezone.utc).isoformat()}
                    )
                    raise RunPaused(str(exc)) from exc
                valid_result = evaluate_selector(
                    artifacts.model,
                    valid_loader,
                    device,
                    len(artifacts.label_to_id),
                    kernel,
                    f"{selector}_valid_final",
                )
                test_result = evaluate_selector(
                    artifacts.model,
                    test_loader,
                    device,
                    len(artifacts.label_to_id),
                    kernel,
                    f"{selector}_test",
                )
                metadata = kernel.metadata()
                trainable_parameters = sum(parameter.numel() for parameter in kernel.parameters())
                row = {
                    "selector": selector,
                    "seed": seed,
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
                torch.save(
                    {"state_dict": kernel.state_dict(), "metadata": metadata},
                    selector_dir / "best_kernel_with_metadata.pt",
                )
                (selector_dir / "metrics.json").write_text(
                    json.dumps(row, indent=2, sort_keys=True), encoding="utf-8"
                )
                worker_statuses[selector].update(
                    {
                        "status": "complete",
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "duration_seconds": round(time.perf_counter() - started, 3),
                    }
                )
                _write_json_atomic(
                    run_dir / "gpu_assignments.json",
                    {
                        "parallel_mode": "model_parallel",
                        "requested_gpu_ids": model_parallel_gpu_ids,
                        "hardware_profile": hardware_profile,
                        "workers": worker_statuses,
                    },
                )
                del kernel
                gc.collect()
                torch.cuda.empty_cache()
                current_selector = None
        except RunPaused:
            now = datetime.now(timezone.utc).isoformat()
            for selector, status in worker_statuses.items():
                if status["status"] == "pending":
                    status.update({"status": "paused", "finished_at": now})
            _write_json_atomic(
                run_dir / "gpu_assignments.json",
                {
                    "parallel_mode": "model_parallel",
                    "requested_gpu_ids": model_parallel_gpu_ids,
                    "hardware_profile": hardware_profile,
                    "workers": worker_statuses,
                },
            )
            _write_root_marker(run_dir, "RUN_PAUSED", stage="model_parallel_selectors", workers=worker_statuses)
            raise
        except BaseException as exc:
            now = datetime.now(timezone.utc).isoformat()
            if current_selector is not None and worker_statuses[current_selector]["status"] == "running":
                worker_statuses[current_selector].update(
                    {
                        "status": "failed",
                        "finished_at": now,
                        "duration_seconds": round(time.perf_counter() - started, 3),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            for selector, status in worker_statuses.items():
                if status["status"] == "pending":
                    status.update({"status": "not_started", "finished_at": now})
            _write_json_atomic(
                run_dir / "gpu_assignments.json",
                {
                    "parallel_mode": "model_parallel",
                    "requested_gpu_ids": model_parallel_gpu_ids,
                    "hardware_profile": hardware_profile,
                    "workers": worker_statuses,
                },
            )
            (run_dir / "RUN_FAILED").write_text(
                json.dumps(
                    {
                        "failed_at_utc": now,
                        "stage": "model_parallel_selectors",
                        "failed_selector": current_selector,
                        "reason": f"{type(exc).__name__}: {exc}",
                        "workers": worker_statuses,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            raise
        _write_json_atomic(
            run_dir / "gpu_assignments.json",
            {
                "parallel_mode": "model_parallel",
                "requested_gpu_ids": model_parallel_gpu_ids,
                "hardware_profile": hardware_profile,
                "workers": worker_statuses,
            },
        )
    else:
        # Release the parent's model copy before selector workers load the shared
        # checkpoint. This is important when the first worker uses GPU 0.
        artifacts.model.to("cpu")
        del valid_loader, test_loader
        if device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        del artifacts
        del train_records
        if gpu_ids:
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
            resume=resuming,
            pause=pause,
            adaptive_memory_state=adaptive_memory_state,
        )
    rows: list[dict[str, Any]] = []
    for selector in selectors:
        metrics_path = run_dir / "selectors" / selector / "metrics.json"
        if not _valid_selector_metrics(metrics_path, selector):
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
        "parallel_mode": "model_parallel" if model_parallel_devices else ("selector_parallel" if gpu_ids else "serial"),
        "model_parallel": {
            "enabled": bool(model_parallel_devices),
            "physical_gpu_ids": model_parallel_gpu_ids,
            "device_map": artifacts.model.model_parallel_metadata() if model_parallel_devices else {"enabled": False},
        },
        "multi_gpu": {
            "requested_gpu_ids": gpu_ids,
            "selector_parallelism": len(gpu_ids),
            "worker_statuses": worker_statuses,
        },
        "hardware_profile": hardware_profile,
        "adaptive_memory": (
            _read_json(_adaptive_state_path(run_dir))
            if hardware_profile.get("adaptive")
            else None
        ),
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
            "model_parallel_gpu_ids": model_parallel_gpu_ids,
            "model_parallel_device_map": artifacts.model.model_parallel_metadata() if model_parallel_devices else {"enabled": False},
            "pair_chunk_size": int(hardware_profile["pair_chunk_size"]),
            "activation_checkpointing": bool(hardware_profile["activation_checkpointing"]),
            "memory_strategy": "adaptive" if hardware_profile.get("adaptive") else "fixed",
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
        f"- parallel mode: `{'model_parallel' if model_parallel_devices else ('selector_parallel' if gpu_ids else 'serial')}`",
        f"- model-parallel physical GPUs: `{model_parallel_gpu_ids}`",
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
    (run_dir / "RUN_PAUSED").unlink(missing_ok=True)
    (run_dir / "RUN_FAILED").unlink(missing_ok=True)
    (run_dir / "RUN_COMPLETE").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )
    return 0


def main() -> int:
    args = parse_args()
    pause = PauseController()
    pause.install()
    try:
        return _run(args, pause)
    except RunPaused as exc:
        run_dir = getattr(args, "_run_dir", None)
        if run_dir is not None and not (run_dir / "RUN_PAUSED").exists():
            _write_root_marker(run_dir, "RUN_PAUSED", reason=str(exc))
            _record_resume_event(run_dir, event="paused", reason=str(exc))
        print(f"[q-triad] safely paused: {exc}", flush=True)
        return PAUSED_EXIT_CODE
    except BaseException as exc:
        run_dir = getattr(args, "_run_dir", None)
        if run_dir is not None and not (run_dir / "RUN_FAILED").exists():
            _write_root_marker(run_dir, "RUN_FAILED", reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        pause.close()


if __name__ == "__main__":
    raise SystemExit(main())
