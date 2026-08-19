#!/usr/bin/env python3
"""Run the frozen seed-7 consensus estimator as a two-GPU stage preflight.

The quantum bundle and matched product-state control train concurrently in
independent processes on distinct physical GPUs. This validates stage-level
parallel execution before the frozen five-seed run; it is not DDP.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_q_consensus_quantum_estimator_canary as canary  # noqa: E402
import run_q_consensus_quantum_estimator_frozen_multiseed as frozen  # noqa: E402


SCHEMA_VERSION = "q-attention.q-consensus-quantum-estimator-single-seed-multigpu.v1"
STAGE_SCHEMA_VERSION = f"{SCHEMA_VERSION}.stage.v1"
PREFLIGHT_SEED = 7
STAGES = ("quantum_controls", "classical_control")
STAGE_SELECTORS = {
    "quantum_controls": (
        "disabled",
        "q_consensus_quantum",
        "q_consensus_shuffled_query",
        "q_consensus_magnitude",
    ),
    "classical_control": ("classical_consensus_control",),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/q_consensus_quantum_estimator_frozen_multiseed.json",
    )
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage", choices=STAGES, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--stage-output-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--stage-gpu-id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--expected-commit", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--expected-config-sha256", default=None, help=argparse.SUPPRESS)
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def resolve_path(path: str | Path) -> Path:
    result = Path(path)
    if not result.is_absolute():
        result = ROOT / result
    return result.resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def source_hashes() -> dict[str, str]:
    paths = (
        Path("experiments/run_q_consensus_error_witness_prescreen_toy.py"),
        Path("experiments/run_q_consensus_quantum_estimator_canary.py"),
        Path("experiments/run_q_consensus_quantum_estimator_frozen_multiseed.py"),
        Path("experiments/run_q_consensus_quantum_estimator_single_seed_multigpu.py"),
        Path("src/q_attention/plugins/q_consensus_quantum_estimator.py"),
    )
    return {path.as_posix(): sha256(ROOT / path) for path in paths}


def build_assignments(gpu_ids: list[int]) -> list[dict[str, Any]]:
    if len(gpu_ids) < len(STAGES):
        raise ValueError("single-seed multi-GPU preflight requires at least two GPUs")
    selected = gpu_ids[: len(STAGES)]
    if len(set(selected)) != len(STAGES):
        raise ValueError("single-seed stages require distinct physical GPU IDs")
    return [
        {"stage": stage, "gpu_id": gpu_id}
        for stage, gpu_id in zip(STAGES, selected, strict=True)
    ]


def seed_config(master_config: dict[str, Any]) -> dict[str, Any]:
    if PREFLIGHT_SEED != frozen.FROZEN_SEEDS[0]:
        raise ValueError("preflight seed must remain the first frozen seed")
    return {
        "schema_version": "q-attention.q-consensus-quantum-estimator-canary.v1",
        "experiment_name": "q_consensus_quantum_estimator_canary",
        "selectors": list(frozen.FROZEN_SELECTORS),
        "seed": PREFLIGHT_SEED,
        "device": "cuda",
        "dataset": master_config["dataset"],
        "estimator": master_config["estimator"],
        "training": master_config["training"],
        "gate": master_config["seed_gate"],
        "output_root": "runs/q_consensus_quantum_estimator_single_seed_multigpu",
    }


def make_streams(config: dict[str, Any], device: torch.device) -> dict[str, dict[str, torch.Tensor]]:
    dataset = config["dataset"]
    return {
        "train": canary.task.make_split(PREFLIGHT_SEED, int(dataset["train_size"]), device),
        "calibration": canary.task.make_split(
            PREFLIGHT_SEED + 1000, int(dataset["calibration_size"]), device
        ),
        "valid": canary.task.make_split(
            PREFLIGHT_SEED + 10000, int(dataset["valid_size"]), device
        ),
        "test": canary.task.make_split(
            PREFLIGHT_SEED + 20000, int(dataset["test_size"]), device
        ),
    }


def baseline_metrics(
    split: dict[str, torch.Tensor], frames: torch.Tensor
) -> dict[str, Any]:
    logits, _ = canary.task.v1.baseline_logits(
        split["scores"], split["key"], split["query"], frames
    )
    replay, _ = canary.task.v1.baseline_logits(
        split["scores"], split["key"], split["query"], frames
    )
    prediction = logits.argmax(dim=-1)
    return {
        "accuracy": float(prediction.eq(split["labels"]).float().mean()),
        "replay_error": float((replay - logits).abs().max()),
        "queries": int(split["labels"].numel()),
    }


def run_stage(
    stage: str,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    frames = canary.task.v1.relation_frames(device)
    streams = make_streams(config, device)
    kind = "quantum" if stage == "quantum_controls" else "classical"
    estimator = canary.build_estimator(kind, PREFLIGHT_SEED, frames, config, device)
    training = canary.train_estimator(estimator, streams["train"], config)
    selectors: dict[str, dict[str, Any]] = {}
    baseline: dict[str, dict[str, Any]] = {}
    for split_name, split in streams.items():
        baseline[split_name] = baseline_metrics(split, frames)
        selectors[split_name] = {}
        for selector in STAGE_SELECTORS[stage]:
            selector_estimator = None if selector == "disabled" else estimator
            selectors[split_name][selector] = canary.evaluate(
                selector,
                selector_estimator,
                split,
                frames,
                int(config["dataset"]["batch_size"]),
            )
    return {
        "stage": stage,
        "seed": PREFLIGHT_SEED,
        "selectors": selectors,
        "baseline": baseline,
        "training": {kind: training},
        "estimators": {kind: estimator.metadata()},
        "trainable_parameters": {
            kind: sum(parameter.numel() for parameter in estimator.parameters())
        },
    }


def run_stage_worker(args: argparse.Namespace, config_path: Path) -> int:
    if args.stage is None:
        raise SystemExit("internal stage is missing")
    if (
        not args.stage_output_dir
        or args.stage_gpu_id is None
        or not args.expected_commit
        or not args.expected_config_sha256
    ):
        raise SystemExit("internal stage arguments are incomplete")
    master_config = frozen.load_config(config_path)
    config = seed_config(master_config)
    if sha256(config_path) != args.expected_config_sha256:
        raise SystemExit("frozen config hash changed before stage start")
    provenance = frozen.git_provenance(require_clean=True)
    if provenance["git_commit"] != args.expected_commit:
        raise SystemExit("Git commit changed before stage start")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(args.stage_gpu_id):
        raise SystemExit("stage CUDA_VISIBLE_DEVICES does not match its physical GPU assignment")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("stage must see exactly one CUDA GPU")
    output_dir = resolve_path(args.stage_output_dir)
    if not output_dir.is_relative_to((ROOT / "runs").resolve()):
        raise SystemExit("stage output must be inside runs/")
    output_dir.mkdir(parents=True, exist_ok=False)
    payload = run_stage(args.stage, config, torch.device("cuda"))
    payload.update(
        {
            "schema_version": STAGE_SCHEMA_VERSION,
            "status": "complete",
            "formal_preflight_stage": True,
            "config_path": config_path.relative_to(ROOT).as_posix(),
            "config_sha256": args.expected_config_sha256,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": "cuda:0",
                "physical_gpu_id": args.stage_gpu_id,
                "cuda_device": torch.cuda.get_device_name(0),
                "visible_device_count": torch.cuda.device_count(),
            },
            "provenance": {
                **provenance,
                "source_sha256": source_hashes(),
            },
        }
    )
    write_json(output_dir / "stage_summary.json", payload)
    (output_dir / "STAGE_COMPLETE").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "stage": args.stage,
                "physical_gpu_id": args.stage_gpu_id,
                "output": str(output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


def build_gate(
    baseline: dict[str, dict[str, Any]],
    training: dict[str, dict[str, Any]],
    selectors: dict[str, dict[str, dict[str, Any]]],
    config: dict[str, Any],
    *,
    stage_execution_complete: bool,
    distinct_physical_gpus: bool,
    stage_time_overlap_seconds: float,
    parameter_matched: bool,
) -> dict[str, Any]:
    gate_config = config["gate"]
    quantum = "q_consensus_quantum"
    classical = "classical_consensus_control"
    shuffled = "q_consensus_shuffled_query"
    valid = selectors["valid"]
    test = selectors["test"]
    conditions = {
        "stage_execution_complete": stage_execution_complete,
        "distinct_physical_gpus": distinct_physical_gpus,
        "stage_time_overlap": stage_time_overlap_seconds > 0.0,
        "parameter_matched": parameter_matched,
        "baseline_replay": all(item["replay_error"] == 0.0 for item in baseline.values()),
        "baseline_non_saturated": float(gate_config["baseline_accuracy_min"])
        <= baseline["valid"]["accuracy"]
        <= float(gate_config["baseline_accuracy_max"])
        and float(gate_config["baseline_accuracy_min"])
        <= baseline["test"]["accuracy"]
        <= float(gate_config["baseline_accuracy_max"]),
        "quantum_heldout_gain": valid[quantum]["accuracy_delta"]
        >= float(gate_config["minimum_accuracy_delta"])
        and test[quantum]["accuracy_delta"]
        >= float(gate_config["minimum_accuracy_delta"]),
        "quantum_no_harm": valid[quantum]["harm_rate"]
        <= float(gate_config["maximum_harm_rate"])
        and test[quantum]["harm_rate"]
        <= float(gate_config["maximum_harm_rate"]),
        "quantum_beats_classical": valid[quantum]["accuracy_delta"]
        - valid[classical]["accuracy_delta"]
        >= float(gate_config["minimum_quantum_control_margin"])
        and test[quantum]["accuracy_delta"]
        - test[classical]["accuracy_delta"]
        >= float(gate_config["minimum_quantum_control_margin"]),
        "quantum_beats_shuffled": valid[quantum]["accuracy_delta"]
        - valid[shuffled]["accuracy_delta"]
        >= float(gate_config["minimum_control_margin"])
        and test[quantum]["accuracy_delta"]
        - test[shuffled]["accuracy_delta"]
        >= float(gate_config["minimum_control_margin"]),
        "training_gradients_finite": all(
            info["gradient_norm_min"] > 0.0
            and info["gradient_norm_max"] < float(gate_config["maximum_gradient_norm"])
            for info in training.values()
        ),
        "residual_invariants": all(
            split[quantum]["residual_finite"]
            and split[quantum]["residual_zero_sum_error"] <= 1e-5
            and split[quantum]["residual_max_abs"] <= canary.task.v1.MAX_DELTA + 1e-6
            for split in selectors.values()
        ),
    }
    passed = all(conditions.values())
    return {
        **conditions,
        "status": "pass" if passed else "fail",
        "next_multi_seed_authorized": passed,
        "next_real_data_authorized": False,
        "finite_shot_authorized": False,
        "hardware_claim": False,
        "quantum_advantage_claim": False,
    }


def stage_overlap_seconds(execution: list[dict[str, Any]]) -> float:
    if len(execution) != len(STAGES):
        return 0.0
    latest_start = max(float(item["started_at_epoch"]) for item in execution)
    earliest_end = min(float(item["completed_at_epoch"]) for item in execution)
    return max(0.0, earliest_end - latest_start)


def baselines_match(
    first: dict[str, dict[str, Any]], second: dict[str, dict[str, Any]]
) -> bool:
    if set(first) != set(second):
        return False
    return all(
        int(first[split]["queries"]) == int(second[split]["queries"])
        and abs(float(first[split]["accuracy"]) - float(second[split]["accuracy"])) <= 1e-12
        and abs(float(first[split]["replay_error"]) - float(second[split]["replay_error"])) <= 1e-12
        for split in first
    )


def combine_stage_payloads(
    stage_payloads: dict[str, dict[str, Any]],
    execution: list[dict[str, Any]],
    master_config: dict[str, Any],
    *,
    config_path: Path,
    config_sha256: str,
    git_commit: str,
    expected_source_hashes: dict[str, str],
) -> dict[str, Any]:
    if set(stage_payloads) != set(STAGES):
        raise ValueError("preflight requires both frozen stages")
    for stage, payload in stage_payloads.items():
        if payload.get("schema_version") != STAGE_SCHEMA_VERSION:
            raise ValueError(f"stage schema mismatch: {stage}")
        if payload.get("stage") != stage or payload.get("status") != "complete":
            raise ValueError(f"stage identity or status mismatch: {stage}")
        if payload.get("seed") != PREFLIGHT_SEED:
            raise ValueError(f"stage seed mismatch: {stage}")
        if payload.get("config_sha256") != config_sha256:
            raise ValueError(f"stage config hash mismatch: {stage}")
        provenance = payload.get("provenance", {})
        if provenance.get("git_commit") != git_commit or provenance.get("git_dirty") is not False:
            raise ValueError(f"stage Git provenance mismatch: {stage}")
        if provenance.get("source_sha256") != expected_source_hashes:
            raise ValueError(f"stage source hashes mismatch: {stage}")

    quantum_stage = stage_payloads["quantum_controls"]
    classical_stage = stage_payloads["classical_control"]
    if not baselines_match(quantum_stage["baseline"], classical_stage["baseline"]):
        raise ValueError("stage baseline metrics differ")
    baseline = quantum_stage["baseline"]
    training = {**quantum_stage["training"], **classical_stage["training"]}
    estimators = {**quantum_stage["estimators"], **classical_stage["estimators"]}
    parameter_counts = {
        **quantum_stage["trainable_parameters"],
        **classical_stage["trainable_parameters"],
    }
    selectors: dict[str, dict[str, dict[str, Any]]] = {}
    expected_splits = set(baseline)
    if set(quantum_stage["selectors"]) != expected_splits or set(classical_stage["selectors"]) != expected_splits:
        raise ValueError("stage split sets differ")
    for split in baseline:
        selectors[split] = {
            **quantum_stage["selectors"][split],
            **classical_stage["selectors"][split],
        }
        if set(selectors[split]) != set(frozen.FROZEN_SELECTORS):
            raise ValueError(f"combined selector allowlist is incomplete: {split}")

    execution_complete = len(execution) == len(STAGES) and all(
        item.get("success") is True for item in execution
    )
    physical_gpu_ids = [int(item["gpu_id"]) for item in execution]
    overlap = stage_overlap_seconds(execution)
    parameter_matched = parameter_counts.get("quantum") == parameter_counts.get("classical")
    config = seed_config(master_config)
    gate = build_gate(
        baseline,
        training,
        selectors,
        config,
        stage_execution_complete=execution_complete,
        distinct_physical_gpus=len(set(physical_gpu_ids)) == len(STAGES),
        stage_time_overlap_seconds=overlap,
        parameter_matched=parameter_matched,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "formal_preflight": True,
        "formal_experiment": False,
        "run_type": "frozen_single_seed_multigpu_stage_preflight",
        "experiment_name": "q_consensus_quantum_estimator_single_seed_multigpu",
        "dataset_identity": config["dataset"]["identity"],
        "seed": PREFLIGHT_SEED,
        "selectors_allowlist": list(frozen.FROZEN_SELECTORS),
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": config_sha256,
        "provenance": {
            "git_commit": git_commit,
            "git_dirty": False,
            "source_sha256": expected_source_hashes,
        },
        "parallelism": {
            "type": "within_seed_stage_parallel",
            "ddp": False,
            "physical_gpu_ids": physical_gpu_ids,
            "stage_time_overlap_seconds": overlap,
            "assignments": [
                {"stage": item["stage"], "gpu_id": item["gpu_id"]}
                for item in sorted(execution, key=lambda row: STAGES.index(row["stage"]))
            ],
        },
        "baseline": baseline,
        "training": training,
        "estimators": estimators,
        "trainable_parameters": parameter_counts,
        "selectors": selectors,
        "stage_execution": execution,
        "gate": gate,
    }


def worker_command(
    *,
    stage: str,
    gpu_id: int,
    output_dir: Path,
    config_path: Path,
    commit: str,
    config_sha256: str,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
        "--stage",
        stage,
        "--stage-output-dir",
        str(output_dir),
        "--stage-gpu-id",
        str(gpu_id),
        "--expected-commit",
        commit,
        "--expected-config-sha256",
        config_sha256,
    ]


def execute_stage(command: list[str], stage: str, gpu_id: int) -> dict[str, Any]:
    output_dir = Path(command[command.index("--stage-output-dir") + 1])
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    started_epoch = time.time()
    started = datetime.now(timezone.utc)
    process = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    completed_epoch = time.time()
    completed = datetime.now(timezone.utc)
    if process.stdout:
        print(f"[{stage}][gpu {gpu_id}] {process.stdout.strip()}", flush=True)
    if process.stderr:
        print(f"[{stage}][gpu {gpu_id}][stderr] {process.stderr.strip()}", file=sys.stderr, flush=True)
    return {
        "stage": stage,
        "gpu_id": gpu_id,
        "output_dir": str(output_dir),
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "started_at_epoch": started_epoch,
        "completed_at_epoch": completed_epoch,
        "elapsed_seconds": round(completed_epoch - started_epoch, 3),
        "returncode": process.returncode,
        "success": process.returncode == 0 and (output_dir / "STAGE_COMPLETE").is_file(),
    }


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    try:
        master_config = frozen.load_config(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.stage is not None:
        return run_stage_worker(args, config_path)
    if any(
        value is not None
        for value in (
            args.stage_output_dir,
            args.stage_gpu_id,
            args.expected_commit,
            args.expected_config_sha256,
        )
    ):
        raise SystemExit("internal stage arguments require --stage")
    try:
        gpu_ids = frozen.parse_gpu_ids(args.gpus)
        assignments = build_assignments(gpu_ids)
        provenance = frozen.git_provenance(require_clean=not args.dry_run)
    except (ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
    config_digest = sha256(config_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = resolve_path(
        args.output_dir
        or Path("runs/q_consensus_quantum_estimator_single_seed_multigpu") / stamp
    )
    if not output_dir.is_relative_to((ROOT / "runs").resolve()):
        raise SystemExit("output directory must be inside runs/")
    planned = [
        {
            **assignment,
            "output_dir": str(output_dir / "stages" / assignment["stage"]),
        }
        for assignment in assignments
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_only",
                    "formal_preflight_started": False,
                    "seed": PREFLIGHT_SEED,
                    "parallelism": "within_seed_stage_parallel",
                    "ddp": False,
                    "config_sha256": config_digest,
                    "worktree_dirty": provenance["git_dirty"],
                    "available_gpu_ids": gpu_ids,
                    "assignments": planned,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if output_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    expected_hashes = source_hashes()
    write_json(
        output_dir / "multi_gpu_manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "formal_preflight": True,
            "seed": PREFLIGHT_SEED,
            "git_commit": provenance["git_commit"],
            "git_dirty": provenance["git_dirty"],
            "config_path": config_path.relative_to(ROOT).as_posix(),
            "config_sha256": config_digest,
            "source_sha256": expected_hashes,
            "parallelism": "within_seed_stage_parallel",
            "ddp": False,
            "available_gpu_ids": gpu_ids,
            "assignments": planned,
        },
    )
    commands = [
        worker_command(
            stage=item["stage"],
            gpu_id=item["gpu_id"],
            output_dir=Path(item["output_dir"]),
            config_path=config_path,
            commit=provenance["git_commit"],
            config_sha256=config_digest,
        )
        for item in planned
    ]
    execution: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(STAGES)) as pool:
        futures = {
            pool.submit(execute_stage, command, item["stage"], item["gpu_id"]): item["stage"]
            for command, item in zip(commands, planned, strict=True)
        }
        for future in as_completed(futures):
            execution.append(future.result())
    execution.sort(key=lambda item: STAGES.index(item["stage"]))
    write_json(
        output_dir / "stage_execution_summary.json",
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "execution_success": all(item["success"] for item in execution),
            "stage_time_overlap_seconds": stage_overlap_seconds(execution),
            "results": execution,
        },
    )
    if not all(item["success"] for item in execution):
        (output_dir / "MULTIGPU_PREFLIGHT_FAILED").write_text(
            datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
        )
        return 1
    stage_payloads = {
        item["stage"]: json.loads(
            (Path(item["output_dir"]) / "stage_summary.json").read_text(encoding="utf-8")
        )
        for item in execution
    }
    try:
        payload = combine_stage_payloads(
            stage_payloads,
            execution,
            master_config,
            config_path=config_path,
            config_sha256=config_digest,
            git_commit=provenance["git_commit"],
            expected_source_hashes=expected_hashes,
        )
    except (KeyError, TypeError, ValueError) as exc:
        (output_dir / "MULTIGPU_PREFLIGHT_FAILED").write_text(
            f"{datetime.now(timezone.utc).isoformat()} combine_error={exc}\n",
            encoding="utf-8",
        )
        raise SystemExit(str(exc)) from exc
    write_json(output_dir / "run_summary.json", payload)
    (output_dir / "MULTIGPU_PREFLIGHT_COMPLETE").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output_dir.relative_to(ROOT)),
                "gate": payload["gate"],
                "parallelism": payload["parallelism"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
