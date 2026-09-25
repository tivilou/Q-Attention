#!/usr/bin/env python3
"""Run the frozen five-seed consensus quantum-estimator validation.

The public CLI intentionally exposes no seed, training, selector, or gate
overrides. Formal execution requires a clean Git checkout. The internal worker
arguments are used only by the scheduler after it validates the frozen config.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_q_consensus_quantum_estimator_canary as canary  # noqa: E402


SCHEMA_VERSION = "q-attention.q-consensus-quantum-estimator-frozen-multiseed.v1"
SEED_SUMMARY_SCHEMA = "q-attention.q-consensus-quantum-estimator-frozen-seed.v1"
FROZEN_SEEDS = (7, 11, 13, 17, 23)
FROZEN_SELECTORS = canary.SELECTORS
FROZEN_DATASET = {
    "identity": "synthetic_dynamic_address_consensus_error_witness_v2",
    "train_size": 512,
    "calibration_size": 256,
    "valid_size": 256,
    "test_size": 256,
    "batch_size": 128,
}
FROZEN_ESTIMATOR = {
    "register_qubits": 3,
    "depth": 2,
    "angle_scale": 1.0,
    "seed_offset": 7331,
}
FROZEN_TRAINING = {
    "steps": 120,
    "lr": 0.03,
    "weight_decay": 0.0001,
    "gradient_clip": 5.0,
}
FROZEN_SEED_GATE = {
    "baseline_accuracy_min": 0.7,
    "baseline_accuracy_max": 0.9,
    "minimum_accuracy_delta": 0.03,
    "maximum_harm_rate": 0.02,
    "minimum_quantum_control_margin": 0.0,
    "minimum_control_margin": 0.02,
    "maximum_gradient_norm": 100.0,
}
FROZEN_AGGREGATE_GATE = {
    "required_seed_count": 5,
    "minimum_seed_gate_passes": 4,
    "minimum_quantum_gain_ci95_lower": 0.0,
    "minimum_quantum_product_mean_margin": 0.0,
    "maximum_quantum_product_sign_flip_p": 0.05,
    "minimum_quantum_shuffled_mean_margin": 0.02,
    "minimum_quantum_magnitude_mean_margin": 0.02,
    "maximum_pooled_harm_rate": 0.02,
    "require_all_invariants": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/q_consensus_quantum_estimator_frozen_multiseed.json",
    )
    parser.add_argument(
        "--gpus",
        default="auto",
        help="comma-separated physical GPU IDs, or auto",
    )
    parser.add_argument(
        "--preflight-summary",
        default=None,
        help="passed frozen single-seed run_summary.json",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--expected-commit", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--expected-config-sha256", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


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


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported frozen multi-seed config schema")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": "q_consensus_quantum_estimator_frozen_multiseed",
        "seeds": list(FROZEN_SEEDS),
        "selectors": list(FROZEN_SELECTORS),
        "device": "cuda",
        "dataset": FROZEN_DATASET,
        "estimator": FROZEN_ESTIMATOR,
        "training": FROZEN_TRAINING,
        "seed_gate": FROZEN_SEED_GATE,
        "aggregate_gate": FROZEN_AGGREGATE_GATE,
        "output_root": "runs/q_consensus_quantum_estimator_frozen_multiseed",
    }
    if payload != expected:
        changed = sorted(
            key for key in set(payload) | set(expected) if payload.get(key) != expected.get(key)
        )
        raise ValueError(f"frozen protocol differs in fields: {changed}")
    return payload


def git_provenance(*, require_clean: bool) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    dirty = bool(status.strip())
    if require_clean and dirty:
        raise ValueError("formal run requires a clean Git worktree")
    return {"git_commit": commit, "git_dirty": dirty}


def parse_gpu_ids(spec: str) -> list[int]:
    if spec == "auto":
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        parts = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    else:
        parts = [item.strip() for item in spec.split(",") if item.strip()]
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError("--gpus must be auto or a comma-separated list of GPU IDs")
    gpu_ids: list[int] = []
    for part in parts:
        gpu_id = int(part)
        if gpu_id in gpu_ids:
            raise ValueError(f"duplicate GPU ID: {gpu_id}")
        subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu_id),
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        gpu_ids.append(gpu_id)
    return gpu_ids


def gpu_name(gpu_id: int) -> str:
    return subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu_id),
            "--query-gpu=name",
            "--format=csv,noheader",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()[0]


def source_hashes() -> dict[str, str]:
    paths = (
        Path("experiments/run_q_consensus_error_witness_prescreen_toy.py"),
        Path("experiments/run_q_consensus_quantum_estimator_canary.py"),
        Path("src/q_attention/plugins/q_consensus_quantum_estimator.py"),
    )
    return {path.as_posix(): sha256(ROOT / path) for path in paths}


def validate_single_seed_preflight(
    path: Path,
    *,
    expected_commit: str,
    expected_config_sha256: str,
) -> dict[str, Any]:
    runs_root = (ROOT / "runs").resolve()
    path = path.resolve()
    if not path.is_relative_to(runs_root) or path.name != "run_summary.json":
        raise ValueError("preflight summary must be a run_summary.json inside runs/")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != (
        "q-attention.q-consensus-quantum-estimator-single-seed.v1"
    ):
        raise ValueError("preflight summary schema mismatch")
    if payload.get("formal_preflight") is not True or payload.get("seed") != 7:
        raise ValueError("preflight must be the frozen formal seed-7 preflight")
    if payload.get("config_sha256") != expected_config_sha256:
        raise ValueError("preflight used a different frozen config")
    provenance = payload.get("provenance", {})
    if (
        provenance.get("git_commit") != expected_commit
        or provenance.get("git_dirty") is not False
    ):
        raise ValueError("preflight Git provenance differs from the full run")
    parallelism = payload.get("parallelism", {})
    if (
        parallelism.get("type") != "single_seed_single_gpu"
        or parallelism.get("ddp") is not False
        or not isinstance(parallelism.get("physical_gpu_id"), int)
        or parallelism.get("workers_on_gpu") != 1
    ):
        raise ValueError("preflight did not use one worker on one physical GPU")
    runtime = payload.get("runtime", {})
    if float(runtime.get("elapsed_seconds", 0.0)) <= 0.0:
        raise ValueError("preflight is missing a positive complete-seed runtime")
    gate = payload.get("gate", {})
    if gate.get("status") != "pass" or gate.get("next_multi_seed_authorized") is not True:
        raise ValueError("preflight scientific gate did not authorize multi-seed execution")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
        "seed": payload["seed"],
        "physical_gpu_id": parallelism["physical_gpu_id"],
        "elapsed_seconds": runtime["elapsed_seconds"],
        "gate_status": gate["status"],
    }


def canary_config(config: dict[str, Any], seed: int, output_root: Path) -> dict[str, Any]:
    return {
        "schema_version": "q-attention.q-consensus-quantum-estimator-canary.v1",
        "experiment_name": "q_consensus_quantum_estimator_canary",
        "selectors": list(FROZEN_SELECTORS),
        "seed": seed,
        "device": "cuda",
        "dataset": config["dataset"],
        "estimator": config["estimator"],
        "training": config["training"],
        "gate": config["seed_gate"],
        "output_root": output_root.relative_to(ROOT).as_posix(),
    }


def run_worker(args: argparse.Namespace, config_path: Path) -> int:
    config = load_config(config_path)
    if args.worker_seed not in FROZEN_SEEDS:
        raise SystemExit("internal worker seed is outside the frozen seed set")
    if not args.worker_output_dir or not args.expected_commit or not args.expected_config_sha256:
        raise SystemExit("internal worker arguments are incomplete")
    output_dir = resolve_path(args.worker_output_dir)
    if not output_dir.is_relative_to((ROOT / "runs").resolve()):
        raise SystemExit("worker output must be inside runs/")
    if sha256(config_path) != args.expected_config_sha256:
        raise SystemExit("frozen config hash changed before worker start")
    provenance = git_provenance(require_clean=True)
    if provenance["git_commit"] != args.expected_commit:
        raise SystemExit("Git commit changed before worker start")
    output_dir.mkdir(parents=True, exist_ok=False)
    seed_config_path = output_dir / "seed_config.json"
    write_json(seed_config_path, canary_config(config, args.worker_seed, output_dir))
    payload = canary.run(
        argparse.Namespace(
            config=str(seed_config_path),
            device="cuda",
            output_root=str(output_dir),
        )
    )
    payload["canary_schema_version"] = payload["schema_version"]
    payload["schema_version"] = SEED_SUMMARY_SCHEMA
    payload["formal_experiment"] = True
    payload["run_type"] = "frozen_multiseed_synthetic_validation"
    payload["master_config_path"] = config_path.relative_to(ROOT).as_posix()
    payload["provenance"] = {
        **provenance,
        "master_config_sha256": args.expected_config_sha256,
        "source_sha256": source_hashes(),
    }
    write_json(output_dir / "run_summary.json", payload)
    (output_dir / "SEED_COMPLETE").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "seed": args.worker_seed,
                "seed_gate": payload["gate"]["status"],
                "output": str(output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


def worker_command(
    *,
    config_path: Path,
    seed: int,
    output_dir: Path,
    commit: str,
    config_sha256: str,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
        "--worker-seed",
        str(seed),
        "--worker-output-dir",
        str(output_dir),
        "--expected-commit",
        commit,
        "--expected-config-sha256",
        config_sha256,
    ]


def execute_seed(command: list[str], gpu_id: int) -> dict[str, Any]:
    seed = int(command[command.index("--worker-seed") + 1])
    output_dir = command[command.index("--worker-output-dir") + 1]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    started = datetime.now(timezone.utc)
    process = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    completed = datetime.now(timezone.utc)
    if process.stdout:
        print(f"[seed {seed}][gpu {gpu_id}] {process.stdout.strip()}", flush=True)
    if process.stderr:
        print(f"[seed {seed}][gpu {gpu_id}][stderr] {process.stderr.strip()}", file=sys.stderr, flush=True)
    return {
        "seed": seed,
        "gpu_id": gpu_id,
        "output_dir": output_dir,
        "returncode": process.returncode,
        "success": process.returncode == 0 and (Path(output_dir) / "SEED_COMPLETE").is_file(),
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "elapsed_seconds": round((completed - started).total_seconds(), 3),
    }


def build_gpu_schedules(
    assignments: list[dict[str, Any]], commands: list[list[str]]
) -> dict[int, list[list[str]]]:
    if len(assignments) != len(commands):
        raise ValueError("assignment and command counts differ")
    schedules: dict[int, list[list[str]]] = {}
    for assignment, command in zip(assignments, commands, strict=True):
        gpu_id = int(assignment["gpu_id"])
        schedules.setdefault(gpu_id, []).append(command)
    return schedules


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    try:
        config = load_config(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.worker_seed is not None:
        return run_worker(args, config_path)
    if any((args.worker_output_dir, args.expected_commit, args.expected_config_sha256)):
        raise SystemExit("internal worker arguments cannot be used without --worker-seed")
    try:
        gpu_ids = parse_gpu_ids(args.gpus)
        provenance = git_provenance(require_clean=not args.dry_run)
    except (ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
    config_digest = sha256(config_path)
    preflight = None
    if args.preflight_summary is not None:
        try:
            preflight = validate_single_seed_preflight(
                resolve_path(args.preflight_summary),
                expected_commit=provenance["git_commit"],
                expected_config_sha256=config_digest,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(str(exc)) from exc
    if not args.dry_run and preflight is None:
        raise SystemExit(
            "formal multi-seed run requires --preflight-summary from a passed "
            "frozen single-seed run"
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = resolve_path(
        args.output_dir or Path(config["output_root"]) / stamp
    )
    if not output_dir.is_relative_to((ROOT / "runs").resolve()):
        raise SystemExit("output directory must be inside runs/")
    assignments = [
        {
            "seed": seed,
            "gpu_id": gpu_ids[index % len(gpu_ids)],
            "output_dir": str(output_dir / f"seed_{seed}"),
        }
        for index, seed in enumerate(FROZEN_SEEDS)
    ]
    commands = [
        worker_command(
            config_path=config_path,
            seed=item["seed"],
            output_dir=Path(item["output_dir"]),
            commit=provenance["git_commit"],
            config_sha256=config_digest,
        )
        for item in assignments
    ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_only",
                    "formal_experiment_started": False,
                    "worktree_dirty": provenance["git_dirty"],
                    "config_sha256": config_digest,
                    "seeds": list(FROZEN_SEEDS),
                    "gpus": gpu_ids,
                    "worker_count": min(len(FROZEN_SEEDS), len(gpu_ids)),
                    "assignments": assignments,
                    "preflight": preflight,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if output_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "formal_experiment": True,
        "run_type": "frozen_multiseed_synthetic_validation",
        "git_commit": provenance["git_commit"],
        "git_dirty": provenance["git_dirty"],
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": config_digest,
        "source_sha256": source_hashes(),
        "seeds": list(FROZEN_SEEDS),
        "gpus": [{"id": gpu_id, "name": gpu_name(gpu_id)} for gpu_id in gpu_ids],
        "worker_count": min(len(FROZEN_SEEDS), len(gpu_ids)),
        "assignments": assignments,
        "preflight": preflight,
    }
    write_json(output_dir / "multi_seed_manifest.json", manifest)
    results: list[dict[str, Any]] = []
    status_lock = threading.Lock()

    def persist_status() -> None:
        write_json(
            output_dir / "multi_seed_status.json",
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "completed": sorted(results, key=lambda item: item["seed"]),
            },
        )

    schedules = build_gpu_schedules(assignments, commands)

    def run_gpu_schedule(gpu_id: int, scheduled_commands: list[list[str]]) -> None:
        for command in scheduled_commands:
            result = execute_seed(command, gpu_id)
            with status_lock:
                results.append(result)
                persist_status()

    with ThreadPoolExecutor(max_workers=len(schedules)) as pool:
        futures = [
            pool.submit(run_gpu_schedule, gpu_id, scheduled_commands)
            for gpu_id, scheduled_commands in schedules.items()
        ]
        for future in as_completed(futures):
            future.result()
    execution_success = len(results) == len(FROZEN_SEEDS) and all(
        result["success"] for result in results
    )
    execution_summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "execution_success": execution_success,
        "results": sorted(results, key=lambda item: FROZEN_SEEDS.index(item["seed"])),
    }
    write_json(output_dir / "multi_seed_execution_summary.json", execution_summary)
    marker = "MULTI_SEED_COMPLETE" if execution_success else "MULTI_SEED_FAILED"
    (output_dir / marker).write_text(execution_summary["completed_at"] + "\n", encoding="utf-8")
    if not execution_success:
        print(f"multi-seed execution failed: {output_dir}", file=sys.stderr)
        return 1
    summarize = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/summarize_q_consensus_quantum_estimator_frozen_multiseed.py"),
            "--group-dir",
            str(output_dir),
        ],
        cwd=ROOT,
    )
    if summarize.returncode != 0:
        return summarize.returncode
    print(f"multi-seed execution complete: {output_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
