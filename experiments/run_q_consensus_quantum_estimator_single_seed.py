#!/usr/bin/env python3
"""Run the frozen seed-7 consensus estimator on one physical GPU.

This is the required execution and scientific preflight before the frozen
five-seed run. It preserves the canary task, estimators, selectors, training
budget, and gate without exposing seed or hyperparameter overrides.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_q_consensus_quantum_estimator_canary as canary  # noqa: E402
import run_q_consensus_quantum_estimator_frozen_multiseed as frozen  # noqa: E402


SCHEMA_VERSION = "q-attention.q-consensus-quantum-estimator-single-seed.v1"
PREFLIGHT_SEED = 7


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/q_consensus_quantum_estimator_frozen_multiseed.json",
    )
    parser.add_argument("--gpu", default="auto", help="one physical GPU ID, or auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--physical-gpu-id", type=int, default=None, help=argparse.SUPPRESS)
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
        Path("experiments/run_q_consensus_quantum_estimator_single_seed.py"),
        Path("src/q_attention/plugins/q_consensus_quantum_estimator.py"),
    )
    return {path.as_posix(): sha256(ROOT / path) for path in paths}


def choose_gpu(spec: str) -> int:
    if spec == "auto":
        ids = frozen.parse_gpu_ids("auto")
        return ids[0]
    if not spec.isdigit():
        raise ValueError("--gpu must be auto or one non-negative physical GPU ID")
    gpu_id = int(spec)
    frozen.parse_gpu_ids(str(gpu_id))
    return gpu_id


def seed_config(master_config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if frozen.FROZEN_SEEDS[0] != PREFLIGHT_SEED:
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
        "output_root": output_dir.relative_to(ROOT).as_posix(),
    }


def promote_canary_payload(
    payload: dict[str, Any],
    *,
    elapsed_seconds: float,
    physical_gpu_id: int,
    config_path: Path,
    config_sha256: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    result = dict(payload)
    result["canary_schema_version"] = result["schema_version"]
    result["schema_version"] = SCHEMA_VERSION
    result["formal_preflight"] = True
    result["formal_experiment"] = False
    result["run_type"] = "frozen_single_seed_single_gpu_preflight"
    result["experiment_name"] = "q_consensus_quantum_estimator_single_seed"
    result["config_path"] = config_path.relative_to(ROOT).as_posix()
    result["config_sha256"] = config_sha256
    result["runtime"] = {
        "elapsed_seconds": elapsed_seconds,
        "scope": "complete seed including both estimator training and all selector evaluation",
    }
    result["parallelism"] = {
        "type": "single_seed_single_gpu",
        "ddp": False,
        "physical_gpu_id": physical_gpu_id,
        "workers_on_gpu": 1,
    }
    result["provenance"] = provenance
    gate = dict(result["gate"])
    gate["next_multi_seed_authorized"] = gate.get("status") == "pass"
    gate["next_real_data_authorized"] = False
    gate["finite_shot_authorized"] = False
    gate["hardware_claim"] = False
    gate["quantum_advantage_claim"] = False
    result["gate"] = gate
    return result


def run_worker(args: argparse.Namespace, config_path: Path) -> int:
    if (
        not args.worker_output_dir
        or args.physical_gpu_id is None
        or not args.expected_commit
        or not args.expected_config_sha256
    ):
        raise SystemExit("internal worker arguments are incomplete")
    output_dir = resolve_path(args.worker_output_dir)
    if not output_dir.is_relative_to((ROOT / "runs").resolve()):
        raise SystemExit("worker output must be inside runs/")
    master_config = frozen.load_config(config_path)
    if sha256(config_path) != args.expected_config_sha256:
        raise SystemExit("frozen config hash changed before worker start")
    git = frozen.git_provenance(require_clean=True)
    if git["git_commit"] != args.expected_commit:
        raise SystemExit("Git commit changed before worker start")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu_id):
        raise SystemExit("worker CUDA_VISIBLE_DEVICES does not match the physical GPU assignment")

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("single-seed worker must see exactly one CUDA GPU")
    output_dir.mkdir(parents=True, exist_ok=False)
    seed_config_path = output_dir / "seed_config.json"
    write_json(seed_config_path, seed_config(master_config, output_dir))
    torch.cuda.synchronize()
    started = time.perf_counter()
    payload = canary.run(
        argparse.Namespace(
            config=str(seed_config_path),
            device="cuda",
            output_root=str(output_dir),
        )
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    payload = promote_canary_payload(
        payload,
        elapsed_seconds=elapsed,
        physical_gpu_id=args.physical_gpu_id,
        config_path=config_path,
        config_sha256=args.expected_config_sha256,
        provenance={
            **git,
            "source_sha256": source_hashes(),
        },
    )
    write_json(output_dir / "run_summary.json", payload)
    (output_dir / "SINGLE_SEED_COMPLETE").write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output_dir.relative_to(ROOT)),
                "elapsed_seconds": elapsed,
                "physical_gpu_id": args.physical_gpu_id,
                "gate": payload["gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    try:
        frozen.load_config(config_path)
    except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.worker:
        return run_worker(args, config_path)
    if any(
        value is not None
        for value in (
            args.worker_output_dir,
            args.physical_gpu_id,
            args.expected_commit,
            args.expected_config_sha256,
        )
    ):
        raise SystemExit("internal worker arguments require --worker")
    try:
        gpu_id = choose_gpu(args.gpu)
        provenance = frozen.git_provenance(require_clean=not args.dry_run)
    except (ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
    config_digest = sha256(config_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = resolve_path(
        args.output_dir
        or Path("runs/q_consensus_quantum_estimator_single_seed") / stamp
    )
    if not output_dir.is_relative_to((ROOT / "runs").resolve()):
        raise SystemExit("output directory must be inside runs/")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run_only",
                    "formal_preflight_started": False,
                    "seed": PREFLIGHT_SEED,
                    "parallelism": "single_seed_single_gpu",
                    "physical_gpu_id": gpu_id,
                    "config_sha256": config_digest,
                    "worktree_dirty": provenance["git_dirty"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if output_dir.exists():
        raise SystemExit(f"refusing to reuse output directory: {output_dir}")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(config_path),
        "--worker",
        "--worker-output-dir",
        str(output_dir),
        "--physical-gpu-id",
        str(gpu_id),
        "--expected-commit",
        provenance["git_commit"],
        "--expected-config-sha256",
        config_digest,
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    completed = subprocess.run(command, cwd=ROOT, env=environment)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
