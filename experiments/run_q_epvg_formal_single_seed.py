#!/usr/bin/env python3
"""Run the frozen Q-EPVG Re-TACRED formal single-seed contract."""

from __future__ import annotations

import argparse
import subprocess
import json
import sys
from pathlib import Path
from typing import Any

import torch

import run_q_epvg_scheduler_base as _base
from run_q_epvg_transfer_base import (
    evaluate,
    build_kernel as _build_epvg_kernel,
    metric_delta,
    train_kernel,
)

CUDA_OOM_EXIT_CODE = _base.CUDA_OOM_EXIT_CODE
MEMORY_PRESSURE_EXIT_CODE = _base.MEMORY_PRESSURE_EXIT_CODE
is_cuda_oom_error = _base.is_cuda_oom_error

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "retacred_q_epvg_formal_single_seed.json"
SELECTOR_WORKER_PATH = ROOT / "experiments" / "run_q_epvg_selector_worker.py"
_base.DEFAULT_CONFIG = DEFAULT_CONFIG
_base.FORMAL_RUN_NAME = "retacred_q_epvg_formal_single_seed"
_base.SELECTOR_WORKER_PATH = SELECTOR_WORKER_PATH
_base.FORMAL_RUN_NAME = "retacred_q_epvg_formal_single_seed"
_base.FORMAL_CONFIG_SCHEMAS = {
    "q-attention.q-epvg-formal-single-seed.v1",
}
_base.COUNTERFACTUAL_MODES = set()
_base_selector_resume_contract = _base.selector_resume_contract
_base_selector_resume_contract_compatible = (
    _base.selector_resume_contract_compatible
)


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _epvg_source_contract(*, counterfactual: bool = False) -> dict[str, Any]:
    del counterfactual
    paths = {
        "runner": ROOT / "experiments" / "run_q_epvg_formal_single_seed.py",
        "scheduler": ROOT / "experiments" / "run_q_epvg_scheduler_base.py",
        "worker": SELECTOR_WORKER_PATH,
        "baseline_trainer": ROOT / "experiments" / "train_relation_baseline.py",
        "kernel_trainer": ROOT / "experiments" / "run_q_epvg_transfer_base.py",
        "batch_resume": ROOT / "src" / "q_attention" / "experiments" / "batch_resume.py",
        "relation_steering": ROOT / "src" / "q_attention" / "experiments" / "relation_steering.py",
        "relation_model": ROOT / "src" / "q_attention" / "models" / "relation_transformer.py",
        "relation_task": ROOT / "src" / "q_attention" / "tasks" / "relation.py",
        "attention_adapter": ROOT / "src" / "q_attention" / "adapters" / "q_epvg_attention.py",
        "attention_intervention_model": ROOT / "src" / "q_attention" / "models" / "relation_transformer.py",
        "q_epvg": ROOT / "src" / "q_attention" / "plugins" / "q_epvg.py",
    }
    from q_attention.experiments.batch_resume import file_contract

    return {
        "git_revision": _git_output("rev-parse", "HEAD"),
        "files": {name: file_contract(path) for name, path in paths.items()},
    }


def selector_resume_contract(**kwargs: Any) -> dict[str, Any]:
    contract = _base_selector_resume_contract(**kwargs)
    contract["stage"] = "q_epvg_selector"
    contract["source"] = _epvg_source_contract()
    return contract


def selector_resume_contract_compatible(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    return _base_selector_resume_contract_compatible(previous, current)


# The shared scheduler calls these names through its own module namespace;
# replace them there as well as exporting them for the worker.
_base._source_contract = _epvg_source_contract
_base.selector_resume_contract = selector_resume_contract
_base.selector_resume_contract_compatible = selector_resume_contract_compatible


def build_kernel(
    mode: str,
    model: torch.nn.Module,
    seed: int,
    config: dict[str, Any],
    *,
    pair_chunk_size: int | None | object = _base._DEFAULT_PAIR_CHUNK,
    pair_chunk_divisor: int = 1,
    activation_checkpointing: bool | None = None,
    model_parallel_devices: tuple[torch.device, ...] = (),
) -> torch.nn.Module | None:
    if mode == "disabled":
        return None
    kernel = dict(config.get("kernel", {}))
    kernel["seed_offset"] = int(kernel.get("seed_offset", 3000))
    args = argparse.Namespace(**kernel)
    if pair_chunk_size is not _base._DEFAULT_PAIR_CHUNK:
        if pair_chunk_size is None:
            args.query_chunk_size = None
        else:
            args.query_chunk_size = max(1, int(pair_chunk_size) // max(1, int(pair_chunk_divisor)))
    del activation_checkpointing, model_parallel_devices
    built = _build_epvg_kernel(mode, model, seed, args, config_override=kernel)
    # Query chunking is an execution control owned by the intervention adapter,
    # not by QEPVGConfig (and therefore must not enter the scientific config).
    built.query_chunk_size = getattr(args, "query_chunk_size", None)
    return built


def evaluate_selector(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    label_count: int,
    kernel: torch.nn.Module | None,
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


_base.build_kernel = build_kernel
_base.evaluate_selector = evaluate_selector
_base.metric_delta = metric_delta
_base.train_kernel = train_kernel


def _rewrite_summary(run_dir: Path, config: dict[str, Any]) -> None:
    path = run_dir / "run_summary.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "q-attention.q-epvg-formal-single-seed.run.v1"
    payload["name"] = config["name"]
    payload["candidate"] = config["candidate"]
    payload["matched_control"] = config["matched_control"]
    payload["structural_control"] = config["structural_control"]
    payload["method"] = "q_epvg"
    payload["selectors"] = list(config["selectors"])
    payload["claim_limits"] = config.get("claim_limits", {})
    gates = payload.get("gates", {})
    gates["candidate_minus_structural_macro_f1"] = gates.pop(
        "candidate_minus_random_group_macro_f1", gates.get("candidate_minus_structural_macro_f1")
    )
    gates["random_phase_control_completed"] = gates.pop(
        "random_group_control_completed", True
    )
    gates.pop("practical_gain_gate", None)
    payload["candidate_minus_structural"] = payload.pop("candidate_minus_random_group", {})
    payload.pop("candidate_minus_random_group_macro_f1", None)
    payload["gates"] = gates
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_data = {
        "schema_version": "q-attention.run-summary-data.v1",
        "experiment": payload.get("name"),
        "seed": payload.get("seed"),
        "selectors": payload.get("selectors", []),
        "data": payload.get("data", {}),
        "rows": payload.get("rows", []),
        "gates": payload.get("gates", {}),
        "provenance": payload.get("provenance", {}),
    }
    (run_dir / "run_summary.data").write_text(
        json.dumps(summary_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path = run_dir / "run_summary.md"
    if md_path.is_file():
        md_path.write_text(
            "\n".join(
                [
                    "# Q-EPVG Re-TACRED Formal Single Seed",
                    "",
                    f"- candidate: `{config['candidate']}`",
                    f"- matched classical control: `{config['matched_control']}`",
                    f"- structural phase control: `{config['structural_control']}`",
                    f"- selectors: `{', '.join(config['selectors'])}`",
                    f"- test used for training or selection: `{payload.get('test_used_for_training_or_selection', False)}`",
                    "",
                    "This report is a single complete seed under a frozen collaborator-only contract; no multi-seed claim is authorized by this handoff.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )


def main() -> int:
    cli = argparse.ArgumentParser(add_help=False)
    cli.add_argument("--output-dir", type=Path)
    cli.add_argument("--resume", type=Path)
    cli_args, _ = cli.parse_known_args()
    result = _base.main()
    # The base scheduler writes the complete marker only after all selector
    # artifacts exist. Rewrite naming metadata after that marker, without
    # changing metrics or checkpoints.
    if result == 0:
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        if cli_args.resume is not None:
            run_dir = cli_args.resume if cli_args.resume.is_absolute() else ROOT / cli_args.resume
        elif cli_args.output_dir is not None:
            run_dir = cli_args.output_dir if cli_args.output_dir.is_absolute() else ROOT / cli_args.output_dir
        else:
            run_root = ROOT / "runs" / config["name"]
            candidates = sorted(run_root.glob("*_seed13")) if run_root.is_dir() else []
            run_dir = candidates[-1] if candidates else None
        if run_dir is not None:
            _rewrite_summary(run_dir, config)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
