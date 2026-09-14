#!/usr/bin/env python3
"""Run one Q-CEASC selector worker for the formal scheduler."""

from __future__ import annotations

import argparse
import gc
import hashlib
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
    move_batch,
)
from q_attention.adapters import AttentionScoreKernelAdapter  # noqa: E402
from q_attention.adapters.encoder import resolve_module  # noqa: E402
from q_attention.experiments.batch_resume import (  # noqa: E402
    PAUSED_EXIT_CODE,
    TrainingMemoryPressure,
    TrainingPaused,
    atomic_write_json,
)
from q_attention.tasks.relation import load_relation_jsonl  # noqa: E402
from run_q_causal_value_evidence_relation_transfer import (  # noqa: E402
    hook_config,
    train_kernel,
)
from run_retacred_qceasc_formal_single_seed import (  # noqa: E402
    CUDA_OOM_EXIT_CODE,
    MEMORY_PRESSURE_EXIT_CODE,
    build_kernel,
    evaluate_selector,
    is_cuda_oom_error,
    selector_resume_contract,
    selector_resume_contract_compatible,
    metric_delta,
)


MIN_WORKER_FREE_MIB = 8 * 1024
MEMORY_PRESSURE_POLL_INTERVAL_STEPS = 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class CudaMemoryPressureMonitor:
    """Reclaim only this worker's idle allocator cache after complete updates."""

    def __init__(
        self,
        *,
        selector: str,
        enabled: bool,
        restart_on_pressure: bool,
        poll_interval_steps: int = MEMORY_PRESSURE_POLL_INTERVAL_STEPS,
    ) -> None:
        self.selector = selector
        self.enabled = enabled
        self.restart_on_pressure = restart_on_pressure
        self.poll_interval_steps = max(1, int(poll_interval_steps))

    @staticmethod
    def _mib(value: int) -> int:
        return int(value // (1024 * 1024))

    def _snapshot(self) -> dict[str, int]:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        return {
            "free_mib": self._mib(free_bytes),
            "total_mib": self._mib(total_bytes),
            "allocated_mib": self._mib(allocated),
            "reserved_mib": self._mib(reserved),
        }

    @staticmethod
    def _minimum_free_mib(total_mib: int) -> int:
        # Keep a modest allocation margin, capped so large cards are not
        # needlessly treated as under pressure.
        return max(512, min(2 * 1024, total_mib // 20))

    def __call__(
        self, *, epoch: int, total_batches: int, cursor: Any
    ) -> dict[str, Any] | None:
        if (
            not self.enabled
            or not torch.cuda.is_available()
            or int(cursor.global_step) % self.poll_interval_steps != 0
        ):
            return None
        try:
            before = self._snapshot()
        except Exception as exc:  # diagnostics must never stop a valid update loop
            print(
                json.dumps(
                    {
                        "event": "memory_pressure_sample_failed",
                        "selector": self.selector,
                        "error": str(exc),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return None
        minimum_free_mib = self._minimum_free_mib(before["total_mib"])
        cached_mib = max(0, before["reserved_mib"] - before["allocated_mib"])
        under_pressure = before["free_mib"] < minimum_free_mib
        fragmented = (
            before["free_mib"] < 2 * minimum_free_mib
            and cached_mib >= max(512, before["total_mib"] // 20)
        )
        if not under_pressure and not fragmented:
            return None

        # gc collects only unreachable objects; empty_cache returns only idle
        # blocks from this process's PyTorch allocator. Neither can release
        # active training tensors or memory owned by another CUDA process.
        gc.collect()
        torch.cuda.empty_cache()
        try:
            after = self._snapshot()
        except Exception as exc:  # reclaim succeeded even if the second sample is unavailable
            print(
                json.dumps(
                    {
                        "event": "memory_pressure_sample_failed",
                        "selector": self.selector,
                        "phase": "after_reclaim",
                        "error": str(exc),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return None
        event = {
            "event": "memory_pressure_reclaim",
            "selector": self.selector,
            "epoch": int(epoch),
            "batch": int(cursor.next_batch_index),
            "batches": int(total_batches),
            "global_step": int(cursor.global_step),
            "minimum_free_mib": minimum_free_mib,
            "trigger": "low_free" if under_pressure else "fragmented_cache",
            "before": before,
            "after": after,
            "reclaimed_reserved_mib": max(
                0, before["reserved_mib"] - after["reserved_mib"]
            ),
            "reclaimed_free_mib": max(0, after["free_mib"] - before["free_mib"]),
        }
        print(json.dumps(event, sort_keys=True), flush=True)
        if after["free_mib"] >= minimum_free_mib or not self.restart_on_pressure:
            return None
        return event


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


def write_case_study(
    *,
    model: torch.nn.Module,
    kernel: torch.nn.Module,
    records: list[Any],
    artifacts: Any,
    device: torch.device,
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    selector: str,
) -> None:
    """Record frozen validation examples and baseline-versus-selector deltas."""
    case_config = config.get("case_study", {})
    indices = [int(index) for index in case_config.get("record_indices", [0, 1, 2])]
    if not indices or any(index < 0 or index >= len(records) for index in indices):
        raise ValueError("case_study.record_indices must point inside the validation split")
    selected = [records[index] for index in indices]
    loader = make_relation_loader(
        selected,
        artifacts.vocab,
        artifacts.label_to_id,
        batch_size=len(selected),
    )
    batch = move_batch(next(iter(loader)), device)
    model.eval()
    kernel.eval()
    with torch.no_grad():
        baseline_logits = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["subject_mask"],
            batch["object_mask"],
        )
    captures: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    adapter = AttentionScoreKernelAdapter(model, model.score_module_paths, kernel)
    try:
        adapter.attach(hook_config(batch))
        for layer_index, path in enumerate(model.score_module_paths):
            def capture(
                _module: torch.nn.Module,
                inputs: tuple[object, ...],
                output: object,
                index: int = layer_index,
            ) -> None:
                if isinstance(inputs[0], torch.Tensor) and isinstance(output, torch.Tensor):
                    captures[index] = (inputs[0].detach(), output.detach())

            handles.append(resolve_module(model, path).register_forward_hook(capture))
        with torch.no_grad():
            selector_logits = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["subject_mask"],
                batch["object_mask"],
            )
    finally:
        for handle in handles:
            handle.remove()
        adapter.remove()
    baseline_predictions = baseline_logits.argmax(dim=-1).cpu().tolist()
    selector_predictions = selector_logits.argmax(dim=-1).cpu().tolist()
    labels = batch["labels"].cpu().tolist()
    residual_rms = {
        str(layer): float((steered - base).square().mean().sqrt().cpu())
        for layer, (base, steered) in captures.items()
    }
    cases = []
    for local_index, (record_index, record) in enumerate(zip(indices, selected)):
        cases.append(
            {
                "split": str(case_config.get("split", "valid")),
                "record_index": record_index,
                "tokens": list(record.tokens),
                "label": record.label,
                "label_id": int(labels[local_index]),
                "baseline_prediction": int(baseline_predictions[local_index]),
                "selector_prediction": int(selector_predictions[local_index]),
                "baseline_correct": bool(baseline_predictions[local_index] == labels[local_index]),
                "selector_correct": bool(selector_predictions[local_index] == labels[local_index]),
                "logit_delta_l2": float(
                    (selector_logits[local_index] - baseline_logits[local_index]).norm().cpu()
                ),
                "residual_rms_by_layer": residual_rms,
            }
        )
    payload = {
        "schema_version": "q-attention.q-ceasc-case-study.v1",
        "selector": selector,
        "status": "observed",
        "selection_rule": "Indices are frozen in the formal config before training; this file is written after selector completion and is never used for optimization or selection.",
        "split": str(case_config.get("split", "valid")),
        "record_indices": indices,
        "kernel_metadata": kernel.metadata(),
        "provenance": {
            "git_revision": _git_revision(),
            "plugin_sha256": _sha256(
                ROOT / "src" / "q_attention" / "plugins" / "q_ceasc_score.py"
            ),
            "plugin_core_sha256": _sha256(
                ROOT / "src" / "q_attention" / "plugins" / "q_ceasc.py"
            ),
        },
        "cases": cases,
    }
    (output_dir / "case_study.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Keep the project-wide sample-trace contract alongside the legacy,
    # Q-CEASC-specific case-study payload used by existing reports.
    stages_by_case = []
    for case in cases:
        sample_id = f"{selector}:{case['split']}:{case['record_index']}"
        stages_by_case.append(
            {
                "sample_id": sample_id,
                "split_position": case["record_index"],
                "stages": [
                    {
                        "stage": "data",
                        "status": "observed",
                        "observed_fields": {
                            "split": case["split"],
                            "tokens": case["tokens"],
                        },
                    },
                    {
                        "stage": "preprocess",
                        "status": "observed",
                        "observed_fields": {"token_count": len(case["tokens"])},
                    },
                    {
                        "stage": "training",
                        "status": "observed",
                        "observed_fields": {
                            "selector": selector,
                            "epochs": int(config["kernel"]["epochs"]),
                        },
                    },
                    {"stage": "retrieval", "status": "not_applicable"},
                    {
                        "stage": "scoring",
                        "status": "observed",
                        "observed_fields": {
                            "logit_delta_l2": case["logit_delta_l2"],
                            "residual_rms_by_layer": case["residual_rms_by_layer"],
                        },
                    },
                    {
                        "stage": "selection",
                        "status": "observed",
                        "observed_fields": {
                            "baseline_prediction": case["baseline_prediction"],
                            "selector_prediction": case["selector_prediction"],
                        },
                    },
                    {
                        "stage": "context",
                        "status": "observed",
                        "observed_fields": {"token_count": len(case["tokens"])},
                    },
                    {"stage": "generation", "status": "not_applicable"},
                    {
                        "stage": "evaluation",
                        "status": "observed",
                        "observed_fields": {
                            "label_id": case["label_id"],
                            "baseline_correct": case["baseline_correct"],
                            "selector_correct": case["selector_correct"],
                        },
                    },
                    {
                        "stage": "diagnosis",
                        "status": "observed",
                        "observed_fields": {"kernel_metadata": payload["kernel_metadata"]},
                    },
                ],
            }
        )
    sample_trace = {
        "schema_version": "sample-trace.v1",
        "trace_id": f"{output_dir.name}:{selector}",
        "experiment": {
            "run_id": output_dir.parent.parent.name,
            "dataset": "retacred.valid",
            "code_revision": _git_revision(),
            "config_sha256": _sha256(config_path),
            "model_identity": "relation-transformer-q-ceasc",
            "seed": int(config["seed"]),
        },
        "sample_selection": {
            "rule": payload["selection_rule"],
            "population_scope": "retacred.valid",
            "seed": int(config["seed"]),
            "selected_count": len(stages_by_case),
            "selected_sample_ids": [item["sample_id"] for item in stages_by_case],
        },
        "coverage": {
            "data": "observed",
            "preprocess": "observed",
            "training": "observed",
            "retrieval": "not_applicable",
            "scoring": "observed",
            "selection": "observed",
            "context": "observed",
            "generation": "not_applicable",
            "evaluation": "observed",
            "diagnosis": "observed",
        },
        "samples": stages_by_case,
    }
    (output_dir / "sample_trace.json").write_text(
        json.dumps(sample_trace, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume this selector from its compatible post-update checkpoint",
    )
    parser.add_argument("--checkpoint-every-batches", type=int, default=50)
    parser.add_argument(
        "--pair-chunk-size",
        default="all",
        help="maximum pair chunk, or all for every pair in the physical micro-batch",
    )
    parser.add_argument(
        "--pair-chunk-divisor", type=int, default=1,
        help="divide an all-pairs chunk by this positive power-of-two fallback divisor",
    )
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--activation-checkpointing", type=int, choices=(0, 1), default=None)
    parser.add_argument(
        "--adaptive-memory",
        action="store_true",
        help="allow the parent scheduler to resume this worker with a lower memory tier",
    )
    parser.add_argument(
        "--elastic-resume",
        action="store_true",
        help="allow an explicit execution-only contract migration during resume",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.log_every_batches <= 0:
        raise ValueError("--log-every-batches must be positive")
    if args.checkpoint_every_batches <= 0:
        raise ValueError("--checkpoint-every-batches must be positive")
    if args.pair_chunk_size == "all":
        args.pair_chunk_size = None
    else:
        try:
            args.pair_chunk_size = int(args.pair_chunk_size)
        except ValueError as exc:
            raise ValueError("--pair-chunk-size must be a positive integer or all") from exc
        if args.pair_chunk_size <= 0:
            raise ValueError("--pair-chunk-size must be positive")
    if args.pair_chunk_divisor <= 0:
        raise ValueError("--pair-chunk-divisor must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    check_worker_gpu_capacity(args.device)
    device = choose_device(args.device)
    artifacts = load_relation_run(args.baseline_dir, device)
    data_dir = args.data_dir
    train_records = load_relation_jsonl(data_dir / "train.jsonl")
    valid_records = load_relation_jsonl(data_dir / "valid.jsonl")
    test_records = load_relation_jsonl(data_dir / "test.jsonl")
    kernel_config = config["kernel"]
    logical_batch_size = int(kernel_config["batch_size"])
    if args.micro_batch_size is None:
        args.micro_batch_size = logical_batch_size
    if args.micro_batch_size <= 0 or args.micro_batch_size > logical_batch_size:
        raise ValueError("--micro-batch-size must be in (0, logical batch size]")
    if args.micro_batch_size * args.gradient_accumulation_steps != logical_batch_size:
        raise ValueError(
            "--micro-batch-size * --gradient-accumulation-steps must equal the logical batch size"
        )
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
        pair_chunk_divisor=args.pair_chunk_divisor,
        activation_checkpointing=(
            None
            if args.activation_checkpointing is None
            else bool(args.activation_checkpointing)
        ),
    ).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_args = argparse.Namespace(
        batch_size=logical_batch_size,
        epochs=int(kernel_config["epochs"]),
        kernel_lr=float(kernel_config["lr"]),
        log_every_batches=args.log_every_batches,
        batch_resume=True,
        resume=args.resume,
        checkpoint_every_batches=args.checkpoint_every_batches,
        resume_contract=selector_resume_contract(
            config_path=args.config,
            baseline_dir=args.baseline_dir,
            data_dir=data_dir,
            selector=args.selector,
            seed=args.seed,
            pair_chunk_size=args.pair_chunk_size,
            pair_chunk_divisor=args.pair_chunk_divisor,
            micro_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            activation_checkpointing=args.activation_checkpointing,
            adaptive_memory=args.adaptive_memory,
        ),
        resume_contract_compatible=(
            selector_resume_contract_compatible if args.elastic_resume else None
        ),
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        memory_pressure_monitor=CudaMemoryPressureMonitor(
            selector=args.selector,
            enabled=device.type == "cuda",
            restart_on_pressure=args.adaptive_memory,
        ),
    )
    try:
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
    except TrainingPaused:
        print(
            json.dumps(
                {"event": "run_paused", "selector": args.selector}, sort_keys=True
            ),
            flush=True,
        )
        return PAUSED_EXIT_CODE
    except TrainingMemoryPressure as exc:
        event = {
            "event": "memory_pressure_restart",
            "selector": args.selector,
            "checkpoint": str(args.output_dir / "checkpoints" / "latest.pt"),
            "diagnostics": exc.diagnostics,
        }
        atomic_write_json(args.output_dir / "memory_pressure_event.json", event)
        print(json.dumps(event, sort_keys=True), flush=True)
        return MEMORY_PRESSURE_EXIT_CODE
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
    metadata: dict[str, Any] = kernel.metadata()
    trainable_parameters = sum(parameter.numel() for parameter in kernel.parameters())
    torch.save(
        {"state_dict": kernel.state_dict(), "metadata": metadata},
        args.output_dir / "best_kernel_with_metadata.pt",
    )
    write_case_study(
        model=artifacts.model,
        kernel=kernel,
        records=valid_records,
        artifacts=artifacts,
        device=device,
        config=config,
        config_path=args.config,
        output_dir=args.output_dir,
        selector=args.selector,
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
    try:
        raise SystemExit(main())
    except BaseException as exc:
        if is_cuda_oom_error(exc):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            selector = sys.argv[sys.argv.index("--selector") + 1] if "--selector" in sys.argv else "<unknown>"
            print(
                json.dumps(
                    {"event": "cuda_oom", "selector": selector, "error": str(exc)},
                    sort_keys=True,
                ),
                flush=True,
            )
            raise SystemExit(CUDA_OOM_EXIT_CODE) from exc
        raise
