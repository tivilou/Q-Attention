"""Run the bounded realistic cost and numerical-stability preflight.

This script exercises the isolated Q-CEASC counterfactual score wrapper on
synthetic tensors across the declared context-length ceiling. It measures the
leave-one-out evaluation budget, wall time, and peak device memory, then
probes float32/float64, masks, zero-norm keys, extreme magnitudes, and
backpropagation. It is a preflight only: it never authorizes a formal run.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

try:
    import resource
except ImportError:  # pragma: no cover - Windows staging worktrees
    resource = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.plugins.q_ceasc import QCEASCConfig  # noqa: E402
from q_attention.plugins.q_ceasc_counterfactual import (  # noqa: E402
    QCEASCCounterfactualConfig,
    QCEASCCounterfactualScoreKernelConfig,
    build_qceasc_counterfactual,
    build_qceasc_counterfactual_score_kernel,
)


DEFAULT_FORWARD_CASES = (
    {"batch_size": 1, "context_size": 16, "query_chunk_sizes": [1, 4, 16]},
    {"batch_size": 1, "context_size": 64, "query_chunk_sizes": [1, 4, 16, 64]},
    {"batch_size": 1, "context_size": 128, "query_chunk_sizes": [1, 4, 16, 64]},
    {"batch_size": 1, "context_size": 256, "query_chunk_sizes": [1, 4, 16, 64]},
    {"batch_size": 2, "context_size": 128, "query_chunk_sizes": [4, 16, 64]},
    {"batch_size": 2, "context_size": 256, "query_chunk_sizes": [4, 16, 64]},
)
DEFAULT_DTYPE_CASES = (
    {"dtype": "float32", "batch_size": 1, "context_size": 16, "query_chunk_sizes": [4, 16]},
    {"dtype": "float64", "batch_size": 1, "context_size": 16, "query_chunk_sizes": [4, 16]},
    {"dtype": "float64", "batch_size": 1, "context_size": 64, "query_chunk_sizes": [4, 16]},
)


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dtype(name: str) -> torch.dtype:
    values = {"float32": torch.float32, "float64": torch.float64}
    if name not in values:
        raise ValueError(f"unsupported dtype: {name}")
    return values[name]


def _device_memory(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {}
    index = device.index if device.index is not None else torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {
        "device_index": int(index),
        "device_name": properties.name,
        "total_bytes": int(properties.total_memory),
        "allocated_bytes": int(torch.cuda.memory_allocated(index)),
        "reserved_bytes": int(torch.cuda.memory_reserved(index)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
    }


def _rss_bytes() -> int:
    # Linux reports KiB; this is only a fallback when the selected device is CPU.
    if resource is None:
        return 0
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _seeded_inputs(
    *,
    batch_size: int,
    heads: int,
    context_size: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    query = torch.randn(
        batch_size, heads, context_size, head_dim, generator=generator, dtype=dtype
    ).mul(scale).to(device=device)
    key = torch.randn(
        batch_size, heads, context_size, head_dim, generator=generator, dtype=dtype
    ).mul(scale).to(device=device)
    attention = torch.ones(batch_size, context_size, dtype=torch.bool, device=device)
    attention[:, -1] = False
    subject = torch.zeros_like(attention)
    object_ = torch.zeros_like(attention)
    subject[:, 0] = True
    object_[:, min(1, context_size - 1)] = True
    return query, key, attention, subject, object_


def _score_config(
    raw: dict[str, Any], *, chunk_size: int, leave_one_out_chunk_size: int, dtype: torch.dtype
) -> QCEASCCounterfactualScoreKernelConfig:
    del dtype  # Parameters remain float32; inputs exercise the promoted path.
    return QCEASCCounterfactualScoreKernelConfig(
        num_layers=int(raw.get("num_layers", 1)),
        num_heads=int(raw.get("num_heads", 1)),
        head_dim=int(raw["head_dim"]),
        auxiliary_qubits=int(raw["auxiliary_qubits"]),
        depth=int(raw["depth"]),
        support_width=int(raw["support_width"]),
        action_rank=int(raw["action_rank"]),
        angle_scale=float(raw["angle_scale"]),
        max_gain=float(raw["max_gain"]),
        initial_gain=float(raw["initial_gain"]),
        span_rcond=float(raw["span_rcond"]),
        query_chunk_size=int(chunk_size),
        leave_one_out_chunk_size=int(leave_one_out_chunk_size),
        max_context_size=int(raw["max_context_size"]),
        seed=int(raw["seed"]),
    )


def _expected_cost(
    *,
    batch_size: int,
    heads: int,
    context_size: int,
    chunk_size: int,
    leave_one_out_chunk_size: int,
    head_dim: int,
) -> dict[str, int]:
    effective_query_chunk = min(context_size, chunk_size)
    effective_loo_chunk = min(context_size, leave_one_out_chunk_size)
    return {
        "active_context_evaluations_per_query_row": context_size + 1,
        "base_evaluations": batch_size * heads * context_size * (context_size + 1),
        "base_kernel_calls_per_head": (
            (context_size + effective_query_chunk - 1) // effective_query_chunk
        )
        * (1 + (context_size + effective_loo_chunk - 1) // effective_loo_chunk),
        "query_chunks_per_head": (
            context_size + effective_query_chunk - 1
        ) // effective_query_chunk,
        "leave_one_out_chunks_per_query_call": (
            context_size + effective_loo_chunk - 1
        ) // effective_loo_chunk,
        "peak_influence_elements_per_constructor_call": (
            batch_size * effective_query_chunk * context_size * head_dim
        ),
        "peak_expanded_context_elements": (
            batch_size
            * effective_query_chunk
            * effective_loo_chunk
            * context_size
            * head_dim
        ),
        "output_elements": batch_size * heads * context_size * context_size,
    }


def _run_forward_case(
    raw: dict[str, Any],
    *,
    device: torch.device,
    batch_size: int,
    context_size: int,
    chunk_size: int,
    leave_one_out_chunk_size: int,
    dtype_name: str,
    seed_offset: int,
) -> dict[str, Any]:
    dtype = _dtype(dtype_name)
    heads = int(raw.get("num_heads", 1))
    head_dim = int(raw["head_dim"])
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    before_rss = _rss_bytes()
    query, key, attention, subject, object_ = _seeded_inputs(
        batch_size=batch_size,
        heads=heads,
        context_size=context_size,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
        seed=int(raw["seed"]) + seed_offset,
    )
    config = _score_config(
        raw,
        chunk_size=chunk_size,
        leave_one_out_chunk_size=leave_one_out_chunk_size,
        dtype=dtype,
    )
    kernel = build_qceasc_counterfactual_score_kernel(
        "q_ceasc_counterfactual", config
    ).to(device=device)
    started = time.perf_counter()
    status = "ok"
    error = None
    output: torch.Tensor | None = None
    try:
        with torch.inference_mode():
            output = kernel(
                query,
                key,
                layer_index=0,
                attention_mask=attention,
                subject_mask=subject,
                object_mask=object_,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        finite = bool(output is not None and torch.isfinite(output).all())
        zero_sum_error = (
            float(output.sum(dim=-1).abs().max().detach().cpu())
            if output is not None
            else float("inf")
        )
        active = attention & ~subject & ~object_
        # The score tensor is (batch, heads, query_token, key_token):
        # invalid/entity keys occupy the last axis, while invalid attention
        # rows occupy the query axis.
        key_mask = (~active)[:, None, None, :]
        query_mask = (~attention)[:, None, :, None]
        masked = key_mask | query_mask
        masked_error = (
            float(output.masked_select(masked[:, None, :, None]).abs().max().detach().cpu())
            if output is not None and bool(masked.any())
            else 0.0
        )
        if not finite:
            status = "nonfinite"
    except RuntimeError as exc:
        status = "oom" if "out of memory" in str(exc).lower() else "error"
        error = str(exc)
        finite = False
        zero_sum_error = float("inf")
        masked_error = float("inf")
    elapsed = time.perf_counter() - started
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        memory = _device_memory(device)
        total = max(int(memory.get("total_bytes", 0)), 1)
        peak_fraction = float(memory["peak_allocated_bytes"] / total)
        reserved_fraction = float(memory["peak_reserved_bytes"] / total)
    else:
        memory = {"rss_before_bytes": before_rss, "rss_after_bytes": _rss_bytes()}
        peak_fraction = None
        reserved_fraction = None
    result = {
        "status": status,
        "error": error,
        "dtype": dtype_name,
        "batch_size": batch_size,
        "heads": heads,
        "context_size": context_size,
        "query_chunk_size": chunk_size,
        "leave_one_out_chunk_size": leave_one_out_chunk_size,
        "elapsed_seconds": elapsed,
        "finite": finite,
        "zero_sum_error": zero_sum_error,
        "masked_error": masked_error,
        "memory": memory,
        "peak_allocated_fraction": peak_fraction,
        "peak_reserved_fraction": reserved_fraction,
        "expected_cost": _expected_cost(
            batch_size=batch_size,
            heads=heads,
            context_size=context_size,
            chunk_size=chunk_size,
            leave_one_out_chunk_size=leave_one_out_chunk_size,
            head_dim=head_dim,
        ),
        "output_shape": list(output.shape) if output is not None else None,
    }
    del output, kernel, query, key, attention, subject, object_
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _run_stability_probes(raw: dict[str, Any], *, device: torch.device) -> dict[str, Any]:
    results: dict[str, Any] = {}
    head_dim = int(raw["head_dim"])
    leave_one_out_chunk_size = int(raw.get("leave_one_out_chunk_size", 64))
    context_size = min(
        int(raw.get("stability_context_size", 32)), int(raw["max_context_size"])
    )
    base_kwargs = {
        "query_dim": head_dim,
        "key_dim": head_dim,
        "auxiliary_qubits": int(raw["auxiliary_qubits"]),
        "depth": int(raw["depth"]),
        "support_width": int(raw["support_width"]),
        "action_rank": int(raw["action_rank"]),
        "angle_scale": float(raw["angle_scale"]),
        "max_gain": float(raw["max_gain"]),
        "initial_gain": float(raw["initial_gain"]),
        "span_rcond": float(raw["span_rcond"]),
        "seed": int(raw["seed"]) + 701,
    }
    for dtype_name in ("float32", "float64"):
        dtype = _dtype(dtype_name)
        query, key, valid, subject, object_ = _seeded_inputs(
            batch_size=2,
            heads=1,
            context_size=context_size,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            seed=int(raw["seed"]) + 801,
        )
        key[0, 0, 2] = 0.0
        valid[1].zero_()
        config = QCEASCConfig(**base_kwargs)
        kernel = build_qceasc_counterfactual(
            "q_ceasc_counterfactual",
            QCEASCCounterfactualConfig(
                base=config,
                leave_one_out_chunk_size=leave_one_out_chunk_size,
            ),
        ).to(device=device)
        # The direct constructor consumes one query vector per context row;
        # the score wrapper above expands the same inputs to all query tokens.
        result = kernel.evaluate(
            query[:, 0, 0, :], key[:, 0], valid, subject | object_
        )
        zero_norm_score = float(result.influence_scores[0, 2].abs().detach().cpu())
        empty_residual = float(result.residual[1].abs().max().detach().cpu())
        finite = bool(
            torch.isfinite(result.residual).all()
            and torch.isfinite(result.influence_vectors).all()
        )
        masked_error = float(
            result.residual.masked_select(
                ~(valid & ~(subject | object_))
            ).abs().max().detach().cpu()
        )
        del result, kernel, query, key, valid, subject, object_
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        results[f"mask_zero_norm_{dtype_name}"] = {
            "finite": finite,
            "zero_norm_influence_score": zero_norm_score,
            "empty_row_residual_max": empty_residual,
            "masked_error": masked_error,
        }

        query, key, valid, subject, object_ = _seeded_inputs(
            batch_size=1,
            heads=1,
            context_size=min(16, context_size),
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            seed=int(raw["seed"]) + 811,
            scale=1e6,
        )
        result = build_qceasc_counterfactual(
            "q_ceasc_counterfactual",
            QCEASCCounterfactualConfig(
                base=config,
                leave_one_out_chunk_size=leave_one_out_chunk_size,
            ),
        ).to(device=device).evaluate(
            query[:, 0, 0, :], key[:, 0], valid, subject | object_
        )
        results[f"extreme_scale_{dtype_name}"] = {
            "finite": bool(torch.isfinite(result.residual).all()),
            "zero_sum_error": float(
                result.residual.sum(dim=-1).abs().max().detach().cpu()
            ),
        }
        del result, query, key, valid, subject, object_
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Backpropagation intentionally uses a smaller context because every
    # leave-one-out graph is retained until backward.
    gradient_context = min(
        int(raw.get("gradient_context_size", 16)), int(raw["max_context_size"])
    )
    dtype = torch.float32
    query, key, valid, subject, object_ = _seeded_inputs(
        batch_size=1,
        heads=1,
        context_size=gradient_context,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
        seed=int(raw["seed"]) + 821,
    )
    query.requires_grad_()
    key.requires_grad_()
    config = QCEASCConfig(**base_kwargs)
    kernel = build_qceasc_counterfactual(
        "q_ceasc_counterfactual",
        QCEASCCounterfactualConfig(
            base=config,
            leave_one_out_chunk_size=leave_one_out_chunk_size,
        ),
    ).to(device=device)
    status = "ok"
    error = None
    try:
        result = kernel.evaluate(
            query[:, 0, 0, :], key[:, 0], valid, subject | object_
        )
        loss = result.residual.square().mean() + result.projected_support.square().mean()
        loss.backward()
        parameters = list(kernel.parameters())
        finite = bool(
            torch.isfinite(loss)
            and query.grad is not None
            and torch.isfinite(query.grad).all()
            and key.grad is not None
            and torch.isfinite(key.grad).all()
            and all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in parameters
            )
        )
        if not finite:
            status = "nonfinite"
        gradient_norm = float(
            torch.sqrt(
                sum(
                    parameter.grad.detach().square().sum()
                    for parameter in parameters
                    if parameter.grad is not None
                )
            ).detach().cpu()
        )
    except RuntimeError as exc:
        status = "oom" if "out of memory" in str(exc).lower() else "error"
        error = str(exc)
        finite = False
        gradient_norm = None
    results["backward_float32"] = {
        "status": status,
        "error": error,
        "finite": finite,
        "gradient_context_size": gradient_context,
        "gradient_norm": gradient_norm,
    }
    del kernel, query, key, valid, subject, object_
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def _gate_summary(
    raw: dict[str, Any], forward: list[dict[str, Any]], stability: dict[str, Any]
) -> dict[str, Any]:
    max_fraction = float(raw.get("max_peak_memory_fraction", 0.8))
    max_seconds = float(raw.get("max_case_seconds", 180.0))
    max_evaluations = int(raw.get("max_base_evaluations_per_case", 200000))
    context_ceiling = int(raw["max_context_size"])
    successful = [item for item in forward if item["status"] == "ok"]
    memory_ok = all(
        item["peak_allocated_fraction"] is None
        or item["peak_allocated_fraction"] <= max_fraction
        for item in successful
    ) and bool(successful)
    runtime_ok = all(
        item["elapsed_seconds"] <= max_seconds for item in successful
    ) and bool(successful)
    finite_ok = all(
        item["finite"]
        and item["masked_error"] <= 1e-6
        and item["zero_sum_error"] <= 2e-5
        for item in successful
    ) and bool(successful)
    budget_ok = all(
        item["expected_cost"]["base_evaluations"] <= max_evaluations
        for item in forward
    )
    long_context_ok = any(
        item["context_size"] == context_ceiling and item["status"] == "ok"
        for item in forward
    )
    stability_ok = all(
        item.get("finite", False)
        and item.get("zero_norm_influence_score", 1.0) <= 1e-6
        and item.get("empty_row_residual_max", 1.0) <= 1e-6
        and item.get("masked_error", 1.0) <= 1e-6
        for name, item in stability.items()
        if name.startswith("mask_zero_norm_")
    ) and all(
        item.get("finite", False) and item.get("zero_sum_error", 1.0) <= 2e-5
        for name, item in stability.items()
        if name.startswith("extreme_scale_")
    ) and stability.get("backward_float32", {}).get("status") == "ok" and stability.get(
        "backward_float32", {}
    ).get("finite", False)
    no_oom_or_error = bool(forward) and all(
        item["status"] == "ok" for item in forward
    )
    return {
        "status": "pass"
        if all(
            (
                no_oom_or_error,
                memory_ok,
                runtime_ok,
                finite_ok,
                budget_ok,
                long_context_ok,
                stability_ok,
            )
        )
        else "fail",
        "no_oom_or_error": no_oom_or_error,
        "memory_within_budget": memory_ok,
        "runtime_within_budget": runtime_ok,
        "forward_finite_and_invariant_safe": finite_ok,
        "evaluation_budget_within_declared_limit": budget_ok,
        "declared_context_ceiling_reached": long_context_ok,
        "numerical_stability": stability_ok,
        "thresholds": {
            "max_peak_memory_fraction": max_fraction,
            "max_case_seconds": max_seconds,
            "max_base_evaluations_per_case": max_evaluations,
            "max_context_size": context_ceiling,
        },
        "formal_data_authorized": False,
        "collaborator_handoff_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--resume-progress",
        action="store_true",
        help="reuse completed forward cases from preflight_progress.jsonl",
    )
    args = parser.parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device or raw.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    progress_path = args.output / "preflight_progress.jsonl"
    forward: list[dict[str, Any]] = []
    if args.resume_progress and progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("kind") == "forward_case" and isinstance(item.get("result"), dict):
                forward.append(item["result"])
    else:
        progress_path.unlink(missing_ok=True)
    leave_one_out_chunk_size = int(raw.get("leave_one_out_chunk_size", 64))
    completed_keys = {
        (
            item.get("dtype"),
            int(item.get("batch_size", -1)),
            int(item.get("context_size", -1)),
            int(item.get("query_chunk_size", -1)),
            int(item.get("leave_one_out_chunk_size", -1)),
        )
        for item in forward
    }

    def _record_forward(result: dict[str, Any]) -> None:
        forward.append(result)
        key = (
            result["dtype"],
            result["batch_size"],
            result["context_size"],
            result["query_chunk_size"],
            result["leave_one_out_chunk_size"],
        )
        completed_keys.add(key)
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": "forward_case", "result": result}, ensure_ascii=True) + "\n")

    started = time.perf_counter()
    for case_index, case in enumerate(raw.get("forward_cases", DEFAULT_FORWARD_CASES)):
        for chunk_index, chunk_size in enumerate(case["query_chunk_sizes"]):
            key = (
                "float32",
                int(case["batch_size"]),
                int(case["context_size"]),
                int(chunk_size),
                leave_one_out_chunk_size,
            )
            if key in completed_keys:
                continue
            _record_forward(
                _run_forward_case(
                    raw,
                    device=device,
                    batch_size=int(case["batch_size"]),
                    context_size=int(case["context_size"]),
                    chunk_size=int(chunk_size),
                    leave_one_out_chunk_size=leave_one_out_chunk_size,
                    dtype_name="float32",
                    seed_offset=case_index * 100 + chunk_index,
                )
            )
    for case_index, case in enumerate(raw.get("dtype_cases", DEFAULT_DTYPE_CASES)):
        for chunk_index, chunk_size in enumerate(case["query_chunk_sizes"]):
            key = (
                str(case["dtype"]),
                int(case["batch_size"]),
                int(case["context_size"]),
                int(chunk_size),
                leave_one_out_chunk_size,
            )
            if key in completed_keys:
                continue
            _record_forward(
                _run_forward_case(
                    raw,
                    device=device,
                    batch_size=int(case["batch_size"]),
                    context_size=int(case["context_size"]),
                    chunk_size=int(chunk_size),
                    leave_one_out_chunk_size=leave_one_out_chunk_size,
                    dtype_name=str(case["dtype"]),
                    seed_offset=900 + case_index * 100 + chunk_index,
                )
            )
    try:
        stability = _run_stability_probes(raw, device=device)
    except Exception as exc:  # preserve completed cost cases for diagnosis
        stability = {
            "runner_error": {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "finite": False,
            }
        }
    (args.output / "stability_probes.json").write_text(
        json.dumps(stability, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    elapsed = time.perf_counter() - started
    gate = _gate_summary(raw, forward, stability)
    report = {
        "schema_version": "q-attention.qceasc-counterfactual-influence-preflight.v1",
        "project": "Q-Attention / q-ceasc-counterfactual-influence-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "device_summary": _device_memory(device)
        if device.type == "cuda"
        else {"cpu_rss_peak_bytes": _rss_bytes()},
        "elapsed_seconds": elapsed,
        "git_revision": _git_revision(),
        "config": raw,
        "cost_model": {
            "base_evaluations_per_query_row": "N+1 (full context plus one leave-one-out recomputation per context position)",
            "score_wrapper_base_evaluations": "batch x heads x N x (N+1), independent of query_chunk_size",
            "score_wrapper_batched_base_kernel_calls": "batch x heads x ceil(N/query_chunk_size) x (1 + ceil(N/leave_one_out_chunk_size))",
            "query_chunk_role": "bounds the query rows evaluated in one constructor call; retained graphs can still scale with all query chunks during training",
            "leave_one_out_chunk_role": "bounds the counterfactual worlds materialized in one base call; it changes kernel-call overhead, not the exact N+1 evaluation budget",
            "peak_influence_tensor": "batch x min(N, query_chunk_size) x N x head_dim per constructor call",
            "peak_expanded_context": "batch x min(N, query_chunk_size) x min(N, leave_one_out_chunk_size) x N x head_dim per batched counterfactual call",
        },
        "forward_cases": forward,
        "stability_probes": stability,
        "formal_data_acceptance_criteria": {
            "all_declared_context_lengths_finish_without_oom_or_runtime_error": True,
            "float32_forward_and_representative_backward_are_finite": True,
            "float64_and_extreme_scale_probes_are_finite": True,
            "masked_entity_and_empty_rows_are_exactly_zero_within_tolerance": True,
            "peak_allocated_memory_fraction_within_threshold": float(
                raw.get("max_peak_memory_fraction", 0.8)
            ),
            "per_case_runtime_within_seconds": float(raw.get("max_case_seconds", 180.0)),
            "evaluation_budget_is_explicit_and_within_limit": int(
                raw.get("max_base_evaluations_per_case", 200000)
            ),
            "context_ceiling": int(raw["max_context_size"]),
        },
        "promotion_gate": gate,
        "formal_data_authorized": False,
        "collaborator_handoff_authorized": False,
    }
    report_path = args.output / "qceasc_counterfactual_preflight_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    report["artifact_hashes"] = {
        "report": _sha256(report_path),
        "progress": _sha256(progress_path),
        "stability": _sha256(args.output / "stability_probes.json"),
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": gate["status"],
                "elapsed_seconds": elapsed,
                "report": str(report_path),
                "promotion_gate": gate,
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
