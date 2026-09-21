"""Run the bounded grouped-counterfactual toy and cost/stability preflight.

This is synthetic validation only.  It does not load Re-TACRED, train a model,
or authorize a collaborator handoff.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.plugins.q_ceasc import QCEASCConfig  # noqa: E402
from q_attention.plugins.q_ceasc_counterfactual import (  # noqa: E402
    QCEASCCounterfactualConfig,
    CounterfactualQCEASC,
)
from q_attention.plugins.q_ceasc_grouped_counterfactual import (  # noqa: E402
    QCEASCGroupedCounterfactualConfig,
    GroupedCounterfactualQCEASC,
    build_fixed_group_manifest,
    build_seeded_random_group_manifest,
)


SEEDS = (13, 29, 53, 71, 89)
GROUP_SIZES = (1, 2, 4, 8)
CONTEXT_SIZES = (16, 32, 64)
HEAD_DIM = 6


def _sha256_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _make_inputs(seed: int, context_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    query = torch.randn(1, HEAD_DIM, generator=generator).to(device=device)
    key = torch.randn(1, context_size, HEAD_DIM, generator=generator).to(device=device)
    valid = torch.ones(1, context_size, dtype=torch.bool, device=device)
    valid[:, -1] = False
    entity = torch.zeros_like(valid)
    entity[:, 0] = True
    entity[:, min(1, context_size - 1)] = True
    return query, key, valid, entity


def _base_config(seed: int) -> QCEASCConfig:
    return QCEASCConfig(
        query_dim=HEAD_DIM,
        key_dim=HEAD_DIM,
        auxiliary_qubits=3,
        depth=2,
        support_width=2,
        action_rank=2,
        angle_scale=1.0,
        max_gain=0.25,
        initial_gain=0.05,
        span_rcond=1e-6,
        seed=13091 + int(seed),
        eps=1e-8,
    )


def _grouped(seed: int, group_size: int, mode: str) -> GroupedCounterfactualQCEASC:
    return GroupedCounterfactualQCEASC(
        QCEASCGroupedCounterfactualConfig(
            base=_base_config(seed), group_size=group_size, eps=1e-8
        ),
        mode,
    )


def _per_key(seed: int) -> CounterfactualQCEASC:
    return CounterfactualQCEASC(
        QCEASCCounterfactualConfig(base=_base_config(seed), leave_one_out_chunk_size=64),
        "q_ceasc_counterfactual",
    )


def _run(
    fn: Callable[[], Any],
    device: torch.device,
) -> tuple[Any, float, int]:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
    else:
        peak = 0
    return result, time.perf_counter() - started, peak


def _summary(result: Any, elapsed: float, peak_bytes: int, active: torch.Tensor) -> dict[str, Any]:
    residual = result.residual.detach()
    masked = (~active).to(device=residual.device)
    masked_error = float(residual.masked_select(masked).abs().max().cpu()) if bool(masked.any()) else 0.0
    zero_sum_error = float(residual.sum(dim=-1).abs().max().cpu())
    summary: dict[str, Any] = {
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": peak_bytes,
        "finite": bool(torch.isfinite(residual).all()),
        "mask_entity_zero_error": masked_error,
        "zero_sum_error": zero_sum_error,
    }
    if hasattr(result, "group_ids"):
        group_ids = result.group_ids.detach()
        present = group_ids >= 0
        active_group_ids = group_ids[present]
        active_group_count = int(torch.unique(active_group_ids).numel()) if active_group_ids.numel() else 0
        norms = result.diagnostics["group_influence_norms"].detach()
        summary.update(
            {
                "group_evaluation_count": int(result.diagnostics["group_evaluation_count"]),
                "active_group_count": active_group_count,
                "group_manifest_sha256": _sha256_tensor(group_ids),
                "group_influence_dispersion": float(norms.masked_select(present).var(unbiased=False).cpu())
                if bool(present.any())
                else 0.0,
                "target_free": bool(result.diagnostics["target_free"]),
                "context_only": bool(result.diagnostics["context_only"]),
            }
        )
    else:
        scores = result.diagnostics["influence_norms"].detach()
        summary.update(
            {
                "leave_one_out_count": int(result.diagnostics["leave_one_out_count"].max().cpu()),
                "influence_dispersion": float(scores.masked_select(active).var(unbiased=False).cpu()),
                "target_free": bool(result.diagnostics["target_free"]),
                "context_only": bool(result.diagnostics["context_only"]),
            }
        )
    return summary


def _invariants(summary: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if not summary["finite"]:
        failures.append("nonfinite_output")
    if summary["mask_entity_zero_error"] > 1e-6:
        failures.append("mask_entity_nonzero")
    if summary["zero_sum_error"] > 1e-5:
        failures.append("zero_sum_failure")
    if not summary["target_free"] or not summary["context_only"]:
        failures.append("target_or_context_contract_failure")
    return failures


def _fixture_checks(device: torch.device) -> dict[str, Any]:
    query = torch.zeros(1, HEAD_DIM, device=device)
    key = torch.zeros(1, 6, HEAD_DIM, device=device)
    key[:, 2] = 1.0
    key[:, 3] = 1.0
    key[:, 4] = -1.0
    key[:, 5] = -1.0
    valid = torch.ones(1, 6, dtype=torch.bool, device=device)
    entity = torch.zeros_like(valid)
    manifest = torch.tensor([[0, 0, 1, 2, 3, 3]], device=device)
    grouped = _grouped(13, 2, "q_ceasc_grouped_counterfactual").to(device)
    original = grouped.evaluate(query, key, valid, entity, manifest)
    permutation = torch.tensor([2, 3, 0, 1, 4, 5], device=device)
    permuted = grouped.evaluate(
        query,
        key[:, permutation],
        valid[:, permutation],
        entity[:, permutation],
        manifest[:, permutation],
    )
    inverse = torch.argsort(permutation)
    return {
        "symmetric_group_equal": bool(torch.allclose(original.group_scores[:, 1], original.group_scores[:, 2], atol=1e-5)),
        "permutation_equivariant": bool(torch.allclose(permuted.residual[:, inverse], original.residual, atol=1e-5)),
        "fixture_finite": bool(torch.isfinite(original.residual).all()),
        "fixture_manifest_sha256": _sha256_tensor(manifest),
    }


def run(device: torch.device) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    fatal: list[str] = []
    dispersion_improvements: list[dict[str, Any]] = []
    for seed in SEEDS:
        for context_size in CONTEXT_SIZES:
            query, key, valid, entity = _make_inputs(seed, context_size, device)
            active = valid & ~entity
            per_key, elapsed, peak = _run(lambda: _per_key(seed).to(device).evaluate(query, key, valid, entity), device)
            per_summary = _summary(per_key, elapsed, peak, active)
            fatal.extend(f"seed={seed},context={context_size}:per_key:{x}" for x in _invariants(per_summary))
            for group_size in GROUP_SIZES:
                positional = build_fixed_group_manifest(valid, entity, group_size=group_size)
                random_manifest = build_seeded_random_group_manifest(
                    valid, entity, group_size=group_size, seed=seed + 1000
                )
                quantum = _grouped(seed, group_size, "q_ceasc_grouped_counterfactual").to(device)
                classical = _grouped(seed, group_size, "classical_grouped_counterfactual").to(device)
                q_result, elapsed, peak = _run(
                    lambda: quantum.evaluate(query, key, valid, entity, positional), device
                )
                q_summary = _summary(q_result, elapsed, peak, active)
                fatal.extend(f"seed={seed},context={context_size},group={group_size}:quantum:{x}" for x in _invariants(q_summary))
                c_result, c_elapsed, c_peak = _run(
                    lambda: classical.evaluate(query, key, valid, entity, positional), device
                )
                c_summary = _summary(c_result, c_elapsed, c_peak, active)
                r_result, r_elapsed, r_peak = _run(
                    lambda: quantum.evaluate(query, key, valid, entity, random_manifest), device
                )
                r_summary = _summary(r_result, r_elapsed, r_peak, active)
                q_c_gap = float((q_result.residual - c_result.residual).abs().max().cpu())
                q_r_gap = float((q_result.residual - r_result.residual).abs().max().cpu())
                if q_c_gap <= 1e-7:
                    fatal.append(f"seed={seed},context={context_size},group={group_size}:exact_classical_replay")
                if group_size > 1 and q_r_gap <= 1e-7:
                    fatal.append(f"seed={seed},context={context_size},group={group_size}:exact_random_replay")
                if q_summary["group_evaluation_count"] <= 0:
                    fatal.append(f"seed={seed},context={context_size},group={group_size}:no_group_evaluations")
                improvement = per_summary["influence_dispersion"] - q_summary["group_influence_dispersion"]
                dispersion_improvements.append(
                    {"seed": seed, "context_size": context_size, "group_size": group_size, "delta_per_key_minus_grouped": improvement}
                )
                records.append(
                    {
                        "seed": seed,
                        "context_size": context_size,
                        "group_size": group_size,
                        "positional": q_summary,
                        "random": r_summary,
                        "classical": c_summary,
                        "per_key": per_summary,
                        "quantum_classical_max_abs_gap": q_c_gap,
                        "quantum_random_max_abs_gap": q_r_gap,
                        "dispersion_delta_per_key_minus_grouped": improvement,
                    }
                )
                del q_result, c_result, r_result, quantum, classical
            del per_key
    fixture = _fixture_checks(device)
    if not fixture["symmetric_group_equal"] or not fixture["permutation_equivariant"] or not fixture["fixture_finite"]:
        fatal.append("fixture_invariant_failure")
    best = max(dispersion_improvements, key=lambda item: item["delta_per_key_minus_grouped"])
    return {
        "schema_version": "qceasc-grouped-counterfactual-preflight.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": {"platform": platform.platform(), "python": sys.version.split()[0]},
        "device": str(device),
        "seeds": list(SEEDS),
        "group_sizes": list(GROUP_SIZES),
        "context_sizes": list(CONTEXT_SIZES),
        "contract": {
            "label_free_manifest": True,
            "positional_manifest_frozen_before_scoring": True,
            "random_manifest_frozen_before_scoring": True,
            "matched_classical_control": True,
            "per_key_predecessor_control": True,
            "target_free": True,
            "source_first_trace": {
                "synthetic_case_id": "grouped-counterfactual-preflight",
                "query_key_tensors": "seeded_synthetic_tensors",
                "group_manifest_sha256_per_record": True,
                "targets_enter_plugin": False,
            },
        },
        "fixture_checks": fixture,
        "records": records,
        "dispersion": {
            "best_delta_per_key_minus_grouped": best,
            "has_any_positive_delta": best["delta_per_key_minus_grouped"] > 0.0,
            "null_policy": "inconclusive_if_no_positive_delta",
        },
        "fatal_failures": fatal,
        "status": "fail" if fatal else "pass",
        "promotion": "bounded_toy_passes_cost_and_invariants_only; formal_single_seed_still_requires_explicit_author_gate" if not fatal else "blocked_by_fatal_gate",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    device = _device(args.device)
    result = run(device)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "status": result["status"]}, ensure_ascii=False))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
