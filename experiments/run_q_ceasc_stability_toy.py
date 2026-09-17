"""Run the bounded stability-certified Q-CEASC mechanism screen.

The runner reuses the frozen Q-CEASC toy manifest, compares one-view and
two-view controls, and records the agreement gate as a causal intermediate.
It is toy-only and never authorizes a natural-task or collaborator run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.plugins.q_ceasc import QCEASCConfig, build_qceasc
from q_attention.plugins.q_ceasc_score import QCEASCScoreKernelConfig
from q_attention.plugins.q_ceasc_stability import (
    QCEASCStabilityConfig,
    build_qceasc_stability,
    build_qceasc_stability_score_kernel,
)

from run_q_ceasc_toy import (
    GENERATION_CASES,
    _build_manifest,
    _copy_quantum_state,
    _evaluate_generation,
    _move_tensors,
    _permutation_error,
    _shuffle_training_context,
    _shuffled_generation_case,
    _target_support,
    _train,
)


STABILITY_MODES = (
    "q_ceasc_stability",
    "classical_stability",
    "quantum_product_stability",
)
REPORT_MODES = (
    "q_ceasc",
    *STABILITY_MODES,
    "fixed_bank",
    "random_support",
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


def _stability_config(base: QCEASCConfig, raw: dict[str, Any]) -> QCEASCStabilityConfig:
    return QCEASCStabilityConfig(
        base=base,
        view_seed_stride=int(raw["view_seed_stride"]),
        agreement_threshold=float(raw["agreement_threshold"]),
    )


def _stability_summary(result: Any) -> dict[str, Any]:
    diagnostics = result.diagnostics
    return {
        "agreement_mean": float(diagnostics["agreement"].mean().detach().cpu()),
        "agreement_min": float(diagnostics["agreement"].min().detach().cpu()),
        "agreement_gate_mean": float(
            diagnostics["agreement_gate"].mean().detach().cpu()
        ),
        "agreement_gate_active_fraction": float(
            diagnostics["gate_active_fraction"].mean().detach().cpu()
        ),
        "view_one_residual_norm": float(
            diagnostics["view_one_residual_norm"].mean().detach().cpu()
        ),
        "view_two_residual_norm": float(
            diagnostics["view_two_residual_norm"].mean().detach().cpu()
        ),
        "gated_residual_norm": float(
            diagnostics["gated_residual_norm"].mean().detach().cpu()
        ),
    }


def _mismatched_view_summary(kernel: Any, current: dict[str, torch.Tensor], swapped: dict[str, torch.Tensor]) -> dict[str, float]:
    first = kernel.view_one.evaluate(
        current["query"], current["key"], current["valid"], current["entity"]
    )
    second = kernel.view_two.evaluate(
        swapped["query"], swapped["key"], swapped["valid"], swapped["entity"]
    )
    active = current["valid"] & ~current["entity"]
    weights = active.to(dtype=first.residual.dtype)
    numerator = (first.residual * second.residual * weights).sum(dim=-1)
    denominator = torch.linalg.vector_norm(first.residual, dim=-1) * torch.linalg.vector_norm(
        second.residual, dim=-1
    )
    agreement = torch.where(
        denominator > kernel.config.eps,
        numerator / denominator.clamp_min(kernel.config.eps),
        torch.zeros_like(numerator),
    ).clamp(-1.0, 1.0)
    threshold = kernel.config.agreement_threshold
    gate = (agreement - threshold).clamp_min(0.0)
    if threshold > 0.0:
        gate = gate / (1.0 - threshold)
    return {
        "agreement_mean": float(agreement.mean().detach().cpu()),
        "agreement_gate_mean": float(gate.mean().detach().cpu()),
    }


def _score_wrapper_smoke(config: QCEASCConfig, raw: dict[str, Any]) -> dict[str, Any]:
    score_config = QCEASCScoreKernelConfig(
        num_layers=1,
        num_heads=2,
        head_dim=config.key_dim,
        auxiliary_qubits=config.auxiliary_qubits,
        depth=config.depth,
        support_width=config.support_width,
        action_rank=config.action_rank,
        angle_scale=config.angle_scale,
        max_gain=config.max_gain,
        initial_gain=config.initial_gain,
        span_rcond=config.span_rcond,
        query_chunk_size=2,
        seed=config.seed,
    )
    kernel = build_qceasc_stability_score_kernel(
        "q_ceasc_stability",
        score_config,
        view_seed_stride=int(raw["view_seed_stride"]),
        agreement_threshold=float(raw["agreement_threshold"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 4401)
    query = torch.randn(1, 2, 4, config.key_dim, generator=generator)
    key = torch.randn(1, 2, 4, config.key_dim, generator=generator)
    attention = torch.ones(1, 4, dtype=torch.bool)
    subject = torch.tensor([[1, 0, 0, 0]], dtype=torch.bool)
    object_ = torch.tensor([[0, 1, 0, 0]], dtype=torch.bool)
    residual = kernel(
        query,
        key,
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
    )
    return {
        "finite": bool(torch.isfinite(residual).all()),
        "shape": list(residual.shape),
        "zero_sum_error": float(residual.sum(dim=-1).abs().max().detach().cpu()),
        "parameter_count": kernel.parameter_count,
    }


def evaluate_seed(
    seed: int,
    config: QCEASCConfig,
    *,
    raw: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    seed_config = replace(config, seed=config.seed + seed)
    manifest = _move_tensors(
        _build_manifest(
            seed,
            seed_config,
            training_batch_size=int(raw["training_batch_size"]),
            generation_context_size=int(raw["generation_context_size"]),
        ),
        device,
    )
    reference = build_qceasc("q_ceasc", seed_config).to(device=device)
    action_span = reference.action_span.detach().clone()
    stability_config = _stability_config(seed_config, raw)
    kernels: dict[str, Any] = {
        "q_ceasc": reference,
        "q_ceasc_stability": build_qceasc_stability(
            "q_ceasc_stability", stability_config, action_span=action_span
        ).to(device=device),
        "classical_stability": build_qceasc_stability(
            "classical_stability", stability_config, action_span=action_span
        ).to(device=device),
        "quantum_product_stability": build_qceasc_stability(
            "quantum_product_stability", stability_config, action_span=action_span
        ).to(device=device),
        "fixed_bank": build_qceasc(
            "fixed_bank", seed_config, action_span=action_span
        ).to(device=device),
        "random_support": build_qceasc(
            "random_support", seed_config, action_span=action_span
        ).to(device=device),
    }
    shuffled_manifest = _shuffle_training_context(manifest)
    training = {
        mode: _train(
            kernels[mode],
            manifest,
            steps=int(raw["steps"]),
            learning_rate=float(raw["learning_rate"]),
        )
        for mode in ("q_ceasc", *STABILITY_MODES)
    }
    _copy_quantum_state(kernels["q_ceasc"], kernels["fixed_bank"])

    generation: dict[str, Any] = {}
    stability_trace: dict[str, Any] = {}
    for case_name in GENERATION_CASES:
        case = manifest["generation"][case_name]
        case_results: dict[str, Any] = {
            mode: _evaluate_generation(kernels[mode], case)
            for mode in REPORT_MODES
        }
        shuffled_case, source_name = _shuffled_generation_case(manifest, case_name)
        shuffled_current = _evaluate_generation(kernels["q_ceasc_stability"], case)
        shuffled_swapped = _evaluate_generation(
            kernels["q_ceasc_stability"], shuffled_case
        )
        case_results["shuffled_context_source_case"] = source_name
        case_results["shuffled_context_output_difference"] = float(
            (
                shuffled_current["result"].residual
                - shuffled_swapped["result"].residual
            )
            .abs()
            .max()
            .detach()
            .cpu()
        )
        stability_result = case_results["q_ceasc_stability"]["result"]
        stability_trace[case_name] = _stability_summary(stability_result)
        mismatched = _mismatched_view_summary(
            kernels["q_ceasc_stability"], case, shuffled_case
        )
        stability_trace[case_name]["mismatched_view"] = mismatched
        for metrics in case_results.values():
            if isinstance(metrics, dict):
                metrics.pop("result", None)
        generation[case_name] = case_results

    permutation_error = _permutation_error(
        kernels["q_ceasc_stability"], manifest["generation"]["out_of_bank"]
    )
    parameter_counts = {mode: kernels[mode].parameter_count for mode in REPORT_MODES}
    stability_training = training["q_ceasc_stability"]
    trace = {
        "training": {
            "stage": "training",
            "granularity": "micro_batch",
            "fixed_case_id": f"qceasc-stability-seed{seed}-train-first-microbatch",
            "batch_id": 0,
            "context_size": int(manifest["training"]["key"].shape[1]),
            "input_shapes": {
                "query": list(manifest["training"]["query"].shape),
                "key": list(manifest["training"]["key"].shape),
                "valid": list(manifest["training"]["valid"].shape),
                "entity": list(manifest["training"]["entity"].shape),
            },
            "input_dtypes": {
                "query": str(manifest["training"]["query"].dtype),
                "key": str(manifest["training"]["key"].dtype),
            },
            "parameter_count": kernels["q_ceasc_stability"].parameter_count,
            "diagnostics": stability_training["diagnostics"],
            "gradient_norm": stability_training["gradient_norm"],
            "agreement_gate_mean": stability_training["diagnostics"].get(
                "agreement_gate_mean"
            ),
        },
        "generation": [],
        "evaluation": [],
        "diagnosis": {
            "stage": "diagnosis",
            "permutation_error": permutation_error,
            "training_active_parameter_count": stability_training[
                "active_parameter_count"
            ],
            "training_unused_parameter_count": stability_training[
                "unused_parameter_count"
            ],
            "stability_cases": stability_trace,
        },
    }
    for case_name, case_results in generation.items():
        for mode in REPORT_MODES:
            metrics = case_results[mode]
            entry = {
                "stage": "generation",
                "case_id": f"qceasc-stability-seed{seed}-{case_name}",
                "mode": mode,
                "selected_key": metrics["selected_key"],
                "wrong_top1_residual": metrics["wrong_top1_residual"],
                "rescue_position_support": metrics["rescue_position_support"],
                "residual_l1": metrics["residual_l1"],
                "out_of_bank_support_mass": metrics["out_of_bank_support_mass"],
                "finite": metrics["finite"],
            }
            if mode == "q_ceasc_stability":
                entry.update(stability_trace[case_name])
            trace["generation"].append(entry)
        stability_metrics = case_results["q_ceasc_stability"]
        trace["evaluation"].append(
            {
                "stage": "evaluation",
                "case_id": f"qceasc-stability-seed{seed}-{case_name}",
                "mode": "q_ceasc_stability",
                "projection_idempotence_error": stability_metrics[
                    "projection_idempotence_error"
                ],
                "span_leakage_norm": stability_metrics["span_leakage_norm"],
                "support_basis_entropy": stability_metrics["support_basis_entropy"],
                "auxiliary_state_norm": stability_metrics["auxiliary_state_norm"],
                "entangling_covariance_norm": stability_metrics[
                    "entangling_covariance_norm"
                ],
                "agreement_gate_mean": stability_trace[case_name][
                    "agreement_gate_mean"
                ],
            }
        )
    return {
        "seed": seed,
        "seed_config_seed": seed_config.seed,
        "parameter_counts": parameter_counts,
        "training": training,
        "generation": generation,
        "trace": trace,
        "score_wrapper_smoke": _score_wrapper_smoke(seed_config, raw),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device or raw.get("device", "cpu"))
    if device.type != "cpu":
        torch.cuda.set_device(device)
    base_config = QCEASCConfig(
        query_dim=int(raw["query_dim"]),
        key_dim=int(raw["key_dim"]),
        auxiliary_qubits=int(raw["auxiliary_qubits"]),
        depth=int(raw["depth"]),
        support_width=int(raw["support_width"]),
        action_rank=int(raw["action_rank"]),
        angle_scale=float(raw["angle_scale"]),
        max_gain=float(raw["max_gain"]),
        initial_gain=float(raw["initial_gain"]),
        span_rcond=float(raw["span_rcond"]),
    )
    started = time.perf_counter()
    results = [
        evaluate_seed(int(seed), base_config, raw=raw, device=device)
        for seed in raw["seeds"]
    ]
    elapsed = time.perf_counter() - started
    q_results = [
        result["generation"]["out_of_bank"]["q_ceasc_stability"]
        for result in results
    ]
    fixed_results = [
        result["generation"]["out_of_bank"]["fixed_bank"] for result in results
    ]
    corrupted_results = [
        result["generation"]["corrupted_primary"]["q_ceasc_stability"]
        for result in results
    ]
    gates = {
        "finite_and_invariants": all(
            item["finite"]
            and item["projection_idempotence_error"] < 2e-5
            and item["span_leakage_norm"] < 2e-5
            and item["mask_entity_zero_error"] < 1e-7
            and item["zero_sum_error"] < 1e-6
            for result in results
            for case in GENERATION_CASES
            for item in [result["generation"][case]["q_ceasc_stability"]]
        ),
        "nonzero_out_of_span_support": all(
            item["out_of_bank_support_mass"] > 1e-6 for item in q_results
        ),
        "out_of_bank_recall_above_fixed_bank": sum(
            item["selected_key"] == 1 for item in q_results
        )
        > sum(item["selected_key"] == 1 for item in fixed_results),
        "wrong_top1_safety": all(
            item["wrong_top1_residual"] <= 1e-8 for item in corrupted_results
        ),
        "agreement_gate_not_collapsed": all(
            result["trace"]["diagnosis"]["stability_cases"]["out_of_bank"][
                "agreement_gate_mean"
            ]
            > 0.05
            for result in results
        ),
        "agreement_responds_to_context_swap": all(
            result["trace"]["diagnosis"]["stability_cases"]["out_of_bank"][
                "mismatched_view"
            ]["agreement_gate_mean"]
            < result["trace"]["diagnosis"]["stability_cases"]["out_of_bank"][
                "agreement_gate_mean"
            ]
            + 1e-6
            for result in results
        ),
        "stability_differs_from_single_view": all(
            abs(
                result["generation"]["out_of_bank"]["q_ceasc_stability"][
                    "residual_l1"
                ]
                - result["generation"]["out_of_bank"]["q_ceasc"]["residual_l1"]
            )
            > 1e-8
            for result in results
        ),
        "matched_classical_replay_gap": all(
            abs(
                result["generation"]["out_of_bank"]["q_ceasc_stability"][
                    "residual_l1"
                ]
                - result["generation"]["out_of_bank"]["classical_stability"][
                    "residual_l1"
                ]
            )
            > 1e-8
            for result in results
        ),
        "matched_parameter_budget": all(
            result["parameter_counts"]["q_ceasc_stability"]
            == result["parameter_counts"]["classical_stability"]
            for result in results
        ),
        "permutation_equivariance": all(
            result["trace"]["diagnosis"]["permutation_error"] < 4e-5
            for result in results
        ),
        "score_wrapper_smoke": all(
            result["score_wrapper_smoke"]["finite"]
            and result["score_wrapper_smoke"]["zero_sum_error"] < 1e-6
            for result in results
        ),
        "bounded_runtime": elapsed <= float(raw["max_runtime_seconds"]),
    }
    report = {
        "schema_version": "q-attention.qceasc-stability-toy.v1",
        "project": "Q-Attention / q-ceasc-stability-certified-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "config": raw,
        "seeds": [int(seed) for seed in raw["seeds"]],
        "controls": list(REPORT_MODES),
        "plugin_metadata": build_qceasc_stability(
            "q_ceasc_stability", _stability_config(base_config, raw)
        ).metadata(),
        "results": results,
        "promotion_gate": {
            **gates,
            "status": "pass" if all(gates.values()) else "fail",
            "formal_data_authorized": False,
            "collaborator_handoff_authorized": False,
        },
        "resource_usage": {
            "elapsed_seconds": elapsed,
            "auxiliary_qubits": base_config.auxiliary_qubits,
            "two_view_parameter_multiplier": 2,
        },
        "provenance": {
            "git_revision": _git_revision(),
            "config_sha256": _sha256(args.config),
            "script_sha256": _sha256(Path(__file__)),
            "plugin_sha256": _sha256(
                Path(__file__).parents[1]
                / "src"
                / "q_attention"
                / "plugins"
                / "q_ceasc_stability.py"
            ),
        },
        "claim_boundary": (
            "Toy mechanism evidence only; no natural-task utility, novelty, "
            "reproducibility, or quantum-advantage claim."
        ),
    }
    trace = {
        "schema_version": "sample-trace.v1",
        "project": report["project"],
        "status": "observed",
        "selection_rule": (
            "First fixed training micro-batch plus two frozen generation fixtures "
            "are selected before outputs; a mismatched-view diagnostic swaps only "
            "the context key bank for the second view."
        ),
        "training": [item["trace"]["training"] for item in results],
        "generation": [
            entry
            for item in results
            for entry in item["trace"]["generation"]
        ],
        "evaluation": [
            entry
            for item in results
            for entry in item["trace"]["evaluation"]
        ],
        "diagnosis": [item["trace"]["diagnosis"] for item in results],
        "target_boundary": (
            "Targets and corrective support are evaluation-only metadata and do "
            "not enter the two-view gate or trace payload."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "qceasc_stability_toy_report.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    (args.output / "qceasc_stability_trace.json").write_text(
        json.dumps(trace, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if report["promotion_gate"]["status"] != "pass":
        raise SystemExit("Q-CEASC stability toy gate failed")


if __name__ == "__main__":
    main()
