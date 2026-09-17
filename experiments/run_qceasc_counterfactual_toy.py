"""Run the bounded Q-CEASC counterfactual-influence mechanism screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
SRC = ROOT / "src"
for path in (ROOT, SCRIPT_DIR, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from q_attention.plugins.q_ceasc import QCEASCConfig, build_qceasc  # noqa: E402
from q_attention.plugins.q_ceasc_counterfactual import (  # noqa: E402
    QCEASC_COUNTERFACTUAL_CONTROL_MODES,
    QCEASCCounterfactualConfig,
    QCEASCCounterfactualScoreKernelConfig,
    build_qceasc_counterfactual,
    build_qceasc_counterfactual_score_kernel,
)


GENERATION_CASES = ("unique_context_key", "symmetric_keys")
ALL_MODES = ("q_ceasc_counterfactual", "classical_counterfactual", "quantum_product_counterfactual")


def _seeded_normal(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def _move_tensors(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    return value


def _context_features(query: torch.Tensor, key: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    weights = active.to(dtype=key.dtype)
    mean_key = (key * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0)
    return torch.cat((query, mean_key), dim=-1)


def _target_support(
    query: torch.Tensor,
    key: torch.Tensor,
    active: torch.Tensor,
    *,
    kernel: Any,
    hidden_projection: torch.Tensor,
) -> torch.Tensor:
    context = _context_features(query, key, active)
    coefficients = torch.tanh(
        context @ hidden_projection.to(device=context.device, dtype=context.dtype)
    )
    dictionary = kernel.support_dictionary.to(device=context.device, dtype=context.dtype)
    raw = coefficients @ dictionary.transpose(0, 1)
    complement = kernel.orthogonal_complement.to(device=context.device, dtype=context.dtype)
    return raw @ complement


def _case(query: torch.Tensor, key: torch.Tensor, valid: torch.Tensor, entity: torch.Tensor) -> dict[str, Any]:
    return {"query": query, "key": key, "valid": valid, "entity": entity}


def _build_manifest(
    seed: int,
    config: QCEASCConfig,
    *,
    training_batch_size: int,
    generation_context_size: int,
) -> dict[str, Any]:
    if training_batch_size < 2 or generation_context_size < 4:
        raise ValueError("toy fixtures require at least two training and four context rows")
    reference = build_qceasc("q_ceasc", config)
    hidden_projection = _seeded_normal(
        (reference.context_dim, config.support_width), seed + 991
    ) / float(reference.context_dim) ** 0.5
    generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    train_query = torch.randn(
        training_batch_size, config.query_dim, generator=generator
    )
    train_key = torch.randn(
        training_batch_size, generation_context_size, config.key_dim, generator=generator
    )
    train_valid = torch.ones(training_batch_size, generation_context_size, dtype=torch.bool)
    train_valid[0, -1] = False
    train_entity = torch.zeros_like(train_valid)
    train_entity[1, 0] = True

    query = _seeded_normal((1, config.query_dim), seed + 1201)
    null = _seeded_normal((config.key_dim,), seed + 1202) * 0.15
    unique = _seeded_normal((config.key_dim,), seed + 1203) * 1.8
    unique_key = torch.stack(
        (null, unique, null, null, null, null)[:generation_context_size], dim=0
    ).unsqueeze(0)
    symmetric_vector = _seeded_normal((config.key_dim,), seed + 1204)
    symmetric_key = symmetric_vector.view(1, 1, -1).expand(
        1, generation_context_size, config.key_dim
    ).clone()
    valid = torch.ones(1, generation_context_size, dtype=torch.bool)
    entity = torch.zeros_like(valid)
    return {
        "seed": seed,
        "hidden_projection": hidden_projection,
        "training": {
            "query": train_query,
            "key": train_key,
            "valid": train_valid,
            "entity": train_entity,
        },
        "generation": {
            "unique_context_key": _case(query, unique_key, valid, entity),
            "symmetric_keys": _case(query.clone(), symmetric_key, valid.clone(), entity.clone()),
        },
    }


def _train(
    kernel: Any,
    manifest: dict[str, Any],
    *,
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    parameters = list(kernel.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    batch = manifest["training"]
    target = _target_support(
        batch["query"],
        batch["key"],
        batch["valid"] & ~batch["entity"],
        kernel=kernel,
        hidden_projection=manifest["hidden_projection"],
    ).detach()
    losses: list[float] = []
    last_result: Any = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        last_result = kernel.evaluate(batch["query"], batch["key"], batch["valid"], batch["entity"])
        predicted = torch.nn.functional.normalize(
            last_result.projected_support, dim=-1, eps=kernel.config.eps
        )
        expected = torch.nn.functional.normalize(target, dim=-1, eps=kernel.config.eps)
        loss = (1.0 - (predicted * expected).sum(dim=-1)).mean()
        loss = loss + 0.01 * last_result.residual.square().mean()
        loss.backward()
        active = [parameter for parameter in parameters if parameter.grad is not None]
        if not active or not all(torch.isfinite(parameter.grad).all() for parameter in active):
            raise RuntimeError("non-finite or absent active gradient during counterfactual toy training")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    assert last_result is not None
    gradient_norm = torch.sqrt(
        sum(parameter.grad.detach().square().sum() for parameter in parameters if parameter.grad is not None)
    )
    diagnostics = last_result.diagnostics
    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "active_parameter_count": sum(parameter.numel() for parameter in parameters if parameter.grad is not None),
        "unused_parameter_count": sum(parameter.numel() for parameter in parameters if parameter.grad is None),
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "diagnostics": {
            "influence_score_variance": float(diagnostics["influence_score_variance"].mean().cpu()),
            "influence_norm_variance": float(diagnostics["influence_norm_variance"].mean().cpu()),
            "counterfactual_effect_norm": float(diagnostics["counterfactual_effect_norm"].mean().cpu()),
            "mask_entity_zero_error": float(diagnostics["mask_entity_zero_error"].cpu()),
            "zero_sum_error": float(diagnostics["zero_sum_error"].max().cpu()),
        },
    }


def _metrics(result: Any) -> dict[str, Any]:
    diagnostics = result.diagnostics
    return {
        "residual_l1": float(result.residual[0].abs().sum().detach().cpu()),
        "selected_key": int(result.residual[0].argmax().item()),
        "influence_scores": diagnostics["influence_scores"][0].detach().cpu().tolist(),
        "influence_norms": diagnostics["influence_norms"][0].detach().cpu().tolist(),
        "influence_score_variance": float(diagnostics["influence_score_variance"].mean().cpu()),
        "influence_norm_variance": float(diagnostics["influence_norm_variance"].mean().cpu()),
        "counterfactual_effect_norm": float(diagnostics["counterfactual_effect_norm"].mean().cpu()),
        "out_of_span_support_norm": float(diagnostics["out_of_span_support_norm"].mean().cpu()),
        "mask_entity_zero_error": float(diagnostics["mask_entity_zero_error"].cpu()),
        "zero_sum_error": float(diagnostics["zero_sum_error"].max().cpu()),
        "projection_idempotence_error": diagnostics["projection_idempotence_error"],
        "span_leakage_norm": float(diagnostics["span_leakage_norm"].max().cpu()),
        "finite": bool(torch.isfinite(result.residual).all()),
        "result": result,
    }


def _permutation_error(kernel: Any, case: dict[str, Any]) -> float:
    size = case["key"].shape[1]
    permutation = torch.roll(torch.arange(size), shifts=1)
    original = kernel.evaluate(case["query"], case["key"], case["valid"], case["entity"])
    permuted = kernel.evaluate(
        case["query"], case["key"][:, permutation], case["valid"][:, permutation], case["entity"][:, permutation]
    )
    restored = torch.zeros_like(permuted.residual)
    restored[:, permutation] = permuted.residual
    return float((original.residual - restored).abs().max().detach().cpu())


def _score_smoke(raw: dict[str, Any]) -> dict[str, Any]:
    config = QCEASCCounterfactualScoreKernelConfig(
        num_layers=1,
        num_heads=2,
        head_dim=int(raw["score_head_dim"]),
        auxiliary_qubits=int(raw["score_auxiliary_qubits"]),
        depth=int(raw["depth"]),
        support_width=int(raw["support_width"]),
        action_rank=int(raw["action_rank"]),
        query_chunk_size=2,
        max_context_size=int(raw["score_max_context_size"]),
        seed=int(raw["score_seed"]),
    )
    kernel = build_qceasc_counterfactual_score_kernel("q_ceasc_counterfactual", config)
    generator = torch.Generator(device="cpu").manual_seed(int(raw["score_seed"]) + 1)
    query = torch.randn(2, 2, 5, config.head_dim, generator=generator)
    key = torch.randn(2, 2, 5, config.head_dim, generator=generator)
    attention = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
    subject = torch.tensor([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
    object_ = torch.tensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=torch.bool)
    residual = kernel(query, key, layer_index=0, attention_mask=attention, subject_mask=subject, object_mask=object_)
    return {
        "finite": bool(torch.isfinite(residual).all()),
        "shape": list(residual.shape),
        "zero_sum_error": float(residual.sum(dim=-1).abs().max().cpu()),
        "parameter_count": kernel.parameter_count,
    }


def evaluate_seed(seed: int, raw: dict[str, Any], device: torch.device) -> dict[str, Any]:
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
        seed=seed,
    )
    manifest = _move_tensors(
        _build_manifest(
            seed,
            base_config,
            training_batch_size=int(raw["training_batch_size"]),
            generation_context_size=int(raw["generation_context_size"]),
        ),
        device,
    )
    reference = build_qceasc("q_ceasc", base_config)
    action_span = reference.action_span.detach().clone().to(device=device)
    kernels = {
        mode: build_qceasc_counterfactual(
            mode,
            QCEASCCounterfactualConfig(base=base_config),
            action_span=action_span,
        ).to(device=device)
        for mode in ALL_MODES
    }
    direct = build_qceasc("q_ceasc", base_config, action_span=action_span).to(device=device)
    training = {mode: _train(kernels[mode], manifest, steps=int(raw["steps"]), learning_rate=float(raw["learning_rate"])) for mode in ALL_MODES}
    generation: dict[str, Any] = {}
    for case_name in GENERATION_CASES:
        case = manifest["generation"][case_name]
        case_results = {mode: _metrics(kernels[mode].evaluate(case["query"], case["key"], case["valid"], case["entity"])) for mode in ALL_MODES}
        direct_result = direct.evaluate(case["query"], case["key"], case["valid"], case["entity"])
        case_results["direct_q_ceasc"] = {
            "residual_l1": float(direct_result.residual[0].abs().sum().detach().cpu()),
            "finite": bool(torch.isfinite(direct_result.residual).all()),
            "result": direct_result,
        }
        candidate_result = case_results["q_ceasc_counterfactual"]["result"]
        case_results["q_ceasc_counterfactual"]["direct_replay_gap"] = float(
            (candidate_result.residual - direct_result.residual).abs().max().detach().cpu()
        )
        classical_result = case_results["classical_counterfactual"]["result"]
        case_results["q_ceasc_counterfactual"]["classical_replay_gap"] = float(
            (candidate_result.residual - classical_result.residual).abs().max().detach().cpu()
        )
        for metrics in case_results.values():
            metrics.pop("result", None)
        generation[case_name] = case_results
    permutation_error = _permutation_error(kernels["q_ceasc_counterfactual"], manifest["generation"]["unique_context_key"])
    unique_scores = generation["unique_context_key"]["q_ceasc_counterfactual"]["influence_scores"]
    null_scores = [score for index, score in enumerate(unique_scores) if index != 1]
    unique_key_gap = abs(unique_scores[1] - sum(null_scores) / max(len(null_scores), 1))
    symmetric_scores = generation["symmetric_keys"]["q_ceasc_counterfactual"]["influence_scores"]
    trace = {
        "training": {
            "stage": "training",
            "granularity": "micro_batch",
            "fixed_case_id": f"qceasc-counterfactual-seed{seed}-train-first-microbatch",
            "batch_id": 0,
            "context_size": int(manifest["training"]["key"].shape[1]),
            "parameter_count": kernels["q_ceasc_counterfactual"].parameter_count,
            "valid_context_mask": manifest["training"]["valid"].detach().cpu().int().tolist(),
            "diagnostics": training["q_ceasc_counterfactual"]["diagnostics"],
            "gradient_norm": training["q_ceasc_counterfactual"]["gradient_norm"],
            "target_boundary": "labels and targets are outside the plugin and trace",
        },
        "generation": [
            {
                "stage": "generation",
                "fixed_case_id": f"qceasc-counterfactual-seed{seed}-{case_name}",
                "case_name": case_name,
                "query": manifest["generation"][case_name]["query"].detach().cpu().tolist(),
                "key": manifest["generation"][case_name]["key"].detach().cpu().tolist(),
                "valid_context_mask": manifest["generation"][case_name]["valid"].detach().cpu().int().tolist(),
                "entity_mask": manifest["generation"][case_name]["entity"].detach().cpu().int().tolist(),
                "modes": {mode: generation[case_name][mode] for mode in ALL_MODES},
            }
            for case_name in GENERATION_CASES
        ],
        "diagnosis": {
            "stage": "diagnosis",
            "permutation_error": permutation_error,
            "unique_key_influence_gap": unique_key_gap,
            "symmetric_influence_scores": symmetric_scores,
            "symmetric_score_range": max(symmetric_scores) - min(symmetric_scores),
            "leave_one_out_contract": "each active key is removed once; invalid/entity keys receive zero action",
        },
    }
    return {
        "seed": seed,
        "parameter_counts": {mode: kernels[mode].parameter_count for mode in ALL_MODES},
        "training": training,
        "generation": generation,
        "trace": trace,
    }


def _git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device or raw.get("device", "cpu"))
    started = time.perf_counter()
    results = [evaluate_seed(int(seed), raw, device) for seed in raw["seeds"]]
    elapsed = time.perf_counter() - started
    unique = [result["generation"]["unique_context_key"] for result in results]
    symmetric = [result["generation"]["symmetric_keys"] for result in results]
    candidate_unique = [item["q_ceasc_counterfactual"] for item in unique]
    candidate_symmetric = [item["q_ceasc_counterfactual"] for item in symmetric]
    gates = {
        "finite_and_invariants": all(
            item["finite"] and item["mask_entity_zero_error"] < 1e-7 and item["zero_sum_error"] < 1e-6 and item["projection_idempotence_error"] < 2e-5 and item["span_leakage_norm"] < 2e-5
            for result in results for case in GENERATION_CASES for mode in ALL_MODES for item in [result["generation"][case][mode]]
        ),
        "leave_one_out_non_degenerate": all(item["counterfactual_effect_norm"] > 1e-6 and item["influence_score_variance"] > 1e-10 for item in candidate_unique),
        "symmetric_key_equivariance": all(result["trace"]["diagnosis"]["symmetric_score_range"] < 1e-5 for result in results),
        "unique_key_discriminates": all(result["trace"]["diagnosis"]["unique_key_influence_gap"] > 1e-6 for result in results),
        "classical_counterfactual_non_degenerate": all(item["counterfactual_effect_norm"] > 1e-6 and item["influence_score_variance"] > 1e-10 for result in results for item in [result["generation"]["unique_context_key"]["classical_counterfactual"]]),
        "quantum_differs_from_direct": all(item["direct_replay_gap"] > 1e-6 for item in candidate_unique),
        "quantum_differs_from_classical": all(item["classical_replay_gap"] > 1e-6 for item in candidate_unique),
        "matched_parameter_budget": all(len(set(result["parameter_counts"].values())) == 1 for result in results),
        "permutation_equivariance": all(result["trace"]["diagnosis"]["permutation_error"] < 3e-5 for result in results),
        "score_wrapper_smoke": _score_smoke(raw),
        "bounded_runtime": elapsed <= float(raw["max_runtime_seconds"]),
    }
    gates["score_wrapper_smoke_ok"] = bool(gates["score_wrapper_smoke"]["finite"] and gates["score_wrapper_smoke"]["zero_sum_error"] < 1e-6)
    gates["status"] = "pass" if all(value is True for key, value in gates.items() if key != "score_wrapper_smoke") else "fail"
    gates["formal_data_authorized"] = False
    gates["collaborator_handoff_authorized"] = False
    report = {
        "schema_version": "q-attention.qceasc-counterfactual-influence-toy.v1",
        "project": "Q-Attention / q-ceasc-counterfactual-influence-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "elapsed_seconds": elapsed,
        "cost_profile": {
            "toy_training_context_size": int(raw["training_batch_size"]),
            "toy_generation_context_size": int(raw["generation_context_size"]),
            "base_evaluations_per_generation_query": int(raw["generation_context_size"]) + 1,
            "score_wrapper_query_chunk_size": 2,
            "score_wrapper_max_context_size": int(raw["score_max_context_size"]),
            "peak_influence_elements_per_evaluation": int(raw["generation_context_size"]) * int(raw["key_dim"]),
            "formal_data_status": "not_authorized_until_context_cost_preflight",
        },
        "git_revision": _git_revision(),
        "config": raw,
        "seeds": [int(seed) for seed in raw["seeds"]],
        "controls": list(ALL_MODES) + ["direct_q_ceasc"],
        "results": results,
        "promotion_gate": gates,
        "formal_data_authorized": False,
        "collaborator_handoff_authorized": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "qceasc_counterfactual_toy_report.json"
    trace_path = args.output / "qceasc_counterfactual_trace.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    trace = {"schema_version": "sample-trace.v1", "project": report["project"], "selection_rule": "First fixed training micro-batch plus unique-key and symmetric-key generation fixtures frozen before outcomes.", "seeds": [result["trace"] for result in results]}
    trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    report["artifact_hashes"] = {"report": _sha256(report_path), "trace": _sha256(trace_path)}
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": gates["status"], "elapsed_seconds": elapsed, "report": str(report_path), "trace": str(trace_path), "promotion_gate": gates}, indent=2, ensure_ascii=True, default=str))


if __name__ == "__main__":
    main()
