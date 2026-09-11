"""Run the bounded Q-CEASC mechanism screen.

The manifest fixes training and generation cases before any model output is
inspected.  This runner is toy-only: it emits compact mechanism traces and
never connects the plugin to the production attention hook.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import subprocess
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


BASE_MODES = (
    "q_ceasc",
    "quantum_product",
    "classical_span",
    "fixed_bank",
    "random_support",
)
MODES = BASE_MODES + ("shuffled_context",)
GENERATION_CASES = ("corrupted_primary", "out_of_bank")


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


def _context_features(
    query: torch.Tensor,
    key: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
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
    kernel,
    hidden_projection: torch.Tensor,
) -> torch.Tensor:
    context = _context_features(query, key, active)
    coefficients = torch.tanh(
        context
        @ hidden_projection.to(device=context.device, dtype=context.dtype)
    )
    dictionary = kernel.support_dictionary.to(
        device=context.device, dtype=context.dtype
    )
    raw = coefficients @ dictionary.transpose(0, 1)
    complement = kernel.orthogonal_complement.to(
        device=context.device, dtype=context.dtype
    )
    return raw @ complement


def _generation_case(
    query: torch.Tensor,
    key: torch.Tensor,
    valid: torch.Tensor,
    entity: torch.Tensor,
    *,
    primary_index: int,
    rescue_index: int,
) -> dict[str, Any]:
    baseline_logits = torch.einsum("bnd,bd->bn", key, query)
    if int(baseline_logits.argmax(dim=-1).item()) != primary_index:
        raise RuntimeError("frozen corrupted-primary fixture did not select primary key 0")
    return {
        "query": query,
        "key": key,
        "valid": valid,
        "entity": entity,
        "baseline_logits": baseline_logits,
        "primary_index": primary_index,
        "rescue_index": rescue_index,
    }


def _build_manifest(
    seed: int,
    config: QCEASCConfig,
    *,
    training_batch_size: int,
    generation_context_size: int,
) -> dict[str, Any]:
    if training_batch_size < 2:
        raise ValueError("training_batch_size must be at least 2 for cross-example shuffling")
    if generation_context_size < 6:
        raise ValueError("generation_context_size must be at least 6 for the frozen fixture")
    reference = build_qceasc("q_ceasc", config)
    hidden_projection = _seeded_normal(
        (reference.context_dim, config.support_width), seed + 991
    ) / float(reference.context_dim) ** 0.5

    generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    train_query = torch.randn(
        training_batch_size, config.query_dim, generator=generator, dtype=torch.float32
    )
    train_key = torch.randn(
        training_batch_size,
        generation_context_size,
        config.key_dim,
        generator=generator,
        dtype=torch.float32,
    )
    train_valid = torch.ones(
        training_batch_size, generation_context_size, dtype=torch.bool
    )
    train_valid[0, -1] = False
    train_entity = torch.zeros_like(train_valid)
    train_entity[1, 0] = True

    # The corrupted-primary fixture has no corrective out-of-span key.
    p = 3.0 * reference.span_basis[:, 0]
    distractor = _seeded_normal((config.key_dim,), seed + 71) * 0.15
    zero = torch.zeros_like(p)
    corrupted_key = torch.stack(
        (p, -p, distractor, -distractor, zero, zero),
        dim=0,
    )
    corrupted_key = torch.cat(
        (
            corrupted_key,
            torch.zeros(
                generation_context_size - 6, config.key_dim, dtype=corrupted_key.dtype
            ),
        ),
        dim=0,
    ).unsqueeze(0)
    corrupted_query = p.unsqueeze(0)
    corrupted_valid = torch.ones(
        1, generation_context_size, dtype=torch.bool
    )
    corrupted_entity = torch.zeros_like(corrupted_valid)
    corrupted = _generation_case(
        corrupted_query,
        corrupted_key,
        corrupted_valid,
        corrupted_entity,
        primary_index=0,
        rescue_index=1,
    )

    # The out-of-bank fixture adds a rescue direction in P_perp's complement.
    zero_mean = torch.zeros(1, generation_context_size, config.key_dim)
    generation_active = torch.ones_like(corrupted_valid)
    gold = _target_support(
        corrupted_query,
        zero_mean,
        generation_active,
        kernel=reference,
        hidden_projection=hidden_projection,
    )[0]
    rescue = 3.0 * gold - 0.5 * p
    out_of_bank_key = torch.stack(
        (p, rescue, -p, -rescue, distractor, -distractor + 0.2 * p),
        dim=0,
    )
    out_of_bank_key = torch.cat(
        (
            out_of_bank_key,
            torch.zeros(
                generation_context_size - 6, config.key_dim, dtype=out_of_bank_key.dtype
            ),
        ),
        dim=0,
    ).unsqueeze(0)
    out_of_bank = _generation_case(
        corrupted_query.clone(),
        out_of_bank_key,
        corrupted_valid.clone(),
        corrupted_entity.clone(),
        primary_index=0,
        rescue_index=1,
    )

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
            "corrupted_primary": corrupted,
            "out_of_bank": out_of_bank,
        },
    }


def _shuffle_training_context(manifest: dict[str, Any]) -> dict[str, Any]:
    training = manifest["training"]
    batch_size = training["query"].shape[0]
    source_rows = torch.roll(
        torch.arange(batch_size, device=training["key"].device), shifts=1
    )
    shuffled = dict(manifest)
    shuffled["training"] = {
        **training,
        "key": training["key"][source_rows],
        "valid": training["valid"][source_rows],
        "entity": training["entity"][source_rows],
        "source_rows": source_rows,
    }
    return shuffled


def _shuffled_generation_case(
    manifest: dict[str, Any],
    case_name: str,
) -> tuple[dict[str, Any], str]:
    source_name = {
        "corrupted_primary": "out_of_bank",
        "out_of_bank": "corrupted_primary",
    }[case_name]
    source = manifest["generation"][source_name]
    current = manifest["generation"][case_name]
    return {**source, "query": current["query"]}, source_name


def _context_summary(case: dict[str, Any]) -> torch.Tensor:
    """Return the label-free active-key summary used by the constructor."""
    active = case["valid"] & ~case["entity"]
    weights = active.to(dtype=case["key"].dtype)
    return (case["key"] * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0)


def _train(
    kernel,
    manifest: dict[str, Any],
    *,
    steps: int,
    learning_rate: float,
    target_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parameters = list(kernel.parameters())
    if not parameters:
        return {
            "initial_loss": 0.0,
            "final_loss": 0.0,
            "active_parameter_count": 0,
            "unused_parameter_count": 0,
            "gradient_norm": 0.0,
            "diagnostics": {},
        }
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    batch = manifest["training"]
    target_batch = (target_manifest or manifest)["training"]
    target = _target_support(
        target_batch["query"],
        target_batch["key"],
        target_batch["valid"] & ~target_batch["entity"],
        kernel=kernel,
        hidden_projection=manifest["hidden_projection"],
    ).detach()
    losses: list[float] = []
    last_result = None
    last_used: list[torch.nn.Parameter] = []
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        last_result = kernel.evaluate(
            batch["query"], batch["key"], batch["valid"], batch["entity"]
        )
        predicted = torch.nn.functional.normalize(
            last_result.projected_support,
            dim=-1,
            eps=kernel.config.eps,
        )
        expected = torch.nn.functional.normalize(target, dim=-1, eps=kernel.config.eps)
        loss = (1.0 - (predicted * expected).sum(dim=-1)).mean()
        loss = loss + 0.01 * last_result.residual.square().mean()
        loss.backward()
        last_used = [parameter for parameter in parameters if parameter.grad is not None]
        if not last_used or not all(
            torch.isfinite(parameter.grad).all() for parameter in last_used
        ):
            raise RuntimeError(
                "non-finite or absent active gradient during Q-CEASC toy training"
            )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    assert last_result is not None
    gradient_norm = torch.sqrt(
        sum(parameter.grad.detach().square().sum() for parameter in last_used)
    )
    diagnostics = {
        "auxiliary_state_norm": float(
            last_result.diagnostics["auxiliary_state_norm"].mean().cpu()
        ),
        "entangling_covariance_norm": float(
            last_result.diagnostics["entangling_covariance_norm"].mean().cpu()
        ),
        "support_basis_entropy": float(
            last_result.diagnostics["support_basis_entropy"].mean().cpu()
        ),
        "out_of_span_residual_norm": float(
            last_result.diagnostics["out_of_span_residual_norm"].mean().cpu()
        ),
        "mask_entity_zero_error": float(
            last_result.diagnostics["mask_entity_zero_error"].cpu()
        ),
        "zero_sum_error": float(
            last_result.diagnostics["zero_sum_error"].max().cpu()
        ),
    }
    return {
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "active_parameter_count": sum(parameter.numel() for parameter in last_used),
        "unused_parameter_count": sum(
            parameter.numel() for parameter in parameters if parameter.grad is None
        ),
        "gradient_norm": float(gradient_norm.cpu()),
        "diagnostics": diagnostics,
    }


def _copy_quantum_state(source, target) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    shared = {
        name: value for name, value in source_state.items() if name in target_state
    }
    target.load_state_dict(shared, strict=False)


def _evaluate_generation(kernel, case: dict[str, Any]) -> dict[str, Any]:
    result = kernel.evaluate(
        case["query"], case["key"], case["valid"], case["entity"]
    )
    residual = result.residual[0]
    projected = result.projected_support[0]
    return {
        "wrong_top1_residual": float(
            residual[case["primary_index"]].detach().cpu()
        ),
        "rescue_position_support": float(
            residual[case["rescue_index"]].detach().cpu()
        ),
        "selected_key": int(residual.argmax().item()),
        "out_of_bank_support_mass": float(
            result.diagnostics["out_of_span_support_norm"].mean().detach().cpu()
        ),
        "in_span_support_norm": float(
            result.diagnostics["in_span_support_norm"].mean().detach().cpu()
        ),
        "residual_l1": float(residual.abs().sum().detach().cpu()),
        "projection_idempotence_error": result.diagnostics[
            "projection_idempotence_error"
        ],
        "span_leakage_norm": float(
            result.diagnostics["span_leakage_norm"].max().detach().cpu()
        ),
        "support_basis_entropy": float(
            result.diagnostics["support_basis_entropy"].mean().detach().cpu()
        ),
        "auxiliary_state_norm": float(
            result.diagnostics["auxiliary_state_norm"].mean().detach().cpu()
        ),
        "entangling_covariance_norm": float(
            result.diagnostics["entangling_covariance_norm"].mean().detach().cpu()
        ),
        "mask_entity_zero_error": float(
            result.diagnostics["mask_entity_zero_error"].detach().cpu()
        ),
        "zero_sum_error": float(
            result.diagnostics["zero_sum_error"].max().detach().cpu()
        ),
        "finite": bool(torch.isfinite(result.residual).all()),
        "result": result,
    }


def _permutation_error(kernel, case: dict[str, Any]) -> float:
    size = case["key"].shape[1]
    permutation = torch.roll(
        torch.arange(size, device=case["key"].device), shifts=1
    )
    permuted = kernel.evaluate(
        case["query"],
        case["key"][:, permutation],
        case["valid"][:, permutation],
        case["entity"][:, permutation],
    )
    restored = torch.zeros_like(permuted.residual)
    restored[:, permutation] = permuted.residual
    original = kernel.evaluate(
        case["query"], case["key"], case["valid"], case["entity"]
    )
    return float((original.residual - restored).abs().max().detach().cpu())


def evaluate_seed(
    seed: int,
    config: QCEASCConfig,
    *,
    steps: int,
    learning_rate: float,
    training_batch_size: int,
    generation_context_size: int,
    device: torch.device,
) -> dict[str, Any]:
    seed_config = replace(config, seed=config.seed + seed)
    manifest = _move_tensors(
        _build_manifest(
            seed,
            seed_config,
            training_batch_size=training_batch_size,
            generation_context_size=generation_context_size,
        ),
        device,
    )
    action_span = build_qceasc("q_ceasc", seed_config).action_span.detach().clone().to(
        device=device
    )
    kernels = {
        mode: build_qceasc(
            "q_ceasc" if mode == "shuffled_context" else mode,
            seed_config,
            action_span=action_span,
        ).to(device=device)
        for mode in MODES
    }
    shuffled_manifest = _shuffle_training_context(manifest)
    training = {
        "q_ceasc": _train(
            kernels["q_ceasc"], manifest, steps=steps, learning_rate=learning_rate
        ),
        "quantum_product": _train(
            kernels["quantum_product"], manifest, steps=steps, learning_rate=learning_rate
        ),
        "classical_span": _train(
            kernels["classical_span"], manifest, steps=steps, learning_rate=learning_rate
        ),
        "shuffled_context": _train(
            kernels["shuffled_context"],
            shuffled_manifest,
            steps=steps,
            learning_rate=learning_rate,
            target_manifest=manifest,
        ),
    }
    _copy_quantum_state(kernels["q_ceasc"], kernels["fixed_bank"])

    generation: dict[str, Any] = {}
    for case_name in GENERATION_CASES:
        case = manifest["generation"][case_name]
        case_results = {
            "baseline_top1": int(case["baseline_logits"].argmax().item()),
            "q_ceasc": _evaluate_generation(kernels["q_ceasc"], case),
            "quantum_product": _evaluate_generation(kernels["quantum_product"], case),
            "classical_span": _evaluate_generation(kernels["classical_span"], case),
            "fixed_bank": _evaluate_generation(kernels["fixed_bank"], case),
            "random_support": _evaluate_generation(kernels["random_support"], case),
        }
        shuffled_case, source_name = _shuffled_generation_case(manifest, case_name)
        shuffled_current = _evaluate_generation(kernels["shuffled_context"], case)
        shuffled_swapped = _evaluate_generation(
            kernels["shuffled_context"], shuffled_case
        )
        case_results["shuffled_context"] = shuffled_swapped
        case_results["shuffled_context_source_case"] = source_name
        case_results["shuffled_context_key_difference"] = float(
            (case["key"] - shuffled_case["key"]).abs().max().detach().cpu()
        )
        case_results["shuffled_context_difference"] = float(
            (_context_summary(case) - _context_summary(shuffled_case))
            .abs()
            .max()
            .detach()
            .cpu()
        )
        case_results["shuffled_context_output_difference"] = float(
            (shuffled_current["result"].residual - shuffled_swapped["result"].residual)
            .abs()
            .max()
            .detach()
            .cpu()
        )
        shuffled_current.pop("result")
        quantum_result = case_results["q_ceasc"].pop("result")
        classical_result = case_results["classical_span"].pop("result")
        for mode in (
            "quantum_product",
            "fixed_bank",
            "random_support",
            "shuffled_context",
        ):
            case_results[mode].pop("result")
        case_results["q_ceasc"]["classical_replay_gap"] = float(
            (quantum_result.residual - classical_result.residual)
            .abs()
            .max()
            .detach()
            .cpu()
        )
        generation[case_name] = case_results

    permutation_error = _permutation_error(
        kernels["q_ceasc"], manifest["generation"]["out_of_bank"]
    )
    parameter_counts = {mode: kernels[mode].parameter_count for mode in MODES}
    trace = {
        "training": {
            "stage": "training",
            "granularity": "micro_batch",
            "fixed_case_id": f"qceasc-seed{seed}-train-first-microbatch",
            "batch_id": 0,
            "context_size": int(manifest["training"]["key"].shape[1]),
            "parameter_count": kernels["q_ceasc"].parameter_count,
            "valid_context_mask": manifest["training"]["valid"].detach().cpu().int().tolist(),
            "diagnostics": training["q_ceasc"]["diagnostics"],
            "gradient_norm": training["q_ceasc"]["gradient_norm"],
        },
        "scoring": [],
        "generation": [],
        "evaluation": [],
        "diagnosis": {
            "stage": "diagnosis",
            "effective_span_rank": int(kernels["q_ceasc"].span_basis.shape[1]),
            "effective_complement_rank": int(
                kernels["q_ceasc"].config.key_dim
                - kernels["q_ceasc"].span_basis.shape[1]
            ),
            "span_rcond": kernels["q_ceasc"].config.span_rcond,
            "permutation_error": permutation_error,
            "training_active_parameter_count": training["q_ceasc"]["active_parameter_count"],
            "training_unused_parameter_count": training["q_ceasc"]["unused_parameter_count"],
        },
    }
    for case_name, case_results in generation.items():
        for mode in MODES:
            metrics = case_results[mode]
            trace["scoring"].append(
                {
                    "stage": "scoring",
                    "case_id": f"qceasc-seed{seed}-{case_name}",
                    "mode": mode,
                    "residual_l1": metrics["residual_l1"],
                    "mask_entity_zero_error": metrics["mask_entity_zero_error"],
                    "zero_sum_error": metrics["zero_sum_error"],
                    "finite": metrics["finite"],
                }
            )
            trace["generation"].append(
                {
                    "stage": "generation",
                    "case_id": f"qceasc-seed{seed}-{case_name}",
                    "mode": mode,
                    "out_of_bank_support_mass": metrics["out_of_bank_support_mass"],
                    "selected_key": metrics["selected_key"],
                    "wrong_top1_residual": metrics["wrong_top1_residual"],
                    "rescue_position_support": metrics["rescue_position_support"],
                    "residual_l1": metrics["residual_l1"],
                    "support_mass_by_action_family": {
                        "in_span": metrics["in_span_support_norm"],
                        "out_of_span": metrics["out_of_bank_support_mass"],
                    },
                    "mask_entity_zero_sum_errors": {
                        "mask_entity": metrics["mask_entity_zero_error"],
                        "zero_sum": metrics["zero_sum_error"],
                    },
                    "context_swap": (
                        {
                            "source_case": case_results["shuffled_context_source_case"],
                            "key_difference": case_results["shuffled_context_key_difference"],
                            "summary_difference": case_results["shuffled_context_difference"],
                            "output_difference": case_results["shuffled_context_output_difference"],
                        }
                        if mode == "shuffled_context"
                        else None
                    ),
                    "finite": metrics["finite"],
                }
            )
            trace["evaluation"].append(
                {
                    "stage": "evaluation",
                    "case_id": f"qceasc-seed{seed}-{case_name}",
                    "mode": mode,
                    "projection_idempotence_error": metrics["projection_idempotence_error"],
                    "span_leakage_norm": metrics["span_leakage_norm"],
                    "support_basis_entropy": metrics["support_basis_entropy"],
                    "auxiliary_state_norm": metrics["auxiliary_state_norm"],
                    "entangling_covariance_norm": metrics["entangling_covariance_norm"],
                }
            )
    return {
        "seed": seed,
        "seed_config_seed": seed_config.seed,
        "parameter_counts": parameter_counts,
        "training": training,
        "generation": generation,
        "trace": trace,
    }


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
    config = QCEASCConfig(
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
        evaluate_seed(
            int(seed),
            config,
            steps=int(raw["steps"]),
            learning_rate=float(raw["learning_rate"]),
            training_batch_size=int(raw["training_batch_size"]),
            generation_context_size=int(raw["generation_context_size"]),
            device=device,
        )
        for seed in raw["seeds"]
    ]
    elapsed = time.perf_counter() - started
    out_of_bank_q = [
        result["generation"]["out_of_bank"]["q_ceasc"] for result in results
    ]
    out_of_bank_fixed = [
        result["generation"]["out_of_bank"]["fixed_bank"] for result in results
    ]
    corrupted_q = [
        result["generation"]["corrupted_primary"]["q_ceasc"] for result in results
    ]
    gates = {
        "finite_and_invariants": all(
            item["finite"]
            and item["projection_idempotence_error"] < 2e-5
            and (
                item["span_leakage_norm"] < 2e-5
                if mode != "fixed_bank"
                else item["out_of_bank_support_mass"] < 2e-5
            )
            and item["mask_entity_zero_error"] < 1e-7
            and item["zero_sum_error"] < 1e-6
            for result in results
            for case in GENERATION_CASES
            for mode in MODES
            for item in [result["generation"][case][mode]]
        ),
        "nonzero_out_of_span_support": all(
            item["out_of_bank_support_mass"] > 1e-6 for item in out_of_bank_q
        ),
        "out_of_bank_recall_above_fixed_bank": sum(
            item["selected_key"] == 1 for item in out_of_bank_q
        )
        > sum(item["selected_key"] == 1 for item in out_of_bank_fixed),
        "wrong_top1_safety": all(
            item["wrong_top1_residual"] <= 1e-8 for item in corrupted_q
        ),
        "not_exactly_replayed_by_classical_span": all(
            result["generation"]["out_of_bank"]["q_ceasc"]["classical_replay_gap"]
            > 1e-6
            for result in results
        ),
        "matched_parameter_budget": all(
            len(set(result["parameter_counts"].values())) == 1 for result in results
        ),
        "permutation_equivariance": all(
            result["trace"]["diagnosis"]["permutation_error"] < 3e-5
            for result in results
        ),
        "shuffled_context_executed": all(
            result["generation"][case]["shuffled_context_source_case"] != case
            and result["generation"][case]["shuffled_context_key_difference"] > 1e-8
            and result["generation"][case]["shuffled_context_difference"] > 1e-6
            and result["generation"][case]["shuffled_context_output_difference"] > 1e-8
            and result["generation"][case]["shuffled_context"]["finite"]
            for result in results
            for case in GENERATION_CASES
        ),
        "bounded_runtime": elapsed <= float(raw["max_runtime_seconds"]),
    }
    report = {
        "schema_version": "q-attention.qceasc-toy.v3",
        "project": "Q-Attention / q-ceasc-context-entangled-auxiliary-support-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "config": raw,
        "seeds": [int(seed) for seed in raw["seeds"]],
        "controls": list(MODES),
        "plugin_metadata": build_qceasc("q_ceasc", config).metadata(),
        "results": results,
        "promotion_gate": {
            **gates,
            "status": "pass" if all(gates.values()) else "fail",
            "formal_data_authorized": False,
            "collaborator_handoff_authorized": False,
        },
        "resource_usage": {
            "elapsed_seconds": elapsed,
            "auxiliary_qubits": config.auxiliary_qubits,
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
                / "q_ceasc.py"
            ),
        },
        "claim_boundary": "Toy mechanism evidence only; no natural-task utility, novelty, or quantum-advantage claim.",
    }
    trace = {
        "schema_version": "sample-trace.v1",
        "project": report["project"],
        "status": "observed",
        "selection_rule": "First fixed training micro-batch plus two generation fixtures are frozen before outcomes; shuffled_context preserves the current query and target semantics while swapping only the source context key bank.",
        "training": [item["trace"]["training"] for item in results],
        "scoring": [entry for item in results for entry in item["trace"]["scoring"]],
        "generation": [entry for item in results for entry in item["trace"]["generation"]],
        "evaluation": [entry for item in results for entry in item["trace"]["evaluation"]],
        "diagnosis": [item["trace"]["diagnosis"] for item in results],
        "target_boundary": "Targets, labels, baseline predictions, and gold corrective directions are evaluation-only metadata and are excluded from trace payloads.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "qceasc_toy_report.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    (args.output / "qceasc_trace.json").write_text(
        json.dumps(trace, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if report["promotion_gate"]["status"] != "pass":
        raise SystemExit("Q-CEASC toy gate failed")


if __name__ == "__main__":
    main()
