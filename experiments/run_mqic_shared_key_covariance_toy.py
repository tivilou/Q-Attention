"""Bounded CPU toy screen for MQIC shared-key query covariance."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

STAGING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGING_ROOT / "src"))

from q_attention.plugins.q_multi_query_covariance import (  # noqa: E402
    ClassicalMultiQueryCovarianceKernel,
    MultiQueryCovarianceConfig,
    QuantumMultiQueryCovarianceKernel,
)


def _load_config(path: Path) -> tuple[dict[str, Any], MultiQueryCovarianceConfig]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(MultiQueryCovarianceConfig)}
    values = {key: value for key, value in raw.items() if key in allowed}
    return raw, MultiQueryCovarianceConfig(**values)


def _fixture(seed: int, *, config: MultiQueryCovarianceConfig) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch, tokens = 16, 5
    query = torch.randn(batch, config.num_heads, tokens, config.head_dim, generator=generator)
    key = torch.randn(batch, config.num_heads, tokens, config.head_dim, generator=generator)
    attention = torch.ones(batch, tokens, dtype=torch.bool)
    attention[:, -1] = False
    subject = torch.zeros(batch, tokens, dtype=torch.bool)
    object_ = torch.zeros(batch, tokens, dtype=torch.bool)
    subject[:, 0] = True
    object_[:, 1] = True
    query_mask = attention.clone()
    if config.head_dim >= 2:
        key[:, :, 2, :2] = query[:, :, 0, :2] + torch.tensor(
            [0.7, -0.35], dtype=query.dtype
        )
    return query, key, attention, subject, object_, query_mask


def _forward(kernel: torch.nn.Module, fixture: tuple[torch.Tensor, ...]) -> torch.Tensor:
    query, key, attention, subject, object_, query_mask = fixture
    return kernel(
        query,
        key,
        scores=torch.zeros(query.shape[0], query.shape[1], query.shape[2], key.shape[2]),
        layer_index=0,
        attention_mask=attention,
        subject_mask=subject,
        object_mask=object_,
        query_mask=query_mask,
    )


def _train(
    kernel: torch.nn.Module,
    fixture: tuple[torch.Tensor, ...],
    *,
    steps: int,
    learning_rate: float,
) -> dict[str, float]:
    optimizer = torch.optim.Adam(kernel.parameters(), lr=learning_rate)
    last_loss = torch.tensor(0.0)
    last_gradient_norm = torch.tensor(0.0)
    for _ in range(steps):
        output = _forward(kernel, fixture)
        target = output[:, :, 0, 2].mean()
        competitor = output[:, :, 0, 3].mean()
        covariance = kernel.last_connected_covariance
        assert covariance is not None
        loss = F.softplus(0.01 - target + competitor) + 0.01 * covariance.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [
            parameter.grad.detach().norm()
            for parameter in kernel.parameters()
            if parameter.grad is not None
        ]
        last_gradient_norm = torch.stack(gradients).norm()
        optimizer.step()
        last_loss = loss.detach()
    return {"loss": float(last_loss), "gradient_norm": float(last_gradient_norm)}


def _permutation_fixture(seed: int, *, config: MultiQueryCovarianceConfig) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed + 1000)
    batch, query_tokens, key_tokens = 8, 2, 5
    query = torch.randn(batch, config.num_heads, query_tokens, config.head_dim, generator=generator)
    key = torch.randn(batch, config.num_heads, key_tokens, config.head_dim, generator=generator)
    attention = torch.ones(batch, key_tokens, dtype=torch.bool)
    subject = torch.zeros(batch, query_tokens, dtype=torch.bool)
    object_ = torch.zeros(batch, query_tokens, dtype=torch.bool)
    subject[:, 0] = True
    object_[:, 1] = True
    query_mask = torch.ones(batch, query_tokens, dtype=torch.bool)
    return query, key, attention, subject, object_, query_mask


def _metrics(
    kernel: QuantumMultiQueryCovarianceKernel | ClassicalMultiQueryCovarianceKernel,
    fixture: tuple[torch.Tensor, ...],
) -> dict[str, float | int]:
    query, key, attention, subject, object_, query_mask = fixture
    start = time.perf_counter()
    with torch.no_grad():
        output = _forward(kernel, fixture)
        repeat = _forward(kernel, fixture)
    runtime = time.perf_counter() - start
    connected = kernel.last_connected_covariance
    independent = kernel.last_independent_product
    assert connected is not None and independent is not None
    valid_context = attention[:, None, None, :] & ~subject[:, None, None, :] & ~object_[:, None, None, :]
    active_role = (subject | object_) & query_mask
    row_sum = (output * valid_context.to(output.dtype)).sum(dim=-1)
    row_sum = (
        row_sum.masked_fill(~active_role[:, None, :], 0.0).abs().amax()
        if bool(active_role.any())
        else torch.tensor(0.0)
    )
    entity_zero = output.masked_fill(~(subject | object_)[:, None, None, :], 0.0).abs().amax()
    mask_zero = output.masked_fill(attention[:, None, None, :], 0.0).abs().amax()
    query_zero = output.masked_fill(query_mask[:, None, :, None], 0.0).abs().amax()
    swapped = _forward(kernel, (query, key, attention, object_, subject, query_mask))

    permutation = torch.tensor([2, 4, 0, 3, 1])
    perm_fixture = _permutation_fixture(int(query[0, 0, 0, 0].abs().item() * 10000), config=kernel.config)
    permuted_fixture = (
        perm_fixture[0],
        perm_fixture[1][:, :, permutation],
        perm_fixture[2][:, permutation],
        perm_fixture[3],
        perm_fixture[4],
        perm_fixture[5],
    )
    base_perm = _forward(kernel, perm_fixture)
    permuted = _forward(kernel, permuted_fixture)
    return {
        "pair_count": kernel.last_pair_count,
        "pair_score_std": float(connected.std()),
        "connected_covariance_std": float(connected.std()),
        "score_variance": float(connected.var()),
        "independent_product_gap": float((connected - independent).abs().mean()),
        "residual_rms": float(output.square().mean().sqrt()),
        "context_zero_sum_max": float(row_sum),
        "entity_mask_zero_max": float(entity_zero),
        "attention_mask_zero_max": float(mask_zero),
        "query_mask_zero_max": float(query_zero),
        "query_pair_swap_error": float((swapped + output).abs().amax()),
        "key_permutation_equivariance_error": float(
            (permuted - base_perm[:, :, :, permutation]).abs().amax()
        ),
        "deterministic_replay_error": float((repeat - output).abs().amax()),
        "runtime_seconds": runtime,
    }


def _run_seed(
    seed: int,
    *,
    config: MultiQueryCovarianceConfig,
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    fixture = _fixture(seed, config=config)
    quantum = QuantumMultiQueryCovarianceKernel(config)
    classical = ClassicalMultiQueryCovarianceKernel(config)
    classical.load_state_dict(quantum.state_dict(), strict=True)
    quantum_train = _train(quantum, fixture, steps=steps, learning_rate=learning_rate)
    classical_train = _train(classical, fixture, steps=steps, learning_rate=learning_rate)
    quantum_metrics = _metrics(quantum, fixture)
    classical_metrics = _metrics(classical, fixture)
    quantum_output = _forward(quantum, fixture)
    classical_output = _forward(classical, fixture)
    return {
        "seed": seed,
        "selectors": {
            "mqic_quantum": {**quantum_train, **quantum_metrics},
            "mqic_classical": {**classical_train, **classical_metrics},
        },
        "quantum_control_gap": float((quantum_output - classical_output).abs().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU toy screen for MQIC")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_config, config = _load_config(args.config)
    seeds = [int(seed) for seed in raw_config.get("seeds", [13, 29, 53])]
    steps = int(raw_config.get("toy_train_steps", 8))
    learning_rate = float(raw_config.get("learning_rate", 0.01))
    results = [
        _run_seed(seed, config=config, steps=steps, learning_rate=learning_rate)
        for seed in seeds
    ]
    metrics = [metric for result in results for metric in result["selectors"].values()]
    checks = {
        "finite": all(
            all(torch.isfinite(torch.tensor(value)) for value in metric.values() if isinstance(value, (float, int)))
            for metric in metrics
        ),
        "nonzero_connected_covariance": all(metric["connected_covariance_std"] > 1e-8 for metric in metrics),
        "connected_vs_independent_gap": all(metric["independent_product_gap"] > 1e-8 for metric in metrics),
        "quantum_control_separation": all(result["quantum_control_gap"] > 1e-8 for result in results),
        "mask_and_gauge_contract": all(
            metric["context_zero_sum_max"] < 1e-5
            and metric["entity_mask_zero_max"] == 0.0
            and metric["attention_mask_zero_max"] == 0.0
            and metric["query_mask_zero_max"] == 0.0
            for metric in metrics
        ),
        "symmetry_and_replay": all(
            metric["query_pair_swap_error"] < 5e-5
            and metric["key_permutation_equivariance_error"] < 1e-5
            and metric["deterministic_replay_error"] == 0.0
            for metric in metrics
        ),
        "gradient_flow": all(metric["gradient_norm"] > 1e-8 for metric in metrics),
    }
    report = {
        "schema_version": "q-attention.mqic-shared-key-query-covariance.toy-result.v1",
        "experiment": raw_config.get("experiment", "mqic_shared_key_query_covariance_toy"),
        "selectors": ["mqic_quantum", "mqic_classical"],
        "seeds": seeds,
        "config": raw_config,
        "torch_version": torch.__version__,
        "device": "cpu",
        "results": results,
        "checks": checks,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if not all(checks.values()):
        raise SystemExit("MQIC toy checks failed")


if __name__ == "__main__":
    main()
