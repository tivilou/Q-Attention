from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import sys

STAGING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGING_ROOT / "src"))

from q_attention.plugins.q_query_key_coherent_transport import (  # noqa: E402
    ClassicalQueryKeyCoherentTransportKernel,
    QueryKeyCoherentTransportConfig,
    QuantumQueryKeyCoherentTransportKernel,
)


def _load_config(path: Path) -> tuple[dict[str, Any], QueryKeyCoherentTransportConfig]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(QueryKeyCoherentTransportConfig)}
    values = {key: value for key, value in raw.items() if key in allowed}
    return raw, QueryKeyCoherentTransportConfig(**values)


def _fixture(seed: int, *, config: QueryKeyCoherentTransportConfig) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch, query_tokens, key_tokens = 32, 1, 5
    query = torch.randn(
        batch, config.num_heads, query_tokens, config.head_dim, generator=generator
    )
    key = torch.randn(
        batch, config.num_heads, key_tokens, config.head_dim, generator=generator
    )
    relation = torch.tensor([0.65, -0.35], dtype=query.dtype)
    if config.head_dim >= 2:
        key[:, :, 0, :2] = query[:, :, 0, :2] + relation
    else:
        key[:, :, 0, 0] = query[:, :, 0, 0] + relation[0]
    attention_mask = torch.ones(batch, key_tokens, dtype=torch.bool)
    subject_mask = torch.zeros_like(attention_mask)
    object_mask = torch.zeros_like(attention_mask)
    query_mask = torch.ones(batch, query_tokens, dtype=torch.bool)
    return query, key, attention_mask, subject_mask, object_mask, query_mask


def _forward(
    kernel: torch.nn.Module,
    fixture: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    query, key, attention_mask, subject_mask, object_mask, query_mask = fixture
    return kernel(
        query,
        key,
        scores=torch.zeros(query.shape[0], query.shape[1], query.shape[2], key.shape[2]),
        layer_index=0,
        attention_mask=attention_mask,
        subject_mask=subject_mask,
        object_mask=object_mask,
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
        target = output[:, :, :, 0].mean()
        competitor = output[:, :, :, 1:].amax(dim=-1).mean()
        loss = F.softplus(0.05 - target + competitor)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [parameter.grad.detach().norm() for parameter in kernel.parameters() if parameter.grad is not None]
        last_gradient_norm = torch.stack(gradients).norm()
        optimizer.step()
        last_loss = loss.detach()
    return {
        "loss": float(last_loss),
        "gradient_norm": float(last_gradient_norm),
    }


def _metrics(
    kernel: QuantumQueryKeyCoherentTransportKernel | ClassicalQueryKeyCoherentTransportKernel,
    fixture: tuple[torch.Tensor, ...],
) -> dict[str, float | int]:
    with torch.no_grad():
        output = _forward(kernel, fixture)
        raw = kernel.last_raw_score
        assert raw is not None
        query_mask = fixture[-1]
        context = fixture[2] & ~(fixture[3] | fixture[4])
        row_sum = output.masked_fill(~context[:, None, None, :], 0.0).sum(dim=-1).abs().amax()
        masked_max = output.masked_fill(context[:, None, None, :], 0.0).abs().amax()
        top1 = raw.mean(dim=1).squeeze(1).argmax(dim=-1)
        permutation = torch.tensor([2, 0, 4, 1, 3])
        permuted_fixture = (
            fixture[0], fixture[1][:, :, permutation], fixture[2][:, permutation],
            fixture[3][:, permutation], fixture[4][:, permutation], query_mask,
        )
        permuted_output = _forward(kernel, permuted_fixture)
        permutation_error = (permuted_output - output[:, :, :, permutation]).abs().amax()
        return {
            "pair_score_mean": float(raw.mean()),
            "pair_score_std": float(raw.std()),
            "score_variance": float(raw.var()),
            "residual_rms": float(output.square().mean().sqrt()),
            "target_key_top1": int((top1 == 0).float().mean() * 1000) / 1000,
            "context_zero_sum_max": float(row_sum),
            "masked_entity_zero_max": float(masked_max),
            "permutation_equivariance_error": float(permutation_error),
            "query_masked_zero_max": float(output.masked_fill(query_mask[:, None, :, None], 0.0).abs().amax()),
        }


def _disabled_metrics() -> dict[str, float | int]:
    return {
        "loss": 0.0,
        "gradient_norm": 0.0,
        "pair_score_mean": 0.0,
        "pair_score_std": 0.0,
        "score_variance": 0.0,
        "residual_rms": 0.0,
        "target_key_top1": 0.0,
        "context_zero_sum_max": 0.0,
        "masked_entity_zero_max": 0.0,
        "permutation_equivariance_error": 0.0,
        "query_masked_zero_max": 0.0,
    }


def _run_seed(
    seed: int,
    *,
    config: QueryKeyCoherentTransportConfig,
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    fixture = _fixture(seed, config=config)
    quantum = QuantumQueryKeyCoherentTransportKernel(config)
    classical = ClassicalQueryKeyCoherentTransportKernel(config)
    classical.load_state_dict(quantum.state_dict(), strict=False)
    quantum_train = _train(quantum, fixture, steps=steps, learning_rate=learning_rate)
    classical_train = _train(classical, fixture, steps=steps, learning_rate=learning_rate)
    quantum_metrics = _metrics(quantum, fixture)
    classical_metrics = _metrics(classical, fixture)
    gap = abs(quantum_metrics["pair_score_mean"] - classical_metrics["pair_score_mean"])
    return {
        "seed": seed,
        "selectors": {
            "disabled": _disabled_metrics(),
            "qk_coherent_transport_quantum": {**quantum_train, **quantum_metrics},
            "qk_coherent_transport_classical": {**classical_train, **classical_metrics},
        },
        "quantum_control_gap_mean": float(gap),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU toy screen for qk coherent transport")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_config, config = _load_config(args.config)
    seeds = [int(seed) for seed in raw_config.get("seeds", [config.seed])]
    steps = int(raw_config.get("toy_train_steps", 12))
    learning_rate = float(raw_config.get("learning_rate", 0.01))
    results = [
        _run_seed(seed, config=config, steps=steps, learning_rate=learning_rate)
        for seed in seeds
    ]
    report = {
        "schema_version": "q-attention.qk-coherent-transport.toy-result.v1",
        "experiment": raw_config.get("experiment", "qk_coherent_transport_toy"),
        "selectors": raw_config.get("selectors", ["qk_coherent_transport_quantum", "qk_coherent_transport_classical"]),
        "seeds": seeds,
        "config": raw_config,
        "torch_version": torch.__version__,
        "device": "cpu",
        "results": results,
        "checks": {
            "finite": all(
                metric["context_zero_sum_max"] < 1e-6
                and metric["masked_entity_zero_max"] == 0.0
                and metric["query_masked_zero_max"] == 0.0
                and metric["permutation_equivariance_error"] < 1e-5
                for result in results
                for name, metric in result["selectors"].items()
                if name != "disabled"
            ),
            "nonzero_score_variance": all(
                metric["score_variance"] > 1e-10
                for result in results
                for name, metric in result["selectors"].items()
                if name != "disabled"
            ),
            "quantum_control_separation": all(
                result["quantum_control_gap_mean"] > 1e-7 for result in results
            ),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if not all(report["checks"].values()):
        raise SystemExit("qk coherent transport toy checks failed")


if __name__ == "__main__":
    main()
