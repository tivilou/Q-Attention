"""Bounded Q-EPVG toy kernel with explicit observable/path controls.

This module is intentionally toy/preflight scope.  It exposes the three
declared Pauli observables and three intervention paths without changing the
production attention stack.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn as nn


EPVG_OBSERVABLES = ("zz", "zz_xx", "trainable_pauli_mix")
EPVG_PATHS = ("value_only", "score_value", "query")
EPVG_CONTROLS = ("quantum", "classical", "random_parity")


@dataclass(frozen=True)
class QEPVGConfig:
    num_layers: int = 1
    num_heads: int = 1
    head_dim: int = 4
    observable: str = "zz_xx"
    path: str = "value_only"
    seed: int = 911
    gate_scale: float = 1.0
    score_gain: float = 0.2
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.num_layers <= 0 or self.num_heads <= 0 or self.head_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if self.observable not in EPVG_OBSERVABLES:
            raise ValueError(f"observable must be one of {EPVG_OBSERVABLES}")
        if self.path not in EPVG_PATHS:
            raise ValueError(f"path must be one of {EPVG_PATHS}")
        if self.gate_scale <= 0.0 or self.score_gain < 0.0 or self.eps <= 0.0:
            raise ValueError("invalid scale, gain, or epsilon")


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
    masked = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    result = torch.softmax(masked, dim=-1) * mask.to(scores.dtype)
    return result / result.sum(dim=-1, keepdim=True).clamp_min(eps)


class EntangledParityValueGatingKernel(nn.Module):
    """Q-EPVG observable/path matrix used by the bounded preflight."""

    mechanism_name = "q_epvg"

    def __init__(self, config: QEPVGConfig) -> None:
        super().__init__()
        self.config = config
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        # Fixed reducer and value/query maps are frozen by the toy contract.
        self.register_buffer("reducer", torch.eye(config.head_dim))
        fixed = torch.eye(config.head_dim)
        fixed = fixed + 0.05 * torch.randn(fixed.shape, generator=generator)
        self.register_buffer("linear_map", fixed)
        # Every selector needs a trainable intervention parameter.  The
        # positive log-scale keeps the declared initial gate scale exact while
        # allowing fixed-observable variants to learn their intervention
        # strength during transfer training.
        self.gate_log_scale = nn.Parameter(torch.tensor(math.log(config.gate_scale), dtype=fixed.dtype))
        self.pauli_logits = nn.Parameter(torch.zeros(3))
        self.control = "quantum"
        self.last_trace: dict[str, torch.Tensor] = {}

    def set_control(self, control: str) -> None:
        if control not in EPVG_CONTROLS:
            raise ValueError(f"control must be one of {EPVG_CONTROLS}")
        self.control = control

    def metadata(self) -> dict[str, Any]:
        return {
            "id": "q_epvg",
            "version": "0.1.0",
            "type": "attention_toy_transformer",
            "insertion_point": "bounded_toy_attention",
            "hypothesis": "entangled parity observables can route value, score, or query paths",
            "input_schema": "query,key,value,scores,attention_mask,query_mask",
            "output_schema": "output plus observable/gate/attention trace",
            "requires": ["frozen reducer", "fixed linear mapping"],
            "conflicts": ["production Re-TACRED selectors", "label-aware routing"],
            "deterministic": True,
            "resource_estimate": {"data_qubits": 2, "observables": list(EPVG_OBSERVABLES)},
            "failure_signatures": ["observable collapse", "mask violation", "classical replay", "nonfinite gradients"],
            "observability_contract": {
                "trace_schema": "sample-trace.v1",
                "stages": ["training", "scoring", "generation", "evaluation"],
                "required_intermediates": [
                    "query", "key", "value", "base_attention", "zz", "xx",
                    "observable", "theta", "gate", "score_adjustment", "attention",
                    "routed_values", "query_update", "output", "matched_classical",
                    "random_parity",
                ],
                "target_boundary": "labels and targets are post-evaluation only",
            },
        }

    def _observables(self, query: torch.Tensor, key: torch.Tensor, control: str) -> tuple[torch.Tensor, ...]:
        q_angle = query.mean(dim=-1)
        k_angle = key.mean(dim=-1)
        qz, kz = torch.cos(q_angle), torch.cos(k_angle)
        qx, kx = torch.sin(q_angle), torch.sin(k_angle)
        zz = qz.unsqueeze(-1) * kz.unsqueeze(-2)
        if control == "quantum":
            xx = torch.sin(q_angle.unsqueeze(-1) + k_angle.unsqueeze(-2))
        else:
            xx = qx.unsqueeze(-1) * kx.unsqueeze(-2)
        if control == "random_parity":
            indices = torch.arange(key.shape[-2], device=key.device)
            xx = xx * torch.where(indices.remainder(2).eq(0), 1.0, -1.0)[None, None, None, :]
        return zz, xx, (zz + xx) / 2.0

    def _components(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: torch.Tensor,
        query_mask: torch.Tensor | None,
        control: str,
    ) -> tuple[torch.Tensor, ...]:
        if query_mask is None:
            query_mask = torch.ones(
                query.shape[0], query.shape[2], dtype=torch.bool, device=query.device
            )
        key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
        q_mask = query_mask[:, None, :, None].to(dtype=query.dtype)
        zz, xx, zz_xx = self._observables(query, key, control)
        if self.config.observable == "zz":
            observable = zz
        elif self.config.observable == "zz_xx":
            observable = zz_xx
        else:
            weights = torch.softmax(self.pauli_logits, dim=0)
            observable = weights[0] * zz + weights[1] * xx + weights[2] * zz_xx
        observable = observable.clamp(-1.0, 1.0) * key_mask.to(observable.dtype) * q_mask
        theta = torch.exp(self.gate_log_scale) * observable
        return key_mask, q_mask, zz, xx, observable, theta

    def before_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
        layer_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if attention_mask is None:
            raise ValueError("Q-EPVG requires attention_mask")
        if self.config.path != "query":
            return query, key, value
        key_mask, q_mask, zz, xx, observable, theta = self._components(
            query, key, attention_mask, query_mask, self.control
        )
        pooled = (key * attention_mask[:, None, :, None].to(key.dtype)).sum(dim=2)
        pooled = pooled / attention_mask.sum(dim=1).clamp_min(1).view(-1, 1, 1)
        mapped_query = torch.einsum("bhd,dm->bhm", pooled, self.linear_map)
        query_update = torch.sin(theta).mean(dim=-1, keepdim=True) * mapped_query.unsqueeze(2)
        query_update = query_update.expand_as(query) * q_mask
        self.last_trace = {
            "query": query,
            "key": key,
            "value": value,
            "zz": zz,
            "xx": xx,
            "observable": observable,
            "theta": theta,
            "gate": theta,
            "query_update": query_update,
        }
        return query + query_update, key, value

    def after_scores(
        self,
        scores: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
        layer_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is None:
            raise ValueError("Q-EPVG requires attention_mask")
        key_mask, q_mask, zz, xx, observable, theta = self._components(
            query, key, attention_mask, query_mask, self.control
        )
        base_attention = _masked_softmax(scores, key_mask, self.config.eps)
        score_adjustment = torch.zeros_like(scores)
        attention = base_attention
        query_update = self.last_trace.get("query_update", torch.zeros_like(query))
        routed_values: torch.Tensor = value[:, :, None, :, :].expand(
            -1, -1, query.shape[2], -1, -1
        )
        if self.config.path in {"value_only", "score_value"}:
            mapped_values = torch.einsum("bhkd,dm->bhkm", value, self.linear_map)
            value_angle = theta.unsqueeze(-1)
            routed_values = (
                torch.cos(value_angle) * value[:, :, None, :, :]
                + torch.sin(value_angle) * mapped_values[:, :, None, :, :]
            )
        if self.config.path == "score_value":
            centered = observable - (
                observable * key_mask.to(observable.dtype)
            ).sum(dim=-1, keepdim=True) / key_mask.sum(dim=-1, keepdim=True).clamp_min(1)
            score_adjustment = self.config.score_gain * centered * key_mask.to(scores.dtype)
            scores = scores + score_adjustment
            attention = _masked_softmax(scores, key_mask, self.config.eps)
        elif self.config.path == "query":
            # Query intervention already changed query before score construction.
            routed_values = value[:, :, None, :, :].expand(-1, -1, query.shape[2], -1, -1)
        self.last_trace.update(
            {
                "query": query,
                "key": key,
                "value": value,
                "zz": zz,
                "xx": xx,
                "observable": observable,
                "theta": theta,
                "gate": theta,
                "base_attention": base_attention,
                "score_adjustment": score_adjustment,
                "attention": attention,
                "routed_values": routed_values,
                "query_update": query_update,
            }
        )
        return scores, routed_values

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        scores: torch.Tensor,
        attention_mask: torch.Tensor,
        query_mask: torch.Tensor | None = None,
        control: str = "quantum",
        return_trace: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if control not in EPVG_CONTROLS:
            raise ValueError(f"control must be one of {EPVG_CONTROLS}")
        if query.shape != key.shape or key.shape != value.shape or scores.shape[:3] != query.shape[:3]:
            raise ValueError("query/key/value/scores dimensions are incompatible")
        if attention_mask.shape != (query.shape[0], query.shape[2]):
            raise ValueError("attention_mask must match token dimensions")
        if query_mask is None:
            query_mask = torch.ones(query.shape[0], query.shape[2], dtype=torch.bool, device=query.device)
        key_mask = attention_mask[:, None, None, :]
        q_mask = query_mask[:, None, :, None].to(query.dtype)
        zz, xx, zz_xx = self._observables(query, key, control)
        if self.config.observable == "zz":
            observable = zz
        elif self.config.observable == "zz_xx":
            observable = zz_xx
        else:
            weights = torch.softmax(self.pauli_logits, dim=0)
            observable = weights[0] * zz + weights[1] * xx + weights[2] * zz_xx
        observable = observable.clamp(-1.0, 1.0) * key_mask.to(observable.dtype) * q_mask
        # The toy contract freezes a linear parity-to-angle map.  Keep the
        # angle itself in the trace so downstream checks can distinguish it
        # from a learned/nonlinear gate.
        theta = torch.exp(self.gate_log_scale) * observable
        gate = theta
        base_attention = _masked_softmax(scores, key_mask, self.config.eps)
        score_adjustment = torch.zeros_like(scores)
        attention = base_attention
        query_update = torch.zeros_like(query)
        routed_values = value[:, :, None, :, :].expand(-1, -1, query.shape[2], -1, -1)
        mapped_values = torch.einsum("bhkd,dm->bhkm", value, self.linear_map)
        if self.config.path in {"value_only", "score_value"}:
            value_angle = theta.unsqueeze(-1)
            routed_values = (
                torch.cos(value_angle) * value[:, :, None, :, :]
                + torch.sin(value_angle) * mapped_values[:, :, None, :, :]
            )
        if self.config.path == "score_value":
            centered = observable - (observable * key_mask.to(observable.dtype)).sum(dim=-1, keepdim=True) / key_mask.sum(dim=-1, keepdim=True).clamp_min(1)
            score_adjustment = self.config.score_gain * centered * key_mask.to(scores.dtype)
            attention = _masked_softmax(scores + score_adjustment, key_mask, self.config.eps)
        elif self.config.path == "query":
            pooled = (key * attention_mask[:, None, :, None].to(key.dtype)).sum(dim=2) / attention_mask.sum(dim=1).clamp_min(1).view(-1, 1, 1)
            mapped_query = torch.einsum("bhd,dm->bhm", pooled, self.linear_map)
            query_update = torch.sin(theta).mean(dim=-1, keepdim=True) * mapped_query.unsqueeze(2)
            query_update = query_update.expand_as(query)
            query_update = query_update * q_mask
            query_scores = scores + torch.einsum("bhqd,bhkd->bhqk", query_update, key) / math.sqrt(query.shape[-1])
            attention = _masked_softmax(query_scores, key_mask, self.config.eps)
            routed_values = value[:, :, None, :, :].expand(-1, -1, query.shape[2], -1, -1)
        output = torch.einsum("bhqk,bhqkd->bhqd", attention, routed_values) * q_mask
        if not return_trace:
            return output
        trace = {
            "query": query,
            "key": key,
            "value": value,
            "zz": zz,
            "xx": xx,
            "observable": observable,
            "theta": theta,
            "gate": gate,
            "base_attention": base_attention,
            "score_adjustment": score_adjustment,
            "attention": attention,
            "routed_values": routed_values,
            "query_update": query_update,
            "output": output,
        }
        return output, trace


def build_q_epvg(config: QEPVGConfig) -> EntangledParityValueGatingKernel:
    return EntangledParityValueGatingKernel(config)


__all__ = ["EPVG_CONTROLS", "EPVG_OBSERVABLES", "EPVG_PATHS", "EntangledParityValueGatingKernel", "QEPVGConfig", "build_q_epvg"]
