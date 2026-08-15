"""Causal value-aware quantum evidence transport for attention scores."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from q_attention.plugins.quantum_steering import (
    _data_reuploading_state,
    _seeded_projection,
)


CAUSAL_VALUE_FEATURE_MODES = ("leave_one_out", "key_only")
CAUSAL_VALUE_READOUT_TYPES = (
    "quantum_fidelity_transport",
    "classical_linear_overlap_transport",
)


def _masked_head_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    weights = mask[:, None, :, None].to(device=values.device, dtype=values.dtype)
    denominator = weights.sum(dim=2).clamp_min(eps)
    return (values * weights).sum(dim=2) / denominator


@dataclass(frozen=True)
class CausalValueTransportConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    register_qubits: int = 2
    depth: int = 2
    angle_scale: float = 1.0
    max_transport: float = 0.75
    initial_transport: float = 0.25
    evidence_floor: float = 1e-6
    value_feature_mode: str = "leave_one_out"
    seed: int = 307
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.num_layers <= 0 or self.num_heads <= 0 or self.head_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if self.register_qubits <= 0 or self.depth <= 0:
            raise ValueError("register_qubits and depth must be positive")
        if self.angle_scale <= 0.0:
            raise ValueError("angle_scale must be positive")
        if self.max_transport <= 0.0:
            raise ValueError("max_transport must be positive")
        if not 0.0 < self.initial_transport < self.max_transport:
            raise ValueError(
                "initial_transport must lie inside (0, max_transport)"
            )
        if not 0.0 <= self.evidence_floor < 1.0:
            raise ValueError("evidence_floor must lie inside [0, 1)")
        if self.value_feature_mode not in CAUSAL_VALUE_FEATURE_MODES:
            raise ValueError(
                "value_feature_mode must be one of "
                f"{CAUSAL_VALUE_FEATURE_MODES}"
            )
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")


class CausalValueTransportKernel(nn.Module):
    """Relation-conditioned evidence readout with value-constrained transport.

    The kernel separates evidence formation from attention intervention. Evidence
    is a non-negative overlap between query/relation and token/value states,
    multiplied by the exact leave-one-out output influence magnitude. The
    intervention redistributes only the original context attention mass.
    """

    readout_type = "base"

    def __init__(self, config: CausalValueTransportConfig) -> None:
        super().__init__()
        self.config = config
        qubits = config.register_qubits
        query_dim = 6 * config.head_dim
        token_dim = 8 * config.head_dim
        self.state_dim = 2**qubits
        self.register_qubits = qubits
        self.register_buffer(
            "query_projections",
            torch.stack(
                [
                    _seeded_projection(
                        query_dim,
                        qubits,
                        config.seed + 17 * head,
                    )
                    for head in range(config.num_heads)
                ]
            ),
        )
        self.register_buffer(
            "token_projections",
            torch.stack(
                [
                    _seeded_projection(
                        token_dim,
                        qubits,
                        config.seed + 1009 + 19 * head,
                    )
                    for head in range(config.num_heads)
                ]
            ),
        )
        shape = (config.num_layers, config.num_heads, config.depth, qubits)
        generator = torch.Generator(device="cpu").manual_seed(config.seed + 2003)
        self.query_scales = nn.Parameter(torch.ones(shape))
        self.token_scales = nn.Parameter(torch.ones(shape))
        self.query_biases = nn.Parameter(
            torch.empty(shape).uniform_(-math.pi / 4, math.pi / 4, generator=generator)
        )
        self.token_biases = nn.Parameter(
            torch.empty(shape).uniform_(-math.pi / 4, math.pi / 4, generator=generator)
        )
        initial_ratio = config.initial_transport / config.max_transport
        initial_raw = math.log(initial_ratio) - math.log1p(-initial_ratio)
        self.raw_transport = nn.Parameter(
            torch.full(
                (config.num_layers, config.num_heads),
                initial_raw,
                dtype=torch.float32,
            )
        )

    @property
    def model_dimensions(self) -> tuple[int, int, int]:
        return self.config.num_layers, self.config.num_heads, self.config.head_dim

    def metadata(self) -> dict[str, Any]:
        return {
            "type": self.readout_type,
            "config": asdict(self.config),
            "mechanism": {
                "input": "query,key,value,pre_softmax_scores",
                "causal_witness": "exact_leave_one_out_attention_output_delta",
                "evidence": "nonnegative_quantum_state_overlap_squared",
                "transport": "nonnegative_context_mass_preserving_log_ratio",
                "evidence_support": "convex_base_context_floor",
                "entity_attention": "unchanged",
                "quantum_resource_note": (
                    "overlap can be estimated by a SWAP-test-style circuit; "
                    "simulation does not claim hardware speedup"
                ),
            },
        }

    def transport_fractions(self, layer_index: int) -> torch.Tensor:
        return self.config.max_transport * torch.sigmoid(self.raw_transport[layer_index])

    def _relation_anchor(
        self,
        key: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> torch.Tensor:
        subject = _masked_head_mean(key, subject_mask, self.config.eps)
        object_ = _masked_head_mean(key, object_mask, self.config.eps)
        return torch.cat((subject, object_, subject - object_, subject * object_), dim=-1)

    def _causal_features(
        self,
        scores: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
        masked_scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        base_attention = torch.softmax(masked_scores, dim=-1)
        base_attention = base_attention * key_mask.to(dtype=base_attention.dtype)
        base_output = torch.einsum("bhqk,bhkd->bhqd", base_attention, value)

        # Removing token j and renormalizing the remaining attention gives this
        # exact output delta, up to the numerical floor for a_j near one.
        leave_one_out_delta = (
            base_attention.unsqueeze(-1)
            / (1.0 - base_attention).clamp_min(0.05).unsqueeze(-1)
            * (value[:, :, None, :, :] - base_output[:, :, :, None, :])
        )
        influence = leave_one_out_delta.norm(dim=-1)
        context = attention_mask & ~(subject_mask | object_mask)
        context_mask = context[:, None, None, :].to(dtype=influence.dtype)

        relation = self._relation_anchor(key, subject_mask, object_mask)
        query_relation = relation[:, :, None, :].expand(
            -1, -1, query.shape[2], -1
        )
        query_context = (
            base_output
            if self.config.value_feature_mode == "leave_one_out"
            else torch.zeros_like(base_output)
        )
        query_features = torch.cat((query, query_context, query_relation), dim=-1)

        key_expanded = key[:, :, None, :, :].expand(
            -1, -1, query.shape[2], -1, -1
        )
        relation_expanded = relation[:, :, None, None, :].expand(
            -1, -1, query.shape[2], key.shape[2], -1
        )
        if self.config.value_feature_mode == "leave_one_out":
            value_expanded = value[:, :, None, :, :].expand_as(key_expanded)
            delta = value[:, :, None, :, :] - base_output[:, :, :, None, :]
            token_features = torch.cat(
                (key_expanded, value_expanded, delta, leave_one_out_delta, relation_expanded),
                dim=-1,
            )
            strength = influence
        else:
            zeros = torch.zeros_like(key_expanded)
            token_features = torch.cat(
                (key_expanded, zeros, zeros, zeros, relation_expanded), dim=-1
            )
            strength = torch.ones_like(influence)

        count = context_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean_strength = (strength * context_mask).sum(dim=-1, keepdim=True) / count
        normalized_strength = (
            strength / mean_strength.clamp_min(self.config.eps)
        ).clamp(max=4.0)
        normalized_strength = normalized_strength * context_mask
        return (
            query_features,
            token_features,
            normalized_strength,
            base_attention,
        )

    def _states(
        self,
        query_features: torch.Tensor,
        token_features: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _heads, queries, _ = query_features.shape
        keys = token_features.shape[3]
        query_state = _data_reuploading_state(
            query_features[:, head_index].reshape(batch * queries, -1),
            self.query_projections[head_index],
            self.query_scales[layer_index, head_index],
            self.query_biases[layer_index, head_index],
            angle_scale=self.config.angle_scale,
            eps=self.config.eps,
        ).reshape(batch, queries, self.state_dim)
        token_state = _data_reuploading_state(
            token_features[:, head_index].reshape(batch * queries * keys, -1),
            self.token_projections[head_index],
            self.token_scales[layer_index, head_index],
            self.token_biases[layer_index, head_index],
            angle_scale=self.config.angle_scale,
            eps=self.config.eps,
        ).reshape(batch, queries, keys, self.state_dim)
        return query_state, token_state

    def _readout(
        self,
        query_state: torch.Tensor,
        token_state: torch.Tensor,
    ) -> torch.Tensor:
        overlap = torch.sum(
            query_state[:, :, None, :] * token_state,
            dim=-1,
        )
        if self.readout_type == "quantum_fidelity_transport":
            return overlap.square()
        return 0.5 * (overlap + 1.0)

    def _transport_residual(
        self,
        base_attention: torch.Tensor,
        evidence: torch.Tensor,
        normalized_strength: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        context = attention_mask & ~(subject_mask | object_mask)
        context_mask = context[:, None, None, :].to(dtype=base_attention.dtype)
        evidence = evidence * normalized_strength * context_mask
        evidence_mass = evidence.sum(dim=-1, keepdim=True)
        base_context_mass = (base_attention * context_mask).sum(dim=-1, keepdim=True)
        context_count = context_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        uniform_context = context_mask / context_count
        base_context_distribution = (
            base_attention / base_context_mass.clamp_min(self.config.eps)
        ) * context_mask
        base_context_distribution = torch.where(
            base_context_mass > self.config.eps,
            base_context_distribution,
            uniform_context,
        )
        evidence_distribution = evidence / evidence_mass.clamp_min(self.config.eps)
        evidence_distribution = torch.where(
            evidence_mass > self.config.eps,
            evidence_distribution,
            base_context_distribution,
        )
        evidence_distribution = (
            (1.0 - self.config.evidence_floor) * evidence_distribution
            + self.config.evidence_floor * base_context_distribution
        )
        evidence_distribution = evidence_distribution / evidence_distribution.sum(
            dim=-1, keepdim=True
        ).clamp_min(self.config.eps)
        target_context = base_context_mass * evidence_distribution
        fraction = self.transport_fractions(layer_index).view(1, -1, 1, 1)
        desired = base_attention.clone()
        desired_context = (
            (1.0 - fraction) * base_attention + fraction * target_context
        )
        desired_context_mass = (desired_context * context_mask).sum(
            dim=-1, keepdim=True
        )
        desired_context = desired_context * (
            base_context_mass / desired_context_mass.clamp_min(self.config.eps)
        )
        desired = torch.where(context_mask.bool(), desired_context, desired)
        has_evidence = evidence_mass > self.config.eps
        desired = torch.where(has_evidence, desired, base_attention)
        log_floor = torch.finfo(base_attention.dtype).tiny
        residual = torch.log(desired.clamp_min(log_floor)) - torch.log(
            base_attention.clamp_min(log_floor)
        )
        return residual * attention_mask[:, None, :, None].to(residual.dtype)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor | None = None,
        *,
        scores: torch.Tensor | None = None,
        layer_index: int,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        if value is None or scores is None:
            raise ValueError("causal value transport requires query, key, value, and scores")
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4 or scores.ndim != 4:
            raise ValueError("query, key, value, and scores must be rank four tensors")
        if key.shape != value.shape or scores.shape[:2] != query.shape[:2]:
            raise ValueError("query/key/value/scores dimensions are incompatible")
        if layer_index < 0 or layer_index >= self.config.num_layers:
            raise ValueError("layer_index is outside configured layers")
        query_features, token_features, strength, base_attention = self._causal_features(
            scores,
            query,
            key,
            value,
            attention_mask,
            subject_mask,
            object_mask,
        )
        evidence_by_head = []
        for head_index in range(self.config.num_heads):
            query_state, token_state = self._states(
                query_features,
                token_features,
                layer_index=layer_index,
                head_index=head_index,
            )
            evidence_by_head.append(self._readout(query_state, token_state))
        evidence = torch.stack(evidence_by_head, dim=1)
        return self._transport_residual(
            base_attention,
            evidence,
            strength,
            attention_mask=attention_mask,
            subject_mask=subject_mask,
            object_mask=object_mask,
            layer_index=layer_index,
        )


class QuantumCausalValueTransportKernel(CausalValueTransportKernel):
    """Quantum-primary fidelity evidence with contribution-aware transport."""

    readout_type = "quantum_fidelity_transport"


class ClassicalCausalValueTransportKernel(CausalValueTransportKernel):
    """Matched linear-overlap transport control."""

    readout_type = "classical_linear_overlap_transport"


def build_causal_value_transport_kernel(
    kernel_type: str,
    config: CausalValueTransportConfig,
) -> CausalValueTransportKernel:
    if kernel_type == "quantum":
        return QuantumCausalValueTransportKernel(config)
    if kernel_type == "classical":
        return ClassicalCausalValueTransportKernel(config)
    raise ValueError("kernel_type must be 'quantum' or 'classical'")
