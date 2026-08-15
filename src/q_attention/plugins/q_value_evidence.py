"""Value-aware quantum evidence readout for attention-score intervention."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from q_attention.plugins.quantum_steering import (
    _apply_cnot,
    _data_reuploading_state,
    _raw_gain,
    _seeded_projection,
)


VALUE_FEATURE_MODES = ("leave_one_out", "key_only")
VALUE_READOUT_TYPES = ("quantum_connected", "classical_product")


def _masked_head_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    weights = mask[:, None, :, None].to(device=values.device, dtype=values.dtype)
    denominator = weights.sum(dim=2).clamp_min(eps)
    return (values * weights).sum(dim=2) / denominator


def _z_signs(num_qubits: int, device: torch.device) -> torch.Tensor:
    basis = torch.arange(2**num_qubits, device=device)
    signs = []
    for qubit in range(num_qubits):
        bit = (basis >> (num_qubits - qubit - 1)).bitwise_and(1)
        signs.append(1.0 - 2.0 * bit.to(dtype=torch.float32))
    return torch.stack(signs, dim=0)


@dataclass(frozen=True)
class ValueEvidenceKernelConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    register_qubits: int = 2
    depth: int = 2
    angle_scale: float = 1.0
    max_gain: float = 0.5
    initial_gain: float = 0.05
    value_feature_mode: str = "leave_one_out"
    seed: int = 211
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.num_layers <= 0 or self.num_heads <= 0 or self.head_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if self.register_qubits <= 0 or self.depth <= 0:
            raise ValueError("register_qubits and depth must be positive")
        if self.angle_scale <= 0.0:
            raise ValueError("angle_scale must be positive")
        if self.max_gain <= 0.0:
            raise ValueError("max_gain must be positive")
        if not -self.max_gain < self.initial_gain < self.max_gain:
            raise ValueError("initial_gain must lie inside (-max_gain, max_gain)")
        if self.value_feature_mode not in VALUE_FEATURE_MODES:
            raise ValueError(
                f"value_feature_mode must be one of {VALUE_FEATURE_MODES}"
            )
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")


class ValueEvidenceScoreKernel(nn.Module):
    """Two-register attention intervention driven by query-output evidence.

    The token register receives key, value, and leave-one-out value contribution
    features. The quantum subclass entangles query and token registers and reads
    connected Z correlations; the classical subclass reads the matched product
    correlations without cross-register entanglement.
    """

    readout_type = "base"

    def __init__(self, config: ValueEvidenceKernelConfig) -> None:
        super().__init__()
        self.config = config
        qubits = config.register_qubits
        query_dim = 6 * config.head_dim
        token_dim = 8 * config.head_dim
        self.register_qubits = qubits
        self.total_qubits = 2 * qubits
        self.state_dim = 2**qubits
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
        self.observable_logits = nn.Parameter(
            torch.zeros(config.num_layers, config.num_heads, qubits * qubits)
        )
        self.raw_gains = nn.Parameter(
            _raw_gain(
                config.initial_gain,
                config.max_gain,
                (config.num_layers, config.num_heads),
            )
        )
        self.register_buffer("z_signs", _z_signs(self.total_qubits, torch.device("cpu")))

    @property
    def model_dimensions(self) -> tuple[int, int, int]:
        return self.config.num_layers, self.config.num_heads, self.config.head_dim

    def metadata(self) -> dict[str, Any]:
        return {
            "type": self.readout_type,
            "config": asdict(self.config),
            "mechanism": {
                "input": "query,key,value,base_attention",
                "value_witness": "leave_one_out_value_contribution",
                "quantum_readout": "cross_register_connected_z_correlation",
                "classical_control": "matched_product_z_correlation",
                "attention_intervention": "centered_score_residual",
            },
        }

    def gains(self, layer_index: int) -> torch.Tensor:
        return self.config.max_gain * torch.tanh(self.raw_gains[layer_index])

    def _relation_anchor(
        self,
        key: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> torch.Tensor:
        subject = _masked_head_mean(key, subject_mask, self.config.eps)
        object_ = _masked_head_mean(key, object_mask, self.config.eps)
        return torch.cat((subject, object_, subject - object_, subject * object_), dim=-1)

    def _attention_value_features(
        self,
        scores: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
        masked_scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        base_attention = torch.softmax(masked_scores, dim=-1)
        base_attention = base_attention * key_mask.to(dtype=base_attention.dtype)
        base_output = torch.einsum("bhqk,bhkd->bhqd", base_attention, value)
        value_delta = value[:, :, None, :, :] - base_output[:, :, :, None, :]
        contribution = base_attention.unsqueeze(-1) * value_delta

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
            token_features = torch.cat(
                (
                    key_expanded,
                    value[:, :, None, :, :].expand_as(key_expanded),
                    value_delta,
                    contribution,
                    relation_expanded,
                ),
                dim=-1,
            )
        else:
            zeros = torch.zeros_like(key_expanded)
            token_features = torch.cat(
                (key_expanded, zeros, zeros, zeros, relation_expanded), dim=-1
            )
        return query_features, token_features

    def _register_states(
        self,
        query_features: torch.Tensor,
        token_features: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, heads, queries, _ = query_features.shape
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

    def _pair_features(
        self,
        query_state: torch.Tensor,
        token_state: torch.Tensor,
    ) -> torch.Tensor:
        batch, queries, keys, _ = token_state.shape
        joint = (
            query_state[:, :, None, :, None]
            * token_state[:, :, :, None, :]
        ).reshape(-1, self.state_dim * self.state_dim)
        if self.readout_type == "quantum_connected":
            for qubit in range(self.register_qubits):
                joint = _apply_cnot(
                    joint,
                    qubit,
                    self.register_qubits + qubit,
                    self.total_qubits,
                )
        probabilities = joint.square()
        signs = self.z_signs.to(device=joint.device, dtype=joint.dtype)
        local_query = torch.matmul(probabilities, signs[: self.register_qubits].T)
        local_token = torch.matmul(
            probabilities, signs[self.register_qubits :].T
        )
        pair_features = []
        for query_qubit in range(self.register_qubits):
            for token_qubit in range(self.register_qubits):
                sign = (
                    signs[query_qubit] * signs[self.register_qubits + token_qubit]
                )
                pair = torch.sum(probabilities * sign, dim=-1)
                if self.readout_type == "quantum_connected":
                    pair = pair - local_query[:, query_qubit] * local_token[:, token_qubit]
                pair_features.append(pair)
        return torch.stack(pair_features, dim=-1).reshape(
            batch, queries, keys, self.register_qubits * self.register_qubits
        )

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
        if value is None:
            raise ValueError("value-aware evidence requires the attention value tensor")
        if scores is None:
            raise ValueError("value-aware evidence requires pre-softmax attention scores")
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("query, key, and value must have shape (batch, heads, tokens, dim)")
        if query.shape[:2] != key.shape[:2] or key.shape != value.shape:
            raise ValueError("query/key/value dimensions are incompatible")
        if layer_index < 0 or layer_index >= self.config.num_layers:
            raise ValueError("layer_index is outside configured layers")
        if attention_mask.shape != (scores.shape[0], key.shape[2]):
            raise ValueError("attention_mask must match key tokens")
        query_features, token_features = self._attention_value_features(
            scores,
            query,
            key,
            value,
            attention_mask,
            subject_mask,
            object_mask,
        )
        head_outputs = []
        for head_index in range(self.config.num_heads):
            query_state, token_state = self._register_states(
                query_features,
                token_features,
                layer_index=layer_index,
                head_index=head_index,
            )
            features = self._pair_features(query_state, token_state)
            weights = torch.softmax(
                self.observable_logits[layer_index, head_index], dim=-1
            )
            head_outputs.append(torch.sum(features * weights, dim=-1))
        raw = torch.stack(head_outputs, dim=1)
        key_mask = attention_mask[:, None, None, :].to(dtype=raw.dtype)
        query_mask = attention_mask[:, None, :, None].to(dtype=raw.dtype)
        raw = raw * key_mask
        count = key_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        centered = (raw - (raw * key_mask).sum(dim=-1, keepdim=True) / count) * key_mask
        return centered * self.gains(layer_index).view(1, -1, 1, 1) * query_mask


class QuantumValueEvidenceScoreKernel(ValueEvidenceScoreKernel):
    """Quantum-primary Q-VRES readout using cross-register connected correlations."""

    readout_type = "quantum_connected"


class ClassicalValueEvidenceScoreKernel(ValueEvidenceScoreKernel):
    """Matched classical product-correlation control without cross-register entanglement."""

    readout_type = "classical_product"


def build_value_evidence_score_kernel(
    kernel_type: str,
    config: ValueEvidenceKernelConfig,
) -> ValueEvidenceScoreKernel:
    if kernel_type == "quantum":
        return QuantumValueEvidenceScoreKernel(config)
    if kernel_type == "classical":
        return ClassicalValueEvidenceScoreKernel(config)
    raise ValueError("kernel_type must be 'quantum' or 'classical'")
