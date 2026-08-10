"""Relation-conditioned quantum and matched classical attention-score kernels."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_evidence import (
    EVIDENCE_VIEW_CHOICES,
    RelationEvidenceSelector,
    RelationEvidenceSelectorConfig,
    build_relation_evidence_selector,
)
from .attention_routing import (
    RelationExpertRouterConfig,
    RelationObservableExpertRouter,
    build_relation_expert_router,
)
from .quantum_steering import _data_reuploading_state, _seeded_projection


SCORE_KERNEL_CHECKPOINT_VERSION = 1
CONTINUOUS_MEASUREMENT_READOUTS = (
    "continuous_connected",
    "continuous_measurement",
    "continuous_connected_bank",
    "continuous_measurement_bank",
)
CONTINUOUS_MEASUREMENT_BANK_READOUTS = (
    "continuous_connected_bank",
    "continuous_measurement_bank",
)
CONTINUOUS_CONNECTED_READOUTS = (
    "continuous_connected",
    "continuous_connected_bank",
)
SCORE_READOUT_CHOICES = (
    "fidelity",
    "interference",
    "observable",
    *CONTINUOUS_MEASUREMENT_READOUTS,
)
SCORE_INPUT_ENCODING_CHOICES = ("joint", "factorized_shared")
SCORE_QUERY_SCOPE_CHOICES = ("all", "entities")


@dataclass(frozen=True)
class RelationScoreKernelConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    num_qubits: int = 4
    depth: int = 2
    angle_scale: float = 1.0
    max_gain: float = 0.5
    initial_gain: float = 0.02
    normalize_readout_energy: bool = False
    score_readout: str = "fidelity"
    input_encoding: str = "joint"
    query_scope: str = "all"
    seed: int = 53
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.num_layers <= 0 or self.num_heads <= 0 or self.head_dim <= 0:
            raise ValueError("num_layers, num_heads, and head_dim must be positive")
        if self.num_qubits <= 0 or self.depth <= 0:
            raise ValueError("num_qubits and depth must be positive")
        if self.angle_scale <= 0:
            raise ValueError("angle_scale must be positive")
        if self.max_gain <= 0:
            raise ValueError("max_gain must be positive")
        if not -self.max_gain < self.initial_gain < self.max_gain:
            raise ValueError("initial_gain must lie inside (-max_gain, max_gain)")
        if self.score_readout not in SCORE_READOUT_CHOICES:
            raise ValueError(f"score_readout must be one of {SCORE_READOUT_CHOICES}")
        if self.input_encoding not in SCORE_INPUT_ENCODING_CHOICES:
            raise ValueError(
                f"input_encoding must be one of {SCORE_INPUT_ENCODING_CHOICES}"
            )
        if self.query_scope not in SCORE_QUERY_SCOPE_CHOICES:
            raise ValueError(f"query_scope must be one of {SCORE_QUERY_SCOPE_CHOICES}")
        if self.eps <= 0:
            raise ValueError("eps must be positive")


def _raw_gain(initial_gain: float, max_gain: float, shape: tuple[int, ...]) -> torch.Tensor:
    ratio = torch.tensor(initial_gain / max_gain, dtype=torch.float32)
    return torch.full(shape, float(torch.atanh(ratio).item()), dtype=torch.float32)


def _masked_head_mean(values: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
    weights = mask[:, None, :, None].to(device=values.device, dtype=values.dtype)
    denominator = weights.sum(dim=2).clamp_min(eps)
    return (values * weights).sum(dim=2) / denominator


class RelationAttentionScoreKernel(nn.Module):
    """Common parameterization for quantum and parameter-matched controls."""

    kernel_type = "base"

    def __init__(self, config: RelationScoreKernelConfig) -> None:
        super().__init__()
        self.config = config
        self.evidence_selector: RelationEvidenceSelector | None = None
        self.expert_router: RelationObservableExpertRouter | None = None
        if config.input_encoding == "joint":
            feature_dim = 5 * config.head_dim
            query_projections = torch.stack(
                [
                    _seeded_projection(
                        feature_dim,
                        config.num_qubits,
                        config.seed + 17 * head,
                    )
                    for head in range(config.num_heads)
                ]
            )
            key_projections = torch.stack(
                [
                    _seeded_projection(
                        feature_dim,
                        config.num_qubits,
                        config.seed + 1009 + 19 * head,
                    )
                    for head in range(config.num_heads)
                ]
            )
            relation_projections = torch.empty(
                config.num_heads, 0, config.num_qubits
            )
        else:
            query_projections = torch.stack(
                [
                    _seeded_projection(
                        config.head_dim,
                        config.num_qubits,
                        config.seed + 17 * head,
                    )
                    for head in range(config.num_heads)
                ]
            )
            key_projections = query_projections.clone()
            relation_projections = torch.stack(
                [
                    _seeded_projection(
                        4 * config.head_dim,
                        config.num_qubits,
                        config.seed + 2003 + 23 * head,
                    )
                    for head in range(config.num_heads)
                ]
            )
        self.register_buffer("query_projections", query_projections)
        self.register_buffer("key_projections", key_projections)
        self.register_buffer(
            "relation_projections", relation_projections, persistent=False
        )

        shape = (config.num_layers, config.num_heads, config.depth, config.num_qubits)
        self.query_scales = nn.Parameter(torch.ones(shape))
        self.key_scales = nn.Parameter(torch.ones(shape))
        generator = torch.Generator(device="cpu").manual_seed(config.seed + 2003)
        self.query_biases = nn.Parameter(
            torch.empty(shape).uniform_(-math.pi / 4, math.pi / 4, generator=generator)
        )
        self.key_biases = nn.Parameter(
            torch.empty(shape).uniform_(-math.pi / 4, math.pi / 4, generator=generator)
        )
        self.raw_gains = nn.Parameter(
            _raw_gain(
                config.initial_gain,
                config.max_gain,
                (config.num_layers, config.num_heads),
            )
        )
        if config.score_readout == "observable":
            observable_count = 2 * config.num_qubits
            observable_logits = torch.zeros(
                config.num_layers, config.num_heads, observable_count
            )
            for head_index in range(config.num_heads):
                observable_logits[:, head_index, (3 * head_index) % observable_count] = 1.0
            self.observable_logits = nn.Parameter(observable_logits)
        else:
            self.register_parameter("observable_logits", None)
        if config.score_readout in CONTINUOUS_MEASUREMENT_READOUTS:
            measurement_generator = torch.Generator(device="cpu").manual_seed(
                config.seed + 3001
            )
            basis_count = (
                4
                if config.score_readout in CONTINUOUS_MEASUREMENT_BANK_READOUTS
                else 1
            )
            angle_shape = (
                (config.num_layers, config.num_heads, config.num_qubits)
                if basis_count == 1
                else (
                    config.num_layers,
                    config.num_heads,
                    basis_count,
                    config.num_qubits,
                )
            )
            self.measurement_angles = nn.Parameter(
                torch.empty(*angle_shape).uniform_(
                    -math.pi / 2,
                    math.pi / 2,
                    generator=measurement_generator,
                )
            )
            if basis_count > 1:
                self.readout_logits = nn.Parameter(
                    torch.zeros(config.num_layers, config.num_heads, basis_count)
                )
            else:
                self.register_parameter("readout_logits", None)
        else:
            self.register_parameter("measurement_angles", None)
            self.register_parameter("readout_logits", None)
        self._capture_centered = False
        self._captured_centered: list[torch.Tensor] = []

    @property
    def model_dimensions(self) -> tuple[int, int, int]:
        return self.config.num_layers, self.config.num_heads, self.config.head_dim

    def metadata(self) -> dict[str, Any]:
        return {
            "type": self.kernel_type,
            "config": asdict(self.config),
            "evidence_selector": (
                self.evidence_selector.metadata()
                if self.evidence_selector is not None
                else None
            ),
            "expert_router": (
                self.expert_router.metadata()
                if self.expert_router is not None
                else None
            ),
        }

    def attach_evidence_selector(
        self,
        selector: RelationEvidenceSelector | None,
    ) -> None:
        if selector is not None and selector.model_dimensions != self.model_dimensions:
            raise ValueError("evidence selector dimensions must match the score kernel")
        self.evidence_selector = selector

    def attach_expert_router(
        self,
        router: RelationObservableExpertRouter | None,
    ) -> None:
        if router is not None and self.config.score_readout != "observable":
            raise ValueError("expert routing requires score_readout='observable'")
        if router is not None and router.model_dimensions != self.model_dimensions:
            raise ValueError("expert router dimensions must match the score kernel")
        if (
            router is not None
            and router.config.num_observables != 2 * self.config.num_qubits
        ):
            raise ValueError("expert router observable count must match the score kernel")
        self.expert_router = router

    def gains(self, layer_index: int) -> torch.Tensor:
        self._validate_layer(layer_index)
        return self.config.max_gain * torch.tanh(self.raw_gains[layer_index])

    def observable_weights(self, layer_index: int) -> torch.Tensor | None:
        self._validate_layer(layer_index)
        if self.observable_logits is None:
            return None
        return torch.softmax(self.observable_logits[layer_index], dim=-1)

    def _validate_layer(self, layer_index: int) -> None:
        if not 0 <= layer_index < self.config.num_layers:
            raise ValueError(
                f"layer_index {layer_index} is outside [0, {self.config.num_layers})"
            )

    def _validate_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> None:
        if query.ndim != 4:
            raise ValueError(
                "query must have shape (batch, num_heads, tokens, head_dim)"
            )
        expected_tail = (self.config.num_heads, query.shape[2], self.config.head_dim)
        if query.shape[1:] != expected_tail:
            raise ValueError(
                "query must have shape (batch, num_heads, tokens, head_dim)"
            )
        if key.shape != query.shape:
            raise ValueError("query and key must have the same shape")
        for name, mask in {
            "attention_mask": attention_mask,
            "subject_mask": subject_mask,
            "object_mask": object_mask,
        }.items():
            if mask.shape != (query.shape[0], query.shape[2]):
                raise ValueError(f"{name} must match query batch and token dimensions")

    def _relation_anchor(
        self,
        key: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> torch.Tensor:
        subject = _masked_head_mean(key, subject_mask, self.config.eps)
        object_ = _masked_head_mean(key, object_mask, self.config.eps)
        return torch.cat(
            (subject, object_, subject - object_, subject * object_),
            dim=-1,
        )

    def _relation_features(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        relation_anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relation = relation_anchor
        relation = relation[:, :, None, :].expand(-1, -1, query.shape[2], -1)
        return torch.cat((query, relation), dim=-1), torch.cat((key, relation), dim=-1)

    def _feature_states(
        self,
        features: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
        branch: str,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _observable_features(self, states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _measurement_observable_features(
        self,
        states: torch.Tensor,
        angles: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _connected_correlation_features(
        self,
        measurements: torch.Tensor,
    ) -> torch.Tensor:
        """Return adjacent connected correlations for each measurement expert."""
        qubits = self.config.num_qubits
        if measurements.shape[-1] != 2 * qubits:
            raise ValueError("measurements must contain local and pair observables")
        local = measurements[..., :qubits]
        adjacent = measurements[..., qubits:]
        return adjacent - local * torch.roll(local, shifts=-1, dims=-1)

    def _normalized_connected_features(
        self,
        measurements: torch.Tensor,
    ) -> torch.Tensor:
        connected = self._connected_correlation_features(measurements)
        norm = torch.linalg.vector_norm(connected, dim=-1, keepdim=True)
        normalized = connected / norm.clamp_min(self.config.eps)
        return math.sqrt(self.config.num_qubits) * torch.where(
            norm > self.config.eps,
            normalized,
            torch.zeros_like(normalized),
        )

    def _normalized_measurement_features(
        self,
        measurements: torch.Tensor,
    ) -> torch.Tensor:
        norm = torch.linalg.vector_norm(measurements, dim=-1, keepdim=True)
        normalized = measurements / norm.clamp_min(self.config.eps)
        return math.sqrt(measurements.shape[-1]) * torch.where(
            norm > self.config.eps,
            normalized,
            torch.zeros_like(normalized),
        )

    def _continuous_measurement_kernel(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
    ) -> torch.Tensor:
        if self.measurement_angles is None:
            raise RuntimeError("continuous measurement angles are unavailable")
        angles = self.measurement_angles[layer_index, head_index]
        if angles.ndim == 1:
            angles = angles.unsqueeze(0)
        query_features = self._measurement_observable_features(query_states, angles)
        key_features = self._measurement_observable_features(key_states, angles)
        if self.config.score_readout in CONTINUOUS_CONNECTED_READOUTS:
            query_features = self._normalized_connected_features(query_features)
            key_features = self._normalized_connected_features(key_features)
        else:
            query_features = self._normalized_measurement_features(query_features)
            key_features = self._normalized_measurement_features(key_features)
        basis_scores = torch.einsum(
            "bqef,bkef->beqk", query_features, key_features
        ) / query_features.shape[-1]
        if self.readout_logits is None:
            return basis_scores[:, 0]
        weights = torch.softmax(self.readout_logits[layer_index, head_index], dim=-1)
        return torch.einsum("e,beqk->bqk", weights, basis_scores)

    def _pairwise_kernel(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
        relation_anchor: torch.Tensor,
        routing_mode: str,
        query_context: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        amplitudes = torch.matmul(query_states, key_states.transpose(-1, -2))
        if self.config.score_readout == "fidelity":
            return amplitudes.square()
        if self.config.score_readout == "interference":
            return amplitudes
        if self.config.score_readout in CONTINUOUS_MEASUREMENT_READOUTS:
            return self._continuous_measurement_kernel(
                query_states,
                key_states,
                layer_index=layer_index,
                head_index=head_index,
            )

        query_observables = self._observable_features(query_states)
        key_observables = self._observable_features(key_states)
        components = (
            query_observables[:, :, None, :] * key_observables[:, None, :, :]
        )
        weights = self.observable_weights(layer_index)
        if weights is None:
            raise RuntimeError("observable readout weights are unavailable")
        if self.expert_router is not None:
            if self.expert_router.config.direction_mode in {
                "measurement_aligned",
                "connected_aligned",
            }:
                angles = self.expert_router.measurement_angles(
                    layer_index, head_index
                )
                query_measurements = self._measurement_observable_features(
                    query_states, angles
                )
                key_measurements = self._measurement_observable_features(
                    key_states, angles
                )
                if self.expert_router.config.direction_mode == "connected_aligned":
                    query_measurements = self._normalized_connected_features(
                        query_measurements
                    )
                    key_measurements = self._normalized_connected_features(
                        key_measurements
                    )
                expert_deltas = torch.einsum(
                    "bieo,bjeo->beij",
                    query_measurements,
                    key_measurements,
                ) / query_measurements.shape[-1]
                expert_deltas = expert_deltas - expert_deltas.mean(
                    dim=1, keepdim=True
                )
                base = torch.sum(components * weights[head_index], dim=-1)
                return self.expert_router.route_expert_deltas(
                    expert_deltas,
                    base,
                    relation_anchor,
                    layer_index=layer_index,
                    head_index=head_index,
                    routing_mode=routing_mode,
                    query_context=query_context,
                    query_mask=query_mask,
                )
            return self.expert_router.route_components(
                components,
                weights[head_index],
                relation_anchor,
                layer_index=layer_index,
                head_index=head_index,
                routing_mode=routing_mode,
                query_context=query_context,
                query_mask=query_mask,
            )
        return torch.sum(components * weights[head_index], dim=-1)

    @contextmanager
    def capture_centered_kernels(self) -> Iterator[None]:
        if self._capture_centered:
            raise RuntimeError("centered-kernel capture is already active")
        self._captured_centered.clear()
        self._capture_centered = True
        try:
            yield
        finally:
            self._capture_centered = False
            self._captured_centered.clear()

    def functional_diversity_loss(self) -> torch.Tensor:
        """Penalize squared cosine overlap of captured score kernels across heads."""
        if not self._capture_centered:
            raise RuntimeError("functional diversity requires an active capture context")
        losses: list[torch.Tensor] = []
        for centered in self._captured_centered:
            if centered.shape[1] < 2:
                continue
            head_vectors = centered.transpose(0, 1).reshape(centered.shape[1], -1)
            normalized = F.normalize(
                head_vectors,
                p=2,
                dim=-1,
                eps=self.config.eps,
            )
            similarities = torch.matmul(normalized, normalized.transpose(0, 1))
            off_diagonal = ~torch.eye(
                centered.shape[1],
                dtype=torch.bool,
                device=centered.device,
            )
            losses.append(similarities.masked_select(off_diagonal).square().mean())
        if losses:
            return torch.stack(losses).mean()
        return self.raw_gains.sum() * 0.0

    def _encoding_inputs(
        self,
        features: torch.Tensor,
        *,
        head_index: int,
        branch: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if branch == "query":
            local_projection = self.query_projections[head_index]
        elif branch == "key":
            local_projection = self.key_projections[head_index]
        else:
            raise ValueError("branch must be 'query' or 'key'")
        if self.config.input_encoding == "joint":
            return features, local_projection

        local = F.normalize(
            features[:, : self.config.head_dim].float(),
            p=2,
            dim=-1,
            eps=self.config.eps,
        )
        relation = F.normalize(
            features[:, self.config.head_dim :].float(),
            p=2,
            dim=-1,
            eps=self.config.eps,
        )
        encoded = torch.cat((local, relation), dim=-1)
        projection = torch.cat(
            (local_projection, self.relation_projections[head_index]), dim=0
        )
        return encoded, projection

    def unmodulated_centered_kernel(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        layer_index: int,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
        routing_mode: str = "learned",
    ) -> torch.Tensor:
        """Return the centered score kernel before token-evidence modulation."""
        self._validate_layer(layer_index)
        self._validate_inputs(query, key, attention_mask, subject_mask, object_mask)
        relation_anchor = self._relation_anchor(
            key,
            subject_mask,
            object_mask,
        )
        query_features, key_features = self._relation_features(
            query,
            key,
            relation_anchor,
        )
        batch, _heads, tokens, _features = query_features.shape
        active_queries = attention_mask
        if self.config.query_scope == "entities":
            active_queries = active_queries & (subject_mask | object_mask)
        kernels: list[torch.Tensor] = []
        for head_index in range(self.config.num_heads):
            query_states = self._feature_states(
                query_features[:, head_index].reshape(batch * tokens, -1),
                layer_index=layer_index,
                head_index=head_index,
                branch="query",
            ).reshape(batch, tokens, -1)
            key_states = self._feature_states(
                key_features[:, head_index].reshape(batch * tokens, -1),
                layer_index=layer_index,
                head_index=head_index,
                branch="key",
            ).reshape(batch, tokens, -1)
            kernels.append(
                self._pairwise_kernel(
                    query_states,
                    key_states,
                    layer_index=layer_index,
                    head_index=head_index,
                    relation_anchor=relation_anchor[:, head_index],
                    routing_mode=routing_mode,
                    query_context=query_features[:, head_index],
                    query_mask=active_queries,
                )
            )
        kernel = torch.stack(kernels, dim=1)

        key_mask = attention_mask[:, None, None, :].to(kernel.dtype)
        query_mask = active_queries[:, None, :, None].to(kernel.dtype)
        key_count = key_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        centered = kernel - (kernel * key_mask).sum(dim=-1, keepdim=True) / key_count
        valid_scores = query_mask * key_mask
        centered = centered * valid_scores
        if self.config.normalize_readout_energy:
            valid_count = valid_scores.sum(dim=(-1, -2), keepdim=True).clamp_min(1.0)
            readout_power = (
                centered.square().sum(dim=(-1, -2), keepdim=True) / valid_count
            )
            readout_rms = torch.sqrt(readout_power.clamp_min(self.config.eps**2))
            baseline_scores = torch.matmul(
                query.float(), key.float().transpose(-1, -2)
            ) / math.sqrt(self.config.head_dim)
            baseline_centered = baseline_scores - (
                (baseline_scores * key_mask).sum(dim=-1, keepdim=True) / key_count
            )
            baseline_centered = baseline_centered * valid_scores
            baseline_rms = torch.sqrt(
                baseline_centered.square().sum(dim=(-1, -2), keepdim=True)
                / valid_count
            )
            readout_scale = torch.where(
                readout_power > self.config.eps**2,
                baseline_rms / readout_rms,
                torch.zeros_like(readout_rms),
            )
            centered = centered * readout_scale
        if (
            self.expert_router is not None
            and self.expert_router.config.residual_reference == "baseline"
        ):
            if self.expert_router.config.normalize_routed_energy:
                valid_count = valid_scores.sum(dim=(-1, -2), keepdim=True).clamp_min(1.0)
                routed_power = (
                    centered.square().sum(dim=(-1, -2), keepdim=True) / valid_count
                )
                routed_rms = torch.sqrt(
                    routed_power.clamp_min(self.config.eps**2)
                )
                baseline_scores = torch.matmul(
                    query.float(), key.float().transpose(-1, -2)
                ) / math.sqrt(self.config.head_dim)
                baseline_centered = baseline_scores - (
                    (baseline_scores * key_mask).sum(dim=-1, keepdim=True)
                    / key_count
                )
                baseline_centered = baseline_centered * valid_scores
                baseline_rms = torch.sqrt(
                    baseline_centered.square().sum(
                        dim=(-1, -2), keepdim=True
                    ) / valid_count
                )
                routed_scale = torch.where(
                    routed_power > self.config.eps**2,
                    baseline_rms / routed_rms,
                    torch.zeros_like(routed_rms),
                )
                centered = centered * routed_scale
            centered = centered * self.expert_router.gains(layer_index).view(
                1, -1, 1, 1
            )
        return centered

    def score_residual(self, centered: torch.Tensor, layer_index: int) -> torch.Tensor:
        """Apply the gain owned by the active score intervention."""
        if (
            self.expert_router is not None
            and self.expert_router.config.residual_reference == "baseline"
        ):
            return centered
        return centered * self.gains(layer_index).view(1, -1, 1, 1)

    def evidence_scores(
        self,
        key: torch.Tensor,
        *,
        layer_index: int,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.evidence_selector is None:
            return None
        return self.evidence_selector.token_readouts(
            key,
            layer_index=layer_index,
            attention_mask=attention_mask,
            subject_mask=subject_mask,
            object_mask=object_mask,
        )[1]

    def evidence_readouts(
        self,
        key: torch.Tensor,
        *,
        layer_index: int,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.evidence_selector is None:
            return None
        return self.evidence_selector.token_readouts(
            key,
            layer_index=layer_index,
            attention_mask=attention_mask,
            subject_mask=subject_mask,
            object_mask=object_mask,
        )

    def centered_kernel(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        layer_index: int,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
        routing_mode: str = "learned",
        steering_evidence: torch.Tensor | None = None,
        sufficiency_evidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return centered scores after optional bounded evidence modulation."""
        centered = self.unmodulated_centered_kernel(
            query,
            key,
            layer_index=layer_index,
            attention_mask=attention_mask,
            subject_mask=subject_mask,
            object_mask=object_mask,
            routing_mode=routing_mode,
        )
        if (
            self.evidence_selector is None
            or self.evidence_selector.config.intervention_mode == "direct_bias"
        ):
            return centered
        evidence = steering_evidence
        if evidence is None:
            readouts = self.evidence_readouts(
                key,
                layer_index=layer_index,
                attention_mask=attention_mask,
                subject_mask=subject_mask,
                object_mask=object_mask,
            )
            evidence = None if readouts is None else readouts[0]
            sufficiency_evidence = None if readouts is None else readouts[1]
        if evidence is None:
            return centered
        if sufficiency_evidence is None:
            sufficiency_evidence = evidence
        active_queries = attention_mask
        if self.config.query_scope == "entities":
            active_queries = active_queries & (subject_mask | object_mask)
        return self.evidence_selector.steering_residual(
            centered,
            evidence,
            sufficiency_evidence,
            attention_mask=attention_mask,
            subject_mask=subject_mask,
            object_mask=object_mask,
            query_mask=active_queries,
        )

    def direct_evidence_attention_bias(
        self,
        key: torch.Tensor,
        *,
        layer_index: int,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
        evidence: torch.Tensor | None = None,
        sufficiency_evidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        selector = self.evidence_selector
        if selector is None or selector.config.intervention_mode != "direct_bias":
            return key.new_zeros(
                key.shape[0],
                key.shape[1],
                key.shape[2],
                key.shape[2],
            )
        if evidence is None:
            readouts = self.evidence_readouts(
                key,
                layer_index=layer_index,
                attention_mask=attention_mask,
                subject_mask=subject_mask,
                object_mask=object_mask,
            )
            evidence = None if readouts is None else readouts[0]
            sufficiency_evidence = None if readouts is None else readouts[1]
        if evidence is None:
            raise RuntimeError("direct evidence bias requires evidence scores")
        key_bias = selector.direct_key_bias(
            evidence,
            layer_index=layer_index,
            attention_mask=attention_mask,
            subject_mask=subject_mask,
            object_mask=object_mask,
            sufficiency_scores=sufficiency_evidence,
        )
        active_queries = attention_mask
        if self.config.query_scope == "entities":
            active_queries = active_queries & (subject_mask | object_mask)
        query_mask = active_queries[:, None, :, None].to(key_bias.dtype)
        return key_bias[:, :, None, :] * query_mask

    def counterfactual_attention_bias(
        self,
        key: torch.Tensor,
        *,
        layer_index: int,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
        evidence_view: str,
        random_seed: int,
        detach_random: bool,
        evidence: torch.Tensor | None = None,
        steering_evidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if evidence_view not in EVIDENCE_VIEW_CHOICES:
            raise ValueError(f"evidence_view must be one of {EVIDENCE_VIEW_CHOICES}")
        if evidence_view == "full":
            return key.new_zeros(
                key.shape[0],
                key.shape[1],
                key.shape[2],
                key.shape[2],
            )
        if evidence is None:
            readouts = self.evidence_readouts(
                key,
                layer_index=layer_index,
                attention_mask=attention_mask,
                subject_mask=subject_mask,
                object_mask=object_mask,
            )
            steering_evidence = None if readouts is None else readouts[0]
            evidence = None if readouts is None else readouts[1]
        if evidence is None or self.evidence_selector is None:
            raise RuntimeError("counterfactual views require an evidence selector")
        weights = self.evidence_selector.view_weights(
            evidence,
            view=evidence_view,
            attention_mask=attention_mask,
            subject_mask=subject_mask,
            object_mask=object_mask,
            random_seed=random_seed,
            detach_random=detach_random,
            steering_scores=steering_evidence,
        )
        bias = weights.clamp_min(self.config.eps).log()[:, :, None, :]
        query_mask = attention_mask[:, None, :, None].to(bias.dtype)
        return bias * query_mask

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        layer_index: int,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
        evidence_view: str = "full",
        random_seed: int = 0,
        detach_random: bool = False,
        routing_mode: str = "learned",
    ) -> torch.Tensor:
        evidence_readouts = None
        if self.evidence_selector is not None:
            evidence_readouts = self.evidence_readouts(
                key,
                layer_index=layer_index,
                attention_mask=attention_mask,
                subject_mask=subject_mask,
                object_mask=object_mask,
            )
        centered = self.centered_kernel(
            query,
            key,
            layer_index=layer_index,
            attention_mask=attention_mask,
            subject_mask=subject_mask,
            object_mask=object_mask,
            routing_mode=routing_mode,
            steering_evidence=(
                None if evidence_readouts is None else evidence_readouts[0]
            ),
            sufficiency_evidence=(
                None if evidence_readouts is None else evidence_readouts[1]
            ),
        )
        if self._capture_centered:
            self._captured_centered.append(centered)
        residual = self.score_residual(centered, layer_index)
        if (
            self.evidence_selector is not None
            and self.evidence_selector.config.intervention_mode == "direct_bias"
        ):
            residual = residual + self.direct_evidence_attention_bias(
                key,
                layer_index=layer_index,
                attention_mask=attention_mask,
                subject_mask=subject_mask,
                object_mask=object_mask,
                evidence=(None if evidence_readouts is None else evidence_readouts[0]),
                sufficiency_evidence=(
                    None if evidence_readouts is None else evidence_readouts[1]
                ),
            )
        if evidence_view != "full":
            residual = residual + self.counterfactual_attention_bias(
                key,
                layer_index=layer_index,
                attention_mask=attention_mask,
                subject_mask=subject_mask,
                object_mask=object_mask,
                evidence_view=evidence_view,
                random_seed=random_seed,
                detach_random=detach_random,
                evidence=(None if evidence_readouts is None else evidence_readouts[1]),
                steering_evidence=(
                    None if evidence_readouts is None else evidence_readouts[0]
                ),
            )
        return residual


class QuantumRelationAttentionScoreKernel(RelationAttentionScoreKernel):
    """Fidelity kernel generated by balanced entangled quantum circuits."""

    kernel_type = "quantum"

    def __init__(self, config: RelationScoreKernelConfig) -> None:
        super().__init__(config)
        basis = torch.arange(2**config.num_qubits, dtype=torch.long)
        masks = [1 << qubit for qubit in range(config.num_qubits)]
        masks.extend(
            (1 << qubit) | (1 << ((qubit + 1) % config.num_qubits))
            for qubit in range(config.num_qubits)
        )
        signs: list[torch.Tensor] = []
        for mask in masks:
            parity = torch.zeros_like(basis)
            for bit in range(config.num_qubits):
                if mask & (1 << bit):
                    parity = parity.bitwise_xor((basis >> bit).bitwise_and(1))
            signs.append(1.0 - 2.0 * parity.float())
        self.register_buffer("observable_signs", torch.stack(signs), persistent=False)

    def _observable_features(self, states: torch.Tensor) -> torch.Tensor:
        return torch.matmul(states.square(), self.observable_signs.transpose(0, 1))

    def _measurement_observable_features(
        self,
        states: torch.Tensor,
        angles: torch.Tensor,
    ) -> torch.Tensor:
        num_qubits = self.config.num_qubits
        if angles.ndim != 2 or angles.shape[1] != num_qubits:
            raise ValueError("angles must have shape (experts, num_qubits)")
        basis = torch.arange(states.shape[-1], device=states.device)

        def expectation(x_mask: int, z_mask: int) -> torch.Tensor:
            permutation = basis.bitwise_xor(x_mask)
            parity = torch.zeros_like(basis)
            for bit in range(num_qubits):
                if z_mask & (1 << bit):
                    parity = parity.bitwise_xor(
                        basis.bitwise_right_shift(bit).bitwise_and(1)
                    )
            signs = (1.0 - 2.0 * parity.to(states.dtype)).to(states.device)
            return (states * states[..., permutation] * signs).sum(dim=-1)

        local_z = []
        local_x = []
        pair_zz = []
        pair_zx = []
        pair_xz = []
        pair_xx = []
        for qubit in range(num_qubits):
            next_qubit = (qubit + 1) % num_qubits
            qubit_mask = 1 << qubit
            next_mask = 1 << next_qubit
            local_z.append(expectation(0, qubit_mask))
            local_x.append(expectation(qubit_mask, 0))
            pair_zz.append(expectation(0, qubit_mask | next_mask))
            pair_zx.append(expectation(next_mask, qubit_mask))
            pair_xz.append(expectation(qubit_mask, next_mask))
            pair_xx.append(expectation(qubit_mask | next_mask, 0))
        z = torch.stack(local_z, dim=-1)
        x = torch.stack(local_x, dim=-1)
        zz = torch.stack(pair_zz, dim=-1)
        zx = torch.stack(pair_zx, dim=-1)
        xz = torch.stack(pair_xz, dim=-1)
        xx = torch.stack(pair_xx, dim=-1)
        cosine = torch.cos(angles)
        sine = torch.sin(angles)
        next_cosine = torch.roll(cosine, shifts=-1, dims=-1)
        next_sine = torch.roll(sine, shifts=-1, dims=-1)
        local = z.unsqueeze(-2) * cosine + x.unsqueeze(-2) * sine
        pairs = (
            zz.unsqueeze(-2) * cosine * next_cosine
            + zx.unsqueeze(-2) * cosine * next_sine
            + xz.unsqueeze(-2) * sine * next_cosine
            + xx.unsqueeze(-2) * sine * next_sine
        )
        return torch.cat((local, pairs), dim=-1)

    def _feature_states(
        self,
        features: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
        branch: str,
    ) -> torch.Tensor:
        features, projection = self._encoding_inputs(
            features,
            head_index=head_index,
            branch=branch,
        )
        if branch == "query":
            scales = self.query_scales[layer_index, head_index]
            biases = self.query_biases[layer_index, head_index]
        elif branch == "key":
            scales = self.key_scales[layer_index, head_index]
            biases = self.key_biases[layer_index, head_index]
        else:
            raise ValueError("branch must be 'query' or 'key'")
        return _data_reuploading_state(
            features,
            projection,
            scales,
            biases,
            angle_scale=self.config.angle_scale,
            eps=self.config.eps,
        )


class ClassicalRelationAttentionScoreKernel(RelationAttentionScoreKernel):
    """Parameter-matched separable trigonometric score-kernel control."""

    kernel_type = "classical"

    def _observable_features(self, states: torch.Tensor) -> torch.Tensor:
        qubits = self.config.num_qubits
        local_z = qubits * (
            states[..., :qubits].square() - states[..., qubits:].square()
        )
        adjacent_zz = local_z * torch.roll(local_z, shifts=-1, dims=-1)
        return torch.cat((local_z, adjacent_zz), dim=-1)

    def _measurement_observable_features(
        self,
        states: torch.Tensor,
        angles: torch.Tensor,
    ) -> torch.Tensor:
        qubits = self.config.num_qubits
        if angles.ndim != 2 or angles.shape[1] != qubits:
            raise ValueError("angles must have shape (experts, num_qubits)")
        low = states[..., :qubits]
        high = states[..., qubits:]
        local_z = qubits * (low.square() - high.square())
        local_x = 2 * qubits * low * high
        cosine = torch.cos(angles)
        sine = torch.sin(angles)
        measured = local_z.unsqueeze(-2) * cosine + local_x.unsqueeze(-2) * sine
        adjacent = measured * torch.roll(measured, shifts=-1, dims=-1)
        return torch.cat((measured, adjacent), dim=-1)

    def _feature_states(
        self,
        features: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
        branch: str,
    ) -> torch.Tensor:
        features, projection = self._encoding_inputs(
            features,
            head_index=head_index,
            branch=branch,
        )
        if branch == "query":
            scales = self.query_scales[layer_index, head_index]
            biases = self.query_biases[layer_index, head_index]
        elif branch == "key":
            scales = self.key_scales[layer_index, head_index]
            biases = self.key_biases[layer_index, head_index]
        else:
            raise ValueError("branch must be 'query' or 'key'")
        normalized = F.normalize(features.float(), p=2, dim=-1, eps=self.config.eps)
        angles = self.config.angle_scale * torch.matmul(normalized, projection)
        phase = angles
        for depth_index in range(self.config.depth):
            phase = torch.sin(
                phase + angles * scales[depth_index] + biases[depth_index]
            )
        state = torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)
        return F.normalize(state, p=2, dim=-1, eps=self.config.eps)


def score_residual_to_query_aligned_key_delta(
    query: torch.Tensor,
    score_residual: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Construct the query-aligned key delta represented by a score residual."""
    if query.ndim != 4:
        raise ValueError("query must have shape (batch, heads, query_tokens, head_dim)")
    if score_residual.shape[:3] != query.shape[:3]:
        raise ValueError("score residual must align with query batch/head/token dimensions")
    norm_square = query.float().square().sum(dim=-1).clamp_min(eps)
    scale = math.sqrt(query.shape[-1]) * score_residual / norm_square.unsqueeze(-1)
    return scale.unsqueeze(-1) * query.float().unsqueeze(-2)


def build_relation_attention_score_kernel(
    kernel_type: str,
    config: RelationScoreKernelConfig,
) -> RelationAttentionScoreKernel:
    if kernel_type == "quantum":
        return QuantumRelationAttentionScoreKernel(config)
    if kernel_type == "classical":
        return ClassicalRelationAttentionScoreKernel(config)
    raise ValueError("kernel_type must be 'quantum' or 'classical'")


def save_relation_attention_score_kernel_checkpoint(
    path: str | Path,
    kernel: RelationAttentionScoreKernel,
    *,
    extra_metadata: Mapping[str, Any] | None = None,
) -> None:
    torch.save(
        {
            "format_version": SCORE_KERNEL_CHECKPOINT_VERSION,
            "kernel_metadata": kernel.metadata(),
            "state_dict": kernel.state_dict(),
            "extra_metadata": dict(extra_metadata or {}),
        },
        Path(path),
    )


def load_relation_attention_score_kernel_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[RelationAttentionScoreKernel, dict[str, Any]]:
    try:
        payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        payload = torch.load(Path(path), map_location=map_location)
    if (
        not isinstance(payload, Mapping)
        or payload.get("format_version") != SCORE_KERNEL_CHECKPOINT_VERSION
    ):
        raise ValueError("unsupported relation attention-score kernel checkpoint")
    metadata = payload.get("kernel_metadata")
    state_dict = payload.get("state_dict")
    if not isinstance(metadata, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError("relation attention-score kernel checkpoint is incomplete")
    raw_config = metadata.get("config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("relation attention-score kernel config is missing")
    kernel = build_relation_attention_score_kernel(
        str(metadata.get("type")),
        RelationScoreKernelConfig(**dict(raw_config)),
    )
    raw_selector = metadata.get("evidence_selector")
    if raw_selector is not None:
        if not isinstance(raw_selector, Mapping):
            raise ValueError("relation evidence-selector metadata is invalid")
        raw_selector_config = raw_selector.get("config")
        if not isinstance(raw_selector_config, Mapping):
            raise ValueError("relation evidence-selector config is missing")
        kernel.attach_evidence_selector(
            build_relation_evidence_selector(
                str(raw_selector.get("type")),
                RelationEvidenceSelectorConfig(**dict(raw_selector_config)),
            )
        )
    raw_router = metadata.get("expert_router")
    if raw_router is not None:
        if not isinstance(raw_router, Mapping):
            raise ValueError("relation expert-router metadata is invalid")
        raw_router_config = raw_router.get("config")
        if not isinstance(raw_router_config, Mapping):
            raise ValueError("relation expert-router config is missing")
        kernel.attach_expert_router(
            build_relation_expert_router(
                str(raw_router.get("type")),
                RelationExpertRouterConfig(**dict(raw_router_config)),
            )
        )
    kernel.load_state_dict(state_dict)
    extra = payload.get("extra_metadata")
    return kernel, dict(extra) if isinstance(extra, Mapping) else {}
