"""Stability-certified Q-CEASC support construction.

This module keeps the original Q-CEASC constructor unchanged and composes two
independent, label-free support views.  A residual is applied only in the
direction on which the two views agree.  The same composition is available for
the matched classical and product-state controls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import torch
import torch.nn as nn

from .q_ceasc import (
    QCEASCConfig,
    QCEASCResult,
    ContextEntangledAuxiliarySupportConstructor,
)
from .q_ceasc_score import QCEASCScoreKernel, QCEASCScoreKernelConfig


QCEASC_STABILITY_CONTROL_MODES = (
    "q_ceasc_stability",
    "classical_stability",
    "quantum_product_stability",
)


@dataclass(frozen=True)
class QCEASCStabilityConfig:
    """Configuration for two-view stability certification.

    ``agreement_threshold`` is frozen before an experiment.  The default zero
    threshold means that anti-aligned views abstain while positively aligned
    views are weighted by their cosine agreement; it is not tuned from labels
    or test outcomes.
    """

    base: QCEASCConfig
    view_seed_stride: int = 100_003
    agreement_threshold: float = 0.0
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.view_seed_stride == 0:
            raise ValueError("view_seed_stride must be non-zero")
        if not 0.0 <= self.agreement_threshold < 1.0:
            raise ValueError("agreement_threshold must lie in [0, 1)")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")

    def view_config(self, view_index: int) -> QCEASCConfig:
        if view_index not in (0, 1):
            raise ValueError("view_index must be 0 or 1")
        return replace(
            self.base,
            seed=self.base.seed + view_index * self.view_seed_stride,
        )

    @property
    def query_dim(self) -> int:
        return self.base.query_dim

    @property
    def key_dim(self) -> int:
        return self.base.key_dim

    @property
    def support_width(self) -> int:
        return self.base.support_width

    @property
    def action_rank(self) -> int:
        return self.base.action_rank


@dataclass(frozen=True)
class QCEASCStabilityResult:
    residual: torch.Tensor
    support_vector: torch.Tensor
    projected_support: torch.Tensor
    coefficients: torch.Tensor
    auxiliary_state: torch.Tensor
    view_one: QCEASCResult
    view_two: QCEASCResult
    diagnostics: dict[str, Any]


def _agreement_gate(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    active: torch.Tensor,
    threshold: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-example cosine agreement and a non-negative gate."""

    weights = active.to(device=first.device, dtype=first.dtype)
    dot = (first * second * weights).sum(dim=-1)
    first_norm = torch.sqrt((first.square() * weights).sum(dim=-1).clamp_min(eps))
    second_norm = torch.sqrt((second.square() * weights).sum(dim=-1).clamp_min(eps))
    denominator = first_norm * second_norm
    agreement = (dot / denominator.clamp_min(eps)).clamp(min=-1.0, max=1.0)
    gate = (agreement - threshold).clamp_min(0.0)
    if threshold > 0.0:
        gate = gate / (1.0 - threshold)
    return agreement, gate.clamp(max=1.0)


def _shared_frozen_geometry(
    first: ContextEntangledAuxiliarySupportConstructor,
    second: ContextEntangledAuxiliarySupportConstructor,
) -> None:
    """Share the frozen output geometry, retaining independent view parameters."""

    with torch.no_grad():
        second.action_span.copy_(first.action_span)
        second.span_basis.copy_(first.span_basis)
        second.span_singular_values.copy_(first.span_singular_values)
        second.orthogonal_complement.copy_(first.orthogonal_complement)
        second.support_dictionary.copy_(first.support_dictionary)
        second.context_projection.copy_(first.context_projection)
        second.pair_projection.copy_(first.pair_projection)
        second.classical_feature_tensor.copy_(first.classical_feature_tensor)


def _share_initial_trainable_state(
    first: ContextEntangledAuxiliarySupportConstructor,
    second: ContextEntangledAuxiliarySupportConstructor,
) -> None:
    """Start both views from the same trainable encoder state.

    The views remain separate parameters after initialization; their fixed
    observable banks are the intended independent readout source.
    """

    with torch.no_grad():
        for first_parameter, second_parameter in zip(
            first.parameters(), second.parameters()
        ):
            second_parameter.copy_(first_parameter)


class StabilityCertifiedQCEASC(nn.Module):
    """Two-view Q-CEASC with label-free agreement-conditioned abstention."""

    plugin_type = "q_ceasc_stability_v1"

    def __init__(
        self,
        config: QCEASCStabilityConfig,
        control_mode: str = "q_ceasc_stability",
        *,
        action_span: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if control_mode not in QCEASC_STABILITY_CONTROL_MODES:
            raise ValueError(
                f"control_mode must be one of {QCEASC_STABILITY_CONTROL_MODES}"
            )
        self.config = config
        self.control_mode = control_mode
        base_mode = {
            "q_ceasc_stability": "q_ceasc",
            "classical_stability": "classical_span",
            "quantum_product_stability": "quantum_product",
        }[control_mode]
        first = ContextEntangledAuxiliarySupportConstructor(
            config.view_config(0), base_mode, action_span=action_span
        )
        second = ContextEntangledAuxiliarySupportConstructor(
            config.view_config(1), base_mode, action_span=first.action_span.detach().clone()
        )
        _shared_frozen_geometry(first, second)
        _share_initial_trainable_state(first, second)
        self.view_one = first
        self.view_two = second

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def support_dictionary(self) -> torch.Tensor:
        return self.view_one.support_dictionary

    @property
    def orthogonal_complement(self) -> torch.Tensor:
        return self.view_one.orthogonal_complement

    @property
    def span_basis(self) -> torch.Tensor:
        return self.view_one.span_basis

    def metadata(self) -> dict[str, Any]:
        return {
            "id": self.control_mode,
            "version": "0.1.0",
            "type": self.plugin_type,
            "mechanism": (
                "two independent context-conditioned support views with shared "
                "frozen action geometry and agreement-conditioned residual gating"
            ),
            "control_mode": self.control_mode,
            "input_schema": "query (B,Dq), key (B,N,Dk), valid/entity masks",
            "output_schema": (
                "finite context-only zero-sum bounded residual plus view agreement "
                "and abstention gate"
            ),
            "parameter_count": self.parameter_count,
            "single_view_parameter_count": self.view_one.parameter_count,
            "view_seed_stride": self.config.view_seed_stride,
            "agreement_threshold": self.config.agreement_threshold,
            "shared_frozen_action_geometry": True,
            "shared_frozen_support_dictionary": True,
            "label_free_gate": True,
            "view_one": self.view_one.metadata(),
            "view_two": self.view_two.metadata(),
            "config": asdict(self.config),
        }

    def evaluate(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        valid_context_mask: torch.Tensor,
        entity_mask: torch.Tensor | None = None,
    ) -> QCEASCStabilityResult:
        first = self.view_one.evaluate(query, key, valid_context_mask, entity_mask)
        second = self.view_two.evaluate(query, key, valid_context_mask, entity_mask)
        valid = valid_context_mask.to(device=query.device, dtype=torch.bool)
        entity = (
            torch.zeros_like(valid, dtype=torch.bool)
            if entity_mask is None
            else entity_mask.to(device=query.device, dtype=torch.bool)
        )
        active = valid & ~entity
        agreement, gate = _agreement_gate(
            first.residual,
            second.residual,
            active=active,
            threshold=self.config.agreement_threshold,
            eps=self.config.eps,
        )
        gate_column = gate.unsqueeze(-1)
        residual = 0.5 * (first.residual + second.residual) * gate_column
        support_vector = 0.5 * (first.support_vector + second.support_vector) * gate_column
        projected_support = (
            0.5 * (first.projected_support + second.projected_support) * gate_column
        )
        coefficients = 0.5 * (first.coefficients + second.coefficients) * gate_column
        auxiliary_state = 0.5 * (first.auxiliary_state + second.auxiliary_state)
        diagnostics: dict[str, Any] = {
            "control_mode": self.control_mode,
            "parameter_count": self.parameter_count,
            "single_view_parameter_count": self.view_one.parameter_count,
            "agreement": agreement.detach(),
            "agreement_gate": gate.detach(),
            "gate_active_fraction": (gate > 0).to(dtype=query.dtype).detach(),
            "view_one_residual_norm": torch.linalg.vector_norm(
                first.residual, dim=-1
            ).detach(),
            "view_two_residual_norm": torch.linalg.vector_norm(
                second.residual, dim=-1
            ).detach(),
            "gated_residual_norm": torch.linalg.vector_norm(
                residual, dim=-1
            ).detach(),
            "view_agreement_is_label_free": True,
            "out_of_span_residual_norm": torch.linalg.vector_norm(
                projected_support, dim=-1
            ).detach(),
            "out_of_span_support_norm": torch.linalg.vector_norm(
                projected_support, dim=-1
            ).detach(),
            "in_span_support_norm": 0.5
            * (
                first.diagnostics["in_span_support_norm"]
                + second.diagnostics["in_span_support_norm"]
            ).detach(),
            "projection_idempotence_error": max(
                float(first.diagnostics["projection_idempotence_error"]),
                float(second.diagnostics["projection_idempotence_error"]),
            ),
            "span_leakage_norm": torch.maximum(
                first.diagnostics["span_leakage_norm"],
                second.diagnostics["span_leakage_norm"],
            ).detach(),
            "support_basis_entropy": 0.5
            * (
                first.diagnostics["support_basis_entropy"]
                + second.diagnostics["support_basis_entropy"]
            ).detach(),
            "auxiliary_state_norm": torch.linalg.vector_norm(
                auxiliary_state, dim=-1
            ).real.detach(),
            "entangling_covariance_norm": 0.5
            * (
                first.diagnostics["entangling_covariance_norm"]
                + second.diagnostics["entangling_covariance_norm"]
            ).detach(),
            "mask_entity_zero_error": torch.maximum(
                first.diagnostics["mask_entity_zero_error"],
                second.diagnostics["mask_entity_zero_error"],
            ).detach(),
            "zero_sum_error": residual.sum(dim=-1).abs().detach(),
            "effective_span_rank": int(first.diagnostics["effective_span_rank"]),
            "effective_complement_rank": int(
                first.diagnostics["effective_complement_rank"]
            ),
            "span_rcond": first.diagnostics["span_rcond"],
            "readout": "agreement_gated_signed_bipolar_observable_expectations",
            "context_is_label_free": True,
            "valid_context_token_count": first.diagnostics[
                "valid_context_token_count"
            ],
            "entity_context_token_count": first.diagnostics[
                "entity_context_token_count"
            ],
            "active_context_token_count": first.diagnostics[
                "active_context_token_count"
            ],
            "empty_context_row": first.diagnostics["empty_context_row"],
        }
        return QCEASCStabilityResult(
            residual=residual,
            support_vector=support_vector,
            projected_support=projected_support,
            coefficients=coefficients,
            auxiliary_state=auxiliary_state,
            view_one=first,
            view_two=second,
            diagnostics=diagnostics,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        valid_context_mask: torch.Tensor,
        entity_mask: torch.Tensor | None = None,
    ) -> QCEASCStabilityResult:
        return self.evaluate(query, key, valid_context_mask, entity_mask)


class QCEASCStabilityScoreKernel(nn.Module):
    """Production-shaped score wrapper for a stability-certified residual."""

    kernel_type = "q_ceasc_stability_score_kernel"

    def __init__(
        self,
        config: QCEASCScoreKernelConfig,
        control_mode: str = "q_ceasc_stability",
        *,
        view_seed_stride: int = 100_003,
        agreement_threshold: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if control_mode not in QCEASC_STABILITY_CONTROL_MODES:
            raise ValueError(
                f"control_mode must be one of {QCEASC_STABILITY_CONTROL_MODES}"
            )
        if view_seed_stride == 0:
            raise ValueError("view_seed_stride must be non-zero")
        if not 0.0 <= agreement_threshold < 1.0:
            raise ValueError("agreement_threshold must lie in [0, 1)")
        self.config = config
        self.control_mode = control_mode
        self.view_seed_stride = view_seed_stride
        self.agreement_threshold = agreement_threshold
        self.eps = eps
        base_mode = {
            "q_ceasc_stability": "q_ceasc",
            "classical_stability": "classical_span",
            "quantum_product_stability": "quantum_product",
        }[control_mode]
        first = QCEASCScoreKernel(config, base_mode)
        second = QCEASCScoreKernel(
            replace(config, seed=config.seed + view_seed_stride), base_mode
        )
        for first_constructor, second_constructor in zip(
            first.constructors, second.constructors
        ):
            _shared_frozen_geometry(first_constructor, second_constructor)
            _share_initial_trainable_state(first_constructor, second_constructor)
        self.view_one = first
        self.view_two = second

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def model_dimensions(self) -> tuple[int, int, int]:
        return self.config.num_layers, self.config.num_heads, self.config.head_dim

    def metadata(self) -> dict[str, Any]:
        return {
            "type": self.kernel_type,
            "version": "0.1.0",
            "control_mode": self.control_mode,
            "mechanism": (
                "two independent Q-CEASC score views with shared frozen action "
                "geometry and agreement-conditioned residual gating"
            ),
            "input_schema": "query/key (B,H,T,D), attention and entity masks (B,T)",
            "output_schema": "zero-sum masked agreement-gated residual (B,H,T,T)",
            "num_layers": self.config.num_layers,
            "num_heads": self.config.num_heads,
            "head_dim": self.config.head_dim,
            "parameter_count": self.parameter_count,
            "single_view_parameter_count": self.view_one.parameter_count,
            "view_seed_stride": self.view_seed_stride,
            "agreement_threshold": self.agreement_threshold,
            "shared_frozen_action_geometry": True,
            "shared_frozen_support_dictionary": True,
            "label_free_gate": True,
            "view_one": self.view_one.metadata(),
            "view_two": self.view_two.metadata(),
        }

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
        evidence_view: str = "full",
        random_seed: int = 0,
        detach_random: bool = False,
        routing_mode: str = "learned",
    ) -> torch.Tensor:
        first = self.view_one(
            query,
            key,
            value,
            scores=scores,
            layer_index=layer_index,
            attention_mask=attention_mask,
            subject_mask=subject_mask,
            object_mask=object_mask,
            evidence_view=evidence_view,
            random_seed=random_seed,
            detach_random=detach_random,
            routing_mode=routing_mode,
        )
        second = self.view_two(
            query,
            key,
            value,
            scores=scores,
            layer_index=layer_index,
            attention_mask=attention_mask,
            subject_mask=subject_mask,
            object_mask=object_mask,
            evidence_view=evidence_view,
            random_seed=random_seed,
            detach_random=detach_random,
            routing_mode=routing_mode,
        )
        valid = attention_mask.to(device=query.device, dtype=torch.bool)
        entity = subject_mask.to(device=query.device, dtype=torch.bool) | object_mask.to(
            device=query.device, dtype=torch.bool
        )
        active = (valid & ~entity)[:, None, None, :]
        agreement, gate = _agreement_gate(
            first,
            second,
            active=active.expand_as(first),
            threshold=self.agreement_threshold,
            eps=self.eps,
        )
        return 0.5 * (first + second) * gate.unsqueeze(-1)


def build_qceasc_stability(
    mode: str,
    config: QCEASCStabilityConfig,
    *,
    action_span: torch.Tensor | None = None,
) -> StabilityCertifiedQCEASC:
    if mode not in QCEASC_STABILITY_CONTROL_MODES:
        raise ValueError(f"mode must be one of {QCEASC_STABILITY_CONTROL_MODES}")
    return StabilityCertifiedQCEASC(config, mode, action_span=action_span)


def build_qceasc_stability_score_kernel(
    mode: str,
    config: QCEASCScoreKernelConfig,
    *,
    view_seed_stride: int = 100_003,
    agreement_threshold: float = 0.0,
) -> QCEASCStabilityScoreKernel:
    if mode not in QCEASC_STABILITY_CONTROL_MODES:
        raise ValueError(f"mode must be one of {QCEASC_STABILITY_CONTROL_MODES}")
    return QCEASCStabilityScoreKernel(
        config,
        mode,
        view_seed_stride=view_seed_stride,
        agreement_threshold=agreement_threshold,
    )
