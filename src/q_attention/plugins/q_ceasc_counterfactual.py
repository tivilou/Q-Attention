"""Q-CEASC counterfactual key-influence mechanism.

The constructor first evaluates the support induced by the complete active
context, then removes each active context key once and measures the resulting
change in the projected support.  The diagonal key/support effect becomes a
context-only attention residual.  This is an isolated stage-A mechanism: it is
not wired into the production relation runner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import torch
import torch.nn as nn

from .q_ceasc import (
    QCEASCConfig,
    QCEASCResult,
    ContextEntangledAuxiliarySupportConstructor,
    _active_center,
)


QCEASC_COUNTERFACTUAL_CONTROL_MODES = (
    "q_ceasc_counterfactual",
    "classical_counterfactual",
    "quantum_product_counterfactual",
)


@dataclass(frozen=True)
class QCEASCCounterfactualConfig:
    """Configuration for leave-one-context-key support influence."""

    base: QCEASCConfig
    eps: float = 1e-8
    leave_one_out_chunk_size: int = 64

    def __post_init__(self) -> None:
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")
        if self.leave_one_out_chunk_size <= 0:
            raise ValueError("leave_one_out_chunk_size must be positive")

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
class QCEASCCounterfactualResult:
    residual: torch.Tensor
    support_vector: torch.Tensor
    projected_support: torch.Tensor
    coefficients: torch.Tensor
    auxiliary_state: torch.Tensor
    influence_scores: torch.Tensor
    influence_vectors: torch.Tensor
    full_result: QCEASCResult
    diagnostics: dict[str, Any]


def _base_mode(mode: str) -> str:
    return {
        "q_ceasc_counterfactual": "q_ceasc",
        "classical_counterfactual": "classical_span",
        "quantum_product_counterfactual": "quantum_product",
    }[mode]


def _zero_like_state(result: QCEASCResult) -> torch.Tensor:
    return torch.zeros_like(result.auxiliary_state)


class CounterfactualQCEASC(nn.Module):
    """Q-CEASC with a leave-one-key-out causal influence readout."""

    plugin_type = "q_ceasc_counterfactual_v1"

    def __init__(
        self,
        config: QCEASCCounterfactualConfig,
        control_mode: str = "q_ceasc_counterfactual",
        *,
        action_span: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if control_mode not in QCEASC_COUNTERFACTUAL_CONTROL_MODES:
            raise ValueError(
                "control_mode must be one of "
                f"{QCEASC_COUNTERFACTUAL_CONTROL_MODES}"
            )
        self.config = config
        self.control_mode = control_mode
        self.base = ContextEntangledAuxiliarySupportConstructor(
            config.base,
            _base_mode(control_mode),
            action_span=action_span,
        )

    @property
    def parameter_count(self) -> int:
        return self.base.parameter_count

    @property
    def support_dictionary(self) -> torch.Tensor:
        return self.base.support_dictionary

    @property
    def orthogonal_complement(self) -> torch.Tensor:
        return self.base.orthogonal_complement

    @property
    def span_basis(self) -> torch.Tensor:
        return self.base.span_basis

    def metadata(self) -> dict[str, Any]:
        return {
            "id": self.control_mode,
            "version": "0.1.0",
            "type": self.plugin_type,
            "mechanism": (
                "leave-one-active-context-key recomputation of projected quantum "
                "support followed by a diagonal key/support influence readout"
            ),
            "control_mode": self.control_mode,
            "input_schema": "query (B,Dq), key (B,N,Dk), valid/entity masks",
            "output_schema": "finite context-only zero-sum residual (B,N)",
            "parameter_count": self.parameter_count,
            "base_parameter_count": self.base.parameter_count,
            "leave_one_out": True,
            "leave_one_out_chunk_size": self.config.leave_one_out_chunk_size,
            "cost_model": {
                "base_evaluations_per_query_row": "1 + active_context_key_count",
                "base_kernel_calls_per_query_row": "1 + ceil(context_key_count / leave_one_out_chunk_size)",
                "peak_counterfactual_difference": "batch x leave_one_out_chunk_size x key_dim",
                "output_influence_tensor": "batch x active_context_key_count x key_dim",
                "score_wrapper_scaling": "ceil(query_token_count / query_chunk_size) x (1 + ceil(context_key_count / leave_one_out_chunk_size)) batched calls per head",
                "memory_control": "query_chunk_size and leave_one_out_chunk_size bound each materialized batched counterfactual call; retained training graphs still scale with all query chunks",
            },
            "numerical_stability": {
                "key_norm_floor": self.config.eps,
                "base_projection_floor": self.config.base.eps,
                "zero_norm_key_behavior": "zero normalized probe and finite zero contribution",
                "formal_preflight_required": True,
            },
            "action_span_rank": int(self.base.span_basis.shape[1]),
            "config": asdict(self.config),
            "quantum_resource_note": (
                "Exact statevector simulation establishes functional behavior only; "
                "it does not establish hardware speedup or quantum advantage."
            ),
        }

    def _validate_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        valid_context_mask: torch.Tensor,
        entity_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, active = self.base._validate_inputs(
            query, key, valid_context_mask, entity_mask
        )
        valid = valid_context_mask.to(device=key.device, dtype=torch.bool)
        entity = (
            torch.zeros_like(valid, dtype=torch.bool)
            if entity_mask is None
            else entity_mask.to(device=key.device, dtype=torch.bool)
        )
        return query, key, valid, entity & valid

    def evaluate(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        valid_context_mask: torch.Tensor,
        entity_mask: torch.Tensor | None = None,
    ) -> QCEASCCounterfactualResult:
        query, key, valid, entity = self._validate_inputs(
            query, key, valid_context_mask, entity_mask
        )
        active = valid & ~entity
        full_result = self.base.evaluate(query, key, valid, entity)
        batch, context_size, _ = key.shape
        # Batch several leave-one-out worlds into one base evaluation. The
        # counterfactual definition is unchanged, but this removes the Python
        # call per context key that otherwise dominates long-context runs.
        influence_chunks: list[torch.Tensor] = []
        loo_chunk_size = min(context_size, self.config.leave_one_out_chunk_size)
        for start in range(0, context_size, loo_chunk_size):
            stop = min(context_size, start + loo_chunk_size)
            key_indices = torch.arange(start, stop, device=key.device)
            width = stop - start
            current_row_ids = (
                torch.arange(batch, device=key.device).repeat_interleave(width)
            )
            current_key_indices = key_indices.repeat(batch)
            expanded_query = query.repeat_interleave(width, dim=0)
            expanded_key = key.repeat_interleave(width, dim=0)
            expanded_valid = valid.repeat_interleave(width, dim=0).clone()
            expanded_entity = entity.repeat_interleave(width, dim=0)
            flat_ids = torch.arange(batch * width, device=key.device)
            disable_active = active[current_row_ids, current_key_indices]
            expanded_valid[flat_ids, current_key_indices] = (
                expanded_valid[flat_ids, current_key_indices] & ~disable_active
            )
            leave_one_out = self.base.evaluate(
                expanded_query, expanded_key, expanded_valid, expanded_entity
            )
            full_support = full_result.projected_support.repeat_interleave(
                width, dim=0
            )
            difference = (full_support - leave_one_out.projected_support).reshape(
                batch, width, -1
            )
            influence_chunks.append(difference)
        influence = torch.cat(influence_chunks, dim=1)
        key_norm = torch.linalg.vector_norm(key, dim=-1, keepdim=True)
        key_unit = key / key_norm.clamp_min(self.config.eps)
        influence_scores = (key_unit * influence).sum(dim=-1)
        influence_scores = influence_scores * active.to(dtype=influence_scores.dtype)
        centered = _active_center(influence_scores, active, self.config.base.eps)
        gain_parameter = (
            self.base.classical_raw_gain
            if self.base.control_mode == "classical_span"
            else self.base.raw_gain
        )
        gain = self.config.base.max_gain * torch.tanh(
            gain_parameter.to(dtype=query.dtype)
        )
        residual = centered * gain
        masked = ~active
        masked_error = (
            residual.masked_select(masked).abs().max()
            if bool(masked.any())
            else residual.new_zeros(())
        )
        zero_sum_error = residual.sum(dim=-1).abs()
        influence_norm = torch.linalg.vector_norm(influence, dim=-1)
        active_count = active.sum(dim=-1)
        diagnostics: dict[str, Any] = {
            "control_mode": self.control_mode,
            "parameter_count": self.parameter_count,
            "base_parameter_count": self.base.parameter_count,
            "leave_one_out_count": active_count.detach(),
            "active_context_token_count": active_count.detach(),
            "influence_scores": influence_scores.detach(),
            "influence_norms": influence_norm.detach(),
            "influence_score_variance": influence_scores.var(
                dim=-1, unbiased=False
            ).detach(),
            "influence_norm_variance": influence_norm.var(
                dim=1, unbiased=False
            ).detach(),
            "counterfactual_effect_norm": torch.linalg.vector_norm(
                influence, dim=-1
            ).amax(dim=-1).detach(),
            "full_support_norm": torch.linalg.vector_norm(
                full_result.projected_support, dim=-1
            ).detach(),
            "out_of_span_support_norm": full_result.diagnostics[
                "out_of_span_support_norm"
            ].detach(),
            "in_span_support_norm": full_result.diagnostics[
                "in_span_support_norm"
            ].detach(),
            "projection_idempotence_error": full_result.diagnostics[
                "projection_idempotence_error"
            ],
            "span_leakage_norm": full_result.diagnostics["span_leakage_norm"].detach(),
            "support_basis_entropy": full_result.diagnostics[
                "support_basis_entropy"
            ].detach(),
            "auxiliary_state_norm": full_result.diagnostics[
                "auxiliary_state_norm"
            ].detach(),
            "entangling_covariance_norm": full_result.diagnostics[
                "entangling_covariance_norm"
            ].detach(),
            "mask_entity_zero_error": masked_error.detach(),
            "zero_sum_error": zero_sum_error.detach(),
            "finite": bool(torch.isfinite(residual).all()),
            "context_only": True,
            "target_free": True,
        }
        return QCEASCCounterfactualResult(
            residual=residual,
            support_vector=full_result.support_vector,
            projected_support=full_result.projected_support,
            coefficients=full_result.coefficients,
            auxiliary_state=full_result.auxiliary_state,
            influence_scores=influence_scores,
            influence_vectors=influence,
            full_result=full_result,
            diagnostics=diagnostics,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        valid_context_mask: torch.Tensor,
        entity_mask: torch.Tensor | None = None,
    ) -> QCEASCCounterfactualResult:
        return self.evaluate(query, key, valid_context_mask, entity_mask)


@dataclass(frozen=True)
class QCEASCCounterfactualScoreKernelConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    auxiliary_qubits: int = 3
    depth: int = 2
    support_width: int = 2
    action_rank: int = 2
    angle_scale: float = 1.0
    max_gain: float = 0.25
    initial_gain: float = 0.05
    span_rcond: float = 1e-6
    query_chunk_size: int = 4096
    leave_one_out_chunk_size: int = 64
    max_context_size: int = 256
    seed: int = 13091
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.num_layers <= 0 or self.num_heads <= 0 or self.head_dim <= 0:
            raise ValueError("num_layers, num_heads, and head_dim must be positive")
        if self.query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        if self.leave_one_out_chunk_size <= 0:
            raise ValueError("leave_one_out_chunk_size must be positive")
        if self.max_context_size <= 0:
            raise ValueError("max_context_size must be positive")
        if self.action_rank >= self.head_dim:
            raise ValueError("action_rank must leave a non-empty score complement")


class QCEASCCounterfactualScoreKernel(nn.Module):
    """Headwise wrapper for the bounded toy score contract."""

    kernel_type = "q_ceasc_counterfactual_score_kernel"

    def __init__(
        self,
        config: QCEASCCounterfactualScoreKernelConfig,
        control_mode: str = "q_ceasc_counterfactual",
    ) -> None:
        super().__init__()
        if control_mode not in QCEASC_COUNTERFACTUAL_CONTROL_MODES:
            raise ValueError(
                "control_mode must be one of "
                f"{QCEASC_COUNTERFACTUAL_CONTROL_MODES}"
            )
        self.config = config
        self.control_mode = control_mode
        # Optional inference-only hook used by the formal Case Study writer.
        # Training and ordinary evaluation leave this unset, so the numerical
        # path and gradients are unchanged.
        self.capture_callback: Callable[[dict[str, Any]], None] | None = None
        constructors: list[CounterfactualQCEASC] = []
        for layer_index in range(config.num_layers):
            for head_index in range(config.num_heads):
                base = QCEASCConfig(
                    query_dim=config.head_dim,
                    key_dim=config.head_dim,
                    auxiliary_qubits=config.auxiliary_qubits,
                    depth=config.depth,
                    support_width=config.support_width,
                    action_rank=config.action_rank,
                    angle_scale=config.angle_scale,
                    max_gain=config.max_gain,
                    initial_gain=config.initial_gain,
                    span_rcond=config.span_rcond,
                    seed=config.seed + 101 * layer_index + 17 * head_index,
                    eps=config.eps,
                )
                constructors.append(
                    CounterfactualQCEASC(
                        QCEASCCounterfactualConfig(
                            base=base,
                            eps=config.eps,
                            leave_one_out_chunk_size=config.leave_one_out_chunk_size,
                        ),
                        control_mode,
                    )
                )
        self.constructors = nn.ModuleList(constructors)

    @property
    def model_dimensions(self) -> tuple[int, int, int]:
        """Expose the adapter contract shared by production score kernels."""
        return self.config.num_layers, self.config.num_heads, self.config.head_dim

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def metadata(self) -> dict[str, Any]:
        """Return the stable wrapper contract used by runners and reports."""
        return {
            "type": self.kernel_type,
            "version": "1.0.0",
            "control_mode": self.control_mode,
            "mechanism": "per-key leave-one-out counterfactual influence over projected support",
            "input_schema": "query/key (B,H,T,D), attention and entity masks (B,T)",
            "output_schema": "zero-sum masked score residual (B,H,T,T)",
            "num_layers": self.config.num_layers,
            "num_heads": self.config.num_heads,
            "head_dim": self.config.head_dim,
            "parameter_count": self.parameter_count,
            "leave_one_out": True,
            "leave_one_out_chunk_size": self.config.leave_one_out_chunk_size,
            "query_chunk_size": self.config.query_chunk_size,
            "cost_model": {
                "base_evaluations_per_query_row": "1 + active_context_key_count",
                "counterfactual_chunking": "leave_one_out_chunk_size",
                "exact_evaluation_budget": "query_rows x (1 + active_context_key_count)",
            },
            "per_head_constructor": self.constructors[0].metadata(),
            "quantum_resource_note": (
                "Exact statevector simulation establishes functional behavior only; "
                "it does not establish hardware speedup or quantum advantage."
            ),
            "config": asdict(self.config),
        }

    def _constructor(self, layer_index: int, head_index: int) -> CounterfactualQCEASC:
        if not 0 <= layer_index < self.config.num_layers:
            raise ValueError("layer_index is outside configured layer range")
        if not 0 <= head_index < self.config.num_heads:
            raise ValueError("head_index is outside configured head range")
        return self.constructors[layer_index * self.config.num_heads + head_index]

    def fit_transport(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        return None

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
        del value, scores, evidence_view, random_seed, detach_random, routing_mode
        if query.ndim != 4 or key.shape != query.shape:
            raise ValueError("query and key must have shape (batch, heads, tokens, head_dim)")
        batch, heads, tokens, head_dim = query.shape
        if (heads, head_dim) != (self.config.num_heads, self.config.head_dim):
            raise ValueError("query shape does not match counterfactual score dimensions")
        if attention_mask.shape != (batch, tokens):
            raise ValueError("attention_mask must have shape (batch, tokens)")
        if tokens > self.config.max_context_size:
            raise ValueError(
                "context length exceeds the declared counterfactual preflight ceiling"
            )
        if subject_mask.shape != (batch, tokens) or object_mask.shape != (batch, tokens):
            raise ValueError("subject/object masks must have shape (batch, tokens)")
        if query.device != key.device:
            raise ValueError("query and key must be on the same device")
        valid = attention_mask.to(device=query.device, dtype=torch.bool)
        entity = (
            subject_mask.to(device=query.device, dtype=torch.bool)
            | object_mask.to(device=query.device, dtype=torch.bool)
        )
        output_heads: list[torch.Tensor] = []
        chunk_size = min(tokens, int(self.config.query_chunk_size))
        for head_index in range(heads):
            head_query = query[:, head_index]
            head_key = key[:, head_index]
            chunks: list[torch.Tensor] = []
            constructor = self._constructor(layer_index, head_index)
            for start in range(0, tokens, chunk_size):
                stop = min(tokens, start + chunk_size)
                width = stop - start
                query_rows = head_query[:, start:stop, :].reshape(batch * width, head_dim)
                key_rows = head_key.repeat_interleave(width, dim=0)
                valid_rows = valid.repeat_interleave(width, dim=0)
                entity_rows = entity.repeat_interleave(width, dim=0)
                result = constructor.evaluate(query_rows, key_rows, valid_rows, entity_rows)
                if self.capture_callback is not None:
                    self.capture_callback(
                        {
                            "layer_index": int(layer_index),
                            "head_index": int(head_index),
                            "query_start": int(start),
                            "query_stop": int(stop),
                            "batch_size": int(batch),
                            "query_width": int(width),
                            "residual": result.residual.reshape(batch, width, tokens),
                            "projected_support": result.projected_support.reshape(batch, width, -1),
                            "coefficients": result.coefficients.reshape(batch, width, -1),
                            "auxiliary_state": result.auxiliary_state.reshape(batch, width, -1),
                            "influence_scores": result.influence_scores.reshape(batch, width, tokens),
                        }
                    )
                chunks.append(result.residual.reshape(batch, width, tokens))
            output_heads.append(torch.cat(chunks, dim=1))
        residual = torch.stack(output_heads, dim=1)
        return residual * valid[:, None, :, None].to(dtype=residual.dtype)


def build_qceasc_counterfactual(
    mode: str,
    config: QCEASCCounterfactualConfig,
    *,
    action_span: torch.Tensor | None = None,
) -> CounterfactualQCEASC:
    if mode not in QCEASC_COUNTERFACTUAL_CONTROL_MODES:
        raise ValueError(
            "mode must be one of " f"{QCEASC_COUNTERFACTUAL_CONTROL_MODES}"
        )
    return CounterfactualQCEASC(config, mode, action_span=action_span)


def build_qceasc_counterfactual_score_kernel(
    mode: str,
    config: QCEASCCounterfactualScoreKernelConfig,
) -> QCEASCCounterfactualScoreKernel:
    if mode not in QCEASC_COUNTERFACTUAL_CONTROL_MODES:
        raise ValueError(
            "mode must be one of " f"{QCEASC_COUNTERFACTUAL_CONTROL_MODES}"
        )
    return QCEASCCounterfactualScoreKernel(config, mode)
