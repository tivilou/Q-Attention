"""Grouped counterfactual Q-CEASC support influence.

This isolated toy mechanism replaces the predecessor's one-key deletion with a
predeclared group intervention.  A group manifest is a context-only integer
tensor supplied before scoring; labels, predictions, and gold relations never
enter the constructor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import torch
import torch.nn as nn

from .q_ceasc import QCEASCConfig, QCEASCResult, ContextEntangledAuxiliarySupportConstructor, _active_center


QCEASC_GROUPED_COUNTERFACTUAL_CONTROL_MODES = (
    "q_ceasc_grouped_counterfactual",
    "classical_grouped_counterfactual",
    "random_grouped_counterfactual",
)


def _base_mode(mode: str) -> str:
    return {
        "q_ceasc_grouped_counterfactual": "q_ceasc",
        "classical_grouped_counterfactual": "classical_span",
        "random_grouped_counterfactual": "q_ceasc",
    }[mode]


@dataclass(frozen=True)
class QCEASCGroupedCounterfactualConfig:
    """Configuration for a label-free group leave-one-out intervention."""

    base: QCEASCConfig
    group_size: int = 2
    group_intervention_chunk_size: int = 64
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.group_intervention_chunk_size <= 0:
            raise ValueError("group_intervention_chunk_size must be positive")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")


@dataclass(frozen=True)
class QCEASCGroupedCounterfactualResult:
    residual: torch.Tensor
    support_vector: torch.Tensor
    projected_support: torch.Tensor
    coefficients: torch.Tensor
    auxiliary_state: torch.Tensor
    group_ids: torch.Tensor
    group_masked_supports: torch.Tensor
    group_influence_vectors: torch.Tensor
    group_scores: torch.Tensor
    member_scores: torch.Tensor
    full_result: QCEASCResult
    diagnostics: dict[str, Any]


def build_fixed_group_manifest(
    valid_context_mask: torch.Tensor,
    entity_mask: torch.Tensor | None,
    *,
    group_size: int,
) -> torch.Tensor:
    """Return a frozen positional group manifest, with -1 outside active context."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if valid_context_mask.ndim != 2:
        raise ValueError("valid_context_mask must have shape (batch, context_keys)")
    valid = valid_context_mask.to(dtype=torch.bool)
    entity = (
        torch.zeros_like(valid)
        if entity_mask is None
        else entity_mask.to(device=valid.device, dtype=torch.bool)
    )
    if entity.shape != valid.shape:
        raise ValueError("entity_mask must match valid_context_mask")
    positions = torch.arange(valid.shape[1], device=valid.device).unsqueeze(0)
    manifest = (positions // group_size).expand_as(valid).clone()
    return manifest.masked_fill(~(valid & ~entity), -1)


def build_seeded_random_group_manifest(
    valid_context_mask: torch.Tensor,
    entity_mask: torch.Tensor | None,
    *,
    group_size: int,
    seed: int,
) -> torch.Tensor:
    """Return a reproducible label-free random grouping with matched group sizes.

    The only inputs are masks, positions, a fixed seed, and the group size.  It
    is a structural control: the caller freezes this manifest before scoring
    and supplies it exactly as it supplies the positional manifest.
    """
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if valid_context_mask.ndim != 2:
        raise ValueError("valid_context_mask must have shape (batch, context_keys)")
    valid = valid_context_mask.to(dtype=torch.bool)
    entity = (
        torch.zeros_like(valid)
        if entity_mask is None
        else entity_mask.to(device=valid.device, dtype=torch.bool)
    )
    if entity.shape != valid.shape:
        raise ValueError("entity_mask must match valid_context_mask")
    manifest = torch.full_like(valid, -1, dtype=torch.long)
    for row_index in range(valid.shape[0]):
        active_positions = torch.nonzero(valid[row_index] & ~entity[row_index], as_tuple=False).flatten()
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + row_index)
        shuffled = active_positions.cpu()[torch.randperm(active_positions.numel(), generator=generator)]
        manifest[row_index, shuffled.to(device=valid.device)] = torch.arange(
            shuffled.numel(), device=valid.device, dtype=torch.long
        ) // group_size
    return manifest


class GroupedCounterfactualQCEASC(nn.Module):
    """Q-CEASC with a predeclared group-wise support influence readout."""

    plugin_type = "q_ceasc_grouped_counterfactual_v1"

    def __init__(
        self,
        config: QCEASCGroupedCounterfactualConfig,
        control_mode: str = "q_ceasc_grouped_counterfactual",
        *,
        action_span: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if control_mode not in QCEASC_GROUPED_COUNTERFACTUAL_CONTROL_MODES:
            raise ValueError(
                "control_mode must be one of "
                f"{QCEASC_GROUPED_COUNTERFACTUAL_CONTROL_MODES}"
            )
        self.config = config
        self.control_mode = control_mode
        self.base = ContextEntangledAuxiliarySupportConstructor(
            config.base, _base_mode(control_mode), action_span=action_span
        )

    @property
    def parameter_count(self) -> int:
        return self.base.parameter_count

    def metadata(self) -> dict[str, Any]:
        return {
            "id": self.control_mode,
            "version": "0.1.0",
            "type": self.plugin_type,
            "mechanism": (
                "predeclared label-free group removal of projected support, "
                "size-normalized group action score, and member distribution"
            ),
            "control_mode": self.control_mode,
            "input_schema": "query (B,Dq), key (B,N,Dk), masks, group_ids (B,N)",
            "output_schema": "finite context-only zero-sum residual (B,N)",
            "parameter_count": self.parameter_count,
            "group_size": self.config.group_size,
            "cost_model": {
                "base_evaluations_per_query_row": "1 + active_group_count",
                "base_kernel_calls_per_query_row": "1 + ceil(active_group_count / group_intervention_chunk_size)",
                "group_intervention_chunk_size": self.config.group_intervention_chunk_size,
                "group_manifest": "frozen_label_free positional blocks unless explicitly supplied; random control uses a separately frozen seeded manifest with the same group sizes",
                "random_grouping": self.control_mode == "random_grouped_counterfactual",
                "group_intervention": "all active members of one declared group are masked together",
            },
            "numerical_stability": {
                "key_norm_floor": self.config.eps,
                "base_projection_floor": self.config.base.eps,
                "zero_norm_key_behavior": "zero normalized member probe and finite zero contribution",
                "formal_preflight_required": True,
            },
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
        group_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, active = self.base._validate_inputs(
            query, key, valid_context_mask, entity_mask
        )
        valid = valid_context_mask.to(device=key.device, dtype=torch.bool)
        entity = (
            torch.zeros_like(valid)
            if entity_mask is None
            else entity_mask.to(device=key.device, dtype=torch.bool)
        ) & valid
        if group_ids is None:
            manifest = build_fixed_group_manifest(
                valid, entity, group_size=self.config.group_size
            )
        else:
            if group_ids.shape != key.shape[:2]:
                raise ValueError("group_ids must have shape (batch, context_keys)")
            if group_ids.device != key.device:
                raise ValueError("group_ids must be on the same device as key")
            if group_ids.is_floating_point() or group_ids.is_complex():
                raise ValueError("group_ids must be an integer tensor")
            manifest = group_ids.to(dtype=torch.long)
            if bool((manifest[active] < 0).any()):
                raise ValueError("every active key must have a non-negative group id")
            if bool((manifest[active] >= key.shape[1]).any()):
                raise ValueError("group ids must be smaller than context_keys")
            manifest = manifest.masked_fill(~active, -1)
        return query, key, valid, entity, manifest

    def evaluate(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        valid_context_mask: torch.Tensor,
        entity_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
    ) -> QCEASCGroupedCounterfactualResult:
        query, key, valid, entity, manifest = self._validate_inputs(
            query, key, valid_context_mask, entity_mask, group_ids
        )
        active = valid & ~entity
        full_result = self.base.evaluate(query, key, valid, entity)
        batch, context_size, _ = key.shape
        group_influence = key.new_zeros((batch, context_size, key.shape[-1]))
        group_supports = key.new_zeros((batch, context_size, key.shape[-1]))
        group_present = torch.zeros((batch, context_size), dtype=torch.bool, device=key.device)
        group_present_any = torch.zeros(context_size, dtype=torch.bool, device=key.device)
        present_group_ids = torch.unique(manifest[active], sorted=True)
        chunk_size = min(
            max(1, int(present_group_ids.numel())),
            self.config.group_intervention_chunk_size,
        )
        group_kernel_calls = 0
        for start in range(0, int(present_group_ids.numel()), chunk_size):
            group_ids = present_group_ids[start : start + chunk_size]
            width = int(group_ids.numel())
            member_mask = active[:, None, :] & (
                manifest[:, None, :] == group_ids[None, :, None]
            )
            present = member_mask.any(dim=-1)
            expanded_query = query.repeat_interleave(width, dim=0)
            expanded_key = key.repeat_interleave(width, dim=0)
            expanded_valid = (
                valid[:, None, :].expand(batch, width, context_size) & ~member_mask
            ).reshape(batch * width, context_size)
            expanded_entity = entity.repeat_interleave(width, dim=0)
            without_group = self.base.evaluate(
                expanded_query, expanded_key, expanded_valid, expanded_entity
            )
            without_support = without_group.projected_support.reshape(batch, width, -1)
            difference = full_result.projected_support[:, None, :] - without_support
            present_float = present.unsqueeze(-1).to(dtype=key.dtype)
            group_influence[:, group_ids] = difference * present_float
            group_supports[:, group_ids] = without_support * present_float
            group_present[:, group_ids] = present
            group_present_any[group_ids] = present.any(dim=0)
            group_kernel_calls += 1

        key_norm = torch.linalg.vector_norm(key, dim=-1, keepdim=True)
        key_unit = key / key_norm.clamp_min(self.config.eps)
        member_counts = torch.zeros((batch, context_size), dtype=key.dtype, device=key.device)
        group_probe = key.new_zeros((batch, context_size, key.shape[-1]))
        for group_id in range(context_size):
            member_mask = active & (manifest == group_id)
            count = member_mask.sum(dim=-1).to(dtype=key.dtype)
            member_counts[:, group_id] = count
            group_probe[:, group_id] = (
                (key_unit * member_mask.unsqueeze(-1).to(dtype=key.dtype)).sum(dim=1)
                / count.clamp_min(1.0).unsqueeze(-1)
            )
        raw_group_scores = (group_probe * group_influence).sum(dim=-1)
        raw_group_scores = raw_group_scores * group_present.to(dtype=raw_group_scores.dtype)
        safe_manifest = manifest.clamp_min(0)
        gathered_scores = raw_group_scores.gather(1, safe_manifest)
        gathered_counts = member_counts.gather(1, safe_manifest).clamp_min(1.0)
        member_scores = (gathered_scores / gathered_counts) * active.to(dtype=key.dtype)
        centered = _active_center(member_scores, active, self.config.base.eps)
        gain_parameter = (
            self.base.classical_raw_gain
            if self.base.control_mode == "classical_span"
            else self.base.raw_gain
        )
        gain = self.config.base.max_gain * torch.tanh(gain_parameter.to(dtype=query.dtype))
        residual = centered * gain
        masked = ~active
        masked_error = (
            residual.masked_select(masked).abs().max()
            if bool(masked.any())
            else residual.new_zeros(())
        )
        diagnostics: dict[str, Any] = {
            "control_mode": self.control_mode,
            "parameter_count": self.parameter_count,
            "group_manifest": manifest.detach(),
            "active_group_count": group_present.sum(dim=-1).detach(),
            "group_evaluation_count": int(group_present_any.sum().item()),
            "group_kernel_call_count": group_kernel_calls,
            "group_member_counts": member_counts.detach(),
            "group_scores": raw_group_scores.detach(),
            "group_influence_norms": torch.linalg.vector_norm(group_influence, dim=-1).detach(),
            "group_influence_variance": torch.linalg.vector_norm(group_influence, dim=-1).var(dim=-1, unbiased=False).detach(),
            "member_score_variance": member_scores.var(dim=-1, unbiased=False).detach(),
            "mask_entity_zero_error": masked_error.detach(),
            "zero_sum_error": residual.sum(dim=-1).abs().detach(),
            "finite": bool(torch.isfinite(residual).all()),
            "context_only": True,
            "target_free": True,
            "full_support_norm": torch.linalg.vector_norm(full_result.projected_support, dim=-1).detach(),
            "out_of_span_support_norm": full_result.diagnostics["out_of_span_support_norm"].detach(),
            "projection_idempotence_error": full_result.diagnostics["projection_idempotence_error"],
            "span_leakage_norm": full_result.diagnostics["span_leakage_norm"].detach(),
        }
        return QCEASCGroupedCounterfactualResult(
            residual=residual,
            support_vector=full_result.support_vector,
            projected_support=full_result.projected_support,
            coefficients=full_result.coefficients,
            auxiliary_state=full_result.auxiliary_state,
            group_ids=manifest,
            group_masked_supports=group_supports,
            group_influence_vectors=group_influence,
            group_scores=raw_group_scores,
            member_scores=member_scores,
            full_result=full_result,
            diagnostics=diagnostics,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        valid_context_mask: torch.Tensor,
        entity_mask: torch.Tensor | None = None,
        group_ids: torch.Tensor | None = None,
    ) -> QCEASCGroupedCounterfactualResult:
        return self.evaluate(query, key, valid_context_mask, entity_mask, group_ids)


@dataclass(frozen=True)
class QCEASCGroupedCounterfactualScoreKernelConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    group_size: int = 2
    group_intervention_chunk_size: int = 64
    auxiliary_qubits: int = 3
    depth: int = 2
    support_width: int = 2
    action_rank: int = 2
    angle_scale: float = 1.0
    max_gain: float = 0.25
    initial_gain: float = 0.05
    span_rcond: float = 1e-6
    query_chunk_size: int = 4096
    max_context_size: int = 256
    seed: int = 13091
    random_group_seed: int = 13053
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.num_layers <= 0 or self.num_heads <= 0 or self.head_dim <= 0:
            raise ValueError("num_layers, num_heads, and head_dim must be positive")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.group_intervention_chunk_size <= 0:
            raise ValueError("group_intervention_chunk_size must be positive")
        if self.query_chunk_size <= 0 or self.max_context_size <= 0:
            raise ValueError("query_chunk_size and max_context_size must be positive")
        if self.action_rank >= self.head_dim:
            raise ValueError("action_rank must leave a non-empty score complement")


class QCEASCGroupedCounterfactualScoreKernel(nn.Module):
    """Headwise score wrapper using the frozen positional group manifest."""

    kernel_type = "q_ceasc_grouped_counterfactual_score_kernel"

    def __init__(
        self,
        config: QCEASCGroupedCounterfactualScoreKernelConfig,
        control_mode: str = "q_ceasc_grouped_counterfactual",
    ) -> None:
        super().__init__()
        if control_mode not in QCEASC_GROUPED_COUNTERFACTUAL_CONTROL_MODES:
            raise ValueError(
                "control_mode must be one of "
                f"{QCEASC_GROUPED_COUNTERFACTUAL_CONTROL_MODES}"
            )
        self.config = config
        self.control_mode = control_mode
        self.capture_callback: Callable[[dict[str, Any]], None] | None = None
        constructors: list[GroupedCounterfactualQCEASC] = []
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
                    GroupedCounterfactualQCEASC(
                        QCEASCGroupedCounterfactualConfig(
                            base=base,
                            group_size=config.group_size,
                            group_intervention_chunk_size=config.group_intervention_chunk_size,
                            eps=config.eps,
                        ),
                        control_mode,
                    )
                )
        self.constructors = nn.ModuleList(constructors)

    @property
    def model_dimensions(self) -> tuple[int, int, int]:
        return self.config.num_layers, self.config.num_heads, self.config.head_dim

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def metadata(self) -> dict[str, Any]:
        return {
            "type": self.kernel_type,
            "version": "0.1.0",
            "control_mode": self.control_mode,
            "mechanism": "frozen group leave-one-out counterfactual influence over projected support",
            "input_schema": "query/key (B,H,T,D), attention and entity masks (B,T)",
            "output_schema": "zero-sum masked score residual (B,H,T,T)",
            "num_layers": self.config.num_layers,
            "num_heads": self.config.num_heads,
            "head_dim": self.config.head_dim,
            "group_size": self.config.group_size,
            "group_intervention_chunk_size": self.config.group_intervention_chunk_size,
            "random_group_seed": self.config.random_group_seed,
            "parameter_count": self.parameter_count,
            "cost_model": {
                "base_evaluations_per_query_row": "1 + active_group_count",
                "exact_evaluation_budget": "query_rows x (1 + active_group_count)",
            },
            "per_head_constructor": self.constructors[0].metadata(),
            "config": asdict(self.config),
        }

    def _constructor(self, layer_index: int, head_index: int) -> GroupedCounterfactualQCEASC:
        if not 0 <= layer_index < self.config.num_layers:
            raise ValueError("layer_index is outside configured layer range")
        if not 0 <= head_index < self.config.num_heads:
            raise ValueError("head_index is outside configured head range")
        return self.constructors[layer_index * self.config.num_heads + head_index]

    def fit_transport(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

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
        del value, scores, evidence_view, detach_random, routing_mode
        if query.ndim != 4 or key.shape != query.shape:
            raise ValueError("query and key must have shape (batch, heads, tokens, head_dim)")
        batch, heads, tokens, head_dim = query.shape
        if (heads, head_dim) != (self.config.num_heads, self.config.head_dim):
            raise ValueError("query shape does not match grouped score dimensions")
        if attention_mask.shape != (batch, tokens):
            raise ValueError("attention_mask must have shape (batch, tokens)")
        if subject_mask.shape != (batch, tokens) or object_mask.shape != (batch, tokens):
            raise ValueError("subject/object masks must have shape (batch, tokens)")
        if tokens > self.config.max_context_size:
            raise ValueError("context length exceeds the declared grouped preflight ceiling")
        if query.device != key.device:
            raise ValueError("query and key must be on the same device")
        valid = attention_mask.to(device=query.device, dtype=torch.bool)
        entity = subject_mask.to(device=query.device, dtype=torch.bool) | object_mask.to(device=query.device, dtype=torch.bool)
        output_heads: list[torch.Tensor] = []
        chunk_size = min(tokens, int(self.config.query_chunk_size))
        for head_index in range(heads):
            head_query = query[:, head_index]
            head_key = key[:, head_index]
            chunks: list[torch.Tensor] = []
            constructor = self._constructor(layer_index, head_index)
            random_manifest = None
            if self.control_mode == "random_grouped_counterfactual":
                random_manifest = build_seeded_random_group_manifest(
                    valid,
                    entity,
                    group_size=self.config.group_size,
                    seed=(
                        int(self.config.random_group_seed)
                        + int(random_seed)
                        + 1009 * int(head_index)
                    ),
                )
            for start in range(0, tokens, chunk_size):
                stop = min(tokens, start + chunk_size)
                width = stop - start
                query_rows = head_query[:, start:stop, :].reshape(batch * width, head_dim)
                key_rows = head_key.repeat_interleave(width, dim=0)
                valid_rows = valid.repeat_interleave(width, dim=0)
                entity_rows = entity.repeat_interleave(width, dim=0)
                if random_manifest is not None:
                    manifest_rows = random_manifest.repeat_interleave(width, dim=0)
                else:
                    manifest_rows = build_fixed_group_manifest(
                        valid_rows, entity_rows, group_size=self.config.group_size
                    )
                result = constructor.evaluate(
                    query_rows, key_rows, valid_rows, entity_rows, manifest_rows
                )
                if self.capture_callback is not None:
                    self.capture_callback(
                        {
                            "layer_index": int(layer_index),
                            "head_index": int(head_index),
                            "query_start": int(start),
                            "query_stop": int(stop),
                            "auxiliary_state": result.auxiliary_state.reshape(batch, width, -1),
                            "coefficients": result.coefficients.reshape(batch, width, -1),
                            "group_manifest": result.group_ids.reshape(batch, width, tokens),
                            "projected_support": result.projected_support.reshape(batch, width, -1),
                            "group_masked_supports": result.group_masked_supports.reshape(batch, width, tokens, -1),
                            "group_influence_vectors": result.group_influence_vectors.reshape(batch, width, tokens, -1),
                            "group_scores": result.group_scores.reshape(batch, width, tokens),
                            "member_scores": result.member_scores.reshape(batch, width, tokens),
                            "residual": result.residual.reshape(batch, width, tokens),
                        }
                    )
                chunks.append(result.residual.reshape(batch, width, tokens))
            output_heads.append(torch.cat(chunks, dim=1))
        residual = torch.stack(output_heads, dim=1)
        return residual * valid[:, None, :, None].to(dtype=residual.dtype)


def build_qceasc_grouped_counterfactual(
    mode: str,
    config: QCEASCGroupedCounterfactualConfig,
    *,
    action_span: torch.Tensor | None = None,
) -> GroupedCounterfactualQCEASC:
    if mode not in QCEASC_GROUPED_COUNTERFACTUAL_CONTROL_MODES:
        raise ValueError(f"mode must be one of {QCEASC_GROUPED_COUNTERFACTUAL_CONTROL_MODES}")
    return GroupedCounterfactualQCEASC(config, mode, action_span=action_span)


def build_qceasc_grouped_counterfactual_score_kernel(
    mode: str,
    config: QCEASCGroupedCounterfactualScoreKernelConfig,
) -> QCEASCGroupedCounterfactualScoreKernel:
    if mode not in QCEASC_GROUPED_COUNTERFACTUAL_CONTROL_MODES:
        raise ValueError(f"mode must be one of {QCEASC_GROUPED_COUNTERFACTUAL_CONTROL_MODES}")
    return QCEASCGroupedCounterfactualScoreKernel(config, mode)
