"""Production attention-score wrapper for the Q-CEASC mechanism.

The stage-A constructor operates on one query and a context key bank.  This
wrapper applies it independently to each attention head and query token,
then returns a zero-sum residual aligned with the attention score matrix.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from .q_ceasc import (
    QCEASCConfig,
    QCEASC_CONTROL_MODES,
    ContextEntangledAuxiliarySupportConstructor,
)


QCEASC_SCORE_CONTROL_MODES = ("q_ceasc", "classical_span", "quantum_product")


@dataclass(frozen=True)
class QCEASCScoreKernelConfig:
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
    seed: int = 13091
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.num_layers <= 0 or self.num_heads <= 0 or self.head_dim <= 0:
            raise ValueError("num_layers, num_heads, and head_dim must be positive")
        if self.query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        if self.action_rank >= self.head_dim:
            raise ValueError("action_rank must leave a non-empty score complement")


class QCEASCScoreKernel(nn.Module):
    """Headwise Q-CEASC score residual with bounded token materialization."""

    kernel_type = "q_ceasc_score_kernel"

    def __init__(
        self,
        config: QCEASCScoreKernelConfig,
        control_mode: str = "q_ceasc",
    ) -> None:
        super().__init__()
        if control_mode not in QCEASC_SCORE_CONTROL_MODES:
            raise ValueError(
                f"control_mode must be one of {QCEASC_SCORE_CONTROL_MODES}"
            )
        base_mode = "classical_span" if control_mode == "classical_span" else control_mode
        if base_mode not in QCEASC_CONTROL_MODES:
            raise ValueError(f"unsupported Q-CEASC base mode: {base_mode}")
        self.config = config
        self.control_mode = control_mode
        constructors: list[ContextEntangledAuxiliarySupportConstructor] = []
        for layer_index in range(config.num_layers):
            for head_index in range(config.num_heads):
                constructors.append(
                    ContextEntangledAuxiliarySupportConstructor(
                        QCEASCConfig(
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
                        ),
                        base_mode,
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
        per_constructor = self.constructors[0].metadata()
        return {
            "type": self.kernel_type,
            "version": "1.0.0",
            "control_mode": self.control_mode,
            "mechanism": (
                "context-conditioned auxiliary support construction followed by "
                "frozen action-span orthogonal projection and key-aligned score residual"
            ),
            "input_schema": "query/key (B,H,T,D), attention and entity masks (B,T)",
            "output_schema": "zero-sum masked score residual (B,H,T,T)",
            "num_layers": self.config.num_layers,
            "num_heads": self.config.num_heads,
            "head_dim": self.config.head_dim,
            "query_chunk_size": self.config.query_chunk_size,
            "parameter_count": self.parameter_count,
            "per_head_constructor": per_constructor,
            "quantum_resource_note": (
                "Exact statevector simulation establishes functional behavior only; "
                "it does not establish hardware speedup or quantum advantage."
            ),
            "config": asdict(self.config),
        }

    def _constructor(self, layer_index: int, head_index: int) -> ContextEntangledAuxiliarySupportConstructor:
        if not 0 <= layer_index < self.config.num_layers:
            raise ValueError("layer_index is outside configured layer range")
        if not 0 <= head_index < self.config.num_heads:
            raise ValueError("head_index is outside configured head range")
        return self.constructors[layer_index * self.config.num_heads + head_index]

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
            raise ValueError("query shape does not match Q-CEASC score-kernel dimensions")
        if attention_mask.shape != (batch, tokens):
            raise ValueError("attention_mask must have shape (batch, tokens)")
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
                chunks.append(result.residual.reshape(batch, width, tokens))
            output_heads.append(torch.cat(chunks, dim=1))
        residual = torch.stack(output_heads, dim=1)
        query_valid = valid[:, None, :, None].to(dtype=residual.dtype)
        return residual * query_valid


def build_qceasc_score_kernel(
    mode: str,
    config: QCEASCScoreKernelConfig,
) -> QCEASCScoreKernel:
    if mode not in QCEASC_SCORE_CONTROL_MODES:
        raise ValueError(f"mode must be one of {QCEASC_SCORE_CONTROL_MODES}")
    return QCEASCScoreKernel(config, mode)
