"""Coherent signed-path quantum walk for bounded attention-score transport."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn as nn


COHERENT_PATH_KERNEL_TYPES = (
    "quantum_signed",
    "quantum_unsigned",
    "classical_diffusion",
)


@dataclass(frozen=True)
class CoherentAttentionPathConfig:
    num_layers: int
    num_heads: int
    max_transport: float = 2.0
    initial_transport: float = 1.0
    walk_time: float = math.pi / 4.0
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.num_layers <= 0 or self.num_heads <= 0:
            raise ValueError("model dimensions must be positive")
        if self.max_transport <= 0.0:
            raise ValueError("max_transport must be positive")
        if not 0.0 < self.initial_transport < self.max_transport:
            raise ValueError("initial_transport must lie inside (0, max_transport)")
        if self.walk_time <= 0.0:
            raise ValueError("walk_time must be positive")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")


class CoherentAttentionPathKernel(nn.Module):
    """Map a square signed score graph to a bounded context residual."""

    kernel_type = "base"

    def __init__(self, config: CoherentAttentionPathConfig) -> None:
        super().__init__()
        self.config = config
        ratio = config.initial_transport / config.max_transport
        initial_raw = math.log(ratio) - math.log1p(-ratio)
        self.raw_transport = nn.Parameter(
            torch.full((config.num_layers, config.num_heads), initial_raw)
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "id": f"q_wap_{self.kernel_type}",
            "version": "0.1.0",
            "type": self.kernel_type,
            "insertion_point": "pre_softmax_attention_scores",
            "hypothesis": "signed path interference exposes context evidence beyond nonnegative diffusion",
            "input_schema": "square scores plus attention/entity masks; labels and targets prohibited",
            "output_schema": "finite context-only zero-sum bounded score residual",
            "requires": [],
            "conflicts": [
                kind for kind in COHERENT_PATH_KERNEL_TYPES if kind != self.kernel_type
            ],
            "deterministic": True,
            "resource_estimate": (
                "one exact matrix exponential per batch/head"
                if self.kernel_type.startswith("quantum")
                else "two batched transition-matrix multiplications"
            ),
            "failure_signatures": [
                "no signed-phase separation",
                "matched classical diffusion parity",
                "residual invariant failure",
            ],
            "config": asdict(self.config),
            "quantum_resource_note": (
                "Exact matrix-exponential simulation establishes functional behavior only; "
                "an ideal implementation requires signed Hamiltonian simulation and position readout."
            ),
        }

    def transport_fractions(self, layer_index: int) -> torch.Tensor:
        if layer_index < 0 or layer_index >= self.config.num_layers:
            raise ValueError("layer_index is outside the configured model")
        return self.config.max_transport * torch.sigmoid(
            self.raw_transport[layer_index]
        )

    def _validate(
        self,
        scores: torch.Tensor,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> None:
        if scores.ndim != 4 or scores.shape[-2] != scores.shape[-1]:
            raise ValueError("Q-WAP requires square scores shaped [batch, heads, nodes, nodes]")
        if scores.shape[1] != self.config.num_heads:
            raise ValueError("score head count does not match the Q-WAP config")
        expected = (scores.shape[0], scores.shape[-1])
        for name, mask in (
            ("attention_mask", attention_mask),
            ("subject_mask", subject_mask),
            ("object_mask", object_mask),
        ):
            if mask.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
        if not torch.isfinite(scores).all():
            raise ValueError("scores must be finite before masking")

    def _hermitian_graph(
        self,
        scores: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        node_mask = attention_mask.to(dtype=torch.bool)
        pair_mask = node_mask[:, None, :, None] & node_mask[:, None, None, :]
        graph = 0.5 * (scores + scores.transpose(-1, -2))
        graph = graph * pair_mask.to(dtype=graph.dtype)
        diagonal = torch.diagonal(graph, dim1=-2, dim2=-1)
        graph = graph - torch.diag_embed(diagonal)
        return graph

    def _quantum_probabilities(self, graph: torch.Tensor) -> torch.Tensor:
        complex_dtype = (
            torch.complex128 if graph.dtype == torch.float64 else torch.complex64
        )
        unitary = torch.matrix_exp(
            (-1j * self.config.walk_time) * graph.to(dtype=complex_dtype)
        )
        return unitary.abs().square().to(dtype=graph.dtype)

    def _classical_probabilities(self, graph: torch.Tensor) -> torch.Tensor:
        adjacency = graph.abs()
        degree = adjacency.sum(dim=-1, keepdim=True)
        transition = adjacency / degree.clamp_min(self.config.eps)
        isolated = degree.squeeze(-1) <= self.config.eps
        if isolated.any():
            identity = torch.eye(
                graph.shape[-1], device=graph.device, dtype=graph.dtype
            ).view(1, 1, graph.shape[-1], graph.shape[-1])
            transition = torch.where(isolated[..., None], identity, transition)
        return transition @ transition

    def _path_probabilities(
        self,
        graph: torch.Tensor,
    ) -> torch.Tensor:
        if self.kernel_type == "quantum_signed":
            return self._quantum_probabilities(graph)
        if self.kernel_type == "quantum_unsigned":
            return self._quantum_probabilities(graph.abs())
        if self.kernel_type == "classical_diffusion":
            return self._classical_probabilities(graph)
        raise RuntimeError(f"unsupported Q-WAP kernel type: {self.kernel_type}")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor | None = None,
        *,
        scores: torch.Tensor,
        layer_index: int,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
        **_kwargs: Any,
    ) -> torch.Tensor:
        del query, key, value
        self._validate(scores, attention_mask, subject_mask, object_mask)
        graph = self._hermitian_graph(scores, attention_mask)
        path = self._path_probabilities(graph)

        key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
        masked_scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        base_attention = torch.softmax(masked_scores, dim=-1)
        context = attention_mask & ~(subject_mask | object_mask)
        context_mask = context[:, None, None, :].to(dtype=scores.dtype)
        query_mask = attention_mask[:, None, :, None].to(dtype=scores.dtype)

        base_context = base_attention * context_mask
        base_distribution = base_context / base_context.sum(
            dim=-1, keepdim=True
        ).clamp_min(self.config.eps)
        path_context = path * context_mask
        path_total = path_context.sum(dim=-1, keepdim=True)
        path_distribution = path_context / path_total.clamp_min(self.config.eps)
        path_distribution = torch.where(
            path_total > self.config.eps,
            path_distribution,
            base_distribution,
        )

        fraction = self.transport_fractions(layer_index).view(1, -1, 1, 1)
        residual = fraction * (path_distribution - base_distribution)
        residual = residual * context_mask * query_mask
        return residual.to(dtype=scores.dtype)


class QuantumSignedAttentionPathKernel(CoherentAttentionPathKernel):
    kernel_type = "quantum_signed"


class QuantumUnsignedAttentionPathKernel(CoherentAttentionPathKernel):
    kernel_type = "quantum_unsigned"


class ClassicalAttentionPathDiffusionKernel(CoherentAttentionPathKernel):
    kernel_type = "classical_diffusion"


def build_coherent_attention_path_kernel(
    kernel_type: str,
    config: CoherentAttentionPathConfig,
) -> CoherentAttentionPathKernel:
    classes = {
        "quantum_signed": QuantumSignedAttentionPathKernel,
        "quantum_unsigned": QuantumUnsignedAttentionPathKernel,
        "classical_diffusion": ClassicalAttentionPathDiffusionKernel,
    }
    try:
        cls = classes[kernel_type]
    except KeyError as error:
        raise ValueError(
            f"kernel_type must be one of {COHERENT_PATH_KERNEL_TYPES}"
        ) from error
    return cls(config)

