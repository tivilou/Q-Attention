"""Direct query-key coherent transport score plugins.

The quantum branch prepares one small state for each query-key pair, applies a
trainable controlled phase and local post-rotations for each configured circuit
depth, and reads an X tensor Z transport observable.  The matched classical
branch keeps the same projections, phase/post-rotation parameters, gain, masks,
and pair budget, but evaluates a separable interaction bank per depth.

Both branches implement the existing label-free pre-softmax score-residual
contract: only valid non-entity context keys receive action and each query row
is centered to zero sum.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn as nn


QK_COHERENT_TRANSPORT_KERNEL_TYPES = ("quantum", "classical")


@dataclass(frozen=True)
class QueryKeyCoherentTransportConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    register_qubits: int = 1
    depth: int = 1
    angle_scale: float = 1.0
    max_phase: float = math.pi
    max_post_rotation: float = math.pi / 2.0
    max_transport: float = 0.25
    initial_phase: float = 0.6
    initial_post_rotation: float = 0.35
    initial_transport: float = 0.05
    pair_chunk_size: int = 4096
    seed: int = 2718
    eps: float = 1e-8

    def __post_init__(self) -> None:
        dimensions = (
            self.num_layers,
            self.num_heads,
            self.head_dim,
            self.register_qubits,
            self.depth,
            self.pair_chunk_size,
        )
        if min(dimensions) <= 0:
            raise ValueError("model, circuit, and chunk dimensions must be positive")
        if self.angle_scale <= 0.0:
            raise ValueError("angle_scale must be positive")
        if self.max_phase <= 0.0 or self.max_post_rotation <= 0.0:
            raise ValueError("phase and post-rotation bounds must be positive")
        if self.max_transport <= 0.0:
            raise ValueError("max_transport must be positive")
        if abs(self.initial_phase) >= self.max_phase:
            raise ValueError("initial_phase must lie inside the phase bound")
        if abs(self.initial_post_rotation) >= self.max_post_rotation:
            raise ValueError("initial_post_rotation must lie inside the rotation bound")
        if abs(self.initial_transport) >= self.max_transport:
            raise ValueError("initial_transport must lie inside the transport bound")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")


def _seeded_projection(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(input_dim, output_dim, generator=generator)
    return projection / math.sqrt(float(input_dim))


def _ry_matrix(angle: torch.Tensor) -> torch.Tensor:
    half = angle / 2.0
    c = torch.cos(half)
    s = torch.sin(half)
    return torch.stack(
        (torch.stack((c, -s), dim=-1), torch.stack((s, c), dim=-1)), dim=-2
    )


class QueryKeyCoherentTransportKernel(nn.Module):
    """A direct query-key transport observable and its matched control."""

    kernel_type = "quantum"
    entangled = True

    def __init__(self, config: QueryKeyCoherentTransportConfig) -> None:
        super().__init__()
        self.config = config
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.pair_chunk_size = config.pair_chunk_size
        self.register_qubits = config.register_qubits
        projections = torch.stack(
            [
                _seeded_projection(
                    config.head_dim,
                    config.register_qubits,
                    config.seed + 97 * role,
                )
                for role in range(2)
            ]
        )
        self.register_buffer("input_projections", projections)
        angle_shape = (
            config.num_layers,
            config.num_heads,
            2,
            config.register_qubits,
        )
        gate_shape = (
            config.num_layers,
            config.num_heads,
            config.depth,
            config.register_qubits,
        )
        post_shape = (
            config.num_layers,
            config.num_heads,
            config.depth,
            2,
            config.register_qubits,
        )
        generator = torch.Generator(device="cpu").manual_seed(config.seed + 401)
        self.angle_scales = nn.Parameter(torch.ones(angle_shape))
        self.angle_biases = nn.Parameter(
            torch.empty(angle_shape).uniform_(-math.pi / 4.0, math.pi / 4.0, generator=generator)
        )
        raw_phase = math.atanh(config.initial_phase / config.max_phase)
        raw_post = math.atanh(config.initial_post_rotation / config.max_post_rotation)
        raw_transport = math.atanh(config.initial_transport / config.max_transport)
        self.raw_phase = nn.Parameter(
            torch.full(
                gate_shape,
                raw_phase,
            )
        )
        self.raw_post_rotation = nn.Parameter(
            torch.full(
                post_shape,
                raw_post,
            )
        )
        self.raw_transport = nn.Parameter(
            torch.full((config.num_layers, config.num_heads), raw_transport)
        )
        observable = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, -1.0],
             [1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        self.register_buffer("transport_observable", observable, persistent=False)
        self._last_raw_score: torch.Tensor | None = None
        self._last_residual: torch.Tensor | None = None

    @property
    def model_dimensions(self) -> tuple[int, int, int]:
        return self.num_layers, self.num_heads, self.head_dim

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def last_raw_score(self) -> torch.Tensor | None:
        return self._last_raw_score

    @property
    def last_residual(self) -> torch.Tensor | None:
        return self._last_residual

    def _validate_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        scores: torch.Tensor | None,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
        query_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if query.ndim != 4 or key.ndim != 4:
            raise ValueError("query and key must be rank four")
        if query.shape[0] != key.shape[0] or query.shape[1] != key.shape[1]:
            raise ValueError("query and key batch/head dimensions must match")
        if query.shape[1] != self.num_heads:
            raise ValueError(
                f"query/key head count {query.shape[1]} does not match configured num_heads {self.num_heads}"
            )
        if query.shape[-1] != self.head_dim or key.shape[-1] != self.head_dim:
            raise ValueError("query and key dimensions do not match the configuration")
        batch, _heads, query_tokens, _dim = query.shape
        key_tokens = key.shape[2]
        if query_tokens <= 0 or key_tokens <= 0:
            raise ValueError("query and key must contain at least one token")
        pair_shape = (batch, query_tokens, key_tokens)

        def normalize_mask(name: str, mask: torch.Tensor) -> torch.Tensor:
            if mask.ndim == 2 and mask.shape == (batch, key_tokens):
                return mask[:, None, :].expand(pair_shape)
            if mask.ndim == 3 and mask.shape == pair_shape:
                return mask
            raise ValueError(
                f"{name} must have shape {(batch, key_tokens)} or {pair_shape}; got {tuple(mask.shape)}"
            )

        normalized_attention = normalize_mask("attention_mask", attention_mask)
        normalized_subject = normalize_mask("subject_mask", subject_mask)
        normalized_object = normalize_mask("object_mask", object_mask)
        if query_mask is None:
            if attention_mask.ndim == 3:
                normalized_query = attention_mask.to(dtype=torch.bool).any(dim=-1)
            elif query_tokens == key_tokens:
                normalized_query = attention_mask.to(dtype=torch.bool)
            else:
                normalized_query = torch.ones(
                    (batch, query_tokens), dtype=torch.bool, device=query.device
                )
        elif query_mask.ndim == 2 and query_mask.shape == (batch, query_tokens):
            normalized_query = query_mask
        else:
            raise ValueError(
                f"query_mask must have shape {(batch, query_tokens)}; got {tuple(query_mask.shape)}"
            )
        if scores is not None and scores.shape != (batch, self.num_heads, query_tokens, key_tokens):
            raise ValueError("scores shape is incompatible with query/key")
        if not torch.isfinite(query).all() or not torch.isfinite(key).all():
            raise ValueError("query and key must be finite")
        return (
            normalized_attention.to(device=query.device, dtype=torch.bool),
            normalized_subject.to(device=query.device, dtype=torch.bool),
            normalized_object.to(device=query.device, dtype=torch.bool),
            normalized_query.to(device=query.device, dtype=torch.bool),
        )

    def _angles(
        self,
        value: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
        role: int,
    ) -> torch.Tensor:
        projection = self.input_projections[role].to(device=value.device, dtype=value.dtype)
        projected = torch.matmul(value, projection)
        angle_scales = self.angle_scales[layer_index, head_index, role].to(
            device=value.device, dtype=value.dtype
        )
        angle_biases = self.angle_biases[layer_index, head_index, role].to(
            device=value.device, dtype=value.dtype
        )
        return self.config.angle_scale * (
            projected * angle_scales + angle_biases
        )

    def _single_qubit_quantum_score(
        self,
        query_angles: torch.Tensor,
        key_angles: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
        qubit: int,
    ) -> torch.Tensor:
        q_state = torch.stack(
            (torch.cos(query_angles / 2.0), torch.sin(query_angles / 2.0)), dim=-1
        )
        k_state = torch.stack(
            (torch.cos(key_angles / 2.0), torch.sin(key_angles / 2.0)), dim=-1
        )
        state = torch.einsum("na,nb->nab", q_state, k_state)
        complex_dtype = torch.complex128 if query_angles.dtype == torch.float64 else torch.complex64
        state = state.reshape(-1, 4).to(complex_dtype).reshape(-1, 2, 2)
        for depth_index in range(self.config.depth):
            phase = self.config.max_phase * torch.tanh(
                self.raw_phase[layer_index, head_index, depth_index, qubit].to(
                    device=query_angles.device, dtype=query_angles.dtype
                )
            )
            phase_vector = torch.stack(
                (
                    torch.ones_like(query_angles),
                    torch.ones_like(query_angles),
                    torch.ones_like(query_angles),
                    torch.exp(1j * phase).expand_as(query_angles),
                ),
                dim=-1,
            ).to(dtype=complex_dtype)
            state = (state.reshape(-1, 4) * phase_vector).reshape(-1, 2, 2)
            post_q = self.config.max_post_rotation * torch.tanh(
                self.raw_post_rotation[layer_index, head_index, depth_index, 0, qubit].to(
                    device=query_angles.device, dtype=query_angles.dtype
                )
            )
            post_k = self.config.max_post_rotation * torch.tanh(
                self.raw_post_rotation[layer_index, head_index, depth_index, 1, qubit].to(
                    device=query_angles.device, dtype=query_angles.dtype
                )
            )
            q_matrix = _ry_matrix(post_q).to(dtype=complex_dtype).expand(state.shape[0], -1, -1)
            k_matrix = _ry_matrix(post_k).to(dtype=complex_dtype).expand(state.shape[0], -1, -1)
            state = torch.einsum("nij,njk->nik", q_matrix, state)
            state = torch.einsum("nac,nbc->nab", state, k_matrix)
        observable = self.transport_observable.to(device=state.device, dtype=state.dtype)
        transformed = torch.matmul(state.reshape(-1, 4), observable.transpose(0, 1))
        return (state.reshape(-1, 4).conj() * transformed).sum(dim=-1).real

    def _single_qubit_classical_score(
        self,
        query_angles: torch.Tensor,
        key_angles: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
        qubit: int,
    ) -> torch.Tensor:
        depth_scores = []
        for depth_index in range(self.config.depth):
            post_q = self.config.max_post_rotation * torch.tanh(
                self.raw_post_rotation[layer_index, head_index, depth_index, 0, qubit].to(
                    device=query_angles.device, dtype=query_angles.dtype
                )
            )
            post_k = self.config.max_post_rotation * torch.tanh(
                self.raw_post_rotation[layer_index, head_index, depth_index, 1, qubit].to(
                    device=query_angles.device, dtype=query_angles.dtype
                )
            )
            q_total = query_angles + post_q
            k_total = key_angles + post_k
            local_xz = torch.sin(q_total) * torch.cos(k_total)
            local_zz = torch.cos(q_total) * torch.cos(k_total)
            phase = self.config.max_phase * torch.tanh(
                self.raw_phase[layer_index, head_index, depth_index, qubit].to(
                    device=query_angles.device, dtype=query_angles.dtype
                )
            )
            depth_scores.append(torch.cos(phase) * local_xz + torch.sin(phase) * local_zz)
        return torch.stack(depth_scores, dim=-1).mean(dim=-1)

    def _pair_score(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
    ) -> torch.Tensor:
        query_angles = self._angles(query, layer_index=layer_index, head_index=head_index, role=0)
        key_angles = self._angles(key, layer_index=layer_index, head_index=head_index, role=1)
        scores = []
        for qubit in range(self.register_qubits):
            if self.entangled:
                score = self._single_qubit_quantum_score(
                    query_angles[:, qubit], key_angles[:, qubit],
                    layer_index=layer_index, head_index=head_index, qubit=qubit,
                )
            else:
                score = self._single_qubit_classical_score(
                    query_angles[:, qubit], key_angles[:, qubit],
                    layer_index=layer_index, head_index=head_index, qubit=qubit,
                )
            scores.append(score)
        return torch.stack(scores, dim=-1).mean(dim=-1)

    @staticmethod
    def _center_context(
        score: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
        max_transport: float,
        eps: float,
    ) -> torch.Tensor:
        valid = attention_mask.to(device=score.device, dtype=torch.bool)
        entities = subject_mask.to(device=score.device, dtype=torch.bool) | object_mask.to(device=score.device, dtype=torch.bool)
        context = valid & ~entities
        weights = context.to(dtype=score.dtype)
        count = weights.sum(dim=-1, keepdim=True)
        centered = score - (score * weights).sum(dim=-1, keepdim=True) / count.clamp_min(eps)
        centered = centered * context.to(dtype=score.dtype)
        magnitude = centered.abs().amax(dim=(-1, -2), keepdim=True)
        scale = max_transport / magnitude.clamp_min(eps)
        return centered * torch.minimum(scale, torch.ones_like(scale))

    def metadata(self) -> dict[str, Any]:
        return {
            "id": f"qk_coherent_transport_{self.kernel_type}",
            "version": "0.2.0",
            "type": "query_key_coherent_transport",
            "insertion_point": "pre_softmax_attention_scores",
            "hypothesis": "direct query-key cross-register transport can expose a stronger continuous pair relation than relation-coordinate curvature",
            "input_schema": "query,key,scores?,attention_mask,subject_mask,object_mask,query_mask?; masks are [batch,key] or [batch,query,key]; labels and relation IDs prohibited",
            "output_schema": "finite bounded centered context-only score residual [batch,heads,query,key]",
            "requires": ["query/key projections", "mask-aware score hook"],
            "conflicts": ["q_rpec_quantum", "q_rpec_local_control", "q_triad", "qness", "q_wap", "qcdd", "qccw", "stacked_score_interventions"],
            "deterministic": True,
            "resource_estimate": {
                "data_qubits": 2 * self.register_qubits,
                "ancilla_qubits": 0,
                "depth": self.config.depth,
                "pair_budget": "one transport evaluation per query-key pair",
                "trainable_parameter_count": self.parameter_count,
            },
            "failure_signatures": [
                "quantum score collapses to the separable control",
                "score or residual variance is near numerical noise",
                "relation or permutation discriminator fails",
                "masked/entity key receives action",
                "context residual is not zero-sum",
                "nonfinite score or gradient",
            ],
            "observability_contract": {
                "trace_schema": "qk-coherent-transport-trace.v1",
                "emitted_fields": ["pair_score_mean", "pair_score_std", "residual_rms", "gradient_norm", "quantum_control_gap", "target_key_top1", "permutation_equivariance_error"],
                "checks": ["finite", "nonzero_score_variance", "context_zero_sum", "entity_mask_zero", "query_key_permutation_equivariance", "deterministic_replay", "parameter_match"],
                "deterministic_replay": True,
            },
            "config": asdict(self.config),
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
        query_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        del value
        attention_mask, subject_mask, object_mask, query_mask = self._validate_inputs(
            query, key, scores, attention_mask, subject_mask, object_mask, query_mask
        )
        if not 0 <= layer_index < self.num_layers:
            raise ValueError("layer_index is outside configured layers")
        batch, heads, query_tokens, _dim = query.shape
        key_tokens = key.shape[2]
        pair_count = query_tokens * key_tokens
        total_pairs = batch * pair_count
        residuals: list[torch.Tensor] = []
        raw_scores: list[torch.Tensor] = []
        for head_index in range(heads):
            q = query[:, head_index]
            k = key[:, head_index]
            chunks: list[torch.Tensor] = []
            for start in range(0, total_pairs, self.pair_chunk_size):
                stop = min(start + self.pair_chunk_size, total_pairs)
                flat = torch.arange(start, stop, device=query.device)
                batch_index = torch.div(flat, pair_count, rounding_mode="floor")
                within = flat.remainder(pair_count)
                query_index = torch.div(within, key_tokens, rounding_mode="floor")
                key_index = within.remainder(key_tokens)
                chunks.append(
                    self._pair_score(
                        q[batch_index, query_index],
                        k[batch_index, key_index],
                        layer_index=layer_index,
                        head_index=head_index,
                    )
                )
            score = torch.cat(chunks, dim=0).reshape(batch, query_tokens, key_tokens)
            raw_scores.append(score)
            gain = self.config.max_transport * torch.tanh(
                self.raw_transport[layer_index, head_index].to(query.device)
            )
            residuals.append(
                self._center_context(
                    score * gain,
                    attention_mask=attention_mask,
                    subject_mask=subject_mask,
                    object_mask=object_mask,
                    max_transport=self.config.max_transport,
                    eps=self.config.eps,
                )
            )
        raw = torch.stack(raw_scores, dim=1)
        residual = torch.stack(residuals, dim=1)
        residual = residual * query_mask[:, None, :, None].to(dtype=residual.dtype)
        self._last_raw_score = raw.detach()
        self._last_residual = residual.detach()
        return residual


class QuantumQueryKeyCoherentTransportKernel(QueryKeyCoherentTransportKernel):
    kernel_type = "quantum"
    entangled = True


class ClassicalQueryKeyCoherentTransportKernel(QueryKeyCoherentTransportKernel):
    kernel_type = "classical"
    entangled = False


def build_query_key_coherent_transport_kernel(
    kernel_type: str,
    config: QueryKeyCoherentTransportConfig,
) -> QueryKeyCoherentTransportKernel:
    classes = {
        "quantum": QuantumQueryKeyCoherentTransportKernel,
        "classical": ClassicalQueryKeyCoherentTransportKernel,
    }
    try:
        cls = classes[kernel_type]
    except KeyError as error:
        raise ValueError(
            f"kernel_type must be one of {QK_COHERENT_TRANSPORT_KERNEL_TYPES}"
        ) from error
    return cls(config)
