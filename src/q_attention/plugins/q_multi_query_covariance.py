"""Shared-key two-role query covariance attention plugins.

MQIC first pools subject and object query rows into two role registers.  Each
role is coupled to the same context-key register, and the quantum branch
reads the connected covariance

    E[Z_s Z_o | k] - E[Z_s | k] E[Z_o | k].

The classical branch uses the same projections, trainable tensors, gain
bound, masks, and pair budget but evaluates an explicit tensor-product
covariance control without a joint statevector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn as nn


MQIC_CONTROL_MODES = ("quantum", "classical")


@dataclass(frozen=True)
class MultiQueryCovarianceConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    num_qubits: int = 2
    depth: int = 1
    angle_scale: float = 1.0
    max_phase: float = math.pi
    max_post_rotation: float = math.pi / 2.0
    max_covariance: float = 0.25
    initial_phase: float = 0.6
    initial_post_rotation: float = 0.35
    initial_gain: float = 0.05
    pair_chunk_size: int | None = 4096
    seed: int = 7919
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if min(
            self.num_layers,
            self.num_heads,
            self.head_dim,
            self.num_qubits,
            self.depth,
        ) <= 0:
            raise ValueError("model and circuit dimensions must be positive")
        if self.angle_scale <= 0.0:
            raise ValueError("angle_scale must be positive")
        if self.max_phase <= 0.0 or self.max_post_rotation <= 0.0:
            raise ValueError("phase and rotation bounds must be positive")
        if self.max_covariance <= 0.0:
            raise ValueError("max_covariance must be positive")
        if abs(self.initial_phase) >= self.max_phase:
            raise ValueError("initial_phase must lie inside the phase bound")
        if abs(self.initial_post_rotation) >= self.max_post_rotation:
            raise ValueError("initial_post_rotation must lie inside the rotation bound")
        if abs(self.initial_gain) >= self.max_covariance:
            raise ValueError("initial_gain must lie inside the covariance bound")
        if self.pair_chunk_size is not None and self.pair_chunk_size <= 0:
            raise ValueError("pair_chunk_size must be positive or None")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")


def _seeded_projection(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(input_dim, output_dim, generator=generator)
    return projection / math.sqrt(float(input_dim))


def _raw_bounded(value: float, bound: float) -> float:
    return math.atanh(value / bound)


def _ry_matrix(angle: torch.Tensor) -> torch.Tensor:
    half = angle / 2.0
    c = torch.cos(half)
    s = torch.sin(half)
    return torch.stack(
        (torch.stack((c, -s), dim=-1), torch.stack((s, c), dim=-1)), dim=-2
    )


class MultiQueryCovarianceKernel(nn.Module):
    """Common parameterization and score-hook contract for MQIC branches."""

    kernel_type = "base"
    entangled = False

    def __init__(self, config: MultiQueryCovarianceConfig) -> None:
        super().__init__()
        self.config = config
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.register_qubits = config.num_qubits

        projections = torch.stack(
            [
                torch.stack(
                    [
                        _seeded_projection(
                            config.head_dim,
                            config.num_qubits,
                            config.seed + 97 * role + 13 * head,
                        )
                        for head in range(config.num_heads)
                    ]
                )
                for role in range(3)
            ]
        )
        self.register_buffer("input_projections", projections)

        angle_shape = (
            config.num_layers,
            config.num_heads,
            config.depth,
            3,
            config.num_qubits,
        )
        generator = torch.Generator(device="cpu").manual_seed(config.seed + 401)
        self.angle_scales = nn.Parameter(torch.ones(angle_shape))
        self.angle_biases = nn.Parameter(
            torch.empty(angle_shape).uniform_(-math.pi / 4.0, math.pi / 4.0, generator=generator)
        )
        self.raw_couplings = nn.Parameter(
            torch.full(
                (config.num_layers, config.num_heads, config.depth, config.num_qubits, 3),
                _raw_bounded(config.initial_phase, config.max_phase),
            )
        )
        self.raw_post_rotation = nn.Parameter(
            torch.full(
                (config.num_layers, config.num_heads, config.depth, 3, config.num_qubits),
                _raw_bounded(config.initial_post_rotation, config.max_post_rotation),
            )
        )
        self.raw_gain = nn.Parameter(
            torch.full(
                (config.num_layers, config.num_heads),
                _raw_bounded(config.initial_gain, config.max_covariance),
            )
        )

        self._last_raw_score: torch.Tensor | None = None
        self._last_connected_covariance: torch.Tensor | None = None
        self._last_independent_product: torch.Tensor | None = None
        self._last_residual: torch.Tensor | None = None
        self._last_pair_count: int = 0

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
    def last_connected_covariance(self) -> torch.Tensor | None:
        return self._last_connected_covariance

    @property
    def last_independent_product(self) -> torch.Tensor | None:
        return self._last_independent_product

    @property
    def last_residual(self) -> torch.Tensor | None:
        return self._last_residual

    @property
    def last_pair_count(self) -> int:
        return self._last_pair_count

    @staticmethod
    def _pair_mask(
        name: str,
        mask: torch.Tensor,
        *,
        batch: int,
        query_tokens: int,
        key_tokens: int,
    ) -> torch.Tensor:
        if mask.ndim == 2 and mask.shape == (batch, key_tokens):
            return mask[:, None, :].expand(batch, query_tokens, key_tokens)
        if mask.ndim == 3 and mask.shape == (batch, query_tokens, key_tokens):
            return mask
        raise ValueError(
            f"{name} must have shape {(batch, key_tokens)} or "
            f"{(batch, query_tokens, key_tokens)}; got {tuple(mask.shape)}"
        )

    @staticmethod
    def _role_masks(
        mask: torch.Tensor,
        *,
        batch: int,
        query_tokens: int,
        key_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask.ndim == 2:
            query_mask = (
                mask
                if mask.shape == (batch, query_tokens)
                else torch.zeros((batch, query_tokens), dtype=torch.bool, device=mask.device)
            )
            key_mask = (
                mask
                if mask.shape == (batch, key_tokens)
                else torch.zeros((batch, key_tokens), dtype=torch.bool, device=mask.device)
            )
            if mask.shape not in {(batch, query_tokens), (batch, key_tokens)}:
                raise ValueError("role mask must match query or key token dimensions")
            return query_mask, key_mask
        if mask.ndim == 3 and mask.shape == (batch, query_tokens, key_tokens):
            return mask.any(dim=-1), mask.any(dim=1)
        raise ValueError("role mask must be rank two or a query-key mask")

    def _validate_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        scores: torch.Tensor | None,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
        query_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        if query.ndim != 4 or key.ndim != 4:
            raise ValueError("query and key must be rank four")
        if query.shape[:2] != key.shape[:2]:
            raise ValueError("query and key batch/head dimensions must match")
        if query.shape[1] != self.num_heads:
            raise ValueError("query/key head count does not match configuration")
        if query.shape[-1] != self.head_dim or key.shape[-1] != self.head_dim:
            raise ValueError("query/key head dimensions do not match configuration")
        batch, _heads, query_tokens, _dim = query.shape
        key_tokens = key.shape[2]
        if scores is not None and scores.shape != (batch, self.num_heads, query_tokens, key_tokens):
            raise ValueError("scores shape is incompatible with query/key")
        if not torch.isfinite(query).all() or not torch.isfinite(key).all():
            raise ValueError("query and key must be finite")

        attention_pair = self._pair_mask(
            "attention_mask",
            attention_mask,
            batch=batch,
            query_tokens=query_tokens,
            key_tokens=key_tokens,
        ).to(device=query.device, dtype=torch.bool)
        subject_query, subject_key = self._role_masks(
            subject_mask,
            batch=batch,
            query_tokens=query_tokens,
            key_tokens=key_tokens,
        )
        object_query, object_key = self._role_masks(
            object_mask,
            batch=batch,
            query_tokens=query_tokens,
            key_tokens=key_tokens,
        )
        if query_mask is None:
            active_query = attention_pair.any(dim=-1)
        elif query_mask.ndim == 2 and query_mask.shape == (batch, query_tokens):
            active_query = query_mask.to(device=query.device, dtype=torch.bool)
        else:
            raise ValueError("query_mask must have shape (batch, query_tokens)")
        return (
            attention_pair,
            subject_query.to(device=query.device, dtype=torch.bool),
            subject_key.to(device=query.device, dtype=torch.bool),
            object_query.to(device=query.device, dtype=torch.bool),
            object_key.to(device=query.device, dtype=torch.bool),
            active_query,
        )

    def _angles(
        self,
        value: torch.Tensor,
        *,
        role: int,
        layer_index: int,
        head_index: int,
    ) -> torch.Tensor:
        if role < 2:
            projection = self.input_projections[:2, head_index].mean(dim=0)
        else:
            projection = self.input_projections[role, head_index]
        projection = projection.to(device=value.device, dtype=value.dtype)
        projected = value @ projection
        if role < 2:
            scales = self.angle_scales[layer_index, head_index, :, :2].mean(dim=1)
            biases = self.angle_biases[layer_index, head_index, :, :2].mean(dim=1)
        else:
            scales = self.angle_scales[layer_index, head_index, :, role]
            biases = self.angle_biases[layer_index, head_index, :, role]
        scales = scales.to(
            device=value.device, dtype=value.dtype
        )
        biases = biases.to(
            device=value.device, dtype=value.dtype
        )
        return self.config.angle_scale * (projected[:, None, :] * scales + biases)

    def _coupling(self, layer_index: int, head_index: int, depth_index: int, qubit: int) -> torch.Tensor:
        return self.config.max_phase * torch.tanh(
            self.raw_couplings[layer_index, head_index, depth_index, qubit]
        )

    def _post(self, layer_index: int, head_index: int, depth_index: int, role: int, qubit: int) -> torch.Tensor:
        if role < 2:
            raw = self.raw_post_rotation[layer_index, head_index, depth_index, :2, qubit].mean()
        else:
            raw = self.raw_post_rotation[layer_index, head_index, depth_index, role, qubit]
        return self.config.max_post_rotation * torch.tanh(
            raw
        )

    @staticmethod
    def _apply_local_rotation(
        state: torch.Tensor,
        matrix: torch.Tensor,
        axis: int,
    ) -> torch.Tensor:
        """Apply a 2x2 local gate without mixing the register axes."""
        register_axis = axis + 1  # leading dimension is the flattened pair index
        moved = state.movedim(register_axis, -1)
        rotated = torch.matmul(moved, matrix.transpose(0, 1))
        return rotated.movedim(-1, register_axis)

    def _quantum_single_qubit_covariance(
        self,
        subject_angles: torch.Tensor,
        object_angles: torch.Tensor,
        key_angles: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
        qubit: int,
        _symmetric: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if _symmetric:
            forward = self._quantum_single_qubit_covariance(
                subject_angles,
                object_angles,
                key_angles,
                layer_index=layer_index,
                head_index=head_index,
                qubit=qubit,
                _symmetric=False,
            )
            reverse = self._quantum_single_qubit_covariance(
                object_angles,
                subject_angles,
                key_angles,
                layer_index=layer_index,
                head_index=head_index,
                qubit=qubit,
                _symmetric=False,
            )
            return (
                (forward[0] + reverse[0]) / 2.0,
                (forward[1] + reverse[1]) / 2.0,
            )
        q_s = torch.stack((torch.cos(subject_angles / 2.0), torch.sin(subject_angles / 2.0)), dim=-1)
        q_o = torch.stack((torch.cos(object_angles / 2.0), torch.sin(object_angles / 2.0)), dim=-1)
        q_k = torch.stack((torch.cos(key_angles / 2.0), torch.sin(key_angles / 2.0)), dim=-1)
        state = torch.einsum("na,nb,nc->nabc", q_s, q_o, q_k)
        complex_dtype = torch.complex128 if subject_angles.dtype == torch.float64 else torch.complex64
        state = state.to(complex_dtype)
        for depth_index in range(self.config.depth):
            # A local key rotation is applied before the shared-key coupling;
            # otherwise a post-coupling key-only unitary would disappear when
            # the key register is traced out of the role covariance.
            key_matrix = _ry_matrix(
                self._post(layer_index, head_index, depth_index, 2, qubit).to(
                    device=subject_angles.device, dtype=subject_angles.dtype
                )
            ).to(complex_dtype)
            state = self._apply_local_rotation(state, key_matrix, 2)
            phase_sk, phase_ok, phase_sok = self._coupling(
                layer_index, head_index, depth_index, qubit
            ).to(device=subject_angles.device, dtype=subject_angles.dtype)
            phase_role = (phase_sk + phase_ok) / 2.0
            phase = torch.zeros(8, device=subject_angles.device, dtype=subject_angles.dtype)
            phase[5] = phase_role + phase_sok  # |101>
            phase[6] = phase_role + phase_sok  # |110>
            phase[7] = 2.0 * phase_role + phase_sok  # |111>
            state = state * torch.exp(1j * phase).to(complex_dtype).reshape(1, 2, 2, 2)
            for role, axis in ((0, 0), (1, 1)):
                matrix = _ry_matrix(
                    self._post(layer_index, head_index, depth_index, role, qubit).to(
                        device=subject_angles.device, dtype=subject_angles.dtype
                    )
                ).to(complex_dtype)
                state = self._apply_local_rotation(state, matrix, axis)
        probabilities = state.abs().square()
        z = torch.tensor([1.0, -1.0], device=state.device, dtype=probabilities.dtype)
        mean_s = (probabilities * z.view(1, 2, 1, 1)).sum(dim=(1, 2, 3))
        mean_o = (probabilities * z.view(1, 1, 2, 1)).sum(dim=(1, 2, 3))
        joint = (probabilities * z.view(1, 2, 1, 1) * z.view(1, 1, 2, 1)).sum(dim=(1, 2, 3))
        return joint - mean_s * mean_o, mean_s * mean_o

    def _classical_single_qubit_covariance(
        self,
        subject_angles: torch.Tensor,
        object_angles: torch.Tensor,
        key_angles: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
        qubit: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        connected: list[torch.Tensor] = []
        independent: list[torch.Tensor] = []
        for depth_index in range(self.config.depth):
            s = subject_angles[:, depth_index] + self._post(layer_index, head_index, depth_index, 0, qubit)
            o = object_angles[:, depth_index] + self._post(layer_index, head_index, depth_index, 1, qubit)
            k = key_angles[:, depth_index] + self._post(layer_index, head_index, depth_index, 2, qubit)
            zs, zo, zk = torch.cos(s), torch.cos(o), torch.cos(k)
            phase_sk, phase_ok, phase_sok = torch.tanh(
                self._coupling(layer_index, head_index, depth_index, qubit)
            )
            phase_role = (phase_sk + phase_ok) / 2.0
            # Explicit tensor-product covariance control: no joint state is
            # formed, while the shared-key factors and all coupling weights
            # remain present in the forward and backward graphs.
            tensor_term = 0.5 * (
                phase_role * zs * zk
                + phase_role * zo * zk
                + phase_sok * zs * zo * zk
            )
            connected.append(tensor_term)
            independent.append(zs * zo)
        return torch.stack(connected, dim=-1).mean(dim=-1), torch.stack(independent, dim=-1).mean(dim=-1)

    def _score_from_angles(
        self,
        subject_angles: torch.Tensor,
        object_angles: torch.Tensor,
        key_angles: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        connected: list[torch.Tensor] = []
        independent: list[torch.Tensor] = []
        for qubit in range(self.config.num_qubits):
            if self.entangled:
                current, null = self._quantum_single_qubit_covariance(
                    subject_angles[:, :, qubit].reshape(-1),
                    object_angles[:, :, qubit].reshape(-1),
                    key_angles[:, :, qubit].reshape(-1),
                    layer_index=layer_index,
                    head_index=head_index,
                    qubit=qubit,
                )
            else:
                current, null = self._classical_single_qubit_covariance(
                    subject_angles[:, :, qubit],
                    object_angles[:, :, qubit],
                    key_angles[:, :, qubit],
                    layer_index=layer_index,
                    head_index=head_index,
                    qubit=qubit,
                )
            connected.append(current)
            independent.append(null)
        return torch.stack(connected, dim=-1).mean(dim=-1), torch.stack(independent, dim=-1).mean(dim=-1)

    def metadata(self) -> dict[str, Any]:
        return {
            "id": f"mqic_shared_key_query_covariance_{self.kernel_type}",
            "version": "0.1.0",
            "type": "shared_key_query_covariance",
            "insertion_point": "pre_softmax_attention_scores",
            "hypothesis": "a shared context key can couple pooled subject and object query roles through a connected covariance observable",
            "input_schema": "query,key,scores?,layer_index,attention_mask,subject_mask,object_mask,query_mask?; labels and relation IDs prohibited",
            "output_schema": "finite bounded context-only residual [batch,heads,query_tokens,key_tokens]",
            "requires": ["subject/object role masks", "mask-aware score hook"],
            "conflicts": ["q_rpec_quantum", "q_triad", "qk_coherent_transport", "stacked_score_interventions"],
            "deterministic": True,
            "resource_estimate": {
                "data_qubits": 3 * self.register_qubits,
                "ancilla_qubits": 0,
                "depth": self.config.depth,
                "pair_budget": "one shared-key covariance evaluation per batch-key pair",
                "trainable_parameter_count": self.parameter_count,
            },
            "failure_signatures": [
                "connected covariance collapses to independent product",
                "query-pair swap prediction fails",
                "key permutation equivariance fails",
                "masked/entity key receives action",
                "active context row is not zero-sum",
                "nonfinite score or gradient",
            ],
            "observability_contract": {
                "trace_schema": "mqic-shared-key-query-covariance-trace.v1",
                "emitted_fields": ["connected_covariance_std", "independent_product_gap", "residual_rms", "gradient_norm", "pair_count", "query_pair_swap_error", "key_permutation_equivariance_error"],
                "checks": ["finite", "nonzero_connected_covariance", "connected_vs_independent_gap", "context_zero_sum", "entity_mask_zero", "query_pair_swap", "key_permutation_equivariance", "deterministic_replay", "parameter_match"],
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
        (
            attention_pair,
            subject_query_mask,
            subject_key_mask,
            object_query_mask,
            object_key_mask,
            active_query,
        ) = self._validate_inputs(
            query,
            key,
            scores,
            attention_mask,
            subject_mask,
            object_mask,
            query_mask,
        )
        if not 0 <= layer_index < self.num_layers:
            raise ValueError("layer_index is outside configured layers")
        batch, heads, query_tokens, _dim = query.shape
        key_tokens = key.shape[2]
        context_keys = attention_pair.any(dim=1) & ~subject_key_mask & ~object_key_mask
        subject_query_mask = subject_query_mask & active_query
        object_query_mask = object_query_mask & active_query
        subject_count = subject_query_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(query.dtype)
        object_count = object_query_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(query.dtype)

        raw_scores: list[torch.Tensor] = []
        connected_scores: list[torch.Tensor] = []
        independent_scores: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []
        chunk_size = self.config.pair_chunk_size or max(1, batch * key_tokens)
        self._last_pair_count = batch * key_tokens

        for head_index in range(heads):
            query_head = query[:, head_index]
            key_head = key[:, head_index]
            subject_pooled = (query_head * subject_query_mask[:, :, None].to(query.dtype)).sum(dim=1) / subject_count
            object_pooled = (query_head * object_query_mask[:, :, None].to(query.dtype)).sum(dim=1) / object_count
            subject_all = subject_pooled[:, None, :].expand(batch, key_tokens, self.head_dim).reshape(-1, self.head_dim)
            object_all = object_pooled[:, None, :].expand(batch, key_tokens, self.head_dim).reshape(-1, self.head_dim)
            key_all = key_head.reshape(-1, self.head_dim)
            connected_chunks: list[torch.Tensor] = []
            independent_chunks: list[torch.Tensor] = []
            for start in range(0, batch * key_tokens, chunk_size):
                stop = min(start + chunk_size, batch * key_tokens)
                subject_angles = self._angles(subject_all[start:stop], role=0, layer_index=layer_index, head_index=head_index)
                object_angles = self._angles(object_all[start:stop], role=1, layer_index=layer_index, head_index=head_index)
                key_angles = self._angles(key_all[start:stop], role=2, layer_index=layer_index, head_index=head_index)
                connected, independent = self._score_from_angles(
                    subject_angles,
                    object_angles,
                    key_angles,
                    layer_index=layer_index,
                    head_index=head_index,
                )
                connected_chunks.append(connected)
                independent_chunks.append(independent)
            connected = torch.cat(connected_chunks).reshape(batch, key_tokens)
            independent = torch.cat(independent_chunks).reshape(batch, key_tokens)
            gain = self.config.max_covariance * torch.tanh(self.raw_gain[layer_index, head_index]).to(query.dtype)
            connected_scaled = connected * gain
            role_sign = subject_query_mask.to(query.dtype) - object_query_mask.to(query.dtype)
            raw = role_sign[:, :, None] * connected_scaled[:, None, :]
            valid_context = context_keys[:, None, :]
            active_role = (role_sign.abs() > 0) & active_query
            row_mask = valid_context & active_role[:, :, None]
            denominator = row_mask.to(query.dtype).sum(dim=-1, keepdim=True).clamp_min(self.config.eps)
            centered = raw - (raw * row_mask.to(query.dtype)).sum(dim=-1, keepdim=True) / denominator
            residual = centered * row_mask.to(query.dtype)
            raw_scores.append(raw)
            connected_scores.append(connected_scaled)
            independent_scores.append(independent)
            residuals.append(residual)

        raw_score = torch.stack(raw_scores, dim=1)
        connected_covariance = torch.stack(connected_scores, dim=1)
        independent_product = torch.stack(independent_scores, dim=1)
        residual = torch.stack(residuals, dim=1)
        self._last_raw_score = raw_score.detach()
        self._last_connected_covariance = connected_covariance.detach()
        self._last_independent_product = independent_product.detach()
        self._last_residual = residual.detach()
        return residual


class QuantumMultiQueryCovarianceKernel(MultiQueryCovarianceKernel):
    kernel_type = "quantum"
    entangled = True


class ClassicalMultiQueryCovarianceKernel(MultiQueryCovarianceKernel):
    kernel_type = "classical"
    entangled = False


def build_multi_query_covariance_kernel(
    kernel_type: str,
    config: MultiQueryCovarianceConfig,
) -> MultiQueryCovarianceKernel:
    classes = {
        "quantum": QuantumMultiQueryCovarianceKernel,
        "classical": ClassicalMultiQueryCovarianceKernel,
    }
    try:
        cls = classes[kernel_type]
    except KeyError as error:
        raise ValueError(f"kernel_type must be one of {MQIC_CONTROL_MODES}") from error
    return cls(config)
