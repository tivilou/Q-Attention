"""Isolated Q-CEASC support-expansion toy.

Q-CEASC constructs a context-conditioned auxiliary state, decodes signed
observable coefficients, and projects the resulting support vector outside a
frozen action span.  This module is deliberately stage-A only: it is not wired
to the production attention runner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn as nn


QCEASC_CONTROL_MODES = (
    "q_ceasc",
    "quantum_product",
    "classical_span",
    "fixed_bank",
    "random_support",
)


def _seeded_normal(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def _orthonormal_columns(matrix: torch.Tensor) -> torch.Tensor:
    basis, _ = torch.linalg.qr(matrix.float(), mode="reduced")
    return basis


def _apply_ry(
    state: torch.Tensor,
    angles: torch.Tensor,
    qubit: int,
    num_qubits: int,
) -> torch.Tensor:
    view = state.reshape(
        state.shape[0],
        2**qubit,
        2,
        2 ** (num_qubits - qubit - 1),
    )
    low = view[:, :, 0, :]
    high = view[:, :, 1, :]
    cosine = torch.cos(angles / 2).view(-1, 1, 1)
    sine = torch.sin(angles / 2).view(-1, 1, 1)
    return torch.stack(
        (cosine * low - sine * high, sine * low + cosine * high),
        dim=2,
    ).reshape_as(state)


def _apply_cnot(
    state: torch.Tensor,
    control: int,
    target: int,
    num_qubits: int,
) -> torch.Tensor:
    indices = torch.arange(2**num_qubits, device=state.device)
    control_mask = 1 << (num_qubits - control - 1)
    target_mask = 1 << (num_qubits - target - 1)
    permutation = torch.where(
        (indices & control_mask) != 0,
        indices ^ target_mask,
        indices,
    )
    return state[:, permutation]


def _apply_rzz(
    state: torch.Tensor,
    angle: torch.Tensor,
    control: int,
    target: int,
    num_qubits: int,
) -> torch.Tensor:
    indices = torch.arange(2**num_qubits, device=state.device)
    control_mask = 1 << (num_qubits - control - 1)
    target_mask = 1 << (num_qubits - target - 1)
    control_z = torch.where(
        (indices & control_mask) == 0,
        torch.ones_like(indices, dtype=state.real.dtype),
        -torch.ones_like(indices, dtype=state.real.dtype),
    )
    target_z = torch.where(
        (indices & target_mask) == 0,
        torch.ones_like(indices, dtype=state.real.dtype),
        -torch.ones_like(indices, dtype=state.real.dtype),
    )
    phase_angle = -0.5 * angle[:, None] * control_z[None, :] * target_z[None, :]
    phase = torch.exp(torch.complex(torch.zeros_like(phase_angle), phase_angle))
    return state * phase.to(dtype=state.dtype)


def _product_state(angles: torch.Tensor) -> torch.Tensor:
    local = torch.stack(
        (torch.cos(angles / 2), torch.sin(angles / 2)),
        dim=-1,
    )
    state = torch.ones(
        angles.shape[0],
        1,
        device=angles.device,
        dtype=angles.dtype,
    )
    for qubit in range(angles.shape[1]):
        state = (
            state.unsqueeze(-1) * local[:, qubit].unsqueeze(1)
        ).reshape(angles.shape[0], -1)
    return state / torch.linalg.vector_norm(state, dim=-1, keepdim=True).clamp_min(1e-12)


def _active_center(logits: torch.Tensor, active: torch.Tensor, eps: float) -> torch.Tensor:
    weights = active.to(device=logits.device, dtype=logits.dtype)
    count = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    centered = logits - (logits * weights).sum(dim=-1, keepdim=True) / count
    return centered * weights


def _entangling_covariance_norm(
    state: torch.Tensor,
    num_qubits: int,
) -> torch.Tensor:
    """Return the norm of pairwise Z covariance under the computational readout."""
    probabilities = state.abs().square()
    if num_qubits < 2:
        return torch.zeros(
            state.shape[0], device=state.device, dtype=probabilities.dtype
        )
    indices = torch.arange(2**num_qubits, device=state.device)
    z_values = []
    for qubit in range(num_qubits):
        mask = 1 << (num_qubits - qubit - 1)
        z_values.append(
            torch.where(
                (indices & mask) == 0,
                torch.ones_like(indices, dtype=probabilities.dtype),
                -torch.ones_like(indices, dtype=probabilities.dtype),
            )
        )
    means = torch.stack([probabilities @ values for values in z_values], dim=-1)
    covariances = []
    for first in range(num_qubits):
        for second in range(first + 1, num_qubits):
            pair_values = z_values[first] * z_values[second]
            pair_mean = probabilities @ pair_values
            covariances.append(pair_mean - means[:, first] * means[:, second])
    return torch.stack(covariances, dim=-1).norm(dim=-1)


@dataclass(frozen=True)
class QCEASCConfig:
    query_dim: int
    key_dim: int
    auxiliary_qubits: int = 3
    depth: int = 2
    support_width: int = 2
    action_rank: int = 2
    angle_scale: float = 1.0
    max_gain: float = 0.25
    initial_gain: float = 0.05
    span_rcond: float = 1e-6
    seed: int = 13091
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.query_dim <= 0 or self.key_dim <= 0:
            raise ValueError("query_dim and key_dim must be positive")
        if not 2 <= self.auxiliary_qubits <= 6:
            raise ValueError("auxiliary_qubits must be in [2, 6]")
        if self.depth <= 0:
            raise ValueError("depth must be positive")
        state_dim = 2**self.auxiliary_qubits
        if not 0 < self.support_width <= min(self.key_dim, state_dim):
            raise ValueError("support_width must fit key and state dimensions")
        if not 0 < self.action_rank < self.key_dim:
            raise ValueError("action_rank must lie in [1, key_dim)")
        if self.angle_scale <= 0.0 or self.eps <= 0.0:
            raise ValueError("angle_scale and eps must be positive")
        if self.max_gain <= 0.0 or not abs(self.initial_gain) < self.max_gain:
            raise ValueError("initial_gain must lie inside max_gain")
        if self.span_rcond <= 0.0:
            raise ValueError("span_rcond must be positive")


@dataclass(frozen=True)
class QCEASCResult:
    residual: torch.Tensor
    support_vector: torch.Tensor
    projected_support: torch.Tensor
    coefficients: torch.Tensor
    auxiliary_state: torch.Tensor
    diagnostics: dict[str, Any]


class ContextEntangledAuxiliarySupportConstructor(nn.Module):
    """Context-conditioned auxiliary support constructor with matched controls."""

    plugin_type = "q_ceasc_stage_a"

    def __init__(
        self,
        config: QCEASCConfig,
        control_mode: str = "q_ceasc",
        *,
        action_span: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if control_mode not in QCEASC_CONTROL_MODES:
            raise ValueError(f"control_mode must be one of {QCEASC_CONTROL_MODES}")
        self.config = config
        self.control_mode = control_mode
        self.state_dim = 2**config.auxiliary_qubits
        self.context_dim = config.query_dim + config.key_dim
        self.pair_count = config.auxiliary_qubits * (config.auxiliary_qubits - 1) // 2
        self.quantum_core_parameter_count = (
            2 * config.depth * config.auxiliary_qubits
            + 2 * config.depth * self.pair_count
        )

        if action_span is None:
            action_span = _seeded_normal(
                (config.key_dim, config.action_rank), config.seed + 101
            )
        if action_span.shape != (config.key_dim, config.action_rank):
            raise ValueError("action_span must have shape (key_dim, action_rank)")
        action_span = action_span.detach().float()
        span_u, singular_values, _ = torch.linalg.svd(action_span, full_matrices=False)
        cutoff = singular_values.max().clamp_min(config.eps) * config.span_rcond
        effective_rank = int((singular_values > cutoff).sum().item())
        if effective_rank <= 0 or effective_rank >= config.key_dim:
            raise ValueError("action_span must leave a non-empty orthogonal complement")
        span_basis = span_u[:, :effective_rank]
        identity = torch.eye(
            config.key_dim,
            dtype=torch.float32,
            device=span_basis.device,
        )
        complement = identity - span_basis @ span_basis.transpose(0, 1)
        self.register_buffer("action_span", action_span.float())
        self.register_buffer("span_basis", span_basis)
        self.register_buffer("span_singular_values", singular_values)
        self.register_buffer("orthogonal_complement", complement)

        support_dictionary = _orthonormal_columns(
            _seeded_normal((config.key_dim, config.support_width), config.seed + 211)
        )
        self.register_buffer("support_dictionary", support_dictionary)

        observables = _seeded_normal(
            (config.support_width, self.state_dim), config.seed + 307
        ).sign()
        self.register_buffer("signed_observables", observables)

        self.register_buffer(
            "context_projection",
            _seeded_normal((self.context_dim, config.auxiliary_qubits), config.seed + 401)
            / math.sqrt(float(self.context_dim)),
        )
        self.register_buffer(
            "pair_projection",
            _seeded_normal((self.context_dim, self.pair_count), config.seed + 503)
            / math.sqrt(float(self.context_dim)),
        )

        generator = torch.Generator(device="cpu").manual_seed(config.seed + 601)
        shape = (config.depth, config.auxiliary_qubits)
        pair_shape = (config.depth, self.pair_count)
        if control_mode == "classical_span":
            self.classical_coefficients = nn.Parameter(
                torch.zeros(self.quantum_core_parameter_count)
            )
            self.classical_raw_gain = nn.Parameter(
                torch.tensor(math.atanh(config.initial_gain / config.max_gain))
            )
        else:
            self.local_scales = nn.Parameter(torch.ones(shape))
            self.local_biases = nn.Parameter(
                torch.empty(shape).uniform_(-math.pi / 4, math.pi / 4, generator=generator)
            )
            self.pair_scales = nn.Parameter(torch.ones(pair_shape))
            self.pair_biases = nn.Parameter(
                torch.empty(pair_shape).uniform_(-math.pi / 8, math.pi / 8, generator=generator)
            )
            self.raw_gain = nn.Parameter(
                torch.tensor(math.atanh(config.initial_gain / config.max_gain))
            )

        classical_width = self.quantum_core_parameter_count
        self.register_buffer(
            "classical_feature_tensor",
            _seeded_normal(
                (self.context_dim, classical_width, config.key_dim), config.seed + 701
            )
            / math.sqrt(float(self.context_dim * max(classical_width, 1))),
        )
        random_support = _seeded_normal(
            (config.key_dim,), config.seed + 809
        ).to(device=complement.device)
        random_support = complement @ random_support
        random_support = random_support / torch.linalg.vector_norm(random_support).clamp_min(
            config.eps
        )
        self.register_buffer("random_projected_support", random_support)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def metadata(self) -> dict[str, Any]:
        complement_rank = self.config.key_dim - int(self.span_basis.shape[1])
        return {
            "id": self.control_mode,
            "version": "0.1.0",
            "type": self.plugin_type,
            "mechanism": "context-conditioned auxiliary support with frozen-span orthogonal projection",
            "control_mode": self.control_mode,
            "input_schema": "query (B,Dq), key (B,N,Dk), valid/entity masks",
            "output_schema": "finite context-only zero-sum bounded residual (B,N)",
            "parameter_count": self.parameter_count,
            "quantum_core_parameter_count": self.quantum_core_parameter_count + 1,
            "state_dim": self.state_dim,
            "auxiliary_qubits": self.config.auxiliary_qubits,
            "span_rank": int(self.span_basis.shape[1]),
            "complement_rank": complement_rank,
            "span_rcond": self.config.span_rcond,
            "readout": "signed_bipolar_observable_expectations",
            "support_dictionary": "frozen_label_free orthonormal columns",
            "classical_control_semantics": (
                "matched-parameter D-dimensional random-feature span control"
                if self.control_mode == "classical_span"
                else None
            ),
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if query.ndim != 2 or query.shape[-1] != self.config.query_dim:
            raise ValueError("query must have shape (batch, query_dim)")
        if key.ndim != 3 or key.shape[-1] != self.config.key_dim:
            raise ValueError("key must have shape (batch, context_keys, key_dim)")
        if query.shape[0] != key.shape[0]:
            raise ValueError("query and key batch sizes must match")
        if query.device != key.device:
            raise ValueError("query and key must be on the same device")
        if valid_context_mask.shape != key.shape[:2]:
            raise ValueError("valid_context_mask must have shape (batch, context_keys)")
        if entity_mask is None:
            entity_mask = torch.zeros_like(valid_context_mask, dtype=torch.bool)
        if entity_mask.shape != key.shape[:2]:
            raise ValueError("entity_mask must have shape (batch, context_keys)")
        if query.is_complex() or key.is_complex() or not query.is_floating_point() or not key.is_floating_point():
            raise ValueError("query and key must be real floating-point tensors")
        dtype = torch.promote_types(query.dtype, key.dtype)
        if dtype not in (torch.float32, torch.float64):
            dtype = torch.float32
        query = query.to(dtype=dtype)
        key = key.to(dtype=dtype)
        if not torch.isfinite(query).all() or not torch.isfinite(key).all():
            raise ValueError("query and key must be finite")
        active = valid_context_mask.to(device=key.device, dtype=torch.bool) & ~entity_mask.to(
            device=key.device, dtype=torch.bool
        )
        return query, key, active

    def _context_features(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        weights = active.to(device=key.device, dtype=key.dtype)
        count = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean_key = (key * weights.unsqueeze(-1)).sum(dim=1) / count
        return torch.cat((query, mean_key), dim=-1)

    def _auxiliary_state(self, context: torch.Tensor, *, entangle: bool) -> torch.Tensor:
        dtype = context.dtype
        device = context.device
        projection = self.context_projection.to(device=device, dtype=dtype)
        base_angles = self.config.angle_scale * torch.matmul(context, projection)
        state = _product_state(math.pi * torch.tanh(base_angles) + math.pi / 2).to(
            dtype=torch.complex128 if dtype == torch.float64 else torch.complex64
        )
        pair_projection = self.pair_projection.to(device=device, dtype=dtype)
        pair_angles = (
            self.config.angle_scale * torch.matmul(context, pair_projection)
            if self.pair_count
            else context.new_zeros((context.shape[0], 0))
        )
        for depth_index in range(self.config.depth):
            local_argument = (
                base_angles * self.local_scales[depth_index].to(dtype=dtype)
                + self.local_biases[depth_index].to(dtype=dtype)
            )
            local = math.pi * torch.tanh(local_argument)
            for qubit in range(self.config.auxiliary_qubits):
                state = _apply_ry(state, local[:, qubit], qubit, self.config.auxiliary_qubits)
            if entangle:
                pair_index = 0
                for control in range(self.config.auxiliary_qubits):
                    for target in range(control + 1, self.config.auxiliary_qubits):
                        angle_argument = (
                            pair_angles[:, pair_index]
                            * self.pair_scales[depth_index, pair_index].to(dtype=dtype)
                            + self.pair_biases[depth_index, pair_index].to(dtype=dtype)
                        )
                        angle = math.pi * torch.tanh(angle_argument)
                        state = _apply_rzz(
                            state,
                            angle,
                            control,
                            target,
                            self.config.auxiliary_qubits,
                        )
                        pair_index += 1
                for control in range(self.config.auxiliary_qubits):
                    state = _apply_cnot(
                        state,
                        control,
                        (control + 1) % self.config.auxiliary_qubits,
                        self.config.auxiliary_qubits,
                    )
        return state / torch.linalg.vector_norm(state, dim=-1, keepdim=True).clamp_min(
            self.config.eps
        )

    def _signed_coefficients(self, state: torch.Tensor) -> torch.Tensor:
        probabilities = state.abs().square()
        observables = self.signed_observables.to(
            device=state.device, dtype=probabilities.dtype
        )
        return torch.matmul(probabilities, observables.transpose(0, 1))

    def _quantum_support(
        self,
        context: torch.Tensor,
        *,
        entangle: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self._auxiliary_state(context, entangle=entangle)
        coefficients = self._signed_coefficients(state)
        dictionary = self.support_dictionary.to(
            device=context.device, dtype=context.dtype
        )
        support = torch.matmul(coefficients, dictionary.transpose(0, 1))
        return support, coefficients, state

    def _classical_support(self, context: torch.Tensor) -> torch.Tensor:
        features = self.classical_feature_tensor.to(
            device=context.device, dtype=context.dtype
        )
        coefficients = self.classical_coefficients.to(dtype=context.dtype)
        return torch.einsum("bc,cpd,p->bd", context, features, coefficients)

    def _project(self, support: torch.Tensor) -> torch.Tensor:
        complement = self.orthogonal_complement.to(
            device=support.device, dtype=support.dtype
        )
        return torch.matmul(support, complement)

    def evaluate(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        valid_context_mask: torch.Tensor,
        entity_mask: torch.Tensor | None = None,
    ) -> QCEASCResult:
        query, key, active = self._validate_inputs(
            query, key, valid_context_mask, entity_mask
        )
        valid = valid_context_mask.to(device=key.device, dtype=torch.bool)
        entity = (
            torch.zeros_like(valid, dtype=torch.bool)
            if entity_mask is None
            else entity_mask.to(device=key.device, dtype=torch.bool)
        )
        valid_context_token_count = valid.sum(dim=-1)
        entity_context_token_count = (valid & entity).sum(dim=-1)
        active_context_token_count = active.sum(dim=-1)
        context = self._context_features(query, key, active)
        if self.control_mode == "classical_span":
            support = self._classical_support(context)
            coefficients = context.new_zeros((context.shape[0], self.config.support_width))
            state = torch.zeros(
                context.shape[0],
                self.state_dim,
                device=context.device,
                dtype=torch.complex128 if context.dtype == torch.float64 else torch.complex64,
            )
        elif self.control_mode == "random_support":
            support = self.random_projected_support.to(
                device=context.device, dtype=context.dtype
            ).expand(context.shape[0], -1)
            coefficients = context.new_zeros((context.shape[0], self.config.support_width))
            state = torch.zeros(
                context.shape[0],
                self.state_dim,
                device=context.device,
                dtype=torch.complex128 if context.dtype == torch.float64 else torch.complex64,
            )
        else:
            entangle = self.control_mode != "quantum_product"
            support, coefficients, state = self._quantum_support(context, entangle=entangle)

        complement = self.orthogonal_complement.to(
            device=context.device, dtype=context.dtype
        )
        identity = torch.eye(
            self.config.key_dim, device=context.device, dtype=context.dtype
        )
        span_projector = identity - complement
        if self.control_mode == "fixed_bank":
            projected = torch.matmul(support, span_projector)
        elif self.control_mode == "random_support":
            projected = support
        else:
            projected = self._project(support)

        logits = torch.einsum("bnd,bd->bn", key, projected)
        centered = _active_center(logits, active, self.config.eps)
        raw_gain = self.classical_raw_gain if self.control_mode == "classical_span" else self.raw_gain
        gain = self.config.max_gain * torch.tanh(raw_gain.to(dtype=context.dtype))
        residual = centered * gain
        projection_error = torch.linalg.matrix_norm(
            torch.matmul(complement, complement) - complement
        )
        span_leakage = torch.linalg.vector_norm(
            torch.matmul(projected, span_projector), dim=-1
        )
        if self.control_mode == "fixed_bank":
            out_of_span_support = support.new_zeros((support.shape[0],))
            in_span_support = projected
        else:
            out_of_span_support = projected
            in_span_support = support - projected
        masked = ~active
        masked_error = (
            residual.masked_select(masked).abs().max()
            if bool(masked.any())
            else residual.new_zeros(())
        )
        zero_sum_error = residual.sum(dim=-1).abs()
        return QCEASCResult(
            residual=residual,
            support_vector=support,
            projected_support=projected,
            coefficients=coefficients,
            auxiliary_state=state,
            diagnostics={
                "control_mode": self.control_mode,
                "parameter_count": self.parameter_count,
                "quantum_core_parameter_count": self.quantum_core_parameter_count + 1,
                "auxiliary_state_norm": torch.linalg.vector_norm(state, dim=-1).real.detach(),
                "entangling_covariance_norm": _entangling_covariance_norm(
                    state, self.config.auxiliary_qubits
                ).detach(),
                "support_basis_entropy": (
                    -(coefficients.abs() / coefficients.abs().sum(dim=-1, keepdim=True).clamp_min(self.config.eps))
                    * torch.log((coefficients.abs() / coefficients.abs().sum(dim=-1, keepdim=True).clamp_min(self.config.eps)).clamp_min(self.config.eps))
                ).sum(dim=-1).detach(),
                "out_of_span_residual_norm": torch.linalg.vector_norm(projected, dim=-1).detach(),
                "out_of_span_support_norm": torch.linalg.vector_norm(
                    out_of_span_support, dim=-1
                ).detach(),
                "in_span_support_norm": torch.linalg.vector_norm(
                    in_span_support, dim=-1
                ).detach(),
                "projection_idempotence_error": float(projection_error.detach().cpu()),
                "span_leakage_norm": span_leakage.detach(),
                "mask_entity_zero_error": masked_error.detach(),
                "zero_sum_error": zero_sum_error.detach(),
                "effective_span_rank": int(self.span_basis.shape[1]),
                "effective_complement_rank": self.config.key_dim - int(self.span_basis.shape[1]),
                "span_rcond": self.config.span_rcond,
                "readout": "signed_bipolar_observable_expectations",
                "context_is_label_free": True,
                "valid_context_token_count": valid_context_token_count.detach(),
                "entity_context_token_count": entity_context_token_count.detach(),
                "active_context_token_count": active_context_token_count.detach(),
                "empty_context_row": (active_context_token_count == 0).detach(),
            },
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        valid_context_mask: torch.Tensor,
        entity_mask: torch.Tensor | None = None,
    ) -> QCEASCResult:
        return self.evaluate(query, key, valid_context_mask, entity_mask)


def build_qceasc(
    mode: str,
    config: QCEASCConfig,
    *,
    action_span: torch.Tensor | None = None,
) -> ContextEntangledAuxiliarySupportConstructor:
    if mode not in QCEASC_CONTROL_MODES:
        raise ValueError(f"mode must be one of {QCEASC_CONTROL_MODES}")
    return ContextEntangledAuxiliarySupportConstructor(
        config,
        mode,
        action_span=action_span,
    )
