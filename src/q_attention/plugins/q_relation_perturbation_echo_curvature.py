"""Relation-perturbation echo curvature attention-score plugins.

Q-RPEC measures a symmetric second response to a relation-anchor perturbation.
The quantum branch applies a relation-key controlled phase before a mixed
three-register Pauli readout (XXX plus a configurable XZZ term); the matched
control uses the same local encoding and parameters without the cross-register
phase.  The phase rescales only XXX while XZZ remains input-dependent, so the
quantum branch is not an input-independent rescaling of its control.  Both
branches expose the same label-free, context-only centered score residual
interface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


Q_RPEC_KERNEL_TYPES = ("quantum", "local_control")


@dataclass(frozen=True)
class RelationPerturbationEchoConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    num_qubits: int = 2
    angle_scale: float = 1.0
    perturbation: float = 0.2
    max_coupling: float = math.pi / 2.0
    max_gain: float = 0.5
    mixed_readout_weight: float = 0.5
    initial_gain: float = 0.02
    seed: int = 271
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.num_layers <= 0 or self.num_heads <= 0 or self.head_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if self.num_qubits <= 0 or self.num_qubits > 4:
            raise ValueError("num_qubits must be in [1, 4]")
        if self.angle_scale <= 0.0 or self.perturbation <= 0.0:
            raise ValueError("angle_scale and perturbation must be positive")
        if self.max_coupling <= 0.0 or self.max_gain <= 0.0:
            raise ValueError("max_coupling and max_gain must be positive")
        if not math.isfinite(self.mixed_readout_weight) or self.mixed_readout_weight <= 0.0:
            raise ValueError("mixed_readout_weight must be finite and positive")
        if not -self.max_gain < self.initial_gain < self.max_gain:
            raise ValueError("initial_gain must lie inside (-max_gain, max_gain)")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")


def _seeded_projection(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(input_dim, output_dim, generator=generator)
    return projection / math.sqrt(float(input_dim))


def _product_state(angles: torch.Tensor) -> torch.Tensor:
    """Return normalized product amplitudes for [N, qubits] rotation angles."""
    local = torch.stack((torch.cos(angles / 2.0), torch.sin(angles / 2.0)), dim=-1)
    state = local[:, 0]
    for qubit in range(1, local.shape[1]):
        state = (state.unsqueeze(-1) * local[:, qubit].unsqueeze(-2)).reshape(
            state.shape[0], -1
        )
    return F.normalize(state, p=2, dim=-1)


class RelationPerturbationEchoCurvatureKernel(nn.Module):
    """Standalone Q-RPEC score kernel with an explicit local control."""

    kernel_type = "quantum"
    uses_cross_register_echo = True

    def __init__(
        self,
        config: RelationPerturbationEchoConfig,
        *,
        pair_chunk_size: int | None = 256,
        pair_chunk_divisor: int = 1,
        activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if pair_chunk_size is not None and pair_chunk_size <= 0:
            raise ValueError("pair_chunk_size must be positive")
        if pair_chunk_divisor <= 0:
            raise ValueError("pair_chunk_divisor must be positive")
        self.config = config
        # None means one chunk containing all pairs for the current physical
        # micro-batch. A divisor lets the adaptive runner halve that chunk
        # without guessing sequence padding lengths in the parent process.
        self.pair_chunk_size = None if pair_chunk_size is None else int(pair_chunk_size)
        self.pair_chunk_divisor = int(pair_chunk_divisor)
        self.last_total_pairs = 0
        self.last_resolved_pair_chunk_size = 0
        self.activation_checkpointing = bool(activation_checkpointing)
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.max_gain = config.max_gain
        self.eps = config.eps
        projections = torch.stack(
            [
                _seeded_projection(
                    config.head_dim,
                    config.num_qubits,
                    config.seed + 101 * role,
                )
                for role in range(3)
            ]
        )
        self.register_buffer("input_projections", projections)
        shape = (config.num_layers, config.num_heads, 3, config.num_qubits)
        generator = torch.Generator(device="cpu").manual_seed(config.seed + 2001)
        self.angle_scales = nn.Parameter(torch.ones(shape))
        self.angle_biases = nn.Parameter(
            torch.empty(shape).uniform_(-math.pi / 4, math.pi / 4, generator=generator)
        )
        coupling_shape = (config.num_layers, config.num_heads, config.num_qubits)
        # Start with a small, nonzero echo so the quantum/control discriminator
        # is observable at initialization while remaining smoothly trainable.
        self.raw_coupling = nn.Parameter(torch.full(coupling_shape, 0.35))
        ratio = torch.tensor(config.initial_gain / config.max_gain)
        self.raw_gains = nn.Parameter(
            torch.full((config.num_layers, config.num_heads), float(torch.atanh(ratio)))
        )
        self.register_buffer("observable_flip", self._make_observable_flip(), persistent=False)
        self.register_buffer("xzz_flip", self._make_xzz_flip(), persistent=False)
        self.register_buffer("xzz_sign", self._make_xzz_sign(), persistent=False)
        self.register_buffer("phase_masks", self._make_phase_masks(), persistent=False)

    @property
    def model_dimensions(self) -> tuple[int, int, int]:
        return self.num_layers, self.num_heads, self.head_dim

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _make_observable_flip(self) -> torch.Tensor:
        total = 3 * self.config.num_qubits
        indices = torch.arange(2**total)
        flip = torch.zeros_like(indices)
        for register in range(3):
            for qubit in range(self.config.num_qubits):
                position = register * self.config.num_qubits + qubit
                flip = flip | (1 << (total - position - 1))
        return indices ^ flip

    def _make_xzz_flip(self) -> torch.Tensor:
        """Return basis indices after applying X to every query qubit only."""
        total = 3 * self.config.num_qubits
        indices = torch.arange(2**total)
        flip = 0
        for qubit in range(self.config.num_qubits):
            flip |= 1 << (total - qubit - 1)
        return indices ^ flip

    def _make_xzz_sign(self) -> torch.Tensor:
        """Return the Z-relation/Z-key eigenvalue for every basis state."""
        total = 3 * self.config.num_qubits
        indices = torch.arange(2**total)
        sign = torch.ones(indices.shape, dtype=torch.float32)
        for register in (1, 2):
            for qubit in range(self.config.num_qubits):
                position = register * self.config.num_qubits + qubit
                bit_mask = 1 << (total - position - 1)
                bit = ((indices & bit_mask) != 0).to(torch.float32)
                sign = sign * (1.0 - 2.0 * bit)
        return sign

    def _make_phase_masks(self) -> torch.Tensor:
        total = 3 * self.config.num_qubits
        indices = torch.arange(2**total)
        masks = []
        for qubit in range(self.config.num_qubits):
            relation_position = self.config.num_qubits + qubit
            key_position = 2 * self.config.num_qubits + qubit
            relation_mask = 1 << (total - relation_position - 1)
            key_mask = 1 << (total - key_position - 1)
            masks.append(((indices & relation_mask) != 0).to(torch.float32) * ((indices & key_mask) != 0).to(torch.float32))
        return torch.stack(masks, dim=-1)

    def _validate_inputs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: torch.Tensor,
        subject_mask: torch.Tensor,
        object_mask: torch.Tensor,
    ) -> None:
        if query.ndim != 4 or key.shape != query.shape:
            raise ValueError("query and key must have shape (batch, heads, tokens, head_dim)")
        if query.shape[1] != self.num_heads or query.shape[-1] != self.head_dim:
            raise ValueError("query shape does not match configured model dimensions")
        expected = (query.shape[0], query.shape[2])
        for name, mask in (("attention_mask", attention_mask), ("subject_mask", subject_mask), ("object_mask", object_mask)):
            if mask.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
        if not torch.isfinite(query).all() or not torch.isfinite(key).all():
            raise ValueError("query and key must be finite")

    def _role_angles(
        self,
        value: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
        role: int,
    ) -> torch.Tensor:
        projected = torch.matmul(value.float(), self.input_projections[role].to(value.device))
        return self.config.angle_scale * (
            projected * self.angle_scales[layer_index, head_index, role]
            + self.angle_biases[layer_index, head_index, role]
        )

    def _observable_statevector_reference(
        self,
        query: torch.Tensor,
        relation: torch.Tensor,
        key: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
    ) -> torch.Tensor:
        query_state = _product_state(self._role_angles(query, layer_index=layer_index, head_index=head_index, role=0))
        relation_state = _product_state(self._role_angles(relation, layer_index=layer_index, head_index=head_index, role=1))
        key_state = _product_state(self._role_angles(key, layer_index=layer_index, head_index=head_index, role=2))
        state = query_state
        state = (state.unsqueeze(-1) * relation_state.unsqueeze(-2)).reshape(state.shape[0], -1)
        state = (state.unsqueeze(-1) * key_state.unsqueeze(-2)).reshape(state.shape[0], -1)
        if self.uses_cross_register_echo:
            coupling = self.config.max_coupling * torch.tanh(self.raw_coupling[layer_index, head_index])
            phase = torch.exp(1j * torch.matmul(self.phase_masks.to(state.device), coupling.to(state.device)))
            state = state.to(torch.complex64) * phase
        else:
            # Keep the matched control's parameter schema and optimizer contract
            # identical while disabling the cross-register interaction exactly.
            # Reduce the parameter tensor to a scalar so broadcasting cannot
            # change the state shape while preserving an exact zero gradient.
            state = state + self.raw_coupling[layer_index, head_index].sum().to(state.device) * 0.0
        xxx_flipped = state[:, self.observable_flip.to(state.device)]
        xzz_flipped = state[:, self.xzz_flip.to(state.device)]
        xzz_flipped = xzz_flipped * self.xzz_sign.to(state.device, dtype=state.real.dtype)
        xxx = (state.conj() * xxx_flipped).sum(dim=-1).real
        xzz = (state.conj() * xzz_flipped).sum(dim=-1).real
        return xxx + self.config.mixed_readout_weight * xzz

    def _observable(
        self,
        query: torch.Tensor,
        relation: torch.Tensor,
        key: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
    ) -> torch.Tensor:
        """Evaluate the exact mixed Pauli echo without materializing a statevector.

        The circuit is a tensor product of real R_y product states, followed by
        a diagonal relation-key phase. The readout is the sum of two Pauli
        strings, XXX and XZZ (X on the query register, Z on relation and key).
        The phase rescales only the XXX term; the XZZ term remains phase
        invariant. Their sum therefore cannot collapse to an input-independent
        scalar multiple of the matched local control.
        """
        query_angles = self._role_angles(
            query, layer_index=layer_index, head_index=head_index, role=0
        )
        relation_angles = self._role_angles(
            relation, layer_index=layer_index, head_index=head_index, role=1
        )
        key_angles = self._role_angles(
            key, layer_index=layer_index, head_index=head_index, role=2
        )
        query_sine = torch.sin(query_angles)
        xxx_factors = (
            query_sine
            * torch.sin(relation_angles)
            * torch.sin(key_angles)
        )
        xzz_factors = (
            query_sine
            * torch.cos(relation_angles)
            * torch.cos(key_angles)
        )
        if self.uses_cross_register_echo:
            coupling = self.config.max_coupling * torch.tanh(
                self.raw_coupling[layer_index, head_index]
            )
            xxx_factors = xxx_factors * torch.cos(coupling / 2.0).square()
        else:
            # Keep the matched control's parameter schema and exact zero
            # coupling gradient while disabling the cross-register phase.
            xxx_factors = xxx_factors + self.raw_coupling[layer_index, head_index].sum() * 0.0
        return xxx_factors.prod(dim=-1) + self.config.mixed_readout_weight * xzz_factors.prod(dim=-1)

    def _pair_observable(
        self,
        query: torch.Tensor,
        relation: torch.Tensor,
        key: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
    ) -> torch.Tensor:
        return self._observable(query, relation, key, layer_index=layer_index, head_index=head_index)

    def _curvature(
        self,
        query: torch.Tensor,
        relation: torch.Tensor,
        key: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
    ) -> torch.Tensor:
        direction = torch.ones_like(relation) / math.sqrt(float(self.head_dim))
        delta = self.config.perturbation
        plus = self._pair_observable(query, relation + delta * direction, key, layer_index=layer_index, head_index=head_index)
        center = self._pair_observable(query, relation, key, layer_index=layer_index, head_index=head_index)
        minus = self._pair_observable(query, relation - delta * direction, key, layer_index=layer_index, head_index=head_index)
        return (plus - 2.0 * center + minus) / (delta * delta)

    def metadata(self) -> dict[str, Any]:
        return {
            "id": f"q_rpec_{self.kernel_type}",
            "version": "0.2.0",
            "type": "relation_perturbation_echo_curvature",
            "insertion_point": "pre_softmax_attention_scores",
            "hypothesis": "mixed Pauli relation-anchor curvature exposes input-dependent relation-key sensitivity beyond static density readout",
            "input_schema": "query,key,attention_mask,subject_mask,object_mask; labels prohibited",
            "output_schema": "finite centered context-only score residual [batch,heads,query,key]",
            "requires": [
                "label-free relation anchor",
                "three-point symmetric echo",
                "mixed XXX plus XZZ Pauli readout",
            ],
            "conflicts": ["q_triad", "qness", "q_wap", "qcdd", "qccw", "stacked_score_interventions"],
            "deterministic": True,
            "resource_estimate": {
                "data_qubits": 3 * self.config.num_qubits,
                "ancilla_qubits": 0,
                "echo_evaluations_per_pair": 3,
                "cross_register_operation": "relation-key controlled phase" if self.uses_cross_register_echo else "none",
                "trainable_parameter_count": self.parameter_count,
            },
            "failure_signatures": [
                "curvature is zero or relation-insensitive",
                "quantum branch exactly replays local control",
                "nonfinite score or gradient",
                "entity/masked key receives action",
                "context residual is not zero-sum",
            ],
            "config": asdict(self.config),
            "execution": {
                "pair_chunk_size": (
                    "all" if self.pair_chunk_size is None else self.pair_chunk_size
                ),
                "pair_chunk_divisor": self.pair_chunk_divisor,
                "activation_checkpointing": self.activation_checkpointing,
            },
            "inference_mode": "label_free",
            "target_input": "query,key,subject_mask,object_mask",
            "key_action_scope": "non_entity_context_only",
            "readout": "symmetric_second_finite_difference_of_mixed_XXX_plus_XZZ_echo",
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
        **_: Any,
    ) -> torch.Tensor:
        del value, scores
        self._validate_inputs(query, key, attention_mask, subject_mask, object_mask)
        if not 0 <= layer_index < self.num_layers:
            raise ValueError("layer_index is outside configured layers")
        batch, heads, query_tokens, _dim = query.shape
        key_tokens = key.shape[2]
        mask = attention_mask.to(device=query.device, dtype=torch.bool)
        subject_weights = subject_mask.to(device=key.device, dtype=key.dtype)
        object_weights = object_mask.to(device=key.device, dtype=key.dtype)
        subject = (key * subject_weights[:, None, :, None]).sum(dim=2) / subject_weights.sum(dim=1).clamp_min(1.0)[:, None, None]
        object_ = (key * object_weights[:, None, :, None]).sum(dim=2) / object_weights.sum(dim=1).clamp_min(1.0)[:, None, None]
        relation = subject - object_
        context = mask & ~(subject_mask.to(device=mask.device, dtype=torch.bool) | object_mask.to(device=mask.device, dtype=torch.bool))
        context_weights = context.to(device=query.device, dtype=query.dtype)
        key_count = context_weights.sum(dim=1).clamp_min(1.0)
        residuals: list[torch.Tensor] = []
        for head_index in range(heads):
            q = query[:, head_index]
            k = key[:, head_index]
            r = relation[:, head_index]
            chunks: list[torch.Tensor] = []
            pair_count = query_tokens * key_tokens
            total_pairs = batch * pair_count
            if self.pair_chunk_size is None:
                chunk_size = (total_pairs + self.pair_chunk_divisor - 1) // self.pair_chunk_divisor
            else:
                chunk_size = self.pair_chunk_size
            chunk_size = max(1, int(chunk_size))
            self.last_total_pairs = int(total_pairs)
            self.last_resolved_pair_chunk_size = int(chunk_size)
            for start in range(0, total_pairs, chunk_size):
                stop = min(start + chunk_size, total_pairs)
                flat = torch.arange(start, stop, device=q.device)
                batch_index = torch.div(flat, pair_count, rounding_mode="floor")
                within_batch = flat.remainder(pair_count)
                query_index = torch.div(within_batch, key_tokens, rounding_mode="floor")
                key_index = within_batch.remainder(key_tokens)
                q_chunk = q[batch_index, query_index]
                r_chunk = r[batch_index]
                k_chunk = k[batch_index, key_index]
                if self.training and self.activation_checkpointing:
                    score_chunk = checkpoint(
                        lambda qv, rv, kv: self._curvature(
                            qv,
                            rv,
                            kv,
                            layer_index=layer_index,
                            head_index=head_index,
                        ),
                        q_chunk,
                        r_chunk,
                        k_chunk,
                        use_reentrant=False,
                    )
                else:
                    score_chunk = self._curvature(
                        q_chunk,
                        r_chunk,
                        k_chunk,
                        layer_index=layer_index,
                        head_index=head_index,
                    )
                chunks.append(score_chunk)
            score = torch.cat(chunks, dim=0).reshape(batch, query_tokens, key_tokens)
            score = score - (score * context_weights[:, None, :]).sum(dim=-1, keepdim=True) / key_count[:, None, None]
            gain = self.max_gain * torch.tanh(self.raw_gains[layer_index, head_index].to(query.device))
            residuals.append(score * gain)
        residual = torch.stack(residuals, dim=1)
        query_mask = mask[:, None, :, None].to(residual.dtype)
        context_mask = context[:, None, None, :].to(residual.dtype)
        return residual * query_mask * context_mask


class LocalRelationEchoCurvatureControl(RelationPerturbationEchoCurvatureKernel):
    """Matched control with identical local encoding and no echo coupling."""

    kernel_type = "local_control"
    uses_cross_register_echo = False


def build_relation_perturbation_echo_curvature(
    mode: str,
    config: RelationPerturbationEchoConfig,
    *,
    pair_chunk_size: int | None = 256,
    pair_chunk_divisor: int = 1,
    activation_checkpointing: bool = True,
) -> RelationPerturbationEchoCurvatureKernel:
    classes = {
        "quantum": RelationPerturbationEchoCurvatureKernel,
        "local_control": LocalRelationEchoCurvatureControl,
    }
    try:
        cls = classes[mode]
    except KeyError as error:
        raise ValueError(f"mode must be one of {Q_RPEC_KERNEL_TYPES}") from error
    return cls(
        config,
        pair_chunk_size=pair_chunk_size,
        pair_chunk_divisor=pair_chunk_divisor,
        activation_checkpointing=activation_checkpointing,
    )
