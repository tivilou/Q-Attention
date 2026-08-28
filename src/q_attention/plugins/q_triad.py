"""Q-TRIAD statevector kernel and an explicit classical density control.

The kernel is intentionally small and deterministic.  It is useful for
attention experiments because the relation register is an independent input,
while the classical density expansion exposes the attribution ceiling.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


QTRIAD_CONTROL_MODES = (
    "q_triad",
    "classical_density_tensor",
    "classical_fourier_cp",
    "quantum_product",
    "quantum_random",
)


@dataclass(frozen=True)
class QTriadConfig:
    input_dim: int = 4
    num_qubits: int = 2
    depth: int = 1
    angle_scale: float = 1.0
    control_mode: str = "q_triad"
    seed: int = 131
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.num_qubits <= 0 or self.depth <= 0:
            raise ValueError("input_dim, num_qubits, and depth must be positive")
        if self.num_qubits > 5:
            raise ValueError("num_qubits must be at most five for the statevector kernel")
        if self.angle_scale <= 0.0 or self.eps <= 0.0:
            raise ValueError("angle_scale and eps must be positive")
        if self.control_mode not in QTRIAD_CONTROL_MODES:
            raise ValueError(f"control_mode must be one of {QTRIAD_CONTROL_MODES}")


@dataclass(frozen=True)
class QTriadResult:
    score: torch.Tensor
    query_features: torch.Tensor
    relation_features: torch.Tensor
    key_features: torch.Tensor
    gamma: torch.Tensor
    diagnostics: dict[str, Any]


def _seeded_projection(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(input_dim, output_dim, generator=generator)
    return projection / math.sqrt(float(input_dim))


def _tensor_product(states: list[torch.Tensor]) -> torch.Tensor:
    result = states[0]
    for state in states[1:]:
        result = (result.unsqueeze(-1) * state.unsqueeze(-2)).reshape(
            result.shape[0], -1
        )
    return result


def _product_state(local_states: torch.Tensor) -> torch.Tensor:
    state = torch.ones(
        local_states.shape[0],
        1,
        device=local_states.device,
        dtype=local_states.dtype,
    )
    for qubit in range(local_states.shape[1]):
        state = (state.unsqueeze(-1) * local_states[:, qubit].unsqueeze(-2)).reshape(
            local_states.shape[0], -1
        )
    return F.normalize(state, p=2, dim=-1)


def _pauli_features(theta: torch.Tensor, phase: float) -> torch.Tensor:
    sine = torch.sin(theta)
    return torch.stack(
        (
            torch.ones_like(theta),
            sine * math.cos(phase),
            sine * math.sin(phase),
            torch.cos(theta),
        ),
        dim=-1,
    )


def _apply_pauli_x_all(state: torch.Tensor) -> torch.Tensor:
    return state.flip(dims=(-1,))


class QTriadKernel(nn.Module):
    """Three-register Q-R-K interaction with an exact density expansion."""

    plugin_type = "q_triad"

    def __init__(self, config: QTriadConfig) -> None:
        super().__init__()
        self.config = config
        projections = torch.stack(
            [
                _seeded_projection(
                    config.input_dim,
                    config.num_qubits,
                    config.seed + 101 * role,
                )
                for role in range(3)
            ]
        )
        relation_projection = _seeded_projection(
            config.input_dim, config.num_qubits, config.seed + 701
        )
        random_relation_projection = _seeded_projection(
            config.input_dim, config.num_qubits, config.seed + 1701
        )
        self.register_buffer("input_projections", projections)
        self.register_buffer("relation_projection", relation_projection)
        self.register_buffer("random_relation_projection", random_relation_projection)

        shape = (config.depth, 3, config.num_qubits)
        generator = torch.Generator(device="cpu").manual_seed(config.seed + 2001)
        self.theta_scales = nn.Parameter(torch.ones(shape))
        self.theta_biases = nn.Parameter(
            torch.empty(shape).uniform_(-math.pi / 4, math.pi / 4, generator=generator)
        )
        self.gamma_scales = nn.Parameter(torch.ones(config.depth, config.num_qubits))
        self._phase_offsets = (math.pi / 4, math.pi / 4, math.pi / 4)
        self._state_dim = 2 ** (3 * config.num_qubits)
        self.register_buffer("phase_signs", self._make_phase_signs(), persistent=False)

    def _make_phase_signs(self) -> torch.Tensor:
        signs = []
        total_qubits = 3 * self.config.num_qubits
        for basis_index in range(self._state_dim):
            bits = [
                1.0 if (basis_index & (1 << (total_qubits - 1 - qubit))) == 0 else -1.0
                for qubit in range(total_qubits)
            ]
            signs.append(
                [
                    bits[index]
                    * bits[self.config.num_qubits + index]
                    * bits[2 * self.config.num_qubits + index]
                    for index in range(self.config.num_qubits)
                ]
            )
        return torch.tensor(signs, dtype=torch.float32)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def metadata(self) -> dict[str, Any]:
        return {
            "type": self.plugin_type,
            "config": asdict(self.config),
            "parameter_count": self.parameter_count,
            "num_qubits": 3 * self.config.num_qubits,
            "state_dim": self._state_dim,
            "interaction_order": 3,
            "classical_expansion_rank_upper_bound": 2 ** self.config.num_qubits,
            "readout": "global_XXX",
        }

    def _role_features(
        self,
        value: torch.Tensor,
        role: int,
        layer_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = torch.matmul(
            value.float(), self.input_projections[role].to(device=value.device)
        )
        theta = self.config.angle_scale * (
            projected * self.theta_scales[layer_index, role]
            + self.theta_biases[layer_index, role]
        )
        features = _pauli_features(theta, self._phase_offsets[role])
        amplitudes = torch.stack(
            (
                torch.cos(theta / 2),
                torch.exp(
                    torch.complex(
                        torch.zeros_like(theta),
                        torch.full_like(theta, self._phase_offsets[role]),
                    )
                )
                * torch.sin(theta / 2),
            ),
            dim=-1,
        )
        return features, _product_state(amplitudes)

    def _gamma(self, relation: torch.Tensor, layer_index: int) -> torch.Tensor:
        projection = (
            self.random_relation_projection
            if self.config.control_mode == "quantum_random"
            else self.relation_projection
        )
        projected = torch.matmul(relation.float(), projection.to(device=relation.device))
        return self.config.angle_scale * torch.tanh(
            projected * self.gamma_scales[layer_index]
        )

    def _classical_expansion(
        self,
        query_features: torch.Tensor,
        relation_features: torch.Tensor,
        key_features: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        score = torch.zeros(query_features.shape[0], device=query_features.device)
        for selected in itertools.product((0, 1), repeat=self.config.num_qubits):
            coefficient = torch.ones_like(gamma[..., 0])
            query_factor = torch.ones_like(coefficient)
            relation_factor = torch.ones_like(coefficient)
            key_factor = torch.ones_like(coefficient)
            for qubit, use_y in enumerate(selected):
                angle_index = 2 if use_y else 1
                coefficient = coefficient * (
                    torch.sin(gamma[..., qubit])
                    if use_y
                    else torch.cos(gamma[..., qubit])
                )
                query_factor = query_factor * query_features[:, qubit, angle_index]
                relation_factor = relation_factor * relation_features[:, qubit, angle_index]
                key_factor = key_factor * key_features[:, qubit, angle_index]
            score = score + coefficient * query_factor * relation_factor * key_factor
        return score

    def _quantum_score(
        self,
        query_states: torch.Tensor,
        relation_states: torch.Tensor,
        key_states: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        state = _tensor_product([query_states, relation_states, key_states])
        signs = self.phase_signs.to(device=state.device)
        for qubit in range(self.config.num_qubits):
            phase = torch.exp(
                torch.complex(
                    torch.zeros_like(gamma[:, qubit]),
                    -0.5 * gamma[:, qubit],
                )[:, None]
                * signs[:, qubit][None, :]
            )
            state = state * phase
        return torch.sum(state.conj() * _apply_pauli_x_all(state), dim=-1).real

    def forward(
        self,
        query: torch.Tensor,
        relation: torch.Tensor,
        key: torch.Tensor,
        *,
        layer_index: int | None = None,
    ) -> QTriadResult:
        if query.ndim != 2 or relation.shape != query.shape or key.shape != query.shape:
            raise ValueError("query, relation, and key must share shape (batch, input_dim)")
        if query.shape[-1] != self.config.input_dim:
            raise ValueError("input feature dimension does not match config")
        layers = range(self.config.depth) if layer_index is None else (layer_index,)
        scores = []
        latest: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        for index in layers:
            if not 0 <= index < self.config.depth:
                raise ValueError("layer_index is outside configured depth")
            query_features, query_states = self._role_features(query, 0, index)
            relation_features, relation_states = self._role_features(relation, 1, index)
            key_features, key_states = self._role_features(key, 2, index)
            gamma = self._gamma(relation, index)
            if self.config.control_mode == "quantum_product":
                gamma = torch.zeros_like(gamma)
            if self.config.control_mode in ("classical_density_tensor", "classical_fourier_cp"):
                score = self._classical_expansion(
                    query_features, relation_features, key_features, gamma
                )
            else:
                score = self._quantum_score(
                    query_states, relation_states, key_states, gamma
                )
            scores.append(score)
            latest = query_features, relation_features, key_features, gamma
        assert latest is not None
        query_features, relation_features, key_features, gamma = latest
        return QTriadResult(
            score=torch.stack(scores, dim=0).mean(dim=0),
            query_features=query_features,
            relation_features=relation_features,
            key_features=key_features,
            gamma=gamma,
            diagnostics={
                "parameter_count": self.parameter_count,
                "state_dim": self._state_dim,
                "num_qubits": 3 * self.config.num_qubits,
                "interaction_order": 3,
                "cp_expansion_rank": 2 ** self.config.num_qubits,
            },
        )


class QTriadAttentionScoreKernel(nn.Module):
    """Q-TRIAD score intervention for the relation Transformer hook.

    The relation input is a signed subject/object anchor computed from frozen
    key projections.  Labels and relation IDs never enter the forward path.
    """

    kernel_type = "q_triad"

    def __init__(
        self,
        *,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        num_qubits: int = 2,
        circuit_depth: int = 2,
        angle_scale: float = 1.0,
        max_gain: float = 0.5,
        initial_gain: float = 0.02,
        seed: int = 131,
        control_mode: str = "q_triad",
        eps: float = 1e-8,
        pair_chunk_size: int = 256,
        activation_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if num_layers <= 0 or num_heads <= 0 or head_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if not -max_gain < initial_gain < max_gain or max_gain <= 0.0:
            raise ValueError("initial_gain must lie inside (-max_gain, max_gain)")
        if control_mode not in {"q_triad", "classical_density_tensor", "quantum_product", "quantum_random"}:
            raise ValueError("unsupported Q-TRIAD attention control mode")
        if pair_chunk_size <= 0:
            raise ValueError("pair_chunk_size must be positive")
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_gain = max_gain
        self.eps = eps
        self.control_mode = control_mode
        self.circuit_depth = circuit_depth
        self.pair_chunk_size = pair_chunk_size
        self.activation_checkpointing = bool(activation_checkpointing)
        self.kernels = nn.ModuleList(
            [
                QTriadKernel(
                    QTriadConfig(
                        input_dim=head_dim,
                        num_qubits=num_qubits,
                        depth=circuit_depth,
                        angle_scale=angle_scale,
                        control_mode=control_mode,
                        seed=seed + 1009 * layer + 31 * head,
                        eps=eps,
                    )
                )
                for layer in range(num_layers)
                for head in range(num_heads)
            ]
        )
        ratio = torch.tensor(initial_gain / max_gain, dtype=torch.float32)
        raw = torch.full((num_layers, num_heads), float(torch.atanh(ratio).item()))
        self.raw_gains = nn.Parameter(raw)

    @property
    def model_dimensions(self) -> tuple[int, int, int]:
        return self.num_layers, self.num_heads, self.head_dim

    def _kernel(self, layer_index: int, head_index: int) -> QTriadKernel:
        return self.kernels[layer_index * self.num_heads + head_index]

    def metadata(self) -> dict[str, Any]:
        return {
            "type": self.kernel_type,
            "control_mode": self.control_mode,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "circuit_depth": self.circuit_depth,
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters()),
            "relation_anchor": "subject_mean_minus_object_mean",
            "inference_mode": "label_free",
            "target_input": "query,key,subject_mask,object_mask",
            "readout": "centered_pre_softmax_score_residual",
            "key_action_scope": "non_entity_context_only",
            "pair_chunk_size": self.pair_chunk_size,
            "activation_checkpointing": self.activation_checkpointing,
        }

    def _score_pairs(
        self,
        kernel: QTriadKernel,
        query: torch.Tensor,
        relation: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        """Score query-key pairs in bounded chunks.

        The old implementation materialized three batch*query*key tensors
        before constructing the statevector. Indexing one chunk at a time
        avoids those large expanded copies. During training, checkpoint each
        chunk so statevector intermediates are recomputed during backward
        instead of being retained for every pair in the attention matrix.
        """
        batch, query_tokens, _dim = query.shape
        key_tokens = key.shape[1]
        total_pairs = batch * query_tokens * key_tokens
        parameter_inputs = tuple(
            parameter for parameter in kernel.parameters() if parameter.requires_grad
        )
        use_checkpoint = self.activation_checkpointing and torch.is_grad_enabled() and bool(parameter_inputs)

        def score_chunk(
            chunk_query: torch.Tensor,
            chunk_relation: torch.Tensor,
            chunk_key: torch.Tensor,
            *_parameters: torch.Tensor,
        ) -> torch.Tensor:
            del _parameters
            return kernel(chunk_query, chunk_relation, chunk_key).score

        chunks: list[torch.Tensor] = []
        pairs_per_batch = query_tokens * key_tokens
        for start in range(0, total_pairs, self.pair_chunk_size):
            stop = min(start + self.pair_chunk_size, total_pairs)
            flat_index = torch.arange(start, stop, device=query.device)
            batch_index = torch.div(flat_index, pairs_per_batch, rounding_mode="floor")
            remainder = flat_index.remainder(pairs_per_batch)
            query_index = torch.div(remainder, key_tokens, rounding_mode="floor")
            key_index = remainder.remainder(key_tokens)
            chunk_query = query[batch_index, query_index]
            chunk_relation = relation[batch_index]
            chunk_key = key[batch_index, key_index]
            if use_checkpoint:
                chunks.append(
                    checkpoint(
                        score_chunk,
                        chunk_query,
                        chunk_relation,
                        chunk_key,
                        *parameter_inputs,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                )
            else:
                chunks.append(score_chunk(chunk_query, chunk_relation, chunk_key))
        return torch.cat(chunks, dim=0).reshape(batch, query_tokens, key_tokens)

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
        if query.ndim != 4 or key.shape != query.shape:
            raise ValueError("query and key must have shape (batch, heads, tokens, head_dim)")
        if query.shape[1] != self.num_heads or query.shape[-1] != self.head_dim:
            raise ValueError("query shape does not match configured model dimensions")
        if not 0 <= layer_index < self.num_layers:
            raise ValueError("layer_index is outside configured layers")
        batch, heads, query_tokens, dim = query.shape
        key_tokens = key.shape[2]
        mask = attention_mask.to(device=query.device, dtype=torch.bool)
        if mask.shape != (batch, key_tokens):
            raise ValueError("attention_mask must match batch and key tokens")
        if subject_mask.shape != mask.shape or object_mask.shape != mask.shape:
            raise ValueError("subject/object masks must match attention_mask")
        subject_weights = subject_mask.to(device=key.device, dtype=key.dtype)
        object_weights = object_mask.to(device=key.device, dtype=key.dtype)
        subject = (key * subject_weights[:, None, :, None]).sum(dim=2) / subject_weights.sum(dim=1).clamp_min(1.0)[:, None, None]
        object_ = (key * object_weights[:, None, :, None]).sum(dim=2) / object_weights.sum(dim=1).clamp_min(1.0)[:, None, None]
        relation = subject - object_
        active_queries = mask[:, None, :, None]
        context_mask = mask & ~(subject_mask.to(device=mask.device, dtype=torch.bool) | object_mask.to(device=mask.device, dtype=torch.bool))
        context_key_mask = context_mask[:, None, None, :]
        context_weights = context_mask.to(device=query.device, dtype=query.dtype)
        key_count = context_weights.sum(dim=1).clamp_min(1.0)
        residuals: list[torch.Tensor] = []
        for head_index in range(heads):
            q = query[:, head_index, :, :]
            r = relation[:, head_index, :]
            k = key[:, head_index, :, :]
            score = self._score_pairs(self._kernel(layer_index, head_index), q, r, k)
            score = score - (score * context_weights[:, None, :]).sum(dim=-1, keepdim=True) / key_count[:, None, None]
            gain = self.max_gain * torch.tanh(self.raw_gains[layer_index, head_index])
            residuals.append(score * gain)
        residual = torch.stack(residuals, dim=1)
        return residual * active_queries * context_key_mask


def build_qtriad(mode: str, config: QTriadConfig | None = None) -> QTriadKernel:
    if mode not in QTRIAD_CONTROL_MODES:
        raise ValueError(f"mode must be one of {QTRIAD_CONTROL_MODES}")
    config = config or QTriadConfig(control_mode=mode)
    if config.control_mode != mode:
        raise ValueError("mode and config.control_mode must match")
    return QTriadKernel(config)
