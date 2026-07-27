"""Identifiable observable-expert routing for relation score kernels."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantum_steering import _data_reuploading_state, _seeded_projection


EXPERT_ROUTER_TYPES = ("quantum", "classical")
EXPERT_DIRECTION_MODES = (
    "fixed",
    "task_aligned",
    "measurement_aligned",
    "connected_aligned",
)
MEASUREMENT_DIRECTION_MODES = ("measurement_aligned", "connected_aligned")
ROUTING_MODES = ("learned", "uniform")
ROUTER_RESIDUAL_REFERENCES = ("core", "baseline")
ROUTER_CONDITIONING_CHOICES = ("relation", "query", "query_expert")
QUERY_ROUTING_CONDITIONING = ("query", "query_expert")


@dataclass(frozen=True)
class RelationExpertRouterConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    num_observables: int
    num_experts: int = 4
    router_qubits: int = 2
    depth: int = 2
    angle_scale: float = 1.0
    max_gain: float = 0.5
    initial_gain: float = 0.05
    residual_reference: str = "core"
    normalize_routed_energy: bool = False
    routing_conditioning: str = "relation"
    trainable_projection: bool = False
    direction_mode: str = "fixed"
    seed: int = 97
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if min(
            self.num_layers,
            self.num_heads,
            self.head_dim,
            self.num_observables,
        ) <= 0:
            raise ValueError("router model dimensions must be positive")
        if self.router_qubits <= 0:
            raise ValueError("router_qubits must be positive")
        if self.num_experts != 2**self.router_qubits:
            raise ValueError("num_experts must equal 2 ** router_qubits")
        if self.num_observables < self.num_experts - 1:
            raise ValueError("num_observables must support distinct zero-mean experts")
        if self.depth <= 0 or self.angle_scale <= 0 or self.max_gain <= 0:
            raise ValueError("depth, angle_scale, and max_gain must be positive")
        if not -self.max_gain < self.initial_gain < self.max_gain:
            raise ValueError("initial_gain must lie inside (-max_gain, max_gain)")
        if self.residual_reference not in ROUTER_RESIDUAL_REFERENCES:
            raise ValueError(
                f"residual_reference must be one of {ROUTER_RESIDUAL_REFERENCES}"
            )
        if self.routing_conditioning not in ROUTER_CONDITIONING_CHOICES:
            raise ValueError(
                f"routing_conditioning must be one of {ROUTER_CONDITIONING_CHOICES}"
            )
        if self.direction_mode not in EXPERT_DIRECTION_MODES:
            raise ValueError(
                f"direction_mode must be one of {EXPERT_DIRECTION_MODES}"
            )
        if self.direction_mode in MEASUREMENT_DIRECTION_MODES and self.num_observables % 2:
            raise ValueError(
                "measurement-based directions require local and pair observables per qubit"
            )
        if self.eps <= 0:
            raise ValueError("eps must be positive")


def _raw_gain(initial_gain: float, max_gain: float, shape: tuple[int, ...]) -> torch.Tensor:
    ratio = torch.tensor(initial_gain / max_gain, dtype=torch.float32)
    return torch.full(shape, float(torch.atanh(ratio).item()), dtype=torch.float32)


def _hadamard(order: int) -> torch.Tensor:
    if order <= 0 or order & (order - 1):
        raise ValueError("Hadamard order must be a positive power of two")
    matrix = torch.ones(1, 1)
    while matrix.shape[0] < order:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return matrix


def _expert_codes(config: RelationExpertRouterConfig) -> torch.Tensor:
    base = _hadamard(config.num_experts)[:, 1:]
    base = F.normalize(base, p=2, dim=-1)
    codes: list[torch.Tensor] = []
    for head_index in range(config.num_heads):
        head = torch.zeros(config.num_experts, config.num_observables)
        indices = (
            torch.arange(config.num_experts - 1) + 2 * head_index
        ).remainder(config.num_observables)
        head[:, indices] = base
        codes.append(head)
    return torch.stack(codes)


class RelationObservableExpertRouter(nn.Module):
    """Route zero-mean observable expert directions from a relation anchor."""

    router_type = "base"

    def __init__(self, config: RelationExpertRouterConfig) -> None:
        super().__init__()
        self.config = config
        if config.routing_conditioning == "relation":
            feature_dim = 4 * config.head_dim
        elif config.routing_conditioning == "query":
            feature_dim = 5 * config.head_dim
        else:
            feature_dim = 5 * config.head_dim + 2 * config.num_experts
        relation_projections = torch.stack(
            [
                _seeded_projection(
                    feature_dim,
                    config.router_qubits,
                    config.seed + 23 * head,
                )
                for head in range(config.num_heads)
            ]
        )
        if config.trainable_projection:
            self.relation_projections = nn.Parameter(relation_projections)
        else:
            self.register_buffer("relation_projections", relation_projections)
        expert_codes = _expert_codes(config)
        self.register_buffer("expert_codes", expert_codes, persistent=False)
        if config.direction_mode == "task_aligned":
            generator = torch.Generator(device="cpu").manual_seed(config.seed + 811)
            initial_directions = expert_codes.unsqueeze(0).expand(
                config.num_layers, -1, -1, -1
            ).clone()
            initial_directions.add_(
                0.01
                * torch.randn(
                    initial_directions.shape,
                    generator=generator,
                    dtype=initial_directions.dtype,
                )
            )
            self.expert_direction_parameters = nn.Parameter(initial_directions)
            self.register_parameter("expert_measurement_angles", None)
        elif config.direction_mode in MEASUREMENT_DIRECTION_MODES:
            self.register_parameter("expert_direction_parameters", None)
            measurement_qubits = config.num_observables // 2
            expert_offsets = torch.arange(config.num_experts).float().unsqueeze(1)
            qubit_offsets = torch.arange(measurement_qubits).float().unsqueeze(0)
            initial_angles = math.pi * (
                expert_offsets / config.num_experts
                + qubit_offsets / (2 * measurement_qubits)
            )
            initial_angles = initial_angles.reshape(
                1, 1, config.num_experts, measurement_qubits
            ).expand(config.num_layers, config.num_heads, -1, -1).clone()
            generator = torch.Generator(device="cpu").manual_seed(config.seed + 811)
            initial_angles.add_(
                0.01
                * torch.randn(
                    initial_angles.shape,
                    generator=generator,
                    dtype=initial_angles.dtype,
                )
            )
            self.expert_measurement_angles = nn.Parameter(initial_angles)
        else:
            self.register_parameter("expert_direction_parameters", None)
            self.register_parameter("expert_measurement_angles", None)
        shape = (
            config.num_layers,
            config.num_heads,
            config.depth,
            config.router_qubits,
        )
        generator = torch.Generator(device="cpu").manual_seed(config.seed + 1009)
        self.router_scales = nn.Parameter(torch.ones(shape))
        self.router_biases = nn.Parameter(
            torch.empty(shape).uniform_(-math.pi / 4, math.pi / 4, generator=generator)
        )
        self.raw_gains = nn.Parameter(
            _raw_gain(
                config.initial_gain,
                config.max_gain,
                (config.num_layers, config.num_heads),
            )
        )
        self._capture_routing = False
        self._captured_probabilities: list[tuple[int, int, torch.Tensor]] = []
        self._captured_probability_masks: list[torch.Tensor | None] = []
        self._captured_expert_deltas: list[tuple[int, int, torch.Tensor]] = []

    @property
    def model_dimensions(self) -> tuple[int, int, int]:
        return self.config.num_layers, self.config.num_heads, self.config.head_dim

    def metadata(self) -> dict[str, Any]:
        return {"type": self.router_type, "config": asdict(self.config)}

    def gains(self, layer_index: int) -> torch.Tensor:
        self._validate_layer(layer_index)
        return self.config.max_gain * torch.tanh(self.raw_gains[layer_index])

    def direction_codes(
        self,
        layer_index: int,
        head_index: int,
    ) -> torch.Tensor:
        """Return centered observable coefficients for one layer and head."""
        self._validate_layer(layer_index)
        if not 0 <= head_index < self.config.num_heads:
            raise ValueError(
                f"head_index {head_index} is outside [0, {self.config.num_heads})"
            )
        if self.expert_direction_parameters is None:
            return self.expert_codes[head_index]
        codes = self.expert_direction_parameters[layer_index, head_index]
        centered = codes - codes.mean(dim=0, keepdim=True)
        target_norm = math.sqrt(self.config.num_experts)
        return centered * (
            target_norm / centered.norm().clamp_min(self.config.eps)
        )

    def direction_diversity_loss(self) -> torch.Tensor:
        """Keep learned observable directions close to a centered simplex."""
        if (
            self.expert_direction_parameters is None
            and self.expert_measurement_angles is None
        ):
            return self.raw_gains.sum() * 0.0
        direction_parameters = (
            self.expert_direction_parameters
            if self.expert_direction_parameters is not None
            else self.expert_measurement_angles
        )
        assert direction_parameters is not None
        target_cosine = -1.0 / (self.config.num_experts - 1)
        off_diagonal = ~torch.eye(
            self.config.num_experts,
            dtype=torch.bool,
            device=direction_parameters.device,
        )
        losses: list[torch.Tensor] = []
        for layer_index in range(self.config.num_layers):
            for head_index in range(self.config.num_heads):
                if self.expert_measurement_angles is None:
                    representations = self.direction_codes(
                        layer_index, head_index
                    )
                else:
                    angles = self.expert_measurement_angles[
                        layer_index, head_index
                    ]
                    representations = torch.cat(
                        (torch.cos(angles), torch.sin(angles)), dim=-1
                    )
                    representations = representations - representations.mean(
                        dim=0, keepdim=True
                    )
                normalized = F.normalize(
                    representations,
                    p=2,
                    dim=-1,
                    eps=self.config.eps,
                )
                cosine = torch.matmul(normalized, normalized.transpose(0, 1))
                losses.append(
                    (cosine.masked_select(off_diagonal) - target_cosine)
                    .square()
                    .mean()
                )
        return torch.stack(losses).mean()

    def measurement_angles(
        self,
        layer_index: int,
        head_index: int,
    ) -> torch.Tensor:
        """Return expert-specific local measurement axes."""
        self._validate_layer(layer_index)
        if self.expert_measurement_angles is None:
            raise RuntimeError(
                "measurement angles require a measurement-based direction mode"
            )
        if not 0 <= head_index < self.config.num_heads:
            raise ValueError(
                f"head_index {head_index} is outside [0, {self.config.num_heads})"
            )
        return self.expert_measurement_angles[layer_index, head_index]

    def _validate_layer(self, layer_index: int) -> None:
        if not 0 <= layer_index < self.config.num_layers:
            raise ValueError(
                f"layer_index {layer_index} is outside [0, {self.config.num_layers})"
            )

    def _router_state(
        self,
        features: torch.Tensor,
        projection: torch.Tensor,
        scales: torch.Tensor,
        biases: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def head_probabilities(
        self,
        relation_anchor: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
        routing_mode: str,
        query_context: torch.Tensor | None = None,
        expert_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_layer(layer_index)
        if routing_mode not in ROUTING_MODES:
            raise ValueError(f"routing_mode must be one of {ROUTING_MODES}")
        if relation_anchor.ndim != 2 or relation_anchor.shape[-1] != 4 * self.config.head_dim:
            raise ValueError("relation_anchor must have shape (batch, 4 * head_dim)")
        router_features = relation_anchor
        if self.config.routing_conditioning in QUERY_ROUTING_CONDITIONING:
            if (
                query_context is None
                or query_context.ndim != 3
                or query_context.shape[0] != relation_anchor.shape[0]
                or query_context.shape[-1] != 5 * self.config.head_dim
            ):
                raise ValueError(
                    "query_context must have shape (batch, tokens, 5 * head_dim)"
                )
            router_features = query_context
        if self.config.routing_conditioning == "query_expert":
            if (
                expert_context is None
                or expert_context.ndim != 3
                or expert_context.shape[:2] != router_features.shape[:2]
                or expert_context.shape[-1] != 2 * self.config.num_experts
            ):
                raise ValueError(
                    "expert_context must have shape (batch, tokens, 2 * num_experts)"
                )
            router_features = torch.cat((router_features, expert_context), dim=-1)
        if routing_mode == "uniform":
            return router_features.new_full(
                (*router_features.shape[:-1], self.config.num_experts),
                1.0 / self.config.num_experts,
            )
        normalized = F.normalize(
            router_features.float(),
            p=2,
            dim=-1,
            eps=self.config.eps,
        )
        state = self._router_state(
            normalized.reshape(-1, normalized.shape[-1]),
            self.relation_projections[head_index],
            self.router_scales[layer_index, head_index],
            self.router_biases[layer_index, head_index],
        )
        probabilities = state.square().reshape(
            *router_features.shape[:-1], self.config.num_experts
        )
        return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
            self.config.eps
        )

    @contextmanager
    def capture_routing(self) -> Iterator[None]:
        if self._capture_routing:
            raise RuntimeError("expert-routing capture is already active")
        self._captured_probabilities.clear()
        self._captured_probability_masks.clear()
        self._captured_expert_deltas.clear()
        self._capture_routing = True
        try:
            yield
        finally:
            self._capture_routing = False
            self._captured_probabilities.clear()
            self._captured_probability_masks.clear()
            self._captured_expert_deltas.clear()

    def captured_probabilities(self) -> tuple[tuple[int, int, torch.Tensor], ...]:
        if not self._capture_routing:
            raise RuntimeError("expert-routing capture is not active")
        return tuple(self._captured_probabilities)

    def captured_probability_masks(self) -> tuple[torch.Tensor | None, ...]:
        if not self._capture_routing:
            raise RuntimeError("expert-routing capture is not active")
        return tuple(self._captured_probability_masks)

    def captured_expert_deltas(self) -> tuple[tuple[int, int, torch.Tensor], ...]:
        if not self._capture_routing:
            raise RuntimeError("expert-routing capture is not active")
        return tuple(self._captured_expert_deltas)

    def information_components(self) -> dict[str, torch.Tensor]:
        """Measure routing information and dead experts per layer and head."""
        if not self._capture_routing:
            raise RuntimeError("routing information loss requires active capture")
        if not self._captured_probabilities:
            zero = self.raw_gains.sum() * 0.0
            return {
                "conditional_entropy": zero,
                "marginal_entropy": zero,
                "mutual_information": zero,
                "dead_expert_barrier": zero,
            }
        conditional_entropies: list[torch.Tensor] = []
        marginal_entropies: list[torch.Tensor] = []
        dead_expert_barriers: list[torch.Tensor] = []
        for (
            _layer_index,
            _head_index,
            captured,
        ), mask in zip(
            self._captured_probabilities,
            self._captured_probability_masks,
            strict=True,
        ):
            if mask is not None:
                captured = captured[mask]
            probabilities = captured.reshape(
                -1, self.config.num_experts
            ).clamp_min(self.config.eps)
            conditional_entropies.append(
                -(probabilities * probabilities.log()).sum(dim=-1).mean()
            )
            marginal = probabilities.mean(dim=0)
            marginal_entropies.append(-(marginal * marginal.log()).sum())
            uniform = torch.full_like(marginal, 1.0 / self.config.num_experts)
            dead_expert_barriers.append(
                torch.sum(uniform * (uniform.log() - marginal.log()))
            )
        conditional_entropy = torch.stack(conditional_entropies).mean()
        marginal_entropy = torch.stack(marginal_entropies).mean()
        dead_expert_barrier = torch.stack(dead_expert_barriers).mean()
        return {
            "conditional_entropy": conditional_entropy,
            "marginal_entropy": marginal_entropy,
            "mutual_information": marginal_entropy - conditional_entropy,
            "dead_expert_barrier": dead_expert_barrier,
        }

    def information_loss(self) -> torch.Tensor:
        """Maximize per-head routing information while preventing dead experts."""
        components = self.information_components()
        return (
            -components["mutual_information"]
            + components["dead_expert_barrier"]
        )

    def route_components(
        self,
        components: torch.Tensor,
        base_weights: torch.Tensor,
        relation_anchor: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
        routing_mode: str,
        query_context: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add a routed zero-mean expert direction to the static core readout."""
        if components.ndim != 4 or components.shape[-1] != self.config.num_observables:
            raise ValueError("components must have shape (batch, query, key, observables)")
        expert_deltas = torch.einsum(
            "bijo,eo->beij",
            components,
            self.direction_codes(layer_index, head_index),
        )
        base = torch.sum(components * base_weights, dim=-1)
        return self.route_expert_deltas(
            expert_deltas,
            base,
            relation_anchor,
            layer_index=layer_index,
            head_index=head_index,
            routing_mode=routing_mode,
            query_context=query_context,
            query_mask=query_mask,
        )

    def route_expert_deltas(
        self,
        expert_deltas: torch.Tensor,
        base: torch.Tensor,
        relation_anchor: torch.Tensor,
        *,
        layer_index: int,
        head_index: int,
        routing_mode: str,
        query_context: torch.Tensor | None = None,
        query_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Route precomputed centered expert score directions."""
        if expert_deltas.ndim != 4:
            raise ValueError(
                "expert_deltas must have shape (batch, experts, query, key)"
            )
        if expert_deltas.shape[1] != self.config.num_experts:
            raise ValueError("expert_deltas expert dimension does not match router")
        if base.shape != (
            expert_deltas.shape[0],
            expert_deltas.shape[2],
            expert_deltas.shape[3],
        ):
            raise ValueError("base must match expert batch, query, and key dimensions")
        if routing_mode == "uniform":
            if self.config.residual_reference == "baseline":
                return torch.zeros_like(base)
            return base
        expert_context = None
        if self.config.routing_conditioning == "query_expert":
            expert_mean = expert_deltas.mean(dim=-1).transpose(1, 2)
            expert_power = expert_deltas.square().mean(dim=-1).transpose(1, 2)
            expert_rms = torch.sqrt(
                expert_power.clamp_min(self.config.eps**2)
            )
            expert_context = torch.cat((expert_mean, expert_rms), dim=-1)
        probabilities = self.head_probabilities(
            relation_anchor,
            layer_index=layer_index,
            head_index=head_index,
            routing_mode=routing_mode,
            query_context=query_context,
            expert_context=expert_context,
        )
        if self._capture_routing:
            if not expert_deltas.requires_grad:
                expert_deltas.requires_grad_(True)
            self._captured_probabilities.append(
                (layer_index, head_index, probabilities)
            )
            self._captured_probability_masks.append(
                query_mask
                if self.config.routing_conditioning in QUERY_ROUTING_CONDITIONING
                else None
            )
            self._captured_expert_deltas.append(
                (layer_index, head_index, expert_deltas)
            )
        if self.config.routing_conditioning in QUERY_ROUTING_CONDITIONING:
            routed = torch.einsum("bie,beij->bij", probabilities, expert_deltas)
        else:
            routed = torch.einsum("be,beij->bij", probabilities, expert_deltas)
        if self.config.residual_reference == "baseline":
            return routed
        return base + self.gains(layer_index)[head_index] * routed


class QuantumRelationObservableExpertRouter(RelationObservableExpertRouter):
    """Entangled Born router over observable expert directions."""

    router_type = "quantum"

    def _router_state(
        self,
        features: torch.Tensor,
        projection: torch.Tensor,
        scales: torch.Tensor,
        biases: torch.Tensor,
    ) -> torch.Tensor:
        return _data_reuploading_state(
            features,
            projection,
            scales,
            biases,
            angle_scale=self.config.angle_scale,
            eps=self.config.eps,
        )


class ClassicalRelationObservableExpertRouter(RelationObservableExpertRouter):
    """Parameter-matched separable product-state routing control."""

    router_type = "classical"

    def _router_state(
        self,
        features: torch.Tensor,
        projection: torch.Tensor,
        scales: torch.Tensor,
        biases: torch.Tensor,
    ) -> torch.Tensor:
        angles = self.config.angle_scale * torch.matmul(features, projection)
        phase = angles
        for depth_index in range(self.config.depth):
            phase = torch.sin(
                phase + angles * scales[depth_index] + biases[depth_index]
            )
        amplitudes = torch.ones(features.shape[0], 1, device=features.device)
        for qubit in range(self.config.router_qubits):
            low = torch.cos(phase[:, qubit]).unsqueeze(-1)
            high = torch.sin(phase[:, qubit]).unsqueeze(-1)
            amplitudes = torch.cat((amplitudes * low, amplitudes * high), dim=-1)
        return F.normalize(amplitudes, p=2, dim=-1, eps=self.config.eps)


def build_relation_expert_router(
    router_type: str,
    config: RelationExpertRouterConfig,
) -> RelationObservableExpertRouter:
    if router_type == "quantum":
        return QuantumRelationObservableExpertRouter(config)
    if router_type == "classical":
        return ClassicalRelationObservableExpertRouter(config)
    raise ValueError(f"router_type must be one of {EXPERT_ROUTER_TYPES}")
