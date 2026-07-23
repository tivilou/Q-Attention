"""Composable quantum steering plugins for attention key projections."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


PLUGIN_NAMES = ("headwise_projector", "evidence_gate", "expert_bank")


@dataclass(frozen=True)
class QuantumSteeringContext:
    """Runtime tensors exposed to every quantum steering plugin."""

    keys: torch.Tensor
    layer_index: int
    attention_mask: torch.Tensor | None = None
    steering_mask: torch.Tensor | None = None
    subject_mask: torch.Tensor | None = None
    object_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class SteeringContribution:
    """A plugin contribution: an operator delta, a gate, or both."""

    plugin_name: str
    delta: torch.Tensor | None = None
    gate: torch.Tensor | None = None


@dataclass(frozen=True)
class HeadwiseQuantumProjectorConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    depth: int = 2
    rank: int = 4
    max_gain: float = 0.5
    initial_gain: float = 0.05
    seed: int = 31

    def __post_init__(self) -> None:
        _validate_dimensions(
            self.num_layers,
            self.num_heads,
            self.head_dim,
            require_power_of_two=True,
        )
        if self.depth <= 0:
            raise ValueError("depth must be positive")
        if not 0 < self.rank <= self.head_dim:
            raise ValueError("rank must be between 1 and head_dim")
        _validate_gain(self.initial_gain, self.max_gain)


@dataclass(frozen=True)
class QuantumEvidenceGateConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    num_qubits: int = 4
    depth: int = 2
    angle_scale: float = 1.0
    seed: int = 37
    eps: float = 1e-8

    def __post_init__(self) -> None:
        _validate_dimensions(self.num_layers, self.num_heads, self.head_dim)
        if self.num_qubits <= 0 or self.depth <= 0:
            raise ValueError("num_qubits and depth must be positive")
        if self.angle_scale <= 0:
            raise ValueError("angle_scale must be positive")


@dataclass(frozen=True)
class QuantumExpertBankConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    num_experts: int = 4
    projector_depth: int = 2
    router_qubits: int = 4
    router_depth: int = 2
    rank: int = 4
    angle_scale: float = 1.0
    max_gain: float = 0.5
    initial_gain: float = 0.05
    seed: int = 41
    eps: float = 1e-8

    def __post_init__(self) -> None:
        _validate_dimensions(
            self.num_layers,
            self.num_heads,
            self.head_dim,
            require_power_of_two=True,
        )
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.projector_depth <= 0 or self.router_depth <= 0 or self.router_qubits <= 0:
            raise ValueError("projector and router dimensions must be positive")
        if self.num_experts > 2**self.router_qubits:
            raise ValueError("num_experts cannot exceed the router state dimension")
        if not 0 < self.rank <= self.head_dim:
            raise ValueError("rank must be between 1 and head_dim")
        if self.angle_scale <= 0:
            raise ValueError("angle_scale must be positive")
        _validate_gain(self.initial_gain, self.max_gain)


def _validate_dimensions(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    *,
    require_power_of_two: bool = False,
) -> None:
    if num_layers <= 0 or num_heads <= 0 or head_dim <= 0:
        raise ValueError("num_layers, num_heads, and head_dim must be positive")
    if require_power_of_two and head_dim & (head_dim - 1):
        raise ValueError("head_dim must be a power of two for an exact quantum projector")


def _validate_gain(initial_gain: float, max_gain: float) -> None:
    if max_gain <= 0:
        raise ValueError("max_gain must be positive")
    if not -max_gain < initial_gain < max_gain:
        raise ValueError("initial_gain must lie strictly inside (-max_gain, max_gain)")


def _num_qubits(state_dim: int) -> int:
    return int(math.log2(state_dim))


def _raw_gain(initial_gain: float, max_gain: float, shape: tuple[int, ...]) -> torch.Tensor:
    ratio = torch.tensor(initial_gain / max_gain, dtype=torch.float32)
    return torch.full(shape, float(torch.atanh(ratio).item()), dtype=torch.float32)


def _seeded_projection(input_dim: int, output_dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(input_dim, output_dim, generator=generator, dtype=torch.float32)
    return projection / math.sqrt(float(input_dim))


def _apply_ry(state: torch.Tensor, angles: torch.Tensor, qubit: int, num_qubits: int) -> torch.Tensor:
    view = state.reshape(state.shape[0], 2**qubit, 2, 2 ** (num_qubits - qubit - 1))
    low = view[:, :, 0, :]
    high = view[:, :, 1, :]
    cosine = torch.cos(angles / 2).view(-1, 1, 1)
    sine = torch.sin(angles / 2).view(-1, 1, 1)
    return torch.stack(
        (cosine * low - sine * high, sine * low + cosine * high),
        dim=2,
    ).reshape_as(state)


def _apply_cnot(state: torch.Tensor, control: int, target: int, num_qubits: int) -> torch.Tensor:
    indices = torch.arange(2**num_qubits, device=state.device)
    control_mask = 1 << (num_qubits - control - 1)
    target_mask = 1 << (num_qubits - target - 1)
    permutation = torch.where((indices & control_mask) != 0, indices ^ target_mask, indices)
    return state[:, permutation]


def _entangle_ring(state: torch.Tensor, num_qubits: int) -> torch.Tensor:
    if num_qubits <= 1:
        return state
    for control in range(num_qubits):
        state = _apply_cnot(state, control, (control + 1) % num_qubits, num_qubits)
    return state


def _product_state(angles: torch.Tensor) -> torch.Tensor:
    state = torch.ones(angles.shape[0], 1, device=angles.device, dtype=angles.dtype)
    for qubit in range(angles.shape[1]):
        local = torch.stack(
            (torch.cos(angles[:, qubit] / 2), torch.sin(angles[:, qubit] / 2)),
            dim=-1,
        )
        state = (state.unsqueeze(-1) * local.unsqueeze(1)).reshape(angles.shape[0], -1)
    return state


def _data_reuploading_state(
    features: torch.Tensor,
    projection: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor,
    *,
    angle_scale: float,
    eps: float,
) -> torch.Tensor:
    if features.ndim != 2:
        raise ValueError("quantum circuit features must be a matrix")
    num_qubits = projection.shape[1]
    normalized = F.normalize(features.float(), p=2, dim=-1, eps=eps)
    angles = angle_scale * torch.matmul(normalized, projection.to(features.device))
    state = _product_state(angles)
    for depth_index in range(scales.shape[0]):
        layer_angles = angles * scales[depth_index] + biases[depth_index]
        for qubit in range(num_qubits):
            state = _apply_ry(state, layer_angles[:, qubit], qubit, num_qubits)
        state = _entangle_ring(state, num_qubits)
    return F.normalize(state, p=2, dim=-1, eps=eps)


def _unitary_projector(angles: torch.Tensor, rank: int) -> torch.Tensor:
    if angles.ndim != 2:
        raise ValueError("unitary circuit angles must have shape (depth, num_qubits)")
    num_qubits = angles.shape[1]
    state_dim = 2**num_qubits
    evolved_basis = torch.eye(state_dim, device=angles.device, dtype=angles.dtype)
    for depth_index in range(angles.shape[0]):
        for qubit in range(num_qubits):
            shared_angle = angles[depth_index, qubit].expand(state_dim)
            evolved_basis = _apply_ry(evolved_basis, shared_angle, qubit, num_qubits)
        evolved_basis = _entangle_ring(evolved_basis, num_qubits)
    basis = evolved_basis[:rank].transpose(0, 1)
    projector = torch.matmul(basis, basis.transpose(0, 1))
    return 0.5 * (projector + projector.transpose(0, 1))


def _split_heads(keys: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    if keys.ndim != 3:
        raise ValueError("keys must have shape (batch, tokens, dim)")
    if keys.shape[-1] != num_heads * head_dim:
        raise ValueError(
            f"key dim {keys.shape[-1]} does not match num_heads * head_dim={num_heads * head_dim}"
        )
    return keys.reshape(keys.shape[0], keys.shape[1], num_heads, head_dim)


def _validate_mask(mask: torch.Tensor | None, keys: torch.Tensor, name: str) -> None:
    if mask is not None and mask.shape != keys.shape[:2]:
        raise ValueError(f"{name} shape {mask.shape} must match key batch/token shape {keys.shape[:2]}")


def _validate_context(
    context: QuantumSteeringContext,
    *,
    num_layers: int,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    if not 0 <= context.layer_index < num_layers:
        raise ValueError(f"layer_index {context.layer_index} is outside [0, {num_layers})")
    _validate_mask(context.attention_mask, context.keys, "attention_mask")
    _validate_mask(context.steering_mask, context.keys, "steering_mask")
    _validate_mask(context.subject_mask, context.keys, "subject_mask")
    _validate_mask(context.object_mask, context.keys, "object_mask")
    return _split_heads(context.keys, num_heads, head_dim)


def _masked_head_mean(keys: torch.Tensor, mask: torch.Tensor | None, name: str) -> torch.Tensor:
    if mask is None:
        raise ValueError(f"{name} is required for relation-conditioned quantum plugins")
    weights = mask.to(device=keys.device, dtype=keys.dtype)
    denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return torch.sum(keys * weights[:, :, None, None], dim=1) / denominator[:, :, None]


def _relation_pair_features(
    keys: torch.Tensor,
    subject_mask: torch.Tensor | None,
    object_mask: torch.Tensor | None,
) -> torch.Tensor:
    subject = _masked_head_mean(keys, subject_mask, "subject_mask")
    object_ = _masked_head_mean(keys, object_mask, "object_mask")
    return torch.cat((subject, object_, subject - object_, subject * object_), dim=-1)


def _token_evidence_features(
    keys: torch.Tensor,
    subject_mask: torch.Tensor | None,
    object_mask: torch.Tensor | None,
) -> torch.Tensor:
    subject = _masked_head_mean(keys, subject_mask, "subject_mask")[:, None, :, :]
    object_ = _masked_head_mean(keys, object_mask, "object_mask")[:, None, :, :]
    subject = subject.expand(-1, keys.shape[1], -1, -1)
    object_ = object_.expand(-1, keys.shape[1], -1, -1)
    return torch.cat(
        (
            keys,
            subject,
            object_,
            keys - subject,
            keys - object_,
            subject - object_,
            subject * object_,
            keys * subject,
            keys * object_,
        ),
        dim=-1,
    )


class QuantumSteeringPlugin(nn.Module):
    """Base class for independently trainable steering plugins."""

    plugin_name = "base"

    def config_dict(self) -> dict[str, Any]:
        config = getattr(self, "config", None)
        return asdict(config) if config is not None else {}

    def forward(self, context: QuantumSteeringContext) -> SteeringContribution:
        raise NotImplementedError


class HeadwiseQuantumProjectorPlugin(QuantumSteeringPlugin):
    """Exact per-head projectors generated by real-amplitude quantum unitaries."""

    plugin_name = "headwise_projector"

    def __init__(self, config: HeadwiseQuantumProjectorConfig) -> None:
        super().__init__()
        self.config = config
        num_qubits = _num_qubits(config.head_dim)
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        angles = 0.05 * torch.randn(
            config.num_layers,
            config.num_heads,
            config.depth,
            num_qubits,
            generator=generator,
        )
        self.angles = nn.Parameter(angles)
        self.raw_gains = nn.Parameter(
            _raw_gain(config.initial_gain, config.max_gain, (config.num_layers, config.num_heads))
        )

    def projectors(self, layer_index: int) -> torch.Tensor:
        return torch.stack(
            [
                _unitary_projector(self.angles[layer_index, head_index], self.config.rank)
                for head_index in range(self.config.num_heads)
            ],
            dim=0,
        )

    def forward(self, context: QuantumSteeringContext) -> SteeringContribution:
        keys = _validate_context(
            context,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
        )
        projectors = self.projectors(context.layer_index)
        projected = torch.einsum("bthd,hde->bthe", keys, projectors)
        gains = self.config.max_gain * torch.tanh(self.raw_gains[context.layer_index])
        delta = projected * gains.view(1, 1, -1, 1)
        return SteeringContribution(self.plugin_name, delta=delta.reshape_as(context.keys))


class QuantumEvidenceGatePlugin(QuantumSteeringPlugin):
    """Token-level signed evidence gates measured from a quantum circuit."""

    plugin_name = "evidence_gate"

    def __init__(self, config: QuantumEvidenceGateConfig) -> None:
        super().__init__()
        self.config = config
        feature_dim = 9 * config.head_dim
        self.register_buffer(
            "feature_projection",
            _seeded_projection(feature_dim, config.num_qubits, config.seed),
        )
        generator = torch.Generator(device="cpu").manual_seed(config.seed + 1)
        self.data_scales = nn.Parameter(
            torch.ones(config.num_layers, config.num_heads, config.depth, config.num_qubits)
        )
        self.biases = nn.Parameter(
            0.05
            * torch.randn(
                config.num_layers,
                config.num_heads,
                config.depth,
                config.num_qubits,
                generator=generator,
            )
        )
        indices = torch.arange(2**config.num_qubits)
        first_qubit = 1 << (config.num_qubits - 1)
        observable = torch.where((indices & first_qubit) == 0, 1.0, -1.0)
        self.register_buffer("z_observable", observable)

    def gates(self, context: QuantumSteeringContext) -> torch.Tensor:
        keys = _validate_context(
            context,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
        )
        features = _token_evidence_features(keys, context.subject_mask, context.object_mask)
        batch, tokens = context.keys.shape[:2]
        head_gates: list[torch.Tensor] = []
        for head_index in range(self.config.num_heads):
            states = _data_reuploading_state(
                features[:, :, head_index, :].reshape(batch * tokens, -1),
                self.feature_projection,
                self.data_scales[context.layer_index, head_index],
                self.biases[context.layer_index, head_index],
                angle_scale=self.config.angle_scale,
                eps=self.config.eps,
            )
            expectation = torch.matmul(states.square(), self.z_observable.to(states.device))
            head_gates.append(expectation.reshape(batch, tokens))
        gates = torch.stack(head_gates, dim=2)
        return gates.unsqueeze(-1).expand(-1, -1, -1, self.config.head_dim).reshape_as(context.keys)

    def forward(self, context: QuantumSteeringContext) -> SteeringContribution:
        return SteeringContribution(self.plugin_name, gate=self.gates(context))


class QuantumExpertBankPlugin(QuantumSteeringPlugin):
    """Relation-conditioned routing over a bank of exact quantum projectors."""

    plugin_name = "expert_bank"

    def __init__(self, config: QuantumExpertBankConfig) -> None:
        super().__init__()
        self.config = config
        projector_qubits = _num_qubits(config.head_dim)
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        self.expert_angles = nn.Parameter(
            0.05
            * torch.randn(
                config.num_layers,
                config.num_experts,
                config.num_heads,
                config.projector_depth,
                projector_qubits,
                generator=generator,
            )
        )
        self.register_buffer(
            "router_projection",
            _seeded_projection(4 * config.head_dim, config.router_qubits, config.seed + 1),
        )
        self.router_scales = nn.Parameter(
            torch.ones(
                config.num_layers,
                config.num_heads,
                config.router_depth,
                config.router_qubits,
            )
        )
        self.router_biases = nn.Parameter(
            0.05
            * torch.randn(
                config.num_layers,
                config.num_heads,
                config.router_depth,
                config.router_qubits,
                generator=generator,
            )
        )
        self.raw_gains = nn.Parameter(
            _raw_gain(config.initial_gain, config.max_gain, (config.num_layers, config.num_heads))
        )

    def expert_projectors(self, layer_index: int) -> torch.Tensor:
        return torch.stack(
            [
                torch.stack(
                    [
                        _unitary_projector(
                            self.expert_angles[layer_index, expert_index, head_index],
                            self.config.rank,
                        )
                        for head_index in range(self.config.num_heads)
                    ],
                    dim=0,
                )
                for expert_index in range(self.config.num_experts)
            ],
            dim=0,
        )

    def routing_weights(self, context: QuantumSteeringContext) -> torch.Tensor:
        keys = _validate_context(
            context,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
        )
        pair_features = _relation_pair_features(keys, context.subject_mask, context.object_mask)
        weights: list[torch.Tensor] = []
        for head_index in range(self.config.num_heads):
            states = _data_reuploading_state(
                pair_features[:, head_index, :],
                self.router_projection,
                self.router_scales[context.layer_index, head_index],
                self.router_biases[context.layer_index, head_index],
                angle_scale=self.config.angle_scale,
                eps=self.config.eps,
            )
            probabilities = states.square()[:, : self.config.num_experts] + self.config.eps
            weights.append(probabilities / probabilities.sum(dim=-1, keepdim=True))
        return torch.stack(weights, dim=1)

    def forward(self, context: QuantumSteeringContext) -> SteeringContribution:
        keys = _validate_context(
            context,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
        )
        weights = self.routing_weights(context)
        projectors = self.expert_projectors(context.layer_index)
        dynamic_projectors = torch.einsum("bhe,ehij->bhij", weights, projectors)
        projected = torch.einsum("bthd,bhde->bthe", keys, dynamic_projectors)
        gains = self.config.max_gain * torch.tanh(self.raw_gains[context.layer_index])
        delta = projected * gains.view(1, 1, -1, 1)
        return SteeringContribution(self.plugin_name, delta=delta.reshape_as(context.keys))


class ComposableQuantumSteering(nn.Module):
    """Compose any subset of operator and evidence-gate plugins."""

    def __init__(
        self,
        plugins: Sequence[QuantumSteeringPlugin] = (),
        *,
        operator_reduction: str = "mean",
        identity_gain: float = 0.05,
    ) -> None:
        super().__init__()
        if operator_reduction not in {"sum", "mean"}:
            raise ValueError("operator_reduction must be 'sum' or 'mean'")
        if not math.isfinite(identity_gain):
            raise ValueError("identity_gain must be finite")
        names = [plugin.plugin_name for plugin in plugins]
        if len(set(names)) != len(names):
            raise ValueError("quantum plugin names must be unique within a stack")
        dimensions = {
            (
                plugin.config.num_layers,
                plugin.config.num_heads,
                plugin.config.head_dim,
            )
            for plugin in plugins
        }
        if len(dimensions) > 1:
            raise ValueError("all quantum plugins must use the same model dimensions")
        self.plugins = nn.ModuleList(plugins)
        self.operator_reduction = operator_reduction
        self.identity_gain = float(identity_gain)
        self.model_dimensions = next(iter(dimensions)) if dimensions else None

    @property
    def active_plugin_names(self) -> tuple[str, ...]:
        return tuple(plugin.plugin_name for plugin in self.plugins)

    def metadata(self) -> dict[str, Any]:
        return {
            "plugins": [
                {"name": plugin.plugin_name, "config": plugin.config_dict()}
                for plugin in self.plugins
            ],
            "operator_reduction": self.operator_reduction,
            "identity_gain": self.identity_gain,
        }

    def forward(
        self,
        keys: torch.Tensor,
        *,
        layer_index: int,
        attention_mask: torch.Tensor | None = None,
        steering_mask: torch.Tensor | None = None,
        subject_mask: torch.Tensor | None = None,
        object_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = QuantumSteeringContext(
            keys=keys,
            layer_index=layer_index,
            attention_mask=attention_mask,
            steering_mask=steering_mask,
            subject_mask=subject_mask,
            object_mask=object_mask,
        )
        if not self.plugins:
            return keys

        contributions = [plugin(context) for plugin in self.plugins]
        deltas = [item.delta for item in contributions if item.delta is not None]
        gates = [item.gate for item in contributions if item.gate is not None]
        for delta in deltas:
            if delta.shape != keys.shape:
                raise ValueError("plugin delta must match key shape")
        for gate in gates:
            if gate.shape != keys.shape:
                raise ValueError("plugin gate must match key shape")

        if deltas:
            combined_delta = torch.stack(deltas, dim=0).sum(dim=0)
            if self.operator_reduction == "mean":
                combined_delta = combined_delta / len(deltas)
        elif gates:
            combined_delta = self.identity_gain * keys
        else:
            return keys

        for gate in gates:
            combined_delta = combined_delta * gate

        mask = steering_mask if steering_mask is not None else attention_mask
        if mask is not None:
            _validate_mask(mask, keys, "effective steering mask")
            combined_delta = combined_delta * mask.to(keys.device, keys.dtype).unsqueeze(-1)
        return keys + combined_delta


def normalize_plugin_names(value: str | Sequence[str]) -> tuple[str, ...]:
    names = [item.strip() for item in value.split(",")] if isinstance(value, str) else list(value)
    names = [name for name in names if name]
    unknown = sorted(set(names) - set(PLUGIN_NAMES))
    if unknown:
        raise ValueError(f"unknown quantum plugins: {unknown}")
    return tuple(dict.fromkeys(names))


def build_quantum_steering(
    plugin_names: str | Sequence[str],
    *,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    operator_reduction: str = "mean",
    identity_gain: float = 0.05,
) -> ComposableQuantumSteering:
    """Build a composable stack using the default plugin architectures."""
    names = normalize_plugin_names(plugin_names)
    plugins: list[QuantumSteeringPlugin] = []
    for name in names:
        if name == "headwise_projector":
            plugins.append(
                HeadwiseQuantumProjectorPlugin(
                    HeadwiseQuantumProjectorConfig(num_layers, num_heads, head_dim)
                )
            )
        elif name == "evidence_gate":
            plugins.append(
                QuantumEvidenceGatePlugin(
                    QuantumEvidenceGateConfig(num_layers, num_heads, head_dim)
                )
            )
        elif name == "expert_bank":
            plugins.append(
                QuantumExpertBankPlugin(
                    QuantumExpertBankConfig(num_layers, num_heads, head_dim)
                )
            )
    return ComposableQuantumSteering(
        plugins,
        operator_reduction=operator_reduction,
        identity_gain=identity_gain,
    )


def build_quantum_steering_from_metadata(
    metadata: Mapping[str, Any],
) -> ComposableQuantumSteering:
    """Reconstruct an exact plugin stack from checkpoint metadata."""
    raw_plugins = metadata.get("plugins")
    if not isinstance(raw_plugins, list):
        raise ValueError("plugin metadata must contain a plugin list")
    plugins: list[QuantumSteeringPlugin] = []
    config_types = {
        "headwise_projector": (
            HeadwiseQuantumProjectorConfig,
            HeadwiseQuantumProjectorPlugin,
        ),
        "evidence_gate": (
            QuantumEvidenceGateConfig,
            QuantumEvidenceGatePlugin,
        ),
        "expert_bank": (
            QuantumExpertBankConfig,
            QuantumExpertBankPlugin,
        ),
    }
    for item in raw_plugins:
        if not isinstance(item, Mapping):
            raise ValueError("each plugin metadata item must be a mapping")
        name = item.get("name")
        raw_config = item.get("config")
        if name not in config_types or not isinstance(raw_config, Mapping):
            raise ValueError(f"invalid plugin checkpoint entry: {name!r}")
        config_type, plugin_type = config_types[name]
        plugins.append(plugin_type(config_type(**dict(raw_config))))
    return ComposableQuantumSteering(
        plugins,
        operator_reduction=str(metadata.get("operator_reduction", "mean")),
        identity_gain=float(metadata.get("identity_gain", 0.05)),
    )


def save_quantum_steering_checkpoint(
    path: str | Path,
    steering: ComposableQuantumSteering,
    *,
    extra_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save plugin parameters independently from the frozen base model."""
    payload = {
        "format_version": 1,
        "plugin_metadata": steering.metadata(),
        "state_dict": steering.state_dict(),
        "extra_metadata": dict(extra_metadata or {}),
    }
    torch.save(payload, Path(path))


def load_quantum_steering_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[ComposableQuantumSteering, dict[str, Any]]:
    """Load a standalone plugin checkpoint and its experiment metadata."""
    try:
        payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, Mapping) or payload.get("format_version") != 1:
        raise ValueError("unsupported quantum steering checkpoint")
    plugin_metadata = payload.get("plugin_metadata")
    state_dict = payload.get("state_dict")
    if not isinstance(plugin_metadata, Mapping) or not isinstance(state_dict, Mapping):
        raise ValueError("quantum steering checkpoint is incomplete")
    steering = build_quantum_steering_from_metadata(plugin_metadata)
    steering.load_state_dict(state_dict)
    extra_metadata = payload.get("extra_metadata")
    return steering, dict(extra_metadata) if isinstance(extra_metadata, Mapping) else {}
