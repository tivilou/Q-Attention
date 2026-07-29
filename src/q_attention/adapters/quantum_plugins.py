"""Hook adapter for composable quantum steering plugins."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
import torch.nn as nn

from q_attention.adapters.encoder import resolve_module
from q_attention.plugins import ComposableQuantumSteering


@dataclass(frozen=True)
class QuantumPluginHookConfig:
    attention_mask: torch.Tensor | None = None
    steering_mask: torch.Tensor | None = None
    subject_mask: torch.Tensor | None = None
    object_mask: torch.Tensor | None = None


class QuantumPluginSteeringAdapter:
    """Attach a trainable plugin stack without changing the base model."""

    def __init__(
        self,
        model: nn.Module,
        key_module_paths: Sequence[str],
        steering: ComposableQuantumSteering,
    ) -> None:
        if not key_module_paths:
            raise ValueError("at least one key module path is required")
        if (
            steering.model_dimensions is not None
            and len(key_module_paths) != steering.model_dimensions[0]
        ):
            raise ValueError("key module path count must match plugin num_layers")
        self.model = model
        self.key_module_paths = tuple(key_module_paths)
        self.steering_module = steering
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    @property
    def attached(self) -> bool:
        return bool(self._handles)

    def attach(self, config: QuantumPluginHookConfig) -> None:
        self.remove()

        def make_hook(layer_index: int):
            def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> object:
                return self._steer_output(output, layer_index=layer_index, config=config)

            return hook

        for layer_index, path in enumerate(self.key_module_paths):
            module = resolve_module(self.model, path)
            self._handles.append(module.register_forward_hook(make_hook(layer_index)))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @contextmanager
    def steering(self, config: QuantumPluginHookConfig) -> Iterator[None]:
        self.attach(config)
        try:
            yield
        finally:
            self.remove()

    def _steer_tensor(
        self,
        keys: torch.Tensor,
        *,
        layer_index: int,
        config: QuantumPluginHookConfig,
    ) -> torch.Tensor:
        return self.steering_module(
            keys,
            layer_index=layer_index,
            attention_mask=config.attention_mask,
            steering_mask=config.steering_mask,
            subject_mask=config.subject_mask,
            object_mask=config.object_mask,
        )

    def _steer_output(
        self,
        output: object,
        *,
        layer_index: int,
        config: QuantumPluginHookConfig,
    ) -> object:
        if isinstance(output, torch.Tensor):
            return self._steer_tensor(output, layer_index=layer_index, config=config)
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            steered = self._steer_tensor(output[0], layer_index=layer_index, config=config)
            return (steered, *output[1:])
        if isinstance(output, list) and output and isinstance(output[0], torch.Tensor):
            steered = self._steer_tensor(output[0], layer_index=layer_index, config=config)
            return [steered, *output[1:]]
        raise TypeError("key module output must be a Tensor or start with a Tensor")
