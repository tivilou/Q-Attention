"""Generic encoder key-steering adapter.

The adapter attaches forward hooks to key-projection modules and replaces their
output with ``k' = k + gPk`` for selected span positions. It is intentionally
model-agnostic: callers provide module paths such as ``layers.0.key_proj``.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Iterator, Sequence

import torch
import torch.nn as nn

from q_attention.steering import apply_key_steering


@dataclass(frozen=True)
class KeySteeringHookConfig:
    """Runtime key-steering configuration."""

    projector: torch.Tensor | Mapping[str, torch.Tensor]
    mask: torch.Tensor
    gain: float | Mapping[str, float] = 1.0


def resolve_module(model: nn.Module, path: str) -> nn.Module:
    """Resolve a dotted module path on a PyTorch module.

    Numeric path components index into ``ModuleList`` or ``Sequential`` objects.
    """
    if not path:
        raise ValueError("module path cannot be empty")

    current: nn.Module | object = model
    for part in path.split("."):
        if part.isdigit():
            try:
                current = current[int(part)]  # type: ignore[index]
            except Exception as exc:  # pragma: no cover - defensive branch
                raise ValueError(f"cannot index '{part}' while resolving '{path}'") from exc
        else:
            if not hasattr(current, part):
                raise ValueError(f"module path '{path}' cannot resolve component '{part}'")
            current = getattr(current, part)

    if not isinstance(current, nn.Module):
        raise ValueError(f"resolved path '{path}' is not a torch.nn.Module")
    return current


class EncoderKeySteeringAdapter:
    """Attach key-steering hooks to selected encoder modules."""

    def __init__(self, model: nn.Module, key_module_paths: Sequence[str]) -> None:
        if not key_module_paths:
            raise ValueError("at least one key module path is required")
        self.model = model
        self.key_module_paths = tuple(key_module_paths)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    @property
    def attached(self) -> bool:
        """Return whether hooks are currently attached."""
        return bool(self._handles)

    def attach(self, config: KeySteeringHookConfig) -> None:
        """Attach hooks using the provided projector, mask, and gain."""
        self.remove()
        mask = config.mask

        if isinstance(config.projector, torch.Tensor):
            projectors = {path: config.projector for path in self.key_module_paths}
        elif isinstance(config.projector, Mapping):
            if any(not isinstance(path, str) for path in config.projector):
                raise TypeError("layer projector paths must be strings")
            missing = sorted(set(self.key_module_paths) - set(config.projector))
            unexpected = sorted(set(config.projector) - set(self.key_module_paths))
            if missing or unexpected:
                raise ValueError(f"layer projector paths do not match adapter paths; missing={missing}, unexpected={unexpected}")
            projectors = dict(config.projector)
            if any(not isinstance(projector, torch.Tensor) for projector in projectors.values()):
                raise TypeError("every layer projector must be a tensor")
        else:
            raise TypeError("projector must be a tensor or a mapping from module path to tensor")

        if isinstance(config.gain, Mapping):
            if any(not isinstance(path, str) for path in config.gain):
                raise TypeError("layer gain paths must be strings")
            missing = sorted(set(self.key_module_paths) - set(config.gain))
            unexpected = sorted(set(config.gain) - set(self.key_module_paths))
            if missing or unexpected:
                raise ValueError(f"layer gain paths do not match adapter paths; missing={missing}, unexpected={unexpected}")
            gains = {path: float(config.gain[path]) for path in self.key_module_paths}
        elif isinstance(config.gain, (int, float)):
            gains = {path: float(config.gain) for path in self.key_module_paths}
        else:
            raise TypeError("gain must be a number or a mapping from module path to number")
        if any(not math.isfinite(gain) for gain in gains.values()):
            raise ValueError("steering gains must be finite")

        def make_hook(projector: torch.Tensor, gain: float):
            def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> object:
                return self._steer_output(output, projector=projector, mask=mask, gain=gain)

            return hook

        for path in self.key_module_paths:
            module = resolve_module(self.model, path)
            self._handles.append(module.register_forward_hook(make_hook(projectors[path], gains[path])))

    def remove(self) -> None:
        """Remove all active hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @contextmanager
    def steering(self, config: KeySteeringHookConfig) -> Iterator[None]:
        """Context manager that attaches hooks and always removes them."""
        self.attach(config)
        try:
            yield
        finally:
            self.remove()

    @staticmethod
    def _steer_output(output: object, *, projector: torch.Tensor, mask: torch.Tensor, gain: float) -> object:
        """Apply steering to tensor outputs or the first tensor inside tuples/lists."""
        if isinstance(output, torch.Tensor):
            return apply_key_steering(output, projector, mask=mask, gain=gain)

        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            steered = apply_key_steering(output[0], projector, mask=mask, gain=gain)
            return (steered, *output[1:])

        if isinstance(output, list) and output and isinstance(output[0], torch.Tensor):
            steered = apply_key_steering(output[0], projector, mask=mask, gain=gain)
            return [steered, *output[1:]]

        raise TypeError("key module output must be a Tensor or a tuple/list with a Tensor as first item")
