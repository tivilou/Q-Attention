"""Model-bound adapter for the Q-EPVG query/key/value intervention boundary."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

import torch
import torch.nn as nn

from q_attention.adapters.encoder import resolve_module


class QEPVGAttentionAdapter:
    """Attach one Q-EPVG kernel to every steerable attention layer."""

    def __init__(self, model: nn.Module, kernels: Sequence[nn.Module]) -> None:
        paths = tuple(getattr(model, "score_module_paths"))
        if len(paths) != len(kernels):
            raise ValueError("kernel count must match model attention layer count")
        self.model = model
        self.paths = paths
        self.kernels = tuple(kernels)
        self._originals: list[nn.Module] = []

    def attach(self) -> None:
        self.remove()
        self._originals = []
        for path, kernel in zip(self.paths, self.kernels):
            module = resolve_module(self.model, path.replace("score_intervention", "attention_intervention"))
            self._originals.append(module)
            parent_path, _, child = path.replace("score_intervention", "attention_intervention").rpartition(".")
            parent = resolve_module(self.model, parent_path)
            setattr(parent, child, kernel)

    def remove(self) -> None:
        if not self._originals:
            return
        for path, original in zip(self.paths, self._originals):
            intervention_path = path.replace("score_intervention", "attention_intervention")
            parent_path, _, child = intervention_path.rpartition(".")
            setattr(resolve_module(self.model, parent_path), child, original)
        self._originals = []

    @property
    def traces(self) -> list[dict[str, torch.Tensor]]:
        return [getattr(kernel, "last_trace", {}) for kernel in self.kernels]

    @contextmanager
    def active(self) -> Iterator[None]:
        self.attach()
        try:
            yield
        finally:
            self.remove()


__all__ = ["QEPVGAttentionAdapter"]
