"""Hook adapter for relation-conditioned attention-score kernels."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
import torch.nn as nn

from q_attention.adapters.encoder import resolve_module
from q_attention.plugins.attention_score_kernel import RelationAttentionScoreKernel


@dataclass(frozen=True)
class AttentionScoreHookConfig:
    attention_mask: torch.Tensor
    subject_mask: torch.Tensor
    object_mask: torch.Tensor
    evidence_view: str = "full"
    random_seed: int = 0
    detach_random: bool = False
    routing_mode: str = "learned"


class AttentionScoreKernelAdapter:
    """Attach a score kernel at explicit attention score hook points."""

    def __init__(
        self,
        model: nn.Module,
        score_module_paths: Sequence[str],
        score_kernel: RelationAttentionScoreKernel,
    ) -> None:
        if not score_module_paths:
            raise ValueError("at least one score module path is required")
        if len(score_module_paths) != score_kernel.model_dimensions[0]:
            raise ValueError("score module path count must match kernel num_layers")
        self.model = model
        self.score_module_paths = tuple(score_module_paths)
        self.score_kernel = score_kernel
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    @property
    def attached(self) -> bool:
        return bool(self._handles)

    def attach(self, config: AttentionScoreHookConfig) -> None:
        self.remove()

        def make_hook(layer_index: int):
            def hook(
                _module: nn.Module,
                inputs: tuple[object, ...],
                output: object,
            ) -> torch.Tensor:
                if (
                    len(inputs) != 3
                    or not isinstance(inputs[0], torch.Tensor)
                    or not isinstance(inputs[1], torch.Tensor)
                    or not isinstance(inputs[2], torch.Tensor)
                    or not isinstance(output, torch.Tensor)
                ):
                    raise TypeError(
                        "score hook must receive and return score, query, and key tensors"
                    )
                scores, query, key = inputs
                residual = self.score_kernel(
                    query,
                    key,
                    layer_index=layer_index,
                    attention_mask=config.attention_mask,
                    subject_mask=config.subject_mask,
                    object_mask=config.object_mask,
                    evidence_view=config.evidence_view,
                    random_seed=config.random_seed + layer_index,
                    detach_random=config.detach_random,
                    routing_mode=config.routing_mode,
                )
                if residual.shape != scores.shape:
                    raise ValueError("score residual must match attention score shape")
                return output + residual.to(device=output.device, dtype=output.dtype)

            return hook

        for layer_index, path in enumerate(self.score_module_paths):
            module = resolve_module(self.model, path)
            self._handles.append(module.register_forward_hook(make_hook(layer_index)))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @contextmanager
    def steering(self, config: AttentionScoreHookConfig) -> Iterator[None]:
        self.attach(config)
        try:
            yield
        finally:
            self.remove()
