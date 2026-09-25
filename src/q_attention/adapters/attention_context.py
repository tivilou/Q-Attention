"""Adapter for query/key/value context interventions in relation attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from q_attention.adapters.encoder import resolve_module


@dataclass(frozen=True)
class AttentionContextHookConfig:
    attention_mask: torch.Tensor
    query_mask: torch.Tensor | None = None
    capture_trace: bool = False


class AttentionContextKernelAdapter:
    """Attach a value/context kernel to explicit self-attention hook points."""

    def __init__(
        self,
        model: nn.Module,
        context_module_paths: Sequence[str],
        context_kernel: nn.Module,
        *,
        query_chunk_size: int | None = None,
    ) -> None:
        if not context_module_paths:
            raise ValueError("at least one context module path is required")
        if query_chunk_size is not None and query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive when specified")
        dimensions = getattr(context_kernel, "model_dimensions", None)
        if dimensions is not None and len(context_module_paths) != dimensions[0]:
            raise ValueError("context module path count must match kernel num_layers")
        self.model = model
        self.context_module_paths = tuple(context_module_paths)
        self.context_kernel = context_kernel
        self.query_chunk_size = query_chunk_size
        self.last_traces: dict[int, dict[str, torch.Tensor]] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    @property
    def attached(self) -> bool:
        return bool(self._handles)

    def attach(self, config: AttentionContextHookConfig) -> None:
        self.remove()
        self.last_traces.clear()

        def make_hook(layer_index: int):
            def hook(
                _module: nn.Module,
                inputs: tuple[object, ...],
                output: object,
            ) -> torch.Tensor:
                if len(inputs) != 5 or not all(
                    isinstance(item, torch.Tensor) for item in inputs[:4]
                ):
                    raise TypeError(
                        "context hook must receive query, key, value, scores, and mask"
                    )
                query, key, value, scores = inputs[:4]
                if not isinstance(output, (torch.Tensor, type(None))):
                    raise TypeError("context pass-through hook must return a tensor or None")
                attention_mask = config.attention_mask
                query_mask = config.query_mask
                if attention_mask.ndim != 2 or attention_mask.shape != (
                    query.shape[0], key.shape[2]
                ):
                    raise ValueError("attention_mask must match batch and key token dimensions")
                if query_mask is None:
                    query_mask = attention_mask
                if query_mask.shape != (query.shape[0], query.shape[2]):
                    raise ValueError("query_mask must match batch and query token dimensions")

                def run_kernel(
                    q: torch.Tensor,
                    q_scores: torch.Tensor,
                    q_mask: torch.Tensor,
                    *,
                    capture: bool,
                ) -> torch.Tensor:
                    result = self.context_kernel(
                        q,
                        key,
                        value,
                        scores=q_scores,
                        layer_index=layer_index,
                        attention_mask=attention_mask,
                        query_mask=q_mask,
                        return_trace=capture,
                    )
                    if capture:
                        if not isinstance(result, tuple) or len(result) != 2:
                            raise TypeError("trace capture requires (context, trace) output")
                        context, trace = result
                        if not isinstance(trace, dict):
                            raise TypeError("context kernel trace must be a dictionary")
                        self.last_traces[layer_index] = trace
                    else:
                        context = result[0] if isinstance(result, tuple) else result
                    if not isinstance(context, torch.Tensor) or context.shape != q.shape:
                        raise ValueError("context kernel output must match query shape")
                    return context

                capture = config.capture_trace
                chunk_size = self.query_chunk_size
                if capture or chunk_size is None or query.shape[2] <= chunk_size:
                    return run_kernel(query, scores, query_mask, capture=capture)

                chunks = []
                for start in range(0, query.shape[2], chunk_size):
                    stop = min(start + chunk_size, query.shape[2])
                    chunks.append(
                        run_kernel(
                            query[:, :, start:stop],
                            scores[:, :, start:stop],
                            query_mask[:, start:stop],
                            capture=False,
                        )
                    )
                return torch.cat(chunks, dim=2)

            return hook

        for layer_index, path in enumerate(self.context_module_paths):
            module = resolve_module(self.model, path)
            self._handles.append(module.register_forward_hook(make_hook(layer_index)))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
