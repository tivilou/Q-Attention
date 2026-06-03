"""Utilities for converting token spans into boolean masks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

Span = tuple[int, int]


def _validate_span(span: Span, length: int) -> None:
    start, end = span
    if start < 0 or end < 0:
        raise ValueError(f"span values must be non-negative: {span}")
    if start >= end:
        raise ValueError(f"span start must be smaller than end: {span}")
    if end > length:
        raise ValueError(f"span end {end} exceeds length {length}")


def span_mask(length: int, spans: Iterable[Span], *, device=None) -> torch.Tensor:
    """Create a boolean mask of shape ``(length,)`` from half-open spans."""
    if length <= 0:
        raise ValueError("length must be positive")
    mask = torch.zeros(length, dtype=torch.bool, device=device)
    for span in spans:
        _validate_span(span, length)
        start, end = span
        mask[start:end] = True
    return mask


def batched_span_mask(batch_size: int, length: int, spans: Sequence[Iterable[Span]], *, device=None) -> torch.Tensor:
    """Create a boolean mask of shape ``(batch_size, length)``."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(spans) != batch_size:
        raise ValueError("spans must contain one span iterable per batch item")
    rows = [span_mask(length, item_spans, device=device) for item_spans in spans]
    return torch.stack(rows, dim=0)
