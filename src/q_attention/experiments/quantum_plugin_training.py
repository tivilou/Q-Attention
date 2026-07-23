"""Shared training and evaluation helpers for relation quantum plugins."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from q_attention.adapters import QuantumPluginHookConfig, QuantumPluginSteeringAdapter
from q_attention.metrics import classification_metrics

from .relation_steering import anchor_mask_from_batch, move_batch


def quantum_plugin_hook_config(
    batch: dict[str, torch.Tensor],
    anchor: str,
) -> QuantumPluginHookConfig:
    return QuantumPluginHookConfig(
        attention_mask=batch["attention_mask"],
        steering_mask=anchor_mask_from_batch(batch, anchor),
        subject_mask=batch["subject_mask"],
        object_mask=batch["object_mask"],
    )


def evaluate_relation_quantum_plugins(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    num_labels: int,
    *,
    adapter: QuantumPluginSteeringAdapter | None,
    steering_anchor: str,
) -> dict[str, float]:
    """Evaluate the frozen relation model with an optional plugin stack."""
    model.eval()
    if adapter is not None:
        adapter.steering_module.eval()
    predictions: list[int] = []
    labels: list[int] = []
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            if adapter is None:
                logits = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["subject_mask"],
                    batch["object_mask"],
                )
            else:
                with adapter.steering(
                    quantum_plugin_hook_config(batch, steering_anchor)
                ):
                    logits = model(
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["subject_mask"],
                        batch["object_mask"],
                    )
            loss = F.cross_entropy(logits, batch["labels"])
            total_loss += float(loss.item()) * batch["labels"].shape[0]
            total_items += batch["labels"].shape[0]
            predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
    metrics = classification_metrics(predictions, labels, num_labels)
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics
