"""Training and evaluation helpers for relation attention-score kernels."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from q_attention.adapters import AttentionScoreHookConfig, AttentionScoreKernelAdapter
from q_attention.adapters.encoder import resolve_module
from q_attention.metrics import classification_metrics, correct_label_margin
from q_attention.plugins import RelationAttentionScoreKernel

from .relation_steering import move_batch


RELATION_SELECTION_CHOICES = ("macro_f1_then_loss", "valid_loss")


def relation_selection_score(
    metrics: dict[str, float],
    selection_metric: str,
) -> tuple[float, float]:
    """Return a deterministic validation-only checkpoint score."""
    if selection_metric == "macro_f1_then_loss":
        return metrics["macro_f1"], -metrics["loss"]
    if selection_metric == "valid_loss":
        return -metrics["loss"], metrics["macro_f1"]
    raise ValueError(
        f"selection_metric must be one of {RELATION_SELECTION_CHOICES}"
    )


class _ScalarAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.square_total = 0.0
        self.absolute_total = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().double().reshape(-1)
        if values.numel() == 0:
            return
        self.count += values.numel()
        self.total += float(values.sum().item())
        self.square_total += float(values.square().sum().item())
        self.absolute_total += float(values.abs().sum().item())
        self.minimum = min(self.minimum, float(values.min().item()))
        self.maximum = max(self.maximum, float(values.max().item()))

    def summary(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "rms": None,
                "mean_abs": None,
                "min": None,
                "max": None,
            }
        mean = self.total / self.count
        variance = max(self.square_total / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "rms": math.sqrt(self.square_total / self.count),
            "mean_abs": self.absolute_total / self.count,
            "min": self.minimum,
            "max": self.maximum,
        }


class GradientNormTracker:
    """Aggregate optimizer-step gradient magnitudes without retaining gradients."""

    def __init__(self, named_parameters: Iterable[tuple[str, nn.Parameter]]) -> None:
        self._parameters = tuple(named_parameters)
        self._steps = 0
        self._l2 = {name: _ScalarAccumulator() for name, _ in self._parameters}
        self._mean_abs = {name: _ScalarAccumulator() for name, _ in self._parameters}
        self._max_abs = {name: _ScalarAccumulator() for name, _ in self._parameters}
        self._steps_with_gradient = {name: 0 for name, _ in self._parameters}

    def update(self) -> None:
        self._steps += 1
        for name, parameter in self._parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            gradient = gradient.detach()
            self._steps_with_gradient[name] += 1
            self._l2[name].update(gradient.float().norm().reshape(1))
            self._mean_abs[name].update(gradient.float().abs().mean().reshape(1))
            self._max_abs[name].update(gradient.float().abs().max().reshape(1))

    def summary(self) -> dict[str, Any]:
        return {
            "optimizer_steps": self._steps,
            "parameters": {
                name: {
                    "steps_with_gradient": self._steps_with_gradient[name],
                    "l2_norm": self._l2[name].summary(),
                    "mean_abs": self._mean_abs[name].summary(),
                    "max_abs": self._max_abs[name].summary(),
                }
                for name, _ in self._parameters
            },
        }


def _masked_values(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values.masked_select(mask.expand_as(values))


def diagnose_relation_attention_score_kernel(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    score_module_paths: Sequence[str],
    score_kernel: RelationAttentionScoreKernel,
) -> dict[str, Any]:
    """Measure whether a score kernel materially changes attention geometry."""
    if len(score_module_paths) != score_kernel.model_dimensions[0]:
        raise ValueError("score module path count must match kernel num_layers")
    model.eval()
    score_kernel.eval()
    layers = [
        {
            "base": _ScalarAccumulator(),
            "centered": _ScalarAccumulator(),
            "residual": _ScalarAccumulator(),
            "attention_tv": _ScalarAccumulator(),
            "attention_max_delta": _ScalarAccumulator(),
            "head_cosine": _ScalarAccumulator(),
        }
        for _ in score_module_paths
    ]
    num_batches = 0

    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            handles: list[torch.utils.hooks.RemovableHandle] = []

            def make_hook(layer_index: int):
                def hook(
                    _module: nn.Module,
                    inputs: tuple[object, ...],
                    output: object,
                ) -> None:
                    if (
                        len(inputs) not in {3, 4}
                        or not all(isinstance(item, torch.Tensor) for item in inputs)
                        or not isinstance(output, torch.Tensor)
                    ):
                        raise TypeError("score diagnostics require score, query, and key tensors")
                    scores, query, key = inputs[:3]
                    centered = score_kernel.centered_kernel(
                        query,
                        key,
                        layer_index=layer_index,
                        attention_mask=batch["attention_mask"],
                        subject_mask=batch["subject_mask"],
                        object_mask=batch["object_mask"],
                    )
                    residual = score_kernel.score_residual(centered, layer_index)
                    valid_scores = (
                        batch["attention_mask"][:, None, :, None]
                        & batch["attention_mask"][:, None, None, :]
                    )
                    layer = layers[layer_index]
                    layer["base"].update(_masked_values(scores, valid_scores))
                    layer["centered"].update(_masked_values(centered, valid_scores))
                    layer["residual"].update(_masked_values(residual, valid_scores))

                    key_mask = batch["attention_mask"][:, None, None, :].bool()
                    masked_scores = scores.masked_fill(
                        ~key_mask, torch.finfo(scores.dtype).min
                    )
                    baseline_attention = torch.softmax(masked_scores, dim=-1)
                    steered_attention = torch.softmax(masked_scores + residual, dim=-1)
                    probability_delta = (steered_attention - baseline_attention).abs()
                    total_variation = 0.5 * probability_delta.sum(dim=-1)
                    query_mask = batch["attention_mask"][:, None, :].bool()
                    layer["attention_tv"].update(
                        _masked_values(total_variation, query_mask)
                    )
                    layer["attention_max_delta"].update(
                        _masked_values(probability_delta, valid_scores)
                    )

                    if centered.shape[1] > 1:
                        vectors = (centered * valid_scores).flatten(start_dim=2)
                        for left in range(centered.shape[1] - 1):
                            for right in range(left + 1, centered.shape[1]):
                                cosine = F.cosine_similarity(
                                    vectors[:, left],
                                    vectors[:, right],
                                    dim=-1,
                                    eps=score_kernel.config.eps,
                                )
                                layer["head_cosine"].update(cosine)

                return hook

            try:
                for layer_index, path in enumerate(score_module_paths):
                    module = resolve_module(model, path)
                    handles.append(module.register_forward_hook(make_hook(layer_index)))
                model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["subject_mask"],
                    batch["object_mask"],
                )
                num_batches += 1
            finally:
                for handle in handles:
                    handle.remove()

    layer_payloads: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(layers):
        base = layer["base"].summary()
        residual = layer["residual"].summary()
        base_rms = float(base["rms"] or 0.0)
        residual_rms = float(residual["rms"] or 0.0)
        head_cosine = layer["head_cosine"].summary()
        observable_weights = score_kernel.observable_weights(layer_index)
        layer_payloads.append(
            {
                "layer_index": layer_index,
                "base_scores": base,
                "centered_kernel": layer["centered"].summary(),
                "score_residual": residual,
                "residual_to_base_rms_ratio": (
                    residual_rms / base_rms if base_rms > 0.0 else None
                ),
                "attention_total_variation": layer["attention_tv"].summary(),
                "attention_probability_delta": layer[
                    "attention_max_delta"
                ].summary(),
                "cross_head_centered_kernel": {
                    "cosine": head_cosine,
                    "mean_abs_cosine": (
                        layer["head_cosine"].absolute_total
                        / layer["head_cosine"].count
                        if layer["head_cosine"].count
                        else None
                    ),
                    "mean_squared_cosine": (
                        layer["head_cosine"].square_total
                        / layer["head_cosine"].count
                        if layer["head_cosine"].count
                        else None
                    ),
                },
                "gains": [
                    float(value)
                    for value in score_kernel.gains(layer_index).detach().cpu().tolist()
                ],
                "observable_weights": (
                    observable_weights.detach().cpu().tolist()
                    if observable_weights is not None
                    else None
                ),
            }
        )
    return {"num_batches": num_batches, "layers": layer_payloads}


def diagnose_relation_attention_score_task_alignment(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    score_module_paths: Sequence[str],
    score_kernel: RelationAttentionScoreKernel,
) -> dict[str, Any]:
    """Compare score residuals with the local task-loss descent direction."""
    model.eval()
    score_kernel.eval()
    adapter = AttentionScoreKernelAdapter(model, score_module_paths, score_kernel)
    actual_loss_change = _ScalarAccumulator()
    first_order_loss_change = _ScalarAccumulator()
    descent_cosine = [_ScalarAccumulator() for _ in score_module_paths]
    gradient_magnitude = [_ScalarAccumulator() for _ in score_module_paths]
    num_batches = 0

    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        with torch.no_grad():
            baseline_logits = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["subject_mask"],
                batch["object_mask"],
            )
            baseline_losses = F.cross_entropy(
                baseline_logits, batch["labels"], reduction="none"
            )

        config = attention_score_hook_config(batch)
        adapter.attach(config)
        base_scores: dict[int, torch.Tensor] = {}
        steered_scores: dict[int, torch.Tensor] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []

        def make_capture_hook(layer_index: int):
            def hook(
                _module: nn.Module,
                inputs: tuple[object, ...],
                output: object,
            ) -> None:
                if (
                    not inputs
                    or not isinstance(inputs[0], torch.Tensor)
                    or not isinstance(output, torch.Tensor)
                ):
                    raise TypeError("task-alignment diagnostics require tensor scores")
                base_scores[layer_index] = inputs[0].detach()
                steered_scores[layer_index] = output

            return hook

        try:
            for layer_index, path in enumerate(score_module_paths):
                module = resolve_module(model, path)
                handles.append(module.register_forward_hook(make_capture_hook(layer_index)))
            steered_logits = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["subject_mask"],
                batch["object_mask"],
            )
            steered_losses = F.cross_entropy(
                steered_logits, batch["labels"], reduction="none"
            )
            ordered_scores = [steered_scores[index] for index in range(len(score_module_paths))]
            gradients = torch.autograd.grad(steered_losses.mean(), ordered_scores)
            actual_loss_change.update(steered_losses.detach() - baseline_losses)

            batch_first_order = 0.0
            valid_scores = (
                batch["attention_mask"][:, None, :, None]
                & batch["attention_mask"][:, None, None, :]
            )
            for layer_index, (scores, gradient) in enumerate(
                zip(ordered_scores, gradients, strict=True)
            ):
                residual = scores - base_scores[layer_index]
                masked_residual = residual * valid_scores
                masked_descent = -gradient * valid_scores
                flat_residual = masked_residual.flatten(start_dim=1)
                flat_descent = masked_descent.flatten(start_dim=1)
                cosine = F.cosine_similarity(
                    flat_residual,
                    flat_descent,
                    dim=-1,
                    eps=score_kernel.config.eps,
                )
                descent_cosine[layer_index].update(cosine)
                gradient_magnitude[layer_index].update(
                    _masked_values(gradient, valid_scores)
                )
                batch_first_order += float(
                    (gradient.detach() * residual.detach()).sum().item()
                )
            first_order_loss_change.update(torch.tensor([batch_first_order]))
            num_batches += 1
        finally:
            for handle in handles:
                handle.remove()
            adapter.remove()

    return {
        "num_batches": num_batches,
        "actual_loss_change": actual_loss_change.summary(),
        "first_order_loss_change": first_order_loss_change.summary(),
        "layers": [
            {
                "layer_index": layer_index,
                "residual_descent_cosine": descent_cosine[layer_index].summary(),
                "task_gradient": gradient_magnitude[layer_index].summary(),
            }
            for layer_index in range(len(score_module_paths))
        ],
    }


def attention_score_hook_config(
    batch: dict[str, torch.Tensor],
    *,
    routing_mode: str = "learned",
) -> AttentionScoreHookConfig:
    return AttentionScoreHookConfig(
        attention_mask=batch["attention_mask"],
        subject_mask=batch["subject_mask"],
        object_mask=batch["object_mask"],
        routing_mode=routing_mode,
    )


def evaluate_relation_attention_score_kernel(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    num_labels: int,
    *,
    adapter: AttentionScoreKernelAdapter | None,
    routing_mode: str = "learned",
) -> dict[str, float]:
    model.eval()
    if adapter is not None:
        adapter.score_kernel.eval()
    predictions: list[int] = []
    labels: list[int] = []
    total_loss = 0.0
    total_margin = 0.0
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
                    attention_score_hook_config(batch, routing_mode=routing_mode)
                ):
                    logits = model(
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["subject_mask"],
                        batch["object_mask"],
                    )
            loss = F.cross_entropy(logits, batch["labels"])
            total_loss += float(loss.item()) * batch["labels"].shape[0]
            total_margin += float(correct_label_margin(logits, batch["labels"]).sum().item())
            total_items += batch["labels"].shape[0]
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
    metrics = classification_metrics(predictions, labels, num_labels)
    metrics["loss"] = total_loss / max(total_items, 1)
    metrics["correct_label_margin"] = total_margin / max(total_items, 1)
    return metrics
