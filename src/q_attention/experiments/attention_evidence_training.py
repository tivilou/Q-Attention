"""Training objectives and diagnostics for counterfactual token evidence."""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from q_attention.adapters import AttentionScoreHookConfig, AttentionScoreKernelAdapter
from q_attention.metrics import correct_label_margin
from q_attention.plugins import (
    EVIDENCE_CORRELATION_CHANNELS,
    RelationAttentionScoreKernel,
)

from .relation_steering import move_batch


COUNTERFACTUAL_OBJECTIVE_CHOICES = (
    "detached_margin",
    "paired_contrast",
    "paired_hinge",
)


class _ScalarAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.square_total = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().double().reshape(-1)
        if values.numel() == 0:
            return
        self.count += values.numel()
        self.total += float(values.sum().item())
        self.square_total += float(values.square().sum().item())
        self.minimum = min(self.minimum, float(values.min().item()))
        self.maximum = max(self.maximum, float(values.max().item()))

    def summary(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        mean = self.total / self.count
        variance = max(self.square_total / self.count - mean * mean, 0.0)
        return {
            "count": self.count,
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self.minimum,
            "max": self.maximum,
        }


def _distillation_kl(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    target = torch.softmax(teacher.detach(), dim=-1)
    return torch.sum(
        target * (target.clamp_min(1e-12).log() - torch.log_softmax(student, dim=-1)),
        dim=-1,
    )


def _view_logits(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    adapter: AttentionScoreKernelAdapter,
    *,
    view: str,
    random_seed: int,
    detach_random: bool,
) -> torch.Tensor:
    hook_config = AttentionScoreHookConfig(
        attention_mask=batch["attention_mask"],
        subject_mask=batch["subject_mask"],
        object_mask=batch["object_mask"],
        evidence_view=view,
        random_seed=random_seed,
        detach_random=detach_random,
    )
    with adapter.steering(hook_config):
        return model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["subject_mask"],
            batch["object_mask"],
        )


def _context_values(
    scores: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    context = (
        batch["attention_mask"]
        & ~(batch["subject_mask"] | batch["object_mask"])
    )[:, None, :]
    return scores.masked_select(context.expand_as(scores))


def _evidence_task_alignment_loss(
    task_loss: torch.Tensor,
    captured_scores: tuple[torch.Tensor, ...],
    batch: dict[str, torch.Tensor],
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align centered token evidence with detached task-loss descent directions."""
    gradients = torch.autograd.grad(
        task_loss,
        captured_scores,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    context = (
        batch["attention_mask"]
        & ~(batch["subject_mask"] | batch["object_mask"])
    )[:, None, :]
    losses: list[torch.Tensor] = []
    alignments: list[torch.Tensor] = []
    variation_floor = max(eps, 1e-6)
    for scores, gradient in zip(captured_scores, gradients, strict=True):
        if gradient is None:
            continue
        mask = context.to(device=scores.device, dtype=scores.dtype)
        count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        centered_scores = scores - (scores * mask).sum(dim=-1, keepdim=True) / count
        utility = -gradient.detach()
        centered_utility = utility - (
            (utility * mask).sum(dim=-1, keepdim=True) / count
        )
        centered_scores = (centered_scores * mask).reshape(-1, scores.shape[-1])
        centered_utility = (centered_utility * mask).reshape(-1, scores.shape[-1])
        active = (
            torch.linalg.vector_norm(centered_scores, dim=-1) > variation_floor
        ) & (
            torch.linalg.vector_norm(centered_utility, dim=-1) > eps
        )
        if not active.any():
            continue
        alignment = F.cosine_similarity(
            centered_scores[active],
            centered_utility[active],
            dim=-1,
            eps=eps,
        )
        alignments.append(alignment)
        losses.append(1.0 - alignment)
    if not losses:
        zero = task_loss.new_zeros(())
        return zero, zero
    return torch.cat(losses).mean(), torch.cat(alignments).mean()


def counterfactual_evidence_objective(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    adapter: AttentionScoreKernelAdapter,
    *,
    counterfactual_weight: float,
    keep_weight: float,
    drop_weight: float,
    budget_weight: float,
    evidence_budget: float,
    rank_margin: float,
    random_seed: int,
    objective_mode: str = "detached_margin",
    task_alignment_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train evidence to preserve keep views and outperform matched random drops."""
    selector = adapter.score_kernel.evidence_selector
    if selector is None:
        raise ValueError("counterfactual evidence training requires an evidence selector")
    if not 0.0 < evidence_budget < 1.0:
        raise ValueError("evidence_budget must lie inside (0, 1)")
    if min(
        counterfactual_weight,
        keep_weight,
        drop_weight,
        budget_weight,
        rank_margin,
        task_alignment_weight,
    ) < 0:
        raise ValueError("counterfactual objective weights must be non-negative")
    if objective_mode not in COUNTERFACTUAL_OBJECTIVE_CHOICES:
        raise ValueError(
            f"objective_mode must be one of {COUNTERFACTUAL_OBJECTIVE_CHOICES}"
        )

    with selector.capture_token_scores():
        full_logits = _view_logits(
            model,
            batch,
            adapter,
            view="full",
            random_seed=random_seed,
            detach_random=False,
        )
        captured_scores = selector.captured_token_scores()
        captured_steering_scores = selector.captured_steering_scores()
        budget_losses: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for scores in captured_scores:
            context = (
                batch["attention_mask"]
                & ~(batch["subject_mask"] | batch["object_mask"])
            )[:, None, :].to(scores.dtype)
            count = context.sum(dim=-1).clamp_min(1.0)
            context_mean = (scores * context).sum(dim=-1) / count
            budget_losses.append((context_mean - evidence_budget).square().mean())
            values = _context_values(scores, batch).clamp(1e-6, 1.0 - 1e-6)
            entropies.append(
                -(values * values.log() + (1.0 - values) * (1.0 - values).log()).mean()
            )
        zero = full_logits.sum() * 0.0
        budget_loss = torch.stack(budget_losses).mean() if budget_losses else zero
        gate_entropy = torch.stack(entropies).mean() if entropies else zero

    keep_logits = _view_logits(
        model,
        batch,
        adapter,
        view="keep",
        random_seed=random_seed,
        detach_random=False,
    )
    drop_logits = _view_logits(
        model,
        batch,
        adapter,
        view="drop",
        random_seed=random_seed,
        detach_random=False,
    )
    if objective_mode in {"paired_contrast", "paired_hinge"}:
        random_keep_logits = _view_logits(
            model,
            batch,
            adapter,
            view="random_keep",
            random_seed=random_seed,
            detach_random=False,
        )
        random_drop_logits = _view_logits(
            model,
            batch,
            adapter,
            view="random_drop",
            random_seed=random_seed,
            detach_random=False,
        )
    else:
        with torch.no_grad():
            random_keep_logits = _view_logits(
                model,
                batch,
                adapter,
                view="random_keep",
                random_seed=random_seed,
                detach_random=True,
            )
            random_drop_logits = _view_logits(
                model,
                batch,
                adapter,
                view="random_drop",
                random_seed=random_seed,
                detach_random=True,
            )

    task_loss = F.cross_entropy(full_logits, batch["labels"])
    if task_alignment_weight > 0.0:
        task_alignment_loss, task_alignment = _evidence_task_alignment_loss(
            task_loss,
            captured_steering_scores,
            batch,
            eps=selector.config.eps,
        )
    else:
        task_alignment_loss = task_loss.new_zeros(())
        task_alignment = task_loss.new_zeros(())
    keep_kl = _distillation_kl(keep_logits, full_logits)
    random_keep_kl = _distillation_kl(random_keep_logits, full_logits)
    drop_margin = correct_label_margin(drop_logits, batch["labels"])
    random_drop_margin = correct_label_margin(random_drop_logits, batch["labels"])
    if objective_mode == "paired_contrast":
        keep_rank = (keep_kl - random_keep_kl).mean()
        drop_rank = (drop_margin - random_drop_margin).mean()
    else:
        keep_rank = F.relu(
            keep_kl - random_keep_kl.detach() + rank_margin
        ).mean()
        drop_rank = F.relu(
            drop_margin - random_drop_margin.detach() + rank_margin
        ).mean()
    keep_loss = keep_kl.mean() + keep_rank
    objective = (
        task_loss
        + counterfactual_weight * (keep_weight * keep_loss + drop_weight * drop_rank)
        + budget_weight * budget_loss
        + task_alignment_weight * task_alignment_loss
    )
    components = {
        "task_loss": task_loss,
        "keep_kl": keep_kl.mean(),
        "random_keep_kl": random_keep_kl.mean(),
        "keep_rank_loss": keep_rank,
        "drop_rank_loss": drop_rank,
        "task_alignment_loss": task_alignment_loss,
        "task_alignment": task_alignment,
        "budget_loss": budget_loss,
        "gate_entropy": gate_entropy,
        "keep_advantage": (random_keep_kl - keep_kl).mean(),
        "drop_advantage": (random_drop_margin - drop_margin).mean(),
    }
    return objective, components


def diagnose_relation_counterfactual_evidence(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    adapter: AttentionScoreKernelAdapter,
    random_repeats: int = 4,
    random_seed: int = 101,
    minimum_advantage: float = 1e-6,
) -> dict[str, Any]:
    """Compare learned evidence masks with size-matched random permutations."""
    selector = adapter.score_kernel.evidence_selector
    if selector is None:
        raise ValueError("counterfactual evidence diagnostics require a selector")
    if random_repeats <= 0:
        raise ValueError("random_repeats must be positive")
    if minimum_advantage < 0.0:
        raise ValueError("minimum_advantage must be non-negative")
    model.eval()
    adapter.score_kernel.eval()
    accumulators = {
        name: _ScalarAccumulator()
        for name in (
            "context_evidence",
            "context_steering",
            "steering_sufficiency_delta",
            "steering_sufficiency_cosine",
            "gate_entropy",
            "full_margin",
            "keep_kl",
            "random_keep_kl",
            "keep_advantage",
            "drop_margin_reduction",
            "random_drop_margin_reduction",
            "drop_advantage",
            "keep_prediction_agreement",
            "random_keep_prediction_agreement",
            "keep_win",
            "drop_win",
        )
    }
    layer_evidence = [_ScalarAccumulator() for _ in range(selector.config.num_layers)]
    layer_steering = [_ScalarAccumulator() for _ in range(selector.config.num_layers)]
    layer_readout_cosine = [
        _ScalarAccumulator() for _ in range(selector.config.num_layers)
    ]
    layer_conditioning_delta = [
        _ScalarAccumulator() for _ in range(selector.config.num_layers)
    ]
    layer_polarity_flip = [
        _ScalarAccumulator() for _ in range(selector.config.num_layers)
    ]
    layer_relation_frame_angle = [
        _ScalarAccumulator() for _ in range(selector.config.num_layers)
    ]
    num_batches = 0

    with torch.no_grad():
        for batch_index, raw_batch in enumerate(loader):
            batch = move_batch(raw_batch, device)
            with selector.capture_token_scores():
                full_logits = _view_logits(
                    model,
                    batch,
                    adapter,
                    view="full",
                    random_seed=random_seed,
                    detach_random=False,
                )
                captured = selector.captured_token_scores()
                captured_steering = selector.captured_steering_scores()
                captured_measurement_weights = (
                    selector.captured_measurement_weights()
                )
                captured_relation_frame_angles = (
                    selector.captured_relation_frame_angles()
                )
                for layer_index, (scores, steering_scores) in enumerate(
                    zip(captured, captured_steering, strict=True)
                ):
                    values = _context_values(scores, batch)
                    steering_values = _context_values(steering_scores, batch)
                    layer_evidence[layer_index].update(values)
                    layer_steering[layer_index].update(steering_values)
                    accumulators["context_evidence"].update(values)
                    accumulators["context_steering"].update(steering_values)
                    accumulators["steering_sufficiency_delta"].update(
                        steering_values - values
                    )
                    context_mask = (
                        batch["attention_mask"]
                        & ~(batch["subject_mask"] | batch["object_mask"])
                    )[:, None, :].to(scores.dtype)
                    count = context_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
                    centered_scores = scores - (
                        (scores * context_mask).sum(dim=-1, keepdim=True) / count
                    )
                    centered_steering = steering_scores - (
                        (steering_scores * context_mask).sum(
                            dim=-1, keepdim=True
                        )
                        / count
                    )
                    score_norm = torch.linalg.vector_norm(
                        centered_scores * context_mask,
                        dim=-1,
                    )
                    steering_norm = torch.linalg.vector_norm(
                        centered_steering * context_mask,
                        dim=-1,
                    )
                    active = (score_norm > selector.config.eps) & (
                        steering_norm > selector.config.eps
                    )
                    if active.any():
                        cosine = F.cosine_similarity(
                            centered_steering[active],
                            centered_scores[active],
                            dim=-1,
                            eps=selector.config.eps,
                        )
                        layer_readout_cosine[layer_index].update(cosine)
                        accumulators["steering_sufficiency_cosine"].update(cosine)
                    values = values.clamp(1e-6, 1.0 - 1e-6)
                    entropy = -(
                        values * values.log()
                        + (1.0 - values) * (1.0 - values).log()
                    )
                    accumulators["gate_entropy"].update(entropy)
                context = batch["attention_mask"] & ~(
                    batch["subject_mask"] | batch["object_mask"]
                )
                for layer_index, head_index, weights in captured_measurement_weights:
                    selected = weights[context]
                    base = selector.observable_weights(layer_index)[head_index]
                    layer_conditioning_delta[layer_index].update(
                        selected - base.unsqueeze(0)
                    )
                    active = (selected.abs() > selector.config.eps) & (
                        base.abs().unsqueeze(0) > selector.config.eps
                    )
                    if active.any():
                        layer_polarity_flip[layer_index].update(
                            (selected[active] * base.expand_as(selected)[active] < 0.0)
                            .to(selected.dtype)
                        )
                for layer_index, _head_index, angles in captured_relation_frame_angles:
                    layer_relation_frame_angle[layer_index].update(angles[context])

            keep_logits = _view_logits(
                model,
                batch,
                adapter,
                view="keep",
                random_seed=random_seed,
                detach_random=False,
            )
            drop_logits = _view_logits(
                model,
                batch,
                adapter,
                view="drop",
                random_seed=random_seed,
                detach_random=False,
            )
            keep_kl = _distillation_kl(keep_logits, full_logits)
            full_margin = correct_label_margin(full_logits, batch["labels"])
            drop_margin = correct_label_margin(drop_logits, batch["labels"])
            full_prediction = full_logits.argmax(dim=-1)
            keep_agreement = (keep_logits.argmax(dim=-1) == full_prediction).float()

            random_keep_kls: list[torch.Tensor] = []
            random_drop_margins: list[torch.Tensor] = []
            random_keep_agreements: list[torch.Tensor] = []
            for repeat in range(random_repeats):
                seed = random_seed + 10007 * batch_index + 97 * repeat
                random_keep = _view_logits(
                    model,
                    batch,
                    adapter,
                    view="random_keep",
                    random_seed=seed,
                    detach_random=True,
                )
                random_drop = _view_logits(
                    model,
                    batch,
                    adapter,
                    view="random_drop",
                    random_seed=seed,
                    detach_random=True,
                )
                random_keep_kls.append(_distillation_kl(random_keep, full_logits))
                random_drop_margins.append(
                    correct_label_margin(random_drop, batch["labels"])
                )
                random_keep_agreements.append(
                    (random_keep.argmax(dim=-1) == full_prediction).float()
                )
            random_keep_kl = torch.stack(random_keep_kls).mean(dim=0)
            random_drop_margin = torch.stack(random_drop_margins).mean(dim=0)
            random_keep_agreement = torch.stack(random_keep_agreements).mean(dim=0)
            keep_advantage = random_keep_kl - keep_kl
            drop_advantage = random_drop_margin - drop_margin

            accumulators["full_margin"].update(full_margin)
            accumulators["keep_kl"].update(keep_kl)
            accumulators["random_keep_kl"].update(random_keep_kl)
            accumulators["keep_advantage"].update(keep_advantage)
            accumulators["drop_margin_reduction"].update(full_margin - drop_margin)
            accumulators["random_drop_margin_reduction"].update(
                full_margin - random_drop_margin
            )
            accumulators["drop_advantage"].update(drop_advantage)
            accumulators["keep_prediction_agreement"].update(keep_agreement)
            accumulators["random_keep_prediction_agreement"].update(
                random_keep_agreement
            )
            accumulators["keep_win"].update((keep_advantage > 0.0).float())
            accumulators["drop_win"].update((drop_advantage > 0.0).float())
            num_batches += 1

    summaries = {name: accumulator.summary() for name, accumulator in accumulators.items()}
    keep_mean = summaries["keep_advantage"]["mean"]
    drop_mean = summaries["drop_advantage"]["mean"]
    keep_win = summaries["keep_win"]["mean"]
    drop_win = summaries["drop_win"]["mean"]
    selectivity_pass = bool(
        keep_mean is not None
        and drop_mean is not None
        and keep_win is not None
        and drop_win is not None
        and keep_mean > minimum_advantage
        and drop_mean > minimum_advantage
        and keep_win > 0.5
        and drop_win > 0.5
    )
    return {
        "num_batches": num_batches,
        "random_repeats": random_repeats,
        "random_seed": random_seed,
        "minimum_advantage": minimum_advantage,
        "evidence_measurement_mode": selector.config.evidence_measurement_mode,
        "selectivity_pass": selectivity_pass,
        "metrics": summaries,
        "layers": [
            {
                "layer_index": layer_index,
                "context_evidence": accumulator.summary(),
                "context_steering": layer_steering[layer_index].summary(),
                "steering_sufficiency_cosine": layer_readout_cosine[
                    layer_index
                ].summary(),
                "observable_weights": selector.observable_weights(layer_index)
                .detach()
                .cpu()
                .tolist(),
                "conditioning_gain": (
                    selector.conditioning_gains(layer_index).detach().cpu().tolist()
                    if selector.raw_conditioning_gains is not None
                    else None
                ),
                "frame_fusion_gain": (
                    selector.frame_fusion_gains(layer_index)
                    .detach()
                    .cpu()
                    .tolist()
                    if selector.raw_frame_fusion_gains is not None
                    else None
                ),
                "reliability_exponent": (
                    selector.reliability_exponents(layer_index)
                    .detach()
                    .cpu()
                    .tolist()
                    if selector.raw_reliability_exponents is not None
                    else None
                ),
                "conditioned_weight_delta": layer_conditioning_delta[
                    layer_index
                ].summary(),
                "conditioned_polarity_flip": layer_polarity_flip[
                    layer_index
                ].summary(),
                "relation_frame_angle": layer_relation_frame_angle[
                    layer_index
                ].summary(),
                "sharpness": selector.sharpness(layer_index).detach().cpu().tolist(),
                "sufficiency_observable_weights": (
                    selector.sufficiency_observable_weights(layer_index)
                    .detach()
                    .cpu()
                    .tolist()
                    if selector.sufficiency_observable_logits is not None
                    else None
                ),
                "sufficiency_sharpness": (
                    selector.sufficiency_sharpness(layer_index)
                    .detach()
                    .cpu()
                    .tolist()
                    if selector.raw_sufficiency_sharpness is not None
                    else None
                ),
            }
            for layer_index, accumulator in enumerate(layer_evidence)
        ],
    }


def diagnose_relation_evidence_task_alignment(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    adapter: AttentionScoreKernelAdapter,
) -> dict[str, Any]:
    """Measure whether high evidence agrees with the local task descent direction."""
    selector = adapter.score_kernel.evidence_selector
    if selector is None:
        raise ValueError("evidence task-alignment diagnostics require a selector")
    model.eval()
    adapter.score_kernel.eval()
    gradient_magnitude = [_ScalarAccumulator() for _ in range(selector.config.num_layers)]
    descent_cosine = [_ScalarAccumulator() for _ in range(selector.config.num_layers)]
    num_batches = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        with selector.capture_token_scores():
            logits = _view_logits(
                model,
                batch,
                adapter,
                view="full",
                random_seed=0,
                detach_random=False,
            )
            captured = selector.captured_token_scores()
            loss = F.cross_entropy(logits, batch["labels"])
            gradients = torch.autograd.grad(loss, captured, allow_unused=True)
            context = (
                batch["attention_mask"]
                & ~(batch["subject_mask"] | batch["object_mask"])
            )[:, None, :]
            for layer_index, (scores, gradient) in enumerate(
                zip(captured, gradients, strict=True)
            ):
                if gradient is None:
                    continue
                mask = context.expand_as(scores)
                gradient_magnitude[layer_index].update(gradient.masked_select(mask))
                weights = mask.to(scores.dtype)
                count = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
                centered_scores = scores - (scores * weights).sum(
                    dim=-1, keepdim=True
                ) / count
                centered_descent = -gradient - ((-gradient) * weights).sum(
                    dim=-1, keepdim=True
                ) / count
                cosine = F.cosine_similarity(
                    (centered_scores * weights).flatten(start_dim=1),
                    (centered_descent * weights).flatten(start_dim=1),
                    dim=-1,
                    eps=selector.config.eps,
                )
                descent_cosine[layer_index].update(cosine)
        num_batches += 1

    return {
        "num_batches": num_batches,
        "layers": [
            {
                "layer_index": layer_index,
                "evidence_gradient": gradient_magnitude[layer_index].summary(),
                "evidence_descent_cosine": descent_cosine[layer_index].summary(),
            }
            for layer_index in range(selector.config.num_layers)
        ],
    }


def diagnose_relation_evidence_measurement_frames(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    adapter: AttentionScoreKernelAdapter,
) -> dict[str, Any]:
    """Measure frame contributions and task gradients in a trained bank."""
    selector = adapter.score_kernel.evidence_selector
    if selector is None:
        raise ValueError("measurement-frame diagnostics require a selector")
    if selector.config.evidence_measurement_mode not in {
        "relation_frame_bank",
        "relation_frame_coherent",
    }:
        raise ValueError(
            "measurement-frame diagnostics require banked relation frames"
        )
    model.eval()
    adapter.score_kernel.eval()
    frame_names = ("z", "x")
    statistics = {
        (layer_index, head_index, frame_index): {
            name: _ScalarAccumulator()
            for name in (
                "contribution",
                "absolute_contribution",
                "contribution_l2",
                "task_gradient",
                "absolute_task_gradient",
                "task_gradient_l2",
                "task_descent_effect",
            )
        }
        for layer_index in range(selector.config.num_layers)
        for head_index in range(selector.config.num_heads)
        for frame_index in range(2)
    }
    channel_statistics = {
        (layer_index, head_index, channel_index, frame_index): {
            name: _ScalarAccumulator()
            for name in (
                "contribution",
                "absolute_contribution",
                "contribution_l2",
                "task_descent_effect",
            )
        }
        for layer_index in range(selector.config.num_layers)
        for head_index in range(selector.config.num_heads)
        for channel_index in range(len(EVIDENCE_CORRELATION_CHANNELS))
        for frame_index in range(2)
    }
    gate_statistics = {
        (layer_index, head_index): {
            "coherence_ratio": _ScalarAccumulator(),
            "effective_x_gate": _ScalarAccumulator(),
        }
        for layer_index in range(selector.config.num_layers)
        for head_index in range(selector.config.num_heads)
    }
    reliability_statistics = {
        (layer_index, head_index, frame_index): {
            "quality": _ScalarAccumulator(),
            "effective_gate": _ScalarAccumulator(),
        }
        for layer_index in range(selector.config.num_layers)
        for head_index in range(selector.config.num_heads)
        for frame_index in range(2)
    }
    num_batches = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        with selector.use_measurement_frame_view("full"):
            with selector.capture_token_scores():
                logits = _view_logits(
                    model,
                    batch,
                    adapter,
                    view="full",
                    random_seed=0,
                    detach_random=False,
                )
                captured = selector.captured_measurement_frame_contributions()
                captured_channels = {
                    (layer_index, head_index): contributions
                    for layer_index, head_index, contributions in (
                        selector.captured_correlation_channel_contributions()
                    )
                }
                captured_gates = selector.captured_coherence_gates()
                captured_reliability = selector.captured_reliability_gates()
                tensors = tuple(item[2] for item in captured)
                loss = F.cross_entropy(logits, batch["labels"])
                gradients = torch.autograd.grad(
                    loss,
                    tensors,
                    allow_unused=True,
                )
                context = batch["attention_mask"] & ~(
                    batch["subject_mask"] | batch["object_mask"]
                )
                for (
                    layer_index,
                    head_index,
                    contributions,
                ), gradient in zip(captured, gradients, strict=True):
                    if gradient is None:
                        continue
                    for frame_index in range(2):
                        values = contributions[..., frame_index].masked_select(context)
                        frame_gradient = gradient[..., frame_index].masked_select(
                            context
                        )
                        accumulators = statistics[
                            (layer_index, head_index, frame_index)
                        ]
                        accumulators["contribution"].update(values)
                        accumulators["absolute_contribution"].update(values.abs())
                        accumulators["contribution_l2"].update(
                            torch.linalg.vector_norm(values).reshape(1)
                        )
                        accumulators["task_gradient"].update(frame_gradient)
                        accumulators["absolute_task_gradient"].update(
                            frame_gradient.abs()
                        )
                        accumulators["task_gradient_l2"].update(
                            torch.linalg.vector_norm(frame_gradient).reshape(1)
                        )
                        accumulators["task_descent_effect"].update(
                            -frame_gradient * values
                        )
                        channel_contributions = captured_channels.get(
                            (layer_index, head_index)
                        )
                        if channel_contributions is not None:
                            for channel_index in range(
                                len(EVIDENCE_CORRELATION_CHANNELS)
                            ):
                                channel_values = channel_contributions[
                                    ..., channel_index, frame_index
                                ].masked_select(context)
                                channel_accumulators = channel_statistics[
                                    (
                                        layer_index,
                                        head_index,
                                        channel_index,
                                        frame_index,
                                    )
                                ]
                                channel_accumulators["contribution"].update(
                                    channel_values
                                )
                                channel_accumulators[
                                    "absolute_contribution"
                                ].update(channel_values.abs())
                                channel_accumulators["contribution_l2"].update(
                                    torch.linalg.vector_norm(
                                        channel_values
                                    ).reshape(1)
                                )
                                channel_accumulators[
                                    "task_descent_effect"
                                ].update(-frame_gradient * channel_values)
                for layer_index, head_index, gates in captured_gates:
                    selected = gates[context]
                    gate_statistics[(layer_index, head_index)][
                        "coherence_ratio"
                    ].update(selected[:, 0])
                    gate_statistics[(layer_index, head_index)][
                        "effective_x_gate"
                    ].update(selected[:, 1])
                for layer_index, head_index, gates in captured_reliability:
                    selected = gates[context]
                    for frame_index in range(2):
                        reliability_statistics[
                            (layer_index, head_index, frame_index)
                        ]["quality"].update(selected[:, frame_index, 0])
                        reliability_statistics[
                            (layer_index, head_index, frame_index)
                        ]["effective_gate"].update(
                            selected[:, frame_index, 1]
                        )
        num_batches += 1

    return {
        "num_batches": num_batches,
        "measurement_frame_view": "full",
        "correlation_mode": selector.config.evidence_correlation_mode,
        "layers": [
            {
                "layer_index": layer_index,
                "heads": [
                    {
                        "head_index": head_index,
                        "coherence_gate": {
                            name: accumulator.summary()
                            for name, accumulator in gate_statistics[
                                (layer_index, head_index)
                            ].items()
                        },
                        "born_reliability": {
                            frame_name: {
                                name: accumulator.summary()
                                for name, accumulator in reliability_statistics[
                                    (layer_index, head_index, frame_index)
                                ].items()
                            }
                            for frame_index, frame_name in enumerate(frame_names)
                        },
                        "frames": {
                            frame_name: {
                                name: accumulator.summary()
                                for name, accumulator in statistics[
                                    (layer_index, head_index, frame_index)
                                ].items()
                            }
                            for frame_index, frame_name in enumerate(frame_names)
                        },
                        "correlation_channels": {
                            channel_name: {
                                frame_name: {
                                    name: accumulator.summary()
                                    for name, accumulator in channel_statistics[
                                        (
                                            layer_index,
                                            head_index,
                                            channel_index,
                                            frame_index,
                                        )
                                    ].items()
                                }
                                for frame_index, frame_name in enumerate(
                                    frame_names
                                )
                            }
                            for channel_index, channel_name in enumerate(
                                EVIDENCE_CORRELATION_CHANNELS
                            )
                        },
                    }
                    for head_index in range(selector.config.num_heads)
                ],
            }
            for layer_index in range(selector.config.num_layers)
        ],
    }
