"""Training and diagnostics for identifiable observable-expert routing."""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from q_attention.adapters import AttentionScoreHookConfig, AttentionScoreKernelAdapter

from .relation_steering import move_batch


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


def _hook_config(
    batch: dict[str, torch.Tensor],
    *,
    routing_mode: str,
) -> AttentionScoreHookConfig:
    return AttentionScoreHookConfig(
        attention_mask=batch["attention_mask"],
        subject_mask=batch["subject_mask"],
        object_mask=batch["object_mask"],
        routing_mode=routing_mode,
    )


def _view_logits(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    adapter: AttentionScoreKernelAdapter,
    *,
    routing_mode: str,
) -> torch.Tensor:
    with adapter.steering(_hook_config(batch, routing_mode=routing_mode)):
        return model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["subject_mask"],
            batch["object_mask"],
        )


def _correct_label_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    correct = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels[:, None], torch.finfo(logits.dtype).min)
    return correct - masked.max(dim=-1).values


def _routing_utility_alignment_loss(
    task_loss: torch.Tensor,
    router: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align routed probability deviations with detached task descent utilities."""
    captured = router.captured_probabilities()
    masks = router.captured_probability_masks()
    probabilities = [item[2] for item in captured]
    gradients = torch.autograd.grad(
        task_loss,
        probabilities,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    losses: list[torch.Tensor] = []
    alignments: list[torch.Tensor] = []
    for probability, gradient, mask in zip(
        probabilities,
        gradients,
        masks,
        strict=True,
    ):
        if gradient is None:
            continue
        probability_centered = probability - probability.mean(dim=-1, keepdim=True)
        probability_selected = probability
        utility = -gradient.detach()
        utility_centered = utility - utility.mean(dim=-1, keepdim=True)
        if mask is not None:
            probability_centered = probability_centered[mask]
            probability_selected = probability_selected[mask]
            utility_centered = utility_centered[mask]
        probability_selected = probability_selected.reshape(
            -1, probability.shape[-1]
        )
        probability_centered = probability_centered.reshape(
            -1, probability.shape[-1]
        )
        utility_centered = utility_centered.reshape(-1, probability.shape[-1])
        active = (
            torch.linalg.vector_norm(probability_centered, dim=-1)
            > router.config.eps
        ) & (
            torch.linalg.vector_norm(utility_centered, dim=-1)
            > router.config.eps
        )
        if active.any():
            probability_active = probability_selected[active]
            utility_active = utility_centered[active]
            utility_scale = torch.sqrt(
                utility_active.square().mean(dim=-1, keepdim=True)
            ).clamp_min(router.config.eps)
            target = torch.softmax(utility_active / utility_scale, dim=-1)
            losses.append(
                F.kl_div(
                    probability_active.clamp_min(router.config.eps).log(),
                    target,
                    reduction="batchmean",
                )
            )
            alignments.append(
                F.cosine_similarity(
                    probability_centered[active],
                    utility_active,
                    dim=-1,
                    eps=router.config.eps,
                )
            )
    if not losses:
        zero = task_loss.new_zeros(())
        return zero, zero
    alignment = torch.cat(alignments).mean()
    return torch.stack(losses).mean(), alignment


def expert_routing_objective(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    adapter: AttentionScoreKernelAdapter,
    *,
    information_weight: float,
    direction_diversity_weight: float = 0.0,
    utility_alignment_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    router = adapter.score_kernel.expert_router
    if router is None:
        raise ValueError("expert-routing training requires an attached router")
    if information_weight < 0:
        raise ValueError("information_weight must be non-negative")
    if direction_diversity_weight < 0:
        raise ValueError("direction_diversity_weight must be non-negative")
    if utility_alignment_weight < 0:
        raise ValueError("utility_alignment_weight must be non-negative")
    with router.capture_routing():
        logits = _view_logits(
            model,
            batch,
            adapter,
            routing_mode="learned",
        )
        task_loss = F.cross_entropy(logits, batch["labels"])
        information = router.information_components()
        information_loss = (
            -information["mutual_information"]
            + information["dead_expert_barrier"]
        )
        direction_diversity_loss = router.direction_diversity_loss()
        if utility_alignment_weight > 0:
            utility_alignment_loss, utility_alignment = (
                _routing_utility_alignment_loss(task_loss, router)
            )
        else:
            utility_alignment_loss = task_loss.new_zeros(())
            utility_alignment = task_loss.new_zeros(())
        objective = (
            task_loss
            + information_weight * information_loss
            + direction_diversity_weight * direction_diversity_loss
            + utility_alignment_weight * utility_alignment_loss
        )
    return objective, {
        "task_loss": task_loss,
        "information_loss": information_loss,
        "direction_diversity_loss": direction_diversity_loss,
        "utility_alignment_loss": utility_alignment_loss,
        "utility_alignment": utility_alignment,
        **information,
    }


def _routing_diagnostic_bucket(num_experts: int) -> dict[str, Any]:
    return {
        "usage": torch.zeros(num_experts, dtype=torch.float64),
        "usage_count": 0,
        "conditional_entropy": _ScalarAccumulator(),
        "max_probability": _ScalarAccumulator(),
        "expert_cosine": _ScalarAccumulator(),
        "expert_rms": [_ScalarAccumulator() for _ in range(num_experts)],
    }


def _summarize_routing_bucket(
    bucket: dict[str, Any],
    *,
    num_experts: int,
    eps: float,
    max_usage_deviation: float,
    max_expert_abs_cosine: float,
    min_mutual_information: float,
) -> tuple[dict[str, Any], bool]:
    usage = bucket["usage"] / max(bucket["usage_count"], 1)
    usage = usage / usage.sum().clamp_min(eps)
    safe_usage = usage.clamp_min(eps)
    marginal_entropy = float(-(safe_usage * safe_usage.log()).sum().item())
    conditional = bucket["conditional_entropy"].summary()
    conditional_mean = float(conditional["mean"] or 0.0)
    mutual_information = marginal_entropy - conditional_mean
    deviation = float((usage - 1.0 / num_experts).abs().max().item())
    cosine = bucket["expert_cosine"].summary()
    mean_abs_cosine = (
        bucket["expert_cosine"].absolute_total / bucket["expert_cosine"].count
        if bucket["expert_cosine"].count
        else None
    )
    mechanism_pass = bool(
        mutual_information > min_mutual_information
        and deviation < max_usage_deviation
        and mean_abs_cosine is not None
        and mean_abs_cosine < max_expert_abs_cosine
        and conditional_mean < math.log(num_experts) - min_mutual_information
    )
    return {
        "mechanism_pass": mechanism_pass,
        "usage": usage.tolist(),
        "max_usage_deviation": deviation,
        "conditional_entropy": conditional,
        "marginal_entropy": marginal_entropy,
        "mutual_information": mutual_information,
        "max_probability": bucket["max_probability"].summary(),
        "expert_cross_cosine": cosine,
        "expert_mean_abs_cosine": mean_abs_cosine,
        "expert_rms": [item.summary() for item in bucket["expert_rms"]],
    }, mechanism_pass


def diagnose_relation_expert_routing(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    adapter: AttentionScoreKernelAdapter,
    max_usage_deviation: float = 0.15,
    max_expert_abs_cosine: float = 0.80,
    min_mutual_information: float = 1e-3,
) -> dict[str, Any]:
    """Measure expert diversity, routing information, balance, and task effect."""
    router = adapter.score_kernel.expert_router
    if router is None:
        raise ValueError("expert-routing diagnostics require an attached router")
    model.eval()
    adapter.score_kernel.eval()
    num_experts = router.config.num_experts
    layers = [
        _routing_diagnostic_bucket(num_experts)
        for _ in range(router.config.num_layers)
    ]
    heads = [
        [
            _routing_diagnostic_bucket(num_experts)
            for _ in range(router.config.num_heads)
        ]
        for _ in range(router.config.num_layers)
    ]
    learned_uniform_kl = _ScalarAccumulator()
    margin_delta = _ScalarAccumulator()
    prediction_change = _ScalarAccumulator()
    num_batches = 0

    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            with router.capture_routing():
                learned_logits = _view_logits(
                    model,
                    batch,
                    adapter,
                    routing_mode="learned",
                )
                captured_probabilities = router.captured_probabilities()
                captured_probability_masks = router.captured_probability_masks()
                captured_deltas = router.captured_expert_deltas()
                for (
                    layer_index,
                    head_index,
                    probabilities,
                ), probability_mask in zip(
                    captured_probabilities,
                    captured_probability_masks,
                    strict=True,
                ):
                    if probability_mask is not None:
                        probabilities = probabilities[probability_mask]
                    probabilities = probabilities.reshape(-1, num_experts)
                    for bucket in (layers[layer_index], heads[layer_index][head_index]):
                        bucket["usage"] += probabilities.double().sum(dim=0).cpu()
                        bucket["usage_count"] += probabilities.shape[0]
                        safe = probabilities.clamp_min(router.config.eps)
                        bucket["conditional_entropy"].update(
                            -(safe * safe.log()).sum(dim=-1)
                        )
                        bucket["max_probability"].update(
                            probabilities.max(dim=-1).values
                        )
                valid = (
                    batch["attention_mask"][:, None, :, None]
                    & batch["attention_mask"][:, None, None, :]
                )
                key_mask = batch["attention_mask"][:, None, None, :]
                key_count = key_mask.sum(dim=-1, keepdim=True).clamp_min(1)
                for layer_index, head_index, deltas in captured_deltas:
                    centered = deltas - (deltas * key_mask).sum(
                        dim=-1, keepdim=True
                    ) / key_count
                    centered = centered * valid
                    vectors = centered.flatten(start_dim=2)
                    for bucket in (layers[layer_index], heads[layer_index][head_index]):
                        for expert_index in range(num_experts):
                            bucket["expert_rms"][expert_index].update(
                                centered[:, expert_index].masked_select(valid[:, 0])
                            )
                        for left in range(num_experts - 1):
                            for right in range(left + 1, num_experts):
                                bucket["expert_cosine"].update(
                                    F.cosine_similarity(
                                        vectors[:, left],
                                        vectors[:, right],
                                        dim=-1,
                                        eps=router.config.eps,
                                    )
                                )

            uniform_logits = _view_logits(
                model,
                batch,
                adapter,
                routing_mode="uniform",
            )
            target = torch.softmax(uniform_logits, dim=-1)
            learned_uniform_kl.update(
                torch.sum(
                    target
                    * (
                        target.clamp_min(1e-12).log()
                        - torch.log_softmax(learned_logits, dim=-1)
                    ),
                    dim=-1,
                )
            )
            margin_delta.update(
                _correct_label_margin(learned_logits, batch["labels"])
                - _correct_label_margin(uniform_logits, batch["labels"])
            )
            prediction_change.update(
                (learned_logits.argmax(dim=-1) != uniform_logits.argmax(dim=-1)).float()
            )
            num_batches += 1

    layer_payloads: list[dict[str, Any]] = []
    mechanism_pass = True
    for layer_index, layer in enumerate(layers):
        aggregate, aggregate_pass = _summarize_routing_bucket(
            layer,
            num_experts=num_experts,
            eps=router.config.eps,
            max_usage_deviation=max_usage_deviation,
            max_expert_abs_cosine=max_expert_abs_cosine,
            min_mutual_information=min_mutual_information,
        )
        head_payloads: list[dict[str, Any]] = []
        head_passes: list[bool] = []
        for head_index, head in enumerate(heads[layer_index]):
            head_payload, head_pass = _summarize_routing_bucket(
                head,
                num_experts=num_experts,
                eps=router.config.eps,
                max_usage_deviation=max_usage_deviation,
                max_expert_abs_cosine=max_expert_abs_cosine,
                min_mutual_information=min_mutual_information,
            )
            head_payloads.append({"head_index": head_index, **head_payload})
            head_passes.append(head_pass)
        layer_pass = aggregate_pass and all(head_passes)
        mechanism_pass = mechanism_pass and layer_pass
        layer_payloads.append(
            {
                "layer_index": layer_index,
                **aggregate,
                "mechanism_pass": layer_pass,
                "aggregate_mechanism_pass": aggregate_pass,
                "heads": head_payloads,
                "routing_gains": router.gains(layer_index).detach().cpu().tolist(),
            }
        )
    return {
        "num_batches": num_batches,
        "mechanism_pass": mechanism_pass,
        "thresholds": {
            "max_usage_deviation": max_usage_deviation,
            "max_expert_abs_cosine": max_expert_abs_cosine,
            "min_mutual_information": min_mutual_information,
        },
        "learned_vs_uniform": {
            "prediction_kl": learned_uniform_kl.summary(),
            "correct_margin_delta": margin_delta.summary(),
            "prediction_change_rate": prediction_change.summary(),
        },
        "layers": layer_payloads,
    }


def diagnose_relation_routing_task_alignment(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    adapter: AttentionScoreKernelAdapter,
) -> dict[str, Any]:
    """Compare routed probability deviations with the task descent direction."""
    router = adapter.score_kernel.expert_router
    if router is None:
        raise ValueError("routing task-alignment diagnostics require a router")
    model.eval()
    adapter.score_kernel.eval()
    gradients = [_ScalarAccumulator() for _ in range(router.config.num_layers)]
    cosines = [_ScalarAccumulator() for _ in range(router.config.num_layers)]
    num_batches = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        with router.capture_routing():
            logits = _view_logits(
                model,
                batch,
                adapter,
                routing_mode="learned",
            )
            captured = router.captured_probabilities()
            captured_masks = router.captured_probability_masks()
            loss = F.cross_entropy(logits, batch["labels"])
            probability_tensors = [item[2] for item in captured]
            probability_gradients = torch.autograd.grad(
                loss,
                probability_tensors,
                allow_unused=True,
            )
            for (
                layer_index,
                _head_index,
                probabilities,
            ), probability_mask, gradient in zip(
                captured,
                captured_masks,
                probability_gradients,
                strict=True,
            ):
                if gradient is None:
                    continue
                if probability_mask is not None:
                    probabilities = probabilities[probability_mask]
                    gradient = gradient[probability_mask]
                probabilities = probabilities.reshape(-1, router.config.num_experts)
                gradient = gradient.reshape(-1, router.config.num_experts)
                gradients[layer_index].update(gradient)
                uniform = 1.0 / router.config.num_experts
                routing_direction = probabilities - uniform
                descent = -gradient
                descent = descent - descent.mean(dim=-1, keepdim=True)
                cosines[layer_index].update(
                    F.cosine_similarity(
                        routing_direction,
                        descent,
                        dim=-1,
                        eps=router.config.eps,
                    )
                )
        num_batches += 1
    return {
        "num_batches": num_batches,
        "layers": [
            {
                "layer_index": layer_index,
                "routing_gradient": gradients[layer_index].summary(),
                "routing_descent_cosine": cosines[layer_index].summary(),
            }
            for layer_index in range(router.config.num_layers)
        ],
    }


def _direction_role_accumulators() -> dict[str, _ScalarAccumulator]:
    return {
        "routed_alignment": _ScalarAccumulator(),
        "uniform_alignment": _ScalarAccumulator(),
        "oracle_alignment": _ScalarAccumulator(),
        "routing_gain_over_uniform": _ScalarAccumulator(),
        "oracle_regret": _ScalarAccumulator(),
        "selected_expert_alignment": _ScalarAccumulator(),
        "top_expert_match": _ScalarAccumulator(),
        "probability_utility_cosine": _ScalarAccumulator(),
    }


def diagnose_relation_expert_direction_alignment(
    model: nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    adapter: AttentionScoreKernelAdapter,
) -> dict[str, Any]:
    """Compare each observable expert with the local task descent direction."""
    router = adapter.score_kernel.expert_router
    if router is None:
        raise ValueError("expert-direction diagnostics require an attached router")
    model.eval()
    adapter.score_kernel.eval()
    num_layers = router.config.num_layers
    num_heads = router.config.num_heads
    num_experts = router.config.num_experts
    roles = ("all", "subject", "object", "context")
    expert_stats = [
        [
            [
                {
                    "alignment": _ScalarAccumulator(),
                    "positive_alignment": _ScalarAccumulator(),
                    "direction_norm": _ScalarAccumulator(),
                    "descent_norm": _ScalarAccumulator(),
                }
                for _ in range(num_experts)
            ]
            for _ in range(num_heads)
        ]
        for _ in range(num_layers)
    ]
    role_stats = [
        [
            {role: _direction_role_accumulators() for role in roles}
            for _ in range(num_heads)
        ]
        for _ in range(num_layers)
    ]
    num_batches = 0

    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        with router.capture_routing():
            logits = _view_logits(
                model,
                batch,
                adapter,
                routing_mode="learned",
            )
            captured_probabilities = router.captured_probabilities()
            captured_probability_masks = router.captured_probability_masks()
            captured_deltas = router.captured_expert_deltas()
            loss = F.cross_entropy(logits, batch["labels"])
            delta_tensors = [item[2] for item in captured_deltas]
            delta_gradients = torch.autograd.grad(
                loss,
                delta_tensors,
                allow_unused=True,
            )

            for (
                layer_index,
                head_index,
                probabilities,
            ), probability_mask, (
                delta_layer,
                delta_head,
                deltas,
            ), gradient in zip(
                captured_probabilities,
                captured_probability_masks,
                captured_deltas,
                delta_gradients,
                strict=True,
            ):
                if (layer_index, head_index) != (delta_layer, delta_head):
                    raise RuntimeError("captured routing tensors are misaligned")
                if gradient is None:
                    continue
                active_queries = probability_mask
                if active_queries is None:
                    active_queries = batch["attention_mask"]
                    if adapter.score_kernel.config.query_scope == "entities":
                        active_queries = active_queries & (
                            batch["subject_mask"] | batch["object_mask"]
                        )
                key_mask = batch["attention_mask"][:, None, None, :].to(
                    deltas.dtype
                )
                query_mask = active_queries[:, None, :, None].to(deltas.dtype)
                valid = query_mask * key_mask
                key_count = key_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
                direction = deltas - (
                    (deltas * key_mask).sum(dim=-1, keepdim=True) / key_count
                )
                direction = direction * valid
                descent = -gradient
                descent = descent - (
                    (descent * key_mask).sum(dim=-1, keepdim=True) / key_count
                )
                descent = descent * valid
                alignment = F.cosine_similarity(
                    direction,
                    descent,
                    dim=-1,
                    eps=router.config.eps,
                )
                direction_norm = torch.linalg.vector_norm(direction, dim=-1)
                descent_norm = torch.linalg.vector_norm(descent, dim=-1)
                utilities = alignment.permute(0, 2, 1)
                if probabilities.ndim == 2:
                    query_probabilities = probabilities[:, None, :].expand(
                        -1, utilities.shape[1], -1
                    )
                else:
                    query_probabilities = probabilities

                for expert_index in range(num_experts):
                    stats = expert_stats[layer_index][head_index][expert_index]
                    selected = alignment[:, expert_index][active_queries]
                    stats["alignment"].update(selected)
                    stats["positive_alignment"].update((selected > 0.0).float())
                    stats["direction_norm"].update(
                        direction_norm[:, expert_index][active_queries]
                    )
                    stats["descent_norm"].update(
                        descent_norm[:, expert_index][active_queries]
                    )

                role_masks = {
                    "all": active_queries,
                    "subject": active_queries & batch["subject_mask"],
                    "object": active_queries & batch["object_mask"],
                    "context": active_queries
                    & ~(batch["subject_mask"] | batch["object_mask"]),
                }
                for role, role_mask in role_masks.items():
                    if not role_mask.any():
                        continue
                    role_utilities = utilities[role_mask]
                    role_probabilities = query_probabilities[role_mask]
                    routed = (role_probabilities * role_utilities).sum(dim=-1)
                    uniform = role_utilities.mean(dim=-1)
                    oracle = role_utilities.max(dim=-1).values
                    selected_expert = role_utilities.gather(
                        1,
                        role_probabilities.argmax(dim=-1, keepdim=True),
                    ).squeeze(1)
                    probability_centered = role_probabilities - role_probabilities.mean(
                        dim=-1, keepdim=True
                    )
                    utility_centered = role_utilities - role_utilities.mean(
                        dim=-1, keepdim=True
                    )
                    probability_utility_cosine = F.cosine_similarity(
                        probability_centered,
                        utility_centered,
                        dim=-1,
                        eps=router.config.eps,
                    )
                    stats = role_stats[layer_index][head_index][role]
                    stats["routed_alignment"].update(routed)
                    stats["uniform_alignment"].update(uniform)
                    stats["oracle_alignment"].update(oracle)
                    stats["routing_gain_over_uniform"].update(routed - uniform)
                    stats["oracle_regret"].update(oracle - routed)
                    stats["selected_expert_alignment"].update(selected_expert)
                    stats["top_expert_match"].update(
                        (
                            role_probabilities.argmax(dim=-1)
                            == role_utilities.argmax(dim=-1)
                        ).float()
                    )
                    stats["probability_utility_cosine"].update(
                        probability_utility_cosine
                    )
        num_batches += 1

    return {
        "num_batches": num_batches,
        "routing_conditioning": router.config.routing_conditioning,
        "layers": [
            {
                "layer_index": layer_index,
                "heads": [
                    {
                        "head_index": head_index,
                        "experts": [
                            {
                                "expert_index": expert_index,
                                **{
                                    name: accumulator.summary()
                                    for name, accumulator in expert_stats[
                                        layer_index
                                    ][head_index][expert_index].items()
                                },
                            }
                            for expert_index in range(num_experts)
                        ],
                        "query_roles": {
                            role: {
                                name: accumulator.summary()
                                for name, accumulator in role_stats[layer_index][
                                    head_index
                                ][role].items()
                            }
                            for role in roles
                        },
                    }
                    for head_index in range(num_heads)
                ],
            }
            for layer_index in range(num_layers)
        ],
    }
