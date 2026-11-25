"""Explicit dense-batch VAT loss; no changes to native model forward/CE.

Known VAT, one finite-difference power iteration. A caller-owned torch.Generator
is mandatory so independent training lanes never race over a global RNG.
"""
import torch
from torch.nn import functional as F


def kl_from_logits(target_logits, prediction_logits):
    target_log = target_logits.detach().double().log_softmax(-1)
    prediction_log = prediction_logits.double().log_softmax(-1)
    return (target_log.exp() * (target_log - prediction_log)).sum(-1).mean()


def masked_unit(value, mask):
    value = torch.where(mask[..., None], value, torch.zeros_like(value))
    norm = value.double().flatten(1).norm(dim=1)
    denominator = torch.where(norm > 0, norm, torch.ones_like(norm))
    return (value.double() / denominator[:, None, None]).to(value.dtype), norm


def objective(head, inputs, labels, *, generator, directed=True,
              alpha=1., epsilon=.01, xi=.001, initial_direction=None):
    if alpha < 0 or epsilon < 0 or xi <= 0:
        raise ValueError('Invalid consistency hyperparameters')
    h, roles, mask = inputs
    logits = head(*inputs)
    ce = F.cross_entropy(logits, labels)
    if alpha == 0:
        return ce, {'ce': ce.detach(), 'kl': ce.detach().new_zeros(()),
                    'disabled': True}
    # Inputs are frozen features, not learnable representations in this graph.
    if h.requires_grad or roles.requires_grad:
        raise ValueError('Frozen input tensors expected')
    if mask.dtype != torch.bool or not bool(mask.any(-1).all()):
        raise ValueError('Each example needs valid tokens')
    if initial_direction is None:
        initial_direction = torch.randn(h.shape, device=h.device, dtype=h.dtype,
                                        generator=generator)
    if initial_direction.shape != h.shape:
        raise ValueError('Direction shape differs')
    direction, initial_norm = masked_unit(initial_direction.detach(), mask)
    _, h_norm = masked_unit(h, mask)
    xi_radius = (xi * h_norm).to(h.dtype)[:, None, None]
    power_norm = None
    if directed:
        probe = direction.detach().requires_grad_(True)
        probe_logits = head(h + xi_radius * probe, roles, mask)
        probe_kl = kl_from_logits(logits, probe_logits)
        gradient, = torch.autograd.grad(probe_kl, probe, create_graph=False)
        direction, power_norm = masked_unit(gradient.detach(), mask)
    direction = direction.detach()
    perturbation = (epsilon * h_norm).to(h.dtype)[:, None, None] * direction
    perturbed_logits = head(h + perturbation, roles, mask)
    penalty = kl_from_logits(logits, perturbed_logits)
    total = ce + alpha * penalty
    if not bool(torch.isfinite(total)):
        raise ValueError('Nonfinite consistency objective')
    return total, {'ce': ce.detach(), 'kl': penalty.detach(), 'disabled': False,
                   'initial_zero_norm': int((initial_norm == 0).sum()),
                   'power_zero_norm': None if power_norm is None else int((power_norm == 0).sum()),
                   'power_norm': None if power_norm is None else power_norm.detach(),
                   'direction': direction, 'perturbation': perturbation.detach(),
                   'target_detached': True}


def backward(head, inputs, labels, **options):
    loss, record = objective(head, inputs, labels, **options)
    loss.backward()
    return loss.detach(), record


def resolve_backward(plugin, baseline, enabled=False):
    """Off returns the very same callable, with no wrapper or extra forward."""
    if not enabled:
        return baseline
    callback = getattr(plugin, 'backward', None)
    if not callable(callback):
        raise ValueError('Explicit scientific loss callback is missing')
    return callback
