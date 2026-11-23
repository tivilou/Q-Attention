"""Frozen-base circuit steering with fixed mention weights and context mass."""
import math
import torch
from torch import nn


class NormalizedMassSteering(nn.Module):
    """Only beta is trainable. Geometry is cached from an immutable native head."""
    def __init__(self, native, circuit, eps=1e-6, smax=1e-3):
        super().__init__()
        self.native = native.eval().requires_grad_(False)
        self.circuit = circuit
        self.eps, self.smax = eps, smax
        theta = (math.pi * torch.tanh(native.effect)).detach().reshape(40, 12)
        def real_state(x):
            return torch.view_as_real(circuit(x.reshape(2, 3, 2)))
        jac = torch.vmap(torch.func.jacfwd(real_state))(theta).detach()
        d = torch.complex(jac[:, :, 0], jac[:, :, 1])
        phi = circuit(theta.reshape(40, 2, 3, 2)).detach()
        horizontal = d - phi[:, :, None] * torch.einsum('rd,rdk->rk', phi.conj(), d)[:, None]
        g = torch.einsum('rdj,rdk->rjk', horizontal.conj(), horizontal).real
        eigen, vectors = torch.linalg.eigh(g)
        regular = (eigen[:, 0] > 0) & (eigen[:, 0] / eigen[:, -1] >= 1e-12)
        # Use exact denominators on admitted branches; safe constants only elsewhere.
        w = (vectors * torch.where(regular[:, None], eigen, torch.ones_like(eigen)).rsqrt()[:, None]) @ vectors.transpose(-1, -2)
        for name, value in [('theta', theta), ('phi', phi), ('dphi', d), ('whitener', w), ('metric_regular', regular)]:
            self.register_buffer(name, value)
        self.beta = nn.Parameter(torch.zeros(40, dtype=theta.dtype, device=theta.device))

    def _row(self, psi, utilities, mention, native_logit, raw_only=False):
        r, dtype, device = 40, self.theta.dtype, self.theta.device
        count = int(mention.sum())
        if count > 10 or count == 0 or bool(mention.all()):
            zero = torch.zeros(r, dtype=dtype, device=device)
            return dict(logits=native_logit + self.beta * 0, direct=native_logit + self.beta * 0,
                        eligible=zero.bool(), admitted=zero.bool(), kappa=zero,
                        raw_error=zero, attention_error=zero, mass_error=zero, first_order_error=zero)
        overlap = psi.conj() @ self.phi.T
        probability = overlap.abs().square()
        j = 2 * (overlap.conj()[:, :, None] * torch.einsum('td,rdk->trk', psi.conj(), self.dphi)).real
        raw = probability + self.eps
        z = raw.sum(0)
        alpha = raw / z
        u = utilities.T
        gl = torch.einsum('tr,trk->rk', (u - (alpha * u).sum(0)) / z, j)
        constraint = j[mention].permute(1, 0, 2)
        if not raw_only:
            constraint = torch.cat((constraint, j[~mention].sum(0)[:, None]), 1)
        jw = constraint @ self.whitener
        left, singular, right = torch.linalg.svd(jw, full_matrices=False)
        ratio = singular[:, -1] / torch.where(singular[:, 0] > 0, singular[:, 0], torch.ones_like(singular[:, 0]))
        projector = torch.eye(12, dtype=dtype, device=device)[None] - right.transpose(-1, -2) @ right
        gradient = (self.whitener @ gl[:, :, None]).squeeze(-1)
        v = (projector @ gradient[:, :, None]).squeeze(-1)
        kappa = v.norm(dim=-1)
        gradient_norm = gradient.norm(dim=-1)
        safe_norm = torch.where(gradient_norm > 0, gradient_norm, torch.ones_like(gradient_norm))
        eligible = self.metric_regular & (ratio >= 1e-6) & (gradient_norm > 0) & (kappa / safe_norm >= 1e-8)
        direction = (self.whitener @ (v / torch.where(eligible, kappa, torch.ones_like(kappa))[:, None])[:, :, None]).squeeze(-1)
        direction = torch.where(eligible[:, None], direction, 0)
        inverse = self.whitener @ ((right.transpose(-1, -2) / torch.where(eligible[:, None], singular, torch.ones_like(singular))[:, None]) @ left.transpose(-1, -2))
        inverse = torch.where(eligible[:, None, None], inverse, 0)
        def quantities(theta):
            state = self.circuit(theta.reshape(40, 2, 3, 2))
            p = (psi.conj() @ state.T).abs().square()
            f = p[mention].T
            if not raw_only:
                f = torch.cat((f, p[~mention].sum(0)[:, None]), 1)
            a = (p + self.eps) / (p + self.eps).sum(0)
            return state, p, a, f
        baseline = probability[mention].T
        if not raw_only:
            baseline = torch.cat((baseline, probability[~mention].sum(0)[:, None]), 1)
        t = self.smax * self.beta.tanh()
        angles = self.theta + t[:, None] * direction
        for _ in range(2):
            _, _, _, f = quantities(angles)
            angles = angles - (inverse @ (f - baseline)[:, :, None]).squeeze(-1)
        state, p, a, f = quantities(angles)
        raw_error = (f - baseline).abs().amax(-1)
        attention_error = (a[mention] - alpha[mention]).abs().amax(0)
        mass_error = (a[~mention].sum(0) - alpha[~mention].sum(0)).abs()
        norm_error = (state.abs().square().sum(-1) - 1).abs()
        admitted = eligible & (raw_error <= 1e-12) & (norm_error <= 1e-12) & (angles.abs().amax(-1) < math.pi)
        if not raw_only:
            admitted = admitted & (attention_error <= 1e-12) & (mass_error <= 1e-12)
        finite = native_logit + ((a - alpha) * u).sum(0)
        direct = native_logit + t * kappa
        return dict(logits=torch.where(admitted, finite, native_logit),
                    direct=torch.where(admitted, direct, native_logit), eligible=eligible,
                    admitted=admitted, kappa=kappa, raw_error=raw_error,
                    attention_error=attention_error, mass_error=mass_error,
                    first_order_error=finite - direct)

    def forward(self, h, roles, mask, details=False, raw_only=False):
        with torch.no_grad():
            baseline = self.native(h, roles, mask, details=True)
        rows = []
        for b in range(len(h)):
            valid = mask[b]
            mention = (roles[b, valid, 2] != 0) | (roles[b, valid, 3] != 0)
            rows.append(self._row(baseline['psi'][b, valid], baseline['utilities'][b][:, valid], mention,
                                  baseline['logits'][b], raw_only))
        result = {k: torch.stack([x[k] for x in rows]) for k in rows[0]}
        result['native_logits'] = baseline['logits']
        return result if details else result['logits']
