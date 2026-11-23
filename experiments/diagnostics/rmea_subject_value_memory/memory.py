"""Trainable subject-addressed signed value read; native flat state is retained."""
import torch
from baselines.retacred.rmea import RMEAHead, EPS

MODES = ('coherent', 'diagonal', 'native')


class AddressedMemoryHead(RMEAHead):
    def __init__(self, readout):
        if readout not in MODES:
            raise ValueError('Unknown addressed-value condition')
        super().__init__('quantum')
        self.readout = readout

    def forward(self, h, roles, mask, details=False):
        d = super().forward(h, roles, mask, details=True)
        if self.readout == 'native':
            return d if details else d['logits']
        if not torch.isin(roles[..., 2][mask], h.new_tensor([0., 1.])).all():
            raise ValueError('Original binary subject-span flag required')
        subject = mask & (roles[..., 2] == 1)
        write = mask & ~subject
        if not subject.any(-1).all() or not write.any(-1).all():
            raise ValueError('Subject and write tokens required for every row')
        q = (d['probability'] + EPS) * subject[:, None, :]
        beta = q / q.sum(-1, keepdim=True)
        # Gather only subject words: O(B*S*T*d), not a dense T-by-T kernel.
        t = h.shape[1]
        slots = torch.arange(t, device=h.device).expand(len(h), -1)
        ids = slots.masked_fill(~subject, t).sort(-1).values[:, :int(subject.sum(-1).max())]
        ids = ids.clamp_max(t - 1)
        psi = d['psi']
        query = psi.gather(1, ids[..., None].expand(-1, -1, 8))
        if self.readout == 'coherent':
            kernel = torch.einsum('bsd,btd->bst', query.conj(), psi).abs().square()
        else:
            kernel = torch.einsum('bsd,btd->bst', query.abs().square(), psi.abs().square())
        b = beta.gather(-1, ids[:, None, :].expand(-1, 40, -1))
        # Clamped padding slots must have zero weight even when t-1 is a subject.
        occupied = torch.arange(ids.shape[1], device=h.device)[None, :] < subject.sum(-1)[:, None]
        b = b * occupied[:, None, :]
        weights = 8 * torch.einsum('brs,bst->brt', b, kernel)
        weights = weights * write[:, None, :] / write.sum(-1)[:, None, None]
        logits = (weights * d['utilities']).sum(-1) + self.bias
        if not details:
            return logits
        return dict(logits=logits, query_alpha=beta, effective_read_weights=weights,
                    subject_mask=subject, write_mask=write, native=d)


def fixtures(initial, inputs):
    """Only changed readout: direct operator parity, padding and gradients."""
    from torch.nn import functional as F
    errors = {}
    for mode in MODES[:2]:
        model = AddressedMemoryHead(mode).double().to(inputs[0].device)
        model.load_state_dict(initial)
        actual = model(*inputs, details=True)
        d = actual['native']; psi = d['psi']; u = d['utilities']
        rho = psi[..., :, None] * psi.conj()[..., None, :]
        q = torch.einsum('brt,btij->brij', actual['query_alpha'].to(rho.dtype), rho)
        w = torch.einsum('brt,btij->brij',
            (u * actual['write_mask'][:, None, :]).to(rho.dtype), rho)
        w = w / actual['write_mask'].sum(-1)[:, None, None, None]
        if mode == 'coherent':
            direct = 8 * torch.einsum('brij,brji->br', q, w).real + model.bias
        else:
            direct = 8 * (q.diagonal(dim1=-2, dim2=-1) * w.diagonal(dim1=-2, dim2=-1)).sum(-1).real + model.bias
        error = float((actual['logits'] - direct).abs().max()); errors[mode] = error
        ga = torch.autograd.grad(actual['logits'].square().mean(), tuple(model.parameters()), retain_graph=True)
        gd = torch.autograd.grad(direct.square().mean(), tuple(model.parameters()))
        errors[mode + '_gradient'] = max(float((a - b).abs().max()) for a, b in zip(ga, gd))
        padded = (F.pad(inputs[0], (0, 0, 0, 2), value=.123),
                  F.pad(inputs[1], (0, 0, 0, 2), value=1.), F.pad(inputs[2], (0, 2), value=False))
        errors[mode + '_padding'] = float((model(*padded) - actual['logits']).abs().max())
    assert max(errors.values()) < 1e-8
    return dict(status='PASS', errors=errors, scope='New readout parity/padding/gradient only; real first update checked in training')


def retrieval_fixture():
    """Exact finite balanced population; known covariance/bottleneck tools."""
    import itertools
    import numpy as np
    d = 8
    had = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2)
    had = np.kron(np.kron(had, had), had)
    states = np.concatenate((np.eye(d), had))
    choices = []
    for positive in itertools.combinations(range(d), d // 2):
        x = -np.ones(d); x[list(positive)] = 1; choices.append(x)
    payload = np.array([np.r_[a, b] for a in choices for b in choices])
    kernel = (states @ states.T)**2
    intact = payload @ kernel.T
    diagonal = states**2
    dephased = payload @ (diagonal @ diagonal.T).T
    covariance = payload.T @ payload / len(payload)
    projector = np.eye(2*d)
    projector[:d, :d] -= np.ones((d, d))/d
    projector[d:, d:] -= np.ones((d, d))/d
    error = float(np.max(abs(intact - payload)))
    covariance_error = float(np.max(abs(covariance - d/(d-1)*projector)))
    mse = float(np.mean((dephased - payload)**2))
    assert error < 1e-12 and covariance_error < 1e-12 and abs(mse - .5) < 1e-12
    return dict(status='PASS', population=len(payload), queries=2*d,
        intact_max_error=error, covariance_max_error=covariance_error,
        particular_diagonal_mse=mse, general_normalized_query_lower_bound=.5,
        assumptions='Payload-independent normalized diagonal queries; arbitrary payload-dependent memory; balanced independent banks; ideal encodings. Not actual contextualized text or finite circuit realization.')
