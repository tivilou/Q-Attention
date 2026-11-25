"""Focused gate checks and compact metrics; no runner/lifecycle ownership."""
import copy
import math
import numpy as np
import torch
from torch.nn import functional as F
from baselines.retacred.rmea import EPS
from plugin import MODES, CorroborationHead, initialize, make_head, sides


def numpy_circuit(angles):
    u = np.eye(8, dtype=complex)
    for layer in angles:
        for wire, (y, z) in enumerate(layer):
            rot = np.diag(np.exp(np.array([-.5j*z, .5j*z]))) @ np.array(
                [[np.cos(y/2), -np.sin(y/2)], [np.sin(y/2), np.cos(y/2)]])
            u = np.kron(np.kron(np.eye(2**wire), rot), np.eye(2**(2-wire))) @ u
        for control, target in ((0, 1), (1, 2), (2, 0)):
            p = [i ^ ((1 << (2-target)) if (i >> (2-control)) & 1 else 0) for i in range(8)]
            u = u[p]
    return u[:, 0]


def reference(model, inputs):
    h, roles, mask = [x.detach().cpu().numpy() for x in inputs]
    a = model.head
    get = lambda t: t.detach().cpu().numpy()
    k = np.concatenate((h, roles), -1) @ get(a.key.weight).T + get(a.key.bias)
    q = get(a.query)
    w = np.array([3-math.sqrt(2), 3+math.sqrt(2)]*6)/24
    base = np.exp(-((k[:, None]-q[None, :, None])**2*w).sum(-1))
    g = np.ones(mask.shape)
    for b in range(len(h)):
        subject = mask[b] & (abs(roles[b, :, 0]) <= abs(roles[b, :, 1]))
        obj = np.flatnonzero(mask[b] & ~subject)
        if not len(obj) or model.kind == 'gaussian_base':
            continue
        states = np.array([numpy_circuit((math.pi*np.tanh(x)).reshape(2, 3, 2)) for x in k[b]])
        for i in np.flatnonzero(subject):
            if model.kind == 'gaussian_corroboration':
                g[b, i] = np.exp(-((k[b, i]-k[b, obj])**2*w).sum(-1)).mean()
            elif model.kind == 'quantum_corroboration':
                g[b, i] = np.mean(abs(states[obj] @ states[i].conj())**2)
            else:
                pop = abs(states)**2
                if model.kind.startswith('normalized_'):
                    pop = pop/np.linalg.norm(pop, axis=-1, keepdims=True)
                g[b, i] = np.mean(pop[obj] @ pop[i])
    raw = (base*g[:, None]+EPS)*mask[:, None]
    alpha = raw/raw.sum(-1, keepdims=True)
    values = h @ get(a.value.weight).T + get(a.value.bias)
    logits = ((alpha @ values)*get(a.decoder)).sum(-1)+get(a.bias)
    return g, logits


def direct_gate(model, inputs):
    """Pair overlaps, independently from the production density-mean contraction."""
    from baselines.retacred.rmea import circuit
    h, roles, mask = inputs
    key = model.head.key(torch.cat((h, roles), -1))
    psi = circuit((math.pi*torch.tanh(key)).reshape(h.shape[:2]+(2, 3, 2)))
    pair = torch.einsum('bti,bsi->bts', psi.conj(), psi).abs().square()
    subject, obj = sides(roles, mask)
    g = (pair*obj[:, None]).sum(-1)/obj.sum(-1).clamp_min(1)[:, None]
    return torch.where(subject & obj.any(-1)[:, None], g, torch.ones_like(g))


def fixtures(models, inputs, labels):
    results = {}
    for kind, original in models.items():
        model = copy.deepcopy(original).to(inputs[0].device)
        small = tuple(x[:2, :6].double() if x.dtype != torch.bool else x[:2, :6] for x in inputs)
        # Exercise a nontrivial cross-side contraction even for an actual prefix
        # whose first words all happen to be closer to the subject.
        small = list(small)
        small[1] = small[1].clone()
        assert (small[2].sum(-1) >= 2).all()
        left = small[2].cumsum(-1) <= (small[2].sum(-1)//2)[:, None]
        small[1][..., 0] = (~left).to(small[1].dtype)
        small[1][..., 1] = left.to(small[1].dtype)
        small = tuple(small)
        subject, obj = sides(small[1], small[2])
        assert (subject.any(-1) & obj.any(-1)).all()
        model64 = copy.deepcopy(model).double()
        if kind in ('real12', 'softmax12'):
            error = float((model64(*small)-model64.native(*small)).abs().max())
        else:
            gate, z = reference(model64, small)
            alpha, g, _ = model64.components(*small)
            error = max(float(np.max(abs(g.detach().cpu().numpy()-gate))),
                        float(np.max(abs(model64(*small).detach().cpu().numpy()-z))))
            assert error < 1e-10, (kind, error)
            padded = (F.pad(small[0], (0, 0, 0, 2), value=.9),
                      F.pad(small[1], (0, 0, 0, 2), value=.8), F.pad(small[2], (0, 2), value=False))
            assert float((model64(*small)-model64(*padded)).abs().max()) < 1e-10
            one_side = (small[0], torch.zeros_like(small[1]), small[2])
            base = copy.deepcopy(model64); base.kind = 'gaussian_base'
            assert float((model64(*one_side)-base(*one_side)).abs().max()) < 1e-10
            assert float((alpha.sum(-1)-1).abs().max()) < 1e-12
            if kind == 'quantum_corroboration':
                actual = model64.components(*small)[1]
                direct = direct_gate(model64, small)
                pa = (model64.head.key.weight, model64.head.key.bias)
                ga = torch.autograd.grad(actual.sum(), pa)
                gb = torch.autograd.grad(direct.sum(), pa)
                assert all(a is not None and b is not None for a, b in zip(ga, gb))
                assert max(float(a.abs().max()) for a in ga) > 1e-12
                errors = [float((a-b).abs().max()) for a, b in zip(ga, gb)]
                assert max(errors) < 1e-9
        assert sum(p.numel() for p in model.parameters()) == 81028
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        before = [p.detach().clone() for p in model.parameters()]
        loss = F.cross_entropy(model(*inputs), labels)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        assert any(not torch.equal(p, old) for p, old in zip(model.parameters(), before))
        assert all(torch.isfinite(p).all() for p in model.parameters())
        results[kind] = dict(numpy_error=error, actual_update=True, parameters=81028)
    return results


def summarize(rows, logits, labels, cfg):
    paired = {}
    target = 'quantum_corroboration'
    for kind, z in logits.items():
        pred = z.argmax(-1)
        cm = np.bincount(40*labels+pred, minlength=1600).reshape(40, 40)
        denominator = cm.sum(0)+cm.sum(1)
        f = np.divide(2*np.diag(cm), denominator, out=np.zeros(40), where=denominator != 0)
        rows[kind].update(all40_macro_f1=float(f.mean()), confusion_matrix=cm.tolist(),
                          class_support=cm.sum(1).tolist(), per_class_f1=f.tolist())
        if kind != target:
            q = logits[target].argmax(-1) == labels
            c = pred == labels
            paired[kind] = dict(repaired=int((q & ~c).sum()), harmed=int((~q & c).sum()),
                                both_correct=int((q & c).sum()), both_wrong=int((~q & ~c).sum()))
    gates = {'macro_beats_'+k: rows[target]['all40_macro_f1'] > v['all40_macro_f1']
             for k, v in rows.items() if k != target}
    return dict(results=rows, paired=paired, gates=gates,
                gate='PROMISING_REQUIRES_BINDING_DIAGNOSTIC' if all(gates.values()) else 'DEFER_EXACT_SCREEN',
                interpretation='All conditions retained; TRAIN-only development. No family impossibility, independent L3 or new theory.',
                cohort_sha256=cfg['cohort_sha256'], L3='NOT_ACCEPTED', L4_T='NOT_ACCEPTED')
