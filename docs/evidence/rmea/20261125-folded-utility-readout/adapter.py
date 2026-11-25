"""Fold identity, independent utility gradients and cloned actual updates."""
import copy
import math
import numpy as np
import torch
from torch.nn import functional as F
from plugin import MODES, SHARED, initialize, make_head


def old_readout(m, inputs):
    h, roles, mask = inputs
    if m.kind != 'gaussian_base':
        return m.head(h, roles, mask)[0]
    a = m.head
    alpha = m.attention(*inputs)
    return (torch.einsum('brt,btd->brd', alpha, a.value(h)) * a.decoder).sum(-1) + a.bias


def fixtures(models, inputs, labels):
    checks = {}
    for kind, original in models.items():
        m = copy.deepcopy(original).to(inputs[0].device)
        with torch.no_grad():
            f32_error = float((m(*inputs) - old_readout(m, inputs)).abs().max())
            assert f32_error < 5e-6
        d = copy.deepcopy(m).double()
        d.fold_initial_utilities()  # exact FP64 fold of the original FP32 coefficients
        x = tuple(t[:2, :6].double() if t.dtype != torch.bool else t[:2, :6] for t in inputs)
        with torch.no_grad():
            f64_error = float((d(*x) - old_readout(d, x)).abs().max())
            assert f64_error < 1e-10
            if kind == 'gaussian_base':
                a = d.head
                key = a.key(torch.cat(x[:2], -1)).cpu().numpy()
                query = a.query.cpu().numpy()
                gamma = np.array([3-np.sqrt(2), 3+np.sqrt(2)] * 6) / 24
                raw = (np.exp(-((key[:, None]-query[None, :, None])**2*gamma).sum(-1)) + 1e-6) * x[2].cpu().numpy()[:, None]
                alpha = raw / raw.sum(-1, keepdims=True)
                expected = (np.einsum('brt,btd->brd', alpha, a.value(x[0]).cpu().numpy()) * a.decoder.cpu().numpy()).sum(-1) + a.bias.cpu().numpy()
                assert np.max(abs(d(*x).cpu().numpy() - expected)) < 1e-10
        got = d(*x)
        alpha = d.attention(*x)
        pooled_h = torch.einsum('brt,btd->brd', alpha, x[0])
        expected = (pooled_h * d.utility.weight).sum(-1) + d.utility.bias
        assert float((got - expected).abs().max()) < 1e-10
        ga = torch.autograd.grad(got.square().mean(), tuple(d.parameters()))
        gb = torch.autograd.grad(expected.square().mean(), tuple(d.parameters()))
        grad_error = max(float((a-b).abs().max()) for a, b in zip(ga, gb))
        assert grad_error < 1e-8
        with torch.no_grad():
            padded = (F.pad(x[0], (0, 0, 0, 2), value=.8), F.pad(x[1], (0, 0, 0, 2), value=.9), F.pad(x[2], (0, 2)))
            assert float((d(*x) - d(*padded)).abs().max()) < 1e-10
        assert sum(p.numel() for p in m.parameters()) == 53828
        old_buffers = {n: m.head.state_dict()[n].clone() for n in SHARED}
        opt = torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.01)
        before = [p.detach().clone() for p in m.parameters()]
        loss = F.cross_entropy(m(*inputs), labels); loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        assert any(not torch.equal(p, q) for p, q in zip(before, m.parameters()))
        assert all(torch.equal(m.head.state_dict()[n], v) for n, v in old_buffers.items())
        assert all(torch.isfinite(p).all() for p in m.parameters())
        checks[kind] = dict(initial_fp32_error=f32_error, folded_fp64_error=f64_error,
            independent_gradient_error=grad_error, parameters=53828, real8_update=True,
            baseline_buffers_unchanged=True, ce=float(loss.detach()))
    return dict(status='PASS', checks=checks, scope='Only folded utility identity/gradient/update; stable core and native attention unchanged')


def summarize(rows, logits, labels, cfg):
    for kind, z in logits.items():
        cm = np.bincount(40*labels + z.argmax(-1), minlength=1600).reshape(40, 40)
        den = cm.sum(0) + cm.sum(1)
        f = np.divide(2*np.diag(cm), den, out=np.zeros(40), where=den != 0)
        rows[kind].update(all40_macro_f1=float(f.mean()), confusion_matrix=cm.tolist(),
                          class_support=cm.sum(1).tolist(), per_class_f1=f.tolist())
    anchors = cfg['reused_native_rows']
    assert set(anchors) == {'quantum', 'dephased', 'gaussian_base', 'gaussian_covariance'}
    comparators = {k:v for k,v in rows.items() if k != 'quantum'}
    comparators.update({'reused/'+k:v for k,v in anchors.items()})
    assert all(v['updates'] == 1024 for v in rows.values())
    assert {v['order_sha256'] for v in rows.values()} == {v['order_sha256'] for v in anchors.values()}
    assert all(v['class_support'] == rows['quantum']['class_support'] for v in anchors.values())
    gates = {'macro_beats_'+k: rows['quantum']['all40_macro_f1'] > v['all40_macro_f1'] for k,v in comparators.items()}
    assert len(gates) == 6
    return dict(results=rows, reused_results=anchors, gates=gates,
        gate='PASS_FOR_FULL_PROTOCOL_DESIGN_ONLY' if all(gates.values()) else 'DEFER_EXACT_FOLDED_UTILITY_SCREEN',
        interpretation='Known linear reparameterization changes optimizer and regularization, not the real-arithmetic hypothesis class. Development screen only, no cause, independent L3 or new T. No LR/scaling/factorization/seed/budget/control rescue.',
        L3='NOT_ACCEPTED', L4_T='NOT_ACCEPTED')
