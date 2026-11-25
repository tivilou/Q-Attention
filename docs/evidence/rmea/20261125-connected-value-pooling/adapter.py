"""Affected moment mathematics, cloned updates and all-condition reporting."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from baselines.retacred.rmea_matched_routers import Head
from plugin import MODES, initialize, make_head, native_kind


def pair_difference(alpha, values):
    left, right = values[..., 0::2], values[..., 1::2]
    product = (left[:, :, None] - left[:, None]) * (right[:, :, None] - right[:, None])
    return .5 * torch.einsum('bri,brj,bijc->brc', alpha, alpha, product)


def fixtures(models, inputs, labels):
    checks = {}
    for kind, original in models.items():
        m = copy.deepcopy(original).to(inputs[0].device).double()
        x = tuple(t[:2, :6].double() if t.dtype != torch.bool else t[:2, :6] for t in inputs)
        with torch.no_grad():
            assert torch.count_nonzero(m.moment_decoder) == 0
            alpha, values, mean, base = m.components(*x)
            endpoint = float((m(*x) - base).abs().max())
            assert endpoint == 0.
            if not kind.startswith('gaussian'):
                native = Head(native_kind(kind)).double().to(inputs[0].device)
                native.head.load_state_dict(m.head.state_dict())
                assert float((base - native(*x)).abs().max()) < 1e-12
            else:
                a = m.head
                key = a.key(torch.cat(x[:2], -1)).cpu().numpy()
                query = a.query.cpu().numpy()
                scale = np.sqrt(np.array([3-np.sqrt(2), 3+np.sqrt(2)] * 6) / 24)
                raw = (np.exp(-((key[:, None]*scale-query[None, :, None]*scale)**2).sum(-1)) + 1e-6) * x[2].cpu().numpy()[:, None]
                attn = raw / raw.sum(-1, keepdims=True)
                ref = (np.einsum('brt,btd->brd', attn, values.cpu().numpy()) * a.decoder.cpu().numpy()).sum(-1) + a.bias.cpu().numpy()
                assert np.max(abs(ref - base.cpu().numpy())) < 1e-10
            m.moment_decoder.copy_(torch.linspace(-.3, .4, 1280, device=values.device, dtype=torch.float64).reshape(40, 32))
        alpha, values, mean, base = m.components(*x)
        actual = m.statistic(alpha, values, mean)
        independent = (torch.einsum('bri,brj,bic,bjd->brcd', alpha, alpha, values[..., 0::2], values[..., 1::2]).diagonal(dim1=-2, dim2=-1)
                       if kind.endswith('mean_product') else pair_difference(alpha, values))
        formula_error = float((actual - independent).abs().max())
        expected = base + (independent * m.moment_decoder).sum(-1)
        got = m(*x)
        assert formula_error < 1e-10 and float((got - expected).abs().max()) < 1e-10
        ga = torch.autograd.grad(got.square().mean(), tuple(m.parameters()))
        gb = torch.autograd.grad(expected.square().mean(), tuple(m.parameters()))
        gradient = max(float((a-b).abs().max()) for a, b in zip(ga, gb))
        assert gradient < 1e-8
        with torch.no_grad():
            padded = (F.pad(x[0], (0, 0, 0, 2), value=.8), F.pad(x[1], (0, 0, 0, 2), value=.9), F.pad(x[2], (0, 2)))
            assert float((m(*x) - m(*padded)).abs().max()) < 1e-10
            if kind.endswith('covariance'):
                shifted = values + torch.linspace(-.7, .8, 64, device=values.device, dtype=values.dtype)
                shifted_mean = torch.einsum('brt,btd->brd', alpha, shifted)
                assert float((actual - m.statistic(alpha, shifted, shifted_mean)).abs().max()) < 1e-10
                one = tuple(t[:1, :1] for t in x)
                oa, ov, om, ob = m.components(*one)
                assert float(m.statistic(oa, ov, om).abs().max()) < 1e-12
        update = copy.deepcopy(original).to(inputs[0].device)
        assert sum(p.numel() for p in update.parameters()) == 82308
        opt = torch.optim.AdamW(update.parameters(), lr=.001, weight_decay=.01)
        before = [p.detach().clone() for p in update.parameters()]
        loss = F.cross_entropy(update(*inputs), labels)
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in update.parameters())
        assert update.moment_decoder.grad.abs().max() > 0
        torch.nn.utils.clip_grad_norm_(update.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        assert all(torch.isfinite(p).all() for p in update.parameters())
        assert any(not torch.equal(a, b) for a, b in zip(before, update.parameters()))
        checks[kind] = dict(formula_error=formula_error, gradient_error=gradient,
                            zero_coefficient_endpoint=endpoint, parameters=82308, real8_update=True, ce=float(loss.detach()))
    return dict(status='PASS', checks=checks, scope='New connected/product readout and cloned real8 updates; accepted core unchanged')


def summarize(rows, logits, labels, cfg):
    for kind, z in logits.items():
        pred = z.argmax(-1)
        cm = np.bincount(40*labels+pred, minlength=1600).reshape(40, 40)
        den = cm.sum(0)+cm.sum(1)
        f = np.divide(2*np.diag(cm), den, out=np.zeros(40), where=den != 0)
        rows[kind].update(all40_macro_f1=float(f.mean()), confusion_matrix=cm.tolist(),
                          class_support=cm.sum(1).tolist(), per_class_f1=f.tolist())
    anchors = cfg['reused_native_rows']
    assert set(anchors) == {'quantum', 'dephased', 'gaussian_base'}
    assert {x['order_sha256'] for x in rows.values()} == {x['order_sha256'] for x in anchors.values()}
    assert all(x['class_support'] == rows['quantum_covariance']['class_support'] and x['updates'] == 1024 for x in anchors.values())
    comparators = {k:v for k,v in rows.items() if k != 'quantum_covariance'}
    comparators.update({'reused/'+k:v for k,v in anchors.items()})
    gates = {'macro_beats_'+k:rows['quantum_covariance']['all40_macro_f1'] > v['all40_macro_f1'] for k,v in comparators.items()}
    assert len(gates) == 7
    return dict(results=rows, reused_results=anchors, gates=gates,
                gaussian_connected_beats_product=rows['gaussian_covariance']['all40_macro_f1'] > rows['gaussian_mean_product']['all40_macro_f1'],
                gate='PASS_FOR_FULL_PROTOCOL_DESIGN_ONLY' if all(gates.values()) else 'DEFER_EXACT_CONNECTED_READOUT_SCREEN',
                L3='NOT_ACCEPTED', L4_T='NOT_ACCEPTED',
                interpretation='Known second-order pooling, development screen only; same-word coupling is not a nonclassical resource. Retain all outcomes, no seed/pair/channel/scale/budget/control rescue.')
