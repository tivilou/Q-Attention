"""Changed flow checks and compact all-condition reporting only."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from baselines.retacred.rmea_matched_routers import Head
from plugin import MODES, initialize, make_head


def taylor(A):
    eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device).expand_as(A)
    result, term = eye.clone(), eye
    for k in range(1, 45):
        term = term @ A / k
        result = result + term
    return result


def fixtures(models, inputs, labels):
    checks = {}
    for kind in ('quantum_flow', 'real12_flow'):
        m = copy.deepcopy(models[kind]).to(inputs[0].device).double()
        x = tuple(t[:2, :6].double() if t.dtype != torch.bool else t[:2, :6] for t in inputs)
        alpha, mass, native_mass = m.attention(*x)
        ref, _, _ = m.attention(*x, exponential=taylor)
        forward = float((alpha - ref).abs().max())
        mass_error = float((mass - native_mass).abs().max())
        assert forward < 1e-10 and mass_error < 1e-10
        weights = torch.linspace(-.8, .9, alpha.numel(), device=alpha.device, dtype=torch.float64).reshape_as(alpha)
        params = tuple(p for n, p in m.named_parameters() if 'prepare' in n or 'effect' in n or 'key' in n or 'query' in n)
        ga = torch.autograd.grad((alpha * weights).sum(), params)
        gb = torch.autograd.grad((ref * weights).sum(), params)
        gradient_error = max(float((a - b).abs().max()) for a, b in zip(ga, gb))
        assert gradient_error < 1e-8 and all(torch.isfinite(g).all() for g in ga)
        with torch.no_grad():
            m.time = 0.
            native = Head('quantum' if kind == 'quantum_flow' else 'real12').double().to(inputs[0].device)
            native.load_state_dict(m.state_dict())
            endpoint = float((m(*x) - native(*x)).abs().max())
            assert endpoint < 1e-10
            m.time = np.pi
            z = m(*x)
            padded = (F.pad(x[0], (0, 0, 0, 2), value=.4), F.pad(x[1], (0, 0, 0, 2), value=.3), F.pad(x[2], (0, 2)))
            padding = float((z - m(*padded)).abs().max())
            assert padding < 1e-10
            one = tuple(t[:1, :1] for t in x)
            assert one[2].all()
            utility = (m.head.value(one[0])[:, 0] * m.head.decoder).sum(-1) + m.head.bias
            assert float((m(*one) - utility).abs().max()) < 1e-10
        checks[kind] = dict(matrix_exp_series_error=forward, mass_error=mass_error,
                            gradient_error=gradient_error, native_endpoint_error=endpoint, padding_error=padding)
    for kind, original in models.items():
        m = copy.deepcopy(original).to(inputs[0].device)
        assert sum(p.numel() for p in m.parameters()) == 81028
        opt = torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.01)
        before = [p.detach().clone() for p in m.parameters()]
        loss = F.cross_entropy(m(*inputs), labels)
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        assert all(torch.isfinite(p).all() for p in m.parameters())
        assert any(not torch.equal(a, b) for a, b in zip(before, m.parameters()))
        checks.setdefault(kind, {}).update(parameters=81028, real8_update=True, ce=float(loss.detach()))
    return dict(status='PASS', checks=checks, scope='New mean-state matrix flow, derivatives and cloned8-row update; accepted core unchanged')


def summarize(rows, logits, labels, cfg):
    for kind, z in logits.items():
        pred = z.argmax(-1)
        cm = np.bincount(40 * labels + pred, minlength=1600).reshape(40, 40)
        den = cm.sum(0) + cm.sum(1)
        f = np.divide(2 * np.diag(cm), den, out=np.zeros(40), where=den != 0)
        rows[kind].update(all40_macro_f1=float(f.mean()), confusion_matrix=cm.tolist(),
                          class_support=cm.sum(1).tolist(), per_class_f1=f.tolist())
    anchors = cfg['reused_native_rows']
    assert set(anchors) == {'real12', 'softmax12', 'gaussian_base'}
    assert {x['order_sha256'] for x in rows.values()} == {x['order_sha256'] for x in anchors.values()}
    assert all(x['class_support'] == rows['quantum_flow']['class_support'] and x['updates'] == 1024 for x in anchors.values())
    comparators = {k: v for k, v in rows.items() if k != 'quantum_flow'}
    comparators.update({'reused/' + k: v for k, v in anchors.items()})
    gates = {'macro_beats_' + k: rows['quantum_flow']['all40_macro_f1'] > v['all40_macro_f1'] for k, v in comparators.items()}
    assert len(gates) == 6
    return dict(results=rows, reused_results=anchors, gates=gates,
                gate='PASS_FOR_FULL_PROTOCOL_DESIGN_ONLY' if all(gates.values()) else 'DEFER_EXACT_FLOW_SCREEN',
                L3='NOT_ACCEPTED', L4_T='NOT_ACCEPTED',
                interpretation='Exposed TRAIN development screen. Known density exponentiation and commutator algebra, no new T or independent confirmation; no time/seed/cohort/budget/control rescue.')
