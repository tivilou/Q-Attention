"""Check only new stage/freezing semantics; complete all-40 comparison."""
import copy
import hashlib
import numpy as np
import torch
from torch.nn import functional as F
from plugin import MODES, SHARED, initialize, make_head, make_optimizer, utility_hash


def fixtures(models, inputs, labels):
    clones = {k: copy.deepcopy(m).to(inputs[0].device) for k, m in models.items()}
    cfg = dict(lr=.001, betas=[.9, .999], optimizer_eps=1e-8, weight_decay=.01)
    optimizers = {k: make_optimizer(m, cfg) for k, m in clones.items()}
    warm = {}
    checks = {}
    for kind, m in clones.items():
        assert sum(p.numel() for p in m.parameters()) == 81028
        before = {n: p.detach().clone() for n, p in m.head.named_parameters()}
        m.train(); m.set_training_progress(1, 1024)
        values = m.head.value(inputs[0]); mean = (values * inputs[2][..., None]).sum(1) / inputs[2].sum(1)[:, None]
        expected = mean @ m.head.decoder.T + m.head.bias
        actual = m(*inputs)
        assert float((actual - expected).abs().max()) < 2e-6
        warm[kind] = actual.detach().clone()
        loss = F.cross_entropy(actual, labels); loss.backward()
        shared, router = m.parameter_groups()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        assert all(torch.count_nonzero(p.grad) == 0 for p in router)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True)
        optimizers[kind].step()
        assert any(not torch.equal(p, before[n]) for n, p in m.head.named_parameters() if n in SHARED)
        assert all(torch.equal(p, before[n]) for n, p in m.head.named_parameters() if n not in SHARED)
        checks[kind] = dict(uniform_formula=True, uniform_actual8_update=True, router_frozen=True)
    assert all(torch.equal(warm['quantum'], z) for z in warm.values())
    assert len({utility_hash(m) for m in clones.values()}) == 1
    for kind, m in clones.items():
        opt = optimizers[kind]; opt.zero_grad(set_to_none=True)
        m.set_training_progress(257, 1024)
        before = {n: p.detach().clone() for n, p in m.head.named_parameters()}
        frozen_hash = utility_hash(m)
        assert bytes(m.boundary_fingerprint.cpu().tolist()).hex() == frozen_hash
        logits = m(*inputs)
        # Independent native endpoint where available; Gaussian is fixed prior formula.
        if kind != 'gaussian_base':
            detail = m.head(*inputs, details=True)
            assert float((detail['logits'] - logits).abs().max()) < 2e-6
        F.cross_entropy(logits, labels).backward()
        shared, router = m.parameter_groups()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        assert all(torch.count_nonzero(p.grad) == 0 for p in shared)
        assert any(torch.count_nonzero(p.grad) > 0 for p in router)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        assert utility_hash(m) == frozen_hash
        assert all(torch.equal(p, before[n]) for n, p in m.head.named_parameters() if n in SHARED)
        assert any(not torch.equal(p, before[n]) for n, p in m.head.named_parameters() if n not in SHARED)
        assert opt.router_reset and all(int(opt.state[p]['step']) == 1 for p in router)
        m.eval()
        with torch.no_grad():
            values = m.head.value(inputs[0]); alpha = m.attention(*inputs)
            independent = (alpha * torch.einsum('btd,rd->brt', values, m.head.decoder)).sum(-1) + m.head.bias
            assert float((m(*inputs) - independent).abs().max()) < 2e-6
        checks[kind].update(router_actual8_update=True, shared_frozen_exact=True,
                            router_optimizer_counter_reset=True, native_readout_formula=True)
    return dict(status='PASS', checks=checks, identical_cloned_warmup_utilities=True,
                scope='Only new 256/768 stage, zero inactive gradients and exact freeze semantics. No lifecycle matrix.')


def summarize(rows, logits, labels, cfg):
    for kind, z in logits.items():
        cm = np.bincount(40*labels + z.argmax(-1), minlength=1600).reshape(40, 40)
        den = cm.sum(0) + cm.sum(1)
        f1 = np.divide(2*np.diag(cm), den, out=np.zeros(40), where=den != 0)
        rows[kind].update(all40_macro_f1=float(f1.mean()), confusion_matrix=cm.tolist(),
                          class_support=cm.sum(1).tolist(), per_class_f1=f1.tolist())
    anchor = cfg['reused_native_rows']
    assert set(anchor) == {'gaussian_covariance'}
    q = rows['quantum']['all40_macro_f1']
    gates = dict(macro_beats_dephased=q > rows['dephased']['all40_macro_f1'],
                 macro_beats_gaussian=q > rows['gaussian_base']['all40_macro_f1'],
                 macro_beats_strong_screen=q > anchor['gaussian_covariance']['all40_macro_f1'])
    assert all(v['updates'] == 1024 for v in rows.values())
    assert len({v['order_sha256'] for v in rows.values()}) == 1
    return dict(results=rows, reused_results=anchor, gates=gates,
        gate='PROMISING_COMMON_UTILITY_ROUTE_ONLY' if all(gates.values()) else 'STOP_EXACT_COMMON_UTILITY_PROFILE',
        interpretation='Development diagnostic with frozen common utilities; original identity is known. No independent L3, new theory, family impossibility or seed/warmup/budget rescue.',
        L3='NOT_ACCEPTED', L4_T='NOT_ACCEPTED')
