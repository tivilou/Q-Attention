"""Existing screen adapter: affected head checks and all40 final-only gates."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from plugin import MODES, initialize, make_head


def fixtures(models, inputs, labels):
    originals = {k: {n: t.detach().clone() for n, t in m.state_dict().items()} for k, m in models.items()}
    checks = {}
    for kind, original in models.items():
        device = inputs[0].device
        model = copy.deepcopy(original).to(device=device, dtype=torch.float64)
        h, roles, mask = (inputs[0].double(), inputs[1].double(), inputs[2])
        before = {n: t.detach().clone() for n, t in model.state_dict().items()}
        p, alpha, u = model.components(h, roles, mask)
        z = model(h, roles, mask)
        assert z.shape == (len(h), 40) and torch.isfinite(z).all()
        assert torch.isfinite(p).all() and float(p.min()) >= 0 and float(p.max()) <= 1+1e-12
        assert torch.allclose(alpha.sum(-1), torch.ones_like(alpha.sum(-1)), atol=1e-12, rtol=0)
        assert float(alpha.masked_select(~mask[:, None].expand_as(alpha)).abs().sum()) == 0
        hpad = torch.cat((h, torch.full_like(h[:, :1], 7)), 1)
        rpad = torch.cat((roles, torch.full_like(roles[:, :1], -3)), 1)
        mpad = torch.cat((mask, torch.zeros_like(mask[:, :1])), 1)
        assert torch.allclose(z, model(hpad, rpad, mpad), atol=1e-10, rtol=1e-10)
        single = model(h[:, :1], roles[:, :1], torch.ones_like(mask[:, :1]))
        utility = torch.einsum('bd,rd->br', model.head.value(h[:, 0]), model.head.decoder)+model.head.bias
        assert torch.allclose(single, utility, atol=1e-12, rtol=1e-12)
        loss = F.cross_entropy(z, labels)
        loss.backward()
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in model.parameters())
        phase_rows = (7, 9, 11) if kind == 'quantum_middle' else (1, 3, 5, 7, 9, 11) if kind == 'dephased_middle' else ()
        for row in phase_rows:
            assert torch.count_nonzero(model.head.prepare.weight.grad[row]) == 0
            assert model.head.prepare.bias.grad[row] == 0
        if kind == 'dephased_middle':
            assert torch.count_nonzero(model.head.effect.grad[:, 1, :, 1]) == 0
        fd = {}
        coordinates = [('head.prepare.weight', (0, 0)), ('head.effect', (0, 0, 0, 0)),
                       ('head.value.weight', (0, 0)), ('head.decoder', (0, 0))]
        named = dict(model.named_parameters())
        for name, coordinate in coordinates:
            tensor = named[name]
            analytical = float(tensor.grad[coordinate])
            value = float(tensor[coordinate].detach())
            with torch.no_grad():
                tensor[coordinate] = value+1e-5
                plus = float(F.cross_entropy(model(h, roles, mask), labels))
                tensor[coordinate] = value-1e-5
                minus = float(F.cross_entropy(model(h, roles, mask), labels))
                tensor[coordinate] = value
            numerical = (plus-minus)/2e-5
            error = abs(analytical-numerical)
            assert error <= 1e-7+1e-4*abs(numerical), (kind, name, analytical, numerical)
            fd[name] = dict(analytical=analytical, numerical=numerical, error=error)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01,
                                      betas=(.9, .999), eps=1e-8)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        assert all(torch.isfinite(x).all() for x in model.parameters())
        assert any(not torch.equal(before[n], t) for n, t in model.state_dict().items())
        # Also exercise the exact screen precision on the same eight real rows.
        fp32 = copy.deepcopy(original).to(device=device, dtype=torch.float32)
        opt32 = torch.optim.AdamW(fp32.parameters(), lr=.001, weight_decay=.01, betas=(.9, .999), eps=1e-8)
        z32 = fp32(inputs[0].float(), inputs[1].float(), mask)
        F.cross_entropy(z32, labels).backward()
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in fp32.parameters())
        torch.nn.utils.clip_grad_norm_(fp32.parameters(), 1., error_if_nonfinite=True)
        opt32.step()
        assert all(torch.isfinite(x).all() for x in fp32.parameters())
        checks[kind] = dict(parameters=sum(x.numel() for x in model.parameters()),
                            finite_gradient=True, finite_difference=fd, exact_zero_prepare_rows=list(phase_rows),
                            cloned_real_label_updates=dict(float64=1, float32=1), original_unchanged=True,
                            padding=True, singleton=True, normalized=True)
    assert all(torch.equal(m.state_dict()[n], t) for k, m in models.items() for n, t in originals[k].items())
    return dict(status='PASS_AFFECTED_HEAD_CHECKS', checks=checks,
                limitations='Eight TRAIN rows; clone updates discarded. No task efficacy or equal effective-capacity claim.')


def summarize(rows, logits, labels, cfg):
    labels = np.asarray(labels, dtype=np.int64)
    for kind in MODES:
        pred = logits[kind].argmax(-1)
        cm = np.bincount(labels*40+pred, minlength=1600).reshape(40, 40)
        denominator = cm.sum(0)+cm.sum(1)
        f1 = np.divide(2*np.diag(cm), denominator, out=np.zeros(40, dtype=float), where=denominator != 0)
        rows[kind].update(all40_macro_f1=float(f1.mean()), confusion_matrix=cm.tolist(),
                          class_support=cm.sum(1).tolist(), per_class_f1=f1.tolist())
        assert rows[kind]['order_sha256'] == cfg['order_sha256'] and rows[kind]['updates'] == 1024
        assert rows[kind]['initial_sha256'] == cfg['expected_initial_sha256']
        assert rows[kind]['parameters'] == 81028 and rows[kind]['class_support'] == cfg['assessment_class_support']
    q = rows['quantum_middle']['all40_macro_f1']
    gates = {'Q_beats_'+k: q > rows[k]['all40_macro_f1'] for k in MODES[1:]}
    for k, r in cfg['references'].items():
        assert r['order_sha256'] == cfg['order_sha256'] and r['updates'] == 1024
        assert r['class_support'] == cfg['assessment_class_support']
        gates['Q_beats_reference/'+k] = q > r['all40_macro_f1']
    return dict(results=rows, reused_results=cfg['references'], gates=gates,
                gate='PASS_FOR_FULL_STUDY_PROTOCOL_ONLY' if all(gates.values()) else 'STOP_EXACT_MIDDLE_PLACEMENT_PROFILE',
                interpretation='Final1024, exposed TRAIN development, seed13 only. Ties fail. CE secondary.',
                L3='NOT_ACCEPTED', L4_T='NOT_ACCEPTED')
