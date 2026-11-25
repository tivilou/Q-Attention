"""Existing screen interface: task fixture and frozen six-gate summary only."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from plugin import MODES, initialize, make_head


def fixtures(models, inputs, labels):
    checks = {}
    for kind, original in models.items():
        before = {n:v.clone() for n,v in original.state_dict().items()}
        m = copy.deepcopy(original).to(inputs[0].device)
        query_name = 'head.query' if kind.startswith('real') else 'head.effect'
        optimizer = torch.optim.AdamW(m.parameters(), lr=.001, weight_decay=.01)
        initial_query = dict(m.named_parameters())[query_name].detach().clone()
        optimizer.zero_grad()
        z = m(*inputs)
        loss = F.cross_entropy(z, labels)
        assert loss.isfinite() and z.isfinite().all()
        loss.backward()
        assert all(p.grad is not None and p.grad.isfinite().all() for p in m.parameters())
        gradients = dict(m.named_parameters())[query_name].grad.reshape(40,-1).norm(dim=-1)
        assert gradients.isfinite().all() and (gradients > 0).all()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        assert all(p.isfinite().all() for p in m.parameters())
        assert not torch.equal(initial_query, dict(m.named_parameters())[query_name])
        with torch.no_grad():
            x = tuple(t[:2,:6] for t in inputs)
            padded = (F.pad(x[0],(0,0,0,2),value=.8),F.pad(x[1],(0,0,0,2),value=.9),F.pad(x[2],(0,2)))
            error = float((m(*x)-m(*padded)).abs().max())
            assert error < 1e-4
        assert all(torch.equal(original.state_dict()[n],v) for n,v in before.items())
        count = sum(p.numel() for p in m.parameters())
        assert count == 81028
        checks[kind] = dict(parameters=count,actual8_update=True,padding_max_error=error,
            all40_query_gradient_min_norm=float(gradients.min()),original_models_unchanged=True)
    return dict(status='PASS',scope='Actual first8 fit rows, disposable clone per arm',checks=checks)


def summarize(rows, logits, labels, cfg):
    assert tuple(rows) == MODES
    for kind,z in logits.items():
        cm = np.bincount(40*labels+z.argmax(-1),minlength=1600).reshape(40,40)
        denominator = cm.sum(0)+cm.sum(1)
        f1 = np.divide(2*np.diag(cm),denominator,out=np.zeros(40),where=denominator!=0)
        rows[kind].update(all40_macro_f1=float(f1.mean()),confusion_matrix=cm.tolist(),
            class_support=cm.sum(1).tolist(),per_class_f1=f1.tolist())
    references = cfg['reused_native_rows']
    q = rows[MODES[0]]
    assert len(references)==4
    assert len({v['order_sha256'] for v in list(rows.values())+list(references.values())})==1
    assert all(v['class_support']==q['class_support'] and v['updates']==1024 for v in references.values())
    controls = {k:v for k,v in rows.items() if k!=MODES[0]}
    controls.update({'reused/'+k:v for k,v in references.items()})
    gates = {'macro_beats_'+k:q['all40_macro_f1']>v['all40_macro_f1'] for k,v in controls.items()}
    assert len(gates)==6
    return dict(results=rows,reused_results=references,gates=gates,
        gate='PASS_FOR_FULL_PROTOCOL_DESIGN_ONLY' if all(gates.values()) else 'DEFER_EXACT_RELATION_FRAME_SCREEN',
        L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED',interpretation='Known query-frame measurement; no transferred PGM task guarantee')
