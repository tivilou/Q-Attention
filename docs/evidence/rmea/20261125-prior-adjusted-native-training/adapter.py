"""Only prior-loss semantics, unchanged inference, and actual small updates."""
import copy
import math
import numpy as np
import torch
from torch.nn import functional as F
from plugin import MODES, FIT_COUNTS, initialize, make_head


def fixtures(models, inputs, labels):
    checks = {}
    for kind, original in models.items():
        m = copy.deepcopy(original).to(inputs[0].device)
        before_original = {n: v.clone() for n, v in original.state_dict().items()}
        m.eval()
        with torch.no_grad():
            raw = m.raw_logits(*inputs)
            assert torch.equal(m(*inputs), raw)
            if kind != 'gaussian_base':
                assert torch.equal(raw, m.head(*inputs)[0])
        d = copy.deepcopy(m).double()
        x = tuple(t[:2, :6].double() if t.dtype != torch.bool else t[:2, :6] for t in inputs)
        y = labels[:2]
        raw = d.raw_logits(*x)
        d.train()
        actual = F.cross_entropy(d(*x), y)
        z = raw.detach().clone().requires_grad_()
        n = z.new_tensor(FIT_COUNTS)
        shifted = z + n.log()
        expected = (torch.logsumexp(shifted, -1) - shifted[torch.arange(len(y)), y]).mean()
        assert abs(float(actual-expected)) < 1e-10
        grad = torch.autograd.grad(expected, z)[0]
        onehot = F.one_hot(y, 40).to(z)
        manual = (shifted.softmax(-1) - onehot) / len(y)
        gradient_error = float((grad-manual).abs().max())
        assert gradient_error < 1e-12
        got_grad = torch.autograd.grad(actual, tuple(d.parameters()))
        ref_loss = F.cross_entropy(d.raw_logits(*x) + n.log(), y)
        ref_grad = torch.autograd.grad(ref_loss, tuple(d.parameters()))
        parameter_gradient_error = max(float((a-b).abs().max()) for a,b in zip(got_grad, ref_grad))
        assert parameter_gradient_error < 1e-9
        zz = z.detach().cpu().numpy(); yy = y.cpu().numpy()
        weighted = np.exp(zz-zz.max(-1, keepdims=True)) * np.array(FIT_COUNTS)
        independent_loss = -np.log(weighted[np.arange(len(yy)), yy]/weighted.sum(-1)).mean()
        numpy_error = abs(float(actual)-float(independent_loss))
        assert numpy_error < 1e-10
        uniform = torch.full((40,), 7., dtype=z.dtype, device=z.device)
        assert abs(float(F.cross_entropy(z + uniform.log(), y)-F.cross_entropy(z, y))) < 1e-12
        d.eval()
        with torch.no_grad():
            assert torch.equal(d(*x), d.raw_logits(*x))
            padded = (F.pad(x[0],(0,0,0,2),value=.8), F.pad(x[1],(0,0,0,2),value=.9), F.pad(x[2],(0,2)))
            assert float((d(*x)-d(*padded)).abs().max()) < 1e-10
            if kind == 'gaussian_base':
                a=d.head;key=a.key(torch.cat(x[:2],-1)).cpu().numpy();query=a.query.cpu().numpy()
                gamma=np.array([3-np.sqrt(2),3+np.sqrt(2)]*6)/24
                p=np.exp(-((key[:,None]-query[None,:,None])**2*gamma).sum(-1))
                raw_np=(p+1e-6)*x[2].cpu().numpy()[:,None];alpha=raw_np/raw_np.sum(-1,keepdims=True)
                expected_z=(np.einsum('brt,btd->brd',alpha,a.value(x[0]).cpu().numpy())*a.decoder.cpu().numpy()).sum(-1)+a.bias.cpu().numpy()
                assert np.max(abs(expected_z-d(*x).cpu().numpy())) < 1e-10
        m.train(); opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
        before=[p.detach().clone() for p in m.parameters()]
        loss=F.cross_entropy(m(*inputs),labels);loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
        assert any(not torch.equal(a,b) for a,b in zip(before,m.parameters()))
        assert all(torch.isfinite(p).all() for p in m.parameters())
        assert tuple(m.fit_counts.tolist()) == FIT_COUNTS
        assert all(torch.equal(original.state_dict()[n],v) for n,v in before_original.items())
        assert sum(p.numel() for p in m.parameters()) == 81028
        checks[kind]=dict(eval_raw_logits_exact=True,numpy_loss_error=numpy_error,
            logit_gradient_error=gradient_error,parameter_gradient_error=parameter_gradient_error,
            uniform_counts_ce_parity=True,padding=True,real8_update=True,parameters=81028,
            original_models_unchanged=True,fit_counts_unchanged=True)
    return dict(status='PASS',checks=checks,scope='Known training-only Balanced Softmax and native inference; no core lifecycle matrix')


def summarize(rows, logits, labels, cfg):
    assert tuple(cfg['fit_class_counts']) == FIT_COUNTS
    for kind,z in logits.items():
        cm=np.bincount(40*labels+z.argmax(-1),minlength=1600).reshape(40,40)
        den=cm.sum(0)+cm.sum(1)
        f=np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den!=0)
        rows[kind].update(all40_macro_f1=float(f.mean()),confusion_matrix=cm.tolist(),
            class_support=cm.sum(1).tolist(),per_class_f1=f.tolist(),
            training_loss='CE(raw_logits + log(fit_class_count), labels)',assessment_loss='CE(raw_logits, labels)')
    anchors=cfg['reused_native_rows']
    assert set(anchors)=={'quantum','dephased','gaussian_base','gaussian_covariance'}
    assert {v['order_sha256'] for v in rows.values()}=={v['order_sha256'] for v in anchors.values()}
    assert all(v['updates']==1024 for v in rows.values())
    assert all(v['class_support']==rows['quantum']['class_support'] for v in anchors.values())
    comparators={k:v for k,v in rows.items() if k!='quantum'}
    comparators.update({'reused/'+k:v for k,v in anchors.items()})
    gates={'macro_beats_'+k:rows['quantum']['all40_macro_f1']>v['all40_macro_f1'] for k,v in comparators.items()}
    assert len(gates)==6
    return dict(results=rows,reused_results=anchors,gates=gates,
        gate='PASS_FOR_FULL_PROTOCOL_DESIGN_ONLY' if all(gates.values()) else 'DEFER_EXACT_PRIOR_ADJUSTED_SCREEN',
        interpretation='Known objective transfer, not quantum novelty or a Macro-F1 consistency theorem. Native factorized readout retained. No prior-strength/threshold/LR/seed/budget/subset/control rescue; development screen only.',
        L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED')
