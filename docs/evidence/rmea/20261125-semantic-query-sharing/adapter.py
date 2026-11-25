"""Focused query-map checks and seven frozen development-screen comparisons."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from baselines.retacred.rmea_matched_routers import Head
from plugin import MODES, CODES, initialize, make_head


def fixtures(models, inputs, labels):
    c = np.asarray(CODES['semantic']); p = np.asarray(CODES['permuted'])
    assert c.shape == p.shape == (40, 8)
    assert np.linalg.matrix_rank(c) == np.linalg.matrix_rank(p) == 8
    assert np.max(np.abs(c.T @ c - p.T @ p)) < 1e-12
    assert np.array_equal(c[0], p[0]) and np.array_equal(c[0], [1.,0,0,0,0,0,0,0])
    assert all(not np.array_equal(c[i], p[i]) for i in range(1, 40))
    checks = {}
    for kind, original in models.items():
        original_state = {n: v.clone() for n, v in original.state_dict().items()}
        m = copy.deepcopy(original).to(inputs[0].device)
        source_name = 'head.' + m.source_query_name
        assert source_name not in dict(m.named_parameters())
        assert source_name in dict(m.named_buffers()) and 'code' in dict(m.named_buffers())
        source = getattr(m.head, m.source_query_name).reshape(40,12).double()
        residual = m.code.double().T @ (m.code.double() @ m.query_weights.double() - source)
        normal_error = float(residual.abs().max())
        assert normal_error < 2e-6
        d = copy.deepcopy(m).double().eval()
        x = tuple(t[:2,:6].double() if t.dtype != torch.bool else t[:2,:6] for t in inputs)
        z = d(*x)
        loss = F.cross_entropy(z, labels[:2])
        got = torch.autograd.grad(loss, d.query_weights)[0]
        q = d.mapped_query().detach().requires_grad_()
        qgrad = torch.autograd.grad(F.cross_entropy(d.logits_with_query(*x,q), labels[:2]), q)[0]
        chain_error = float((got - d.code.T @ qgrad).abs().max())
        assert chain_error < 1e-10 and torch.isfinite(got).all() and float(got.abs().max()) > 0
        idx = int(got.abs().argmax()); step = 1e-5
        with torch.no_grad():
            w = d.query_weights.reshape(-1); old = w[idx].item()
            w[idx] = old + step; plus = F.cross_entropy(d(*x), labels[:2])
            w[idx] = old - step; minus = F.cross_entropy(d(*x), labels[:2])
            w[idx] = old
            finite_difference_error = abs(float((plus-minus)/(2*step)) - float(got.reshape(-1)[idx]))
            assert finite_difference_error < 1e-7
            padded = (F.pad(x[0],(0,0,0,2),value=.8),F.pad(x[1],(0,0,0,2),value=.9),F.pad(x[2],(0,2)))
            assert float((d(*x)-d(*padded)).abs().max()) < 1e-10
            if d.native_mode != 'real12':
                native = Head(d.native_mode).double().to(x[0].device)
                native.head.load_state_dict(d.head.state_dict())
                native.head.effect.copy_(d.mapped_query().reshape(40,2,3,2))
                reference_error = float((d(*x)-native(*x)).abs().max())
            else:
                a = d.head; key = a.key(torch.cat(x[:2],-1)).cpu().numpy()
                query = d.mapped_query().cpu().numpy()
                gamma = np.array([3-np.sqrt(2),3+np.sqrt(2)]*6)/24
                prob = np.exp(-((key[:,None]-query[None,:,None])**2*gamma).sum(-1))
                raw = (prob+1e-6)*x[2].cpu().numpy()[:,None]
                alpha = raw/raw.sum(-1,keepdims=True)
                expected = (np.einsum('brt,btd->brd',alpha,a.value(x[0]).cpu().numpy())
                            * a.decoder.cpu().numpy()).sum(-1)+a.bias.cpu().numpy()
                reference_error = float(np.max(abs(expected-d(*x).cpu().numpy())))
            assert reference_error < 1e-10
        buffers = {n:v.clone() for n,v in m.named_buffers()}
        m.train(); opt = torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
        before = m.query_weights.detach().clone()
        loss = F.cross_entropy(m(*inputs),labels); loss.backward()
        assert torch.isfinite(loss) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters())
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True); opt.step()
        assert not torch.equal(before,m.query_weights) and all(torch.isfinite(v).all() for v in m.parameters())
        assert all(torch.equal(dict(m.named_buffers())[n],v) for n,v in buffers.items())
        assert all(torch.equal(original.state_dict()[n],v) for n,v in original_state.items())
        count = sum(v.numel() for v in m.parameters()); assert count == 80644
        checks[kind] = dict(parameters=count,least_squares_normal_error=normal_error,
            query_chain_gradient_error=chain_error,query_finite_difference_error=finite_difference_error,
            native_or_numpy_reference_error=reference_error,padding=True,real8_update=True,
            original_models_unchanged=True,source_query_and_code_buffers_unchanged=True)
    return dict(status='PASS',checks=checks,code_rank=8,positive_rank=7,
        gram_preserved=True,scope='Only fixed code/query-sharing semantics; accepted core reused')


def summarize(rows, logits, labels, cfg):
    for kind,z in logits.items():
        cm = np.bincount(40*labels+z.argmax(-1),minlength=1600).reshape(40,40)
        den = cm.sum(0)+cm.sum(1)
        f = np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den!=0)
        rows[kind].update(all40_macro_f1=float(f.mean()),confusion_matrix=cm.tolist(),
            class_support=cm.sum(1).tolist(),per_class_f1=f.tolist(),
            training_loss='CE(raw_logits, labels)',assessment_loss='CE(raw_logits, labels)')
    assert tuple(rows) == MODES
    anchors = cfg['reused_native_rows']
    assert set(anchors) == {'quantum','dephased','gaussian_base','gaussian_covariance'}
    assert len({v['order_sha256'] for v in list(rows.values())+list(anchors.values())}) == 1
    assert all(v['updates']==1024 for v in rows.values())
    assert all(v['class_support']==rows['quantum_shared']['class_support'] for v in anchors.values())
    comparators = {k:v for k,v in rows.items() if k!='quantum_shared'}
    comparators.update({'reused/'+k:v for k,v in anchors.items()})
    gates = {'macro_beats_'+k:rows['quantum_shared']['all40_macro_f1']>v['all40_macro_f1']
             for k,v in comparators.items()}
    assert len(gates)==7
    return dict(results=rows,reused_results=anchors,gates=gates,
        gate='PASS_FOR_FULL_PROTOCOL_DESIGN_ONLY' if all(gates.values()) else 'DEFER_EXACT_SEMANTIC_QUERY_SCREEN',
        interpretation='Known semantic parameter sharing, not quantum novelty or a causal variance/coherence claim. Same public label-name structure for Q/D/G, deranged Q ablation. Registered count is not effective-capacity equality. No rank/tokenization/threshold/LR/seed/budget/subset/control rescue.',
        L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED')
