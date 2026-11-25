"""Actual fit8 qualification and frozen seven-Macro gate reporting."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from plugin import MODES, initialize, make_head, backward
from consistency import objective


def fixtures(models, inputs, labels):
    checks = {}
    paired_generators = [torch.Generator(device=torch.device('cuda',i)).manual_seed(13) for i in (0,1)]
    paired_draws_equal = []
    # Both lanes draw the same sequence for paired Q arms, not a shared RNG.
    for batch_size in (8,128,8):
        shape = (batch_size, inputs[0].shape[1], inputs[0].shape[2])
        draws = [torch.randn(shape,device=torch.device('cuda',i),dtype=inputs[0].dtype,
                             generator=paired_generators[i]).cpu() for i in (0,1)]
        equal = torch.equal(*draws)
        assert equal, 'Cross-device paired initial direction stream differs'
        paired_draws_equal.append(equal)
    for kind, original in models.items():
        before = {k:v.clone() for k,v in original.state_dict().items()}
        model = copy.deepcopy(original).to(inputs[0].device)
        # Disposable clone with a private generator; real campaign models/RNG untouched.
        generator = torch.Generator(device=inputs[0].device).manual_seed(13)
        loss, record = objective(model,inputs,labels,generator=generator,
                                directed=kind != 'quantum_random_consistency')
        assert bool(record['power_norm'].gt(0).all()) if kind != 'quantum_random_consistency' else True
        assert record['initial_zero_norm'] == 0
        loss.backward()
        assert loss.isfinite() and all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
        query = model.head.query if kind=='gaussian_vat' else model.head.effect
        query_min = float(query.grad.reshape(40,-1).norm(dim=-1).min())
        assert query_min > 0
        h,roles,mask = inputs
        radius = torch.where(mask[...,None],h,0.).double().flatten(1).norm(dim=1)*.01
        actual_radius = record['perturbation'].double().flatten(1).norm(dim=1)
        relative_radius_error = float(((actual_radius-radius).abs()/radius.clamp_min(1e-30)).max())
        assert relative_radius_error < 2e-6
        assert (record['perturbation'][~mask]==0).all()
        optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        assert all(p.isfinite().all() for p in model.parameters())
        assert any(not torch.equal(v,model.state_dict()[k].cpu()) for k,v in before.items())
        with torch.no_grad():
            z = model(*inputs)
            padded = (F.pad(h,(0,0,0,2),value=.8), F.pad(roles,(0,0,0,2),value=.9), F.pad(mask,(0,2)))
            padding_error = float((z-model(*padded)).abs().max())
            assert padding_error < 1e-4
        assert all(torch.equal(v,original.state_dict()[k]) for k,v in before.items())
        assert original.direction_generator is None and original.objective_totals['updates']==0
        checks[kind] = dict(parameters=sum(p.numel() for p in model.parameters()),
            actual8_clone_update=True,all40_query_gradient_min=query_min,
            relative_radius_error=relative_radius_error,padding_max_error=padding_error,
            original_model_rng_unchanged=True,ce=float(record['ce']),kl=float(record['kl']))
        assert checks[kind]['parameters']==81028
    return dict(status='PASS',scope='Disposable actual first8 fit clones; not utility evidence',checks=checks,
                cross_device_seed13_draws_equal=paired_draws_equal)


def summarize(rows, logits, labels, cfg):
    assert tuple(rows) == MODES
    for kind,z in logits.items():
        cm = np.bincount(40*labels+z.argmax(-1),minlength=1600).reshape(40,40)
        denominator = cm.sum(0)+cm.sum(1)
        f1 = np.divide(2*np.diag(cm),denominator,out=np.zeros(40),where=denominator!=0)
        rows[kind].update(all40_macro_f1=float(f1.mean()),confusion_matrix=cm.tolist(),
            class_support=cm.sum(1).tolist(),per_class_f1=f1.tolist())
        assert rows[kind]['objective_components']['updates']==cfg['updates']
    references = cfg['reused_native_rows']
    assert tuple(references)==('quantum','dephased','gaussian_base','gaussian_covariance')
    q = rows['quantum_vat']
    assert len({v['order_sha256'] for v in list(rows.values())+list(references.values())})==1
    assert all(v['class_support']==q['class_support'] and v['updates']==1024 for v in references.values())
    controls = {k:v for k,v in rows.items() if k!='quantum_vat'}
    controls.update({'reused/'+k:v for k,v in references.items()})
    gates = {'macro_beats_'+k:q['all40_macro_f1']>v['all40_macro_f1'] for k,v in controls.items()}
    assert len(gates)==7
    return dict(results=rows,reused_results=references,gates=gates,
        gate='PASS_FOR_FULL_PROTOCOL_DESIGN_ONLY' if all(gates.values()) else 'DEFER_EXACT_OUTPUT_CONSISTENCY_SCREEN',
        L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED',interpretation='Known VAT on frozen embedding inputs; no semantic guarantee or newT')
