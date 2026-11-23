"""Frozen TRAIN32 competition probe; no fitting, selection, new data or model updates."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, v):
    p.write_bytes((json.dumps(v, indent=2, allow_nan=False)+'\n').encode())


def main():
    ap = argparse.ArgumentParser()
    for n in ('root', 'design', 'output', 'progress'):
        ap.add_argument('--'+n, type=Path, required=True)
    ap.add_argument('--attempt', required=True)
    args = ap.parse_args(); cfg = json.loads(args.design.read_bytes()); root = args.root
    for n,h in cfg['code_hashes'].items():
        assert sha(Path(__file__).parent/n) == h
    for n,h in cfg['input_hashes'].items():
        assert sha(root/n) == h
    assert not args.output.exists(); args.output.mkdir(parents=True)
    start = time.monotonic()
    import torch
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(root/'source'))
    spec=importlib.util.spec_from_file_location('qattn_relation_competition',Path(__file__).with_name('competition.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    attention,CompetitiveHead=module.attention,module.CompetitiveHead
    from baselines.retacred.rmea import RMEAHead, circuit, EPS
    source = root/'control/semantic-native-detail-20261123/output'
    rows = json.loads((source/'cases.private.json').read_bytes())
    selection = json.loads((root/'control/semantic-problem-audit-20261123/selection.private.json').read_bytes())
    labels = json.loads((root/'inputs/native-baseline/recovery-inputs/labels.json').read_bytes())
    types = json.loads((root/'control/task-bottleneck-discovery-20261123/cases.private.json').read_bytes())['type_metadata']
    with np.load(root/'control/native-ordered-mass-value-20261123/output/diagnostic.private.npz', allow_pickle=False) as z:
        fit = z['calibration'].copy()
    support={};seen=set()
    for i in np.flatnonzero(fit):
        g,s,o=types[i];support.setdefault((s,o),set()).add(g);seen.add(g)
    head = RMEAHead('quantum').double()
    weights = torch.load(root/'inputs/native-checkpoint/native-final.pt',map_location='cpu',weights_only=True)
    head.load_state_dict(weights); head.eval()
    assert set(CompetitiveHead('competitive').state_dict()) == set(weights)
    with torch.no_grad():
        phi = circuit(torch.pi*torch.tanh(head.effect))
        identity = torch.eye(8, dtype=torch.complex128)
        effects = phi[:,:,None]*phi[:,None,:].conj()+EPS*identity[None]
        frame = effects.sum(0)
        eigenvalues, basis = torch.linalg.eigh(frame)
        assert eigenvalues.min() > 0
        sqrt = (basis*eigenvalues.sqrt())@basis.conj().T
        invsqrt = (basis*eigenvalues.rsqrt())@basis.conj().T
        povm = invsqrt[None]@effects@invsqrt[None]
        povm_error = float((povm.sum(0)-identity).abs().max())
    assert math.isfinite(povm_error)
    errors = dict(native=0., covariance=0., gradient=0., povm=povm_error, frame_probability=0., padding=0.)
    def accumulate(name,value):
        value=float(value)
        assert math.isfinite(value),(name,'nonfinite residual')
        errors[name]=max(errors[name],value)
    records=[]; saved={}; changed=0
    with np.load(source/'details.private.npz',allow_pickle=False) as arrays:
        for rec in rows:
            key=rec['case_id']; old=torch.from_numpy(arrays[key+'_logits'].copy())
            oldalpha=torch.from_numpy(arrays[key+'_alpha'].copy())
            u=torch.from_numpy(arrays[key+'_utilities'].copy())[None]
            q=torch.from_numpy(arrays[key+'_probability'].copy())[None]+EPS
            mask=torch.ones((1,q.shape[-1]),dtype=torch.bool)
            alpha=attention(q,mask); native=attention(q,mask,False)
            accumulate('native',(native[0]-oldalpha).abs().max())
            bias=old-(oldalpha*u[0]).sum(-1)
            z=(alpha*u).sum(-1)[0]+bias
            changed+=int(float((alpha-native).abs().max())>1e-10)
            invmass=q.sum(1,keepdim=True).reciprocal()
            mu=(native*u).sum(-1,keepdim=True)
            delta=(native*(u-mu)*invmass).sum(-1)/(native*invmass).sum(-1)
            accumulate('covariance',(z-old-delta[0]).abs().max())
            differentiable=q.clone().requires_grad_()
            weights_for_loss=torch.linspace(-1.,1.,40,dtype=torch.float64)[None]
            da=attention(differentiable,mask); mean=(da*u).sum(-1,keepdim=True)
            loss=((da*u).sum(-1)*weights_for_loss).sum()
            actual=torch.autograd.grad(loss,differentiable)[0]*differentiable
            d=weights_for_loss[:,:,None]*da*(u-mean)
            expected=d-(differentiable/differentiable.sum(1,keepdim=True))*d.sum(1,keepdim=True)
            accumulate('gradient',(actual-expected).abs().max().detach())
            padded=torch.cat((q,torch.full(q.shape[:2]+(2,),0.25,dtype=q.dtype)),dim=-1)
            pmask=torch.cat((mask,torch.zeros((1,2),dtype=torch.bool)),dim=-1)
            padded_alpha=attention(padded,pmask)
            accumulate('padding',(padded_alpha[:,:,:-2]-alpha).abs().max())
            accumulate('padding',padded_alpha[:,:,-2:].abs().max())
            psi=torch.from_numpy(arrays[key+'_psi'].copy())
            with torch.no_grad():
                transformed=psi@sqrt.T
                transformed/=transformed.norm(dim=-1,keepdim=True)
                measured=torch.einsum('td,rde,te->rt',transformed.conj(),povm,transformed).real
                accumulate('frame_probability',(measured-q[0]/q.sum(1)[0]).abs().max())
            selected=next(r for r in selection if r['case_id']==key)
            pair=tuple(types[selected['row']][1:]);allow=torch.ones(40,dtype=torch.bool)
            if pair in support:
                allow[:]=False;allow[0]=True
                for k in support[pair]|(set(range(40))-seen):allow[k]=True
            g=labels.index(rec['gold_label']);assert allow[g]
            prediction=int(z.masked_fill(~allow,-torch.inf).argmax())
            rivals=z.masked_fill(~allow,-torch.inf).clone();rivals[g]=-torch.inf
            cue=arrays[key+'_cue_indices'];eligible=rec['semantic_gold_cue_eligible']
            records.append(dict(case_id=key,family=rec['family'],gold=g,old_prediction=labels.index(rec['typed_prediction']),
                prediction=prediction,old_margin=rec['original_gold_margin'],margin=float(z[g]-rivals.max()),
                semantic_gold_cue_eligible=eligible,
                old_cue_mass=float(oldalpha[g,cue].sum()) if eligible else None,
                new_cue_mass=float(alpha[0,g,cue].sum()) if eligible else None,
                old_null=float(old[0]),new_null=float(z[0]),old_target=float(old[g]),new_target=float(z[g])))
            saved[key+'_logits']=z.numpy();saved[key+'_alpha']=alpha[0].numpy()
    assert len(records)==32 and changed>0
    assert max(errors.values())<1e-8,errors
    strata={}
    for family in ('positive_missed','positive_correct','null_false_positive','null_correct'):
        r=[v for v in records if v['family']==family]
        strata[family]=dict(rows=len(r),repairs=sum(v['old_prediction']!=v['gold'] and v['prediction']==v['gold'] for v in r),
            damages=sum(v['old_prediction']==v['gold'] and v['prediction']!=v['gold'] for v in r),
            changed=sum(v['prediction']!=v['old_prediction'] for v in r),
            old_correct=sum(v['old_prediction']==v['gold'] for v in r),new_correct=sum(v['prediction']==v['gold'] for v in r))
    supported=[v for v in records if v['family']=='positive_missed' and v['semantic_gold_cue_eligible']]
    assert len(supported)==5
    np.savez(args.output/'probe.private.npz',**saved);save(args.output/'cases.private.json',records)
    report=dict(status='completed',attempt=args.attempt,seconds=time.monotonic()-start,rows=32,
        formula='q=Born+EPS; c_rt=q_rt/sum_s q_st; alpha_rt=c_rt/sum_t c_rt; z_r=sum_t alpha_rt*u_rt+bias_r',
        changed_attention_examples=changed,errors=errors,strata=strata,
        supported_misses=dict(rows=5,repairs=sum(v['prediction']==v['gold'] for v in supported),
                              cue_mass_increases=sum(v['new_cue_mass']>v['old_cue_mass']+1e-10 for v in supported),
                              margins=[dict(old=v['old_margin'],new=v['margin']) for v in supported]),
        optimizer_updates=0,new_model_word_forwards=0,valid_rows=0,test_rows=0,cos_writes=0,
        private_arrays_sha256=sha(args.output/'probe.private.npz'),private_cases_sha256=sha(args.output/'cases.private.json'),
        scope='Fixed exposed outcome-stratified TRAIN32, same checkpoint values and Born scores; label-free transform, labels only for scoring. '
              'Known finite column/row normalization and POVM identity, not new theory. No trained utility, population estimate, '
              'independent evidence, phase attribution or L5. No hyperparameter selection. Negative is only this frozen intervention, not all training.')
    save(args.output/'summary.json',report);save(args.output/'RUN_COMPLETE',dict(summary_sha256=sha(args.output/'summary.json')))
    save(args.progress,dict(schema='research.progress.v1',attempt=args.attempt,phase='completed',current=32,total=32,unit='example',observed_at=time.time(),heartbeat_at=time.time()))
    print(json.dumps(report))


if __name__=='__main__':main()
