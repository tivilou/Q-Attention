"""Calibration-selected shared native effect direction; zero optimizer updates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time
import numpy as np


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,v):
    pending=p.with_suffix('.pending');pending.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');pending.replace(p)


def tables(z,dz,y,cal,tol):
    exp=np.exp(z-z.max(-1,keepdims=True));prob=exp/exp.sum(-1,keepdims=True)
    coeff=prob.copy();coeff[np.arange(len(y)),y]-=1
    ce_dot=np.sum(coeff*dz,-1)
    other=z.copy();other[np.arange(len(y)),y]=-np.inf
    ties=other==other.max(-1,keepdims=True)
    margin_dot=dz[np.arange(len(y)),y]-np.where(ties,dz,-np.inf).max(-1)
    pred=z.argmax(-1);error=pred!=y;out={}
    for part,take in dict(all=np.ones(len(y),bool),calibration=cal,assessment=~cal).items():
        masks=dict(all=take,native_errors=take&error,native_correct=take&~error,
            positive_to_zero=take&(y!=0)&(pred==0),zero_to_positive=take&(y==0)&(pred!=0),
            wrong_positive=take&error&(y!=0)&(pred!=0))
        table={}
        for name,mask in masks.items():
            threshold=tol*np.maximum(1,np.max(np.abs(dz),-1))
            table[name]=dict(n=int(mask.sum()),ce_directional_derivative=float(ce_dot[mask].mean()) if mask.any() else None,
                margin_derivative_mean=float(margin_dot[mask].mean()) if mask.any() else None,
                margin_increases=int((mask&(margin_dot>threshold)).sum()),
                margin_decreases=int((mask&(margin_dot < -threshold)).sum()),
                margin_unresolved=int((mask&(np.abs(margin_dot)<=threshold)).sum()))
        out[part]=table
    return out,ce_dot,margin_dot


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('source','inputs','reference','checkpoint','design','output','progress','allocation'):
        ap.add_argument('--'+n,required=True,type=Path)
    ap.add_argument('--attempt',required=True);a=ap.parse_args()
    cfg=json.loads(a.design.read_bytes());allocation=json.loads(a.allocation.read_bytes())['allocation']
    assert allocation['runner_attempt_id']==a.attempt and len(allocation['gpu_uuids'])==1
    for n,h in cfg['code_hashes'].items():assert digest(Path(__file__).parent/n)==h
    for n,h in cfg['source_hashes'].items():assert digest(a.source/n)==h
    assert digest(a.checkpoint)==cfg['checkpoint_sha256']
    assert digest(a.inputs/'discovery/manifest.json')==cfg['native_manifest_sha256']
    prior=a.reference/'output/diagnostic.private.npz';assert digest(prior)==cfg['split_reference_sha256']
    a.output.mkdir(exist_ok=True);assert not (a.output/'started.json').exists()
    write(a.output/'started.json',dict(attempt=a.attempt,design_sha256=digest(a.design)))
    signal.alarm(cfg['timeout_seconds']);started=time.time()
    import torch
    from torch.nn import functional as F
    sys.path.insert(0,str(a.source))
    from baselines.retacred.rmea import RMEAHead,circuit,EPS
    assert EPS==1e-6
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);assert torch.cuda.device_count()==1
    model=RMEAHead('quantum').double()
    model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True));model.to('cuda');model.eval()
    for p in model.parameters():p.requires_grad_(False)
    model.effect.requires_grad_(True)
    assert [n for n,p in model.named_parameters() if p.requires_grad]==['effect']
    assert model.effect.shape==(40,2,3,2) and model.effect.numel()==480
    initial=model.effect.detach().clone()
    with np.load(prior,allow_pickle=False) as q:
        cal=q['calibration'].copy();groups=q['groups'].copy();y=q['labels'].copy();prior_z=q['logits'].copy()
    assert len(y)==8192 and int(cal.sum())==4192
    assert not set(groups[cal])&set(groups[~cal])
    cache=a.inputs/'discovery/cache'
    data={k:np.load(cache/('train-'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('h','roles','offsets','targets')}
    assert np.array_equal(data['targets'][:8192],y)

    def batch(ids):
        lengths=[int(data['offsets'][i+1]-data['offsets'][i]) for i in ids]
        h=np.zeros((len(ids),max(lengths),1024),np.float64);r=np.zeros(h.shape[:2]+(4,),np.float64);mask=np.zeros(h.shape[:2],bool)
        for j,i in enumerate(ids):
            lo,hi=map(int,data['offsets'][i:i+2]);n=hi-lo
            h[j,:n]=data['h'][lo:hi];r[j,:n]=data['roles'][lo:hi];mask[j,:n]=True
        return tuple(torch.from_numpy(x).to('cuda') for x in (h,r,mask)),torch.tensor(y[ids],device='cuda',dtype=torch.long)

    def progress(phase,current,total):
        write(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,phase=phase,
            current=current,total=total,unit='example',observed_at=time.time(),heartbeat_at=time.time()))

    fit=np.flatnonzero(cal);assessment=np.flatnonzero(~cal);gcal=torch.zeros_like(model.effect)
    # Direction is chosen ONCE using calibration only. No assessment labels
    # enter this pass, no optimizer exists, and source forward is unchanged.
    for lo in range(0,len(fit),cfg['microbatch_size']):
        ids=fit[lo:lo+cfg['microbatch_size']];inputs,target=batch(ids)
        z,_=model(*inputs);loss=F.cross_entropy(z,target,reduction='sum')/len(fit)
        gcal+=torch.autograd.grad(loss,model.effect)[0]
        if lo%128==0:progress('calibration_effect_gradient',min(lo+len(ids),len(fit)),len(fit))
    assert torch.isfinite(gcal).all();norm=float(gcal.norm())
    if norm<=cfg['zero_gradient_tolerance']:
        result=dict(status='no_direction',gradient_norm=norm,optimizer_updates=0,seconds=time.time()-started,
            reason='Specified calibration gradient vanished; no substitute direction',design_sha256=digest(a.design))
        write(a.output/'summary.json',result);write(a.output/'fixtures.json',dict(status='not_applicable'))
        write(a.output/'RUN_COMPLETE',dict(files={n:digest(a.output/n) for n in ('summary.json','fixtures.json')}))
        print(json.dumps(result));return
    direction=-gcal/norm
    np.save(a.output/'direction.private.npy',direction.detach().cpu().numpy(),allow_pickle=False)
    frozen=dict(direction_sha256=digest(a.output/'direction.private.npy'),calibration_gradient_norm=norm,
        chosen_at=time.time(),selection='Negative Euclidean calibration mean-CE gradient in all480 native effect coordinates, normalized once')
    write(a.output/'direction-frozen.json',frozen)
    phi,dphi=torch.autograd.functional.jvp(lambda theta:circuit(math.pi*torch.tanh(theta)),initial,direction)
    assert torch.isfinite(dphi).all()
    h=cfg['finite_difference_step']
    plus=circuit(math.pi*torch.tanh(initial+h*direction));minus=circuit(math.pi*torch.tanh(initial-h*direction))
    phi_error=float((dphi-(plus-minus)/(2*h)).abs().max());assert phi_error<cfg['finite_difference_tolerance']
    ga=torch.zeros_like(model.effect);logits=[];dots=[];max_native_error=0.;max_fd_error=0.
    max_phi_error=0.;assessment_started=time.time()
    assert assessment_started>=frozen['chosen_at']
    for lo in range(0,len(y),cfg['microbatch_size']):
        ids=np.arange(lo,min(lo+cfg['microbatch_size'],len(y)));inputs,target=batch(ids)
        detail=model(*inputs,details=True);z=detail['logits'];psi=detail['psi'].detach();u=detail['utilities'].detach()
        assert torch.allclose(detail['phi'],phi,atol=1e-12,rtol=0)
        native_probability=detail['probability'].detach();mask=inputs[2]
        amplitude=torch.einsum('rd,btd->brt',phi.conj(),psi)
        amplitude_dot=torch.einsum('rd,btd->brt',dphi.conj(),psi)
        p_dot=2*(amplitude.conj()*amplitude_dot).real
        # Differentiate source normalization explicitly; EPS is constant.
        q=(native_probability+EPS)*mask[:,None];normalizer=q.sum(-1,keepdim=True)
        p_dot=p_dot*mask[:,None];pooled=(q*u).sum(-1,keepdim=True)/normalizer
        dz=(p_dot*(u-pooled)).sum(-1)/normalizer.squeeze(-1)
        assert torch.isfinite(dz).all()
        take=np.flatnonzero(~cal[ids])
        if len(take):
            loss=F.cross_entropy(z[take],target[take],reduction='sum')/len(assessment)
            ga+=torch.autograd.grad(loss,model.effect)[0]
        zz=z.detach().cpu().numpy();dd=dz.detach().cpu().numpy()
        assert np.array_equal(zz.argmax(-1),prior_z[ids].argmax(-1))
        error=float(np.max(np.abs(zz-prior_z[ids])));assert error<1e-8
        max_native_error=max(max_native_error,error)
        if lo<cfg['fixture_rows']:
            def direct(effects):
                p=torch.einsum('rd,btd->brt',effects.conj(),psi).abs().square()
                raw=(p+EPS)*mask[:,None];return (raw*u).sum(-1)/raw.sum(-1)+model.bias
            fd=(direct(plus)-direct(minus))/(2*h)
            fe=float((fd-dz).abs().max());assert fe<cfg['finite_difference_tolerance'];max_fd_error=max(max_fd_error,fe)
        logits.append(zz);dots.append(dd)
        max_phi_error=max(max_phi_error,float((detail['phi'].detach()-phi).abs().max()))
        if lo%128==0 or lo+len(ids)==len(y):progress('shared_effect_direction_transfer',lo+len(ids),len(y))
    z=np.concatenate(logits);dz=np.concatenate(dots)
    report,ce_dot,margin_dot=tables(z,dz,y,cal,cfg['derivative_tolerance'])
    inner=float((ga*direction).sum());rowmean=float(ce_dot[~cal].mean());ct=float(ce_dot[cal].mean())
    gradient_error=abs(inner-rowmean);calibration_error=abs(ct+norm)
    assert gradient_error<1e-9 and calibration_error<1e-9
    assert torch.equal(model.effect.detach(),initial)
    assert digest(a.output/'direction.private.npy')==frozen['direction_sha256']
    assert digest(a.checkpoint)==cfg['checkpoint_sha256']
    threshold=cfg['derivative_tolerance']*max(1.,float(ga.norm()))
    sign='negative' if rowmean < -threshold else 'positive' if rowmean>threshold else 'unresolved'
    private=a.output/'diagnostic.private.npz'
    np.savez_compressed(private,logits=z,logit_derivative=dz,labels=y,calibration=cal,groups=groups,
        gcal=gcal.cpu().numpy(),gassessment=ga.cpu().numpy(),direction=direction.cpu().numpy(),
        ce_derivative=ce_dot,margin_derivative=margin_dot)
    fixture=dict(status='PASS',phi_jvp_finite_difference_error=phi_error,first8_logit_finite_difference_error=max_fd_error,
        assessment_gradient_inner_product_error=gradient_error,calibration_gradient_identity_error=calibration_error,
        native_prior_max_error=max_native_error,native_phi_max_error=max_phi_error,weights_unchanged=True)
    summary=dict(status='complete',primary_assessment_ce_derivative=rowmean,primary_sign=sign,
        primary_numerical_threshold=threshold,calibration_gradient_norm=norm,assessment_gradient_norm=float(ga.norm()),
        gradient_cosine=float((gcal*ga).sum()/(gcal.norm()*ga.norm())) if float(ga.norm())>0 else None,
        partitions=report,seconds=time.time()-started,optimizer_updates=0,
        design_sha256=digest(a.design),checkpoint_sha256=cfg['checkpoint_sha256'],diagnostic_sha256=digest(private),
        direction_frozen=frozen,assessment_started_at=assessment_started,claim_scope=cfg['claim_scope'])
    write(a.output/'summary.json',summary);write(a.output/'fixtures.json',fixture)
    write(a.output/'RUN_COMPLETE',dict(files={n:digest(a.output/n) for n in
        ('summary.json','fixtures.json','direction-frozen.json','direction.private.npy','diagnostic.private.npz')}))
    print(json.dumps(dict(status='complete',primary_sign=sign,ce_derivative=rowmean,seconds=summary['seconds'],
        native_errors=report['assessment']['native_errors'],native_correct=report['assessment']['native_correct'])))


if __name__=='__main__':main()
