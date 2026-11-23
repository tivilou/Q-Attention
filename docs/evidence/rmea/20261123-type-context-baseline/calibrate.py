"""Bounded conventional convex calibration, frozen native outputs, no model replay."""
from pathlib import Path
from collections import defaultdict
import argparse
import hashlib
import json
import os
import signal
import sys
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax


def save(p,v):
    p=Path(p);tmp=p.with_suffix('.pending');tmp.write_text(json.dumps(v,indent=2)+'\n');os.replace(tmp,p)


def metrics(y,p):
    cm=np.zeros((40,40),np.int64);np.add.at(cm,(y,p),1)
    tp=cm.diagonal();den=cm.sum(0)+cm.sum(1)
    f=np.divide(2*tp,den,out=np.zeros(40),where=den>0)
    return dict(items=len(y),correct=int(tp.sum()),positive39_macro_f1=float(f[1:].mean()),
        null_false_positive=int(((y==0)&(p>0)).sum()),positive_false_negative=int(((y>0)&(p==0)).sum()),
        positive_wrong_relation=int(((y>0)&(p>0)&(y!=p)).sum()))


def inputs_ok(z,y,n,metadata):
    assert z.shape==(n,40) and np.issubdtype(z.dtype,np.floating) and np.isfinite(z).all()
    assert y.shape==(n,) and np.issubdtype(y.dtype,np.integer) and ((y>=0)&(y<40)).all()
    assert isinstance(metadata,list) and len(metadata)==n
    assert all(isinstance(m,list) and len(m)==3 and isinstance(m[0],int) and m[0]==int(y[i]) and
        isinstance(m[1],str) and m[1] and isinstance(m[2],str) and m[2] for i,m in enumerate(metadata))


def objective(theta,z,y,t,groups,penalty):
    parameters=theta.reshape(groups+1,40)
    scores=z+parameters[0]
    if groups:scores=scores+parameters[t+1]
    value=float(np.mean(logsumexp(scores,axis=1)-scores[np.arange(len(y)),y]))
    delta=softmax(scores,axis=1);delta[np.arange(len(y)),y]-=1;delta/=len(y)
    grad=np.zeros_like(parameters);grad[0]=delta.sum(0)
    if groups:np.add.at(grad,t+1,delta)
    return value+.5*penalty*float(theta@theta),(grad+penalty*parameters).ravel()


def main():
    if sys.flags.optimize:
        raise RuntimeError('Optimized Python is not supported by this checked diagnostic')
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--design',type=Path,required=True)
    parser.add_argument('--progress',type=Path,required=True);parser.add_argument('--attempt',required=True)
    args=parser.parse_args();cfg=json.loads(args.design.read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==cfg['code_sha256']
    for name,expected in cfg['additional_input_hashes'].items():
        assert hashlib.sha256((args.root/name).read_bytes()).hexdigest()==expected
    assert not (args.output/'started.json').exists(),'Preserve this attempt; no scientific retry'
    args.output.mkdir(parents=True);save(args.output/'started.json',dict(attempt=args.attempt,design_sha256=hashlib.sha256(args.design.read_bytes()).hexdigest()))
    signal.alarm(cfg['timeout_seconds']);start=time.monotonic()
    def progress(phase,k):
        save(args.progress,dict(schema='research.progress.v1',attempt=args.attempt,phase=phase,current=k,total=4,unit='stage',observed_at=time.time(),heartbeat_at=time.time()))
    progress('fit_global_and_type_baselines',0)
    original=args.root/'control/task-bottleneck-discovery-20261123'
    p=args.root/'control/native-ordered-mass-value-20261123/output/diagnostic.private.npz'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==cfg['train_predictions_sha256']
    with np.load(p,allow_pickle=False) as a:z=a['logits'].copy();y=a['labels'].copy();fit=a['calibration'].copy()
    meta=json.loads((original/'cases.private.json').read_text())['type_metadata']
    inputs_ok(z,y,8192,meta)
    assert fit.shape==(8192,) and fit.dtype==np.bool_ and fit.sum()==4192 and (~fit).sum()==4000
    cache=json.loads((args.root/'inputs/native-baseline/discovery/cache/metadata.json').read_text())
    proof=json.loads((args.root/'control/type-context-baseline-20261123/calibration-row-provenance.json').read_text())
    producer_proof=json.loads((args.root/'control/type-context-baseline-20261123/prediction-row-provenance.json').read_text())
    for split,selector,n in [('train','logits',8192),('valid','logits[0]',19584)]:
        binding=[q for q in producer_proof['records'] if q['split']==split]
        assert len(binding)==1
        binding=binding[0]
        assert binding['archive_sha256']==cfg[split+'_predictions_sha256'] and binding['selected_array']==selector
        assert binding['shape']==[n,40] and binding['row_id_order_sha256']==proof[split]['row_id_order_sha256']
        assert binding['successful_original_execution_verified']
    for split,expected_n,metadata in [('train',8192,meta)]:
        words=json.loads((args.root/f'inputs/native-word-metadata-20261123/{split}.private.json').read_text())
        ids=[w['id_hash'] for w in words]
        assert len(ids)==expected_n and ids==[w['id_hash'] for w in cache[split][:expected_n]]
        assert hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==proof[split]['row_id_order_sha256']
        assert hashlib.sha256(json.dumps(metadata,separators=(',',':')).encode()).hexdigest()==proof[split]['records_sha256']
        assert proof[split]['all_source_ids_and_types_match'] and proof[split]['word_projection_and_cache_ids_match']
    pairs=sorted({(meta[i][1],meta[i][2]) for i in np.flatnonzero(fit)})
    index={pair:i for i,pair in enumerate(pairs)}
    types=np.array([index.get((m[1],m[2]),-1) for m in meta])
    assert len(pairs)==25 and (types[fit]>=0).all()
    # Focused changed-math check, before either fit; fixed directional finite difference.
    theta=np.linspace(-.05,.05,120);fixture_z=z[fit][:17];fixture_y=y[fit][:17];fixture_t=np.arange(17)%2
    direction=np.cos(np.arange(120));direction/=np.linalg.norm(direction);eps=1e-5
    f,g=objective(theta,fixture_z,fixture_y,fixture_t,2,cfg['penalty'])
    fd=(objective(theta+eps*direction,fixture_z,fixture_y,fixture_t,2,cfg['penalty'])[0]-objective(theta-eps*direction,fixture_z,fixture_y,fixture_t,2,cfg['penalty'])[0])/(2*eps)
    fixture_error=abs(fd-float(g@direction));assert fixture_error<1e-8
    fits={};theta_by_mode={}
    for k,(mode,group_count) in enumerate((('global',0),('type',len(pairs))),1):
        initial=np.zeros((group_count+1)*40)
        result=minimize(objective,initial,args=(z[fit],y[fit],types[fit],group_count,cfg['penalty']),jac=True,method='L-BFGS-B',
            options=dict(maxiter=cfg['max_iterations'],gtol=cfg['gradient_inf']/10,ftol=1e-14,maxls=40))
        f,g=objective(result.x,z[fit],y[fit],types[fit],group_count,cfg['penalty'])
        gradient=float(abs(g).max());assert result.success and gradient<=cfg['gradient_inf']
        assert np.isfinite(result.x).all() and np.isfinite(f) and np.isfinite(g).all() and np.isfinite(result.fun)
        theta_by_mode[mode]=result.x.reshape(group_count+1,40)
        fits[mode]=dict(parameters=len(initial),iterations=int(result.nit),objective=f,gradient_inf=gradient,converged=True)
        progress('fits_complete_'+mode,k)
    # Seal both models before any assessment or VALID scoring; earlier provenance reads bytes.
    np.savez(args.output/'coefficients.private.npz',**theta_by_mode)
    coefficient_sha=hashlib.sha256((args.output/'coefficients.private.npz').read_bytes()).hexdigest()
    save(args.output/'fits.json',dict(fits=fits,coefficient_sha256=coefficient_sha))
    def scores(native,metadata,mode):
        b=theta_by_mode[mode];out=native+b[0]
        if mode=='type':
            for i,m in enumerate(metadata):
                j=index.get((m[1],m[2]))
                if j is not None:out[i]+=b[j+1]
        assert np.isfinite(out).all()
        return out
    results={};saved={}
    for split,native,targets,metadata in [('train_assessment',z[~fit],y[~fit],[meta[i] for i in np.flatnonzero(~fit)])]:
        results[split]={'native':metrics(targets,native.argmax(1))}
        for mode in ('global','type'):
            adjusted=scores(native,metadata,mode);pred=adjusted.argmax(1)
            results[split][mode]=metrics(targets,pred);saved[split+'_'+mode]=pred
    vp=args.root/'control/native-context-dephasing-20261123/output/predictions.private.npz'
    assert hashlib.sha256(vp.read_bytes()).hexdigest()==cfg['valid_predictions_sha256']
    with np.load(vp,allow_pickle=False) as a:vz=a['logits'][0].copy();vy=a['labels'].copy()
    vm=json.loads((original/'validation-cases.private.json').read_text())['metadata']
    inputs_ok(vz,vy,19584,vm)
    vw=json.loads((args.root/'inputs/native-word-metadata-20261123/valid.private.json').read_text());ids=[w['id_hash'] for w in vw]
    assert len(ids)==19584 and ids==[w['id_hash'] for w in cache['valid']]
    assert hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==proof['valid']['row_id_order_sha256']
    assert hashlib.sha256(json.dumps(vm,separators=(',',':')).encode()).hexdigest()==proof['valid']['records_sha256']
    assert proof['valid']['all_source_ids_and_types_match'] and proof['valid']['word_projection_and_cache_ids_match']
    results['development_valid']={'native':metrics(vy,vz.argmax(1))}
    for mode in ('global','type'):
        pred=scores(vz,vm,mode).argmax(1);saved['valid_'+mode]=pred
        results['development_valid'][mode]=metrics(vy,pred)
    progress('development_evaluation_complete',3)
    frozen=json.loads((original/'validation-summary.json').read_text())
    assert frozen['data']['valid_count']==19584 and frozen['data']['prediction_archive_sha256']==cfg['valid_predictions_sha256']
    assert frozen['data']['raw_source_sha256']==proof['valid']['source_sha256']
    assert abs(results['development_valid']['native']['positive39_macro_f1']-frozen['native']['positive39_macro_f1'])<1e-14
    # Recompute the unchanged empirical rule with the same 40-class confusion/39-positive metric.
    support=defaultdict(set);seen=set()
    for i in np.flatnonzero(fit):
        g,st,ot=meta[i];support[(st,ot)].add(g);seen.add(g)
    support_sha=hashlib.sha256(json.dumps(sorted((list(pair),sorted(ids)) for pair,ids in support.items()),separators=(',',':')).encode()).hexdigest()
    assert support_sha==cfg['frozen_support_sha256']
    allow=np.ones(vz.shape,bool)
    for i,(_,st,ot) in enumerate(vm):
        pair=(st,ot)
        if pair not in support:continue
        allow[i]=False;allow[i,0]=True
        for k in support[pair]|(set(range(40))-seen):allow[i,k]=True
    hard_prediction=np.where(allow,vz,-np.inf).argmax(1)
    hard=metrics(vy,hard_prediction)
    assert abs(hard['positive39_macro_f1']-frozen['projected']['positive39_macro_f1'])<1e-14
    assert hard['correct']==frozen['projected']['correct']
    results['development_valid']['frozen_hard_type_rule']=hard
    saved['valid_frozen_hard_type_rule']=hard_prediction
    np.savez(args.output/'predictions.private.npz',**saved)
    r=results['development_valid'];success=(r['type']['positive39_macro_f1']>r['native']['positive39_macro_f1'] and r['type']['positive39_macro_f1']>r['global']['positive39_macro_f1'] and r['type']['positive39_macro_f1']>r['frozen_hard_type_rule']['positive39_macro_f1'])
    report=dict(profile=cfg['profile'],attempt=args.attempt,source=dict(train_predictions_sha256=cfg['train_predictions_sha256'],valid_predictions_sha256=cfg['valid_predictions_sha256'],fit_rows=4192,assessment_rows=4000,type_pairs=len(pairs),penalty=cfg['penalty'],frozen_coefficients_sha256=coefficient_sha),fits=fits,results=results,fixture_directional_gradient_error=fixture_error,
        decision='TYPE_CALIBRATION_EXCEEDS_ALL_FIXED_REFERENCES' if success else 'NO_PROMOTION_CLOSE_EXACT_CALIBRATION_PROFILE',
        scope='Conventional CPU development calibration of frozen Q-RMEA logits with given types; no retrained classical encoder, independent confirmation, quantum advantage or L5. Unequal parameter counts explicitly shown.',
        stop='No penalty, feature, calibration split, budget, comparator or evaluation rescue.',execution=dict(seconds=time.monotonic()-start,model_forwards=0,optimizer_models=2,gpu_compute=0,cos_writes=0))
    save(args.output/'summary.json',report);(args.output/'RUN_COMPLETE').write_text('completed\n');progress('completed',4)
    print(json.dumps(report))


if __name__=='__main__':main()
