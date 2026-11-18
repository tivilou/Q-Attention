"""Original validation data only: compare fixed readout probes, never retrain heads."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'rmea_original_scores'))
import job as original
from common import guard,sha,read,complete

def sigmoid(x):
    a=np.exp(-np.abs(x));return np.where(x>=0,1/(1+a),a/(1+a))

def objective(design,y,w,penalty):
    z=design@w
    return float(np.mean(np.logaddexp(0,z)-y*z)+penalty/2*np.dot(w[1:],w[1:]))

def fit_predict(train,y,test,penalty):
    train=np.asarray(train,np.float64);test=np.asarray(test,np.float64);y=np.asarray(y,np.float64)
    assert train.ndim==test.ndim==2 and train.shape[1]==test.shape[1] and len(y)==len(train)
    assert np.isfinite(train).all() and np.isfinite(test).all() and set(y)=={0.,1.}
    mean=train.mean(0);scale=train.std(0);scale=np.where(scale>1e-12,scale,1)
    X=np.column_stack((np.ones(len(train)),(train-mean)/scale))
    Q=np.column_stack((np.ones(len(test)),(test-mean)/scale));w=np.zeros(X.shape[1])
    w[0]=np.log(y.mean()/(1-y.mean()));reg=np.r_[0.,np.full(len(w)-1,penalty)]
    for step in range(50):
        p=sigmoid(X@w);gradient=X.T@(p-y)/len(y)+reg*w
        if np.max(abs(gradient))<=1e-8:break
        H=X.T@(np.maximum(p*(1-p),1e-12)[:,None]*X)/len(y)+np.diag(reg)
        delta=np.linalg.solve(H,gradient);before=objective(X,y,w,penalty)
        for backtrack in range(24):
            rate=2.**(-backtrack);candidate=w-rate*delta
            if objective(X,y,candidate,penalty)<=before-1e-4*rate*np.dot(gradient,delta):
                w=candidate;break
        else:raise AssertionError('Fixed solver line search failed; no adaptive retry')
    else:raise AssertionError('Fixed solver did not converge; no penalty/budget adjustment')
    assert np.max(abs(X.T@(sigmoid(X@w)-y)/len(y)+reg*w))<=1e-8
    return sigmoid(Q@w),dict(iterations=step,gradient_infinity_norm=float(np.max(abs(gradient))),
        input_features=train.shape[1],fitted_parameters=len(w),penalty=penalty,fit_entries=len(y))

def dispersion(alpha,utilities):
    """Conditional token-utility variance, not Born uncertainty or label probability."""
    assert alpha.dtype==utilities.dtype==np.float32 and alpha.shape==utilities.shape
    a=alpha.astype(np.float64);u=utilities.astype(np.float64)
    mean=(a*u).sum(-1)
    # Centering avoids subtraction of two large nearly equal moments.
    v=(a*(u-mean[...,None])**2).sum(-1)
    assert np.isfinite(v).all() and (v>=0).all()
    return np.log1p(v)

def unit(contract):
    a=np.array([[[.5,.5]],[[.5,.5]]],np.float32)
    u=np.array([[[2.,-2.]],[[0.,0.]]],np.float32)
    assert np.array_equal((a*u).sum(-1),np.zeros((2,1),np.float32))
    assert np.allclose(dispersion(a,u).ravel(),[np.log(5),0],rtol=0,atol=1e-12)
    X=np.arange(-20,20,dtype=np.float64)[:,None]/10;y=(np.arange(40)%3==0).astype(np.float64)
    p,fit=fit_predict(X,y,X,.01);assert (p>0).all() and (p<1).all()
    # Fit-only standardization: unrelated future test points cannot change prediction of X.
    q,_=fit_predict(X,y,np.r_[X,[[1000000.]]],.01);assert np.array_equal(p,q[:len(X)])
    torch,Head=original.torch_setup(contract['original_contract'])
    torch.manual_seed(20261118)
    with torch.inference_mode():
        h=Head('quantum').eval().requires_grad_(False)
        detail=h(torch.randn(2,4,1024),torch.zeros(2,4,4),torch.ones(2,4,dtype=torch.bool),details=True)
        t=dispersion(detail['alpha'].numpy(),detail['utilities'].numpy());assert t.shape==(2,40)
    return dict(status='PASS_UNIT',checks=['equal_mean_distinct_dispersion','fixed_logistic_solver',
        'fit_only_standardization','original_quantum_details'],head_training=False,probe_fit_synthetic=True,cos_requests=0)

def metrics(y,q):
    q=np.asarray(q,np.float64);assert q.shape==y.shape and np.isfinite(q).all()
    # Finite numeric scoring only; deployment decision uses unmodified q strictly > .5.
    p=np.clip(q,np.finfo(np.float64).eps,1-np.finfo(np.float64).eps)
    return dict(logloss=float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),
        brier=float(np.mean((q-y)**2)))

def deployment(gold,scores,subset,q):
    prediction=scores.argmax(1);assert (prediction[subset]==0).all()
    positive=scores[subset,1:].argmax(1)+1;fire=q>.5
    changed=prediction.copy();changed[subset]=np.where(fire,positive,0)
    selected_gold=gold[subset]
    return dict(triggered=int(fire.sum()),correct_positive_recoveries=int((fire&(positive==selected_gold)).sum()),
        true_null_false_positives=int((fire&(selected_gold==0)).sum()),
        positive_wrong_label_reassignments=int((fire&(selected_gold>0)&(positive!=selected_gold)).sum()),
        eligible_positive=int((selected_gold>0).sum()),true_null=int((selected_gold==0).sum()))

def run(args,contract,preflight):
    torch,Head=original.torch_setup(contract['original_contract'])
    data,expected=original.pinned_inputs(args,contract['original_contract'])
    data=data['valid']
    metadata=read(args.inputs/'assets/shared/cache/metadata.json')['valid']
    hashes=[hashlib.sha256(('rmea-dispersion-probe-v1:'+x['id_hash']).encode()).hexdigest() for x in metadata]
    assert len(set(hashes))==len(hashes)
    evaluation=np.array([int(h[:8],16)%3==0 for h in hashes]);assert evaluation.any() and (~evaluation).any()
    cells=[];anchors={};started=time.monotonic()
    with torch.inference_mode():
        for seed in contract['seeds']:
            head=Head('quantum').eval().requires_grad_(False)
            head.load_state_dict(torch.load(args.inputs/f'assets/seed{seed}/checkpoints/quantum-selected.pt',
                map_location='cpu',weights_only=True),strict=True)
            # Existing frozen anchor rule checks all four original stored anchors;
            # no new test predictions, test fitting or test-derived selection.
            anchors[str(seed)]=original.anchors(torch,head,args.inputs,seed,'quantum',contract['original_contract'])
            if preflight:continue
            scores=[];variances=[];gold=[]
            for start in range(0,len(data['targets']),128):
                h,roles,mask,y=original.batch(torch,data,start,128)
                detail=head(h,roles,mask,details=True)
                scores.append(detail['logits'].numpy());gold.append(y)
                variances.append(dispersion(detail['alpha'].numpy(),detail['utilities'].numpy()))
            S=np.concatenate(scores);T=np.concatenate(variances);Y=np.concatenate(gold)
            original_counts=original.count_batch(S,Y)
            ref=expected['cells'][f'{seed}:quantum:valid']
            assert original_counts['total']==ref['items'] and original_counts['confusion']==ref['confusion'],'Invalid original validation bridge'
            null=S.argmax(1)==0;fit=null&~evaluation;test=null&evaluation
            assert fit.sum()>=100 and test.sum()>=100
            target=(Y>0).astype(np.float64)
            feature_sets=dict(score_linear=S.astype(np.float64),
                score_quadratic=np.column_stack((S.astype(np.float64),S.astype(np.float64)**2)),
                score_dispersion=np.column_stack((S.astype(np.float64),T)))
            row=dict(seed=seed,split='original_validation_development_only',fit_entries=int(fit.sum()),
                eval_entries=int(test.sum()),eval_positive=int(target[test].sum()),original_confusion_exact=True,probes={})
            for name,F in feature_sets.items():
                q,fitted=fit_predict(F[fit],target[fit],F[test],contract['penalty'])
                row['probes'][name]=dict(metrics=metrics(target[test],q),fit=fitted,
                    deployment=deployment(Y,S,test,q))
            row['logloss_improvement_vs_linear']=row['probes']['score_linear']['metrics']['logloss']-row['probes']['score_dispersion']['metrics']['logloss']
            row['logloss_improvement_vs_quadratic']=row['probes']['score_quadratic']['metrics']['logloss']-row['probes']['score_dispersion']['metrics']['logloss']
            cells.append(row)
            print(json.dumps(dict(seed=seed,validation_bridge=True,eval_entries=row['eval_entries'],
                improvement_vs_linear=row['logloss_improvement_vs_linear'],improvement_vs_quadratic=row['logloss_improvement_vs_quadratic'])),flush=True)
    if preflight:return dict(status='PASS_FROZEN_INPUT_PREFLIGHT',heads=5,anchors=anchors,anchor_count=20,head_training=False,probe_fit=False)
    assert len(cells)==5
    qualified=all(c['logloss_improvement_vs_linear']>0 and c['logloss_improvement_vs_quadratic']>0 for c in cells)
    return dict(schema='qattn.utility-dispersion.development.v1',status='PASS_DEVELOPMENT_PROBE',cells=cells,
        criterion='ALL5_INCREMENTAL_LOGLOSS' if qualified else 'INCREMENTAL_CRITERION_NOT_MET',
        anchors=anchors,head_training=False,probe_fit=True,original_test_evaluation=False,
        elapsed_seconds=time.monotonic()-started,
        limits=['Original heads were selected on the entire original validation set. Hash-split probe evaluation is development evidence, not independent head/method confirmation.',
            'Only restricted fixed linear/quadratic score probes are compared; neither a gain nor a failure establishes conditional information/sufficiency relative to all measurable score functions.',
            'Extra forty dispersion coordinates and forty squared-score coordinates match feature count but not nonlinear function class or effective capacity.',
            'Variance is an ordinary token utility statistic, not quantum-specific uncertainty or no_relation probability. No changed attention method or L5.',
            'Shared examples across seeds are not independent replications; fixed .5 trigger diagnostics are not a calibrated deployment policy.'],
        candidate='NONE',L5='NOT_ACCEPTED')

def main():
    guard();p=argparse.ArgumentParser();p.add_argument('--action',choices=['unit','preflight','diagnose'],required=True)
    for n in ['inputs','plan','expected','preflight-acceptance']:p.add_argument('--'+n,type=Path)
    p.add_argument('--preflight-acceptance-sha256');p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();c=read(HERE/'contract.json')
    for name,digest in c['dependency_sources'].items():assert sha(ROOT/name)==digest
    assert c['original_contract']==read(HERE.parent/'rmea_original_scores/contract.json')
    if args.action=='diagnose':
        assert sha(args.preflight_acceptance)==args.preflight_acceptance_sha256
        a=read(args.preflight_acceptance)
        assert a['status']=='ACCEPTED_ORIGINAL_INPUT_PREFLIGHT_ONLY' and a['contract_sha256']==sha(HERE/'contract.json')
        assert a['task_state']=='complete' and a['quiescent'] and a['returncode']==0
        assert a['owner_binding_exact_restored'] and a['worker_binding_exact_restored']
        assert a['report']['status']=='PASS_FROZEN_INPUT_PREFLIGHT' and a['report']['anchor_count']==20
    result=unit(c) if args.action=='unit' else run(args,c,args.action=='preflight')
    result.update(action=args.action,contract_sha256=sha(HERE/'contract.json'),environment=c['environment'],
        python_execution=dict(optimize=sys.flags.optimize,PYTHONOPTIMIZE_present='PYTHONOPTIMIZE' in os.environ))
    complete(args.output,args.action,result);print(json.dumps(dict(status=result['status'],action=args.action,head_training=False)))
if __name__=='__main__':main()
