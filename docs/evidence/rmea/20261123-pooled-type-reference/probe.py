"""Bounded classical sentence-feature diagnostic; no native logits enter its fit."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
from pathlib import Path
import argparse
from collections import defaultdict
import hashlib
import json
import resource
import signal
import sys
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp,softmax


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    p=Path(path);tmp=p.with_suffix('.pending')
    tmp.write_bytes((json.dumps(value,indent=2,allow_nan=False)+'\n').encode());os.replace(tmp,p)


def objective(theta,x,y,penalty):
    w=theta.reshape(x.shape[1]+1,40)
    z=x@w[:-1]+w[-1]
    loss=float(np.mean(logsumexp(z,axis=1)-z[np.arange(len(y)),y]))
    residual=softmax(z,axis=1);residual[np.arange(len(y)),y]-=1;residual/=len(y)
    grad=np.concatenate((x.T@residual,residual.sum(0,keepdims=True)),axis=0)
    return loss+.5*penalty*float(theta@theta),(grad+penalty*w).ravel()


def pool(cache,split,count):
    arrays={k:np.load(cache/(split+'-'+k+'.npy'),mmap_mode='r',allow_pickle=False)
            for k in ('h','roles','offsets','targets')}
    h,roles,offsets=arrays['h'],arrays['roles'],arrays['offsets']
    assert h.ndim==2 and h.shape[1]==1024 and roles.shape==(len(h),4)
    assert offsets.shape==(len(arrays['targets'])+1,) and offsets[0]==0 and offsets[-1]==len(h)
    assert np.issubdtype(offsets.dtype,np.integer) and (np.diff(offsets)>0).all() and count<=len(arrays['targets'])
    x=np.empty((count,3072),np.float64);no_context=0
    for i in range(count):
        left,right=map(int,offsets[i:i+2]);a=np.asarray(h[left:right],np.float64);r=roles[left:right]
        assert np.isfinite(a).all() and np.isfinite(r).all()
        assert np.isin(r[:,2:],(0,1)).all()
        subject=r[:,2].astype(bool);obj=r[:,3].astype(bool);context=~(subject|obj)
        assert subject.any() and obj.any()
        x[i,:1024]=a[subject].mean(0);x[i,1024:2048]=a[obj].mean(0)
        x[i,2048:]=a[context].mean(0) if context.any() else 0
        no_context+=int(not context.any())
    y=np.array(arrays['targets'][:count],copy=True)
    assert y.shape==(count,) and np.issubdtype(y.dtype,np.integer) and ((y>=0)&(y<40)).all()
    assert np.isfinite(x).all()
    return x,y,no_context


def main():
    if sys.flags.optimize:raise RuntimeError('Checked diagnostic requires assertions')
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--design',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--progress',type=Path,required=True);parser.add_argument('--attempt',required=True)
    a=parser.parse_args();cfg=json.loads(a.design.read_bytes())
    assert digest(__file__)==cfg['code_sha256']
    for name,expected in cfg['input_hashes'].items():assert digest(a.root/name)==expected,name
    assert not a.output.exists(),'Preserve old attempts, no automatic scientific retry'
    a.output.mkdir(parents=True);save(a.output/'started.json',dict(attempt=a.attempt,design_sha256=digest(a.design)))
    signal.alarm(cfg['timeout_seconds']);started=time.monotonic()
    def progress(phase,k):
        save(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,phase=phase,current=k,total=5,unit='stage',heartbeat_at=time.time(),observed_at=time.time()))
    sys.path.insert(0,str(a.root/'source'))
    from baselines.retacred.common import metrics
    labels=json.loads((a.root/'inputs/native-baseline/recovery-inputs/labels.json').read_bytes())
    assert isinstance(labels,list) and len(labels)==40 and len(set(labels))==40 and labels[0]=='no_relation'
    manifest=json.loads((a.root/'inputs/native-baseline/discovery/manifest.json').read_bytes())
    cache=a.root/'inputs/native-baseline/discovery/cache'
    assert all(digest(cache/n)==r['sha256'] and (cache/n).stat().st_size==r['bytes'] for n,r in cfg['feature_cache'].items())
    for n,r in cfg['feature_cache'].items():assert manifest['cache'][n]==r
    original=a.root/'control/task-bottleneck-discovery-20261123'
    meta=json.loads((original/'cases.private.json').read_bytes())['type_metadata']
    vm=json.loads((original/'validation-cases.private.json').read_bytes())['metadata']
    assert len(meta)==8192 and len(vm)==19584
    with np.load(a.root/'control/native-ordered-mass-value-20261123/output/diagnostic.private.npz',allow_pickle=False) as z:
        fit=z['calibration'].copy();recorded_y=z['labels'].copy();native_train=z['logits'][~fit].argmax(1)
    assert fit.shape==(8192,) and fit.dtype==np.bool_ and fit.sum()==4192 and (~fit).sum()==4000
    assert recorded_y.shape==(8192,)
    proof=json.loads((a.root/'control/type-context-baseline-20261123/calibration-row-provenance.json').read_bytes())
    rowmeta=json.loads((cache/'metadata.json').read_bytes())
    for split,n,rows in [('train',8192,meta),('valid',19584,vm)]:
        ids=[row['id_hash'] for row in rowmeta[split][:n]]
        assert len(ids)==n and hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==proof[split]['row_id_order_sha256']
        assert hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()==proof[split]['records_sha256']
    progress('pool_first8192_train',0)
    raw,y,empty_train=pool(cache,'train',8192)
    assert np.array_equal(y,recorded_y)
    assert all(isinstance(row,list) and len(row)==3 and row[0]==int(y[i]) and all(isinstance(t,str) and t for t in row[1:]) for i,row in enumerate(meta))
    mean=raw[fit].mean(0)
    centered=raw[fit]-mean
    scale=np.sqrt(np.mean(np.sum(centered.reshape(-1,3,1024)**2,axis=2),axis=0))
    assert scale.shape==(3,) and np.isfinite(scale).all() and (scale>0).all()
    raw=(raw-mean)/np.repeat(scale,1024)
    pairs=sorted({tuple(meta[i][1:]) for i in np.flatnonzero(fit)});assert len(pairs)==25
    pair_index={pair:i for i,pair in enumerate(pairs)}
    def features(x,rows,typed):
        if not typed:return x
        onehot=np.zeros((len(rows),len(pairs)))
        for i,m in enumerate(rows):
            j=pair_index.get(tuple(m[1:]))
            if j is not None:onehot[i,j]=1
        return np.concatenate((x,onehot),axis=1)
    # One fixed finite-difference fixture for this linear-head objective.
    fx=raw[fit][:13,:5];fy=y[fit][:13];theta=np.linspace(-.02,.02,240)
    direction=np.cos(np.arange(240));direction/=np.linalg.norm(direction);eps=1e-5
    f,g=objective(theta,fx,fy,cfg['penalty'])
    fd=(objective(theta+eps*direction,fx,fy,cfg['penalty'])[0]-objective(theta-eps*direction,fx,fy,cfg['penalty'])[0])/(2*eps)
    error=abs(fd-g@direction);assert error<1e-8
    coefficients={};fits={};fit_rows=[meta[i] for i in np.flatnonzero(fit)]
    for k,(mode,typed) in enumerate([('untyped',False),('typed',True)],1):
        x=features(raw[fit],fit_rows,typed);initial=np.zeros((x.shape[1]+1)*40)
        result=minimize(objective,initial,args=(x,y[fit],cfg['penalty']),jac=True,method='L-BFGS-B',
            options=dict(maxiter=cfg['max_iterations'],gtol=cfg['gradient_inf']/10,ftol=1e-14,maxls=40))
        f,g=objective(result.x,x,y[fit],cfg['penalty']);grad=float(np.abs(g).max())
        assert result.success and np.isfinite(result.x).all() and np.isfinite(f) and np.isfinite(g).all() and grad<=cfg['gradient_inf'],(mode,result.message,grad)
        coefficients[mode]=result.x.reshape(x.shape[1]+1,40)
        fits[mode]=dict(parameters=len(initial),iterations=int(result.nit),objective=f,gradient_inf=grad,optimizer_success=True)
        progress('fit_complete_'+mode,k)
    np.savez(a.output/'coefficients.private.npz',**coefficients,feature_mean=mean,feature_scale=scale)
    sealed=digest(a.output/'coefficients.private.npz');save(a.output/'fits.json',dict(fits=fits,coefficient_sha256=sealed))
    def score(x,rows,mode):
        x=features(x,rows,mode=='typed');w=coefficients[mode];z=x@w[:-1]+w[-1]
        assert z.shape==(len(rows),40) and np.isfinite(z).all()
        return z
    def measure(y,z):
        pred=z.argmax(1);ce=float(np.mean(logsumexp(z,axis=1)-z[np.arange(len(y)),y]))
        m=metrics(y.tolist(),pred.tolist(),labels,ce)
        return dict(items=len(y),correct=int((pred==y).sum()),ce=ce,positive39_macro_f1=float(np.mean([c['f1'] for c in m['per_class'][1:]])),
            null_false_positive=int(((y==0)&(pred>0)).sum()),positive_false_negative=int(((y>0)&(pred==0)).sum()),positive_wrong_relation=int(((y>0)&(pred>0)&(pred!=y)).sum()),per_class=m['per_class'],confusion_matrix=m['confusion_matrix'])
    predictions={};assessment={};arows=[meta[i] for i in np.flatnonzero(~fit)]
    for mode in ('untyped','typed'):
        z=score(raw[~fit],arows,mode);assessment[mode]=measure(y[~fit],z);predictions['assessment_'+mode]=z.argmax(1)
    del raw,centered,x
    progress('pool_all_development_valid_after_both_fits_sealed',3)
    vx,vy,empty_valid=pool(cache,'valid',19584)
    assert all(isinstance(row,list) and len(row)==3 and row[0]==int(vy[i]) and all(isinstance(t,str) and t for t in row[1:]) for i,row in enumerate(vm))
    vx=(vx-mean)/np.repeat(scale,1024)
    with np.load(a.root/'control/native-context-dephasing-20261123/output/predictions.private.npz',allow_pickle=False) as archive:
        vz=archive['logits'][0].copy();assert np.array_equal(archive['labels'],vy)
    assert vz.shape==(19584,40) and np.isfinite(vz).all()
    support=defaultdict(set);seen=set()
    for i in np.flatnonzero(fit):support[tuple(meta[i][1:])].add(int(y[i]));seen.add(int(y[i]))
    allow=np.ones(vz.shape,bool)
    for i,m in enumerate(vm):
        pair=tuple(m[1:])
        if pair in support:
            allow[i]=False;allow[i,0]=True
            for j in support[pair]|(set(range(40))-seen):allow[i,j]=True
    hard=np.where(allow,vz,-np.inf).argmax(1);native=vz.argmax(1)
    frozen=json.loads((original/'validation-summary.json').read_bytes())
    assert labels==['no_relation']+[row['label'] for row in frozen['per_class']]
    fm=metrics(vy.tolist(),hard.tolist(),labels,None)
    frozen_macro=float(np.mean([m['f1'] for m in fm['per_class'][1:]]))
    assert abs(frozen_macro-frozen['projected']['positive39_macro_f1'])<1e-14 and int((hard==vy).sum())==frozen['projected']['correct']
    validation={};paired={};gates={}
    for mode in ('untyped','typed'):
        z=score(vx,vm,mode);pred=z.argmax(1);validation[mode]=measure(vy,z);predictions['valid_'+mode]=pred
        paired[mode]=dict(repaired=int(((hard!=vy)&(pred==vy)).sum()),damaged=int(((hard==vy)&(pred!=vy)).sum()),
            repaired_positive_false_negatives=int(((vy>0)&(hard==0)&(pred==vy)).sum()),new_positive_false_negatives=int(((vy>0)&(hard!=0)&(pred==0)).sum()))
        gates[mode]=dict(macro_beats_frozen_rule=validation[mode]['positive39_macro_f1']>frozen_macro,net_correct_vs_frozen_rule=paired[mode]['repaired']>paired[mode]['damaged'])
    gates['typed_improves_untyped']=validation['typed']['positive39_macro_f1']>validation['untyped']['positive39_macro_f1']
    np.savez(a.output/'predictions.private.npz',**predictions)
    report=dict(profile=cfg['profile'],attempt=a.attempt,fit_rows=4192,assessment_rows=4000,development_valid_rows=19584,
        features='subject mean, object mean, nonentity-context mean; each centered and block-RMS scaled on fit only; typed adds fit-only type-pair onehot, unseen pair zeros',
        parameters=fits,assessment=assessment,development_valid=validation,paired_vs_frozen_type_rule=paired,gates=gates,
        references=dict(native_macro=frozen['native']['positive39_macro_f1'],frozen_type_rule_macro=frozen_macro,frozen_type_rule_correct=int((hard==vy).sum())),
        coefficient_sha256=sealed,gradient_fixture_error=float(error),empty_context_rows=dict(train=empty_train,valid=empty_valid),
        execution=dict(seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,deep_model_forwards=0,gpu_compute=0,cos_writes=0),
        scope='Bounded exploratory classical head fit on4192exposed TRAIN rows; no native logits in training. Reused VALID and different fit budget from historical Q-RMEA prohibit formal utility/attribution/quantum necessity/L5 claims.',
        stop='No C/penalty/features/normalization/split/iterations/seed rescue. A negative bounded fit does not reject full-data trained classical heads.')
    save(a.output/'summary.json',report);(a.output/'RUN_COMPLETE').write_bytes(b'completed\n');progress('completed',5)
    print(json.dumps(dict(gates=gates,seconds=report['execution']['seconds'],valid_macro={m:validation[m]['positive39_macro_f1'] for m in validation})))


if __name__=='__main__':main()
