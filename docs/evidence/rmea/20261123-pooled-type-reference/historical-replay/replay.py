"""Read-only common-cache replay of the five already completed historical heads."""
from pathlib import Path
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import sys
import threading
import time
import numpy as np


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def save(p,v):
    t=p.with_suffix('.pending');t.write_bytes((json.dumps(v,indent=2,allow_nan=False)+'\n').encode());os.replace(t,p)


def main():
    if sys.flags.optimize:raise RuntimeError('Checked replay requires assertions')
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--design',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--progress',type=Path,required=True);p.add_argument('--attempt',required=True)
    a=p.parse_args();cfg=json.loads(a.design.read_bytes());here=a.design.parent
    assert digest(__file__)==cfg['code_sha256']
    for n,h in cfg['input_hashes'].items():assert digest(a.root/n)==h,n
    assert not a.output.exists();a.output.mkdir(parents=True)
    save(a.output/'started.json',dict(attempt=a.attempt,design_sha256=digest(a.design)))
    start=time.monotonic();sys.path.insert(0,str(a.root/'source'))
    import torch
    from baselines.retacred.rmea import RMEAHead
    from baselines.retacred.common import metrics
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.device_count()==2
    provenance=json.loads((here/'model-provenance.json').read_bytes());labels=provenance['labels'];assert len(labels)==40 and labels[0]=='no_relation'
    cache=a.root/'inputs/native-baseline/discovery/cache';current=json.loads((a.root/'inputs/native-baseline/discovery/manifest.json').read_bytes())
    assert current['config']['encoder_revision']==provenance['config']['encoder_revision']
    assert digest(cache/'metadata.json')==provenance['metadata_sha256']
    data={k:np.load(cache/('valid-'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('h','roles','offsets','targets')}
    assert len(data['targets'])==19584 and data['offsets'].shape==(19585,)
    assert data['offsets'][0]==0 and data['offsets'][-1]==len(data['h'])==len(data['roles'])
    for n,rec in provenance['original_valid_cache'].items():
        if n!='cache/valid-h.npy':assert digest(a.root/'inputs/native-baseline/discovery'/n)==rec['sha256']
    fixture=np.load(here/'valid-h-first2.npy',allow_pickle=False);new=np.asarray(data['h'][:len(fixture)])
    assert fixture.shape==new.shape and fixture.ndim==2 and fixture.shape[1]==1024
    difference=fixture.astype(np.float64)-new.astype(np.float64)
    drift=dict(old_full_h_sha256=provenance['original_valid_cache']['cache/valid-h.npy']['sha256'],current_full_h_sha256=current['cache']['valid-h.npy']['sha256'],
        first2_fixture_elements=int(fixture.size),changed_elements=int(np.count_nonzero(difference)),max_abs=float(np.abs(difference).max()),rms=float(np.sqrt(np.mean(difference**2))),
        scope='First2 cache rows only; no inference that the complete old cache is identical. All methods below receive the same current rebuilt cache.')
    old=a.root/'control/task-bottleneck-discovery-20261123'
    tm=json.loads((old/'cases.private.json').read_bytes())['type_metadata'];vm=json.loads((old/'validation-cases.private.json').read_bytes())['metadata']
    with np.load(a.root/'control/native-ordered-mass-value-20261123/output/diagnostic.private.npz',allow_pickle=False) as z:fit=z['calibration'].copy();train_y=z['labels'].copy()
    assert fit.shape==(8192,) and fit.dtype==np.bool_ and fit.sum()==4192 and len(tm)==8192 and len(vm)==19584
    support=defaultdict(set);seen=set()
    for i in np.flatnonzero(fit):
        g,st,ot=tm[i];assert g==int(train_y[i]);support[(st,ot)].add(g);seen.add(g)
    support_sha=hashlib.sha256(json.dumps(sorted((list(pair),sorted(ids)) for pair,ids in support.items()),separators=(',',':')).encode()).hexdigest()
    assert support_sha==cfg['frozen_support_sha256']
    y=np.array(data['targets']);allow=np.ones((len(y),40),bool)
    for i,(g,st,ot) in enumerate(vm):
        assert g==int(y[i]);pair=(st,ot)
        if pair in support:
            allow[i]=False;allow[i,0]=True
            for k in support[pair]|(set(range(40))-seen):allow[i,k]=True
    assert labels==json.loads((a.root/'inputs/native-baseline/recovery-inputs/labels.json').read_bytes())
    cursor=[0,0];lock=threading.Lock()
    def progress(lane):
        with lock:save(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,phase='historical_five_head_common_cache_replay',current=sum(cursor),total=5*len(y),unit='example',gpu_lanes=cursor.copy(),heartbeat_at=time.time(),observed_at=time.time()))
    progress(0)
    def worker(lane,modes):
        torch.cuda.set_device(lane);device=torch.device('cuda',lane);rows={}
        with torch.inference_mode():
            for mode in modes:
                rec=provenance['models'][mode];checkpoint=here/rec['file'];assert digest(checkpoint)==rec['sha256']
                state=torch.load(checkpoint,map_location='cpu',weights_only=True)
                assert all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in state.values())
                head=RMEAHead(mode).float().to(device);head.load_state_dict(state,strict=True);head.eval()
                logits=np.empty((len(y),40),np.float32)
                for lo in range(0,len(y),32):
                    ids=range(lo,min(lo+32,len(y)));lengths=[int(data['offsets'][i+1]-data['offsets'][i]) for i in ids]
                    h=np.zeros((len(lengths),max(lengths),1024),np.float32);r=np.zeros(h.shape[:2]+(4,),np.float32);mask=np.zeros(h.shape[:2],bool)
                    for j,i in enumerate(ids):
                        left,right=map(int,data['offsets'][i:i+2]);size=right-left
                        h[j,:size]=data['h'][left:right];r[j,:size]=data['roles'][left:right];mask[j,:size]=True
                    assert np.isfinite(h).all() and np.isfinite(r).all()
                    inputs=tuple(torch.from_numpy(x).to(device) for x in (h,r,mask));z=head(*inputs)[0].cpu().numpy()
                    assert z.shape==(len(lengths),40) and np.isfinite(z).all();logits[lo:lo+len(lengths)]=z
                    cursor[lane]+=len(lengths)
                    if lo%512==0:progress(lane)
                np.save(a.output/(mode+'-logits.private.npy'),logits,allow_pickle=False)
                pred=logits.argmax(1);typed=np.where(allow,logits,-np.inf).argmax(1);records={}
                for name,pp in [('raw',pred),('typed',typed)]:
                    m=metrics(y.tolist(),pp.tolist(),labels,None)
                    records[name]=dict(positive39_macro_f1=float(np.mean([q['f1'] for q in m['per_class'][1:]])),correct=int((pp==y).sum()),
                        null_false_positive=int(((y==0)&(pp>0)).sum()),positive_false_negative=int(((y>0)&(pp==0)).sum()),positive_wrong_relation=int(((y>0)&(pp>0)&(pp!=y)).sum()),confusion_matrix=m['confusion_matrix'],per_class=m['per_class'])
                historical=rec['selected_valid'];old_macro=float(np.mean([q['f1'] for q in historical['per_class'][1:]]))
                rows[mode]=dict(**records,repair=int(((pred!=y)&(typed==y)).sum()),damage=int(((pred==y)&(typed!=y)).sum()),historical_valid_macro39=old_macro,
                    current_cache_raw_minus_historical=records['raw']['positive39_macro_f1']-old_macro,original_selected_epoch=rec['selected_epoch'],checkpoint_sha256=rec['sha256'])
                del head,state
        progress(lane);return rows
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,0,['uniform','relation_softmax','real_overlap']),pool.submit(worker,1,['quantum','dephased'])]
        results={}
        for job in jobs:results.update(job.result())
    contrasts={mode:{scope:results['quantum'][scope]['positive39_macro_f1']-results[mode][scope]['positive39_macro_f1'] for scope in ('raw','typed')} for mode in ('uniform','relation_softmax','real_overlap','dephased')}
    report=dict(cohort=provenance['cohort'],attempt=a.attempt,source_sha256=digest(here/'model-provenance.json'),frozen_support_sha256=support_sha,rows=len(y),
        feature_cache_drift=drift,results=results,quantum_minus_each_control=contrasts,seconds=time.monotonic()-start,optimizer_updates=0,test_rows_evaluated=0,cos_writes=0,
        scope='Posthoc same-current-cache diagnostic of fixed old full-data models; original h bytes differ and original selected epochs unchanged. Types add input equally to all5. No new formal training, independent confirmation, quantum necessity or L5 acceptance. Do not replace historical reports.')
    save(a.output/'summary.json',report);(a.output/'RUN_COMPLETE').write_bytes(b'completed\n')
    print(json.dumps(dict(seconds=report['seconds'],macro={m:{s:r[s]['positive39_macro_f1'] for s in ('raw','typed')} for m,r in results.items()},quantum_minus_each_control=contrasts)))


if __name__=='__main__':main()
