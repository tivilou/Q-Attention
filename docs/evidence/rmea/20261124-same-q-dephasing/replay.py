"""R0 replay of completed five-way selected states on TRAIN, with zero updates."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,hashlib,json,threading,time
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(2**20),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    p=path.with_suffix('.pending')
    p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');p.replace(path)


def metrics(y,z):
    prediction=z.argmax(1)
    cm=np.bincount(40*y+prediction,minlength=1600).reshape(40,40)
    denominator=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*np.diag(cm),denominator,out=np.zeros(40),where=denominator!=0)
    shifted=z.astype(np.float64)-z.max(1,keepdims=True)
    return dict(confusion_matrix=cm.tolist(),all40_macro_f1=float(f1.mean()),
        positive39_macro_f1=float(f1[1:].mean()),correct=int(np.trace(cm)),
        null_false_positive=int(((y==0)&(prediction>0)).sum()),
        positive_false_negative=int(((y>0)&(prediction==0)).sum()),
        positive_wrong_relation=int(((y>0)&(prediction>0)&(prediction!=y)).sum()),
        ce=float(np.mean(np.log(np.exp(shifted).sum(1))-shifted[np.arange(len(y)),y])))


def main():
    ap=argparse.ArgumentParser()
    for name in ('root','design','output','progress'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--attempt',required=True);a=ap.parse_args()
    cfg=json.loads(a.design.read_bytes());assert digest(__file__)==cfg['code_sha256']
    completed=a.root/cfg['completed'];source=a.root/cfg['source'];cache=a.root/cfg['cache']
    assert digest(completed/'manifest.json')==cfg['manifest_sha256']
    manifest=json.loads((completed/'manifest.json').read_bytes())
    for name,h in cfg['source_hashes'].items():assert digest(source/name)==h,name
    for name,pin in cfg['train_cache'].items():assert digest(cache/name)==pin['sha256'],name
    assert digest(a.root/cfg['labels'])==cfg['labels_sha256']
    labels=json.loads((a.root/cfg['labels']).read_bytes());assert len(labels)==40 and labels[0]=='no_relation'
    import sys
    sys.path.insert(0,str(source))
    import torch
    from baselines.retacred.rmea_matched_routers import Head
    from baselines.retacred.rmea_token_transport_study import state_identity
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    assert not torch.cuda.is_available()
    data={k:np.load(cache/('train-'+k+'.npy'),mmap_mode='r',allow_pickle=False)
          for k in ('h','roles','offsets','targets')}
    y=np.array(data['targets']);assert y.shape==(58465,) and y.min()==0 and y.max()==39
    assert data['offsets'].shape==(58466,) and data['offsets'][0]==0
    assert data['offsets'][-1]==len(data['h'])==len(data['roles'])
    assert not a.output.exists();a.output.mkdir(parents=True)
    started=time.monotonic();cursor=[0];lock=threading.Lock()
    def progress():
        with lock:save(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,
            phase='completed_five_heads_train_replay',current=sum(cursor),total=5*len(y),unit='example',
            gpu_lanes=cursor.copy(),heartbeat_at=time.time(),observed_at=time.time()))
    progress()
    def worker(lane,modes):
        device=torch.device('cpu');results={}
        with torch.inference_mode():
            for mode in modes:
                pin=cfg['selected'][mode];path=completed/pin['path']
                assert digest(path)==pin['sha256']
                state=torch.load(path,map_location='cpu',weights_only=True)
                assert state_identity(state)==pin['state_sha256']
                model=Head(mode).float().to(device);model.load_state_dict(state,strict=True);model.eval()
                logits=np.empty((len(y),40),dtype=np.float32)
                for lo in range(0,len(y),32):
                    hi=min(lo+32,len(y));lengths=np.diff(data['offsets'][lo:hi+1])
                    h=np.zeros((hi-lo,int(lengths.max()),1024),np.float32)
                    roles=np.zeros(h.shape[:2]+(4,),np.float32);mask=np.zeros(h.shape[:2],bool)
                    for j,i in enumerate(range(lo,hi)):
                        left,right=map(int,data['offsets'][i:i+2]);size=right-left
                        h[j,:size]=data['h'][left:right];roles[j,:size]=data['roles'][left:right];mask[j,:size]=True
                    values=model(*(torch.from_numpy(v).to(device) for v in (h,roles,mask))).cpu().numpy()
                    assert values.shape==(hi-lo,40) and np.isfinite(values).all()
                    logits[lo:hi]=values;cursor[lane]+=hi-lo
                    if lo%1024==0:progress()
                path=a.output/(mode+'-logits.private.npy');np.save(path,logits,allow_pickle=False)
                results[mode]=dict(**metrics(y,logits),selected_checkpoint=pin,
                    logits_sha256=digest(path))
                del model,state,logits
        progress();return results
    results={}
    with ThreadPoolExecutor(max_workers=1) as pool:
        jobs=[pool.submit(worker,0,cfg['conditions'])]
        for job in jobs:results.update(job.result())
    native=json.loads((a.root/cfg['native_report']).read_bytes())
    q=np.load(a.root/cfg['native_logits'],mmap_mode='r',allow_pickle=False).argmax(1)
    assert digest(a.root/cfg['native_logits'])==native['results']['quantum']['logits_sha256']
    p=np.load(a.output/'dephased-logits.private.npy',mmap_mode='r',allow_pickle=False).argmax(1)
    masks=dict(coherence_repairs=(q==y)&(p!=y),coherence_damages=(q!=y)&(p==y),
        both_wrong=(q!=y)&(p!=y),prediction_changes=(q!=p))
    paired={}
    for name,m in masks.items():
        paired[name]=dict(rows=int(m.sum()),per_gold_class=np.bincount(y[m],minlength=40).tolist(),
            quantum_positive_to_null=int((m&(y>0)&(q==0)).sum()),
            quantum_null_to_positive=int((m&(y==0)&(q>0)).sum()),
            quantum_positive_wrong_relation=int((m&(y>0)&(q>0)&(q!=y)).sum()))
    report=dict(profile=cfg['profile'],rows=len(y),results=results,native_quantum_metrics=native['results']['quantum'],
        paired=paired,seconds=time.monotonic()-started,optimizer_updates=0,valid_rows_evaluated=0,test_rows_evaluated=0,cos_writes=0,
        claim='Paired internal full-dephasing intervention at identical completed quantum-selected weights on all TRAIN; not retrained-ablation utility, generalization causality, new method or qualification rescue.')
    save(a.output/'summary.json',report)
    save(a.output/'RUN_COMPLETE',dict(summary_sha256=digest(a.output/'summary.json'),attempt=a.attempt))
    progress();print(json.dumps(dict(rows=len(y),seconds=report['seconds'],optimizer_updates=0)))


if __name__=='__main__':main()
