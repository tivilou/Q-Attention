"""Read accepted immutable models on their original fit cohort; zero updates."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    p=path.with_suffix('.pending')
    p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')
    p.replace(path)


def main():
    ap=argparse.ArgumentParser()
    for n in ('design','allocation','output','progress'):
        ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--attempt',required=True);args=ap.parse_args()
    cfg=json.loads(args.design.read_bytes());a=json.loads(args.allocation.read_bytes())['allocation']
    assert cfg['kind']=='fixed_checkpoint_forward_no_training'
    assert a['runner_attempt_id']==args.attempt and len(set(a['gpu_uuids']))==2
    os.environ.update(CUDA_VISIBLE_DEVICES=','.join(a['gpu_uuids']),CUBLAS_WORKSPACE_CONFIG=':4096:8')
    signal.alarm(cfg['timeout_seconds']);started=time.monotonic()
    for n,h in cfg['code_hashes'].items():assert digest(Path(__file__).parent/n)==h
    for p,h in cfg['input_hashes'].items():assert digest(p)==h,p
    args.output.mkdir(exist_ok=True);assert not (args.output/'started.json').exists()
    save(args.output/'started.json',dict(attempt=args.attempt,design_sha256=digest(args.design)))
    import numpy as np
    import torch
    sys.path.insert(0,cfg['source'])
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.device_count()==2
    cache=Path(cfg['cache']);cohort=json.loads(Path(cfg['cohort']).read_bytes())
    ids=np.asarray(cohort['fit'],dtype=np.int64);assessment=np.asarray(cohort['assessment'],dtype=np.int64)
    assert len(ids)==8192 and len(assessment)==4096 and not set(ids)&set(assessment)
    data={k:np.load(cache/('train-'+k+'.npy'),mmap_mode='r',allow_pickle=False)
          for k in ('h','roles','offsets','targets')}
    y=np.array(data['targets'][ids]);assert np.bincount(y,minlength=40).tolist()==cohort['supports']['fit']
    factories={}
    for name,path in cfg['factory_files'].items():
        spec=importlib.util.spec_from_file_location('accepted_factory_'+name,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);factories[name]=module
    modes=tuple(cfg['models']);lock=threading.Lock();cursors={k:0 for k in modes};total=len(modes)*(8192+128)

    def batch(indices,device):
        lengths=[int(data['offsets'][i+1]-data['offsets'][i]) for i in indices]
        h=np.zeros((len(indices),max(lengths),1024),np.float32)
        roles=np.zeros(h.shape[:2]+(4,),np.float32);mask=np.zeros(h.shape[:2],bool)
        for j,i in enumerate(indices):
            lo,hi=map(int,data['offsets'][i:i+2]);n=hi-lo
            h[j,:n]=data['h'][lo:hi];roles[j,:n]=data['roles'][lo:hi];mask[j,:n]=True
        return tuple(torch.from_numpy(x).to(device) for x in (h,roles,mask))

    def score(model,inputs):
        return torch.cat([model(*(x[lo:lo+4] for x in inputs)) for lo in range(0,len(inputs[0]),4)])

    def report(mode,count):
        with lock:
            cursors[mode]=count
            save(args.progress,dict(schema='research.progress.v1',attempt=args.attempt,
                phase='fixed_fit_readout_'+mode,current=sum(cursors.values()),total=total,unit='evaluated_example',
                children={k:dict(current=v,total=8192+128) for k,v in cursors.items()},
                observed_at=time.time(),heartbeat_at=time.time()))

    def evaluate(mode,gpu):
        item=cfg['models'][mode];device=torch.device('cuda',gpu)
        state=torch.load(item['checkpoint'],map_location='cpu',weights_only=True)
        model=factories[item['factory']].make_head(mode).float();model.load_state_dict(state,strict=True)
        model.to(device).eval()
        with torch.cuda.device(device),torch.inference_mode():
            z=score(model,batch(assessment[:128],device)).cpu().numpy()
            with np.load(item['diagnostic'],allow_pickle=False) as ref:
                assert np.array_equal(ref['assessment_indices'],assessment)
                expected=ref[mode+'_logits'][:128]
            parity=float(np.max(np.abs(z-expected)));assert parity<=cfg['parity_atol'],(mode,parity)
            assert np.array_equal(z.argmax(-1),expected.argmax(-1))
            report(mode,128);zs=[]
            for lo in range(0,len(ids),128):
                zz=score(model,batch(ids[lo:lo+128],device)).cpu().numpy()
                assert zz.shape==(min(128,len(ids)-lo),40) and np.isfinite(zz).all()
                zs.append(zz);report(mode,128+min(lo+128,len(ids)))
        z=np.concatenate(zs);pred=z.argmax(-1)
        cm=np.bincount(40*y+pred,minlength=1600).reshape(40,40)
        den=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den!=0)
        positive=y>0;forced=z[:,1:].argmax(-1)+1
        pcm=np.bincount(39*(y[positive]-1)+(forced[positive]-1),minlength=39**2).reshape(39,39)
        zz=z.astype(np.float64);zz-=zz.max(-1,keepdims=True)
        assert digest(item['checkpoint'])==cfg['input_hashes'][item['checkpoint']]
        return dict(macro=float(f1.mean()),per_class_f1=f1.tolist(),confusion_matrix=cm.tolist(),
            class_support=cm.sum(1).tolist(),correct=int((pred==y).sum()),
            ce=float((np.log(np.exp(zz).sum(-1))-zz[np.arange(len(y)),y]).mean()),
            positive39_confusion=pcm.tolist(),positive_forced_correct=int((positive&(forced==y)).sum()),
            assessment128_parity_max_error=parity,assessment128_argmax_equal=True,
            checkpoint_sha256=cfg['input_hashes'][item['checkpoint']])

    ready=Queue()
    for mode in modes:ready.put(mode)
    def worker(gpu):
        rows={}
        while True:
            try:mode=ready.get_nowait()
            except Empty:return rows
            rows[mode]=evaluate(mode,gpu)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=[f.result() for f in [pool.submit(worker,i) for i in range(2)]]
    rows={k:v for group in results for k,v in group.items()};assert set(rows)==set(modes)
    summary=dict(status='completed',attempt=args.attempt,profile=cfg['profile'],rows=8192,
        results=rows,seconds=time.monotonic()-started,forward_rows=total,optimizer_updates=0,
        code_hashes=cfg['code_hashes'],input_hashes=cfg['input_hashes'],design_sha256=digest(args.design),
        checkpoint_states_unchanged=True,valid_evaluated=False,test_evaluated=False,cos_writes=0,
        gate='DESCRIPTIVE_DIAGNOSTIC_ONLY',L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED',
        limits='Same selected small8192 models on their original fit examples; assessment128 only reproduces stored outputs. Training/evaluation gaps are descriptive, not a variance/optimization cause or generalization theorem. No model or training changes.')
    save(args.output/'summary.json',summary)
    save(args.output/'RUN_COMPLETE',{'summary.json':digest(args.output/'summary.json')})
    save(args.progress,dict(schema='research.progress.v1',attempt=args.attempt,phase='completed',
        current=total,total=total,unit='evaluated_example',observed_at=time.time(),heartbeat_at=time.time()))
    print(json.dumps(dict(seconds=summary['seconds'],macro={k:v['macro'] for k,v in rows.items()})),flush=True)


if __name__=='__main__':main()
