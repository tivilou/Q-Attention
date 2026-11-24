"""Frozen complete TRAIN diagnostic; no fit, heldout, transport or checkpoint writes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse, hashlib, json, math, sys, threading, time
import numpy as np
from endpoints import bounds, numpy_reference, fixtures


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def save(p,v):
    tmp=p.with_suffix('.pending');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def metrics(y,z):
    pred=z.argmax(-1);cm=np.bincount(40*y+pred,minlength=1600).reshape(40,40)
    den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den!=0)
    z=z.astype('float64');z-=z.max(-1,keepdims=True)
    return dict(confusion_matrix=cm.tolist(),all40_macro_f1=float(f.mean()),
        ce=float((np.log(np.exp(z).sum(-1))-z[np.arange(len(y)),y]).mean()),
        correct=int(np.trace(cm)),positive_false_negative=int(((y>0)&(pred==0)).sum()),
        null_false_positive=int(((y==0)&(pred>0)).sum()),
        positive_wrong_relation=int(((y>0)&(pred>0)&(pred!=y)).sum()))


def main():
    ap=argparse.ArgumentParser()
    for n in ('root','design','output','progress'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--attempt',required=True);a=ap.parse_args();cfg=json.loads(a.design.read_bytes())
    for n,h in cfg['code_hashes'].items():assert digest(Path(__file__).parent/n)==h,n
    source=a.root/cfg['source'];completed=a.root/cfg['completed'];cache=completed/'cache'
    assert digest(completed/'manifest.json')==cfg['manifest_sha256']
    for n,h in cfg['source_hashes'].items():assert digest(source/n)==h,n
    for n,v in cfg['train_cache'].items():assert digest(cache/n)==v['sha256'],n
    assert digest(a.root/cfg['labels'])==cfg['labels_sha256']
    labels=json.loads((a.root/cfg['labels']).read_bytes());assert len(labels)==40
    sys.path.insert(0,str(source));import torch
    from torch.nn import functional as F
    from baselines.retacred.rmea import circuit
    from baselines.retacred.rmea_matched_routers import Head
    from baselines.retacred.rmea_token_transport_study import state_identity
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;assert torch.cuda.device_count()==2
    assert not a.output.exists();a.output.mkdir(parents=True)
    fixture_results=fixtures();save(a.output/'fixtures.json',fixture_results)
    data={k:np.load(cache/('train-'+k+'.npy'),mmap_mode='r',allow_pickle=False)
          for k in ('h','roles','offsets','targets')}
    y=np.array(data['targets']);assert y.shape==(58465,) and y.min()==0 and y.max()==39
    assert data['offsets'].shape==(58466,) and data['offsets'][0]==0
    assert data['offsets'][-1]==len(data['h'])==len(data['roles'])
    started=time.monotonic();cursor=[0,0];lock=threading.Lock()
    def progress():
        with lock:save(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,
            phase='fixed_native_routing_headroom',current=sum(cursor),total=2*len(y),unit='example',
            gpu_lanes=cursor.copy(),heartbeat_at=time.time(),observed_at=time.time()))
    progress()
    def worker(lane,mode):
        torch.cuda.set_device(lane);device=torch.device('cuda',lane);begin=time.monotonic()
        pin=cfg['selected'][mode];path=completed/pin['path'];assert digest(path)==pin['sha256']
        state=torch.load(path,map_location='cpu',weights_only=True);assert state_identity(state)==pin['state_sha256']
        model=Head(mode).float().to(device);model.load_state_dict(state,strict=True);model.eval();head=model.head
        logits=np.empty((len(y),40),np.float32);category=np.empty(len(y),np.int8)
        margins=np.empty((len(y),3),np.float64);diag={};numpy_error=0.;readout_error=0.;native_outside=0.
        with torch.inference_mode():
            for lo in range(0,len(y),128):
                assert time.monotonic()-begin<900, 'Frozen per-model time budget exhausted'
                hi=min(lo+128,len(y));lengths=np.diff(data['offsets'][lo:hi+1])
                h=np.zeros((hi-lo,int(lengths.max()),1024),np.float32)
                roles=np.zeros(h.shape[:2]+(4,),np.float32);mask=np.zeros(h.shape[:2],bool)
                for j,i in enumerate(range(lo,hi)):
                    left,right=map(int,data['offsets'][i:i+2]);size=right-left
                    h[j,:size]=data['h'][left:right];roles[j,:size]=data['roles'][left:right];mask[j,:size]=True
                ht,rt,mt=(torch.from_numpy(v).to(device) for v in (h,roles,mask))
                z=model(ht,rt,mt);logits[lo:hi]=z.cpu().numpy()
                u=torch.einsum('btd,rd->brt',head.value(ht),head.decoder)
                if mode=='quantum':
                    k=circuit((math.pi*torch.tanh(head.prepare(torch.cat((ht,rt),-1)))).reshape(ht.shape[:2]+(2,3,2)))
                    q=circuit(math.pi*torch.tanh(head.effect))
                else:
                    k=F.normalize(head.key(torch.cat((ht,rt),-1)),dim=-1);q=F.normalize(head.query,dim=-1)
                result=bounds(k,u,head.bias,mt,q)
                readout_error=max(readout_error,float((z.double()-result['query_logits']).abs().max()))
                native_outside=max(native_outside,float(torch.maximum(result['query_low']-z.double(),z.double()-result['query_high']).clamp_min(0).max()))
                for name,value in result['diagnostic'].items():
                    diag[name]=min(diag.get(name,math.inf),value) if name=='minimum_B_eigenvalue' else max(diag.get(name,0),value)
                if lo==0:
                    for j in range(3):
                        nl,nh=numpy_reference(k[j,mt[j]].cpu().numpy(),u[j][:,mt[j]].cpu().numpy(),head.bias.cpu().numpy())
                        numpy_error=max(numpy_error,float(np.abs(nl-result['query_low'][j].cpu().numpy()).max()),float(np.abs(nh-result['query_high'][j].cpu().numpy()).max()))
                gold=torch.from_numpy(y[lo:hi]).to(device);idx=torch.arange(hi-lo,device=device)
                gaps=[]
                for prefix in ('simplex','query'):
                    lower=result[prefix+'_low'].clone();lower[idx,gold]=-torch.inf
                    gaps.append((result[prefix+'_high'][idx,gold]-lower.max(-1).values).cpu().numpy())
                gs,gq=gaps;tol=result['tolerance'].cpu().numpy()
                assert np.all(gq<=gs+tol), 'Numerically inconsistent nested bounds'
                c=np.full(hi-lo,3,np.int8)
                c[gs < -tol]=0;c[(gs>tol)&(gq < -tol)]=1;c[(gs>tol)&(gq>tol)]=2
                category[lo:hi]=c;margins[lo:hi]=np.stack((gs,gq,tol),-1)
                cursor[lane]=hi
                if lo%1024==0:progress()
        assert time.monotonic()-begin<900,'Frozen per-model time budget exhausted'
        assert numpy_error<1e-6 and readout_error<1e-4,(numpy_error,readout_error)
        assert diag['maximum_generalized_eigenpair_relative_residual']<1e-7
        assert diag['maximum_original_query_outside_bounds']<1e-5
        assert native_outside<1e-5
        assert diag['maximum_query_outside_simplex']<1e-5
        observed=metrics(y,logits);reference=cfg['prior_train_metrics'][mode]
        assert observed['confusion_matrix']==reference['confusion_matrix'],'Native TRAIN confusion matrix differs'
        parity={k:abs(observed[k]-reference[k]) for k in ('ce','all40_macro_f1')}
        assert max(parity.values())<1e-5,parity
        # Reconstructible private arrays remain only on the allocated GPU host.
        np.savez(a.output/(mode+'-diagnostic.private.npz'),category=category,margins=margins,logits=logits)
        wrong=logits.argmax(-1)!=y
        def counts(rows):return np.bincount(category[rows],minlength=4).tolist()
        report=dict(metrics=observed,metric_parity=parity,numerics=diag,readout_max_error=readout_error,
            native_model_logit_outside_bounds=native_outside,exact_prior_confusion_parity=True,
            independent_numpy_max_error=numpy_error,seconds=time.monotonic()-begin,
            categories_all=counts(np.ones(len(y),bool)),categories_original_errors=counts(wrong),
            per_class=[dict(label=labels[i],support=int((y==i).sum()),errors=int(((y==i)&wrong).sum()),
                categories_all=counts(y==i),categories_original_errors=counts((y==i)&wrong)) for i in range(40)])
        progress();elapsed=time.monotonic()-begin
        assert elapsed<900,'Frozen per-model time budget exhausted'
        report['seconds']=elapsed
        return mode,report
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,0,'quantum'),pool.submit(worker,1,'real12')]
        results=dict(j.result() for j in jobs)
    report=dict(profile=cfg['profile'],rows=len(y),labels=labels,results=results,fixtures=fixture_results,
        category_order=['simplex_numerically_unreachable','simplex_reachable_query_numerically_unreachable','free_query_reachable','unresolved'],
        seconds=time.monotonic()-started,optimizer_updates=0,valid_rows_evaluated=0,test_rows_evaluated=0,cos_writes=0,
        claim='Numerical fixed-state TRAIN diagnosis; no rigorous infeasibility certificate, deployable oracle, generalization or new theory/utility claim.')
    save(a.output/'summary.json',report);save(a.output/'RUN_COMPLETE',dict(summary_sha256=digest(a.output/'summary.json'),attempt=a.attempt))
    progress();print(json.dumps(dict(rows=len(y),seconds=report['seconds'],results={m:v['categories_original_errors'] for m,v in results.items()})))


if __name__=='__main__':main()
