"""Selective context dephasing in a frozen native head, development VALID only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,signal,sys,threading,time
from pathlib import Path
import numpy as np


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(p,v):
    q=p.with_suffix('.pending');q.write_bytes((json.dumps(v,indent=2,allow_nan=False)+'\n').encode());q.replace(p)


def describe(z,y,metric):
    pred=z.argmax(-1);shift=z-z.max(-1,keepdims=True)
    ce=np.log(np.exp(shift).sum(-1))-shift[np.arange(len(y)),y]
    m=metric(y.tolist(),pred.tolist(),['no_relation']+[str(i) for i in range(1,40)],float(ce.mean()))
    return dict(ce=float(ce.mean()),positive_macro_f1=float(np.mean([r['f1'] for r in m['per_class'][1:]])),
        accuracy=m['accuracy'],micro_f1=m['micro_f1'],confusion_matrix=m['confusion_matrix'],
        false_negative=int(((y!=0)&(pred==0)).sum()),false_positive=int(((y==0)&(pred!=0)).sum()))


def decision(rows,tol):
    candidate=rows['outer'];gates={}
    for baseline in ('native','bridge','context'):
        gates['macro_beats_'+baseline]=candidate['positive_macro_f1']>rows[baseline]['positive_macro_f1']+tol
        gates['ce_beats_'+baseline]=candidate['ce']<rows[baseline]['ce']-tol
    return gates,'PASS_CONTEXT_INTERVENTION_ONLY' if all(gates.values()) else 'FAIL_CLOSE_CONTEXT_PROFILE'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','inputs','reference','checkpoint','design','output','progress','allocation'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--attempt',required=True);a=ap.parse_args();cfg=json.loads(a.design.read_bytes())
    alloc=json.loads(a.allocation.read_bytes())['allocation']
    assert alloc['runner_attempt_id']==a.attempt and len(set(alloc['gpu_uuids']))==2
    for n,h in cfg['code_hashes'].items():assert digest(Path(__file__).parent/n)==h
    for n,h in cfg['source_hashes'].items():assert digest(a.source/n)==h
    assert digest(a.checkpoint)==cfg['checkpoint_sha256']
    assert digest(a.inputs/'discovery/manifest.json')==cfg['native_manifest_sha256']
    manifest=json.loads((a.inputs/'discovery/manifest.json').read_bytes());cache=a.inputs/'discovery/cache'
    for key in ('h','roles','offsets','targets'):
        name='valid-'+key+'.npy';assert digest(cache/name)==manifest['cache'][name]['sha256']
    word_path=a.reference/'valid.private.json';assert digest(word_path)==cfg['valid_word_sha256']
    rows=json.loads(word_path.read_bytes());assert len(rows)==cfg['valid_count']
    sys.path.insert(0,str(Path(__file__).parent));from qualify import regions
    masks=[];eligible=[]
    for row in rows:
        bridge,outer,e=regions(len(row['tokens']),row['subject'],row['object'])
        if not e:bridge[:]=False;outer[:]=False
        masks.append((bridge,outer));eligible.append(e)
    eligible=np.asarray(eligible,bool)
    a.output.mkdir(exist_ok=True);assert not (a.output/'started.json').exists()
    write(a.output/'started.json',dict(attempt=a.attempt,design_sha256=digest(a.design)))
    signal.alarm(cfg['timeout_seconds']);started=time.time()
    import torch
    sys.path.insert(0,str(a.source));from baselines.retacred.rmea import RMEAHead,EPS
    from baselines.retacred.common import metrics
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert torch.cuda.device_count()==2 and EPS==1e-6
    initial=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    data={n:np.load(cache/('valid-'+n+'.npy'),mmap_mode='r',allow_pickle=False) for n in ('h','roles','offsets','targets')}
    y=np.asarray(data['targets']);n=len(y);assert n==cfg['valid_count'] and set(y)==set(range(40))
    assert data['offsets'].shape==(n+1,) and data['offsets'][-1]==len(data['h'])==len(data['roles'])
    for i,row in enumerate(rows):
        lo,hi=map(int,data['offsets'][i:i+2]);assert hi-lo==len(row['tokens'])
        for span,col in ((row['subject'],2),(row['object'],3)):
            expected=np.zeros(hi-lo,bool);expected[slice(*span)]=True
            assert np.array_equal(expected,np.asarray(data['roles'][lo:hi,col],bool))
    logits=np.empty((4,n,40),np.float64);identity_terms=np.empty((4,n,40),np.float64)
    lock=threading.Lock();cursors=[0,0];modes=cfg['modes'];assert modes==['native','bridge','outer','context']

    def progress(lane,count):
        with lock:
            cursors[lane]=count
            write(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,phase='context_dephasing_full_valid',
                current=sum(cursors),total=n,unit='example',observed_at=time.time(),heartbeat_at=time.time(),gpu_lanes=cursors.copy(),modes=modes))

    def worker(lane,ids):
        device=torch.device('cuda',lane);torch.cuda.set_device(device)
        with torch.inference_mode():
            model=RMEAHead('quantum').double().to(device);model.load_state_dict(initial);model.eval()
            native_error=identity_error=density_error=0.;fixture_done=False
            for lo in range(0,len(ids),cfg['microbatch_size']):
                index=ids[lo:lo+cfg['microbatch_size']];lengths=np.diff(data['offsets'])[index];width=int(lengths.max())
                h=np.zeros((len(index),width,1024),np.float64);roles=np.zeros((len(index),width,4),np.float64)
                valid=np.zeros((len(index),width),bool);region=np.zeros((4,len(index),width),bool)
                for j,i in enumerate(index):
                    left,right=map(int,data['offsets'][i:i+2]);size=right-left
                    h[j,:size]=data['h'][left:right];roles[j,:size]=data['roles'][left:right];valid[j,:size]=True
                    b,o=masks[i];region[1,j,:size]=b;region[2,j,:size]=o;region[3,j,:size]=b|o
                inputs=tuple(torch.from_numpy(x).to(device) for x in (h,roles,valid));mask=torch.from_numpy(region).to(device)
                d=model(*inputs,details=True);psi=d['psi'];phi=d['phi'];p=d['probability'];u=d['utilities']
                diagonal=torch.einsum('rd,btd->brt',phi.abs().square(),psi.abs().square())
                delta=mask[:,:,None,:]*(diagonal-p)[None]
                probability=p[None]+delta;raw=(probability+EPS)*inputs[2][None,:,None,:]
                normalizer=raw.sum(-1,keepdim=True);pooled=(raw*u[None]).sum(-1)/normalizer.squeeze(-1)
                z=pooled+model.bias
                err=float((z[0]-d['logits']).abs().max());native_error=max(native_error,err);assert err<1e-9
                assert torch.equal(z[0].argmax(-1),d['logits'].argmax(-1))
                expected=(delta*(u-d['logits'][:,:,None]+model.bias[None,:,None])[None]).sum(-1)/normalizer.squeeze(-1)
                err=float(((z-z[0])-expected).abs().max());identity_error=max(identity_error,err);assert err<1e-9
                assert torch.isfinite(z).all() and float(probability.min())>=-1e-12
                assert torch.max(torch.abs(delta[3]-delta[1]-delta[2]))<1e-12
                if not fixture_done:
                    # Independently apply the density channel before the effect trace.
                    rho=psi[..., :,None]*psi.conj()[...,None,:]
                    diag=torch.diag_embed(torch.diagonal(rho,dim1=-2,dim2=-1))
                    for k in range(4):
                        changed=torch.where(mask[k,:,:,None,None],diag,rho)
                        assert torch.max(torch.abs(torch.diagonal(changed,dim1=-2,dim2=-1)-torch.diagonal(rho,dim1=-2,dim2=-1)))<1e-12
                        direct=torch.einsum('ri,btij,rj->brt',phi.conj(),changed,phi).real
                        err=float((direct-probability[k]).abs().max());density_error=max(density_error,err);assert err<1e-9
                    model.mode='dephased';full,_=model(*inputs);model.mode='quantum'
                    q=(diagonal+EPS)*inputs[2][:,None,:];direct=(q*u).sum(-1)/q.sum(-1)+model.bias
                    assert torch.max(torch.abs(full-direct))<1e-9
                    fixture_done=True
                logits[:,index,:]=z.cpu().numpy();identity_terms[:,index,:]=expected.cpu().numpy()
                if lo%128==0 or lo+len(index)==len(ids):progress(lane,lo+len(index))
            for name,param in model.state_dict().items():assert torch.equal(param.cpu(),initial[name])
        return dict(lane=lane,rows=len(ids),native_error=native_error,exact_identity_error=identity_error,density_channel_error=density_error)

    progress(0,0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,lane,ids) for lane,ids in enumerate(np.array_split(np.arange(n),2))]
        fixtures=[job.result() for job in jobs]
    assert cursors==[n//2,n-n//2] and np.isfinite(logits).all()
    assert np.max(np.abs(logits[:,~eligible]-logits[0,~eligible]))<1e-9
    result={name:describe(z,y,metrics) for name,z in zip(modes,logits)}
    old=logits[0].argmax(-1)
    for k,name in enumerate(modes):
        new=logits[k].argmax(-1);result[name].update(repaired=int(((old!=y)&(new==y)).sum()),
            damaged=int(((old==y)&(new!=y)).sum()),changed=int((old!=new).sum()))
    gates,gate=decision(result,cfg['tolerance']);assert digest(a.checkpoint)==cfg['checkpoint_sha256']
    private=a.output/'predictions.private.npz';np.savez_compressed(private,logits=logits,labels=y,eligible=eligible,identity_terms=identity_terms)
    summary=dict(profile=cfg['profile'],gate=gate,gates=gates,results=result,seconds=time.time()-started,examples=n,
        eligible_rows=int(eligible.sum()),eligible_positive=int((eligible&(y!=0)).sum()),optimizer_updates=0,
        source_checkpoint_unchanged=True,design_sha256=digest(a.design),checkpoint_sha256=cfg['checkpoint_sha256'],
        predictions_sha256=digest(private),claim_scope=cfg['claim_scope'])
    write(a.output/'summary.json',summary);write(a.output/'fixtures.json',dict(status='PASS',lanes=fixtures,
        all_span_cache_memberships_verified=True,all_ineligible_rows_identity=True,weights_unchanged=True))
    write(a.output/'RUN_COMPLETE',dict(files={n:digest(a.output/n) for n in ('summary.json','fixtures.json','predictions.private.npz')}))
    print(json.dumps(dict(gate=gate,seconds=summary['seconds'],eligible_rows=summary['eligible_rows'],results={k:{m:v[m] for m in ('ce','positive_macro_f1','repaired','damaged')} for k,v in result.items()})))


if __name__=='__main__':main()
