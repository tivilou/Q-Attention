"""One frozen effect step on full development VALID; no optimization or search."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,math,signal,sys,threading,time
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
    return dict(ce=float(ce.mean()),class0_ce=float(ce[y==0].mean()),positive_ce=float(ce[y!=0].mean()),
        positive_macro_f1=float(np.mean([r['f1'] for r in m['per_class'][1:]])),
        accuracy=m['accuracy'],micro_f1=m['micro_f1'],confusion_matrix=m['confusion_matrix'],
        false_negative=int(((y!=0)&(pred==0)).sum()),false_positive=int(((y==0)&(pred!=0)).sum()))


def decision(rows,tol):
    c=rows['common'];n=rows['native']
    gates=dict(class0_ce_decreases=c['class0_ce']<n['class0_ce']-tol,
        positive_ce_decreases=c['positive_ce']<n['positive_ce']-tol,
        macro_beats_native=c['positive_macro_f1']>n['positive_macro_f1']+tol,
        macro_beats_balanced=c['positive_macro_f1']>rows['balanced']['positive_macro_f1']+tol,
        macro_beats_frequency=c['positive_macro_f1']>rows['frequency']['positive_macro_f1']+tol)
    return gates,'PASS_FIXED_STEP_ONLY' if all(gates.values()) else 'FAIL_CLOSE_EXACT_STEP'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('source','inputs','reference','checkpoint','design','output','progress','allocation'):
        ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--attempt',required=True);a=ap.parse_args();cfg=json.loads(a.design.read_bytes())
    alloc=json.loads(a.allocation.read_bytes())['allocation']
    assert alloc['runner_attempt_id']==a.attempt and len(set(alloc['gpu_uuids']))==2
    for n,h in cfg['code_hashes'].items():assert digest(Path(__file__).parent/n)==h
    for n,h in cfg['source_hashes'].items():assert digest(a.source/n)==h
    assert digest(a.checkpoint)==cfg['checkpoint_sha256']
    manifest=a.inputs/'discovery/manifest.json';assert digest(manifest)==cfg['native_manifest_sha256']
    manifest=json.loads(manifest.read_bytes());cache=a.inputs/'discovery/cache'
    for n in ('h','roles','offsets','targets'):
        name='valid-'+n+'.npy';assert digest(cache/name)==manifest['cache'][name]['sha256']
    parent=a.reference
    for n,h in cfg['producer_file_hashes'].items():assert digest(parent/n)==h
    producer=json.loads((parent/'design.json').read_bytes())
    assert producer['modes']==cfg['modes'][1:]
    assert producer['code_hashes']['two_group.py']==cfg['producer_file_hashes']['two_group.py']
    for n,h in cfg['direction_artifact_hashes'].items():assert digest(parent/'output'/n)==h
    complete=json.loads((parent/'output/RUN_COMPLETE').read_bytes())
    for n in ('summary.json','directions-frozen.json','directions.private.npy'):
        assert digest(parent/'output'/n)==complete['files'][n]
    selected=json.loads((parent/'output/summary.json').read_bytes())
    assert selected['gate']=='PASS_LOCAL_DIRECTION_ONLY'
    directions=np.load(parent/'output/directions.private.npy',allow_pickle=False)
    assert directions.shape==(3,40,2,3,2) and np.isfinite(directions).all()
    assert np.max(np.abs(np.linalg.norm(directions.reshape(3,-1),axis=1)-1))<1e-12
    a.output.mkdir(exist_ok=True);assert not (a.output/'started.json').exists()
    write(a.output/'started.json',dict(attempt=a.attempt,design_sha256=digest(a.design),eta=cfg['eta']))
    signal.alarm(cfg['timeout_seconds']);started=time.time()
    import torch
    sys.path.insert(0,str(a.source))
    from baselines.retacred.rmea import RMEAHead,circuit,EPS
    from baselines.retacred.common import metrics
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert torch.cuda.device_count()==2 and EPS==1e-6
    initial=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    data={n:np.load(cache/('valid-'+n+'.npy'),mmap_mode='r',allow_pickle=False) for n in ('h','roles','offsets','targets')}
    y=np.asarray(data['targets']);n=len(y);assert n==cfg['valid_count'] and set(y)==set(range(40))
    assert data['offsets'].shape==(n+1,) and data['offsets'][0]==0 and data['offsets'][-1]==len(data['h'])
    assert len(data['roles'])==len(data['h']) and np.all(np.diff(data['offsets'])>0)
    modes=cfg['modes'];assert modes==['native','frequency','balanced','common']
    logits=np.empty((4,n,40),np.float64);lock=threading.Lock();cursors=[0,0]

    def progress(lane,count):
        with lock:
            cursors[lane]=count
            write(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,
                phase='fixed_step_full_valid',current=sum(cursors),total=n,unit='example',
                observed_at=time.time(),heartbeat_at=time.time(),gpu_lanes=cursors.copy(),modes=modes))

    def worker(lane,ids):
        device=torch.device('cuda',lane);torch.cuda.set_device(device)
        with torch.inference_mode():
            model=RMEAHead('quantum').double().to(device);model.load_state_dict(initial);model.eval()
            theta=model.effect.detach().clone()
            direction=torch.as_tensor(directions,device=device)
            effects=torch.cat((theta[None],theta[None]+cfg['eta']*direction),dim=0)
            phi=circuit(math.pi*torch.tanh(effects));native_error=direct_error=0.;fixture_done=False
            for lo in range(0,len(ids),cfg['microbatch_size']):
                index=ids[lo:lo+cfg['microbatch_size']]
                lengths=np.diff(data['offsets'])[index];width=int(lengths.max())
                h=np.zeros((len(index),width,1024),np.float64);r=np.zeros((len(index),width,4),np.float64)
                m=np.zeros((len(index),width),bool)
                for j,i in enumerate(index):
                    left,right=map(int,data['offsets'][i:i+2]);size=right-left
                    h[j,:size]=data['h'][left:right];r[j,:size]=data['roles'][left:right];m[j,:size]=True
                inputs=tuple(torch.from_numpy(x).to(device) for x in (h,r,m))
                detail=model(*inputs,details=True)
                p=torch.einsum('krd,btd->kbrt',phi.conj(),detail['psi']).abs().square()
                raw=(p+EPS)*inputs[2][None,:,None,:]
                alpha=raw/raw.sum(-1,keepdim=True)
                z=(alpha*detail['utilities'][None]).sum(-1)+model.bias
                err=float((z[0]-detail['logits']).abs().max());native_error=max(native_error,err);assert err<1e-9
                assert torch.equal(z[0].argmax(-1),detail['logits'].argmax(-1))
                if not fixture_done:
                    for k in range(4):
                        model.effect.copy_(effects[k]);direct,_=model(*inputs)
                        err=float((z[k]-direct).abs().max());direct_error=max(direct_error,err);assert err<1e-9
                    model.effect.copy_(theta);fixture_done=True
                logits[:,index,:]=z.cpu().numpy()
                if lo%128==0 or lo+len(index)==len(ids):progress(lane,lo+len(index))
            assert torch.equal(model.effect,theta)
            for name,param in model.state_dict().items():assert torch.equal(param.cpu(),initial[name])
        return dict(lane=lane,rows=len(ids),source_native_max_error=native_error,first_batch_all_modes_direct_error=direct_error)

    progress(0,0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,lane,ids) for lane,ids in enumerate(np.array_split(np.arange(n),2))]
        fixtures=[job.result() for job in jobs]
    assert cursors==[n//2,n-n//2] and np.isfinite(logits).all()
    rows={name:describe(z,y,metrics) for name,z in zip(modes,logits)}
    for k,name in enumerate(modes):
        old=logits[0].argmax(-1);new=logits[k].argmax(-1)
        rows[name].update(repaired=int(((old!=y)&(new==y)).sum()),
            damaged=int(((old==y)&(new!=y)).sum()),changed=int((old!=new).sum()))
    gates,gate=decision(rows,cfg['tolerance'])
    assert digest(a.checkpoint)==cfg['checkpoint_sha256']
    private=a.output/'predictions.private.npz';np.savez_compressed(private,logits=logits,labels=y)
    summary=dict(profile=cfg['profile'],eta=cfg['eta'],gate=gate,gates=gates,results=rows,seconds=time.time()-started,
        examples=n,optimizer_updates=0,source_checkpoint_unchanged=True,design_sha256=digest(a.design),
        checkpoint_sha256=cfg['checkpoint_sha256'],predictions_sha256=digest(private),claim_scope=cfg['claim_scope'])
    write(a.output/'summary.json',summary);write(a.output/'fixtures.json',dict(status='PASS',lanes=fixtures,
        valid_data_hashes_verified=True,direction_seals_verified=True,producer_axis_order_verified=True,source_model_unchanged=True))
    write(a.output/'RUN_COMPLETE',dict(files={n:digest(a.output/n) for n in ('summary.json','fixtures.json','predictions.private.npz')}))
    print(json.dumps(dict(gate=gate,seconds=summary['seconds'],results={k:{m:v[m] for m in ('ce','positive_macro_f1','class0_ce','positive_ce','repaired','damaged')} for k,v in rows.items()})))


if __name__=='__main__':main()
