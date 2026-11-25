"""Full TRAIN fixed-readout intervention, not a model or training change."""
from pathlib import Path
import argparse, hashlib, json, os, signal, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def save(path,value):
    p=path.with_suffix('.pending')
    p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')
    p.replace(path)

def main():
    ap=argparse.ArgumentParser()
    for n in ('design','allocation','output','progress'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--attempt',required=True);args=ap.parse_args()
    cfg=json.loads(args.design.read_bytes());a=json.loads(args.allocation.read_bytes())['allocation']
    assert a['runner_attempt_id']==args.attempt and len(set(a['gpu_uuids']))==2
    os.environ.update(CUDA_VISIBLE_DEVICES=','.join(a['gpu_uuids']),CUBLAS_WORKSPACE_CONFIG=':4096:8')
    signal.alarm(cfg['timeout_seconds']);started=time.monotonic()
    for n,h in cfg['code_hashes'].items():assert sha(Path(__file__).parent/n)==h
    for p,h in cfg['input_hashes'].items():assert sha(p)==h,p
    args.output.mkdir(exist_ok=True);assert not (args.output/'started.json').exists()
    save(args.output/'started.json',dict(attempt=args.attempt,design_sha256=sha(args.design)))
    import numpy as np
    import torch
    sys.path[:0]=[cfg['source'],str(Path(__file__).parent)]
    import plugin
    from baselines.retacred.rmea import EPS
    from baselines.retacred.rmea_token_transport_study import state_identity
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.device_count()==2
    cache=Path(cfg['cache']);n=cfg['rows']
    data={k:np.load(cache/('train-'+k+'.npy'),mmap_mode='r',allow_pickle=False)
          for k in ('h','roles','offsets','targets')}
    gold=np.asarray(data['targets']);assert len(gold)==n and len(data['offsets'])==n+1
    assert gold.min()>=0 and gold.max()<40
    batches=list(range(0,n,cfg['batch_size']));lock=threading.Lock();cursors=[0,0]
    def progress(device,count):
        with lock:
            cursors[device]=count
            save(args.progress,dict(schema='research.progress.v1',attempt=args.attempt,
                phase='fixed_gate_readout',current=sum(cursors),total=n,unit='evaluated_example',
                observed_at=time.time(),heartbeat_at=time.time()))
    def batch(lo,device):
        hi=min(lo+cfg['batch_size'],n);lengths=np.diff(data['offsets'][lo:hi+1]);width=int(lengths.max())
        h=np.zeros((hi-lo,width,1024),np.float32);roles=np.zeros((hi-lo,width,4),np.float32)
        mask=np.zeros((hi-lo,width),bool)
        for j,i in enumerate(range(lo,hi)):
            l,u=map(int,data['offsets'][i:i+2]);h[j,:u-l]=data['h'][l:u]
            roles[j,:u-l]=data['roles'][l:u];mask[j,:u-l]=True
        return hi,tuple(torch.from_numpy(x).to(device) for x in (h,roles,mask))
    def worker(device_id):
        device=torch.device('cuda',device_id)
        state=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True)
        assert state_identity(state)==cfg['checkpoint_state_sha256']
        model=plugin.make_head('quantum_corroboration').float()
        model.load_state_dict(state,strict=True);model.to(device).eval()
        result=[];count=0;errors=dict(native=0.,covariance=0.,margin=0.,fp32_scaled=0.)
        with torch.cuda.device(device),torch.inference_mode():
            for lo in batches[device_id::2]:
                hi,(h,roles,mask)=batch(lo,device);z_on=model(h,roles,mask)
                alpha,gate,base=model.components(h,roles,mask);head=model.head;value=head.value(h)
                replay=(torch.einsum('brt,btd->brd',alpha,value)*head.decoder).sum(-1)+head.bias
                assert torch.allclose(z_on,replay,atol=cfg['replay_atol'],rtol=cfg['replay_rtol'])
                assert torch.equal(z_on.argmax(-1),replay.argmax(-1))
                errors['native']=max(errors['native'],float((z_on-replay).abs().max()))
                raw=(base+EPS)*mask[:,None];off=raw/raw.sum(-1,keepdim=True)
                z_off=(torch.einsum('brt,btd->brd',off,value)*head.decoder).sum(-1)+head.bias
                # Independent FP64 algebra uses actual projected FP32 components.
                b=base.double();g=gate.double();valid=mask[:,None]
                w0=(b+EPS)*valid;w1=(b*g[:,None]+EPS)*valid
                a0=w0/w0.sum(-1,keepdim=True);a1=w1/w1.sum(-1,keepdim=True)
                c=(b*g[:,None]+EPS)/(b+EPS);mc=(a0*c).sum(-1)
                u=torch.einsum('btd,rd->brt',value.double(),head.decoder.double())
                mean=(a0*u).sum(-1);cov=(a0*c*u).sum(-1)-mc*mean;delta=cov/mc
                z0=(a0*u).sum(-1)+head.bias.double();z1=(a1*u).sum(-1)+head.bias.double()
                err=float((z1-z0-delta).abs().max());errors['covariance']=max(errors['covariance'],err)
                assert err<=cfg['identity_atol']
                scaled=max(float(((z0-z_off.double()).abs()/(1+z0.abs())).max()),
                           float(((z1-z_on.double()).abs()/(1+z1.abs())).max()))
                errors['fp32_scaled']=max(errors['fp32_scaled'],scaled)
                assert scaled<=cfg['fp32_scaled_atol']
                y=torch.as_tensor(gold[lo:hi].copy(),device=device,dtype=torch.long);ix=torch.arange(hi-lo,device=device)
                rivals0=z0.clone();rivals0[ix,y]=-torch.inf;s0=rivals0.argmax(-1)
                rivals1=z1.clone();rivals1[ix,y]=-torch.inf
                switch=rivals1.max(-1).values-z1[ix,s0]
                m0=z0[ix,y]-rivals0.max(-1).values;m1=z1[ix,y]-rivals1.max(-1).values
                explained=delta[ix,y]-delta[ix,s0]-switch
                me=float((m1-m0-explained).abs().max());errors['margin']=max(errors['margin'],me)
                assert me<=cfg['identity_atol'] and bool((switch>=-cfg['identity_atol']).all())
                for z in (z_on,z_off,z0,z1,delta):assert bool(torch.isfinite(z).all())
                result.append((lo,hi,z_on.cpu().numpy(),z_off.cpu().numpy(),delta.cpu().numpy(),
                               switch.cpu().numpy(),gate[mask].cpu().numpy()))
                count+=hi-lo;progress(device_id,count)
        return result,errors
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,i) for i in range(2)];results=[f.result() for f in futures]
    on=np.zeros((n,40),np.float32);off=np.zeros_like(on);delta=np.zeros((n,40),np.float64)
    switch=np.zeros(n);seen=np.zeros(n,bool);gates=[]
    errors={k:max(x[1][k] for x in results) for k in results[0][1]}
    for records,_ in results:
        for lo,hi,zo,zf,de,sw,ga in records:
            assert not seen[lo:hi].any();seen[lo:hi]=True
            on[lo:hi]=zo;off[lo:hi]=zf;delta[lo:hi]=de;switch[lo:hi]=sw;gates.append(ga)
    assert seen.all() and sha(cfg['checkpoint'])==cfg['input_hashes'][cfg['checkpoint']]
    def metrics(z):
        pred=z.argmax(-1);cm=np.bincount(40*gold+pred,minlength=1600).reshape(40,40)
        den=cm.sum(0)+cm.sum(1);fs=np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den!=0)
        zz=z.astype(np.float64);zz-=zz.max(-1,keepdims=True)
        ce=np.mean(np.log(np.exp(zz).sum(-1))-zz[np.arange(n),gold])
        return dict(macro=float(fs.mean()),ce=float(ce),correct=int((pred==gold).sum()),
            confusion_matrix=cm.tolist(),positive_to_null=int(cm[1:,0].sum()),null_to_positive=int(cm[0,1:].sum()))
    po,pf=on.argmax(-1),off.argmax(-1);co=po==gold;cf=pf==gold
    def margin(z):
        zz=z.astype(np.float64).copy();target=zz[np.arange(n),gold].copy();zz[np.arange(n),gold]=-np.inf
        return target-zz.max(-1)
    mo,mf=margin(on),margin(off)
    def stats(x):return dict(mean=float(x.mean()),median=float(np.median(x)),positive=int((x>0).sum()),negative=int((x<0).sum()),zero=int((x==0).sum()))
    pair=dict(gate_on_repairs=int((co&~cf).sum()),gate_on_harms=int((~co&cf).sum()),both_correct=int((co&cf).sum()),both_wrong=int((~co&~cf).sum()))
    strata={name:dict(rows=int(sel.sum()),on_minus_off_margin=stats((mo-mf)[sel]))
            for name,sel in [('on_correct',co),('on_wrong',~co)] if sel.any()}
    gate=np.concatenate(gates)
    np.savez(args.output/'diagnostic.private.npz',gold=gold,on=on,off=off,covariance_delta=delta,switch_penalty=switch)
    summary=dict(status='completed',attempt=args.attempt,profile=cfg['profile'],rows=n,seconds=time.monotonic()-started,
        results=dict(gate_on=metrics(on),gate_off_same_parameters=metrics(off)),paired=pair,
        on_minus_off_margin=stats(mo-mf),strata=strata,
        gate_statistics=dict(tokens=len(gate),mean=float(gate.mean()),minimum=float(gate.min()),maximum=float(gate.max())),
        verification_errors=errors,code_hashes=cfg['code_hashes'],input_hashes=cfg['input_hashes'],design_sha256=sha(args.design),
        native_argmax_reconstruction_equal=True,private_diagnostic_sha256=sha(args.output/'diagnostic.private.npz'),
        optimizer_updates=0,cos_writes=0,valid_evaluated=False,test_evaluated=False,
        L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED',gate='TECHNICAL_PASS_DIAGNOSTIC_ONLY',
        limits='Same selected Q, fixed parameters, exposed fullTRAIN. FP64 identity versus actual FP32 correspondence distinguished. No parent TRAIN metrics were stored. No training-causal, generalization, semantic or novelty claim.')
    save(args.output/'summary.json',summary);save(args.output/'RUN_COMPLETE',{'summary.json':sha(args.output/'summary.json')})
    save(args.progress,dict(schema='research.progress.v1',attempt=args.attempt,phase='completed',current=n,total=n,
        unit='evaluated_example',observed_at=time.time(),heartbeat_at=time.time()))
    print(json.dumps(dict(seconds=summary['seconds'],macro={k:v['macro'] for k,v in summary['results'].items()},paired=pair,errors=errors)),flush=True)

if __name__=='__main__':main()
