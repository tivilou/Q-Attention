"""Label-blind full-TRAIN native angle/route census; no fitting or binary output."""
from pathlib import Path
import argparse, hashlib, json, math, signal, sys, time
import numpy as np

QUANTILES=[0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def save(path,value):
    tmp=path.with_suffix('.pending')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def stats(values):
    x=np.asarray(values)
    assert x.size and np.isfinite(x).all()
    return dict(count=int(x.size),mean=float(np.mean(x,dtype=np.float64)),
                quantiles=np.quantile(x,QUANTILES).tolist())

def main():
    ap=argparse.ArgumentParser()
    for n in ('root','design','output','progress'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--attempt',required=True);a=ap.parse_args()
    cfg=json.loads(a.design.read_bytes());source=a.root/cfg['source'];completed=a.root/cfg['completed']
    assert digest(Path(__file__))==cfg['code_sha256']
    for n,h in cfg['source_hashes'].items():assert digest(source/n)==h,n
    assert digest(completed/'manifest.json')==cfg['manifest_sha256']
    # Targets and labels are deliberately absent from both the allowlist and the reads.
    cache=completed/'cache';assert set(cfg['train_cache'])=={'train-h.npy','train-roles.npy','train-offsets.npy'}
    for n,v in cfg['train_cache'].items():assert digest(cache/n)==v['sha256'],n
    state_path=completed/cfg['selected']['path'];assert digest(state_path)==cfg['selected']['sha256']
    sys.path.insert(0,str(source));import torch
    from baselines.retacred.rmea import RMEAHead,circuit,EPS
    from baselines.retacred.rmea_token_transport_study import state_identity
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;assert torch.cuda.device_count()==1
    assert not a.output.exists();a.output.mkdir(parents=True)
    data={k:np.load(cache/('train-'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('h','roles','offsets')}
    n=len(data['offsets'])-1;tokens=int(data['offsets'][-1]);assert n==58465
    assert data['offsets'][0]==0 and tokens==len(data['h'])==len(data['roles'])
    started=time.monotonic()
    def expired(*_):raise TimeoutError('Fixed 120 second scientific compute budget exhausted')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,120)
    def progress(mode,current):
        save(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,
            phase='native_angle_aliasing_'+mode,current=current,total=2*n,unit='example',
            heartbeat_at=time.time(),observed_at=time.time()))
    progress('fixtures',0)
    device=torch.device('cuda',0)
    with torch.inference_mode():
        signs=torch.tensor([[1 if i&(1<<j) else -1 for j in range(12)] for i in range(4096)],device=device,dtype=torch.float64)
        corners=circuit((signs*math.pi).reshape(4096,2,3,2))
        corner=circuit(torch.full((2,3,2),math.pi,device=device,dtype=torch.float64))
        projector_error=float((corners[:,:,None]*corners[:,None,:].conj()-corner[:,None]*corner[None,:].conj()).abs().max())
        assert projector_error<1e-12
    state=torch.load(state_path,map_location='cpu',weights_only=True)
    assert state_identity(state)==cfg['selected']['state_sha256']
    torch.manual_seed(13);initial=RMEAHead('quantum').float()
    initial_id=state_identity({'head.'+k:v for k,v in initial.state_dict().items()})
    models=[('initial_seed13',initial)]
    trained=RMEAHead('quantum').float()
    assert all(k.startswith('head.') for k in state)
    trained.load_state_dict({k[5:]:v for k,v in state.items()},strict=True)
    models.append(('selected_epoch17',trained));results={};checks={}
    with torch.inference_mode():
        for lane,(name,head) in enumerate(models):
            begin=time.monotonic();head.to(device).eval()
            pre=np.empty((tokens,12),np.float32);slope=np.empty_like(pre)
            fidelity=np.empty(tokens,np.float64);distance=np.empty_like(fidelity);loose=np.empty_like(fidelity)
            tv=np.empty((n,40),np.float32);counts=np.zeros((3,12),np.int64)
            route_error=padding_error=bound_error=0.;maximum_norm_error=0.
            phi=circuit(math.pi*torch.tanh(head.effect)).to(torch.complex128)
            corner_prob=(phi.conj()@corner).abs().square().cpu().numpy()
            for lo in range(0,n,128):
                hi=min(lo+128,n);lengths=np.diff(data['offsets'][lo:hi+1])
                h=np.zeros((hi-lo,int(lengths.max()),1024),np.float32)
                roles=np.zeros(h.shape[:2]+(4,),np.float32);mask=np.zeros(h.shape[:2],bool)
                for j,i in enumerate(range(lo,hi)):
                    left,right=map(int,data['offsets'][i:i+2]);size=right-left
                    h[j,:size]=data['h'][left:right];roles[j,:size]=data['roles'][left:right];mask[j,:size]=True
                ht,rt,mt=(torch.from_numpy(v).to(device) for v in (h,roles,mask))
                d=head(ht,rt,mt,details=True)
                x=head.prepare(torch.cat((ht,rt),-1))[mt];t=torch.tanh(x)
                left,right=map(int,(data['offsets'][lo],data['offsets'][hi]))
                pre[left:right]=x.abs().cpu().numpy();slope[left:right]=(1-t.square()).cpu().numpy()
                for j,threshold in enumerate((.95,.99,.999)):counts[j]+=(t.abs()>=threshold).sum(0).cpu().numpy()
                psi=d['psi'][mt].to(torch.complex128);norm=psi.abs().square().sum(-1)
                maximum_norm_error=max(maximum_norm_error,float((norm-1).abs().max()))
                psi=psi/norm.sqrt()[:,None]
                f=(psi@corner.conj()).abs().square().clamp(0,1)
                dist=(1-f).sqrt();b=(math.pi-d['angles'][mt].double().abs()).sum((-1,-2,-3))/2
                bound_error=max(bound_error,float((dist-b.clamp_max(1)).clamp_min(0).max()))
                fidelity[left:right]=f.cpu().numpy();distance[left:right]=dist.cpu().numpy();loose[left:right]=b.cpu().numpy()
                uniform=mt[:,None,:]/mt.sum(-1)[:,None,None]
                tv[lo:hi]=((d['alpha']-uniform).abs().sum(-1)/2).cpu().numpy()
                if lo==0:
                    raw=(d['probability']+EPS)*mt[:,None,:];expected=raw/raw.sum(-1,keepdim=True)
                    route_error=float((expected-d['alpha']).abs().max())
                    ph=torch.cat((ht,torch.ones((len(ht),3,1024),device=device)),1)
                    pr=torch.cat((rt,torch.ones((len(rt),3,4),device=device)),1)
                    pm=torch.cat((mt,torch.zeros((len(mt),3),dtype=torch.bool,device=device)),1)
                    padded=head(ph,pr,pm,details=True)
                    padding_error=max(float((padded['logits']-d['logits']).abs().max()),
                        float((padded['alpha'][...,:ht.shape[1]]-d['alpha']).abs().max()),float(padded['alpha'][...,ht.shape[1]:].abs().max()))
                    assert route_error<1e-6 and padding_error<1e-5
                if lo%2048==0:progress(name,lane*n+hi)
            assert bound_error<1e-5 and maximum_norm_error<1e-5
            results[name]=dict(valid_tokens=tokens,examples=n,
                absolute_preactivation=stats(pre),tanh_slope=stats(slope),
                per_coordinate=[dict(index=j,absolute_preactivation=stats(pre[:,j]),tanh_slope=stats(slope[:,j]),
                    saturated_counts={str(t):int(counts[k,j]) for k,t in enumerate((.95,.99,.999))}) for j in range(12)],
                saturated_coordinate_counts={str(t):int(counts[k].sum()) for k,t in enumerate((.95,.99,.999))},
                corner_fidelity=stats(fidelity),corner_trace_distance=stats(distance),
                untruncated_distance_bound=stats(loose),
                corner_fidelity_counts={str(t):int((fidelity>=t).sum()) for t in (.9,.99,.999)},
                corner_born_probability_by_relation=corner_prob.tolist(),
                corner_born_probability_all_repeated=stats(np.broadcast_to(corner_prob,(n,40))),
                attention_tv_from_uniform=stats(tv),
                attention_tv_by_relation=[stats(tv[:,j]) for j in range(40)],seconds=time.monotonic()-begin)
            checks[name]=dict(route_parity_error=route_error,padding_error=padding_error,
                unit_norm_max_error=maximum_norm_error,trace_distance_bound_max_violation=bound_error)
            head.cpu();del pre,slope,fidelity,distance,loose,tv
            progress(name,(lane+1)*n)
    elapsed=time.monotonic()-started;assert elapsed<120
    signal.setitimer(signal.ITIMER_REAL,0)
    report=dict(profile=cfg['profile'],examples=n,valid_tokens=tokens,quantile_probabilities=QUANTILES,
        results=results,checks=checks,corner_projector_max_error=projector_error,
        initial_state_sha256=initial_id,selected_state_sha256=cfg['selected']['state_sha256'],seconds=elapsed,
        labels_read=0,optimizer_updates=0,valid_rows_evaluated=0,test_rows_evaluated=0,cos_writes=0,
        claim='Descriptive label-blind encoding/route census, not task causation, utility or new theory.')
    save(a.output/'summary.json',report)
    save(a.output/'RUN_COMPLETE',dict(summary_sha256=digest(a.output/'summary.json'),attempt=a.attempt))
    print(json.dumps(dict(seconds=elapsed,examples=n,valid_tokens=tokens,checks=checks)))

if __name__=='__main__':main()
