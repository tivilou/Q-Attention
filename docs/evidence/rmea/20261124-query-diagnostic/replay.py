"""Label-free R0 query/routing diagnosis of the five completed selected states."""
from pathlib import Path
import argparse,hashlib,json,math,sys,time
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    q=p.with_suffix('.pending');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)
def stats(x):
    x=np.asarray(x,dtype=np.float64).reshape(-1)
    assert np.isfinite(x).all()
    return dict(mean=float(x.mean()),q05=float(np.quantile(x,.05)),median=float(np.median(x)),q95=float(np.quantile(x,.95)),min=float(x.min()),max=float(x.max()),n=int(x.size))

def main():
    ap=argparse.ArgumentParser()
    for n in ('root','design','output','progress'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--attempt',required=True);a=ap.parse_args();cfg=json.loads(a.design.read_bytes())
    assert sha(__file__)==cfg['code_sha256']
    source=Path(cfg['source']);cache=Path(cfg['cache']);completed=Path(cfg['completed'])
    assert sha(completed/'manifest.json')==cfg['manifest_sha256']
    for n,h in cfg['source_hashes'].items():assert sha(source/n)==h,n
    # The immutable feature cache was previously hashed; do not reread 4.4 GB.
    assert (cache/'train-h.npy').stat().st_size==cfg['train_cache']['train-h.npy']['bytes']
    for n in ('train-roles.npy','train-offsets.npy'):assert sha(cache/n)==cfg['train_cache'][n]['sha256']
    data={n:np.load(cache/('train-'+n+'.npy'),mmap_mode='r',allow_pickle=False) for n in ('h','roles','offsets')}
    assert data['offsets'].shape==(58466,) and data['offsets'][0]==0 and data['offsets'][-1]==len(data['h'])==len(data['roles'])
    sys.path.insert(0,str(source));import torch
    from torch.nn import functional as F
    from baselines.retacred.rmea_entity_query import EntityQueryHead,anchor,real_circuit
    from baselines.retacred.rmea import circuit
    from baselines.retacred.rmea_token_transport_study import state_identity
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert not a.output.exists();a.output.mkdir(parents=True)
    start=time.monotonic();report={};ii,jj=torch.triu_indices(40,40,1)
    def progress(done):save(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,phase='TRAIN128_query_routing_diagnostic',current=done,total=640,unit='model_example',observed_at=time.time(),heartbeat_at=time.time()))
    progress(0)
    with torch.inference_mode():
        for mi,(kind,pin) in enumerate(cfg['selected'].items()):
            p=completed/pin['path'];assert sha(p)==pin['sha256']
            state=torch.load(p,map_location='cpu',weights_only=True);assert state_identity(state)==pin['state_sha256']
            model=EntityQueryHead(kind).float().eval();model.load_state_dict(state,strict=True)
            out={};parity=0.
            def add(k,x):out.setdefault(k,[]).append(x.detach().cpu().numpy().reshape(-1))
            for lo in range(0,128,16):
                hi=lo+16;lengths=np.diff(data['offsets'][lo:hi+1]);h=np.zeros((16,int(lengths.max()),1024),np.float32)
                roles=np.zeros(h.shape[:2]+(4,),np.float32);mask=np.zeros(h.shape[:2],bool)
                for b,i in enumerate(range(lo,hi)):
                    left,right=map(int,data['offsets'][i:i+2]);n=right-left
                    h[b,:n]=data['h'][left:right];roles[b,:n]=data['roles'][left:right];mask[b,:n]=True
                h,roles,mask=map(torch.from_numpy,(h,roles,mask));head=model.head;x=torch.cat((h,roles),-1);c=anchor(x,roles,mask)
                routes=[]
                for conditioned in (False,True):
                    suffix='conditioned' if conditioned else 'unconditioned'
                    if model.mode in ('quantum','dephased'):
                        ka=math.pi*torch.tanh(head.prepare(x));pre=head.effect.flatten(-3)[None].expand(16,-1,-1)
                        if conditioned:pre=pre+head.prepare(c)[:,None,:]
                        qa=math.pi*pre.tanh()
                        if kind=='real_circuit_context':key=real_circuit(ka);query=real_circuit(qa)
                        else:key=circuit(ka.reshape(h.shape[:2]+(2,3,2)));query=circuit(qa.reshape(16,40,2,3,2))
                        add(suffix+'/angle_pair_rms',((qa[:,ii]-qa[:,jj]).square().mean(-1)).sqrt())
                        add(suffix+'/tanh_slope',1-pre.tanh().square())
                        gram=torch.einsum('brd,bsd->brs',query.conj(),query).abs().square().clamp(0,1)
                        add(suffix+'/pure_query_trace_distance',(1-gram[:,ii,jj]).sqrt())
                        if model.mode=='dephased':
                            probability=query.abs().square()
                            add(suffix+'/diagonal_query_trace_distance',(probability[:,ii]-probability[:,jj]).abs().sum(-1)/2)
                            score=torch.einsum('brd,btd->brt',probability,key.abs().square())
                        else:score=torch.einsum('brd,btd->brt',query.conj(),key).abs().square()
                    else:
                        key=head.key(x);query=head.query[None].expand(16,-1,-1)
                        if conditioned:query=query+head.key(c)[:,None,:]
                        add(suffix+'/query_pair_l2',(query[:,ii]-query[:,jj]).norm(dim=-1))
                        if model.mode=='softmax12':score=torch.einsum('brd,btd->brt',query,key)/math.sqrt(12)
                        else:score=torch.einsum('brd,btd->brt',F.normalize(query,dim=-1),F.normalize(key,dim=-1)).square()
                    if model.mode=='softmax12':alpha=score.masked_fill(~mask[:,None],-torch.inf).softmax(-1)
                    else:
                        raw=(score+1e-6)*mask[:,None];alpha=raw/raw.sum(-1,keepdim=True)
                    expected=model(h,roles,mask,conditioned=conditioned)
                    actual=(torch.einsum('brt,btd->brd',alpha,head.value(h))*head.decoder).sum(-1)+head.bias
                    parity=max(parity,float((expected-actual).abs().max()));assert torch.equal(expected,actual)
                    add(suffix+'/route_pair_tv',(alpha[:,ii]-alpha[:,jj]).abs().sum(-1)/2)
                    add(suffix+'/route_entropy',-(alpha*alpha.clamp_min(1e-30).log()).sum(-1))
                    routes.append(alpha)
                add('conditioning_route_change_tv',(routes[1]-routes[0]).abs().sum(-1)/2)
            report[kind]=dict(metrics={k:stats(np.concatenate(v)) for k,v in out.items()},max_forward_error=parity,checkpoint=pin)
            progress((mi+1)*128)
    result=dict(schema='q-rmea.query-diagnostic.v1',profile=cfg['profile'],selection='first 128 stored TRAIN rows; all 40 relations; no labels opened',rows=128,results=report,seconds=time.monotonic()-start,optimizer_updates=0,valid_rows=0,test_rows=0,cos_writes=0,limitations=['Selected checkpoints were previously selected on VALID.','Within-checkpoint conditioning contrast is diagnostic, not a retrained control or utility test.','No root-cause, generalization, novelty or L3/T conclusion from geometry alone.'])
    save(a.output/'summary.json',result);save(a.output/'RUN_COMPLETE',dict(summary_sha256=sha(a.output/'summary.json'),code_sha256=cfg['code_sha256']))
    print(json.dumps(dict(completed=True,seconds=result['seconds'],max_forward_error=max(v['max_forward_error'] for v in report.values()))))
if __name__=='__main__':main()
