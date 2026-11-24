"""One full TRAIN reciprocal context-only transfer, no fitting or saved arrays."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,copy,hashlib,json,signal,sys,threading,time
import numpy as np

KINDS=('fock1','gaussian3_born','real12')
PAIRS=(('gaussian3_born','fock1'),('real12','fock1'),('fock1','gaussian3_born'),('fock1','real12'))
NAMES=KINDS+tuple(h+'<-'+d for h,d in PAIRS)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def save(p,v):
    t=p.with_suffix('.pending');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def metrics(y,z):
    q=z.argmax(-1);cm=np.bincount(40*y+q,minlength=1600).reshape(40,40)
    den=cm.sum(0)+cm.sum(1);f=np.divide(2*np.diag(cm),den,out=np.zeros(40),where=den!=0)
    x=z.astype(np.float64);x-=x.max(-1,keepdims=True)
    return dict(confusion_matrix=cm.tolist(),all40_macro_f1=float(f.mean()),ce=float((np.log(np.exp(x).sum(-1))-x[np.arange(len(y)),y]).mean()),correct=int((q==y).sum()))
def transfer(h,d,c):
    ch=(h*c[:,None]).sum(-1,keepdim=True);cd=(d*c[:,None]).sum(-1,keepdim=True)
    assert bool(((cd>0)|~c.any(-1)[:,None,None]).all())
    scale=ch/cd.where(cd>0,cd.new_ones(()))
    active=c & (c.sum(-1,keepdim=True)>=2)
    return (d*scale).where(active[:,None],h)
def main():
    ap=argparse.ArgumentParser()
    for n in ('root','design','output','progress'):ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--attempt',required=True);a=ap.parse_args();cfg=json.loads(a.design.read_bytes());start=time.monotonic()
    signal.alarm(cfg['timeout_seconds'])
    source=a.root/cfg['source'];cache=a.root/cfg['completed']/'cache'
    for n,h in cfg['code_hashes'].items():assert digest(Path(__file__).parent/n)==h
    for n,h in cfg['source_hashes'].items():assert digest(source/n)==h,n
    for n,v in cfg['train_cache'].items():assert digest(cache/n)==v['sha256'],n
    sys.path.insert(0,str(source));import torch
    from torch.nn import functional as F
    from baselines.retacred.rmea import RMEAHead
    from baselines.retacred.rmea_optical_fock import initialize_head,optical_probability
    from baselines.retacred.rmea_matched_routers import initialize_head as native_initialize
    from baselines.retacred.rmea_token_transport_study import state_identity
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.device_count()==2 and not a.output.exists();a.output.mkdir(parents=True)
    data={k:np.load(cache/('train-'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('h','roles','offsets','targets')}
    y=np.array(data['targets']);n=len(y);assert n==58465 and data['offsets'].shape==(n+1,) and data['offsets'][0]==0
    assert data['offsets'][-1]==len(data['h'])==len(data['roles']) and y.min()==0 and y.max()==39
    selected={}
    for k,pin in cfg['selected'].items():
        p=a.root/pin['path'];assert digest(p)==pin['sha256'];s=torch.load(p,map_location='cpu',weights_only=True)
        assert state_identity(s)==pin['state_sha256'];selected[k]=s
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(13);initial=RMEAHead('quantum').float().state_dict()
        templates={k:(native_initialize(k,initial,cfg['model_config']) if k=='real12' else initialize_head(k,initial,cfg['model_config'])) for k in KINDS}
    for k,m in templates.items():m.load_state_dict(selected[k],strict=True);m.eval()
    # Explicitly cover empty/singleton contexts and padding, independent of their natural frequency.
    gen=torch.Generator().manual_seed(20261125)
    valid=torch.tensor([[1,1,1,1,1],[1,1,0,0,0],[1,1,1,0,0]],dtype=torch.bool)
    contexts=torch.tensor([[0,0,1,1,1],[0,0,0,0,0],[0,0,1,0,0]],dtype=torch.bool)
    aa=(torch.rand((3,40,5),generator=gen,dtype=torch.float64)+.01)*valid[:,None]
    bb=(torch.rand((3,40,5),generator=gen,dtype=torch.float64)+.01)*valid[:,None]
    aa/=aa.sum(-1,keepdim=True);bb/=bb.sum(-1,keepdim=True);mixed=transfer(aa,bb,contexts)
    assert torch.equal(mixed[1:],aa[1:])
    assert float(((mixed-aa)*(~contexts)[:,None]).abs().max())==0
    assert float((mixed.sum(-1)-1).abs().max())<1e-14
    assert float((transfer(aa,aa,contexts)-aa).abs().max())<1e-14
    z={precision:{k:np.empty((n,40),np.float64 if precision=='float64' else np.float32) for k in NAMES} for precision in ('float32','float64')}
    cursors=[0,0];lock=threading.Lock()
    def progress(lane,value):
        with lock:
            cursors[lane]=value;save(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,phase='frozen_context_transfer',current=sum(cursors),total=n,unit='example',heartbeat_at=time.time(),observed_at=time.time()))
    def worker(lane,lo,hi):
        torch.cuda.set_device(lane);device=torch.device('cuda',lane)
        models={k:copy.deepcopy(m).to(device).eval() for k,m in templates.items()}
        errors=dict(source_forward=0.,numpy_transfer=0.,numpy_weights=0.,self_transfer=0.,entity_weight=0.,context_mass=0.,empty_singleton=0.)
        counts=dict(empty=0,singleton=0,entity_tokens=0)
        with torch.no_grad():
            for left in range(lo,hi,128):
                right=min(left+128,hi);lengths=np.diff(data['offsets'][left:right+1]);width=int(lengths.max())
                hh=np.zeros((right-left,width,1024),np.float32);rr=np.zeros((right-left,width,4),np.float32);mm=np.zeros((right-left,width),bool)
                for j,i in enumerate(range(left,right)):
                    b,e=map(int,data['offsets'][i:i+2]);size=e-b;hh[j,:size]=data['h'][b:e];rr[j,:size]=data['roles'][b:e];mm[j,:size]=True
                h,r,mask=(torch.from_numpy(v).to(device) for v in (hh,rr,mm))
                assert bool(((r[...,2]==0)|(r[...,2]==1)).all() and ((r[...,3]==0)|(r[...,3]==1)).all())
                c=mask & (r[...,2]==0) & (r[...,3]==0);entities=mask&~c;cnt=c.sum(-1)
                counts['empty']+=int((cnt==0).sum());counts['singleton']+=int((cnt==1).sum());counts['entity_tokens']+=int(entities.sum())
                records={}
                for k,m in models.items():
                    block=m.head;key=block.key(torch.cat((h,r),-1))
                    if k=='real12':raw=(torch.einsum('btd,rd->brt',F.normalize(key,dim=-1),F.normalize(block.query,dim=-1)).square()+1e-6)*mask[:,None]
                    else:raw=(optical_probability(key,block.query,k)+1e-6)*mask[:,None]
                    alpha=raw/raw.sum(-1,keepdim=True);value=block.value(h)
                    assert bool(torch.isfinite(alpha).all() and torch.isfinite(value).all())
                    records[k]=(alpha,value,block.decoder,block.bias)
                    actual=(torch.einsum('brt,btd->brd',alpha,value)*block.decoder).sum(-1)+block.bias
                    if left==lo:
                        error=float((actual-m(h,r,mask)).abs().max());assert np.isfinite(error)
                        errors['source_forward']=max(errors['source_forward'],error)
                    self_alpha=transfer(alpha,alpha,c)
                    errors['self_transfer']=max(errors['self_transfer'],float((self_alpha-alpha).abs().max()))
                for precision,dtype in [('float32',torch.float32),('float64',torch.float64)]:
                    converted={k:tuple(v.to(dtype) for v in record) for k,record in records.items()}
                    for k,(alpha,value,decoder,bias) in converted.items():
                        out=(torch.einsum('brt,btd->brd',alpha,value)*decoder).sum(-1)+bias;z[precision][k][left:right]=out.cpu().numpy()
                    for receiver,donor in PAIRS:
                        alpha,value,decoder,bias=converted[receiver];other=converted[donor][0];mixed=transfer(alpha,other,c)
                        entity_error=((mixed-alpha)*entities[:,None]).abs().max();mass_error=((mixed-alpha)*c[:,None]).sum(-1).abs().max()
                        errors['entity_weight']=max(errors['entity_weight'],float(entity_error));errors['context_mass']=max(errors['context_mass'],float(mass_error))
                        identity=(cnt<=1)
                        if bool(identity.any()):errors['empty_singleton']=max(errors['empty_singleton'],float((mixed[identity]-alpha[identity]).abs().max()))
                        out=(torch.einsum('brt,btd->brd',mixed,value)*decoder).sum(-1)+bias
                        assert bool(torch.isfinite(out).all());z[precision][receiver+'<-'+donor][left:right]=out.cpu().numpy()
                        if left==lo and precision=='float64':
                            ah=alpha[:3].cpu().numpy();ad=other[:3].cpu().numpy();cc=c[:3].cpu().numpy();reference=ah.copy()
                            for bi in range(len(ah)):
                                for ri in range(40):
                                    keep=np.flatnonzero(cc[bi]);sh=sum(float(ah[bi,ri,t]) for t in keep);sd=sum(float(ad[bi,ri,t]) for t in keep)
                                    if len(keep)>=2:
                                        for ti in keep:reference[bi,ri,ti]=sh*float(ad[bi,ri,ti])/sd
                            weight_error=float(np.max(abs(reference-mixed[:3].cpu().numpy())));assert np.isfinite(weight_error)
                            errors['numpy_weights']=max(errors['numpy_weights'],weight_error)
                            vv=value[:3].cpu().numpy();dd=decoder.cpu().numpy();bb=bias.cpu().numpy()
                            zz=np.einsum('brt,btd,rd->br',reference,vv,dd)+bb
                            error=float(np.max(abs(zz-out[:3].cpu().numpy())));assert np.isfinite(error)
                            errors['numpy_transfer']=max(errors['numpy_transfer'],error)
                if (left-lo)%2048==0:progress(lane,right-lo)
        assert all(np.isfinite(v) for v in errors.values())
        assert errors['source_forward']<2e-5 and errors['numpy_transfer']<1e-10 and errors['numpy_weights']<1e-12
        assert errors['self_transfer']<2e-6 and errors['entity_weight']==0 and errors['context_mass']<2e-6 and errors['empty_singleton']==0
        for k,m in models.items():assert state_identity({name:v.cpu() for name,v in m.state_dict().items()})==cfg['selected'][k]['state_sha256']
        progress(lane,hi-lo);return dict(errors=errors,counts=counts)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(worker,0,0,n//2),pool.submit(worker,1,n//2,n)];checks=[job.result() for job in jobs]
    reports={precision:{k:metrics(y,x) for k,x in values.items()} for precision,values in z.items()}
    assert reports['float32']['real12']['confusion_matrix']==cfg['prior_train_metrics']['real12']['confusion_matrix']
    transfers={};gate=True
    for receiver,donor in PAIRS:
        name=receiver+'<-'+donor;sign=1 if donor=='fock1' else -1;row={}
        for precision in z:
            delta=reports[precision][name]['all40_macro_f1']-reports[precision][receiver]['all40_macro_f1'];good=z[precision][name].argmax(-1)==y;old=z[precision][receiver].argmax(-1)==y
            row[precision]=dict(macro_delta_hybrid_minus_receiver=delta,oriented_macro_delta=sign*delta,ce_delta=reports[precision][name]['ce']-reports[precision][receiver]['ce'],repaired=int((good&~old).sum()),harmed=int((~good&old).sum()))
            gate=gate and sign*delta>0
        transfers[name]=row
    report=dict(profile=cfg['profile'],rows=n,results=reports,transfers=transfers,four_direction_gate=bool(gate),seconds=time.monotonic()-start,optimizer_updates=0,new_valid=0,new_test=0,cos_writes=0,per_example_binary_outputs=0,selected=cfg['selected'],code_hashes=cfg['code_hashes'],claim='Frozen TRAIN context-position transfer only; no refitted/heldout utility, semantic cause or quantum attribution',L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED')
    save(a.output/'summary.json',report);save(a.output/'fixtures.json',dict(status='PASS',checks=checks))
    save(a.output/'RUN_COMPLETE',{name:digest(a.output/name) for name in ('summary.json','fixtures.json')});signal.alarm(0);print(json.dumps(dict(gate=gate,seconds=report['seconds'],transfers=transfers)))
if __name__=='__main__':main()
