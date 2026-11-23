"""Frozen TRAIN/valid4096 channel diagnostic, two allocated GPUs, no fitting."""
import argparse,hashlib,itertools,json,math,os,signal,sys,threading,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from correlation_order import channel_scores


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2),b''):h.update(part)
    return h.hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.pending');t.write_bytes((json.dumps(v,indent=2)+'\n').encode());t.replace(p)


def unit(circuit):
    device=torch.device('cuda:0');g=torch.Generator().manual_seed(2211)
    a=(torch.randn(2,3,2,3,2,generator=g,dtype=torch.float64)*.2).to(device)
    b=(torch.randn(4,2,3,2,generator=g,dtype=torch.float64)*.2).to(device)
    psi=circuit(a);phi=circuit(b);t=.05;v=channel_scores(psi,phi,t);lam=math.exp(-t)
    eye=torch.eye(2,device=device,dtype=torch.complex128)
    paulis=[eye,torch.tensor([[0,1],[1,0]],device=device,dtype=torch.complex128),torch.tensor([[0,-1j],[1j,0]],device=device,dtype=torch.complex128),torch.diag(torch.tensor([1,-1],device=device,dtype=torch.complex128))]
    rho=phi[:,:,None]*phi[:,None,:].conj();effect=torch.zeros_like(rho)
    for labels in itertools.product(range(4),repeat=3):
        P=torch.kron(torch.kron(paulis[labels[0]],paulis[labels[1]]),paulis[labels[2]])
        weight=math.prod((1+3*lam)/4 if i==0 else (1-lam)/4 for i in labels)
        effect+=weight*(P@rho@P.mH)
    direct=torch.einsum('bti,rij,btj->brt',psi.conj(),effect,psi).real
    exact_purity=torch.einsum('rij,rji->r',effect,effect).real
    control_purity=(1+7*v['kappa'].square())/8
    zero=channel_scores(psi,phi,0);eps=1e-6;small=channel_scores(psi,phi,eps)
    checks=dict(dense_channel_error=float((v['local']-direct).abs().max()),purity_match_error=float((exact_purity-control_purity).abs().max()),zero_error=max(float((zero[n]-zero['native']).abs().max()) for n in ('local','isotropic')),derivative_error=max(float(((small[n]-small['native'])/eps-small['derivative_'+n]).abs().max()) for n in ('local','isotropic')))
    assert max(checks[k] for k in ('dense_channel_error','purity_match_error','zero_error'))<1e-10 and checks['derivative_error']<1e-5,checks
    # Actual source circuit angle derivative, including phi-dependent purity calibration.
    b=b.detach().requires_grad_(True);gradients={}
    for kind in ('local','isotropic'):
        loss=channel_scores(psi,circuit(b),t)[kind].square().mean()
        grad=torch.autograd.grad(loss,b)[0];h=1e-5
        plus=b.detach().clone();minus=b.detach().clone();plus[0,0,0,0]+=h;minus[0,0,0,0]-=h
        fd=(channel_scores(psi,circuit(plus),t)[kind].square().mean()-channel_scores(psi,circuit(minus),t)[kind].square().mean())/(2*h)
        error=float((fd-grad[0,0,0,0]).abs());assert torch.isfinite(grad).all() and error<1e-7
        gradients[kind]=dict(finite=True,gradient_error=error,gradient_norm=float(grad.norm()))
    return dict(**checks,gradients=gradients)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','inputs','checkpoint','design','output','progress'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--attempt',required=True);a=p.parse_args();cfg=json.loads(a.design.read_bytes());signal.alarm(cfg['timeout_seconds'])
    assert not (a.output/'RUN_COMPLETE').exists();a.output.mkdir(parents=True,exist_ok=True)
    assert sha(a.checkpoint)==cfg['checkpoint_sha256'] and sha(a.inputs/'discovery/manifest.json')==cfg['native_manifest_sha256']
    for n,h in cfg['source_hashes'].items():assert sha(a.source/n)==h
    for n,h in cfg['code_hashes'].items():assert sha(Path(__file__).with_name(n))==h
    sys.path.insert(0,str(a.source));from baselines.retacred.rmea import RMEAHead,circuit,EPS
    assert torch.cuda.is_available() and torch.cuda.device_count()==2
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    checks=unit(circuit);lock=threading.Lock();counts={'train':0,'valid':0};started=time.time()
    def progress(split,n):
        with lock:
            counts[split]=n;now=time.time();done=sum(counts.values())
            save(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,phase='frozen_channel_diagnostic',current=done,total=8192,unit='example',observed_at=now,heartbeat_at=now,rate_per_second=done/max(now-started,1e-6),split_counts=dict(counts)))
    def worker(split,device_number):
        device=torch.device('cuda:'+str(device_number));torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device);head=RMEAHead('quantum').double().to(device).eval();head.load_state_dict(torch.load(a.checkpoint,map_location=device,weights_only=True),strict=True)
        cache=a.inputs/'discovery/cache'
        for name in ('h','roles','offsets','targets'):
            f=cache/(split+'-'+name+'.npy');pin=cfg['cache_hashes'][f.name]
            assert f.is_file() and not f.is_symlink() and f.stat().st_size==pin['bytes'] and sha(f)==pin['sha256']
        data={n:np.load(cache/(split+'-'+n+'.npy'),mmap_mode='r',allow_pickle=False) for n in ('h','roles','offsets','targets')}
        assert len(data['targets'])==cfg['split_totals'][split]
        size=cfg['count_per_split'];acc={n:dict(ce=0.,cm=np.zeros((40,40),dtype=np.int64)) for n in ('native','local','isotropic')};derivatives={n:0. for n in ('local','isotropic')};paired_sum=paired_sq=0.;max_reconstruction=max_purity=0.;probability_min=1.;native_parity=0.
        def metrics(row):
            cm=row['cm'];tp=np.diag(cm);den=cm.sum(0)+cm.sum(1);f=np.divide(2*tp,den,out=np.zeros(40,dtype=np.float64),where=den!=0)
            return dict(ce=row['ce']/size,positive39_macro_f1=float(f[1:].mean()),macro40=float(f.mean()),examples=int(cm.sum()))
        with torch.inference_mode():
            for start in range(0,size,cfg['batch_size']):
                end=min(start+cfg['batch_size'],size);indices=list(range(start,end));lengths=[int(data['offsets'][i+1]-data['offsets'][i]) for i in indices];length=max(lengths)
                h=torch.zeros(len(indices),length,1024,device=device,dtype=torch.float64);roles=torch.zeros(len(indices),length,4,device=device,dtype=torch.float64);mask=torch.zeros(len(indices),length,device=device,dtype=torch.bool)
                for j,(i,n) in enumerate(zip(indices,lengths)):
                    lo,hi=map(int,data['offsets'][i:i+2]);h[j,:n]=torch.from_numpy(np.array(data['h'][lo:hi],dtype=np.float64)).to(device);roles[j,:n]=torch.from_numpy(np.array(data['roles'][lo:hi],dtype=np.float64)).to(device);mask[j,:n]=True
                y=torch.from_numpy(np.array(data['targets'][start:end],dtype=np.int64)).to(device);assert ((y>=0)&(y<40)).all()
                d=head(h,roles,mask,details=True);v=channel_scores(d['psi'],d['phi'],cfg['time']);losses={};native_normalizer=None
                for kind in ('native','local','isotropic'):
                    q=v[kind];assert torch.isfinite(q).all() and q.min()>-1e-10 and q.max()<1+1e-10
                    raw=(q+EPS)*mask[:,None];normalizer=raw.sum(-1,keepdim=True);alpha=raw/normalizer;logits=(alpha*d['utilities']).sum(-1)+head.bias
                    if kind=='native':
                        native_parity=max(native_parity,float((logits-d['logits']).abs().max()));native_normalizer=normalizer
                    loss=torch.nn.functional.cross_entropy(logits,y,reduction='none');losses[kind]=loss
                    acc[kind]['ce']+=float(loss.sum());pred=logits.argmax(-1);np.add.at(acc[kind]['cm'],(y.cpu().numpy(),pred.cpu().numpy()),1)
                    probability_min=min(probability_min,float(q.min()))
                base_logits=d['logits'];g=base_logits.softmax(-1);g[torch.arange(len(y),device=device),y]-=1
                centred=d['utilities']-(d['alpha']*d['utilities']).sum(-1,keepdim=True)
                for kind in derivatives:
                    dl=(v['derivative_'+kind]*mask[:,None]*centred/native_normalizer).sum(-1)
                    derivatives[kind]+=float((g*dl).sum())
                delta=losses['local']-losses['isotropic'];paired_sum+=float(delta.sum());paired_sq+=float(delta.square().sum())
                max_reconstruction=max(max_reconstruction,float(((1+v['order_terms'].sum(-1))/8-v['native']).abs().max()));max_purity=max(max_purity,float(v['purity_error']));progress(split,end)
        torch.cuda.synchronize(device);assert native_parity<1e-10 and max_reconstruction<1e-10 and max_purity<1e-10
        result=dict(split=split,count=size,metrics={k:metrics(v) for k,v in acc.items()},native_parity_error=native_parity,pauli_reconstruction_error=max_reconstruction,purity_match_error=max_purity,probability_min=probability_min,ce_direction_at_zero={k:v/size for k,v in derivatives.items()},paired_local_minus_isotropic_ce=paired_sum/size,paired_ce_standard_error=math.sqrt(max(0,paired_sq/size-(paired_sum/size)**2)/size),peak_torch_bytes=torch.cuda.max_memory_allocated(device),physical_visible_device=device_number,seconds=time.time()-started)
        save(a.output/(split+'-summary.json'),result);return result
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={s:pool.submit(worker,s,i) for i,s in enumerate(('train','valid'))};results={s:f.result() for s,f in futures.items()}
    valid=results['valid'];gate=valid['paired_local_minus_isotropic_ce']<=-.001 and valid['metrics']['local']['positive39_macro_f1']>=valid['metrics']['native']['positive39_macro_f1']-.01
    summary=dict(status='complete',profile=cfg['profile'],design_sha256=sha(a.design),unit=checks,results=results,screen_gate=gate,L5='NOT_ACCEPTED',scope='Frozen-model exposed TRAIN/valid diagnostic; no fit, independent confirmation, novelty, whole-family failure or automatic full-run admission.',seconds=time.time()-started)
    save(a.output/'summary.json',summary);save(a.output/'RUN_COMPLETE',dict(profile=cfg['profile'],attempt=a.attempt,design_sha256=sha(a.design)));print(json.dumps(dict(status='complete',screen_gate=gate,results=results,unit=checks)))
if __name__=='__main__':main()
