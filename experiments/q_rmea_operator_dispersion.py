"""Frozen linear head: operator, outcome and uniform dispersion calibration.

No original model parameter is updated; this is a diagnostic of a prospective
value-feedback tangent, not an implementation of the finite-eta filter.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import q_rmea_query_alignment as qa

p = qa.p
ARMS = ('baseline', 'operator', 'outcome', 'uniform')
TRAIN, VALID = 58465, 19584
REVIEW = 'docs/evidence/rmea/20261122-operator-dispersion/review.json'
INPUT_RUN = '/hy-tmp/Q-Attention/worker/attempts/6a2bf35b8d9d4ac4a6b4fe2acb6bc116/output/20261121T034306Z_seed13/discovery'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def design():
    return dict(schema='q-rmea.operator-dispersion.design.v1', checkpoint='linear_coherent-final',
        counts=dict(train=TRAIN, valid=VALID), arms=list(ARMS), original_parameters_frozen=True,
        statistics='F_t=psi_t psi_t^H+eps I; C=sum F; A=sum uF; VQ=(Aphi)^H solve(C,Aphi)/Z-m^2; VC=E_p[u^2]-m^2; VU=uniform variance',
        query='Existing exact linear coherent eight-dimensional query rotation; eps=1e-6 per valid token',
        calibration=dict(objective='Full train40class CE, independent shared signed scalar per nonbaseline arm',
            bounds=[-1.,1.], solver='Monotone derivative; endpoints or scipy.brentq xtol=rtol=1e-12,maxiter100; compare zero and retain zero for improvement<=1e-12',
            boundary='Record bound hit; no expansion or validation selection'),
        primary='Operator-minus-each-of-baseline/outcome/uniform paired CE; all three upper percentile bounds<0 AND fitted operator scalar>0',
        bootstrap=dict(draws=10000, seed=20261122, unit='fixed validation sentence', quantile=1-.05/3,
            conditional='Fixed training fits; no training uncertainty or independent confirmation'),
        parity=dict(probability_atol=1e-9, logit_atol=1e-9, aggregate_atol=1e-10,
            coverage='All training and validation points: whitened eta0 original-interface probabilities/logits; first batch per split also original token-space route'),
        roundoff='Require raw0<=VQ<=VC and VU>=0 within1e-9*(1+abs(VC)+m^2); clip only certified floating violations within that tolerance',
        execution='One3GPU shared allocation; three disjoint train+valid shards; one CPU calibration/bootstrap parent',
        trace='First two valid units: complete relevant numeric arrays and independent numpy operator calculation; partial probe, not sample-trace.v2',
        recovery='Cheap restart from original protected inputs and Git. No new model checkpoint/COS. No automatic retry',
        scope='Diagnostic only. No finite-eta feedback execution, formal new method, quantum advantage, independent confirmation or L5 acceptance')


def hashes():
    return {n:p.digest(ROOT/n) for n in ('experiments/q_rmea_operator_dispersion.py',
        'experiments/q_rmea_query_alignment.py', 'baselines/retacred/rmea_token_transport.py')}


def reviewed():
    require(not sys.flags.optimize, 'Optimized Python is not admitted')
    r = p.read(ROOT/REVIEW)
    require(r.get('verdict') == 'PASS' and r.get('must_fix') == [] and r.get('review_scope') == 'both', 'Exact both-scope review required')
    require(r.get('hashes') == hashes() and r.get('design_sha256') == p.identity(design())
        and r.get('core_sha256') == p.identity(p.sources()), 'Reviewed source/design identity differs')
    require(not p.git('status','--porcelain'), 'Clean source required')


def statistics(psi, phi, utilities, mask, eps=1e-6):
    import torch
    b,t,d = psi.shape
    require(bool(torch.isfinite(psi).all() and torch.isfinite(phi).all() and torch.isfinite(utilities).all()), 'Nonfinite source tensors')
    require(phi.shape[:1] == (b,) and phi.shape[-1] == d, 'Bad effective query')
    require(utilities.shape == (b,phi.shape[1],t) and mask.shape == (b,t) and mask.dtype == torch.bool, 'Bad utility/mask')
    require(bool(mask.any(-1).all()) and eps >= 0, 'Empty token set or epsilon')
    s = psi.masked_fill(~mask[...,None],0)
    eye = torch.eye(d, dtype=psi.dtype, device=psi.device)
    counts = mask.sum(-1).to(psi.real.dtype)
    c = torch.einsum('bti,btj->bij',s,s.conj())+counts[:,None,None]*float(eps)*eye
    a = torch.einsum('bri,bti->brt',phi,s.conj())
    raw = (a.abs().square()+eps)*mask[:,None,:]
    z = raw.sum(-1)
    alpha = raw/z[...,None]
    u = utilities.masked_fill(~mask[:,None,:],0)
    mean = (alpha*u).sum(-1)
    ap = torch.einsum('brt,bti->bri',u*a,s)+eps*u.sum(-1)[...,None]*phi
    solution = torch.linalg.solve(c[:,None],ap[...,None]).squeeze(-1)
    q = (ap.conj()*solution).sum(-1).real/z-mean.square()
    classical = (alpha*u.square()).sum(-1)-mean.square()
    count = mask.sum(-1)[:,None]
    uniform = u.square().sum(-1)/count-(u.sum(-1)/count).square()
    tol = 1e-9*(1+classical.abs()+mean.square())
    require(bool(torch.isfinite(q).all() and torch.isfinite(classical).all() and torch.isfinite(uniform).all()), 'Nonfinite dispersion')
    require(bool(((q>=-tol)&(q<=classical+tol)&(classical>=-tol)&(uniform>=-tol)).all()), 'Operator variance bounds fail')
    # Full zero-filter round trip, with independent Hermitian roots, at every point.
    eigen, vectors = torch.linalg.eigh(c)
    require(bool((eigen>0).all()), 'Nonpositive frame')
    root = (vectors*eigen.sqrt()[:,None,:])@vectors.mH
    inverse_root = (vectors*eigen.rsqrt()[:,None,:])@vectors.mH
    state = torch.einsum('bij,brj->bri',root,phi)/z.sqrt()[...,None]
    zero_phi = torch.einsum('bij,brj->bri',inverse_root,state)
    zero_phi = zero_phi/zero_phi.norm(dim=-1,keepdim=True)
    zero_raw = (torch.einsum('bri,bti->brt',zero_phi,s.conj()).abs().square()+eps)*mask[:,None,:]
    zero_alpha = zero_raw/zero_raw.sum(-1,keepdim=True)
    probability_error = float((zero_alpha-alpha).abs().max())
    logit_error = float(((zero_alpha-alpha)*u).sum(-1).abs().max())
    require(probability_error <= 1e-9 and logit_error <= 1e-9, 'Zero-filter pointwise parity failed')
    clipped_c = classical.clamp_min(0)
    return dict(mean=mean, alpha=alpha, operator=torch.minimum(q.clamp_min(0),clipped_c),
        outcome=clipped_c, uniform=uniform.clamp_min(0), c=c, ap=ap, normalization=z,
        raw_operator=q, zero_alpha=zero_alpha, probability_error=probability_error, logit_error=logit_error,
        bound_roundoff=float(torch.maximum((-q).clamp_min(0),(q-classical).clamp_min(0)).max()))


def loss(z,y):
    import numpy as np
    from scipy.special import logsumexp
    return logsumexp(z,axis=1)-z[np.arange(len(y)),y]


def fit(z,v,y):
    import numpy as np
    from scipy.optimize import brentq
    def derivative(g):
        x=z+g*v; a=x-x.max(1,keepdims=True); w=np.exp(a); w/=w.sum(1,keepdims=True)
        return float(np.mean((w*v).sum(1)-v[np.arange(len(y)),y]))
    lower,upper = derivative(-1.),derivative(1.)
    if lower >= 0: gamma=-1.
    elif upper <= 0: gamma=1.
    else: gamma=float(brentq(derivative,-1.,1.,xtol=1e-12,rtol=1e-12,maxiter=100))
    baseline=float(loss(z,y).mean()); final=float(loss(z+gamma*v,y).mean())
    if baseline-final <= 1e-12: gamma=0.; final=baseline
    require(np.isfinite(final) and final <= baseline+1e-12, 'Scalar fit regressed')
    return dict(gamma=gamma, train_ce=final, baseline_train_ce=baseline,
        derivative=derivative(gamma), boundary_hit=abs(gamma)==1.)


def batches(data,start,stop):
    import numpy as np
    import torch
    for a in range(start,stop,32):
        ids=list(range(a,min(a+32,stop)));lengths=[int(data['offsets'][i+1]-data['offsets'][i]) for i in ids]
        h=np.zeros((len(ids),max(lengths),1024),np.float64); roles=np.zeros(h.shape[:2]+(4,),np.float64);mask=np.zeros(h.shape[:2],bool)
        for j,(i,n) in enumerate(zip(ids,lengths)):
            lo,hi=map(int,data['offsets'][i:i+2]);h[j,:n]=data['h'][lo:hi];roles[j,:n]=data['roles'][lo:hi];mask[j,:n]=True
        yield a,tuple(torch.from_numpy(x).to('cuda') for x in (h,roles,mask))


def event(out,phase,current,total,**kw):
    p.atomic_json(out/'progress.json',dict(schema='research.progress.v1',attempt=Path(os.environ['RESEARCH_OUTPUT_DIR']).parent.name,
        phase=phase,stage=phase,unit='sample',current=current,completed=current,total=total,observed_at=time.time(),
        batch_id='rmea-operator-dispersion-20261122',**kw))


def numpy_probe(psi,phi,u,mask,expected):
    import numpy as np
    errors=[]
    for i in range(len(psi)):
        s=psi[i][mask[i]];f=np.einsum('ti,tj->tij',s,s.conj())+1e-6*np.eye(8)[None]
        c=f.sum(0);eigen,vec=np.linalg.eigh(c);w=(vec*(eigen**-.5)[None])@vec.conj().T
        effects=np.einsum('ij,tjk,kl->til',w,f,w)
        for r in range(len(phi[i])):
            query=phi[i,r];den=(query.conj()@c@query).real
            state=((vec*np.sqrt(eigen)[None])@vec.conj().T)@query/np.sqrt(den)
            operator=np.einsum('t,tij->ij',u[i,r,mask[i]],effects)
            mean=(state.conj()@operator@state).real
            variance=(state.conj()@operator@operator@state).real-mean**2
            errors.append(abs(variance-expected[i,r]))
    require(max(errors,default=0)<1e-9, 'Independent numpy operator variance failed')
    return max(errors,default=0.)


def worker(args):
    import numpy as np
    import torch
    from baselines.retacred.rmea_token_transport import route
    import ctypes
    import signal
    import io
    from baselines.retacred.rmea import RMEAHead
    require(ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGKILL,0,0,0)==0,'Parent-death fencing failed')
    require(os.getppid()==args.parent_pid,'Parent identity changed')
    require(torch.cuda.device_count()==1 and bool(os.environ.get('CUDA_VISIBLE_DEVICES')),'One allocated GPU required')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    checkpoint=args.run/'checkpoints/linear_coherent-final.pt';blob=checkpoint.read_bytes()
    require(hashlib.sha256(blob).hexdigest()==p.read(args.output/'inputs.json')['checkpoint_sha256'],'Checkpoint bytes differ')
    head=RMEAHead('quantum').double();head.load_state_dict(torch.load(io.BytesIO(blob),map_location='cpu',weights_only=True))
    head=head.to('cuda').eval();out=args.output/('shard'+str(args.shard));out.mkdir()
    totals=[TRAIN,VALID];size=sum(n*(args.shard+1)//3-n*args.shard//3 for n in totals)
    done=0; errors=dict(zero_probability=0.,zero_logit=0.,original_route=0.,numpy_operator=0.,roundoff=0.)
    with torch.no_grad():
        for split,n in zip(('train','valid'),totals):
            start,stop=n*args.shard//3,n*(args.shard+1)//3;data=qa.dataset(args.run,split)
            arrays={k:[] for k in ('logits','operator','outcome','uniform')}
            for a,inputs in batches(data,start,stop):
                d=head(*inputs,details=True);g=qa.generator(d['psi'],inputs[2]);alpha,rotation=qa.distribution(d['psi'],d['phi'],inputs[2],g)
                phi=torch.einsum('bij,rj->bri',rotation,d['phi']);s=statistics(d['psi'],phi,d['utilities'],inputs[2]);z=s['mean']+head.bias
                require(torch.allclose(alpha,s['alpha'],atol=1e-12,rtol=1e-12),'Original eight-dimensional probabilities differ')
                if a==start:
                    original=route(d['psi'],d['phi'],inputs[2],kind='linear_coherent')['alpha']
                    error=float((alpha-original).abs().max());errors['original_route']=max(errors['original_route'],error)
                    require(error<2e-12,'Original token route differs')
                reference=(alpha*d['utilities']).sum(-1)+head.bias
                restored=(s['zero_alpha']*d['utilities']).sum(-1)+head.bias
                probability_error=float((s['zero_alpha']-alpha).abs().max());logit_error=float((restored-reference).abs().max())
                require(probability_error<=1e-9 and logit_error<=1e-9,'Pointwise frozen-interface zero-filter parity failed')
                errors['zero_probability']=max(errors['zero_probability'],probability_error);errors['zero_logit']=max(errors['zero_logit'],logit_error)
                errors['roundoff']=max(errors['roundoff'],s['bound_roundoff'])
                for k in arrays:arrays[k].append((z if k=='logits' else s[k]).cpu().numpy())
                if split=='valid' and args.shard==0 and a==0:
                    probe=dict(psi=d['psi'][:2].cpu().numpy(),phi=phi[:2].cpu().numpy(),utilities=d['utilities'][:2].cpu().numpy(),
                        mask=inputs[2][:2].cpu().numpy(),alpha=alpha[:2].cpu().numpy(),logits=z[:2].cpu().numpy(),
                        operator=s['operator'][:2].cpu().numpy(),outcome=s['outcome'][:2].cpu().numpy(),uniform=s['uniform'][:2].cpu().numpy())
                    errors['numpy_operator']=numpy_probe(probe['psi'],probe['phi'],probe['utilities'],probe['mask'],probe['operator'])
                    np.savez_compressed(out/'first2.npz',**probe)
                    p.atomic_json(out/'trace-metadata.json',dict(status='partial numeric probe, not sample-trace.v2',path='first2.npz',sha256=p.digest(out/'first2.npz'),
                        arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in probe.items()}))
                done+=len(inputs[0]);event(out,split,done,size,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'])
            for k,parts in arrays.items():np.save(out/(split+'-'+k+'.npy'),np.concatenate(parts),allow_pickle=False)
    files={x.name:p.digest(x) for x in out.iterdir() if x.suffix in ('.npy','.npz') or x.name=='trace-metadata.json'}
    row=dict(shard=args.shard,current=done,total=size,errors=errors,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],files=files)
    p.atomic_json(out/'summary.json',row);p.atomic_json(out/'WORKER_COMPLETE',dict(summary_sha256=p.digest(out/'summary.json')))


def report(out,phase,states):
    event(out,phase,sum(s.get('current',0) for s in states),TRAIN+VALID,jobs=states)


def unit(out):
    import numpy as np
    import torch
    require(not os.environ.get('CUDA_VISIBLE_DEVICES'),'CPU unit recipe required')
    out.mkdir(parents=True,exist_ok=True);qa.fresh_output(out)
    psi=torch.eye(2,dtype=torch.complex128)[None];phi=torch.tensor([[[1.,1.]]],dtype=torch.complex128)/2**.5
    u=torch.tensor([[[-1.,1.]]],dtype=torch.float64);mask=torch.ones(1,2,dtype=torch.bool)
    a=statistics(psi,phi,u,mask,eps=0.)
    require(abs(float(a['operator'])-1)<1e-12 and abs(float(a['outcome'])-1)<1e-12,'Projective equality fixture failed')
    psi=torch.tensor([[[1.,0.],[1.,0.]]],dtype=torch.complex128);phi=torch.tensor([[[1.,0.]]],dtype=torch.complex128)
    b=statistics(psi,phi,u,mask)
    require(abs(float(b['operator']))<1e-12 and abs(float(b['outcome'])-1)<1e-12,'Indistinguishable utility-noise fixture failed')
    z=np.zeros((2,2));y=np.array([0,1]);v=np.eye(2)
    require(fit(z,v,y)['gamma']==1 and fit(z,-v,y)['gamma']==-1 and fit(z,np.ones((2,2)),y)['gamma']==0,'Convex calibration fixtures failed')
    p.atomic_json(out/'unit-validation.json',dict(gate=True,fixtures=['projective_equality','identical_state_noise','zero_filter_roundtrip','positive_negative_flat_calibration']))
    p.atomic_json(out/'RUN_COMPLETE',dict(artifacts={'unit-validation.json':p.digest(out/'unit-validation.json')}))


def run(args):
    import numpy as np
    from baselines.retacred.orchestrate import gpu_ids
    reviewed();devices=gpu_ids('auto');require(len(devices)==3,'Exactly three allocated GPUs required')
    args.run=args.run.resolve();args.output=args.output.resolve()
    require(not(args.output.is_relative_to(ROOT) or args.output.is_relative_to(args.run) or args.run.is_relative_to(args.output)),'Independent output required')
    qa.fresh_output(args.output);started=time.time();p.atomic_json(args.output/'design.json',design())
    inputs=qa.validate_inputs(args.run)
    target=args.run/'cache/train-targets.npy';m=p.read(args.run/'manifest.json')
    require(not target.is_symlink() and p.digest(target)==m['cache']['train-targets.npy']['sha256'],'Training label identity differs')
    inputs['train_targets']=m['cache']['train-targets.npy'];p.atomic_json(args.output/'inputs.json',inputs)
    jobs=[('shard'+str(i),[sys.executable,'-B',str(Path(__file__).resolve()),'worker','--run',str(args.run),
        '--output',str(args.output),'--parent-pid',str(os.getpid()),'--shard',str(i)],g) for i,g in enumerate(devices)]
    qa.execute(jobs,args.output,'statistics',report=report)
    shards=[]
    for i in range(3):
        d=args.output/('shard'+str(i));r=p.read(d/'summary.json')
        require(p.read(d/'WORKER_COMPLETE')['summary_sha256']==p.digest(d/'summary.json'),'Shard seal differs')
        require(all(p.digest(d/n)==h for n,h in r['files'].items()),'Shard arrays differ');shards.append(r)
    data={s:{k:np.concatenate([np.load(args.output/('shard'+str(i))/(s+'-'+k+'.npy'),allow_pickle=False) for i in range(3)])
        for k in ('logits','operator','outcome','uniform')} for s in ('train','valid')}
    labels={'train':np.load(args.run/'cache/train-targets.npy',allow_pickle=False)}
    require(labels['train'].shape==(TRAIN,),'Training target coverage differs')
    for s,n in [('train',TRAIN),('valid',VALID)]:
        require(all(x.shape==(n,40) and np.isfinite(x).all() for x in data[s].values()),'Full cohort coverage failed')
    calibration={a:fit(data['train']['logits'],data['train'][a],labels['train']) for a in ARMS[1:]}
    p.atomic_json(args.output/'calibration.json',calibration)
    labels['valid']=np.load(args.run/'cache/valid-targets.npy',allow_pickle=False)
    require(labels['valid'].shape==(VALID,),'Validation target coverage differs')
    rows={};losses={};base=data['valid']['logits'];y=labels['valid']
    for a in ARMS:
        z=base if a=='baseline' else base+calibration[a]['gamma']*data['valid'][a]
        losses[a]=loss(z,y);rows[a]=dict(ce=float(losses[a].mean()),positive39_macro_f1=qa.macro(y,z.argmax(1)),count=VALID)
    old=p.read(args.run/'summary.json')['results']['linear_coherent']['valid']
    require(abs(rows['baseline']['ce']-old['ce'])<1e-10 and abs(rows['baseline']['positive39_macro_f1']-old['positive39_macro_f1'])<1e-10,'Frozen original validation does not replay')
    contrasts={a:losses['operator']-losses[a] for a in ('baseline','outcome','uniform')}
    draws={a:[] for a in contrasts};rng=np.random.default_rng(20261122)
    for i in range(10000):
        ids=rng.integers(0,VALID,size=VALID)
        for a,delta in contrasts.items():draws[a].append(float(delta[ids].mean()))
        if i%100==0:event(args.output,'paired_bootstrap',TRAIN+VALID,TRAIN+VALID,resamples=i)
    bounds={a:dict(delta=float(v.mean()),onesided_upper=float(np.quantile(draws[a],1-.05/3))) for a,v in contrasts.items()}
    gate=calibration['operator']['gamma']>0 and all(v['onesided_upper']<0 for v in bounds.values())
    trace=p.read(args.output/'shard0/trace-metadata.json');trace['path']='shard0/first2.npz'
    require(p.digest(args.output/trace['path'])==trace['sha256'],'Root trace payload differs')
    p.atomic_json(args.output/'trace-metadata.json',trace)
    result=dict(schema='q-rmea.operator-dispersion.result.v1',gate=gate,results=rows,calibration=calibration,contrasts=bounds,
        source=p.git('rev-parse','HEAD'),source_hashes=hashes(),core_sha256=p.identity(p.sources()),design_sha256=p.identity(design()),
        seconds=time.time()-started,shards=shards,scope='Frozen scalar readout diagnostic, NOT finite-eta routing or independent evidence/L5')
    p.atomic_json(args.output/'summary.json',result);event(args.output,'complete',TRAIN+VALID,TRAIN+VALID)
    names=('design.json','inputs.json','calibration.json','summary.json','trace-metadata.json')
    p.atomic_json(args.output/'RUN_COMPLETE',dict(artifacts={n:p.digest(args.output/n) for n in names}))
    print(json.dumps(dict(gate=gate,seconds=result['seconds'])),flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('action',choices=['unit','run','worker']);a.add_argument('--run',type=Path,default=Path(INPUT_RUN))
    a.add_argument('--output',type=Path,required=True);a.add_argument('--parent-pid',type=int);a.add_argument('--shard',type=int,choices=range(3));args=a.parse_args()
    if args.action=='worker':reviewed();worker(args)
    elif args.action=='unit':reviewed();unit(args.output)
    else:run(args)

if __name__=='__main__':main()
