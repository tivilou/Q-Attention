"""Changed optical mathematics/model checks; no heldout inference or lifecycle matrix."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np

def operator_fixtures():
    def displacement(z,n):
        a=np.diag(np.sqrt(np.arange(1,n)),1);g=z*a.T-z.conjugate()*a
        eigen,vectors=np.linalg.eigh(1j*g)
        return (vectors*np.exp(-1j*eigen))@vectors.conj().T
    errors=[]
    for z,w in ((.1+.2j,-.3+.15j),(.5+0j,-.5+0j),(.65-.4j,-.15+.5j)):
        f=lambda d:np.exp(-d)*(1-d)**2
        expected=f(abs(z-w)**2);parity_expected=.5*(expected+f(abs(z+w)**2))
        for n in (32,48):
            a=displacement(z,n)[:,1];b=displacement(w,n)[:,1]
            rho=np.outer(a,a.conj());sigma=np.outer(b,b.conj())
            signs=(-1.)**np.arange(n);twirl=lambda x:.5*(x+signs[:,None]*x*signs[None])
            observed=abs(np.vdot(a,b))**2
            parity_observed=np.trace(twirl(rho)@twirl(sigma)).real
            errors.extend((abs(observed-expected),abs(parity_observed-parity_expected)))
    assert max(errors)<3e-13
    return dict(max_operator_error=float(max(errors)),cutoffs=[32,48],complex_point_pairs=3,
        scope='Pointwise operator/parity witness only; production uses exact closed formula, no cutoff')

def gradient_fixtures():
    import torch
    from baselines.retacred.rmea_optical_fock import optical_probability,one_photon
    rng=np.random.default_rng(23);k=rng.normal(0,.7,(1,2,12));q=rng.normal(0,.7,(40,12))
    weights=rng.normal(0,1,(1,40,2));errors={}
    # Independent analytic chain rule in NumPy, including parity's sum branch.
    def analytic(kind):
        kk=k.reshape(1,2,6,2);qq=q.reshape(40,6,2)
        minus=kk[:,None]-qq[None,:,None];plus=kk[:,None]+qq[None,:,None]
        dm=(minus**2).sum(-1)/24;dp=(plus**2).sum(-1)/24
        f=lambda d:np.exp(-d)*(1-d)**2
        df=lambda d:np.exp(-d)*(d-1)*(3-d)
        if kind in ('vacuum_born','gaussian3_born'):
            slope=1 if kind=='vacuum_born' else 3;p=np.exp(-slope*dm.sum(-1))
            derivative=-slope*p[...,None,None]*minus/12
            dk=(weights[...,None,None]*derivative).sum(1)
            dq=-(weights[...,None,None]*derivative).sum((0,2))
        else:
            factors=f(dm) if kind=='fock1' else .5*(f(dm)+f(dp))
            gk=np.zeros_like(minus);gq=np.zeros_like(minus)
            for m in range(6):
                other=np.prod(np.delete(factors,m,axis=-1),axis=-1)
                gm=df(dm[...,m])[...,None]*minus[...,m,:]/12
                gp=df(dp[...,m])[...,None]*plus[...,m,:]/12
                gk[...,m,:]=other[...,None]*(gm if kind=='fock1' else .5*(gm+gp))
                gq[...,m,:]=other[...,None]*(-gm if kind=='fock1' else .5*(-gm+gp))
            p=factors.prod(-1);dk=(weights[...,None,None]*gk).sum(1);dq=(weights[...,None,None]*gq).sum((0,2))
        return p,dk.reshape(k.shape),dq.reshape(q.shape)
    for kind in ('fock1','parity_fock1','vacuum_born','gaussian3_born'):
        a=torch.tensor(k,requires_grad=True);b=torch.tensor(q,requires_grad=True)
        p=optical_probability(a,b,kind);(p*torch.tensor(weights)).sum().backward()
        ep,ek,eq=analytic(kind)
        error=max(np.max(abs(p.detach().numpy()-ep)),np.max(abs(a.grad.numpy()-ek)),np.max(abs(b.grad.numpy()-eq)))
        assert error<2e-12;errors[kind]=float(error)
    d=torch.tensor([0.,1.,10000.],dtype=torch.float64,requires_grad=True)
    v=one_photon(d);v.sum().backward()
    assert torch.isfinite(v).all() and torch.isfinite(d.grad).all() and v[1]==0 and d.grad[1]==0
    assert abs(float(d.grad[0])+3)<1e-14
    expected=np.exp(-.5*((k[:,None]-q[None,:,None])**2).mean(-1))
    actual=optical_probability(torch.tensor(k),torch.tensor(q),'vacuum_born').numpy()
    assert np.max(abs(expected-actual))<3e-16
    return dict(analytic_gradient_errors=errors,vacuum_rbf_identity_error=float(np.max(abs(expected-actual))),
        node_and_large_distance_finite=True,fock_origin_slope=-3)

def main():
    if sys.flags.optimize:raise ValueError('Technical checks require assertions enabled')
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--inputs',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(exist_ok=True)
    assert not (args.output/'technical-started.json').exists(),'Retain first execution record'
    sys.path.insert(0,str(args.source))
    from baselines.retacred import rmea_optical_fock_protocol as p
    start=time.time();p.atomic_json(args.output/'technical-started.json',dict(time=start,source_sha256=p.identity(p.sources()),contract_sha256=p.identity(p.contract())))
    cfg=p.config('discovery');cache=args.inputs/'cache';manifest=args.inputs/'manifest.json'
    assert p.digest(manifest)==p.BASE_MANIFEST
    pins=p.read(manifest)['cache']
    def pinned(name):
        f=cache/name;pin=pins[name]
        assert not f.is_symlink() and f.is_file() and f.stat().st_size==pin['bytes'] and p.digest(f)==pin['sha256']
        return f
    import torch
    from torch.nn import functional as F
    from baselines.retacred.rmea import RMEAHead
    from baselines.retacred.rmea_optical_fock import initialize_head,validate_initialized_head
    from baselines.retacred.rmea_matched_routers import initialize_head as native_initialize
    from baselines.retacred.rmea_frozen_training import FrozenFeatureTraining
    from baselines.retacred import rmea_optical_fock_runtime as runtime
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    math=dict(operator=operator_fixtures(),gradients=gradient_fixtures())
    torch.manual_seed(13);initial=RMEAHead('quantum').float().state_dict()
    roles=np.load(pinned('train-roles.npy'),mmap_mode='r',allow_pickle=False)
    offsets=np.load(pinned('train-offsets.npy'),mmap_mode='r',allow_pickle=False)
    features=np.load(pinned('train-h.npy'),mmap_mode='r',allow_pickle=False)
    targets=np.load(pinned('train-targets.npy'),mmap_mode='r',allow_pickle=False)
    lengths=np.diff(offsets[:9]);n=int(max(lengths))
    h=np.zeros((8,n,1024),np.float32);r=np.zeros((8,n,4),np.float32);mask=np.zeros((8,n),bool)
    for i,length in enumerate(lengths):
        a,b=map(int,offsets[i:i+2]);h[i,:length]=features[a:b];r[i,:length]=roles[a:b];mask[i,:length]=True
    x=tuple(torch.from_numpy(a).to('cuda:0') for a in (h,r,mask));y=torch.tensor(np.array(targets[:8]),device='cuda:0',dtype=torch.long)
    rows={};init_states=[]
    for kind in p.MODES:
        model=initialize_head(kind,initial,cfg);validate_initialized_head(model,kind,initial,cfg);model.cuda()
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        if init_states:assert all(torch.equal(v,init_states[0][k]) for k,v in before.items())
        else:init_states.append(before)
        assert sum(v.numel() for v in model.parameters())==81028
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],betas=tuple(cfg['betas']),eps=cfg['optimizer_eps'],weight_decay=cfg['weight_decay'])
        z=model(*x);assert torch.isfinite(z).all()
        padded=tuple(F.pad(a,(0,0,0,3)) if a.ndim==3 else F.pad(a,(0,3)) for a in x)
        error=float((z-model(*padded)).abs().max());assert error<2e-5
        if kind=='vacuum_born':
            # Normalization matches RBF softmax only when the declared probability floor is absent.
            a=model.head;key=a.key(torch.cat(x[:2],-1));expected=-.5*(key[:,None]-a.query[None,:,None]).square().mean(-1)
            from baselines.retacred.rmea_optical_fock import optical_probability
            prob=optical_probability(key,a.query,kind)
            assert torch.max(abs(prob-torch.exp(expected)))<2e-7
        loss=F.cross_entropy(z,y);assert torch.isfinite(loss)
        loss.backward();assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_norm']);optimizer.step()
        changed=any(not torch.equal(v,before[k]) for k,v in model.state_dict().items());assert changed
        rows[kind]=dict(parameters=81028,padding_error=error,initial_train_ce=float(loss.detach()),
            finite_gradients=True,real_train_updates=1,state_changed=True)
        del model,optimizer,z,loss,before
    row=dict(gate=True,source_sha256=p.identity(p.sources()),contract_sha256=p.identity(p.contract()),config_sha256=p.identity(cfg),
        input_manifest_sha256=p.digest(manifest),math=math,conditions=rows,test_model_rows=0,test_target_reads=0,real_train_updates_per_condition=1,
        environment=FrozenFeatureTraining(p,runtime).execution_environment(),seconds=time.time()-start,
        scope='Changed formulas/model only; no VALID/TEST reads/inference, utility gate or lifecycle recertification')
    p.atomic_json(args.output/'unit-validation.json',row)
    p.atomic_json(args.output/'RUN_COMPLETE',dict(artifacts={'unit-validation.json':p.digest(args.output/'unit-validation.json')}))
    print(json.dumps(dict(gate=True,conditions=list(rows),seconds=row['seconds'],math=math,test_model_rows=0)))

if __name__=='__main__':main()
