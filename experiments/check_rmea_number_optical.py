from pathlib import Path
import argparse,json,sys,time
"""Independent operator and NumPy chain-rule checks for the new kernel only."""
import math
import numpy as np


def operator_fixtures():
    def displacement(z, n):
        a = np.diag(np.sqrt(np.arange(1, n)), 1)
        values, vectors = np.linalg.eigh(1j * (z * a.T - z.conjugate() * a))
        return (vectors * np.exp(-1j * values)) @ vectors.conj().T
    errors = []
    import torch
    from baselines.retacred.rmea_number_optical import mode_probability
    for z, w in ((.1+.2j,-.3+.15j), (.5+0j,-.5+0j), (.65-.4j,-.15+.5j), (2**.5+0j,0j), (1+1j,0j)):
        delta = z-w
        for n in (64,80):
            dz, dw = displacement(z,n), displacement(w,n)
            chi_z=(dz[:,0]+dz[:,2])/math.sqrt(2)
            chi_w=(dw[:,0]+dw[:,2])/math.sqrt(2)
            rho=(np.outer(dz[:,0],dz[:,0].conj())+np.outer(dz[:,2],dz[:,2].conj()))/2
            sigma=(np.outer(dw[:,0],dw[:,0].conj())+np.outer(dw[:,2],dw[:,2].conj()))/2
            coherent=abs(np.vdot(chi_z,chi_w))**2
            mixed=np.trace(rho@sigma).real
            point=torch.tensor([delta.real,delta.imag],dtype=torch.float64)
            for kind, expected in (('number_coherent02',coherent),('number_mixed02',mixed),
                                   ('number_mixed02_purity_normalized',2*mixed)):
                errors.append(abs(float(mode_probability(point,kind))-expected))
    assert max(errors)<3e-13
    return dict(max_operator_error=float(max(errors)),cutoffs=[64,80],complex_pairs=5,
                scope='Pointwise displaced pure/density operators; production uses exact analytic expressions')


def gradient_fixtures():
    import torch
    from baselines.retacred.rmea_number_optical import optical_probability,mode_probability,MODES
    rng=np.random.default_rng(23)
    k=rng.normal(0,1.7,(1,2,12));q=rng.normal(0,1.7,(40,12));weights=rng.normal(size=(1,40,2))
    delta=(k.reshape(1,2,6,2)[:,None]-q.reshape(40,6,2)[None,:,None])/math.sqrt(24)
    x,y=delta[...,0],delta[...,1];s=x*x+y*y;errors={}
    for kind in MODES:
        if kind=='number_coherent02':
            f=1-s+s*s/4+(x*x-y*y)/math.sqrt(2)
            fx=(-2+s+math.sqrt(2))*x;fy=(-2+s-math.sqrt(2))*y
            factor=np.exp(-s)*f*f
            gx=np.exp(-s)*(2*f*fx-2*x*f*f);gy=np.exp(-s)*(2*f*fy-2*y*f*f)
        elif kind.startswith('number_mixed02'):
            l=1-2*s+s*s/2;h=(1+l*l+s*s)/4;dh=(2*l*(-2+s)+2*s)/4
            scale=2 if kind.endswith('purity_normalized') else 1
            factor=scale*np.exp(-s)*h
            gx=scale*np.exp(-s)*2*x*(dh-h);gy=scale*np.exp(-s)*2*y*(dh-h)
        else:
            ax,ay=3-math.sqrt(2),3+math.sqrt(2)
            factor=np.exp(-ax*x*x-ay*y*y);gx=-2*ax*x*factor;gy=-2*ay*y*factor
        dk=np.zeros_like(delta)
        for m in range(6):
            other=np.prod(np.delete(factor,m,axis=-1),axis=-1)
            dk[...,m,0]=other*gx[...,m]/math.sqrt(24)
            dk[...,m,1]=other*gy[...,m]/math.sqrt(24)
        expected_k=(weights[...,None,None]*dk).sum(1).reshape(k.shape)
        expected_q=-(weights[...,None,None]*dk).sum((0,2)).reshape(q.shape)
        a=torch.tensor(k,requires_grad=True);b=torch.tensor(q,requires_grad=True)
        p=optical_probability(a,b,kind);(p*torch.tensor(weights)).sum().backward()
        error=max(np.max(abs(p.detach().numpy()-factor.prod(-1))),
                  np.max(abs(a.grad.numpy()-expected_k)),np.max(abs(b.grad.numpy()-expected_q)))
        assert error<2e-12;errors[kind]=float(error)
        # Exact kernel node, zero distance, and a large finite distance. Both
        # first derivatives and probabilities must stay finite in FP32/FP64.
        for dtype in (torch.float32,torch.float64):
            points=torch.tensor([[0.,0.],[1.,1.],[100.,100.]],dtype=dtype,requires_grad=True)
            v=mode_probability(points,kind);v.sum().backward()
            assert torch.isfinite(v).all() and torch.isfinite(points.grad).all()
            assert (v>=0).all() and (v<=1+1e-6).all()
    return dict(analytic_gradient_errors=errors,all_four_node_origin_large_distance_finite=True,
                purity_scale_not_normalization_invariant_with_EPS=True)

def main():
    if sys.flags.optimize:raise ValueError('Technical checks require assertions enabled')
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--inputs',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(exist_ok=True)
    assert not (args.output/'technical-started.json').exists(),'Retain first execution record'
    sys.path.insert(0,str(args.source))
    from baselines.retacred import rmea_number_optical_protocol as p
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
    from baselines.retacred.rmea_number_optical import initialize_head,validate_initialized_head
    from baselines.retacred.rmea_matched_routers import initialize_head as native_initialize
    from baselines.retacred.rmea_frozen_training import FrozenFeatureTraining
    from baselines.retacred import rmea_number_optical_runtime as runtime
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
