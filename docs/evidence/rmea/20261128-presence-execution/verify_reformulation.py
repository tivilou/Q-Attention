"""Synthetic-only proof checks and paired CPU timing for two candidate routes."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch

from check_presence_phase import original
from presence_phase import MODES, PresenceHead
from sector_reformulation import SectorHead, ring_gather


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def err(a,b):
    return float((a-b).detach().abs().max())


def parameter_errors(a,b):
    return {name:err(p.grad,dict(b.named_parameters())[name].grad) for name,p in a.named_parameters()}


def product_fit():
    """A separable 3-qubit alternative and its exact classical harmonic control."""
    torch.manual_seed(46)
    w0=torch.randn(5,3,2,dtype=torch.float64,requires_grad=True)
    q0=torch.randn(40,3,2,dtype=torch.float64,requires_grad=True)
    values=torch.randn(5,64,dtype=torch.float64)
    dec=torch.randn(40,64,dtype=torch.float64)
    mask=torch.tensor([True, True, False, True, True])
    def amplitude(a):
        y,z=a.unbind(-1)
        return torch.stack((torch.cos(y/2)*torch.exp(-.5j*z),torch.sin(y/2)*torch.exp(.5j*z)),dim=-1)
    def kron(s):
        a,b,c=s.unbind(dim=-2)
        return (a[:, :,None,None]*b[:,None,:,None]*c[:,None,None,:]).flatten(1)
    def independent_pools(w,q,qubit):
        if qubit:
            w,q=kron(amplitude(w)),kron(amplitude(q))
            left,right=w[:,:4]@q[:,:4].T.conj(),w[:,4:]@q[:,4:].T.conj()
            bm=left.abs().square()+right.abs().square()
            first=(left[:,0]+right[:,0]).abs().square()
        else:
            y,z=w.unbind(-1);u,v=q.unbind(-1)
            diagonal=(1+y[:,None,:].cos()*u[None,:,:].cos())/2
            kernel=diagonal+(.5*y[:,None,:].sin()*u[None,:,:].sin()*(z[:,None,:]-v[None,:,:]).cos())
            bm=diagonal[...,0]*kernel[...,1]*kernel[...,2]
            first=kernel[:,0,:].prod(-1)
        p=.25+.5*first
        raw=torch.cat((p[:,None],1-p[:,None],bm[:,1:]),dim=1)
        raw=(raw+1e-6)*mask[:,None]
        pooled=raw.T@values/raw.sum(0)[:,None]
        g=dec[0]@(pooled[0]-pooled[1]);u=(dec[1:]*pooled[2:]).sum(-1)
        z=torch.cat((g.new_zeros(1),g+u.log_softmax(0)))
        return pooled,z
    quantum, q=independent_pools(w0,q0,True)
    classical, c=independent_pools(w0,q0,False)
    u=torch.linspace(.1,1.,40,dtype=q.dtype)
    qgrad=torch.autograd.grad((q*u).sum(),(w0,q0))
    cgrad=torch.autograd.grad((c*u).sum(),(w0,q0))
    gaps={'full64_pools':err(quantum,classical),'all40':err(q,c),
          'word_gradient':err(qgrad[0],cgrad[0]),'query_gradient':err(qgrad[1],cgrad[1])}
    assert max(gaps.values())<4e-13
    return gaps


def main(source):
    torch.set_num_threads(1)
    circuit=original(source)
    assert ring_gather()==[0,7,3,4,6,1,5,2]
    rng=torch.Generator().manual_seed(46)
    random=lambda *s:torch.randn(*s,generator=rng)
    angles=random(4,11,2,3,2).double()
    state=SectorHead(MODES[0],circuit).double()
    state_gap=err(state.pre_final_state(angles).index_select(-1,state.first_ring),circuit(angles))
    assert state_gap<1e-14
    witness=torch.zeros(2,3,2,dtype=torch.float64)
    witness[-1,0,0]=math.pi/2
    final=circuit(witness)
    pre=state.pre_final_state(witness)
    # The native diagram maps pre-ring |100> to |011>. Physical output
    # wire 0 is always zero here, while the frozen pre-ring sectors mix.
    expected=torch.zeros_like(final)
    expected[[0,3]]=2**-.5
    assert err(final,expected)<1e-14
    def block_mass(s,cut):
        return (s[:cut].abs().square().sum().square()+s[cut:].abs().square().sum().square()).item()
    prior=.25+.5*block_mass(pre,4)
    physical=.25+.5*block_mass(final,4)
    assert abs(prior-.5)<1e-14 and abs(physical-.75)<1e-14
    raw={};timings={}
    for mode in MODES:
        a=PresenceHead(mode,circuit)
        b=SectorHead(mode,circuit)
        b.load_state_dict(a.state_dict(),strict=True)
        variants=(False,True) if mode==MODES[0] else (False,)
        for dtype in (torch.float64,torch.float32):
            a.to(dtype);b.to(dtype)
            ramp=1.e-12 if dtype==torch.float64 else 4.e-5
            for intervention in variants:
                h=random(3,19,1024).to(dtype).requires_grad_(True)
                roles=random(3,19,4).to(dtype).requires_grad_(True)
                m=torch.arange(19)[None]<torch.tensor([1,17,19])[:,None];m[1,4]=False
                targets=torch.tensor([0,17,39])
                out_a=a(h,roles,m,details=True,dephase=intervention)
                out_b=b(h,roles,m,details=True,dephase=intervention)
                losses=[torch.nn.functional.cross_entropy(t['logits'],targets) for t in (out_a,out_b)]
                a.zero_grad(set_to_none=True);b.zero_grad(set_to_none=True)
                ah,ar=torch.autograd.grad(losses[0],(h,roles),retain_graph=True)
                bh,br=torch.autograd.grad(losses[1],(h,roles),retain_graph=True)
                losses[0].backward();losses[1].backward()
                g=parameter_errors(a,b)
                diffs={k:err(out_a[k],out_b[k]) for k in out_a}
                diffs.update(h_gradient=err(ah,bh),role_gradient=err(ar,br),loss=err(*losses))
                assert max(diffs.values())<ramp and max(g.values())<ramp,(mode,dtype,diffs,g)
                assert torch.count_nonzero(bh[~m])==0 and torch.count_nonzero(br[~m])==0
                assert all(torch.isfinite(p.grad).all() for p in b.parameters())
                if intervention:
                    assert torch.equal(out_b['positive_scores'], b(h,roles,m,details=True)['positive_scores'])
                key=f'{mode}/{str(dtype)}/{intervention}'
                raw[key]=dict(tensors=diffs,parameters=g)
        # No optimizer or task arrays. Alternating warmed identical repeated micro4 proxy.
        h,roles=random(4,56,1024),random(4,56,4)
        mask=torch.arange(56)[None]<torch.tensor([44,52,55,56])[:,None]
        y=torch.tensor([0,13,39,2])
        def bulk(model):
            start=time.perf_counter()
            for _ in range(12):
                model.zero_grad(set_to_none=True)
                torch.nn.functional.cross_entropy(model(h,roles,mask),y).backward()
            return (time.perf_counter()-start)/12
        bulk(a);bulk(b);pairs=[]
        for i in range(7):
            if i%2:t1,t0=bulk(b),bulk(a)
            else:t0,t1=bulk(a),bulk(b)
            pairs.append((t0,t1))
        timings[mode]={'original_cpu_seconds':statistics.median(a for a,b in pairs),
                       'reformulated_cpu_seconds':statistics.median(b for a,b in pairs),
                       'paired_reformulated_over_original':statistics.median(b/a for a,b in pairs),
                       'all_pairs_seconds':pairs}
    sim=product_fit()
    out=dict(status='PASS_EXACT_SECTOR_REFORMULATION_AND_SEPARABLE_CLASSICAL_IDENTITY',
          native_source_sha256=sha(source),code_sha256=sha(Path(__file__).with_name('sector_reformulation.py')),
          test_sha256=sha(__file__),complex_state_recover_max_error=state_gap,complete_tensor_and_gradient_errors=raw,
          warmed_synthetic_cpu_timing=timings,separable_alternative_exact_control_errors=sim,
          sector_boundary_witness={'frozen_pre_ring_wire0_dephased_presence':prior,'physical_output_wire0_dephased_presence':physical},
          complex_overlap_dense_mac_per_token=dict(original=640,exact_block_sliced=320),
          real_overlap_dense_mac_per_token=dict(original=2560,exact_packed=1280),
          diagnostic_runs=0,new_task_forwards=0,new_optimizer_updates=0,new_gpu_allocations=0,
          scope='Forward/backward synthetic CPU only; timings are not GPU or budget evidence. Exact means over real arithmetic, not bitwise finite-precision identity.')
    print(json.dumps(out,indent=2,allow_nan=False))


if __name__=='__main__':
    main(sys.argv[1])
