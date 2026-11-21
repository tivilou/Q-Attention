"""Focused CPU algebra and mask checks for the frozen diagnostic."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from experiments.q_rmea_mass_direction import arms, macro


def main():
    torch.manual_seed(20261121)
    for lengths in ([1],[3],[2,5]):
        mask=torch.arange(max(lengths))[None,:]<torch.tensor(lengths)[:,None]
        psi=torch.randn(len(lengths),max(lengths),8,dtype=torch.complex128)
        psi=psi/psi.norm(dim=-1,keepdim=True)
        phi=torch.randn(40,8,dtype=torch.complex128);phi=phi/phi.norm(dim=-1,keepdim=True)
        distributions,values=arms(psi,phi,mask)
        for v in distributions.values():
            assert torch.allclose(v.sum(-1),torch.ones_like(v.sum(-1)),atol=1e-12,rtol=0)
            assert (v.masked_select((~mask)[:,None,:].expand_as(v))==0).all()
        assert torch.allclose(values['direction_raw']*values['z'][:,None,:],values['routed_raw'],atol=1e-12,rtol=0)
        altered=psi.clone();altered[~mask]=7+9j
        changed,_=arms(altered,phi,mask)
        assert all(torch.equal(distributions[k],changed[k]) for k in distributions)
        if lengths==[1]:assert all(torch.equal(v,torch.ones_like(v)) for v in distributions.values())
    import numpy as np
    assert macro(np.array([0,1,2]),np.array([0,1,2]))==2/39
    # Explicit cubic witness: first-order row mass drift vanishes; second-order does not.
    x=torch.zeros(3,8,dtype=torch.complex128);x[0,0]=x[1,0]=1;x[2,0]=.5;x[2,1]=3**.5/2
    k=x@x.mH;h=k*k.abs().square();h=h-torch.diag_embed(h.diagonal());c=torch.linalg.eigvalsh(h).abs().max();h=h/c
    comm=h@k-k@h
    assert comm.diagonal().abs().max()<1e-14
    second=-(h@comm-comm@h).diagonal().real
    assert torch.allclose(second[2],-3/(16*c*c),atol=1e-12,rtol=0)
    import os
    if os.environ.get('RESEARCH_OUTPUT_DIR'):
        from baselines.retacred.common import atomic_json
        out=Path(os.environ['RESEARCH_OUTPUT_DIR'])
        atomic_json(out/'unit-validation.json',dict(accepted=True,checks=['factorization','masking','singleton','normalized arms','fixed39 metric','first/second-order witness']))
        atomic_json(out/'RUN_COMPLETE',dict(accepted=True))
    print('PASS: factorization, masking, singleton, normalized arms, fixed39 metric, cubic first/second-order witness')


if __name__=='__main__':main()
