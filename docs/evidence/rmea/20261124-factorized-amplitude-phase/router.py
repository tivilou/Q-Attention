"""Separate real amplitude preparation and commuting phase encoding.

The gates and their algebra are known; this is a task hypothesis, not new theory.
"""
import math
import torch
from baselines.retacred.rmea_matched_routers import Head,initialize_head

NEW=('quantum_factorized','dephased_factorized','real_factorized')
MODES=NEW+('quantum','dephased','real12','softmax12','uniform')

def ry_ring(angles,state=None):
    if state is None:
        state=torch.zeros(angles.shape[:-2]+(8,),device=angles.device,dtype=angles.dtype)
        state[...,0]=1
    for layer in range(2):
        for wire in range(3):
            shape=state.shape;pair=state.reshape(shape[:-1]+(2**wire,2,2**(2-wire)))
            x,y=pair.unbind(-2);a=angles[...,layer,wire]
            c=torch.cos(a/2)[...,None,None];s=torch.sin(a/2)[...,None,None]
            state=torch.stack((c*x-s*y,s*x+c*y),dim=-2).reshape(shape)
        for control,target in ((0,1),(1,2),(2,0)):
            perm=[i^((1<<(2-target)) if (i>>(2-control))&1 else 0) for i in range(8)]
            state=state[...,perm]
    return state

def state12(angles,mode):
    amp=ry_ring(angles[...,:6].reshape(angles.shape[:-1]+(2,3)))
    if mode=='dephased_factorized':return amp
    if mode=='real_factorized':
        return ry_ring(angles[...,6:].reshape(angles.shape[:-1]+(2,3)),amp)
    if mode!='quantum_factorized':raise ValueError(mode)
    # Diagonal Pauli Z0,Z1,Z2,Z0Z1,Z1Z2,Z2Z0. 3 RZ +3 RZZ.
    z=torch.tensor([[1-2*((i>>(2-j))&1) for i in range(8)] for j in range(3)],device=angles.device,dtype=angles.dtype)
    fields=torch.cat((z,z[[0,1,2]]*z[[1,2,0]]),dim=0)
    phase=-.5*torch.einsum('...p,pi->...i',angles[...,6:],fields)
    return amp*torch.exp(1j*phase)

def probability(key,query,mode):
    if mode=='dephased_factorized':
        return torch.einsum('rd,btd->brt',query.square(),key.square())
    return torch.einsum('rd,btd->brt',query.conj(),key).abs().square()

class FactorizedHead(Head):
    def __init__(self,mode):
        super().__init__('quantum' if mode in NEW else mode);self.variant=mode
    def forward(self,h,roles,mask):
        a=self.head
        if self.variant=='uniform':
            w=mask.to(h.dtype);w=w/w.sum(-1,keepdim=True)
            return torch.einsum('bt,btd->bd',w,a.value(h))@a.decoder.T+a.bias
        if self.variant not in NEW:return super().forward(h,roles,mask)
        ka=math.pi*torch.tanh(a.prepare(torch.cat((h,roles),-1)))
        qa=(math.pi*torch.tanh(a.effect)).flatten(-3)
        key=state12(ka,self.variant);query=state12(qa,self.variant)
        score=probability(key,query,self.variant)
        raw=(score+1e-6)*mask[:,None,:];alpha=raw/raw.sum(-1,keepdim=True)
        return (torch.einsum('brt,btd->brd',alpha,a.value(h))*a.decoder).sum(-1)+a.bias

def initialize(initial,seed):
    out={}
    for name in MODES:
        native=initialize_head('quantum' if name in NEW else name,initial,dict(seed=seed,training_dtype='float64'))
        model=FactorizedHead(name).double();model.load_state_dict(native.state_dict());out[name]=model
    return out

def make_head(mode):return FactorizedHead(mode)

def fixtures(models,inputs):
    h,roles,mask=(v.cpu() for v in inputs)
    ref=models[NEW[0]]
    assert all(all(torch.equal(v,models[n].state_dict()[k]) for k,v in ref.state_dict().items()) for n in NEW[1:])
    k=math.pi*torch.tanh(ref.head.prepare(torch.cat((h,roles),-1)))
    q=(math.pi*torch.tanh(ref.head.effect)).flatten(-3)
    norms=[];padding=[]
    for name in NEW:
        ks=state12(k,name);qs=state12(q,name)
        norms.extend([float((ks.abs().square().sum(-1)-1).abs().max()),float((qs.abs().square().sum(-1)-1).abs().max())])
        z=models[name](h,roles,mask);assert torch.isfinite(z).all()
        padded=(torch.nn.functional.pad(h,(0,0,0,3)),torch.nn.functional.pad(roles,(0,0,0,3)),torch.nn.functional.pad(mask,(0,3)))
        padding.append(float((z-models[name](*padded)).abs().max()))
    delta=torch.linspace(-.3,.3,6,dtype=k.dtype)
    shifted_k=k.clone();shifted_q=q.clone();shifted_k[...,6:]+=delta;shifted_q[...,6:]+=delta
    base=probability(state12(k,NEW[0]),state12(q,NEW[0]),NEW[0])
    shifted=probability(state12(shifted_k,NEW[0]),state12(shifted_q,NEW[0]),NEW[0])
    covariance_error=float((base-shifted).abs().max())
    diagonal_error=float((state12(k,NEW[1])-state12(shifted_k,NEW[1])).abs().max())
    zero_k=k.clone();zero_q=q.clone();zero_k[...,6:]=0;zero_q[...,6:]=0
    zero_qprob=probability(state12(zero_k,NEW[0]),state12(zero_q,NEW[0]),NEW[0])
    zero_rprob=probability(state12(zero_k,NEW[2]),state12(zero_q,NEW[2]),NEW[2])
    zero_error=float((zero_qprob-zero_rprob).abs().max())
    assert max(norms+padding+[covariance_error,diagonal_error,zero_error])<1e-12
    return dict(status='PASS',state_norm_error=max(norms),padding_error=max(padding),shared_phase_covariance_error=covariance_error,diagonal_phase_independence_error=diagonal_error,zero_phase_readout_parity_error=zero_error,scope='Only new gate-order/phase fields and real-control correspondence; actual finite gradients/first updates checked in stable runner. Known gate algebra, no T novelty.')

def summarize(rows,logits,labels,cfg):
    q=rows['quantum_factorized'];ds={k:q['all40_macro_f1']-v['all40_macro_f1'] for k,v in rows.items() if k!='quantum_factorized'}
    return dict(results=rows,all40_macro_deltas=ds,gates={k:v>0 for k,v in ds.items()},gate='PASS_FOR_NEW_FULL_PROTOCOL_DESIGN_ONLY' if all(v>0 for v in ds.values()) else 'NO_PROMOTION_EXACT_SCREEN',L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED',scope='Distinct separated amplitude/phase ansatz; native binaryBorn/value/optimizer unchanged. ExposedTRAIN planning, known gates, no novelty/quantum-exclusive or hardware speedup. Equal registered81028scalars, dephased has6414 invariantphase scalars; no equal effective capacity.')
