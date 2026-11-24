"""Linear complex-amplitude Born router and a less constrained real-plane control.

Known complex-to-real identity is an implementation check, not novel theory.
"""
import torch
from torch import nn
from torch.nn import functional as F
from baselines.retacred.rmea_matched_routers import Head,initialize_head

MODES=('quantum_amplitude6','dephased_amplitude6','real_plane12',
       'quantum','dephased','real12','softmax12','uniform')
NEW=MODES[:3]

def J(v):
    x,y=v.chunk(2,dim=-1)
    return torch.cat((-y,x),dim=-1)

def unit(v):
    norm=v.norm(dim=-1,keepdim=True)
    if torch.any(norm<1e-12):raise ValueError('Degenerate amplitude/plane vector')
    return v/norm

def complex6(v):
    x,y=v.chunk(2,dim=-1)
    return torch.complex(x,y)

def probabilities(key,query,mode,query_pair=None):
    key=unit(key);query=unit(query)
    if mode=='quantum_amplitude6':
        return torch.einsum('rd,btd->brt',complex6(query).conj(),complex6(key)).abs().square()
    if mode=='dephased_amplitude6':
        return torch.einsum('rd,btd->brt',complex6(query).abs().square(),complex6(key).abs().square())
    if mode=='real_plane12':
        other=unit(query_pair-(query_pair*query).sum(-1,keepdim=True)*query)
        return torch.einsum('rd,btd->brt',query,key).square()+torch.einsum('rd,btd->brt',other,key).square()
    raise ValueError(mode)

class AmplitudeHead(Head):
    def __init__(self,mode):
        super().__init__('real12' if mode in NEW else mode)
        self.variant=mode
        if mode=='real_plane12':self.head.query_pair=nn.Parameter(J(self.head.query.detach()).clone())
    def forward(self,h,roles,mask):
        a=self.head
        if self.variant=='uniform':
            w=mask.to(h.dtype);w=w/w.sum(-1,keepdim=True)
            return torch.einsum('bt,btd->bd',w,a.value(h))@a.decoder.T+a.bias
        if self.variant not in NEW:return super().forward(h,roles,mask)
        score=probabilities(a.key(torch.cat((h,roles),-1)),a.query,self.variant,getattr(a,'query_pair',None))
        raw=(score+1e-6)*mask[:,None,:];alpha=raw/raw.sum(-1,keepdim=True)
        return (torch.einsum('brt,btd->brd',alpha,a.value(h))*a.decoder).sum(-1)+a.bias

def initialize(initial,seed):
    out={}
    for mode in MODES:
        native=initialize_head('real12' if mode in NEW else mode,initial,dict(seed=seed,training_dtype='float64'))
        model=AmplitudeHead(mode).double()
        state=dict(native.state_dict())
        if mode=='real_plane12':state['head.query_pair']=J(state['head.query']).clone()
        model.load_state_dict(state);out[mode]=model
    return out

def make_head(mode):return AmplitudeHead(mode)

def fixtures(models,inputs):
    h,roles,mask=(v.cpu() for v in inputs)
    q=models['quantum_amplitude6'];d=models['dephased_amplitude6'];p=models['real_plane12']
    for other in (d,p):
        assert all(torch.equal(v,other.state_dict()[k]) for k,v in q.state_dict().items())
    key=q.head.key(torch.cat((h,roles),-1));query=q.head.query
    prob=probabilities(key,query,'quantum_amplitude6')
    # Shared-J real plane is exactly the same Born probability.
    identity=probabilities(key,query,'real_plane12',J(query))
    identity_error=float((identity-prob).abs().max());assert identity_error<1e-12
    score_error=float((q(h,roles,mask)-p(h,roles,mask)).abs().max());assert score_error<1e-12
    angles=torch.linspace(-2,2,key.shape[1],dtype=key.dtype,device=key.device).view(1,-1,1)
    x,y=key.chunk(2,dim=-1);rotated=torch.cat((x*angles.cos()-y*angles.sin(),x*angles.sin()+y*angles.cos()),dim=-1)
    phase_error=float((prob-probabilities(rotated,query,'quantum_amplitude6')).abs().max());assert phase_error<1e-12
    errors=[]
    for model in (q,d,p):
        z=model(h,roles,mask)
        padded=(F.pad(h,(0,0,0,3)),F.pad(roles,(0,0,0,3)),F.pad(mask,(0,3)))
        errors.append(float((z-model(*padded)).abs().max()))
        assert torch.isfinite(z).all()
    assert max(errors)<1e-12
    z=models['uniform'](h,roles,mask)
    pooled=(models['uniform'].head.value(h)*mask[...,None]).sum(1)/mask.sum(1)[:,None]
    uniform_error=float((z-(pooled@models['uniform'].head.decoder.T+models['uniform'].head.bias)).abs().max())
    assert uniform_error<1e-12
    return dict(status='PASS',complex_real_plane_identity_error=identity_error,initial_logit_parity_error=score_error,phase_gauge_invariance_error=phase_error,padding_error=max(errors),uniform_error=uniform_error,scope='Changed amplitudes/plane mechanism only; actual finite gradients and first real update checked by unchanged runner. Known realification, not quantum exclusive or new theory.')

def summarize(rows,logits,labels,cfg):
    q=rows['quantum_amplitude6'];deltas={k:q['all40_macro_f1']-v['all40_macro_f1'] for k,v in rows.items() if k!='quantum_amplitude6'}
    return dict(results=rows,all40_macro_deltas=deltas,gates={k:v>0 for k,v in deltas.items()},gate='PASS_FOR_NEW_FULL_PROTOCOL_DESIGN_ONLY' if all(v>0 for v in deltas.values()) else 'NO_PROMOTION_EXACT_SCREEN',L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED',scope='ExposedTRAIN planning only. Smooth complex-amplitude map replaces nonlinear finite circuit; same binary Born/word pooling, no novelty or hardware speed claim. Real-plane control is strictly less constrained and has480 extra trainable scalars; common input/order/value/optimizer/budget.')
