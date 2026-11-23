"""Input-specific inverse-reference filtering inside the native quantum head."""
import torch
from baselines.retacred.rmea import RMEAHead, EPS


def filtered_effects(reference, phi, diagonal=False):
    if diagonal:
        reference=torch.diag_embed(reference.diagonal(dim1=-2,dim2=-1))
    loaded=reference+torch.eye(8,dtype=reference.dtype,device=reference.device)/8
    vectors=torch.linalg.solve(loaded,phi.T.expand(reference.shape[0],8,len(phi)))
    return (vectors/torch.linalg.vector_norm(vectors,dim=1,keepdim=True)).transpose(1,2)


class NullRelativeHead(RMEAHead):
    def __init__(self, mode):
        assert mode in ('coherent','diagonal','native')
        super().__init__('quantum');self.read_mode=mode

    def forward(self,h,roles,mask,details=False):
        native=super().forward(h,roles,mask,details=True)
        if self.read_mode=='native':
            return native if details else native['logits']
        psi=native['psi'];phi=native['phi']
        reference=torch.einsum('bt,bti,btj->bij',native['alpha'][:,0].to(psi.dtype),psi,psi.conj())
        effects=filtered_effects(reference,phi[1:],self.read_mode=='diagonal')
        probability=torch.einsum('brd,btd->brt',effects.conj(),psi).abs().square()
        raw=(probability+EPS)*mask[:,None,:]
        alpha=raw/raw.sum(-1,keepdim=True)
        positive=(alpha*native['utilities'][:,1:]).sum(-1)+self.bias[1:]
        logits=torch.cat([native['logits'][:,:1],positive],dim=1)
        if not details:return logits
        return dict(logits=logits,alpha=torch.cat([native['alpha'][:,:1],alpha],1),
            probability=torch.cat([native['probability'][:,:1],probability],1),
            reference_density=reference,filtered_positive_effects=effects,
            native_effects=phi,psi=psi,utilities=native['utilities'])


def fixtures(initial,inputs):
    """Only new operator/gradient/padding contracts; no lifecycle replay."""
    device=inputs[0].device;records={}
    for mode in ('coherent','diagonal'):
        model=NullRelativeHead(mode).double().to(device);model.load_state_dict(initial)
        detail=model(*inputs,details=True);nd=RMEAHead.forward(model,*inputs,details=True)
        ket_bra=nd['psi'][:,:,:,None]*nd['psi'].conj()[:,:,None,:]
        n=(nd['alpha'][:,0,:,None,None]*ket_bra).sum(1)
        reference_error=float((n-detail['reference_density']).abs().max())
        assert reference_error<1e-12
        if mode=='diagonal':n=torch.diag_embed(n.diagonal(dim1=-2,dim2=-1))
        # Independent inverse + dense effect/density contraction.
        inverse=torch.linalg.inv(n+torch.eye(8,dtype=n.dtype,device=device)/8)
        v=torch.einsum('bij,rj->bri',inverse,nd['phi'][1:])
        v=v/torch.linalg.vector_norm(v,dim=-1,keepdim=True)
        effect=torch.einsum('bri,brj->brij',v,v.conj())
        density=torch.einsum('bti,btj->btij',nd['psi'],nd['psi'].conj())
        p=torch.einsum('brij,btji->brt',effect,density).real
        raw=(p+EPS)*inputs[2][:,None,:];alpha=raw/raw.sum(-1,keepdim=True)
        z=torch.cat([nd['logits'][:,:1],(alpha*nd['utilities'][:,1:]).sum(-1)+model.bias[1:]],1)
        parameters=list(model.parameters());weights=torch.linspace(.1,1,40,device=device,dtype=torch.float64)
        g1=torch.autograd.grad((detail['logits']*weights).sum(),parameters,retain_graph=True)
        gg=torch.autograd.grad((z*weights).sum(),parameters,retain_graph=True)
        forward_error=float((detail['logits']-z).abs().max())
        grad_error=max(float((a-b).abs().max()) for a,b in zip(g1,gg))
        native0_error=float((detail['logits'][:,0]-nd['logits'][:,0]).abs().max())
        positive_to_null=torch.autograd.grad(detail['logits'][:,1].sum(),model.effect,retain_graph=True)[0][0].norm()
        h,roles,mask=inputs
        padded=(torch.cat([h,torch.zeros_like(h[:,:3])],1),torch.cat([roles,torch.zeros_like(roles[:,:3])],1),
                torch.cat([mask,torch.zeros_like(mask[:,:3])],1))
        padding_error=float((model(*padded)-detail['logits']).abs().max())
        assert forward_error<1e-10 and grad_error<1e-8 and native0_error==0 and padding_error<1e-10
        assert torch.isfinite(positive_to_null) and positive_to_null>0
        records[mode]=dict(reference_error=reference_error,forward_error=forward_error,gradient_error=grad_error,native_class0_error=native0_error,
            padding_error=padding_error,positive_logit_to_null_effect_gradient=float(positive_to_null))
    return dict(status='PASS',source_specific=records,matrix_port=matrix_fixture(device))


def matrix_fixture(device):
    d=8;dtype=torch.complex128;eye=torch.eye(d,dtype=dtype,device=device)
    plus=(eye[0]+eye[1])/2**.5;minus=(eye[0]-eye[1])/2**.5
    psi=torch.stack([plus,minus]);phi=eye[0:1]
    probabilities=(psi@plus.conj()).abs().square();a=(probabilities+EPS)/(probabilities+EPS).sum()
    n=torch.einsum('t,ti,tj->ij',a.to(dtype),psi,psi.conj())[None]
    full=filtered_effects(n,phi)[0,0];diag=filtered_effects(n,phi,True)[0,0]
    pool=lambda effect: ((psi@effect.conj()).abs().square()+EPS)[0]/(((psi@effect.conj()).abs().square()+EPS).sum())
    native_value=pool(phi[0]);full_value=pool(full);diagonal_value=pool(diag)
    assert full_value<native_value and abs(float(diagonal_value-native_value))<1e-12
    isotropic=filtered_effects(eye[None]/8,phi)
    isotropic_error=float((isotropic[0]-phi).abs().max());assert isotropic_error<1e-12
    # Multiplicity counterexample at fixed prepared states. Compare per-token
    # probabilities, whose attention ratio equals their probability+EPS ratio.
    base=torch.stack([eye[0],eye[1]]);query=minus[None]
    ratios=[]
    for weights0 in ([.5,.5],[2/3,1/3]):
        ref=torch.einsum('t,ti,tj->ij',torch.tensor(weights0,dtype=torch.float64,device=device).to(dtype),base,base.conj())
        eff=filtered_effects(ref[None],query)[0,0]
        q=(base@eff.conj()).abs().square()+EPS;ratios.append(float(q[0]/q[1]))
    assert abs(ratios[0]-ratios[1])>.01
    return dict(status='PASS',isotropic_error=isotropic_error,
        native_signal_pool=float(native_value),full_signal_pool=float(full_value),diagonal_signal_pool=float(diagonal_value),
        signal_suppression_counterexample=True,fixed_state_multiplicity_attention_ratios=ratios,
        scope='Ideal state-port algebra, not finite-circuit/text realizability or task improvement')
