"""Qubit CPTP counterfactual geometry and full two-qubit attention logits."""
from __future__ import annotations

import math
import torch
from torch import nn

MODES = ('disabled', 'quantum', 'classical_cosine', 'classical_rbf',
         'diagonal', 'shuffled', 'entity_permuted')


def paulis(reference):
    dtype = torch.complex128 if reference.dtype == torch.float64 else torch.complex64
    return torch.tensor([[[0,1],[1,0]], [[0,-1j],[1j,0]], [[1,0],[0,-1]]],
                        dtype=dtype, device=reference.device)


def rotation(angle, axis):
    p = paulis(angle)[axis]
    eye = torch.eye(2, dtype=p.dtype, device=p.device)
    return torch.cos(angle/2)[...,None,None]*eye-1j*torch.sin(angle/2)[...,None,None]*p


def fidelity(first, second, epsilon=1e-5):
    # Closed-form squared Uhlmann fidelity for 2x2 density matrices.
    overlap = (first @ second).diagonal(dim1=-2, dim2=-1).sum(-1).real
    product = (torch.linalg.det(first).real * torch.linalg.det(second).real).clamp_min(1e-16)
    return (overlap+2*product.sqrt()).clamp(epsilon, 1-epsilon)


def rotate_valid(tensor, mask):
    """Rotate valid token correspondence; independent of batch partition."""
    n = mask.sum(-1)
    positions = torch.arange(mask.shape[-1], device=mask.device)[None]
    indexes = torch.where(mask, (positions+1) % n[:,None], positions)
    extra = (1,)*(tensor.ndim-2)
    return tensor.gather(1,indexes.reshape(*indexes.shape,*extra).expand_as(tensor))


class CounterfactualFidelityAttention(nn.Module):
    def __init__(self, dim, mode='quantum', *, depolarization=1e-4,
                 fidelity_epsilon=1e-5, temperature=.7, depth=2):
        super().__init__()
        if mode not in MODES or depth != 2 or not 0 < depolarization < 1 or temperature <= 0:
            raise ValueError('invalid frozen Q-CFRA configuration')
        self.mode, self.depolarization = mode, depolarization
        self.epsilon, self.temperature = fidelity_epsilon, temperature
        self.projection = nn.Linear(dim,4)
        self.channel = nn.Parameter(torch.randn(6)*.1)
        self.readout = nn.Parameter(torch.randn(2,2,3)*.1)
        self.last = {}

    def geometry(self, hidden, mask, capture=False):
        # hidden [B,3,N,D]. Shared parameters across views; never gold inputs.
        projected = self.projection(hidden)
        self.last = {}
        if self.mode.startswith('classical'):
            x = projected[..., :3] + projected[...,3:4]
            x = torch.tanh(x*torch.exp(self.channel[:3])+self.channel[3:])
            x = nn.functional.normalize(x,dim=-1,eps=1e-8)
            r, rs, ro = x.unbind(1)
            if self.mode == 'classical_cosine':
                fs,fo = ((r*rs).sum(-1)+1)/2, ((r*ro).sum(-1)+1)/2
            else:
                fs,fo = torch.exp(-(r-rs).square().sum(-1)),torch.exp(-(r-ro).square().sum(-1))
            a,b = r-rs,r-ro
            covariance = (a*b).sum(-1)-(r*a).sum(-1)*(r*b).sum(-1)
            if capture:
                self.last.update(projected=projected,real_states=x)
        else:
            amplitudes = torch.complex(projected[...,:2], projected[...,2:])
            norm = amplitudes.abs().square().sum(-1,keepdim=True)
            fallback = torch.zeros_like(amplitudes)
            fallback[...,0] = 1
            amplitudes = torch.where(norm>1e-12,amplitudes/norm.clamp_min(1e-12).sqrt(),fallback)
            pure = amplitudes[..., :,None]*amplitudes.conj()[...,None,:]
            weights = self.channel[3:].softmax(-1)
            # A common fixed noncommuting basis rotation keeps every channel
            # angle observable in the diagonal control as well as the quantum path.
            basis = rotation(self.channel.new_tensor(math.pi/4),1)
            unitaries = torch.stack([basis @ rotation(self.channel[i],i) for i in range(3)])
            rho = sum(weights[i]*(unitaries[i] @ pure @ unitaries[i].mH) for i in range(3))
            identity = torch.eye(2,dtype=rho.dtype,device=rho.device)/2
            rho = (1-self.depolarization)*rho+self.depolarization*identity
            if self.mode == 'diagonal':
                rho = torch.diag_embed(rho.diagonal(dim1=-2,dim2=-1))
            rho0,rhos,rhoo = rho.unbind(1)
            if self.mode == 'shuffled':
                rhos,rhoo = rotate_valid(rhos,mask),rotate_valid(rhoo,mask)
            fs,fo = fidelity(rho0,rhos,self.epsilon),fidelity(rho0,rhoo,self.epsilon)
            def bloch(state):
                return torch.einsum('...ij,aji->...a',state,paulis(hidden)).real
            r,rs,ro = bloch(rho0),bloch(rhos),bloch(rhoo)
            a,b = r-rs,r-ro
            covariance = (a*b).sum(-1)-(r*a).sum(-1)*(r*b).sum(-1)
            commutator = 2*torch.linalg.cross(a,b,dim=-1).norm(dim=-1)
            if capture:
                self.last.update(projected=projected,amplitudes=amplitudes,pure=pure,density=rho,
                                 paired_density=torch.stack((rho0,rhos,rhoo),1),
                                 channel_weights=weights,channel_unitaries=unitaries,
                                 bloch=torch.stack((r,rs,ro),1),commutator=commutator)
        fs,fo = fs.clamp(self.epsilon,1-self.epsilon),fo.clamp(self.epsilon,1-self.epsilon)
        features = torch.stack((torch.logit(fs),torch.logit(fo),covariance),-1)
        if capture:
            self.last.update(fidelity=torch.stack((fs,fo),-1),covariance=covariance,geometry_features=features)
        return features

    def forward(self, scores, geometry, *, capture=False):
        # scores [B,3,H,Q,K]; the primary returns complete logits, not a residual.
        features = geometry[:,None,None,None,:,:].expand(*scores.shape,3)
        inputs = torch.cat((scores[...,None],features),-1)
        # Fixed pre-screen scale avoids saturating near-identical counterfactual fidelities.
        scale = scores.new_tensor([1.,.25,.25,1.])
        angles = math.pi*torch.tanh(inputs*scale)
        if self.mode.startswith('classical'):
            w = self.readout.reshape(4,3)
            outputs = torch.tanh(angles @ w)
            logits = outputs.sum(-1)/self.temperature
            if capture:
                self.last.update(readout_inputs=inputs,readout_angles=angles,real_readout=outputs,logits=logits)
            return logits
        dtype = paulis(scores).dtype
        # Exact Ry tensor-product encoding followed by the two local Rz phases.
        c0,s0=torch.cos(angles[...,0]/2),torch.sin(angles[...,0]/2)
        c1,s1=torch.cos(angles[...,1]/2),torch.sin(angles[...,1]/2)
        amplitude=torch.stack((c0*c1,c0*s1,s0*c1,s0*s1),-1)
        z0,z1=angles[...,2],angles[...,3]
        phase=-.5*torch.stack((z0+z1,z0-z1,-z0+z1,-z0-z1),-1)
        state=amplitude.to(dtype)*torch.exp(1j*phase)
        states=[]
        if capture:
            initial=torch.zeros_like(state);initial[...,0]=1
            states=[initial,state]
        # Shared learned gates are compiled once, rather than once per token pair.
        layers=[]
        cz=scores.new_tensor([1,1,1,-1]).to(dtype)
        for depth in range(2):
            local=[]
            for qubit in range(2):
                local.append(rotation(self.readout[depth,qubit,0],0) @
                             rotation(self.readout[depth,qubit,1],1) @
                             rotation(self.readout[depth,qubit,2],2))
            layers.append(cz[:,None]*torch.kron(local[0],local[1]))
            if capture:
                state=state @ layers[-1].transpose(-1,-2)
                states.append(state)
        if not capture:
            state=state @ (layers[1] @ layers[0]).transpose(-1,-2)
        probabilities = state.abs().square()
        observable = scores.new_tensor([3,-1,-1,-1])
        logits = (probabilities*observable).sum(-1)/self.temperature
        if capture:
            self.last.update(readout_inputs=inputs,readout_angles=angles,
                             circuit_states=torch.stack(states,-2),logits=logits)
        return logits
