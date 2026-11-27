"""Math-only ordered-subsequence routing; no task, runtime or registry installed.

All sums keep increasing distinct indices. D removes subset-path coherence only,
not intrapath memory coherence. Physical postselected masses use C**(-2).
"""
import math
import torch
from torch import nn

EPS=1e-6

def _operator_sums(u,k):
    n,d,_=u.shape
    eye=torch.eye(d,dtype=u.dtype,device=u.device)
    zero=torch.zeros_like(eye)
    pref=[[eye]+[zero]*k]
    for j in range(n):
        prev=pref[-1]
        pref.append([eye]+[prev[a]+u[j]@prev[a-1] for a in range(1,k+1)])
    suffix=[[eye]+[zero]*k for _ in range(n+1)]
    for j in range(n-1,-1,-1):
        prev=suffix[j+1]
        suffix[j]=[eye]+[prev[b]+prev[b-1]@u[j] for b in range(1,k+1)]
    return pref,suffix

def coherent_masses(U,E,mask):
    out=[]
    for b in range(len(U)):
        idx=mask[b].nonzero(as_tuple=True)[0]
        u=U[b,idx];n,d,_=u.shape
        if n==0:raise ValueError('Empty sentence')
        k=min(3,n);count=math.comb(n-1,k-1)
        pref,suffix=_operator_sums(u,k)
        marked=[]
        for t in range(n):
            H=sum(suffix[t+1][k-1-a]@u[t]@pref[t][a] for a in range(k))/count
            psi=H[:,0]
            marked.append(torch.einsum('i,rij,j->r',psi.conj(),E,psi).real)
        g=torch.stack(marked,-1)
        out.append(g.new_zeros(E.shape[0],mask.shape[1]).index_copy(1,idx,g))
    return torch.stack(out)

def dephased_masses(U,E,mask):
    out=[]
    for b in range(len(U)):
        idx=mask[b].nonzero(as_tuple=True)[0]
        u=U[b,idx];n,d,_=u.shape
        if n==0:raise ValueError('Empty sentence')
        k=min(3,n);count=math.comb(n-1,k-1)
        rho=u.new_zeros(d,d);rho[0,0]=1
        z=torch.zeros_like(rho)
        pref=[[rho]+[z]*k]
        for j in range(n):
            previous=pref[-1]
            pref.append([rho]+[previous[a]+u[j]@previous[a-1]@u[j].mH for a in range(1,k+1)])
        suffix=[[E]+[torch.zeros_like(E)]*k for _ in range(n+1)]
        for j in range(n-1,-1,-1):
            previous=suffix[j+1]
            suffix[j]=[E]+[previous[a]+u[j].mH@previous[a-1]@u[j] for a in range(1,k+1)]
        marked=[]
        for t in range(n):
            g=sum(torch.einsum('rij,ji->r',suffix[t+1][k-1-a],u[t]@pref[t][a]@u[t].mH).real for a in range(k))/count**2
            marked.append(g)
        g=torch.stack(marked,-1)
        out.append(g.new_zeros(E.shape[0],mask.shape[1]).index_copy(1,idx,g))
    return torch.stack(out)

def attention(masses,mask):
    raw=(masses+EPS)*mask[:,None]
    return raw/raw.sum(-1,keepdim=True)

def readout(masses,h,mask,value,decoder,bias):
    h=torch.where(mask[...,None],h,0)
    values=torch.where(mask[...,None],value(h),0)
    alpha=attention(masses,mask)
    logits=torch.einsum('brt,btd,rd->br',alpha,values,decoder)+bias
    return dict(logits=logits,alpha=alpha,masses=masses,values=values)

class SignedRealOperatorHead(nn.Module):
    """Independent nonlinear classical comparator specification, not Q replay.

    Its85,748 raw parameters and real16 state are not an effective-capacity match
    certificate. This prototype is only exercised on artificial inputs.
    """
    def __init__(self):
        super().__init__()
        self.prepare=nn.Linear(1028,12)
        self.operator=nn.Sequential(nn.Linear(12,16),nn.Tanh(),nn.Linear(16,256))
        self.terminal=nn.Parameter(torch.randn(40,16)*.2)
        self.value=nn.Linear(1024,64)
        self.decoder=nn.Parameter(torch.randn(40,64)/8)
        self.bias=nn.Parameter(torch.zeros(40))

    def components(self,h,roles,mask):
        h=torch.where(mask[...,None],h,0)
        roles=torch.where(mask[...,None],roles,0)
        raw=self.operator(torch.tanh(self.prepare(torch.cat((h,roles),-1)))).reshape(h.shape[:2]+(16,16))
        A=raw/torch.linalg.matrix_norm(raw,ord='fro',dim=(-2,-1)).clamp_min(1)[...,None,None]
        phi=self.terminal/self.terminal.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        E=(phi[:,:,None]*phi[:,None,:]+torch.eye(16,dtype=A.dtype,device=A.device)[None]/16)/2
        return A,E,h,roles

    def forward(self,h,roles,mask):
        A,E,h,roles=self.components(h,roles,mask)
        return readout(coherent_masses(A,E,mask),h,mask,self.value,self.decoder,self.bias)['logits']
