"""Synthetic-only ordered products for the unchanged history equations.

Not registered in the task runner. No data, optimizer or resource policy.
"""
import torch

def ordered_products(maps):
    """Inclusive prefix/suffix; a later segment B composes as B @ A."""
    prefix=suffix=maps
    offset=1
    while offset<maps.shape[1]:
        prefix=torch.cat((prefix[:,:offset],prefix[:,offset:]@prefix[:,:-offset]),dim=1)
        suffix=torch.cat((suffix[:,offset:]@suffix[:,:-offset],suffix[:,-offset:]),dim=1)
        offset*=2
    return prefix,suffix

def selected_masses(selected,maps,terminal,initial,mask):
    batch,length,dim,_=maps.shape
    eye=torch.eye(dim,dtype=maps.dtype,device=maps.device).expand(batch,1,dim,dim)
    maps=torch.where(mask[:,:,None,None],maps,eye)
    prefix,suffix=ordered_products(maps)
    before=torch.cat((eye,prefix[:,:-1]),dim=1)
    after=torch.cat((suffix[:,1:],eye),dim=1)
    forward=(before@initial[None,None,:,None])[...,0]
    selected_forward=(selected@forward[...,None])[...,0]
    final=(after@selected_forward[...,None])[...,0]
    masses=torch.einsum('rd,btd->brt',terminal.conj(),final).real
    return torch.where(mask[:,None],masses,0)

def quantum_masses(U,E,mask):
    batch,length,d,_=U.shape
    # Row-major vec(U rho U†): (U tensor U*) vec(rho).
    adjoint=torch.einsum('btik,btjl->btijkl',U,U.conj()).reshape(batch,length,d*d,d*d)
    eye=torch.eye(d*d,dtype=U.dtype,device=U.device)
    initial=U.new_zeros(d*d);initial[0]=1
    return selected_masses(adjoint/2,(eye+adjoint)/2,E.reshape(E.shape[0],d*d),initial,mask)

def population_masses(P,e,mask):
    dim=P.shape[-1];eye=torch.eye(dim,dtype=P.dtype,device=P.device)
    initial=P.new_zeros(dim);initial[0]=1
    return selected_masses(P/2,(eye+P)/2,e,initial,mask)
