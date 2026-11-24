"""Exact reused matrix primitives from ordered-context-memory; no sequence runner."""
import torch

def quantum_matrix(angles):
    """Apply the exact native circuit to all computational basis columns."""
    dtype=torch.complex128 if angles.dtype==torch.float64 else torch.complex64
    state=torch.eye(8,dtype=dtype,device=angles.device).expand(angles.shape[:-3]+(8,8))
    for layer in range(2):
        for wire in range(3):
            shape=state.shape
            pair=state.reshape(shape[:-1]+(2**wire,2,2**(2-wire)))
            a,b=pair.unbind(-2)
            y,z=angles[...,layer,wire,:].unbind(-1)
            c,s=torch.cos(y/2)[...,None,None,None],torch.sin(y/2)[...,None,None,None]
            minus=torch.exp(-.5j*z)[...,None,None,None]
            plus=torch.exp(.5j*z)[...,None,None,None]
            state=torch.stack(((c*a-s*b)*minus,(s*a+c*b)*plus),dim=-2).reshape(shape)
        for control,target in ((0,1),(1,2),(2,0)):
            permutation=[i ^ ((1 << (2-target)) if (i >> (2-control)) & 1 else 0) for i in range(8)]
            state=state[...,permutation]
    return state.transpose(-1,-2)

def orthogonal_transition(angles):
    state=torch.eye(12,dtype=angles.dtype,device=angles.device).expand(angles.shape[:-1]+(12,12))
    for layer in range(2):
        order=list(range(12)) if layer==0 else list(range(1,12))+[0]
        inverse=[order.index(i) for i in range(12)]
        pair=state[...,order,:].reshape(angles.shape[:-1]+(6,2,12))
        a,b=pair.unbind(-2);theta=angles[...,6*layer:6*(layer+1)]
        c,s=theta.cos()[...,None],theta.sin()[...,None]
        state=torch.stack((c*a-s*b,s*a+c*b),dim=-2).reshape(angles.shape[:-1]+(12,12))[...,inverse,:]
    return state
