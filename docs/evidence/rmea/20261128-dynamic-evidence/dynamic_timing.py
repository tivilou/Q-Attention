"""Verify midpoint recovery with signal and disturbance timing, plus matched replay."""
from itertools import product
import json
import numpy as np


def run():
    i2=np.eye(2)
    x=np.array([[0.,1.],[1.,0.]])
    z=np.diag([1.,-1.])
    ident=np.eye(8)
    x1=np.kron(np.kron(x,i2),i2)
    z1=np.kron(np.kron(z,i2),i2)
    z2=np.kron(np.kron(i2,z),i2)
    p=(ident+z1@z2)/2
    complement=ident-p
    even_odd=(p,x1@complement)
    v=np.zeros(8,dtype=complex)
    v[[0,6]]=1/np.sqrt(2)
    base=np.outer(v,v.conj())
    phases=np.linspace(.07,5.93,40)
    steps=np.linspace(.17,1.27,9)

    def rotation(t):
        return np.cos(t/2)*ident-1j*np.sin(t/2)*z1

    def conj(u,r):
        return u@r@u.conj().T

    def recover(r):
        return sum(conj(k,r) for k in even_odd)

    def measure(theta):
        q=v.copy()
        q[6]*=np.exp(1j*theta)
        return np.outer(q,q.conj())

    effects=np.stack([measure(t) for t in phases])
    rng=np.random.default_rng(20261128)
    values=rng.normal(size=(len(steps),64))
    decoder=rng.normal(size=(40,64))

    def readout(rho):
        return np.einsum('rij,ji->r',effects,rho).real

    def pool(scores):
        alpha=scores+1e-6
        alpha=alpha/alpha.sum(0,keepdims=True)
        return alpha.T@values

    errors=dict(collision=0.,tolerated_timing=0.,d_erases=0.,classical_probability=0.)
    world_a=[];world_b=[];static_a=[];static_b=[];classical_a=[];classical_b=[]
    for theta in steps:
        # World A: positive signal, later flip. World B: earlier flip,
        # then negative signal. Without a midpoint instrument they collide.
        pa=conj(x1,conj(rotation(theta),base))
        pb=conj(rotation(-theta),conj(x1,base))
        errors['collision']=max(errors['collision'],float(np.max(np.abs(pa-pb))))
        aa=recover(conj(x1,recover(conj(rotation(theta),base))))
        bb=recover(conj(rotation(-theta),recover(conj(x1,base))))
        # A same-sign physical control flips before the signal.
        control=recover(conj(rotation(theta),recover(conj(x1,base))))
        errors['tolerated_timing']=max(errors['tolerated_timing'],float(np.max(np.abs(aa-control))))
        errors['d_erases']=max(errors['d_erases'],float(np.max(np.abs(np.diag(aa)-np.diag(bb)))))
        a,b=readout(aa),readout(bb)
        ca,cb=(1+np.cos(theta-phases))/2,(1+np.cos(-theta-phases))/2
        errors['classical_probability']=max(errors['classical_probability'],float(max(np.max(np.abs(a-ca)),np.max(np.abs(b-cb)))))
        world_a.append(a);world_b.append(b)
        static_a.append(readout(recover(pa)));static_b.append(readout(recover(pb)))
        classical_a.append(ca);classical_b.append(cb)
    pools=[pool(np.stack(a)) for a in [world_a,world_b,static_a,static_b,classical_a,classical_b]]
    readout_logits=lambda a:np.sum(a*decoder,axis=1)
    logits=[readout_logits(a) for a in pools]
    assert max(errors.values())<2e-15
    differences=dict(dynamic_opposite_sign_max_score_change=float(np.max(np.abs(np.stack(world_a)-np.stack(world_b)))),
        dynamic_opposite_sign_max_logit_change=float(np.max(np.abs(logits[0]-logits[1]))),
        final_only_logit_gap=float(np.max(np.abs(logits[2]-logits[3]))),
        matched_stateful_classical_max_logit_error=float(max(np.max(np.abs(logits[0]-logits[4])),np.max(np.abs(logits[1]-logits[5])))))
    assert differences['dynamic_opposite_sign_max_logit_change']>.1
    assert differences['final_only_logit_gap']<1e-13
    assert differences['matched_stateful_classical_max_logit_error']<1e-13
    return dict(status='PASS_DYNAMIC_TIMING_WITNESS_WITH_MATCHED_CLASSICAL_REPLAY', synthetic_class_effects=40,token_count=len(steps),
                full_values_per_token=64,**errors,**differences,
                scope='Intermediate instrument distinguishes timing in this artificial path; a classical model with the same event timing reproduces all results. No natural-input validity claimed.',
                task_forwards=0,updates=0,gpu_allocations=0)


if __name__=='__main__':
    print(json.dumps(run(),indent=2,allow_nan=False))