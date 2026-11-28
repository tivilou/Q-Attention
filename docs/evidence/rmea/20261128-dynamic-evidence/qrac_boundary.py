"""Known 2-to-1 code and its exact single-copy simultaneous-readout boundary."""
from itertools import product
import json
import math
import numpy as np


def run():
    i2=np.eye(2)
    x=np.array([[0.,1.],[1.,0.]])
    z=np.diag([1.,-1.])
    labels=list(product((-1,1),repeat=2))
    axis=(x,z)
    root=math.sqrt(2)
    states=[(i2+(a*x+b*z)/root)/2 for a,b in labels]
    joint=[(i2+(a*x+b*z)/root)/4 for a,b in labels]
    assert np.allclose(sum(joint),i2) and min(np.linalg.eigvalsh(m).min() for m in joint)>-1e-15
    assert max(abs(np.trace(r)-1) for r in states)<1e-15
    assert min(np.linalg.eigvalsh(r).min() for r in states)>-1e-15

    average_late=sum(np.trace(r@(i2+code[j]*axis[j])/2).real
                      for r,code in zip(states,labels) for j in (0,1))/8
    average_dephased=sum(np.trace(np.diag(np.diag(r))@(i2+code[j]*axis[j])/2).real
                         for r,code in zip(states,labels) for j in (0,1))/8
    best=0
    for enc in product((0,1),repeat=4):
        for dec in product((-1,1),repeat=4):
            n=sum(code[j]==dec[2*stored+j] for code,stored in zip(labels,enc) for j in (0,1))
            best=max(best,n)
    assert best==6 and np.isclose(average_late,(1+1/root)/2,atol=1e-15)

    # For any joint two-output-bit decoding POVM, payoff operator W_t
    # accounts for the uniform four states and uniform two scored bits.
    # W_t <= gamma I, so sum Tr(M_t W_t) <= gamma Tr(I).
    def fraction(a,b):
        return sum(u==v for u,v in zip(a,b))/2

    payoff=[sum(fraction(prediction,target)*rho for target,rho in zip(labels,states))/4
            for prediction in labels]
    max_eigen=max(np.linalg.eigvalsh(w)[-1] for w in payoff)
    achieved=sum(np.trace(m@w).real for m,w in zip(joint,payoff))
    bound=2*max_eigen
    assert abs(achieved-.75)<2e-15 and abs(bound-.75)<2e-15

    # A deterministic unrestricted real scalar can encode all four alternatives.
    messages=dict(zip(labels,(0.,1.,2.,3.)))
    lookup={q:b for b,q in messages.items()}
    scalar_replay=sum(lookup[messages[z]][j]==z[j] for z in labels for j in (0,1))/8
    assert scalar_replay==1.
    return dict(status='PASS_QUERY_TIME_AND_PRECISION_BOUNDARY',classical_one_bit_optimum=best/8,
                classic_encoders=16,classic_decoders=16,single_late_quantum_readout=float(average_late),
                same_fixed_code_dephased_late_readout=float(average_dephased),
                simultaneous_one_copy_for_this_code_optimum=float(achieved),
                simultaneous_one_copy_for_this_code_upper_bound=float(bound),
                noiseless_real_scalar_lookup=scalar_replay,
                scope='Uniform 2-bit fixed code fixture. No mapping to Re-TACRED labels, finite-precision constraint, or quantum efficiency claim.',
                task_forward=0,updates=0,gpu_allocations=0)


if __name__=='__main__':
    print(json.dumps(run(),indent=2,allow_nan=False))