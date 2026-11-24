"""Known postselection geometry: a native-circuit counterexample to metric equality.

NumPy FP64 reference, no data/weights/optimization. Not a novelty or utility claim.
"""
from pathlib import Path
import json,math
import numpy as np

def circuit(angles):
    state=np.zeros(angles.shape[:-3]+(8,),np.complex128);state[...,0]=1
    for layer in range(2):
        for wire in range(3):
            shape=state.shape;pair=state.reshape(shape[:-1]+(2**wire,2,2**(2-wire)))
            a,b=np.take(pair,0,axis=-2),np.take(pair,1,axis=-2)
            y,z=angles[...,layer,wire,0],angles[...,layer,wire,1]
            c,s=np.cos(y/2)[...,None,None],np.sin(y/2)[...,None,None]
            minus,plus=np.exp(-.5j*z)[...,None,None],np.exp(.5j*z)[...,None,None]
            state=np.stack(((c*a-s*b)*minus,(s*a+c*b)*plus),axis=-2).reshape(shape)
        for control,target in ((0,1),(1,2),(2,0)):
            permutation=[i^((1<<(2-target)) if (i>>(2-control))&1 else 0) for i in range(8)]
            state=state[...,permutation]
    return state

def query(e):
    angles=np.zeros((2,3,2));angles[1,2,0]=math.pi*math.tanh(e)
    return circuit(angles)

def main():
    rows=[];maximum=0.;h=1e-8;q=query(0);dq=(query(h)-query(-h))/(2*h)
    qfi=float(4*(np.vdot(dq,dq).real-abs(np.vdot(q,dq))**2))
    assert abs(qfi-math.pi**2)<1e-11
    for delta in (.3,.1,.03,.01,.003,.001):
        angles=np.zeros((2,2,3,2));angles[:,1,2,0]=np.array([1,-1])*(math.pi-2*delta)
        keys=circuit(angles);s,c=math.sin(delta),math.cos(delta)
        for eps in (0.,1e-6):
            def alpha(e):
                p=abs(keys.conj()@query(e))**2+eps
                return p/p.sum()
            a=alpha(0);da=(alpha(h)-alpha(-h))/(2*h)
            observed=float((da**2/a).sum());expected=math.pi**2*s*s*c*c/(s*s+eps)**2
            relative=abs(observed-expected)/expected;maximum=max(maximum,relative)
            assert relative<2e-8
            rows.append(dict(delta=delta,epsilon=eps,query_fisher=qfi,
                attention_fisher=observed,analytic_attention_fisher=expected,
                attention_to_query_ratio=observed/qfi,attention=a.tolist(),
                scaled_subPOVM_success_probability=s*s,
                zero_epsilon_success_times_fisher_over_QFI=c*c if eps==0 else None))
    report=dict(status='VERIFIED',rows=rows,finite_difference_step=h,
        maximum_relative_error=maximum,quantum_Fisher_qfi=qfi,
        native_access='Only last-layer wire2 RY differs; angles remain inside (-pi,pi), original fixed CNOT ring twice. Query e=0.',
        result='Physical pure-state Fisher and word-normalized attention Fisher need not agree; normalization can amplify the metric. Standard conditional-information/postselection fact, not new T or observed failure cause.',
        training_updates=0,data_rows=0,checkpoints=0,cos_writes=0)
    (Path(__file__).parent/'conditional-fisher.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],maximum_relative_error=maximum,
        native_epsilon_ratios=[r['attention_to_query_ratio'] for r in rows if r['epsilon']==1e-6])))

if __name__=='__main__':main()
