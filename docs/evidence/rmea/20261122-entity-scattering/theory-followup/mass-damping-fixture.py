"""Pure NumPy geometry fixtures; no model, data, fitting, CUDA or network."""
import json,hashlib
from pathlib import Path
import numpy as np

def evaluate(states,q,e,eta=.1,gamma=1.):
    assert states.shape==(len(q),8) and np.all(q>0) and abs(q.sum()-1)<1e-12
    assert np.allclose(np.linalg.norm(states,axis=1),1,atol=1e-12,rtol=0)
    g=states.conj()@states.T
    h=np.sqrt(q[:,None]*q[None,:])*(g-np.eye(len(q)))+np.diag(e)
    lead=np.full(len(q),eta);lead[:2]+=gamma/2
    a=-h+1j*np.diag(lead)
    inverse=np.linalg.solve(a,np.eye(len(q)))
    transmission=float(gamma**2*abs(inverse[0,1])**2)
    rho=float(q[2:].sum());coeff=q[2:]/(q[2:]-e[2:]+1j*eta)
    c=float(min(1.,abs(g[0,1])))
    cap=rho/np.sqrt(rho*rho+4*eta*eta)
    b=min(float(cap),float(abs(coeff).sum()))
    gap=max(0.,c*np.sqrt(max(0.,1-b*b))-np.sqrt(max(0.,1-c*c))*b)
    f=q[:2]-e[:2]+1j*(eta+gamma/2)
    k0=1+abs(coeff).sum();k1=1+rho/eta;dmax=max(q[:2]/abs(f))
    lower=float(gamma**2*q[0]*q[1]*gap**2/(abs(f.prod())**2*k0**2*(1+(1+c)*k1*dmax)**4))
    assert np.linalg.norm(h-h.conj().T)<1e-12
    assert lower<=transmission+1e-13
    return dict(transmission=transmission,lower_bound=lower,endpoint_overlap=c,cap=float(cap),effective_cap=b,exceeds_epsilon=bool(lower>1e-12))

def main():
    eta=.1;q=np.array([.1,.1,.3,.5]);rho=q[2:].sum();e=np.array([0.,0.,q[2]-rho/2,q[3]-rho/2])
    a=rho/(rho/2+1j*eta);v=np.eye(8,dtype=complex)[0];w=np.eye(8,dtype=complex)[1]
    context=(v+np.exp(-1j*np.angle(a))*w)/np.sqrt(2)
    cmat=np.eye(8)-a*np.outer(context,context.conj())
    obj=cmat@v;obj/=np.linalg.norm(obj)
    states=np.array([w,obj,context,context]);sharp=evaluate(states,q,e)
    assert abs(sharp['endpoint_overlap']-sharp['cap'])<1e-12
    assert sharp['transmission']<1e-24
    overlap=.999
    states[1]=np.sqrt(1-overlap**2)*v+overlap*w
    finite=evaluate(states,q,e);assert finite['lower_bound']>1e-12
    rng=np.random.default_rng(20261122);rows=[];max_ratio=0.
    for n in (3,5,11,33):
        for _ in range(12):
            qi=rng.dirichlet(np.ones(n));ei=rng.uniform(-.9,.9,n)
            st=rng.normal(size=(n,8))+1j*rng.normal(size=(n,8));st/=np.linalg.norm(st,axis=1)[:,None]
            # Include deliberately near-parallel endpoints so the finite bound is tested nonvacuously.
            st[0]=v;st[1]=np.sqrt(1-overlap**2)*w+overlap*v
            row=evaluate(st,qi,ei);rows.append(row)
            coeff=qi[2:]/(qi[2:]-ei[2:]+1j*eta)
            for _ in range(16):
                t=rng.uniform(0,1,n-2);alpha=1-np.dot(coeff,t);beta=np.dot(abs(coeff),np.sqrt(t*(1-t)))
                sine=beta/np.sqrt(abs(alpha)**2+beta**2)
                assert sine<=row['cap']+1e-12
                max_ratio=max(max_ratio,float(sine/row['cap']))
    result=dict(schema='bounded-math-fixture.v1',kind='free_state_relaxation_only',sharp=sharp,finite=finite,random_cases=len(rows),nonvacuous_cases=sum(x['lower_bound']>0 for x in rows),bounds_above_epsilon=sum(x['exceeds_epsilon'] for x in rows),max_sampled_cap_ratio=max_ratio,minimum_transmission_minus_bound=min(x['transmission']-x['lower_bound'] for x in rows),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    Path(__file__).with_name('mass-damping-fixture-result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result))

if __name__=='__main__':main()
