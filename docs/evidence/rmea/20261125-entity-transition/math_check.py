"""Independent finite-dimensional transition and adverse-case calculations."""
from pathlib import Path
import json
import numpy as np

def pure_reference(states,transition,relation,smask,omask,population=False,drop=False):
    s=states[smask];o=states[omask]
    if drop:s=np.eye(states.shape[-1],dtype=states.dtype)[:1]
    result=np.zeros((len(relation),len(transition)))
    for r,v in enumerate(relation):
        for t,u in enumerate(transition):
            for a in s:
                for b in o:
                    if population:result[r,t]+=np.dot(abs(v)**2@abs(b)**2,abs(u)**2@abs(a)**2)
                    else:result[r,t]+=abs(np.vdot(v@b,u@a))**2
            result[r,t]/=len(s)*len(o)
    return result

def run():
    rng=np.random.default_rng(20261125);worst=0.
    for complex_input in (False,True):
        for d in (2,8,12):
            def unitary():
                a=rng.normal(size=(d,d))
                if complex_input:a=a+1j*rng.normal(size=(d,d))
                return np.linalg.qr(a)[0]
            states=np.stack([unitary()[:,0] for _ in range(4)])
            trans=np.stack([unitary() for _ in range(3)]);rel=np.stack([unitary() for _ in range(2)])
            smask=np.array([1,1,0,0],bool);omask=~smask
            s=states[smask].T@states[smask].conj()/2;o=states[omask].T@states[omask].conj()/2
            for pop in (False,True):
                ref=pure_reference(states,trans,rel,smask,omask,pop)
                ans=[]
                for v in rel:
                    row=[]
                    for u in trans:
                        if pop:p=np.dot(abs(v)**2@o.diagonal().real,abs(u)**2@s.diagonal().real)
                        else:p=np.trace(v@o@v.conj().T@u@s@u.conj().T).real
                        row.append(p)
                    ans.append(row)
                worst=max(worst,float(np.max(abs(np.array(ans)-ref))))
                assert np.min(ref)>=-1e-12 and np.max(ref)<=1+1e-12
    e0=np.array([1.,0.]);plus=np.ones(2)/np.sqrt(2);identity=np.eye(2);z=np.diag([1.,-1.])
    assert np.allclose(np.outer(identity@e0,identity@e0),np.outer(z@e0,z@e0))
    witness=[float(abs(np.vdot(plus,u@plus))**2) for u in (identity,z)]
    native=.5;transition=(witness[1]+1e-6)/(sum(witness)+2e-6)
    assert np.allclose(witness,[1.,0.]) and transition<native
    assert worst<1e-12
    return dict(status='PASS',numpy_max_error=worst,completion_witness=witness,same_vacuum_density=True,adverse_gold_score=dict(native=native,transition=transition),scope='Known quantum transition tools; no useful-task or new-theory guarantee')

if __name__=='__main__':
    result=run();Path(__file__).with_name('math-check.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
