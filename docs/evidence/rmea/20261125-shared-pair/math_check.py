"""Independent dense mathematical falsifiers; no natural inputs or model fits."""
from pathlib import Path
import json,numpy as np
EPS=1e-6


def effect(qr,qs):
    c=np.vdot(qr,qs);t=abs(c)/max(abs(c),EPS);eta=c/max(abs(c),EPS)
    v=qr-eta.conjugate()*qs
    return (np.outer(v,v.conj())+(1-t*t)*np.outer(qs,qs.conj()))/2


def numpy_pairs(psi,query,utility,bias,mask,intensity=False):
    classes=len(query);z=np.array(bias,dtype=float);weights={}
    for r in range(classes):
        for s in range(r+1,classes):
            e=effect(query[r],query[s]);g=(np.outer(query[r],query[r].conj())+np.outer(query[s],query[s].conj()))/2
            if intensity:e=np.trace(e).real/np.trace(g).real*g
            p=np.einsum('ti,ij,tj->t',psi.conj(),e,psi).real
            a=(p+EPS)*mask;a/=a.sum();weights[r,s]=a
            d=a@(utility[r]-utility[s]);z[r]+=d/classes;z[s]-=d/classes
    return z,weights


def main():
    rng=np.random.default_rng(20261125);maximum=0.;smallest=1.;largest=0.
    for _ in range(128):
        q=rng.normal(size=(4,8))+1j*rng.normal(size=(4,8));q/=np.linalg.norm(q,axis=-1,keepdims=True)
        psi=rng.normal(size=(7,8))+1j*rng.normal(size=(7,8));psi/=np.linalg.norm(psi,axis=-1,keepdims=True)
        u=rng.normal(size=(4,7));b=rng.normal(size=4);mask=np.array([1,1,1,1,1,0,0])
        for r in range(4):
            for s in range(r+1,4):
                e=effect(q[r],q[s]);v=np.linalg.eigvalsh(e);smallest=min(smallest,float(v.min()));largest=max(largest,float(v.max()))
                a=q[r].conj()@psi.T;aa=q[s].conj()@psi.T;c=np.vdot(q[r],q[s]);eta=c/max(abs(c),EPS)
                raw=(abs(a)**2+abs(aa)**2)/2-(a.conj()*eta*aa).real
                maximum=max(maximum,float(np.max(abs(raw-np.einsum('ti,ij,tj->t',psi.conj(),e,psi).real))))
        z,weights=numpy_pairs(psi,q,u,b,mask)
        phaseq=np.exp(1j*rng.normal(size=(4,1)));phasepsi=np.exp(1j*rng.normal(size=(7,1)))
        zz,_=numpy_pairs(psi*phasepsi,q*phaseq,u,b,mask)
        maximum=max(maximum,float(np.max(abs(z-zz))))
    assert maximum<1e-12 and smallest>-1e-12 and largest<=1+1e-12
    # A third decoder has a nonzero coefficient in a rival margin at fixed routing.
    k=2;token=2;h=1e-5;changed=u.copy();changed[k,token]+=h
    base,_=numpy_pairs(psi,q,u,b,mask);other,_=numpy_pairs(psi,q,changed,b,mask)
    coefficient=(weights[1,k][token]-weights[0,k][token])/4
    numeric=((other[0]-other[1])-(base[0]-base[1]))/h
    assert abs(numeric-coefficient)<1e-9 and abs(coefficient)>1e-4
    # Condorcet winner0 can lose Borda projection: no classification guarantee.
    d=np.array([[0.,1.,1.],[-1.,0.,5.],[-1.,-5.,0.]])
    vote=d.mean(-1);assert np.all(d[0,1:]>0) and vote.argmax()!=0
    result=dict(status='PASS',trials=128,max_formula_or_gauge_error=maximum,effect_min_eigenvalue=smallest,effect_max_eigenvalue=largest,
        third_decoder_coefficient=coefficient,finite_difference_coefficient=numeric,
        condorcet_counterexample=dict(edges=d.tolist(),projected_logits=vote.tolist(),winner=int(vote.argmax())),
        scope='Known PSD/phase/voting mathematics and fixed-readout distinction only; no natural task gain or new T')
    Path(__file__).with_name('math-check.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))


if __name__=='__main__':main()
