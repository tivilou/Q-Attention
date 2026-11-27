"""Synthetic dimension-8/64/40 check. No native model, dataset or accelerator."""
from fractions import Fraction as F
from pathlib import Path
import hashlib,json,math
import numpy as np

EPS=1e-6  # retained native rmea.py
SIGNS=np.array([1]*4+[-1]*4)
J=np.kron(np.eye(32),np.array([[0.,-1.],[1.,0.]]))

def err(a,b):return float(np.max(np.abs(a-b)))

def explicit(psi,phi,eta):
    out=[]
    for b in (1,-1):
        k=np.diag(np.sqrt((1+b*eta*SIGNS)/2))
        out.append(np.array([[abs(np.vdot(f,k@p))**2 for f in phi] for p in psi]))
    return np.stack(out,axis=2)

def compact(psi,phi,eta):
    u=psi[:,:4]@phi[:,:4].conj().T
    v=psi[:,4:]@phi[:,4:].conj().T
    s=abs(u)**2+abs(v)**2+2*math.sqrt(1-eta*eta)*(u*v.conj()).real
    d=eta*(abs(u)**2-abs(v)**2)
    return s,d

def dense_general(psi,phi,eta,kind):
    q=np.outer(psi,psi.conj()); signs=SIGNS[:,None]==SIGNS[None,:]
    if kind=='blocks':q=q*signs
    if kind=='full':q=q*np.eye(8)
    e=np.outer(phi,phi.conj())
    p=[]
    for i in (1,-1):
        k=np.diag(np.sqrt((1+i*eta*SIGNS)/2))
        p.append(float(np.trace(e@k@q@k).real))
    return p

def angle(z):return (1-z*z)/(1+z*z),2*z/(1+z*z)

def read_explicit(probs,values,mask,z,w):
    c,s=angle(z); num=np.zeros((w.shape[0],64)); den=np.zeros(w.shape[0])
    for t in np.flatnonzero(mask):
        for k,b in enumerate((1,-1)):
            mass=probs[t,:,k]+EPS/2
            num+=mass[:,None]*((c*np.eye(64)+b*s*J)@values[t])[None,:]
            den+=mass
    assert (den>0).all()
    features=num/den[:,None]
    return features,np.sum(w*features,axis=1)

def read_compact(s,d,values,mask,z,w):
    c,t=angle(z); sw=(s+EPS)*mask[:,None]; dw=d*mask[:,None]
    den=sw.sum(axis=0)[:,None]
    p=sw.T@values/den; q=dw.T@values/den
    features=c*p+t*q@J.T
    return features,np.einsum('rd,rd->r',w,features)

def arbitrary(rng,n):
    a=rng.normal(size=(n,8))+1j*rng.normal(size=(n,8))
    return a/np.linalg.norm(a,axis=1,keepdims=True)

def test():
    rng=np.random.default_rng(20261127)
    psi=arbitrary(rng,7); phi=arbitrary(rng,40)
    values=rng.normal(size=(7,64)); w=rng.normal(size=(40,64));mask=np.array([1,0,1,1,0,1,1],bool)
    errors={}; beta=0.
    for eta in (0.,.6,1.):
        probs=explicit(psi,phi,eta); s,d=compact(psi,phi,eta)
        errors['born_sum']=max(errors.get('born_sum',0),err(s,probs.sum(-1)))
        errors['born_difference']=max(errors.get('born_difference',0),err(d,probs[:,:,0]-probs[:,:,1]))
        assert np.all(s>=abs(d)-1.e-14) and np.all(s<=1+1.e-14)
        for z in (0.,.5,-.3):
            a,b=read_explicit(probs,values,mask,z,w);c,h=read_compact(s,d,values,mask,z,w)
            errors['full64_features']=max(errors.get('full64_features',0),err(a,c))
            errors['full40_logits']=max(errors.get('full40_logits',0),err(b,h))
            alt=values.copy();alt[~mask]=5.5e4
            errors['mask']=max(errors.get('mask',0),err(read_compact(s,d,alt,mask,z,w)[1],h))
            flip=read_compact(s,-d,values,mask,-z,w)[1]
            errors['branch_relabel']=max(errors.get('branch_relabel',0),err(flip,h))
        if eta==.6:
            for a in range(psi.shape[0]):
                for b in range(phi.shape[0]):
                    p=dense_general(psi[a],phi[b],eta,'blocks')
                    beta=max(beta,abs((p[0]-p[1])-d[a,b]))
    errors['raw_contrast_block_dephase']=beta
    assert max(errors.values())<1.e-12

    # Independent rational witness: all populations identical, equal native and measured mass.
    eta=F(3,5); tau=F(4,5); features=[]
    f=[F(1,2)]*4
    for weights in ((1,1,1,-1),(1,-1,1,1)):
        p=[F(x,2) for x in weights]
        u=sum(a*b for a,b in zip(f[:2],p[:2]));v=sum(a*b for a,b in zip(f[2:],p[2:]))
        native=(u+v)**2
        s=u*u+v*v+2*tau*u*v;d=eta*(u*u-v*v)
        arm_marginal=(1+eta*(sum(x*x for x in p[:2])-sum(x*x for x in p[2:])))/2
        assert native==s==F(1,4) and arm_marginal==F(1,2) and d!=0
        features.append((native,s,d,arm_marginal))
    left,right=features
    assert left[2]==-right[2]==F(3,20)
    no_floor_gap=F(4,5)*(left[2]-right[2])/left[1]
    assert no_floor_gap==F(24,25)
    q=np.zeros((2,8));q[:,[0,1,4,5]]=np.array([[1,1,1,-1],[1,-1,1,1]])/2
    p=np.zeros((1,8));p[0,[0,1,4,5]]=.5
    synthetic=[];dims=np.zeros((1,64));dims[0,0]=1
    dec=np.zeros((1,64));dec[0,1]=1
    for k in range(2):
        a,b=compact(q[k:k+1],p,float(eta));logit=read_compact(a,b,dims,np.ones(1,dtype=bool),.5,dec)[1][0]
        dblock=dense_general(q[k],p[0],float(eta),'blocks')
        dfull=dense_general(q[k],p[0],float(eta),'full')
        assert abs(float(b[0,0])-(dblock[0]-dblock[1]))<1.e-14
        assert abs(dfull[0]-dfull[1])<1.e-14
        synthetic.append(float(logit))
    gap=synthetic[0]-synthetic[1]
    assert abs(gap-float(no_floor_gap)*.25/(.25+EPS))<5.e-15

    # Identifiability of the exact function only, not a claim about a fitted classical control.
    # Scalar rank-one branch weights and the shared 2x2 per-pair gate can be independently learned.
    # Artificial example defeats fixed decoder on one pooled native vector; not this two-pool model.
    return {'status':'PASS_EXACT_REDUCTION_AND_CONDITIONAL_WITNESS',
            'errors':errors,'paired_equal_native_and_sum_mass':'1/4',
            'paired_equal_bare_arm_probability':'1/2','paired_raw_contrasts':['3/20','-3/20'],
            'standalone_witness_logits':synthetic,'gap_with_native_eps':gap,'gap_without_eps':str(no_floor_gap),
            'full_dephase_gap_in_raw_contrast':0,
            'block_dephase_contrast_invariant':True,
            'scope':'Artificial unrestricted 3-qubit states. No claim of exact native reachability, QML utility, asymptotic speed or an independently fitted control.',
            'dimensions':{'qubits':3,'synthetic_words':7,'signed_value':64,'independent_class_effects':40},
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'task_rows':0,'task_forwards':0,'optimizer_updates':0,'gpu_allocations':0}

if __name__=='__main__':print(json.dumps(test(),indent=2,allow_nan=False))
