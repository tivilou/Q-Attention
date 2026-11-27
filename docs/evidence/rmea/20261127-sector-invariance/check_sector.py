"""Fixed artificial amplitudes only: dense reference vs sector recurrence."""
import hashlib,json,math
from fractions import Fraction as F
from pathlib import Path
import numpy as np

H=Path(__file__).resolve().parent
I2=np.eye(2,dtype=complex)
X=np.array([[0,1],[1,0]],complex)
Z=np.diag([1,-1]).astype(complex)
HI=np.array([[1,1],[1,-1]],complex)/math.sqrt(2)
Yp=np.array([[1,-1j],[1j,1]],complex)/2

def sector_recurrence(initial,flags,ops):
    """Use sector masses and ancillary states; no word-space matrices."""
    count=5 # predeclared includes one zero-mass sector
    p=np.bincount(flags,weights=abs(initial)**2,minlength=count)
    a=np.tile(np.array([1.,0.],complex),(count,1))
    for tag,value in ops:
        if tag=='reflect':
            a=a-2*(value@(p@a))[None,:]
        else:
            a=np.einsum('kij,kj->ki',value,a)
    reconstructed=initial[:,None]*a[flags]
    return p,a,reconstructed

def dense(initial,flags,ops):
    """Independent exact operator construction on word x flag x ancilla."""
    n=len(initial);x=np.kron(initial,np.array([1.,0.],complex))
    for tag,value in ops:
        if tag=='reflect':
            W=np.eye(n*2)-2*np.kron(np.outer(initial,initial.conj()),value)
        else:
            W=np.zeros((n*2,n*2),complex)
            for j,s in enumerate(flags):
                W[2*j:2*j+2,2*j:2*j+2]=value[s]
        assert np.max(abs(W.conj().T@W-np.eye(n*2)))<1e-12
        x=W@x
    return x.reshape(n,2)

def close(a,b):
    return float(np.max(abs(np.asarray(a)-np.asarray(b))))

def fixture():
    # Coordinates represent four words, each with class failure/success state.
    # Words 0/1 and 2/3 occupy the two roles, respectively.
    f=np.array([0,1,0,1,2,3,2,3])
    psi=np.array([1,2j,3,-1,1-1j,2,3j,4-1j],complex);psi/=np.linalg.norm(psi)
    gate=np.array([HI, X, np.diag([1,1j]), HI@Z, I2])
    gate2=np.array([Z,I2,HI, X, I2])
    pre=[('control',gate),('reflect',Yp),('control',gate2),('reflect',I2)]
    return f,psi,pre

def probabilities(x):
    # Ancilla measuring |+> (a coherent final operation, not inferred routing).
    prob=abs(x@np.array([1,1])/math.sqrt(2))**2
    return prob.reshape(4,2)[:,1]

f,psi,ops=fixture()
p,a,fast=sector_recurrence(psi,f,ops);slow=dense(psi,f,ops)
errors={'dense_vs_sector':close(fast,slow),'norm':abs(float(np.linalg.norm(fast)**2)-1)}
assert p[4]==0 and np.isfinite(a).all()
good=probabilities(fast)
marg0=abs(psi[1::2])**2
for i,j in ((0,1),(2,3)):
    assert min(good[i],good[j])>0
    errors[f'in_role_ratio_{i}_{j}']=abs(float(good[i]/good[j]-marg0[i]/marg0[j]))
# Trainable, within-sector rephasing of encodings cannot introduce sensitivity
# if preparation reflection follows the new preparation and flags stay fixed.
rot=np.exp(1j*np.arange(8)/7)
_,_,fast2=sector_recurrence(psi*rot,f,ops)
errors['independent_within_sector_rephase_probability']=close(probabilities(fast2),good)
# Full signed 64-value readout: equality to direct dense distribution.
v=np.arange(4*64,dtype=float).reshape(4,64)/128-1.
weights=(good+1e-6)/(good+1e-6).sum()
q=probabilities(slow);q=(q+1e-6)/(q+1e-6).sum()
errors['full_64_value_pool']=close(weights@v,q@v)
# Epsilon inserted after the quantum operation is an external uniform floor.
# It can alter within-sector final normalized ratios even for one good sector,
# despite exact invariance of pre-floor probability ratios.
g=np.array([2.,8.]);manual=lambda c:(c*g+1e-6)/(c*g+1e-6).sum()
eps_only_delta=close(manual(1),manual(.01));assert eps_only_delta>1e-7
# A straightforward mutant applies a noncommuting mix in the word register.
# Equal flag mass and same coordinate probabilities, two opposite input phases.
u=[F(1),F(0),F(2),F(0)]
def rational_mixed(sign):
    x=[F(1),F(2),F(3*sign),F(4)]
    product=sum(a*b for a,b in zip(u,x));den=sum(a*a for a in u)
    y=[a-2*b*product/den for a,b in zip(x,u)]
    assert sum(z*z for z in x)==sum(z*z for z in y)==30
    return y,(y[0]/y[1])**2
wp,rp=rational_mixed(1);wm,rm=rational_mixed(-1)
assert rp==F(81,100) and rm==F(9,4)
v0=np.array(u,float)/math.sqrt(5);P=np.outer(v0,v0)
G=np.diag([1.,1.,0.,0.]);R=np.eye(4)-2*P
errors['noncommuting_reference_plus']=close(R@np.array([1,2,3,4.]),[float(x) for x in wp])
errors['noncommuting_reference_minus']=close(R@np.array([1,2,-3,4.]),[float(x) for x in wm])
commutator=float(np.linalg.norm(P@G-G@P))
assert abs(commutator-2*math.sqrt(2)/5)<1e-12 and max(errors.values())<1e-12
result=dict(status='PASS_FIXED_ARTIFICIAL_SECTOR_INVARIANCE_AND_NONCOMMUTING_BOUNDARY',
    errors=errors,tolerance=1e-12,commuting_sectors_mass=p.tolist(),joint_word_good_and_ancilla_unfloored=good.tolist(),
    epsilon_floor_only_distribution_max_delta=eps_only_delta,
    noncommuting_fixture=dict(raw_input_good_word_ratio='1/4',plus_after='81/100',minus_after='9/4',commutator_frobenius=commutator),
    data_rows=0,task_forwards=0,optimizer_updates=0,gpu_allocations=0,
    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
with (H/'sector-check.json').open('x',encoding='utf-8') as o:json.dump(result,o,indent=2)
print(json.dumps(result))
