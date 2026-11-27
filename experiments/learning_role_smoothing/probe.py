from pathlib import Path
from fractions import Fraction
import hashlib,itertools,json,math,os,time
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
H=Path(__file__).resolve().parent;protocol=json.loads((H/'protocol.json').read_bytes())
start=time.perf_counter();tol=protocol['tolerance'];I=np.eye(2,dtype=complex)
def gate(y,z):
    c=math.cos(y/2);s=math.sin(y/2)
    return np.diag([np.exp(-.5j*z),np.exp(.5j*z)])@np.array([[c,-s],[s,c]])
angles=protocol['fixed_fixture']['input_gate_angles'];U=[gate(*a) for a in angles];n=len(U);d=2
effects=np.asarray([(np.outer(v:=gate(.2+.013*r,.1+.07*r)[:,0],v.conj())+I/d)/2 for r in range(40)])
rho=np.diag([1.,0.]).astype(complex)
def channel(a,u):return (a+u@a@u.conj().T)/2
def adjoint(a,u):return (a+u.conj().T@a@u)/2
def smoother(units,initial):
    states=[initial]
    for u in units:states.append(channel(states[-1],u))
    suffix=effects.copy();masses=np.empty((40,n));complements=np.empty_like(masses)
    for t in reversed(range(n)):
        selected=units[t]@states[t]@units[t].conj().T/2
        masses[:,t]=np.einsum('rij,ji->r',suffix,selected).real
        complements[:,t]=np.einsum('rij,ji->r',suffix,states[t]/2).real
        suffix=np.asarray([adjoint(e,units[t]) for e in suffix])
    p=np.einsum('rij,ji->r',effects,states[-1]).real
    assert np.max(np.abs(masses+complements-p[:,None]))<tol
    assert min(np.linalg.eigvalsh(s).min() for s in states)>-tol
    assert max(abs(np.trace(s)-1) for s in states)<tol
    return masses,masses/masses.sum(-1,keepdims=True),p
g,alpha,p=smoother(U,rho)
enum=np.zeros_like(g);enum_p=np.zeros(40);real_enum=np.zeros_like(g)
def real4(u):return np.block([[u.real,-u.imag],[u.imag,u.real]])
real_effects=np.asarray([real4(e) for e in effects])
for path in itertools.product([0,1],repeat=n):
    psi=np.array([1,0],complex);realpsi=np.array([1,0,0,0],float)
    for u,b in zip(U,path):
        if b:psi=u@psi;realpsi=real4(u)@realpsi
    probs=np.einsum('i,rij,j->r',psi.conj(),effects,psi).real/(2**n)
    real_probs=np.einsum('i,rij,j->r',realpsi,real_effects,realpsi)/(2**n)
    enum_p+=probs
    for t,b in enumerate(path):
        if b:enum[:,t]+=probs;real_enum[:,t]+=real_probs
assert np.max(np.abs(g-enum))<tol and np.max(np.abs(p-enum_p))<tol
assert np.max(np.abs(g-real_enum))<tol
assert np.max(np.abs(alpha.sum(-1)-1))<tol and g.min()>=1/(4*d)-tol
assert alpha.max()<=(d+1)/(n+d)+tol
replacement=[*U[:-1],gate(*protocol['fixed_fixture']['last_word_replacement'])]
g2,a2,p2=smoother(replacement,rho)
ratio_before=float(g[0,0]/g[0,1]);ratio_after=float(g2[0,0]/g2[0,1])
assert abs(ratio_after-ratio_before)>protocol['witness_minimum']
gm,am,pm=smoother(U,I/d);assert np.max(np.abs(am-1/n))<tol
assert np.max(np.abs(gm-1/(2*d)))<tol
values=np.arange(12).reshape(4,3)/10
assert np.max(np.abs(alpha@values-(enum/enum.sum(-1,keepdims=True))@values))<tol
# Born instrument counterexample: same law as source supplementA8, not bodyEq2 shortcut.
plus=np.ones((2,2),complex)/2;projections=[np.diag([1,0]),np.diag([0,1])]
numerators=[float(np.trace(proj@plus@proj@plus).real) for proj in projections]
assert numerators==[.25,.25] and float(np.trace(plus@plus).real)==1.
exact={'numerator_each':str(Fraction(1,4)),'printed_denominator':str(Fraction(1)),
 'printed_probability_sum':str(Fraction(1,2)),'instrument_denominator':str(Fraction(1,2)),'correct_probability_each':str(Fraction(1,2))}
seconds=time.perf_counter()-start;assert seconds<protocol['budget']['CPU_compute_seconds']
out={'status':'PASS_ONE_EXPLORATORY_SMOOTHING_ROLE_CHECK','seconds':seconds,'protocol_sha256':hashlib.sha256((H/'protocol.json').read_bytes()).hexdigest(),
 'fixture':protocol['fixed_fixture'],'prefix_suffix_vs_16history_error':float(np.max(np.abs(g-enum))),
 'independent_real4_history_error':float(np.max(np.abs(g-real_enum))),'terminal_probability_error':float(np.max(np.abs(p-enum_p))),
 'minimum_g':float(g.min()),'minimum_bound':1/(4*d),'max_alpha':float(alpha.max()),'max_alpha_bound':(d+1)/(n+d),
 'future_ratio_witness':{'class':0,'word_indices':[0,1],'changed_word':3,'before':ratio_before,'after':ratio_after,'delta':ratio_after-ratio_before,
 'scope':'Fixed local gates/effect, changes relative weights through future channel; not better task predictions or whole-model separation from contextual classical attention.'},
 'maxmixed_uniform_alpha_error':float(np.max(np.abs(am-1/n))),'source_printed_formula_counterexample':exact,
 'first_class_g':g[0].tolist(),'first_class_alpha':alpha[0].tolist(),'first_class_pooled_value':(alpha@values)[0].tolist(),
 'operational_scope':'Classical fair skip/apply coins with coherent quantum memory and terminalbinaryeffect. No coherence between coin branches, no gold future observation, no physical40way measurement claim.',
 'new_task_rows':0,'new_task_forwards':0,'new_fits':0,'GPU':0,'external_model_calls':0,'new_packages':0,
 'limits':['Synthetic qubit check; original native3qubit input-angle reachability, training gradients, source-neighbor novelty, production code and actual task cost unproved.',
 'No independent researcher validation, empirical utility, newT, L3 or L4-T. Pauli/real replay is not an independently trained strong classical control.',
 'Maxmixed collapse and bounded concentration are genuine restrictions; unitary-mixture memory can mix away context and has no guaranteed long-range benefit.']}
(H/'probe.json').write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(out,ensure_ascii=False))
