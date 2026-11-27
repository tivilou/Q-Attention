from pathlib import Path
from fractions import Fraction
import ast,hashlib,itertools,json,math,os,time
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
H=Path(__file__).resolve().parent
protocol=json.loads((H/'protocol.json').read_bytes());protocol_sha=hashlib.sha256((H/'protocol.json').read_bytes()).hexdigest()
start=time.perf_counter();d=8;tol=protocol['numeric_tolerance']
assert protocol['seed']==20261127 and protocol['budget']['CPU_compute_seconds']==10
rng=np.random.default_rng(protocol['seed'])
I2=np.eye(2,dtype=complex);I=np.eye(d,dtype=complex)
X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]],complex);Z=np.diag([1,-1]).astype(complex)
def kron3(a,b,c):return np.kron(np.kron(a,b),c)
def local(g,w):return kron3(*[g if i==w else I2 for i in range(3)])
def psd_power(a,p):
    w,v=np.linalg.eigh(a);assert w.min()>-1e-12
    if p<0:assert w.min()>0
    return (v*(np.maximum(w,0)**p))@v.conj().T
def circuit(angles):
    s=np.zeros(8,complex);s[0]=1
    for layer in range(2):
        for wire in range(3):
            y,z=angles[layer,wire];c=math.cos(y/2);v=math.sin(y/2)
            ry=np.array([[c,-v],[v,c]],complex);rz=np.diag([np.exp(-.5j*z),np.exp(.5j*z)])
            s=local(rz@ry,wire)@s
        for control,target in [(0,1),(1,2),(2,0)]:
            perm=[i^((1<<(2-target)) if (i>>(2-control))&1 else 0) for i in range(8)]
            s=s[perm]
    return s
def circuit_reference(angles):
    s=[0j]*8;s[0]=1+0j
    for layer in range(2):
        for w in range(3):
            y,z=angles[layer,w];a=complex(math.cos(y/2));b=complex(math.sin(y/2));bit=1<<(2-w)
            before=s[:]
            for j in range(8):
                if j&bit:continue
                u,v=before[j],before[j|bit]
                s[j]=(a*u-b*v)*np.exp(-.5j*z)
                s[j|bit]=(b*u+a*v)*np.exp(.5j*z)
        for c,t in [(0,1),(1,2),(2,0)]:
            before=s[:]
            for j in range(8):s[j^((1<<(2-t)) if j&(1<<(2-c)) else 0)]=before[j]
    return np.asarray(s)

# Source pins and executable marker-only fixture: no torch import or task-model forward.
src=Path('D:/CodexHome/task-artifacts/qattn-role-exchange-audit-20261125/source')
pins={
 str(src/'audit_rmea_tokenization.py'):'5f7860e47a7a0979976e5acc435ed34acede4c96bfac799d79d3018366b6aef8',
 str(src/'rmea_run.py'):'769173b9ab17f76229b0db79b4a7625ac1574427d2d40491be7d5b19d4d0ff25',
 'D:/CodexHome/task-artifacts/qattn-relation-frame-20261125/source/baselines/retacred/rmea.py':'9f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0'}
for path,expected in pins.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
tree=ast.parse((src/'audit_rmea_tokenization.py').read_text(encoding='utf-8'))
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='marked_words')
namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned-marker-fixture>','exec'),namespace)
fixture={'tokens':['entity_a','met','entity_b','near','entity_c'],'subject':[0,1],'object':[2,3]}
left=namespace['marked_words'](fixture);right=namespace['marked_words']({**fixture,'object':[4,5]})
assert left!=right

z=rng.normal(size=(2,3072));W=rng.normal(scale=1/math.sqrt(3072),size=(12,3072))
theta=rng.normal(scale=.2,size=(40,12));beta=np.linspace(-.7,.7,40)
angles=math.pi*np.tanh((z@W.T)[:,None,:]+theta[None,:,:])
angles=angles.reshape(2,40,2,3,2)
vectors=np.asarray([[circuit(a) for a in row] for row in angles])
reference=np.asarray([[circuit_reference(a) for a in row] for row in angles])
gate_error=float(np.max(np.abs(vectors-reference)));assert gate_error<tol
rhos=np.einsum('qri,qrj->qrij',vectors,vectors.conj())
A=np.exp(beta)[None,:,None,None]*(rhos+I/d)/2
def frame(a):
    b=psd_power(a.sum(0),-.5)
    return np.asarray([b@r@b for r in a])
effects=np.asarray([frame(a) for a in A])
assert np.max(np.abs(effects.sum(1)-I))<tol
assert np.linalg.eigvalsh(effects).min()>0
single=effects.trace(axis1=-2,axis2=-1).real/d
J=np.einsum('rij,sji->rs',effects[0],effects[1]).real/d
assert J.min()>0 and np.max(np.abs(single.sum(-1)-1))<tol
assert np.max(np.abs(J.sum(1)-single[0]))<tol and np.max(np.abs(J.sum(0)-single[1]))<tol
J_direct=np.empty((40,40))
for r in range(40):
    k=psd_power(effects[0,r],.5)
    post=k@(I/d)@k.conj().T
    for s in range(40):J_direct[r,s]=np.trace(effects[1,s]@post).real
sequential_error=float(np.max(np.abs(J-J_direct)));assert sequential_error<tol
paulis=np.asarray([kron3(*a)/math.sqrt(d) for a in itertools.product([I2,X,Y,Z],repeat=3)])
coords=np.einsum('qrij,aji->qra',effects,paulis).real
replay=coords[0]@coords[1].T/d
classical_replay_error=float(np.max(np.abs(J-replay)));assert classical_replay_error<tol

finalD=effects*np.eye(d)[None,None,:,:]
single_finalD=finalD.trace(axis1=-2,axis2=-1).real/d
final_dephase_single_error=float(np.max(np.abs(single-single_finalD)));assert final_dephase_single_error<tol
J_finalD=np.einsum('rij,sji->rs',finalD[0],finalD[1]).real/d
upstreamD=np.asarray([frame(a*np.eye(d)[None,:,:]) for a in A])
single_upstreamD=upstreamD.trace(axis1=-2,axis2=-1).real/d
upstream_delta=float(np.max(np.abs(single-single_upstreamD)));assert upstream_delta>1e-6

# A pair-only learning direction: common effect conjugation preserves all singles.
G=local(Y/2,0);r,s=3,11
def rotate(t):return math.cos(t/2)*I-2j*math.sin(t/2)*G
def coupled_loss(t):
    u=rotate(t);e=u@effects[0,r]@u.conj().T
    return -math.log(float(np.trace(e@effects[1,s]).real/d))
derivative_matrix=-1j*(G@effects[0,r]-effects[0,r]@G)
analytic=-float(np.trace(derivative_matrix@effects[1,s]).real)/float(np.trace(effects[0,r]@effects[1,s]).real)
h=1e-5;numeric=(coupled_loss(h)-coupled_loss(-h))/(2*h)
assert abs(analytic-numeric)<protocol['finite_difference_tolerance'] and abs(analytic)>1e-7
rotated=np.asarray([rotate(.7)@e@rotate(.7).conj().T for e in effects[0]])
rotation_single_error=float(np.max(np.abs(rotated.trace(axis1=-2,axis2=-1).real/d-single[0])))
rotation_pair_delta=float(np.max(np.abs(np.einsum('rij,sji->rs',rotated,effects[1]).real/d-J)))
assert rotation_single_error<tol and rotation_pair_delta>1e-7

# Full-rank local marginals do not define a global classical three-label law.
eta=.9;directions=[(0,0,1),(math.sqrt(3)/2,0,-.5),(-math.sqrt(3)/2,0,-.5)]
trine=[]
for nx,ny,nz in directions:
    observable=nx*X+ny*Y+nz*Z
    trine.append(np.asarray([(I2+sign*eta*observable)/2 for sign in [-1,1]]))
correlations=[]
trine_min=1.
for a,b in [(0,1),(1,2),(2,0)]:
    p=np.einsum('rij,sji->rs',trine[a],trine[b]).real/2
    assert np.max(np.abs(p.sum(0)-.5))<tol and np.max(np.abs(p.sum(1)-.5))<tol
    correlations.append(float(p[0,0]+p[1,1]-p[0,1]-p[1,0]));trine_min=min(trine_min,float(np.linalg.eigvalsh(trine[a]).min()))
exact_pair=-Fraction(81,200);exact_sum=3*exact_pair
global_sums=[a*b+b*c+c*a for a,b,c in itertools.product([-1,1],repeat=3)]
assert min(global_sums)==-1 and exact_sum<Fraction(-1)
assert np.max(np.abs(np.asarray(correlations)-float(exact_pair)))<tol and trine_min>0

elapsed=time.perf_counter()-start;assert elapsed<protocol['budget']['CPU_compute_seconds']
out={'status':'PASS_ONE_COEDGE_COMPOSITE_MEASUREMENT_DESIGN_CHECK','protocol_sha256':protocol_sha,
 'seconds':elapsed,'state_dimension':8,'synthetic_queries':2,'class_measurement_outcomes':40,'native_order_synthetic_states':80,
 'gate_reference_error':gate_error,'sequential_vs_HS_error':sequential_error,'real_Pauli64_replay_error':classical_replay_error,
 'POVM_sum_error':float(np.max(np.abs(effects.sum(1)-I))),'minimum_effect_eigenvalue':float(np.linalg.eigvalsh(effects).min()),
 'minimum_joint_probability':float(J.min()),'joint_marginal_max_error':max(float(np.max(np.abs(J.sum(1)-single[0]))),float(np.max(np.abs(J.sum(0)-single[1])))),
 'pair_only_learning_direction':{'synthetic_labels':[r,s],'analytic_derivative':analytic,'central_difference':numeric,'absolute_error':abs(analytic-numeric),
    'single_marginal_rotation_error':rotation_single_error,'paired_joint_rotation_delta':rotation_pair_delta,
    'scope':'Direction in valid effect space, not proof the chosen native angle parametrization reaches that common conjugation direction or actual task gradients improve.'},
 'dephasing':{'final_effect_single_error':final_dephase_single_error,'final_effect_joint_max_change':float(np.max(np.abs(J-J_finalD))),
   'upstream_preparation_single_max_change':upstream_delta,'scope':'Final effect dephasing and upstream preparation dephasing are different interventions; no accuracy or CE gain inferred.'},
 'global_compatibility_counterexample':{'eta':eta,'minimum_effect_eigenvalue':trine_min,'pair_correlations':correlations,
   'exact_pair_correlation':str(exact_pair),'exact_three_pair_sum':str(exact_sum),'classical_three_bit_assignment_floor':min(global_sums),
   'interpretation':'Pair-composite score only; compatible singleton marginals do not certify a global joint-label model. Known elementary probability boundary, not newT or task evidence.'},
 'source_pins':pins,'marker_only_fixture':{'query_marker_sequences_differ':True,'same_entity_span_retained':True,
   'scope':'Structural query-conditioning; no actual encoder forward or hidden-state difference measured.'},
 'new_task_rows':0,'new_task_model_forwards':0,'new_fits':0,'GPU':0,'network':0,'new_packages':0,
 'limits':['One fully specified synthetic mathematical check; not production plugin, formal method admission or independent researcher validation.',
 'Pauli64 replay exposes an exact classical representation for this pair score; use independently trained strong controls, not simulability as a veto.',
 'Frame/POVM/Luders/HS identities are known tools, not original theory or novelty clearance.',
 'Input pooling, effective capacity, actual feature behavior, small-screen coedge coverage and cost, L3 utility and L4-T remain unproved.']}
(H/'probe.json').write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(out,ensure_ascii=False))
