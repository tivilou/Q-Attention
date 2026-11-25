from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math
import numpy as np
r=Path(__file__).resolve().parent
parent=Path('D:/CodexHome/task-artifacts/qattn-native-phase-construction-20261126')
def sha(b):return hashlib.sha256(b).hexdigest()
def unpack(x):return np.asarray(x['real'])+1j*np.asarray(x['imag']) if 'real' in x else np.asarray(x['values'])
def packed(x):
    x=np.asarray(x);return dict(real=x.real.tolist(),imag=x.imag.tolist())
assert not (r/'scope.json').exists()
stage=json.loads((parent/'native-stage.json').read_bytes());oldscope=json.loads((parent/'scope.json').read_bytes())
cases=[]
psi=unpack(stage['quantum']['psi']);phi=unpack(stage['quantum']['phi']);u=unpack(stage['quantum']['utilities'])
for b,label in enumerate(['retained-native-A','retained-native-B']):
    cases.append(dict(id=label,kind='retained constructed native row0 only',psi=packed(psi[b]),phi=packed(phi[0]),u=u[b,0].tolist(),mask=[True,True]))
e=np.eye(8,dtype=complex);p=(e[0]+e[1])/math.sqrt(2);m=(e[0]-e[1])/math.sqrt(2)
base=np.stack([(3*e[0]+4*e[1])/5,(4*e[0]+3j*e[2])/5,(e[1]+1j*e[3])/math.sqrt(2)])
fixtures=[
 ('zero-overlap',np.stack([e[1],e[2]]),[-1,1],[True,True]),
 ('unit-overlap',np.stack([e[0],e[0]]),[-1,1],[True,True]),
 ('mixed-extremes',np.stack([e[0],e[1]]),[-1,1],[True,True]),
 ('constant-utility',np.stack([p,m]),[-2,-2],[True,True]),
 ('complex-three',np.stack([(e[0]+1j*e[1])/math.sqrt(2),(e[0]-1j*e[1])/math.sqrt(2),(e[0]+e[2])/math.sqrt(2)]),[-2,1,3],[True]*3),
 ('nonuniform',base,[-2,.5,3],[True]*3),
 ('word-gauge-nonuniform',base*np.exp(1j*np.array([.2,.7,1.1]))[:,None],[-2,.5,3],[True]*3),
 ('permuted-masked-nonuniform',np.stack([base[2],base[0],e[7],base[1]]),[3,-2,1234,.5],[True,True,False,True])]
for label,states,values,mask in fixtures:cases.append(dict(id=label,kind='general pure-state algebra fixture, not native certified',psi=packed(states),phi=packed(e[0]),u=values,mask=mask))
assert len(cases)==10
scope=dict(status='FROZEN_BEFORE_TANGENT_ALGEBRA',recorded_at=datetime.now(timezone.utc).isoformat(),
 purpose='Resolve derivative-control decomposition for original quantum candidate; no method selection/new theory/task admission.',
 EPS=oldscope['fixed']['EPS'],dtype='complex128/float64',absolute_tolerance=1e-12,
 cases=cases,exact_fraction_cases=['retained-native-A','retained-native-B'],
 gates=['Cphi equals masked tangent sum','direct derivative equals Gram/self+cross decomposition',
        'Gamma equals probability formula','0<=g<=2Gamma*variance','zero bound has zero derivative',
        'retained A eta0/B eta1 and both signed cross terms','gauge/permutation/padding invariance',
        'exact Fraction A/B certificates','source hashes unchanged'],
 source_pins=[dict(path=str(parent/n),sha256=sha((parent/n).read_bytes())) for n in ['native-stage.json','trace.json','scope.json']],
 resources=dict(task_rows=0,feature_reads=0,checkpoint_reads=0,native_head_calls=0,circuit_calls=0,
                optimizer_updates=0,GPU_allocations=0,server_writes=0,COS=0,model_API=0,network_requests=0,package_installs=0),
 budget='one CPU algebra execution, max10 internal CPU seconds; no strength/EPS/norm/seed sweeps',
 stop='Any scientific gate fail stops this derivative-control claim; retain failures, no configuration rescue.',
 scope_limits=['Only row0 reused from two saved native synthetic contexts; eight other states not native certified',
               'Same-agent calculations and exact moments, not independent external audit',
               'No label alignment, trained-control advantage, novelty, newT/L3/L4T'],
 goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
(r/'scope.json').write_bytes((json.dumps(scope,indent=2)+'\n').encode())
(r/'receipt.before-execution.json').write_bytes((json.dumps(dict(scope_sha256=sha((r/'scope.json').read_bytes()),
    code_sha256=sha((r/'decompose.py').read_bytes()),derivation_sha256=sha((r/'derivation.md').read_bytes())),indent=2)+'\n').encode())
print(json.dumps(dict(status=scope['status'],numeric_rows=10,exact_moment_certificates=2,native_calls=0)))
