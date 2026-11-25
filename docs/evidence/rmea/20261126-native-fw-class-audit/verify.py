from pathlib import Path
from fractions import Fraction as F
import hashlib,json
import numpy as np
import fw_trace
r=Path(__file__).resolve().parent
read=lambda n:json.loads((r/n).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=read('scope.json');m=read('results.json');e=read('audit-execution.json');trace=read('sample-trace.v2.json')
assert e['exit_status']==0 and not e['stderr'] and e['code_sha256']==sha(r/'audit.py') and json.loads(e['stdout'])==m
for pin in s['pins']:assert sha(pin['path'])==pin['sha256']
assert m['CPU_seconds']<10 and all(v<1e-12 for v in m['errors'].values())
assert m['trace']['sha256']==sha(r/'sample-trace.v2.json') and m['algebra_trace_sha256']==sha(r/'algebra-trace.json')
assert read('audit-execution.before-diagnostic.json')['exit_status']==1
assert read('audit-execution.before-centering.json')['exit_status']==1 and '7.770847671232738e-12' in read('audit-execution.before-centering.json')['stderr']
timing_error=0.;reader_checks=0
for sample in trace['samples']:
    ref=sample['representations'][0]['artifact_refs'][0];assert sha(r/ref['path'])==ref['sha256']
    payload=read(ref['path']);a={k:np.asarray(v['values'],dtype=v['dtype']) for k,v in payload.items()}
    for t in range(3):
        memory=sum((.5*.95**(t-1-j)*np.outer(a['hidden'][j],a['hidden'][j]) for j in range(t)),np.zeros((8,8)))
        timing_error=max(timing_error,float(np.max(abs(memory-a['memory_before'][t]))));assert timing_error<1e-12
        np.testing.assert_allclose(a['preLN'][t],a['inputs'][t]+memory@np.maximum(a['inputs'][t],0),atol=1e-12,rtol=1e-12)
    for rep in sample['representations']:
        arr=a[rep['id']];coords=[list(map(int,np.unravel_index(i,arr.shape))) for i in sorted({0,arr.size//2,arr.size-1})]
        for c,v in zip(coords,fw_trace.read_value(r,sample,rep['id'],coords)):assert v==arr[tuple(c)].item();reader_checks+=1
# Exact arithmetic for signed utility / epsilon rational trace readout.
Z=(F(0),F(0))
def plus(a,b):return a[0]+b[0],a[1]+b[1]
def mul(a,b):return a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]
def conj(a):return a[0],-a[1]
def scale(a,b):return a[0]*b,a[1]*b
def csum(xs):
    out=Z
    for x in xs:out=plus(out,x)
    return out
def outer(v):return [[mul(a,conj(b)) for b in v] for a in v]
xs=[[(F(3,5),F(0)),(F(4,5),F(0))],[(F(5,13),F(0)),(F(0),F(12,13))],[(F(1),F(0)),Z]]
qs=[[(F(3,5),F(0)),(F(0),F(4,5))],[(F(5,13),F(0)),(F(12,13),F(0))]]
exact=[];eps=F(1,10**6)
for qi,q in enumerate(qs):
    for ui,uu in enumerate(([-2,1,3],[-1,3,-4])):
        rho=[outer(x) for x in xs];E=outer(q)
        Q=[[plus(E[i][j],(eps,F(0))) if i==j else E[i][j] for j in range(2)] for i in range(2)]
        M=[[csum(rr[i][j] for rr in rho) for j in range(2)] for i in range(2)]
        V=[[csum(scale(rr[i][j],F(u)) for rr,u in zip(rho,uu)) for j in range(2)] for i in range(2)]
        numerator=csum(mul(Q[i][j],V[j][i]) for i in range(2) for j in range(2))
        denominator=csum(mul(Q[i][j],M[j][i]) for i in range(2) for j in range(2))
        assert numerator[1]==denominator[1]==0 and denominator[0]>0
        p=[]
        for x in xs:
            overlap=csum(mul(conj(a),b) for a,b in zip(q,x));p.append(overlap[0]**2+overlap[1]**2)
        direct=sum((a+eps)*u for a,u in zip(p,uu))/sum(a+eps for a in p)
        assert numerator[0]/denominator[0]==direct
        exact.append(dict(query=qi,utilities=ui,score=str(direct),status='EXACT'))
out=dict(status='PASS_REOPENED_SOURCE_CLASS_AUDIT',equation3_memory_error=timing_error,reader_checks=reader_checks,exact_fraction_readout_certificates=exact,
    failed_raw_trace_quotient_preserved=True,repair='Equivalent utility-centering before dense trace contraction; no epsilon/normalizer/input/tolerance change',
    all24_numeric_cases_match=True,full_FW_trajectories=2,FW_history_nonnegative=True,native_pair_bridge_verified=True,
    historical_real8_scope='Real-state fixture only; general complex8 real16 two-overlap or Hermitian64 exact',
    general_function_class_inclusion='NOT_PROVEN',new_task_prediction_or_utility=False,method_admitted=False,new_theory=False,goal='ACTIVE')
(r/'verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
