"""Exact rational collision and dephased-chain check; reopen the saved traces."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json
r=Path(__file__).resolve().parent
read=lambda n:json.loads((r/n).read_bytes());result=read('math-results.json');trace=read('math-trace.json')
assert result['status']=='PASS' and result['max_error']<=1e-11
assert len(trace['tuples'])==32
mul=lambda a,b:(a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0])
conj=lambda a:(a[0],-a[1])
zero=(F(0),F(0));A={0:(F(1,2),F(-1,2)),3:(F(1,2),F(1,2))};B={0:(F(1,2),F(1,2)),3:(F(-1,2),F(1,2))}
rho=lambda psi:[[mul(psi.get(i,zero),conj(psi.get(j,zero))) for j in range(8)] for i in range(8)]
assert rho(A)==rho(B) and rho(A)[0][3]==(F(0),F(-1,2))
def mm(a,b):return [[sum(a[i][k]*b[k][j] for k in range(8)) for j in range(8)] for i in range(8)]
def tr(a):return [list(x) for x in zip(*a)]
P=[[F(0) for _ in range(8)] for _ in range(8)]
for initial in range(8):
 final=initial
 for control,target in ((0,1),(1,2),(2,0)):
  if (final>>(2-control))&1:final^=1<<(2-target)
 P[final][initial]=F(1)
M=[[F(1,2) if i in (j,j^4) else F(0) for j in range(8)] for i in range(8)]
W2=mm(P,M);Q=mm(mm(P,P),M);Qd=tr(Q)
dephased=mm(mm(W2,Qd),P)[0][0]
assert dephased==F(1,2)
eps=F(1,1000000);native_probability=rho(A)[0][0][0]/2
assert native_probability==F(1,4)
native_z=((native_probability+eps)-(native_probability+eps))/(2*native_probability+2*eps)
middle_z=(eps-(1+eps))/(1+2*eps)
assert native_z==0 and middle_z==F(-500000,500001)
assert abs(float(middle_z)-result['fixture']['mid_row0'])<1e-11
for name in ('out','mid','dephased'):
 logits=trace['fixture']['logits'][name];alpha=trace['fixture']['alphas'][name]
 probs=trace['fixture']['probabilities'][name];utility=trace['fixture']['utilities']
 assert len(logits)==len(alpha)==len(probs)==len(utility)==40
 for row in range(40):
  denominator=sum(x+float(eps) for x in probs[row])
  expected=[(x+float(eps))/denominator for x in probs[row]]
  assert max(abs(a-b) for a,b in zip(alpha[row],expected))<1e-11
  assert abs(sum(a*u for a,u in zip(alpha[row],utility[row]))-logits[row])<1e-11
 assert abs(sum(trace['fixture']['softmax'][name])-1)<1e-11
for x in trace['tuples']:
 assert all(-1e-11<=p<=1+1e-11 for p in x['probabilities'].values())
 assert max(abs(x['representations'][k]-x['probabilities']['mid']) for k in ('direct','vec','real16'))<1e-11
assert result['resources']==dict(GPU=0,task_forward=0,fit=0,new_capacity=0)
execution=read('math_preflight-execution.json')
assert execution['exit_status']==0 and execution['code_sha256']==hashlib.sha256((r/'math_preflight.py').read_bytes()).hexdigest()
packet=read('review-packet.private.json')
assert packet['code']==(r/'interleaved_math.py').read_text('utf8')
assert read('review.json')['packet_sha256']==hashlib.sha256((r/'review-packet.private.json').read_bytes()).hexdigest()
out=dict(status='PASS_EXACT_RATIONAL_COLLISION_AND_DEPHASED_CHAIN',
 density_entries_checked=64,native_probability=str(native_probability),dephased_probability=str(dephased),
 exact_native_logit=str(native_z),exact_mid_logit=str(middle_z),all40_trace_reopened=True,
 independence='Primary-owned exact Fraction/Markov reconstruction; external reviewer supplied the construction. This is not independent task confirmation.',
 scope='Fixed preparer/utility output distinction, not freely reparameterized family or task-risk separation',
 reviewed_math_unchanged=True,training_admitted=False,goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
(r/'independent-verification.json').write_bytes((json.dumps(out,indent=2)+'\n').encode())
print(json.dumps(out))
