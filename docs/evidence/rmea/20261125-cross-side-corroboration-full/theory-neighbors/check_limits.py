"""CPU algebra fixtures for the stated limits, not a new task experiment."""
from pathlib import Path
import json,math
import numpy as np
r=Path(__file__).resolve().parent
rng=np.random.default_rng(43);eps=1e-6;errors=[];bound_slack=[]
for _ in range(128):
 b=rng.uniform(.05,1,(3,5));g=rng.uniform(.05,1,5)
 w=b*g;alpha=w/w.sum(-1,keepdims=True)
 lhs=math.log(alpha[0,0])+math.log(alpha[1,1])-math.log(alpha[0,1])-math.log(alpha[1,0])
 rhs=math.log(b[0,0])+math.log(b[1,1])-math.log(b[0,1])-math.log(b[1,0])
 errors.append(abs(lhs-rhs));real=w+eps;a=real/real.sum(-1,keepdims=True)
 logratio=math.log(a[0,0])+math.log(a[1,1])-math.log(a[0,1])-math.log(a[1,0])
 tau=min(w[0,0],w[1,1],w[0,1],w[1,0]);bound=2*math.log1p(eps/tau)
 assert abs(logratio-rhs)<=bound+2e-14
 bound_slack.append(bound-abs(logratio-rhs))
dilution=[]
for m in (1,2,4,7):
 # One subject basis state and m object states; only the first matches.
 psi=np.eye(8,dtype=complex);rho=sum(np.outer(v,v.conj()) for v in psi[:m])/m
 gate=float(np.vdot(psi[0],rho@psi[0]).real);assert abs(gate-1/m)<1e-14
 h=.7;weights=np.r_[h*gate+eps,np.full(m,h/m+eps)]
 # Actual EPS is per token, including all m object tokens.
 actual=(weights[0]-weights[1:].sum())/weights.sum()
 expected=(h/m-h-(m-1)*eps)/(h/m+h+(m+1)*eps)
 assert abs(actual-expected)<1e-14
 dilution.append(dict(m=m,gate=gate,actual_per_token_eps_logit=float(actual),expected=float(expected)))
rho=np.eye(2)/2;swap=(1+np.trace(rho@rho))/2;assert swap==.75
plus=np.array([1,1])/np.sqrt(2);minus=np.array([1,-1])/np.sqrt(2)
assert np.allclose(plus**2,minus**2) and abs(np.vdot(plus,minus))<1e-14
rec=dict(status='PASS',cross_ratio_trials=128,max_identity_error=max(errors),epsilon_bound_min_slack=min(bound_slack),
 dilution=dilution,identical_maximally_mixed_swap=float(swap),same_populations_can_be_orthogonal=True,
 optimizer_updates=0,task_rows=0,cos_operations=0,new_theory_certified=False)
(r/'limits-verification.json').write_bytes((json.dumps(rec,indent=2)+'\n').encode());print(json.dumps(rec))
