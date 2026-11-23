"""Fixed synthetic full-span experiment for a local exact-fiber construction.

This is mathematical feasibility, not an admitted model/plugin or a data test.
"""
from pathlib import Path
import cmath,json,math,time
r=Path(__file__).resolve().parent;d=8;eps=1e-6
dot=lambda a,b:sum(x*y for x,y in zip(a,b))
inner=lambda a,b:sum(x.conjugate()*y for x,y in zip(a,b))
norm=lambda a:math.sqrt(sum(abs(x)**2 for x in a))
unit=lambda a:[x/norm(a) for x in a]
real=lambda a:[z.real for z in a]+[z.imag for z in a]
complexify=lambda a:[complex(a[k],a[d+k]) for k in range(d)]
def solve(matrix,rhs):
 n=len(rhs);a=[list(row)+[v] for row,v in zip(matrix,rhs)]
 for k in range(n):
  p=max(range(k,n),key=lambda i:abs(a[i][k]));a[k],a[p]=a[p],a[k]
  assert abs(a[k][k])>1e-13,'Singular fixture: no regular-fiber claim'
  pivot=a[k][k];a[k]=[x/pivot for x in a[k]]
  for i in range(n):
   if i!=k:
    scale=a[i][k];a[i]=[x-scale*y for x,y in zip(a[i],a[k])]
 return [row[-1] for row in a]
mention=[unit([complex(int(j==k))+.05*cmath.exp(1j*(j+1)*(k+1)) for k in range(d)]) for j in range(d)]
# Before column normalization the matrix is strictly diagonally dominant:
# diagonal modulus>=.95, off-diagonal absolute row sum=.35, so full complex rank8.
phi=[cmath.exp(1j*k*k)/math.sqrt(d) for k in range(d)]
context=[unit([cmath.exp(1j*(j+2)*(k+1)/3)+.3*complex(k==j) for k in range(d)]) for j in range(4)]
tokens=mention+context;utilities=[math.sin(k+1) for k in range(len(tokens))];lam=1/8
def grad_prob(psi,w):return [2*x*inner(psi,w) for x in psi]
jac=[[2*x for x in real(phi)]]+[real(grad_prob(p,phi)) for p in mention]
gram=[[dot(a,b) for b in jac] for a in jac]
inverse_cols=[solve(gram,[int(i==k) for i in range(len(gram))]) for k in range(len(gram))]
inverse=[list(x) for x in zip(*inverse_cols)]
b=[[sum(jac[k][i]*inverse[k][j] for k in range(len(jac))) for j in range(len(jac))] for i in range(2*d)]
mv=lambda matrix,v:[dot(row,v) for row in matrix]
right_inverse_error=max(abs(dot(jac[i],[row[k] for row in b])-int(i==k)) for i in range(len(jac)) for k in range(len(jac)))
assert right_inverse_error<1e-10
project=lambda v:[x-y for x,y in zip(v,mv(b,mv(jac,v)))]
raw=[abs(inner(p,phi))**2+eps for p in tokens];z=sum(raw);logit=dot(raw,utilities)/z
ge=real([sum(p[k]*inner(p,phi) for p in context)*2/len(context) for k in range(d)])
gl=[sum((u-logit)*grad_prob(p,phi)[k] for p,u in zip(tokens,utilities))/z for k in range(d)]
gl=real(gl);a=project(ge);c=project(gl)
assert norm(a)>1e-8*norm(ge) and norm(c)>1e-8*norm(gl)
aa=unit(a);cc=unit(c);correlation=dot(aa,cc);sign=1 if correlation>=0 else -1
t=unit([-x-sign*y for x,y in zip(aa,cc)])
assert norm(mv(jac,t))<1e-10 and dot(ge,t)<0 and abs(dot(gl,t))>1e-8
bf=math.sqrt(sum(x*x for row in b for x in row));D=math.sqrt(1+sum(norm(p)**4 for p in mention))
s=min(.01,1/(8*bf*D));tangent=[s*x for x in t]
def q(v):
 w=complexify(v)
 return [norm(w)**2]+[abs(inner(p,w))**2 for p in mention]
normal=[0.]*(2*d)
for iteration in range(48):normal=[-x for x in mv(b,q([x+y for x,y in zip(tangent,normal)]))]
candidate=complexify([x+y+z for x,y,z in zip(real(phi),tangent,normal)])
plain=complexify([x+y for x,y in zip(real(phi),tangent)])
def constraints(w):return [norm(w)**2]+[abs(inner(p,w))**2 for p in mention]
base=constraints(phi)
error=lambda w:max(abs(x-y) for x,y in zip(constraints(w),base))
def energy(w):return sum(abs(inner(p,w))**2 for p in context)/len(context)+lam*sum(abs(x-y)**2 for x,y in zip(w,phi))
def readout(w):
 weights=[abs(inner(p,w))**2+eps for p in tokens]
 return dot(weights,utilities)/sum(weights)
delta_e=energy(candidate)-energy(phi);delta_l=readout(candidate)-readout(phi)
assert error(candidate)<1e-12 and error(plain)>1e-8 and delta_e<-1e-10 and abs(delta_l)>1e-6
assert abs(norm(plain)**2-1-s*s)<1e-14
out=dict(status='PASS',observed_at=time.time(),scope='One fixed synthetic full-span state/decoder fixture; no trained circuit/data/utility/novelty claim',dimension=d,mentions=len(mention),jacobian_rows=len(jac),quotient_dimension_on_this_regular_branch=2*d-len(jac)-1,right_inverse_error=right_inverse_error,step=s,contraction_upper_bound=4*s*bf*D,iterations=48,tangent_only_residual=error(plain),retracted_residual=error(candidate),energy_delta=delta_e,relative_logit_delta=delta_l,energy_first_derivative=dot(ge,t),logit_first_derivative=dot(gl,t),optimizer_updates=0,model_parameters_changed=0,exit_status=0)
(r/'retraction-verification.json').write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps(out))
