"""Independent exact enumeration of the new decision rule; no task data."""
from pathlib import Path
from fractions import Fraction as Q
import hashlib,json
import numpy as np
r=Path(__file__).resolve().parent
p=np.array([[.7,.3,0,0,0,0,0,0],[.4,.6,0,0,0,0,0,0]],dtype=np.float64)
u=np.array([1.2,-.8]);maximum=0.;rows=[]
for x in np.arange(1,6)/6:
    raw=np.array([x,1-x])*(p[:,0]+1e-6);a=raw/raw.sum();mu=a@u
    c=np.sum(a[:,None]*p*(u-mu)[:,None],axis=0);d=(c>0).astype(float)
    m=p@d;alpha=a*(1+m-a@m)
    enumerated=sum(a[t]*p[t,k]*(u[t] if d[k] else mu) for t in range(2) for k in range(8))
    gains=[enumerated-mu,alpha@u-mu,np.maximum(c,0).sum(),.5*abs(c).sum(),a[0]*a[1]*abs(u[0]-u[1])*.5*abs(p[0]-p[1]).sum()]
    maximum=max(maximum,float(np.ptp(gains)))
    assert 0<gains[0]<sum(a*np.maximum(u-mu,0)) and alpha.min()>=0 and abs(alpha.sum()-1)<1e-14
    rows.append(dict(fraction=float(x),gain=float(gains[0])))
assert maximum<1e-10
w=Q(700001,1000000);v=Q(400001,1000000);pole=-v/(w-v)
double_pole_coefficient=w*v*pole*(1-pole)*Q(2)*Q(3,10)/(w-v)**2
assert double_pole_coefficient!=0
# A wrong row can gain while every other row, including the true row, stays fixed.
old=np.array([0.,0.]+[-10.]*38);g=np.array([0.,.25]+[0.]*38)
lse=lambda z:float(np.log(np.exp(z-z.max()).sum())+z.max())
ce_delta=lse(old+g)-lse(old);identity=float(np.log((np.exp(old-old.max())*np.exp(g)).sum()/np.exp(old-old.max()).sum())-g[0])
assert ce_delta>0 and abs(ce_delta-identity)<1e-14
record=dict(status='PASS',five_fraction_enumeration_max_error=maximum,fixtures=rows,rational_double_pole=str(pole),nonzero_double_pole_coefficient=str(double_pole_coefficient),ce_harm_counterexample=ce_delta,scope='Synthetic fixed-state decision algebra only; no novelty, task utility or natural-input witness',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(r/'math-check.json').write_bytes((json.dumps(record,indent=2)+'\n').encode());print(json.dumps(record))
