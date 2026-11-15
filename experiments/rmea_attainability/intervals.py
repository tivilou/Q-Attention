"""Small directed Decimal intervals; exact float input, no sqrt assumption."""
from decimal import Decimal,localcontext,ROUND_FLOOR,ROUND_CEILING
PRECISION=50
def calc(fn,rounding):
    with localcontext() as c:c.prec=PRECISION;c.rounding=rounding;return fn()
class I:
    def __init__(self,lo,hi=None):
        self.lo=lo if isinstance(lo,Decimal) else Decimal(str(lo))
        self.hi=self.lo if hi is None else (hi if isinstance(hi,Decimal) else Decimal(str(hi)))
        assert self.lo.is_finite() and self.hi.is_finite() and self.lo<=self.hi
    @classmethod
    def f(cls,x):return cls(Decimal.from_float(float(x)))
    def __add__(self,b):
        b=b if isinstance(b,I) else I(b)
        return I(calc(lambda:self.lo+b.lo,ROUND_FLOOR),calc(lambda:self.hi+b.hi,ROUND_CEILING))
    __radd__=__add__
    def __neg__(self):return I(self.hi.copy_negate(),self.lo.copy_negate())
    def __sub__(self,b):return self+(-b if isinstance(b,I) else -I(b))
    def __mul__(self,b):
        b=b if isinstance(b,I) else I(b)
        return I(min(calc(lambda:a*z,ROUND_FLOOR) for a in (self.lo,self.hi) for z in (b.lo,b.hi)),max(calc(lambda:a*z,ROUND_CEILING) for a in (self.lo,self.hi) for z in (b.lo,b.hi)))
    __rmul__=__mul__
    def __truediv__(self,b):
        b=b if isinstance(b,I) else I(b);assert b.lo>0 or b.hi<0
        return self*I(calc(lambda:Decimal(1)/b.hi,ROUND_FLOOR),calc(lambda:Decimal(1)/b.lo,ROUND_CEILING))
    def sq(self):
        lower=Decimal(0) if self.lo<=0<=self.hi else min(calc(lambda:a*a,ROUND_FLOOR) for a in (self.lo,self.hi))
        return I(lower,max(calc(lambda:a*a,ROUND_CEILING) for a in (self.lo,self.hi)))
    def absolute(self):
        return I(Decimal(0) if self.lo<=0<=self.hi else min(self.lo.copy_abs(),self.hi.copy_abs()),max(self.lo.copy_abs(),self.hi.copy_abs()))
    def json(self):return dict(lower=str(self.lo),upper=str(self.hi))
def cx(z):return I.f(z.real),I.f(z.imag)
def product(a,b):return a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0]
def norm(v):return sum((x[0].sq()+x[1].sq() for x in v),I(0))
def density(v):
    z=[cx(x) for x in v];n=norm(z);assert n.lo>0
    return [[tuple(x/n for x in product(a,(b[0],-b[1]))) for b in z] for a in z]
def probability(psi,phi):
    a=[cx(z) for z in psi];b=[cx(z) for z in phi];real=I(0);imag=I(0)
    for u,v in zip(a,b):
        z=product(u,(v[0],-v[1]));real+=z[0];imag+=z[1]
    return (real.sq()+imag.sq())/(norm(a)*norm(b))
def physical_allocation(psi,phi,eps):
    p=[probability(v,phi)+eps for v in psi];s=sum(p,I(0));return [v/s for v in p]
def certify(cases,targets,w,tau,eps):
    d=len(cases[0][0]);W=[[(I(0),I(0)) for _ in range(d)] for _ in range(d)];c=I(0);bound=I(0);pos=0
    for states,a in zip(cases,targets):
        n=len(states);ww=[I.f(v) for v in w[pos:pos+n]];aa=[I.f(v) for v in a];pos+=n
        s=sum((x*y for x,y in zip(ww,aa)),I(0));c+=eps*(s*n-sum(ww,I(0)));bound+=tau*(I(1)+eps)*n*sum((x.absolute() for x in ww),I(0))
        for rho,wi in zip(states,ww):
            coefficient=wi-s
            for j in range(d):
                for k in range(d):W[j][k]=(W[j][k][0]+coefficient*rho[j][k][0],W[j][k][1]+coefficient*rho[j][k][1])
    assert pos==len(w)
    center=sum((W[j][j][0] for j in range(d)),I(0))/d;frob=I(0)
    for j in range(d):
        for k in range(d):frob+=(W[j][k][0]-(center if j==k else I(0))).sq()+W[j][k][1].sq()
    delta=(c-center).absolute()-bound;positive=delta.lo>0
    gap=delta.sq()-frob*(I(1)-I(1)/d) if positive else I(-1)
    return dict(certified_infeasible=positive and gap.lo>0,delta_after_tolerance=delta.json(),squared_gap=gap.json(),frobenius_squared=frob.json(),target_tolerance_bound=bound.json(),precision=PRECISION,dimension=d)
