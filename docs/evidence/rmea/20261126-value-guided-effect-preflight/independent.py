"""Independent exact rational moment and counterexample checks, no NumPy."""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json

r = Path(__file__).resolve().parent
cfg = json.loads((r/'scope.json').read_bytes())
eps = F(str(cfg['EPS']))


def add(a,b): return [[x+y for x,y in zip(ra,rb)] for ra,rb in zip(a,b)]
def scale(c,a): return [[c*x for x in row] for row in a]
def dot(q,a): return sum(q[i]*a[i][j]*q[j] for i in range(2) for j in range(2))
def outer(q): return [[x*y for y in q] for x in q]


eye = [[F(1),F(0)],[F(0),F(1)]]
states = [[F(3,5),F(4,5)],[F(5,13),F(-12,13)],[F(8,17),F(15,17)]]
checks = 0
for q in states:
    for utility in ([F(-2),F(1),F(3)],[F(2),F(-4),F(0)],[F(1)]*3,[F(-1),F(2),F(-3)]):
        bt = [add(outer(v),scale(eps,eye)) for v in states]
        b = [[sum(x[i][j] for x in bt) for j in range(2)] for i in range(2)]
        a = [[sum(u*x[i][j] for u,x in zip(utility,bt)) for j in range(2)] for i in range(2)]
        native = sum((sum(x*y for x,y in zip(q,v))**2+eps)*u for v,u in zip(states,utility))
        denominator = sum(sum(x*y for x,y in zip(q,v))**2+eps for v in states)
        z = dot(q,a)/dot(q,b)
        assert z == native/denominator
        c = add(a,scale(-z,b))
        assert dot(q,c) == 0
        cq = [sum(c[i][j]*q[j] for j in range(2)) for i in range(2)]
        derivative = 2*sum(x*x for x in cq)/dot(q,b)
        assert derivative >= 0
        checks += 1

# The six densities have rational entries despite plus/minus state amplitudes.
minus = [[F(1,2),F(-1,2)],[F(-1,2),F(1,2)]]
plus = [[F(1,2),F(1,2)],[F(1,2),F(1,2)]]
rho = [minus,minus,minus,plus,[[F(1),F(0)],[F(0),F(0)]],[[F(0),F(0)],[F(0),F(1)]]]
bt = [add(x,scale(eps,eye)) for x in rho]
b = [[sum(x[i][j] for x in bt) for j in range(2)] for i in range(2)]
a = add(scale(F(-1),bt[-2]),bt[-1])
assert a == [[F(-1),F(0)],[F(0),F(1)]]
assert b == [[3+6*eps,F(-1)],[F(-1),3+6*eps]]
values = [dot([F(1),F(x)],a)/dot([F(1),F(x)],b) for x in (6,100)]
assert values[0] > values[1] > 0
# exp(8)>1+8+8^2/2=41; rival score increases with exp(8).
lower = 2*F(40,42)/(1+2*eps)
assert lower > 1
primary = json.loads((r/'results.json').read_bytes())
assert all(abs(float(v)-n)<1e-14 for v,n in zip(values,primary['nonmonotone_strength']['scores']))
assert primary['forty_class_counterexample']['updated_first2'][1] > float(lower)
out = dict(status='PASS_INDEPENDENT_EXACT_RATIONAL_EFFECT_CHECKS', exact_native_moment_cases=checks,
           exact_nonmonotone_scores=[str(v) for v in values], exponential_lower_bound='exp(8)>41',
           rival_score_certified_lower_bound=str(lower), classification_margin_counterexample_certified=True,
           primary_results_sha256=hashlib.sha256((r/'results.json').read_bytes()).hexdigest(),
           scope='Independent rational moment construction and strict inequalities; not task data, circuit implementation or external review.',
           goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
(r/'independent-results.json').write_bytes((json.dumps(out,indent=2)+'\n').encode())
print(json.dumps(out))
