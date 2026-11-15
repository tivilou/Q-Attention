"""Exact counterexample to a generic Fourier-degree inference for softmax."""
from fractions import Fraction
from itertools import product
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
rows=[];cubic=Fraction(0)
for x in product([-1,1],repeat=3):
    # q1 = log(2)*x1, kj=xj, vj=xj; softmax weights are proportional to 4 or 1.
    raw=[4 if x[0]*z==1 else 1 for z in x]
    y=Fraction(sum(a*z for a,z in zip(raw,x)),sum(raw))
    term=y*x[0]*x[1]*x[2];cubic+=term/Fraction(8)
    rows.append(dict(tokens=x,raw_proportional_weights=raw,output=str(y)))
assert cubic==Fraction(-1,18)
result=dict(status='PASS',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),claim_checked='A dot-product score is pairwise, but a normalized single softmax layer need not have Boolean Fourier degree at most two.',counterexample=dict(heads=1,query_row=1,embedding_dimension=1,q1='log(2)*x1',key='xj',value='xj',cubic_fourier_coefficient=str(cubic),inputs=rows),scope='Exact real arithmetic on three +/-1 tokens. Refutes the generic degree<=2 inference only, not a fixed-precision asymptotic MATCHk lower bound or the entire QHA paper.',original_locator='arxiv:2606.11673v2 p5, high-order attention paragraph; broader lower-bound scope p17-18 inspected separately',role='Literature premise audit, not Q-RMEA task-utility evidence or L5 contribution')
(P/'literature-premise-counterexample.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in result.items() if k!='counterexample'},indent=2));print('exact cubic coefficient:',cubic)
