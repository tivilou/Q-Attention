"""Small deterministic algebra audit, not training or task-utility evidence."""
import cmath,hashlib,json,math,random,sys
from pathlib import Path
P=Path(__file__).resolve().parent
EPS=1e-6
def mat_add(a,b):return [[x+y for x,y in zip(ra,rb)] for ra,rb in zip(a,b)]
def scale(a,s):return [[s*x for x in row] for row in a]
def outer(v):return [[x*y.conjugate() for y in v] for x in v]
def dot(v,a):return sum(v[i].conjugate()*a[i][j]*v[j] for i in range(len(v)) for j in range(len(v))).real
def normalize(v):
    n=math.sqrt(sum(abs(x)**2 for x in v));return [x/n for x in v]
def eig2(a):
    mid=(a[0][0].real+a[1][1].real)/2
    rad=math.sqrt(((a[0][0].real-a[1][1].real)/2)**2+abs(a[0][1])**2)
    return mid-rad,mid+rad
def invsqrt2(a):
    lo,hi=eig2(a);assert lo>0
    if hi-lo<1e-13:return [[1/math.sqrt(lo),0],[0,1/math.sqrt(lo)]]
    b=(hi**-.5-lo**-.5)/(hi-lo);c=lo**-.5-b*lo
    return mat_add(scale(a,b),[[c,0],[0,c]])
def mul(a,b):return [[sum(a[i][k]*b[k][j] for k in range(len(a))) for j in range(len(a))] for i in range(len(a))]
def mv(a,v):return [sum(x*y for x,y in zip(row,v)) for row in a]
def sums(ms):
    a=[[0j]*len(ms[0]) for _ in ms[0]]
    for b in ms:a=mat_add(a,b)
    return a
def main():
    rng=random.Random(20261115); max_direct=0.;max_bound_violation=0.;max_povm=0.
    fixtures=200;queries=25
    for k in range(fixtures):
        sentences=[]
        for j in range(3):
            t=3+j
            states=[normalize([complex(rng.uniform(-1,1),rng.uniform(-1,1)) for _ in range(2)]) for _ in range(t)]
            utilities=[rng.uniform(-2,3) for _ in range(t)]
            aa=[mat_add(outer(v),[[EPS,0],[0,EPS]]) for v in states]
            s=sums(aa);h=sums([scale(a,u) for a,u in zip(aa,utilities)])
            si=invsqrt2(s);lam=eig2(mul(mul(si,h),si))[1]
            g=mat_add(scale(s,lam),scale(h,-1));L=eig2(s)[1]
            sentences.append((states,utilities,aa,s,h,si,lam,g,L))
            ff=sums([mul(mul(si,a),si) for a in aa])
            max_povm=max(max_povm,max(abs(ff[i][j]-(i==j)) for i in range(2) for j in range(2)))
        cert=eig2(sums([scale(g,1/(3*L)) for _,_,_,_,_,_,_,g,L in sentences]))[0]
        for q in range(queries):
            phi=normalize([complex(rng.uniform(-1,1),rng.uniform(-1,1)) for _ in range(2)])
            regret=0
            for states,u,aa,s,h,si,lam,g,L in sentences:
                raw=[abs(sum(x.conjugate()*y for x,y in zip(phi,v)))**2+EPS for v in states]
                score=sum(x*y for x,y in zip(raw,u))/sum(raw)
                quotient=dot(phi,h)/dot(phi,s)
                max_direct=max(max_direct,abs(score-quotient));regret+=(lam-score)/3
            max_bound_violation=max(max_bound_violation,cert-regret)
    lam=(1+EPS)/(1+2*EPS);exactgap=lam-.5
    opposed=[];aligned=[]
    for q in range(301):
        theta=math.pi*q/600;phi=[complex(math.cos(theta)),complex(math.sin(theta))]
        a=(abs(phi[0])**2+EPS)/(1+2*EPS);b=(abs(phi[1])**2+EPS)/(1+2*EPS)
        opposed.append((a+b)/2);aligned.append(a)
    assert max_direct<1e-12 and max_bound_violation<1e-12 and max_povm<1e-10
    assert max(abs(v-.5) for v in opposed)<1e-14 and abs(max(aligned)-lam)<1e-14
    result=dict(status='PASS',role='algebra audit only; not task evidence or novelty certification',random_seed=20261115,fixtures=fixtures,queries_per_fixture=queries,max_direct_quotient_residual=max_direct,max_povm_identity_residual=max_povm,max_certificate_violation=max_bound_violation,opposing_fixture=dict(individual_maximum=lam,shared_mean=.5,exact_gap=exactgap,certificate=1/(2*(1+2*EPS))),aligned_fixture=dict(shared_maximum=max(aligned),certificate=0),scope='d2 relaxed pure states, fixed utilities; actual Q-RMEA d8/ansatz not empirically tested',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),theory_sha256=hashlib.sha256((P/'theory-candidate-v1.md').read_bytes()).hexdigest())
    (P/'theory-algebra-audit.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(result))
if __name__=='__main__':main()
