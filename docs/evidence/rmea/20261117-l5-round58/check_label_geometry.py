"""Exact finite-field/projector audit; no dataset, model, optimizer or training."""
from fractions import Fraction as F
from collections import Counter
from pathlib import Path
import hashlib, json, math

P = Path(__file__).resolve().parent
Z = (0, 0)
ONE = (1, 0)
PHASE = [ONE, (0,1), (-1,0), (0,-1)]
def add(a,b): return (a[0]+b[0],a[1]+b[1])
def mul(a,b): return (a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0])
def scale(a,s): return tuple(tuple((s*z[0],s*z[1]) for z in row) for row in a)
def plus(a,b): return tuple(tuple(add(x,y) for x,y in zip(u,v)) for u,v in zip(a,b))
def mm(a,b):
    return tuple(tuple(tuple(sum(mul(a[i][k],b[k][j])[c] for k in range(8)) for c in range(2))
                       for j in range(8)) for i in range(8))
def tr(a): return tuple(sum(a[i][i][c] for i in range(8)) for c in range(2))
def dag(a): return tuple(tuple((a[j][i][0],-a[j][i][1]) for j in range(8)) for i in range(8))
I = tuple(tuple(ONE if i==j else Z for j in range(8)) for i in range(8))
ZERO = scale(I,0)
def gf(a,b):
    result = 0
    while b:
        if b&1: result ^= a
        b >>= 1; a <<= 1
        if a&8: a ^= 0b1011
    return result
def field_trace(a):
    a2=gf(a,a); ans=a^a2^gf(a2,a2)
    assert ans in [0,1]
    return ans
def rank2(a):
    a=[list(row) for row in a]; rank=0
    for col in range(3):
        pivot=next((i for i in range(rank,3) if a[i][col]),None)
        if pivot is None: continue
        a[pivot],a[rank]=a[rank],a[pivot]
        for i in range(3):
            if i!=rank and a[i][col]: a[i]=[x^y for x,y in zip(a[i],a[rank])]
        rank+=1
    return rank
def pauli(x,z):
    matrix=[[Z for _ in range(8)] for _ in range(8)]
    phase=PHASE[(x&z).bit_count()%4]
    for v in range(8):
        sign=1 if (z&v).bit_count()%2==0 else -1
        matrix[v^x][v]=(sign*phase[0],sign*phase[1])
    return tuple(tuple(row) for row in matrix)
def main():
    e=[1,2,4]
    matrices=[[[field_trace(gf(alpha,gf(x,y))) for y in e] for x in e] for alpha in range(8)]
    for a in matrices:
        assert all(a[i][j]==a[j][i] for i in range(3) for j in range(3))
    assert all(rank2([[a[i][j]^b[i][j] for j in range(3)] for i in range(3)])==3
               for k,a in enumerate(matrices) for b in matrices[k+1:])
    classes=[[pauli(0,1<<(2-i)) for i in range(3)]]
    for a in matrices[:4]:
        classes.append([pauli(1<<(2-i),sum(a[i][j]<<(2-j) for j in range(3))) for i in range(3)])
    projectors=[]
    for generators in classes:
        assert all(dag(g)==g and mm(g,g)==I for g in generators)
        assert all(mm(a,b)==mm(b,a) for i,a in enumerate(generators) for b in generators[i+1:])
        group=[]
        for signs in range(8):
            n=I
            for i,g in enumerate(generators): n=mm(n,plus(I,scale(g,1 if signs&(1<<i) else -1)))
            assert dag(n)==n and mm(n,n)==scale(n,8) and tr(n)==(8,0)
            group.append(n)
        assert sum_matrices(group)==scale(I,8)
        projectors.extend(group)
    gram=[]
    for i,a in enumerate(projectors):
        row=[]
        for j,b in enumerate(projectors):
            t=tr(mm(a,b)); assert t[1]==0
            q=F(t[0],64)
            assert q==(1 if i==j else (0 if i//8==j//8 else F(1,8)))
            row.append(q)
        assert Counter(row)=={F(1):1,F(0):7,F(1,8):32}
        gram.append(row)
    eps=F(1,10**6)
    weight=lambda q:(q+eps)/(q+1+9*eps)
    a1,a0,ac=map(weight,[F(1),F(0),F(1,8)])
    assert a1>ac>a0 and ac==F(1,9)
    permutations=[list(range(40)),[(13*i+7)%40 for i in range(40)]]
    hist=Counter(map(weight,gram[0]))
    # Class y is assumed aligned to P_{pi(y)}. This is NOT invariance of
    # relabelled queries applied to an unchanged physical input encoder.
    for pi in permutations:
        assert sorted(pi)==list(range(40))
        for y in range(40):
            assert Counter(weight(gram[pi[r]][pi[y]]) for r in range(40))==hist
    ce=lambda sign:math.log1p(7*math.exp(float(sign*(a0-a1)))+32*math.exp(float(sign*(ac-a1))))
    numeric={'positive_common_signal_utility_CE':ce(1),'zero_utilities_CE':math.log(40),
             'negative_common_signal_utility_CE':ce(-1)}
    assert ce(1)<math.log(40)<ce(-1)
    encoded=[[[list(z) for z in row] for row in n] for n in projectors]
    payload={'schema':'qattn.exact-label-projectors.v1','denominator':8,
             'construction':'GF(8), polynomial x^3+x+1; symmetric trace forms; Z plus alpha=0,1,2,3.',
             'numerators':encoded}
    raw=(json.dumps(payload,separators=(',',':'))+'\n').encode()
    (P/'label-projectors.json').write_bytes(raw)
    result={'schema':'qattn.label-geometry-audit.v1','status':'PASS','exact_arithmetic':'Gaussian integers and Fraction',
      'source_construction':'quant-ph/0103162v3 Theorem4.4, Lemma4.3; field-trace instantiation independently checked.',
      'dimension':8,'bases':5,'rank_one_projectors':40,'all_1600_overlaps_checked':True,
      'per_row_overlap_counts':{'1':1,'0':7,'1/8':32},
      'signal_alpha':{'correct':str(a1),'same_basis_wrong':str(a0),'cross_basis_wrong':str(ac)},
      'background':'Eight tokens forming the computational basis, sum of projectors I; one signal token aligned to the assigned label projector.',
      'utilities':'Common signed signal utility, zero on background; full40 logits, zero class biases. Abstract fixture only.',
      'assignment_conclusion':'Under the aligned-signal/isotropic-background/common-utility assumptions, every label assignment has the same CE. The proof uses the exact row multiset, not only the two checked permutations.',
      'important_limit':'Does not prove invariance when queries are relabelled but physical inputs/encoder remain fixed; does not rule out training or label-dependent noise effects.',
      'geometry_alone_not_utility':'Positive, zero and negative common signed readouts give benefit, null and harm versus uniform attention in this abstract fixture.',
      'illustrative_float_CE_only':numeric,'projectors_sha256':hashlib.sha256(raw).hexdigest(),
      'native_ansatz_reachability_verified':False,'new_model_forwards':0,'new_data_reads':0,'training':0,'l5_accepted':False}
    (P/'geometry-audit.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps({'status':'PASS','exact_projectors':40,'overlaps':1600,'CE':numeric,'assignment_ranked':False,'training':0}))
def sum_matrices(seq):
    total=ZERO
    for a in seq: total=plus(total,a)
    return total
if __name__=='__main__': main()
