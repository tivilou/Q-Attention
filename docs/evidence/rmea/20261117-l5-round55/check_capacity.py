"""Exact finite product-state capacity fixture; no Torch, model or data forward."""
import ast,hashlib,itertools,json,math
from fractions import Fraction as F
from pathlib import Path
P=Path(__file__).resolve().parent
import argparse
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-dir',type=Path,required=True,help='Pinned rmea.py/frozen_feature_head.py directory; see report for hashes.')
S=parser.parse_args().source_dir.resolve()
def sh(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sh(S/'rmea.py')=='9f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0'
assert sh(S/'frozen_feature_head.py')=='f4e63a43994d2e241c9ae30497dcbcd4bc3a98f04ec6b16746b1e721371b6af9'
source=(S/'frozen_feature_head.py').read_text(encoding='utf8')
assert 'F.normalize(key,dim=-1)' in source and 'F.normalize(self.query,dim=-1)' in source
assert 'scores=(scores+1e-6)*mask[:,None,:]' in source
def ldl_positive(matrix):
    n=len(matrix);L=[[F(i==j) for j in range(n)] for i in range(n)];D=[]
    for i in range(n):
        pivot=matrix[i][i]-sum(L[i][k]**2*D[k] for k in range(i));assert pivot>0
        D.append(pivot)
        for j in range(i+1,n):L[j][i]=(matrix[j][i]-sum(L[j][k]*L[i][k]*D[k] for k in range(i)))/pivot
    assert all(matrix[i][j]==sum(L[i][k]*D[k]*L[j][k] for k in range(min(i,j)+1)) for i in range(n) for j in range(n))
    return D
def rank(matrix):
    a=[list(row) for row in matrix];r=0
    for c in range(len(a[0])):
        index=next((i for i in range(r,len(a)) if a[i][c]),None)
        if index is None:continue
        a[r],a[index]=a[index],a[r];p=a[r][c];a[r]=[x/p for x in a[r]]
        for i in range(r+1,len(a)):
            v=a[i][c]
            if v:a[i]=[x-v*y for x,y in zip(a[i],a[r])]
        r+=1
        if r==len(a):break
    return r
# |0>, |+>, |->, |+i>. Raw RY/RZ arguments are atanh(theta/pi).
raw=[(0,0),(F(1,2),0),(-F(1,2),0),(F(1,2),F(1,2))]
G=[[F(1),F(1,2),F(1,2),F(1,2)],
   [F(1,2),F(1),F(0),F(1,2)],
   [F(1,2),F(0),F(1),F(1,2)],
   [F(1,2),F(1,2),F(1,2),F(1)]]
pivots4=ldl_positive(G);assert pivots4==[F(1),F(3,4),F(2,3),F(1,2)]
codes=list(itertools.product(range(4),repeat=3))[:40]
Q=[[math.prod(G[a][b] for a,b in zip(x,y)) for y in codes] for x in codes]
eps=F(1,10**6);K=[[q+eps for q in row] for row in Q];D=[sum(row) for row in K]
alpha=[[q/d for q in row] for row,d in zip(K,D)]
pivots40=ldl_positive(K);assert len(pivots40)==40 and rank(alpha)==40
pop=[(F(1),F(0))]+[(F(1,2),F(1,2))]*3
Gdp=[[sum(x*y for x,y in zip(a,b)) for b in pop] for a in pop]
Kdp=[[math.prod(Gdp[a][b] for a,b in zip(x,y))+eps for y in codes] for x in codes]
assert rank(Kdp)==8
assert sum(Q[0])==F(45,4) and alpha[0][0]==F(1000001,11250040)
assert alpha[0][0]-F(1,40)==F(71875,1125004)>0
assert all(abs(v)<1 for pair in raw for v in pair)
# Exact rational LDL establishes rank; this scalar number is illustration only.
lam=(5-math.sqrt(17))/4;singular_lb=lam**3/(40*(1+float(eps)))
out=dict(schema='qattn.source-native-capacity-audit.v1',status='PASS_EXACT_HEAD_API_SCOPE',
 source_hashes={n:sh(S/n) for n in ['rmea.py','frozen_feature_head.py']},
 witness=dict(tokens=40,relations=40,valid_mask='All40,including subject/object rows38/39',
              single_qubit_states=['0','+','-','+i'],code_order='First40 lexicographic triples from0..3',
              source_map='H[0:6]=atanh of six finite physical RY/RZ angle fractions; prepare selects these into first layer, second layer zero. Same common two CNOT rings cancel in overlaps. Effects use same40codes. H[6] is a spare first-token utility indicator; all role coefficients zero.',
              raw_angle_fractions=[[str(x) for x in pair] for pair in raw],
              single_gram_ldl_pivots=[str(x) for x in pivots4],whole40_positive_ldl_pivots=40,
              ordinary_attention_rank=40,dephased_same_fixture_rank=8,source_floating_forward_executed=False),
 real_control=dict(global_rank_upper_bound=37,unit_query_or_unit_key_branch_upper_bound=36,
                   reason='Symmetric8x8 rank-one lift has36coordinates; epsilon adds at most one constant feature. If either full query-side or key-side vector set has unit norm, trace identity absorbs the constant without a37th coordinate. F.normalize clamp allows nonunit vectors, so37 is the unconditional source-family bound.',
                   logical_scope='Any real8 head fails to match this40x40 attention exactly on the chosen admitted H/R input; no strict inclusion or whole-predictor risk theorem.'),
 quantitative_bound=dict(exact_expression='sigma_min(alpha)>=((5-sqrt(17))/4)^3/[40(1+epsilon)]',
                          best_rank37_frobenius_error_at_least='sqrt(3)*((5-sqrt(17))/4)^3/[40(1+epsilon)]',
                          scalar_illustration_singular_bound=singular_lb,scalar_illustration_error_bound=math.sqrt(3)*singular_lb,
                          numeric_illustrations_are_not_interval_certificates=True),
 full40_readout=dict(zero_utility_case='All40logits zero under every attention, CE=log40; so structural separation need not change predictions.',
                    signed_fixture='Only value.W[0,6]=1 and decoder[0,0]=s=+1or-1; all other readout parameters/biases0. Targety=0, other39logits0. Same rank40 alpha either improves or worsens complete40CE versus uniform.',
                    alpha00=str(alpha[0][0]),uniform_alpha00='1/40',signed_target_logit_delta='s*71875/1125004',
                    loss_reason='CE_y0(z)=log(expz+39)-z strictly decreases with target logit.'),
 limitations=['Known density-feature/PSD dimension tools; no substantive novelty claim.',
 'Engineered admitted H/R inputs, not actual encoder-image/text/learned-checkpoint evidence.',
 'Real64 lifted classical features reproduce the kernel; no quantum-exclusivity/hardware/resource superiority claim.',
 'Dot-softmax may have full ordinary rank; this witness does not separate from that control.',
 'All exact statements are mathematical, not bitwise native floating execution.',
 'Native value/readout is coadapted; capacity alone is not utility, risk or total-gain mediation.'],
 new_native_model_forwards=0,new_data_reads=0,new_training=0,l5_accepted=False)
(P/'capacity-audit.json').write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(dict(status=out['status'],ordinary_rank=40,real_global_bound=37,dephased_rank=8,full40_CE_both_signs=True,l5_accepted=False)))
