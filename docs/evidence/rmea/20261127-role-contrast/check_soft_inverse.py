"""Exact-rational audit of retained aggregate matrices; no data loaders or models."""
from pathlib import Path
from collections import deque
from fractions import Fraction as F
import hashlib,json,math,sys

def rank(rows):
    a=[r[:] for r in rows];r=0
    for k in range(len(a[0]) if a else 0):
        pivot=next((i for i in range(r,len(a)) if a[i][k]),None)
        if pivot is None:continue
        a[r],a[pivot]=a[pivot],a[r]
        z=a[r][k];a[r]=[x/z for x in a[r]]
        for i in range(r+1,len(a)):
            z=a[i][k]
            if z:a[i]=[x-z*y for x,y in zip(a[i],a[r])]
        r+=1
        if r==len(a):break
    return r

def digest(b):return hashlib.sha256(b).hexdigest()

def exact(m):
    size=len(m);assert all(len(x)==size for x in m)
    assert all(type(m[i][j]) is int and m[i][j]>=0 and m[i][j]==m[j][i] for i in range(size) for j in range(size))
    support=[i for i in range(size) if sum(m[i])]
    visited=set();cs=[]
    for root in support:
        if root in visited:continue
        q=deque([root]);col={root:0};bip=True
        while q:
            i=q.popleft()
            for j in support:
                if not m[i][j]:continue
                if j not in col:col[j]=1-col[i];q.append(j)
                elif col[i]==col[j]:bip=False
        visited.update(col)
        ids=sorted(col);n=len(ids)
        t=[[F(m[i][j],sum(m[i])) for j in ids] for i in ids]
        t2=[[sum(t[i][k]*t[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
        fixed=n-rank([[t2[i][j]-int(i==j) for j in range(n)] for i in range(n)])
        assert fixed==(2 if bip else 1)
        pi=[F(sum(m[i]),sum(sum(m[j]) for j in ids)) for i in ids]
        assert all(sum(pi[i]*t[i][j] for i in range(n))==pi[j] for j in range(n))
        assert all(pi[i]*t[i][j]==pi[j]*t[j][i] for i in range(n) for j in range(n))
        directed_different=sum(m[i][j] for i in ids for j in ids if i!=j)
        assert directed_different%2==0
        cs.append(dict(class_ids=ids,period=2 if bip else 1,dim_left_fixed_T2=fixed,
                       directed_observations=sum(sum(m[i]) for i in ids),
                       directed_label_disagreements=directed_different,
                       pi=[str(f) for f in pi]))
    assert visited==set(support)
    return dict(supported_classes=len(support),unsupported_classes=size-len(support),components=cs,
        directed_obs=sum(sum(row) for row in m),directed_disagreements=sum(c['directed_label_disagreements'] for c in cs),
        directed_disagreements_in_period1=sum(c['directed_label_disagreements'] for c in cs if c['period']==1),
        fixed_dimension=sum(c['dim_left_fixed_T2'] for c in cs))

def coverage(r):
    f=r['results']['fit']
    distinct=[x for x in f['fit_label_tables'] if x['labels'][0]!=x['labels'][1]]
    stable=[x for x in distinct if x['groups']>=2]
    return dict(retained_fit_complete_pairs=f['within_role_complete_reciprocal_pairs'],
        retained_fit_directional_pairs=f['distinct_label_pairs'],supported_fit_different_label_types_in_two_groups=len(stable),
        minimum_fit_different_label_types_in_two_groups=2,
        fit_support_gate=len(stable)>=2,
        selection_pairs_label_blind=r['results']['selection']['within_role_complete_reciprocal_pairs'],
        qualification_groups_label_blind_upper_bound=r['results']['qualification']['within_role_reciprocal_groups'],
        absolute_floor_groups_if_all_directional=5,
        nonindependence='Metadata capacity only; qualification labels not indexed. The all-wins illustration is not power and does not certify independence.')

def main():
    orig=Path(sys.argv[1]).read_bytes();b=Path(sys.argv[2]).read_bytes()
    a=json.loads(orig);r=json.loads(b)
    cases={k:exact(m['observed_reverse_label_matrix']) for k,m in a['results'].items()}
    assert cases['canonical_train']['directed_obs']==2484 and cases['original_fit8192']['directed_obs']==98
    assert exact([[0,2],[2,0]])['components'][0]['period']==2
    assert exact([[2,2],[2,0]])['components'][0]['period']==1
    assert exact([[0,1,1],[1,0,1],[1,1,0]])['components'][0]['dim_left_fixed_T2']==1
    gate=coverage(r)
    # Sum-of-margins can call an assignment correct even if one observed row fails.
    margin_a,margin_b=12.,-2.
    summed_assignment=math.log1p(math.exp(-margin_a-margin_b))
    each_row_rival=math.log1p(math.exp(-margin_a))+math.log1p(math.exp(-margin_b))
    assert summed_assignment<.00005 and each_row_rival>2.
    report=dict(status='PASS_EXACT_SOFT_INVERSE_AND_FROZEN_SUPPORT_QUALIFICATION',
        exact_matrix=cases,support=gate,toy=dict(margin_a=margin_a,margin_b=margin_b,
          summed_margin_loss=summed_assignment,each_row_rival_loss=each_row_rival),
        conclusion='NONE_FROZEN_ROLE_CONTRAST_SUPPORT' if not gate['fit_support_gate'] else 'FURTHER_QUALIFICATION_REQUIRED',
        record_hashes=dict(canonical_hist=digest(orig),roster=digest(b),script=digest(Path(__file__).read_bytes())),
        new_task_forwards=0,optimizer_updates=0,gpu_allocations=0)
    if len(sys.argv)>3:
        with Path(sys.argv[3]).open('x',encoding='utf-8') as out:out.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
