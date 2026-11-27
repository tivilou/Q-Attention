"""Reconcile the released aggregate and exercise rank ties on artificial inputs."""
from pathlib import Path
import hashlib,itertools,json,sys
import numpy as np
from rank_readonly import levels

h=Path(__file__).resolve().parent;x=json.loads((h/'rank-result.json').read_bytes());contract=json.loads((h/'frozen-scope.json').read_bytes())
assert hashlib.sha256((h/'rank_readonly.py').read_bytes()).hexdigest()==contract['audit_source_sha256']
assert x['source_hashes']==contract['input_sha256'] and x['exact_confusion_parity']==4
assert len(x['pairwise'])==6 and x['positive']+x['null']==x['total']==4096
assert all(x[k]==0 for k in ('task_forwards','optimizer_updates','gpu_allocations','remote_writes','checkpoint_decodes','feature_arrays_read','official_splits_read','individual_records_exported'))
for name,r in x['reports'].items():
    cnt=r['positive_codes'];hist=r['positive_rank_histogram_1_to_39']
    assert sum(hist)==sum(cnt)==x['positive'];assert sum(cnt[1:4])==r['positive_to_null'] and sum(cnt[4:])==r['positive_to_wrong_positive']
    assert r['positive_rank1_all']==cnt[0]+cnt[1]==hist[0] and r['positive_rank_top3_all']==sum(hist[:3])==sum(cnt[i] for i in [0,1,2,4])
    assert r['null_false_positive']+r['null_correct']==x['null']
for a,b in itertools.combinations(('Q','D','G','Gcov'),2):
    r=x['pairwise'][a+'__'+b];m=r['positive_codes_matched'];n=r['null_fp_status_a_by_b'];ra=x['reports'][a];rb=x['reports'][b]
    assert list(map(sum,m))==ra['positive_codes'] and list(map(sum,zip(*m)))==rb['positive_codes']
    assert r['positive_correct_a_only']==sum(m[0][1:]) and r['positive_correct_b_only']==sum(row[0] for row in m[1:])
    assert list(map(sum,n))==[ra['null_correct'],ra['null_false_positive']] and list(map(sum,zip(*n)))==[rb['null_correct'],rb['null_false_positive']]
    assert r['all_correct_a_only']==r['positive_correct_a_only']+n[0][1] and r['all_correct_b_only']==r['positive_correct_b_only']+n[1][0]
    assert sum(r[k] for k in ('gold_positive_rank_a_better','gold_positive_rank_b_better','gold_positive_rank_equal'))==x['positive']
    assert r['top3_a_only']-r['top3_b_only']==ra['positive_rank_top3_all']-rb['positive_rank_top3_all']
y=np.array([1,1,3,6,3,6,0]);scores=np.array([[1,4,3,1,0,0,0],[4,4,3,1,0,0,0],[4,3,1,3,0,0,0],
    [4,3,3,3,3,3,1],[0,3,3,3,2,0,0],[0,5,5,5,5,4,1],[1,1,0,0,0,0,0]],float)
k,p,c,t=levels(scores,y)
assert c.tolist()==list(range(6)) and p[-1]==0 and k.tolist()==[1,1,2,6,3,6]
assert t['tied_null_with_gold']==1 and t['positive_gold_tied_with_another_positive']==2
q=x['reports']['Q']['positive_codes'];g=x['reports']['Gcov']['positive_codes']
assert sum(q[1:])==485 and sum(g[1:])==481 and x['reports']['Q']['null_false_positive']-x['reports']['Gcov']['null_false_positive']==55
print(json.dumps(dict(status='PASS_AGGREGATE_CROSS_TABS_AND_ALL_SIX_SYNTHETIC_CATEGORIES',paired_tables=6,positive_categories=6,synthetic_ties_checked=True,remote_executions=0,rows_outside_synthetic_checked_here=0)))
