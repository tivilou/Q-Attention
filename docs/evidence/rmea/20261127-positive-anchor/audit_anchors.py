"""Recompute three frozen Q-RMEA evidence anchors; no data/model execution."""
import hashlib,itertools,json,statistics,sys
from fractions import Fraction as F
from pathlib import Path

FRESH='.codex-artifacts/qattn-rmea-implementation-20261026/fresh5-report-audit-20261030/report'
FULL='.codex-artifacts/full-protocol-fair-reference-20261124/owner-run'
SCREEN='.codex-artifacts/distinct-token-evidence-20261125'
A='docs/evidence/rmea/20261115-l4-acceptance'
B='docs/evidence/rmea/20261115-l4-statistical-audit/audit.json'
T975=2.7764451051977987  # Student-t .975 at df4, fixed original summary statistic

def load(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def f1(m,n):
    assert len(m)==40 and all(len(r)==40 for r in m) and sum(map(sum,m))==n
    assert all(isinstance(v,int) and v>=0 for row in m for v in row)
    ds=[sum(m[i])+sum(row[i] for row in m) for i in range(40)]
    return [F(2*m[i][i],d) if d else F() for i,d in enumerate(ds)]
def avg(v):return sum(v)/len(v)
def stats(v):
    v=list(map(float,v));n=len(v);m=statistics.mean(v);sd=statistics.stdev(v)
    p=sum(abs(sum(c*x for c,x in zip(s,v)))>=abs(sum(v))-1e-14 for s in itertools.product((-1,1),repeat=n))/2**n
    radius=T975*sd/n**.5
    return dict(delta=v,mean=m,positive=sum(x>0 for x in v),ci95=[m-radius,m+radius],exact_two_sided_sign_flip=p)
def exact_macro(rec,n,key):
    v=f1(rec['confusion_matrix'],n)
    assert abs(float(avg(v))-rec[key])<3e-15,(float(avg(v)),rec[key])
    return v
def delete_boundary(q,c):
    delta=[b-a for a,b in zip(q,c)];total=sum(delta);assert total>0
    rem=total
    for k,z in enumerate(sorted(delta,reverse=True),start=1):
        rem-=z
        if rem<0:break
    else:raise AssertionError('The exact positive-class deficit was never reversed')
    loo=[float((total-x)/39) for x in delta]
    return dict(classical_minus_q_min_after_one_deletion=min(loo),max_after_one_deletion=max(loo),
                any_single_label_deletion_reverses=min(loo)<0,oracle_min_deletions_to_strictly_reverse=k,
                scope='Worst-case deletion arithmetic only. No labels removed from the frozen metric and no class selection for execution.')

def audit(r):
    accept=load(r/A/'acceptance.json');older=load(r/B);fs=load(r/FRESH/'summary.json')
    assert accept['status']=='ACCEPTED_L4' and accept['conditional'] and 'quantum-specific advantage' in accept['not_supported']
    protected=[r/p for p in [accept['card'],*accept['preserved_hashes']]]
    assert [sha(p) for p in protected]==[accept['card_sha256'],*accept['preserved_hashes'].values()]
    assert all(sha(r/FRESH/k)==h for k,h in older['report_hashes'].items())
    order=accept['seeds']; assert order==[29,53,71,97,113] and len(fs['per_seed'])==5 and not fs['discovery_seed13_included']
    comparators=('uniform','relation_softmax','real_overlap','dephased')
    av={}
    for s in order:
        rec=fs['per_seed'][str(s)];assert rec['seed']==s and rec['full_data_training'] and not rec['test_selection'] and rec['common_initialization_equal']
        av[s]={k:avg(exact_macro(t['test'],13418,'macro_f1')) for k,t in rec['results'].items()}
    a={c:stats(av[s]['quantum']-av[s][c] for s in order) for c in comparators}
    for c,z in a.items():
        prior=fs['utility_statistics'] if c=='uniform' else fs['attribution_contrasts'][c]
        assert abs(z['mean']-prior['mean'])<3e-15 and z['positive']==prior['positive_seeds']
        assert all(abs(i-j)<3e-12 for i,j in zip(z['ci95'],prior['ci95'])) and z['exact_two_sided_sign_flip']==prior['exact_two_sided_sign_flip_p']
    ce=stats(j['ce_delta'] for j in older['rows'])
    assert abs(ce['mean']-older['contrasts']['frozen_mechanism_ce']['mean'])<3e-15 and ce['positive']==5
    lower_envelope={str(s):min(av[s]['quantum']-av[s][c] for c in comparators[1:]) for s in order}
    assert sum(i>0 for i in lower_envelope.values())==2

    full=r/FULL
    analysis=load(full/'result-analysis.json');f=load(full/'completed-report/summary.json')
    assert all(sha(full/'completed-report'/k)==v for k,v in analysis['report_sha256'].items())
    assert f['full_data_training'] and f['seed']==13 and not f['qualification_gate'] and analysis['disposition']=='CLOSE_EXACT_QUALIFICATION_PROFILE'
    fvec={};aggregate_checks=0
    for k,v in f['results'].items():
        assert v['updates']==9140 and len(v['history'])==20
        assert v['parameters']==(68200 if k=='uniform' else 81028), (k,v['parameters'])
        for role,n in [('valid',19584),('test',13418)]:
            vec=exact_macro(v[role],n,'macro_f1');aggregate_checks+=1
            if role=='test':fvec[k]=vec
    ff={c:float(avg(fvec['quantum'])-avg(v)) for c,v in fvec.items() if c!='quantum'}
    assert sum(i>0 for i in ff.values())==1

    sc=r/SCREEN; z=load(sc/'summary.json');sv=load(sc/'verification.json');decision=load(sc/'primary-disposition.json')
    assert sha(sc/'summary.json')==sv['exact_remote_hashes']['summary.json'] and sha(sc/'fixtures.json')==sv['exact_remote_hashes']['fixtures.json']
    assert z['assessment_items']==4096 and z['train_items']==8192 and not z['valid_read'] and not z['test_read']
    assert z['gate']==decision['decision']=='DEFER_EXACT_DISTINCT_TOKEN_SCREEN' and decision['full_data']==decision['multi_seed']=='NOT_STARTED'
    assert sv['all_fits_before_assessment'] and sv['all40_coverage'] and sv['sentence_group_overlap']==0
    allv={**z['results'],**{'reused/'+k:v for k,v in z['reused_results'].items()}}
    svec={k:exact_macro(x,4096,'all40_macro_f1') for k,x in allv.items()};q=svec['quantum_distinct']
    checked={'macro_beats_'+k:avg(q)>avg(v) for k,v in svec.items() if k!='quantum_distinct'}
    assert checked==z['gates']==decision['gates'] and sum(checked.values())==10 and len(checked)==13
    names=('dephased_distinct','gaussian_distinct','reused/earlier/native_gaussian_covariance','reused/previous/quantum_shared_mean_product')
    cc={k:float(avg(q)-avg(svec[k])) for k in names}
    cm_full=delete_boundary(fvec['quantum'],fvec['real12']); cm_screen=delete_boundary(q,svec[names[2]])
    cm_matched_gaussian=delete_boundary(q,svec[names[1]])
    return dict(status='PASS_THREE_FROZEN_ANCHORS_NO_SAME_CORE_REOPEN',classification='NONE_SAME_CORE_POSITIVE_ANCHOR',
        fresh5=dict(seeds=order,macro_pairs=a,ce_frozen_alignment=ce,q_minus_best_nonuniform_arm_by_seed={s:float(v) for s,v in lower_envelope.items()},q_beats_all_three_nonuniform_same_seed=sum(v>0 for v in lower_envelope.values()),attribution='Original statistical claim is ordinary pooling utility plus matched permutation sensitivity, not Q-specific benefit.'),
        gen49=dict(macro={k:float(avg(v)) for k,v in fvec.items()},q_delta=ff,qualification='CLOSE_EXACT_QUALIFICATION_PROFILE',passed_required=1,total_required=4,no_single_class_reversal=cm_full),
        gen98=dict(macro={k:float(avg(svec[k])) for k in ('quantum_distinct',*names)},q_delta=cc,failed_frozen_gates=[k for k,v in checked.items() if not v],passed_required=10,total_required=13,full_data='NOT_STARTED', versus_gcov_class_deletion=cm_screen,versus_new_gaussian_class_deletion=cm_matched_gaussian),
        recomputed_confusion_matrices=5*5+aggregate_checks+len(svec),records_sha256={k:sha(r/p) for k,p in {'accepted_card':A+'/acceptance.json','fresh5':FRESH+'/summary.json','gen49':FULL+'/completed-report/summary.json','gen98':SCREEN+'/summary.json'}.items()},
        interpretive_scope='No pooled cross-study estimates. Leave-one-label arithmetic is descriptive, not class resampling, row pairing, or a changed evaluation.',
        task_rows=0,task_forwards=0,optimizer_updates=0,gpu_allocations=0,script_sha256=sha(Path(__file__)))

if __name__=='__main__':print(json.dumps(audit(Path(sys.argv[1]).resolve()),indent=2,allow_nan=False))
