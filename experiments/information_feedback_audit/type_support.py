cfg={'at': '2026-11-26T13:15:50.082709+00:00', 'scope': 'One saved-output comparison, all frozen11 models; original40-class TRAIN assessment only', 'cohort_sha256': '8fb8393bbe2241929748283f2bbaa219682696409857ec15678585bcd0b88ff0', 'sources': [{'name': 'context-spectral-flow-20261125', 'modes': ['quantum', 'dephased'], 'summary_sha256': 'ca8d6f75cc8726580004a0c432ddee8d3dae36a51222b13a7c15da09b43a50c6', 'diagnostic_sha256': 'bd97826f169479d0af7716e9711343ec3f571d69d0e0c346ee2100f8d18fcd19'}, {'name': 'staged-bottleneck-design-20261125', 'modes': ['gaussian_base'], 'summary_sha256': 'f6382f2771a04c34a74dfd28598cd25818408c4c018c510bd2693d0bcfc6e094', 'diagnostic_sha256': '30a2615a0cbb9676b2c985574972d07b54149166790911dd5da64e1659eefa50'}, {'name': 'connected-value-pooling-20261125', 'modes': ['quantum_covariance', 'gaussian_covariance', 'quantum_mean_product', 'gaussian_mean_product', 'dephased_covariance'], 'summary_sha256': 'ae04452aa0a79f52542c31a04d8b911b3c06c33c478fe685d9de6461a1d35f95', 'diagnostic_sha256': '87e60431f0a5a76af415db3fdc0d6e206a04a8a23473560d0426d55b49eb6040'}, {'name': 'distinct-token-evidence-20261125', 'modes': ['quantum_distinct', 'dephased_distinct', 'gaussian_distinct'], 'summary_sha256': 'dc3ed63bc46f15fb193b456eb302fc68435f48469d701e81677e34938262c2c7', 'diagnostic_sha256': '3677d55f2e6d3c64df6c9789c76f7113b66a79ee1efbee7531bc9bf2bf5d1307'}], 'model_order': ['quantum', 'dephased', 'gaussian_base', 'quantum_covariance', 'gaussian_covariance', 'quantum_mean_product', 'gaussian_mean_product', 'dephased_covariance', 'quantum_distinct', 'dephased_distinct', 'gaussian_distinct'], 'fixed_outputs': ['all40 counts and Macro', 'deployed-argmax pair table vs strong Gcov for all40 true classes', 'global/null/positive correctness masks', 'top5 Gcov error cells by count then classID'], 'no_strata_threshold_search': True, 'old_positive39_diagnostic': 'Already available; does not count as new deployed-pair evidence', 'worker_sha256': '4a84cbb1c12aa9e37fde6670b1bb4dae6fe9517046d03b078abccfe9c253e995', 'new_fits': 0, 'new_model_forwards': 0}

from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,time
import numpy as np
t=time.monotonic();root=Path('/hy-tmp/Q-Attention')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cohort=root/'control/staged-bottleneck-design-20261125/cohort.private.json'
assert sha(cohort)==cfg['cohort_sha256']
split=json.loads(cohort.read_bytes());fit=np.array(split['fit']);ids=np.array(split['assessment'])
assert len(fit)==8192 and len(ids)==4096 and not set(fit)&set(ids)
cache=root/'inputs/native-baseline/discovery/cache'
assert sha(cache/'metadata.json')=='9e6eac9ad582a4ef4e1a8a12190bfef3b8f0a18c5efbd9072f91f31563656c0e'
assert sha(cache/'train-targets.npy')=='6ce409b586a421995f7d70906f062a7e91fe9d35a04be5359a2f58a3485ee604'
metadata=json.loads((cache/'metadata.json').read_bytes())['train']
y=np.load(cache/'train-targets.npy',allow_pickle=False)
raw=root/'inputs/native-baseline/recovery-inputs/train.jsonl'
assert sha(raw)=='f1051d574a44df3ecbd036c1de4fbbe0a372c768f6ec990330e2cea4a657269f'
rows=[json.loads(s) for s in raw.read_bytes().splitlines() if s.strip()]
lookup={hashlib.sha256(str(r['metadata']['id']).encode()).hexdigest():r for r in rows}
assert len(lookup)==len(rows)==len(metadata)==len(y)==58465
ordered=[lookup[m['id_hash']] for m in metadata]
pairs=[(r['metadata']['subj_type'],r['metadata']['obj_type']) for r in ordered]
mapping={}
for r,label in zip(ordered,y):
    label=int(label)
    if r['label'] in mapping:assert mapping[r['label']]==label
    else:mapping[r['label']]=label
assert len(mapping)==len(set(mapping.values()))==40
groups=[hashlib.sha256(json.dumps(r['tokens'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest() for r in ordered]
assert not {groups[i] for i in fit}&{groups[i] for i in ids}
assert all(groups[int(i)]==split['selected_groups'][str(int(i))] for i in np.r_[fit,ids])
support=defaultdict(Counter)
for i in fit:support[pairs[i]][int(y[i])]+=1
known=np.array([pairs[i] in support for i in ids])
gold=y[ids];gold_supported=np.array([gold[j]==0 or support[pairs[i]][int(gold[j])]>0 for j,i in enumerate(ids)])
report={};pins=[]
for source in cfg['sources']:
    folder=root/'control'/source['name']/'output';sp=folder/'summary.json';dp=folder/'diagnostic.private.npz'
    assert sha(sp)==source['summary_sha256'] and sha(dp)==source['diagnostic_sha256']
    summary=json.loads(sp.read_bytes())
    with np.load(dp,allow_pickle=False) as data:
        assert np.array_equal(data['assessment_indices'],ids) and np.array_equal(data['labels'],y)
        for mode in source['modes']:
            logits=data[mode+'_logits'];assert logits.shape==(4096,40) and np.isfinite(logits).all()
            pred=logits.argmax(-1);correct=pred==gold;error=~correct
            ps=np.array([pred[j]==0 or support[pairs[i]][int(pred[j])]>0 for j,i in enumerate(ids)])
            cm=np.bincount(40*gold+pred,minlength=1600).reshape(40,40)
            assert cm.tolist()==summary['results'][mode]['confusion_matrix']
            den=cm.sum(0)+cm.sum(1);macro=float(np.divide(2*cm.diagonal(),den,out=np.zeros(40),where=den>0).mean())
            strata={}
            for title,mask in [('all_errors',error),('gold_null_errors',error&(gold==0)),('positive_to_null',error&(gold>0)&(pred==0)),('positive_wrong_positive',error&(gold>0)&(pred>0))]:
                a=int((mask&known&gold_supported&~ps).sum())
                b=int((mask&known&gold_supported&ps).sum())
                c=int((mask&known&~gold_supported).sum())
                d=int((mask&~known).sum())
                assert a+b+c+d==int(mask.sum())
                strata[title]={'n':int(mask.sum()),'fit_known_gold_supported_pred_unsupported':a,
                    'fit_known_gold_and_pred_supported':b,'fit_known_gold_unsupported':c,'fit_pair_unseen':d}
            report[mode]={'all40_macro':macro,'correct':int(correct.sum()),'errors':strata,
                'correct_but_empirically_unsupported':int((correct&known&~ps).sum())}
    pins.append({'name':source['name'],'summary_sha256':sha(sp),'diagnostic_sha256':sha(dp)})
    assert time.monotonic()-t<60
assert list(report)==cfg['model_order']
print(json.dumps({'status':'PASS_EXPLORATORY_FIT_TYPE_SUPPORT_DIAGNOSTIC','models':report,
 'fit_rows':len(fit),'assessment_rows':len(ids),'fit_type_pairs':len(support),
 'assessment_pair_unseen':int((~known).sum()),'assessment_gold_nonnull_unsupported':int(((gold>0)&~gold_supported).sum()),
 'assessment_gold_nonnull_unsupported_by_class':np.bincount(gold[(gold>0)&~gold_supported],minlength=40).tolist(),
 'fit_class_counts':np.bincount(y[fit],minlength=40).tolist(),
 'source_pins':pins,'train_source_sha256':sha(raw),'cohort_sha256':sha(cohort),
 'seconds':time.monotonic()-t,'new_fits':0,'new_model_forwards':0,'gpu_initializations':0,
 'valid_test_read':False,'server_writes':0,'raw_rows_exported':False,
 'limit':'FIT support absence is not a semantic constraint; retrospective error strata do not certify benefit of a filter or quantum type mechanism.'}))
