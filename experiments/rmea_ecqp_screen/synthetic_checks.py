"""Synthetic checks for the actual metrics and nine-condition runner."""
import copy
import tempfile
from pathlib import Path

import numpy as np
import torch


def metric_checks(metrics):
    gold=np.array([0,1,1,2,2,3]);native=np.array([1,1,0,2,2,0]);perfect=gold.copy()
    observed=metrics.from_confusion(metrics.confusion(gold,native))
    assert observed['macro_f1']==1.5/39 and observed['accuracy']==.5
    groups=['a','a','b','c','c','d'];names={'native_quantum':native,'ecqp_quantum':perfect}
    actual=metrics.paired_group_bootstrap(gold,names,groups,replicates=23,seed=4)
    permutation=np.array([5,1,3,0,4,2])
    reordered=metrics.paired_group_bootstrap(gold[permutation],{n:p[permutation] for n,p in names.items()},
        [groups[i] for i in permutation],replicates=23,seed=4)
    assert reordered==actual and actual['intervals']['native_quantum']['delta_vs_native_quantiles']==[0.,0.]
    # Independent oracle: explicitly materialize repeated groups and count each
    # label using Python, not the optimized group sufficient statistics.
    unique=sorted(set(groups));rng=np.random.default_rng(4);draws={n:[] for n in names}
    for _ in range(23):
        draw=rng.integers(0,len(unique),size=len(unique))
        indices=[i for j in draw for i,g in enumerate(groups) if g==unique[j]]
        for name,pred in names.items():
            total=0.
            for label in range(1,40):
                tp=sum(gold[i]==label and pred[i]==label for i in indices)
                support=sum(gold[i]==label for i in indices);guessed=sum(pred[i]==label for i in indices)
                if support+guessed:total+=2*tp/(support+guessed)
            draws[name].append(total/39)
    delta=np.array(draws['ecqp_quantum'])-draws['native_quantum']
    np.testing.assert_allclose(actual['intervals']['ecqp_quantum']['delta_vs_native_quantiles'],
        np.quantile(delta,[.025,.975],method='linear'),rtol=0,atol=1e-15)
    fake={n:dict(macro_f1=.2,ce=1.,support=[1]*40) for n in
        ['native_quantum','ecqp_quantum','mismatch_quantum','square_quantum']}
    fake['ecqp_quantum']['macro_f1']=.3
    assert metrics.decision(fake,[-.01,.2],[1]*40)=='INCONCLUSIVE_NO_PROMOTION'
    missing=[1]*40;missing[1]=0
    assert metrics.decision(fake,[.01,.2],missing)=='INCONCLUSIVE_NO_PROMOTION'
    assert metrics.decision(fake,[.01,.2],[1]*40)=='EXPLORATORY_SCREEN_POSITIVE'
    fake['ecqp_quantum'].update(macro_f1=.1,ce=1.2)
    assert metrics.decision(fake,[-.2,.1],missing)=='CATASTROPHIC_FIXED_SCREEN'
    try:metrics.decision(fake,[float('nan'),.2],[1]*40)
    except AssertionError:pass
    else:raise AssertionError('Nonfinite interval accepted')
    return ['hand_count_fixed39_macro_and_accuracy','group_bootstrap_vs_explicit_repeated_group_oracle',
        'joint_row_permutation_and_native_zero_interval','interval_missing_class_catastrophic_and_invalid_precedence']


def training_checks(runner,design,metrics):
    rng=np.random.default_rng(71);h=rng.normal(0,.02,size=(32,1024)).astype('<f2')
    roles=np.zeros((32,4),dtype='<f4');roles[::4,2]=1;roles[1::4,3]=1
    ids=[f'synthetic-{i}' for i in range(8)];cohort=[dict(group=f'g-{i}',id_hash=s) for i,s in enumerate(ids)]
    splits={'fit':np.arange(4),'eval':np.arange(4,8)}
    donors={name:{ids[int(i)]:ids[int(indices[(j+1)%len(indices)])] for j,i in enumerate(indices)}
        for name,indices in splits.items()}
    summaries=[]
    for i in range(8):
        a=torch.from_numpy(h[4*i:4*i+4].astype(np.float32))[None]
        r=torch.from_numpy(roles[4*i:4*i+4])[None]
        _,_,entity=runner.ECQPHead.checked_inputs(a,r,torch.ones(1,4,dtype=torch.bool))
        summaries.append(entity.mean(1)[0].numpy())
    data=dict(h=h,roles=roles,offsets=np.arange(0,33,4),targets=np.arange(8,dtype=np.int64),
        metadata=[dict(id_hash=s) for s in ids],cohort=cohort,splits=splits,donors=donors,
        lookup={s:i for i,s in enumerate(ids)},summaries=np.stack(summaries))
    h_before=h.copy();roles_before=roles.copy()
    try:runner.batch(data,np.array([4]),'fit')
    except AssertionError:pass
    else:raise AssertionError('Evaluation sample entered fit batch')
    b=runner.batch(data,np.array([0,1]),'fit',True)
    np.testing.assert_array_equal(b[-1].numpy(),data['summaries'][[1,2]])
    with tempfile.TemporaryDirectory(prefix='ecqp-screen-draft-') as folder:
        result=runner.train_conditions(data,copy.deepcopy(design),Path(folder)/'final')
        assert set(result)=={c['id'] for c in design['conditions']}
        assert len({r['initial_shared_state_sha256'] for r in result.values()})==1
        assert all(r['final_epoch']==4 and len(r['epochs'])==4 for r in result.values())
        assert all(np.isfinite(r['final']['logits']).all() for r in result.values())
        assert result['ecqp_quantum']['gradient_max_abs']['head.B']>0
        summary=runner.summarize(data,result,metrics)
        assert summary['decision']=='INCONCLUSIVE_NO_PROMOTION' and summary['formal_handoff'] is False
        assert len(summary['metrics'])==9 and summary['uncertainty']['replicates']==2000
        assert summary['L5']=='NOT_ACCEPTED'
    np.testing.assert_array_equal(h,h_before);np.testing.assert_array_equal(roles,roles_before)
    return ['fit_eval_batch_isolation_and_fixed_donor','all9_actual_optimizer_final_checkpoint_roundtrips',
        'identical_shared_initial_state_and_finite_gradients','no_input_mutation',
        'actual_nine_condition_summary_bootstrap_and_missing_class_no_promotion']
