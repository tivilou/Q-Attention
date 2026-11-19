"""Five matched conditions through the unchanged shared training loop."""
from experiments.rmea_utility_measurement.cross_fold import CrossFoldUtilityHead
from experiments.rmea_utility_measurement.train import TrainHead, run as shared_run

CONDITIONS = ('native', 'full', 'simplex', 'self_fold', 'cross_fold')


def head_factory(kind):
    if kind in CONDITIONS[:3]:
        return TrainHead(kind)
    if kind in CONDITIONS[3:]:
        return CrossFoldUtilityHead(kind)
    raise ValueError('Undeclared matched training condition')


def run(arguments, design):
    assert design['conditions'] == list(CONDITIONS)
    result = shared_run(arguments, design, conditions=CONDITIONS, head_factory=head_factory)
    rows = result['results']; candidate = rows['cross_fold']['metrics']
    comparisons = {}
    for name in CONDITIONS[:-1]:
        metrics = rows[name]['metrics']; delta = candidate['macro_f1'] - metrics['macro_f1']
        comparisons[name] = dict(macro_f1=delta,
            macro_relative=delta / abs(metrics['macro_f1']) if metrics['macro_f1'] else None,
            ce=candidate['ce'] - metrics['ce'], accuracy=candidate['accuracy'] - metrics['accuracy'])
    macro_pass = all(comparisons[name]['macro_f1'] > 0 for name in CONDITIONS[:-1])
    ce_pass = all(comparisons[name]['ce'] <= 0 for name in ('native', 'self_fold'))
    reference_decision = result['decision']
    reference_joint_positive = result.pop('joint_macro_ce_positive')
    result.update(status='CROSS_FOLD_JOINT_TRAINING_COMPLETE',
        decision='CROSS_FOLD_DISCOVERY_GATE_PASS' if macro_pass and ce_pass else 'CROSS_FOLD_SCREEN_INCONCLUSIVE_NO_PROMOTION',
        cross_fold_vs=comparisons, cross_fold_macro_gate=macro_pass, cross_fold_ce_gate=ce_pass,
        reference_full_vs_native_decision=reference_decision,
        reference_full_vs_native_joint_macro_ce_positive=reference_joint_positive,
        architecture_completion=False, mechanism_attribution=False, noise_independence_established=False,
        independent_confirmation=False, formal_handoff=False, L5='NOT_ACCEPTED')
    return result
