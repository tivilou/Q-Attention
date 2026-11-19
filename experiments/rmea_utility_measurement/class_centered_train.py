"""Six matched conditions through the shared training loop; no global hooks."""
from experiments.rmea_utility_measurement.class_centered import ClassCenteredUtilityHead
from experiments.rmea_utility_measurement.train import TrainHead, run as shared_run

CONDITIONS = ('native', 'full', 'simplex', 'centered_native', 'centered_full', 'construction_only')


def head_factory(kind):
    if kind in CONDITIONS[:3]:
        return TrainHead(kind)
    if kind in CONDITIONS[3:]:
        return ClassCenteredUtilityHead(kind)
    raise ValueError('Undeclared matched training condition')


def run(arguments, design):
    assert design['conditions'] == list(CONDITIONS)
    result = shared_run(arguments, design, conditions=CONDITIONS, head_factory=head_factory)
    rows = result['results']; candidate = rows['centered_full']['metrics']
    comparisons = {}
    for name in CONDITIONS:
        if name == 'centered_full':
            continue
        metrics = rows[name]['metrics']
        comparisons[name] = dict(macro_f1=candidate['macro_f1'] - metrics['macro_f1'],
                                 macro_relative=(candidate['macro_f1'] - metrics['macro_f1']) / abs(metrics['macro_f1']) if metrics['macro_f1'] else None,
                                 ce=candidate['ce'] - metrics['ce'], accuracy=candidate['accuracy'] - metrics['accuracy'])
    macro_pass = all(comparisons[name]['macro_f1'] > 0 for name in ['native', 'full', 'simplex', 'centered_native'])
    ce_pass = all(comparisons[name]['ce'] <= 0 for name in ['native', 'centered_native'])
    reference_decision = result['decision']
    reference_joint_positive = result.pop('joint_macro_ce_positive')
    result.update(status='CLASS_CENTERED_JOINT_TRAINING_COMPLETE',
        decision='CLASS_CENTERED_DISCOVERY_GATE_PASS' if macro_pass and ce_pass else 'CLASS_CENTERED_SCREEN_INCONCLUSIVE_NO_PROMOTION',
        centered_full_vs=comparisons, centered_full_macro_gate=macro_pass, centered_full_ce_gate=ce_pass,
        reference_full_vs_native_decision=reference_decision,
        reference_full_vs_native_joint_macro_ce_positive=reference_joint_positive, architecture_completion=False,
        mechanism_attribution=False, independent_confirmation=False, formal_handoff=False, L5='NOT_ACCEPTED')
    return result
