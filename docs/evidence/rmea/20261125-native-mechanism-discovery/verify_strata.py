"""Independent standard-library verification of exported aggregate counts."""
from pathlib import Path
import hashlib, json, math
r = Path(__file__).resolve().parent
v = json.loads((r / 'feature-strata.json').read_bytes())
assert v['strata_design_sha256'] == hashlib.sha256((r / 'strata-design.json').read_bytes()).hexdigest()
assert v['analysis_code_sha256'] == hashlib.sha256((r / 'stratify.py').read_bytes()).hexdigest()
models = ('quantum_distinct', 'dephased_distinct', 'gaussian_distinct')
all_rows = v['overall']
support = all_rows['support']
maximum = 0.


def check(row):
    global maximum
    assert len(row['support']) == 40 and sum(row['support']) == row['rows']
    pairs = row['paired_by_class']
    assert len(pairs) == 40 and all(len(x) == 8 for x in pairs)
    assert [sum(x) for x in pairs] == row['support']
    codes = [sum(x[k] for x in pairs) for k in range(8)]
    assert codes == row['paired_correct_q_g_d_codes']
    assert sum(codes[k] for k in (4, 5)) == row['q_correct_g_wrong']
    assert sum(codes[k] for k in (2, 3)) == row['g_correct_q_wrong']
    for m, bit in zip(models, (4, 1, 2)):
        cm = row['confusion_matrices'][m]
        assert len(cm) == 40 and all(len(x) == 40 for x in cm)
        assert [sum(x) for x in cm] == row['support']
        assert all(isinstance(n, int) and n >= 0 for x in cm for n in x)
        f1 = []
        for c in range(40):
            denominator = sum(cm[c]) + sum(cm[j][c] for j in range(40))
            f1.append(2 * cm[c][c] / denominator if denominator else 0.)
            assert sum(pairs[c][k] for k in range(8) if k & bit) == cm[c][c]
        maximum = max(maximum, abs(sum(f1) / 40 - row['metrics'][m]['all40_macro_f1']))
        assert max(abs(a-b) for a,b in zip(f1, row['metrics'][m]['per_class_f1'])) < 1e-12
        assert sum(cm[c][c] for c in range(40)) == row['metrics'][m]['correct']
    delta = sum((row['confusion_matrices'][models[0]][c][c] -
                 row['confusion_matrices'][models[2]][c][c]) / support[c] for c in range(40)) / 40
    assert abs(delta - row['additive_global_balanced_accuracy_q_minus_g']) < 1e-12


check(all_rows)
for rows in v['partitions'].values():
    for row in rows:
        check(row)
    assert sum(row['rows'] for row in rows) == 4096
    for c in range(40):
        assert sum(row['support'][c] for row in rows) == support[c]
        for k in range(8):
            assert sum(row['paired_by_class'][c][k] for row in rows) == all_rows['paired_by_class'][c][k]
        for m in models:
            for d in range(40):
                assert sum(row['confusion_matrices'][m][c][d] for row in rows) == all_rows['confusion_matrices'][m][c][d]
    assert abs(sum(row['additive_global_balanced_accuracy_q_minus_g'] for row in rows) - all_rows['additive_global_balanced_accuracy_q_minus_g']) < 1e-12
assert maximum < 1e-12


def count(codes, positive):
    classes = range(1, 40) if positive else (0,)
    return sum(all_rows['paired_by_class'][c][k] for c in classes for k in codes)


carryover = dict(status='DESCRIPTIVE_EXISTING_GEN98_COUNTS',
    q_correct_g_wrong=dict(positive=count((4, 5), True), null=count((4, 5), False)),
    g_correct_q_wrong=dict(positive=count((2, 3), True), null=count((2, 3), False)),
    all_threeway_codes=all_rows['paired_correct_q_g_d_codes'],
    blind_audit_four_strata=dict(Q_unique_correct_positive=count((4,), True),
        Q_unique_correct_null=count((4,), False), Q_unique_wrong_positive=count((3,), True),
        Q_unique_wrong_null=count((3,), False)),
    definition='Unique Q correct: Q correct and D/G both wrong, code4; unique Q wrong: Q wrong and D/G both correct, code3.',
    semantic_support_verified=False, cause_verified=False, new_model_evaluations=0)
(r / 'paired-carryover.json').write_text(json.dumps(carryover, indent=2) + '\n', encoding='utf8')
record = dict(status='PASS', implementation='independent standard-library aggregate recomputation',
    source_sha256=hashlib.sha256((r / 'feature-strata.json').read_bytes()).hexdigest(),
    macro_max_error=maximum, all40_support_cm_and_paired_agree=True,
    all_partitions_reconstruct_global=True, rows=4096,
    global_correct={m:all_rows['metrics'][m]['correct'] for m in models},
    net_q_minus_g_correct=all_rows['q_correct_g_wrong']-all_rows['g_correct_q_wrong'],
    length_net_q_minus_g={row['name']:row['q_correct_g_wrong']-row['g_correct_q_wrong'] for row in v['partitions']['length']},
    every_nonempty_bin_negative_global_class_weighted_recall_contribution=all(
        row['additive_global_balanced_accuracy_q_minus_g'] < 0
        for rows in v['partitions'].values() for row in rows if row['rows']),
    no_new_model_execution=True, newT=False, L3='NOT_ACCEPTED', L4_T='NOT_ACCEPTED')
(r / 'strata-verification.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf8')
print(json.dumps(dict(verification=record, carryover=carryover)))
