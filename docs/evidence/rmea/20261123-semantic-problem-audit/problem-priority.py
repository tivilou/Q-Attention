"""R0 task-headroom arithmetic from completed, public aggregate evidence only."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / '.codex-artifacts/pooled-type-reference-20261123/historical-replay'


def macro(matrix, null):
    size = len(matrix)
    scores = []
    for i in range(size):
        if i == null:
            continue
        denominator = sum(matrix[i]) + sum(row[i] for row in matrix)
        scores.append(2 * matrix[i][i] / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


def main():
    source_paths = [SOURCE / name for name in ('summary.json', 'residual-summary.json')]
    summary, residual = [json.loads(path.read_bytes()) for path in source_paths]
    baseline = summary['results']['relation_softmax']['typed']
    names = [row['label'] for row in baseline['per_class']]
    null = names.index('no_relation')
    original = baseline['confusion_matrix']
    total = sum(map(sum, original))
    assert total == summary['rows'] == residual['rows'] == 19584
    before = macro(original, null)
    assert abs(before - baseline['positive39_macro_f1']) < 1e-14
    scenarios = []
    for mode in ('shared_positive_fn', 'shared_fn_zero_rule_exclusions',
                 'all_positive_fn', 'all_null_fp', 'all_wrong_positive'):
        matrix = [row.copy() for row in original]
        repairs = 0
        if mode.startswith('shared'):
            for row in residual['per_class']:
                if mode == 'shared_fn_zero_rule_exclusions' and row['gold_excluded_by_fixed_rule']:
                    continue
                i = names.index(row['label'])
                count = row['all_four_fn']
                assert 0 <= count <= matrix[i][null]
                matrix[i][null] -= count
                matrix[i][i] += count
                repairs += count
        elif mode == 'all_positive_fn':
            for i in range(len(names)):
                if i != null:
                    count = matrix[i][null]
                    matrix[i][null] = 0
                    matrix[i][i] += count
                    repairs += count
        elif mode == 'all_null_fp':
            for j in range(len(names)):
                if j != null:
                    count = matrix[null][j]
                    matrix[null][j] = 0
                    matrix[null][null] += count
                    repairs += count
        else:
            for i in range(len(names)):
                if i == null:
                    continue
                for j in range(len(names)):
                    if j in (i, null):
                        continue
                    count = matrix[i][j]
                    matrix[i][j] = 0
                    matrix[i][i] += count
                    repairs += count
        assert sum(map(sum, matrix)) == total
        assert list(map(sum, matrix)) == list(map(sum, original))
        assert sum(matrix[i][i] for i in range(len(names))) == baseline['correct'] + repairs
        scenarios.append(dict(name=mode, repairs=repairs, macro39=macro(matrix, null),
                              absolute_delta=macro(matrix, null) - before))
    assert scenarios[0]['repairs'] == residual['all_four_positive_fn'] == 1155
    assert scenarios[1]['repairs'] == 721
    assert scenarios[2]['repairs'] == baseline['positive_false_negative'] == 1976
    assert scenarios[3]['repairs'] == baseline['null_false_positive'] == 1046
    assert scenarios[4]['repairs'] == baseline['positive_wrong_relation'] == 409
    per_class = []
    for row in residual['per_class']:
        i = names.index(row['label'])
        modified = [values.copy() for values in original]
        modified[i][null] -= row['all_four_fn']
        modified[i][i] += row['all_four_fn']
        per_class.append(dict(label=row['label'], support=row['support'],
                              shared_fn=row['all_four_fn'],
                              gold_rule_exclusions=row['gold_excluded_by_fixed_rule'],
                              single_class_shared_fn_repair_macro_delta=macro(modified, null)-before))
    assert abs(sum(row['single_class_shared_fn_repair_macro_delta'] for row in per_class)
               - scenarios[0]['absolute_delta']) < 1e-13
    small = [row for row in per_class if row['support'] < 100]
    result = dict(
        schema='task-headroom-assessment.v1',
        created_at=datetime.now(timezone.utc).isoformat(),
        source_hashes={str(path.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in source_paths},
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope=dict(dataset='Re-TACRED 1.1', population='all 19584 reused development VALID rows',
                   inputs='same rebuilt frozen encoder cache; same TRAIN-only empirical type rule for all models',
                   comparator='historical seed13 relation_softmax, fixed selected epoch19',
                   labels='unchanged original labels', metric='positive39 macro F1',
                   formal_metric='historical formal metrics unchanged', nominal_maximum_not_assumed_attainable=True),
        baseline_macro39=before,
        bound=dict(status='exact_under_stated_assumptions', kind='label-aware empirical repair oracle',
                   assumptions='Correct exactly the named existing errors using gold labels; all other predictions fixed. '
                               'This does not establish a label-free rule, shared-parameter/circuit reachability, '
                               'annotation truth, expected gain, independent evidence or a population ceiling.'),
        scenarios=scenarios,
        legal_shared_fn_count=residual['all_four_positive_fn_with_gold_allowed'],
        legal_subset_note='721 is a conservative subset of the 1070 legal shared misses: retain only relation classes '
                          'with zero gold exclusions anywhere in this reused VALID population. Posthoc descriptive '
                          'selection, not a deployment rule or a new model target.',
        frequency_sensitivity=dict(posthoc_cutoff=100, classes=len(small),
                                   positive_rows=sum(row['support'] for row in small),
                                   shared_fn=sum(row['shared_fn'] for row in small),
                                   oracle_macro_delta=sum(row['single_class_shared_fn_repair_macro_delta'] for row in small),
                                   warning='No class selection, resampling, loss weighting or tuning authorized by this table.'),
        per_class=per_class,
        decision='Task problem warrants research: recover explicit entity-pair relations currently rejected as null '
                 'without increasing false positives; assess all positive classes, including rare relations. '
                 'Next derive a label-free shared evidence-versus-null hypothesis with equal-input classical control. '
                 'Do not select a quantum mechanism from oracle arithmetic.',
        runtime_scope=dict(new_training=0, model_forwards=0, gpu_jobs=0, new_valid_predictions=0,
                           test_rows=0, cos_writes=0, new_raw_windows_payload=0),
        verification='PASS: original Macro reproduced; counts, total, per-gold-row support, exact repairs '
                     'and classwise Macro additivity checked.')
    target = HERE / 'problem-priority.json'
    assert not target.exists(), 'Immutable completed result exists; inspect it instead of rerunning.'
    target.write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode())
    print(json.dumps(dict(baseline=before, scenarios=scenarios,
                          frequency_sensitivity=result['frequency_sensitivity'], verification='PASS')))


if __name__ == '__main__':
    main()
