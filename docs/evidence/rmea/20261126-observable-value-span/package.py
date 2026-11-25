"""Reopen frozen code, actual execution and all-class gates; no task replay."""
from pathlib import Path
import hashlib
import json
import ast

r = Path(__file__).resolve().parent


def sha(b):
    return hashlib.sha256(b).hexdigest()


def read(n):
    return json.loads((r / n).read_bytes())


s, d, v = read('summary.json'), read('design.json'), read('verification.json')
f, inp, pre = read('fit-lock.json'), read('input-verification.json'), read('local-preflight.json')
execution, audit, ae = read('execution.json'), read('statistics-audit.json'), read('statistics-audit-execution.json')
assert s['status'] == 'COMPLETE' and s['decision'] == 'STOP_FIXED_REPRESENTATION_VALUE_REPLACEMENT'
assert s['gates'] == dict(mean_MSE=False, each_class_MSE=False, teacher_agreement=False)
assert d['gate'] == dict(assessment_mean_all40_normalized_MSE_max=.01,
                         assessment_each_class_normalized_MSE_max=.04, teacher_argmax_agreement_min=.99)
for arm, m in s['metrics']['assessment'].items():
    a = m['normalized_MSE_all40']
    assert len(a) == len(m['MSE_all40']) == 40
    assert abs(sum(a) / 40 - m['mean_normalized_MSE']) < 1e-14
    assert max(a) == m['max_normalized_MSE']
    assert m['examples'] == 4096 and m['agreement_count'] == 2770
    assert m['teacher_argmax_agreement'] == m['agreement_count'] / m['examples']
    assert sum(a) / 40 > .01 and max(a) > .04 and m['teacher_argmax_agreement'] < .99
assert s['quantum_solver']['rank'] == f['quantum_solver']['rank'] == 64
assert s['diagonal_solver']['rank'] == f['diagonal_solver']['rank'] == 8
assert f['assessment_forward_count_at_lock'] == v['assessment_forward_count_at_fit_lock'] == 0
assert f['coefficient_sha256'] == s['private_coefficient_sha256'] == audit['coefficient_sha256']
assert v['coefficient_hash_unchanged'] and v['effective_teacher_state_unchanged']
assert max(v['max_errors'].values()) < 1e-12 and v['numpy_matrix_replay_examples'] == 2
assert inp['status'] == 'PASS_EXACT_INPUTS' and len(inp['checks']) == len(d['input_hashes']) == 51
for path, h in d['input_hashes'].items():
    assert inp['checks'][path]['exists'] and inp['checks'][path]['sha256'] == h
assert inp['cohort_fit'] == 8192 and inp['cohort_assessment'] == 4096 and inp['cohort_group_overlap'] == 0
assert pre['status'] == 'PASS_LOCAL_PREFLIGHT' and not pre['external_research_review']
assert all(sha((r / n).read_bytes()) == h for n, h in pre['files'].items())
assert execution['exit_code'] == 0 and execution['stderr'] == ''
for n, h in execution['file_hashes'].items():
    if n != 'host_run.py':
        assert sha((r / n).read_bytes()) == h
literal = json.loads(execution['stdout'].splitlines()[-1])
assert literal['status'] == s['status'] and literal['gates'] == s['gates']
assert literal['assessment'] == s['metrics']['assessment']['quantum_span']
assert s['unique_TRAIN_feature_examples'] == 12288 and s['example_passes'] == 20480
assert s['native_head_forwards'] == 320 and s['native_circuit_executions'] == 640
assert s['linear_solutions'] == 2 and s['GPU_count'] == 1 and 0 < s['elapsed_seconds'] < 600
assert all(s[k] == 0 for k in ('optimizer_updates', 'gold_label_decodes', 'new_official_VALID_TEST_examples'))
complete = read('RUN_COMPLETE')
assert complete['summary_sha256'] == sha((r / 'summary.json').read_bytes())
assert complete['verification_sha256'] == sha((r / 'verification.json').read_bytes())
assert audit['status'] == 'PASS_SAME_SOLUTION_SUFFICIENT_STATISTICS_AUDIT'
assert not audit['private_arrays_returned'] and audit['new_native_forwards'] == audit['new_feature_or_checkpoint_reads'] == 0
assert ae['exit_status'] == 0 and ae['stderr'] == '' and json.loads(ae['stdout']) == audit
code = ae['argv'][-1]
assert sha(code.encode()) == ae['remote_code_sha256']
ast.parse(code)
(r / 'statistics_audit_code.py').write_bytes(code.encode())
for x in audit['checks'].values():
    assert x['relative_coefficient_difference'] < 1e-12
    assert x['relative_normal_equation_residual'] < 1e-12
    assert x['max_fit_MSE_difference_sufficient_vs_second_pass'] < 1e-11
close, monitor = read('allocation-closeout.json'), read('last-observation.json')
e = monitor['evidence']
assert close['state'] == 'released' and close['generation'] == e['generation'] == 102
assert close['allocation_id'] == e['allocation_id'] and e['campaign_status'] == 'completed'
assert all(e[k] for k in ('all_jobs_terminal', 'quiescent', 'no_remaining_consumers', 'resources_reusable'))
assert monitor['progress']['current'] == monitor['progress']['total'] == 20480
resource = {k: e[k] for k in ('allocation_id', 'generation', 'run_id', 'runner_attempt_id', 'campaign_status',
                            'all_jobs_terminal', 'quiescent', 'no_remaining_consumers', 'resources_reusable')}
resource.update(allocation_state=close['state'], controller_release_receipt_sha256=sha((r / 'allocation-closeout.json').read_bytes()),
                controller_terminal_monitor_sha256=sha((r / 'last-observation.json').read_bytes()),
                private_arrays_collected=False, task_execution_exit_code=execution['exit_code'])
(r / 'resource-verification.json').write_bytes((json.dumps(resource, indent=2) + '\n').encode())
n = read('next-direction.json')
assert not n['selected_task_method'] and n['goal'] == s['goal'] == 'ACTIVE'
assert all(x == 0 for x in n['resources'].values())
assert n['newL3'] == n['newL4T'] == s['newL3'] == s['newL4T'] == 'NOT_ACCEPTED'
names = ('runner.py', 'span_math.py', 'check_math.py', 'math-check.json', 'protocol.md', 'design.json',
         'local-preflight.json', 'preparation.json', 'input_probe.py', 'input-verification.json', 'summary.json',
         'verification.json', 'fit-lock.json', 'RUN_COMPLETE', 'execution.json', 'statistics-audit.json',
         'statistics-audit-execution.json', 'statistics_audit_code.py', 'private-output-inventory.json',
         'resource-verification.json', 'analysis.md', 'next-direction.json')
raw = {n: (r / n).read_bytes() for n in names}
for name, b in raw.items():
    if name.endswith('.py'): ast.parse(b.decode('utf8'))
    if name.endswith('.json'): json.loads(b)
out = dict(status='PASS_COMPLETED_DIAGNOSTIC_EVIDENCE_PACKAGE', scientific_gates_passed=0, scientific_gates_total=3,
           all40_gate_recomputation=True, literal_execution_exit0_verified=True, frozen_code_and_inputs_verified=True,
           same_solution_statistics_audit_verified=True, generation102_released=True,
           public_evidence_sha256={n: sha(b) for n, b in raw.items()}, external_research_review=False,
           goal='ACTIVE', newL3='NOT_ACCEPTED', newL4T='NOT_ACCEPTED')
(r / 'package-verification.json').write_bytes((json.dumps(out, indent=2) + '\n').encode())
print(json.dumps({k: v for k, v in out.items() if k != 'public_evidence_sha256'}))
