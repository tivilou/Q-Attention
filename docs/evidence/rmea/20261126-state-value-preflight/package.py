"""Reopen the exact math, source, execution and prospective-input records."""
from pathlib import Path
import ast
import hashlib
import json

r = Path(__file__).resolve().parent
names = ('scope.json', 'receipt.before-execution.json', 'source-pins.json', 'derivation.md',
         'preflight.py', 'preflight-execution.json', 'results.json', 'independent.py',
         'independent-execution.json', 'independent-results.json', 'history-comparison.json',
         'readout-span-protocol.md', 'next-direction.json', 'task-input-preview.json',
         'verification.json', 'analysis.md')


def sha(b):
    return hashlib.sha256(b).hexdigest()


raw = {n: (r / n).read_bytes() for n in names}
receipt = json.loads(raw['receipt.before-execution.json'])
assert all(sha(raw[n]) == h for n, h in receipt['files'].items())
scope, v = json.loads(raw['scope.json']), json.loads(raw['verification.json'])
a, b = json.loads(raw['results.json']), json.loads(raw['independent-results.json'])
assert a['status'] == 'PASS_DENSE_OPERATOR_IDENTITIES'
assert b['status'] == 'PASS_INDEPENDENT_EXACT_RATIONAL_CHECKS'
assert a['random_cases'] == 80 and a['random_token_density_matrices'] == 560
assert b['rank_one_positive_events_checked'] == 48 and b['zero_events_left_undefined'] == 12
assert b['primary_results_sha256'] == sha(raw['results.json'])
assert max(a['max_absolute_errors'].values()) < scope['tolerance']['dense_identity_absolute']
for prefix in ('preflight', 'independent'):
    run = json.loads(raw[prefix + '-execution.json'])
    assert run['code_sha256'] == sha(raw[prefix + '.py'])
    assert run['scope_sha256'] == sha(raw['scope.json'])
    assert run['exit_status'] == 0 and run['stderr'] == ''
    assert json.loads(run['stdout'])['status'] == json.loads(raw['results.json' if prefix == 'preflight' else 'independent-results.json'])['status']
    imports = set()
    for node in ast.walk(ast.parse(raw[prefix + '.py'].decode())):
        if isinstance(node, ast.Import): imports.update(x.name for x in node.names)
        if isinstance(node, ast.ImportFrom): imports.add(node.module)
    assert imports <= {'pathlib', 'hashlib', 'json', 'ast', 'numpy', 'fractions'}
for x in json.loads(raw['source-pins.json']).values():
    source = Path(x.get('local_copy', x.get('path')))
    assert sha(source.read_bytes()) == x['sha256'], source
preview = json.loads(raw['task-input-preview.json'])
assert preview['status'] == 'PASS_EXISTING_METADATA_PIN_PREVIEW_ONLY'
assert not preview['server_availability_checked'] and not preview['checkpoint_opened']
for x in preview['files'].values():
    assert sha(Path(x['path']).read_bytes()) == x['sha256']
next_action = json.loads(raw['next-direction.json'])
assert not next_action['selected_plugin'] and not next_action['fixed_inputs']['availability_freshly_verified_this_turn']
assert next_action['fixed_inputs']['checkpoint_sha256'] == '8ec179f4605a901ea1aeeeaefaacc20d78a0b202f236a2d41c1a37b06220cf3c'
assert v['status'] == 'PASS_READOUT_ALGEBRA_AND_HISTORY_PRECHECK'
assert not any(v[k] for k in ('task_method_selected', 'new_theory', 'independent_task_confirmation', 'external_research_review', 'task_information_protocol_executed'))
assert all(v[k] == 0 for k in scope['resource'])
assert v['goal'] == next_action['goal'] == 'ACTIVE'
assert v['newL3'] == v['newL4T'] == 'NOT_ACCEPTED'
out = {'status': 'PASS_READOUT_PREFLIGHT_EVIDENCE_PACKAGE', 'literal_two_execution_exit0_verified': True,
       'preregistered_scope_and_code_hashes_verified': True, 'source_and_existing_input_receipts_reopened': True,
       'prospective_task_diagnostic_executed': False,
       'public_evidence_sha256': {n: sha(x) for n, x in raw.items()},
       'goal': 'ACTIVE', 'newL3': 'NOT_ACCEPTED', 'newL4T': 'NOT_ACCEPTED'}
(r / 'package-verification.json').write_bytes((json.dumps(out, ensure_ascii=False, indent=2) + '\n').encode())
print(json.dumps({'status': out['status'], 'evidence_files': len(raw), 'task_diagnostic_executed': False, 'goal': 'ACTIVE'}))
