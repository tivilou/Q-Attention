"""Verify retained evidence, literal execution, and a report-only allowlist."""
from pathlib import Path
import hashlib
import json
import ast

r = Path(__file__).resolve().parent
names = ('audit.py', 'execution.json', 'source-pins.json', 'interface-ledger.json',
         'verification.json', 'analysis.md', 'next-direction.json', 'lookup-observation.json')


def sha(b):
    return hashlib.sha256(b).hexdigest()


raw = {n: (r / n).read_bytes() for n in names}
v = json.loads(raw['verification.json'])
ledger = json.loads(raw['interface-ledger.json'])
pins = json.loads(raw['source-pins.json'])
run = json.loads(raw['execution.json'])
next_action = json.loads(raw['next-direction.json'])
assert v['status'] == 'PASS_MECHANICAL_INTERFACE_AUDIT'
assert ledger['decision'] == next_action['source_decision'] == 'NONE_CURRENT_GRAPH_INPUT_QUALIFICATION'
assert run['exit_status'] == 0 and run['stderr'] == ''
assert run['code_sha256'] == sha(raw['audit.py'])
literal = json.loads(run['stdout'])
assert literal['status'] == v['status'] and literal['decision'] == ledger['decision']
assert literal['existing_train_records'] == 58465 and literal['new_task_rows'] == 0
assert not literal['graph_qualified'] and literal['producer_binding'] == 'UNVERIFIED'
assert v['goal'] == next_action['goal'] == 'ACTIVE'
assert v['newL3'] == v['newL4T'] == 'NOT_ACCEPTED'
assert not any(v[k] for k in ('new_method_selected', 'new_theory', 'independent_task_confirmation'))
zero_fields = ('task_rows', 'feature_checkpoint_reads', 'head_forwards', 'synthetic_circuit_executions',
               'new_updates', 'new_gpu_allocations', 'server_writes', 'COS', 'model_API_calls', 'new_original_acquisitions')
assert all(v[k] == 0 for k in zero_fields)
assert not next_action['selected_plugin']
for row in pins['files'].values():
    assert sha(Path(row['path']).read_bytes()) == row['sha256'], row['path']
tree = ast.parse(raw['audit.py'].decode())
imports = set()
for n in ast.walk(tree):
    if isinstance(n, ast.Import):
        imports.update(a.name for a in n.names)
    if isinstance(n, ast.ImportFrom):
        imports.add(n.module)
assert imports <= {'pathlib', 'datetime', 'ast', 'hashlib', 'json'}
assert ledger['canonical_train_summary_reused']['new_task_rows_read'] == 0
assert ledger['historical_converter']['producer_binding_to_current_canonical_data'] == 'UNVERIFIED'
assert not ledger['historical_converter']['synthetic_canary_is_full_data_producer_receipt']
assert not ledger['qualified_original_word_graph']
assert len(raw['analysis.md']) < 14000
out = {
    'status': 'PASS_INTERFACE_EVIDENCE_PACKAGE',
    'literal_command_stdout_stderr_exit_verified': True,
    'all_input_source_hashes_reopened': True,
    'audit_imports_static_only': sorted(imports),
    'public_evidence_sha256': {n: sha(b) for n, b in raw.items()},
    'scientific_scope': 'Graph qualification NONE, not a task utility or theory result',
    'goal': 'ACTIVE', 'newL3': 'NOT_ACCEPTED', 'newL4T': 'NOT_ACCEPTED'
}
(r / 'package-verification.json').write_bytes((json.dumps(out, ensure_ascii=False, indent=2) + '\n').encode())
print(json.dumps({'status': out['status'], 'evidence_files': len(raw), 'literal_exit_status': run['exit_status'], 'goal': 'ACTIVE'}))
