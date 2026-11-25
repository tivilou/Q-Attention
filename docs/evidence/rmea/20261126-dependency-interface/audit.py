"""Read existing summaries and Python AST only; never load task rows or models."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json

ROOT = Path('E:/Claude.Code/Paper.Writing/Q-Attention')
OUT = Path(__file__).resolve().parent
PREVIOUS = Path('D:/CodexHome/task-artifacts/qattn-trained-kernel-source-20261126/next-direction.json')
INPUTS = {
    'existing_input_inspection': ROOT / '.codex-artifacts/qattn-rmea-readiness-20261026/input-audit.json',
    'canonical_reference': ROOT / '.codex-artifacts/reference-results-20261017/repo/reports/retacred_reference_suite/20261012T112015Z/luke_core_seed13/data_manifest.json',
    'historical_converter': ROOT / '.codex-artifacts/q-srm-implementation/source/src/q_attention/tasks/relation_formats.py',
    'historical_record_io': ROOT / '.codex-artifacts/q-srm-implementation/source/src/q_attention/tasks/relation.py',
    'historical_synthetic_canary': ROOT / '.codex-artifacts/q-srm-implementation/canary-manifest.json',
    'pinned_inspection_code': Path('D:/CodexHome/task-artifacts/qattn-role-exchange-audit-20261125/source/audit_rmea_inputs.py'),
    'pinned_extractor': Path('D:/CodexHome/task-artifacts/qattn-role-exchange-audit-20261125/source/rmea_run.py'),
    'pinned_native_head': Path('D:/CodexHome/task-artifacts/qattn-relation-frame-20261125/source/baselines/retacred/rmea.py'),
    'existing_native_source_receipt': Path('D:/CodexHome/task-artifacts/qattn-role-exchange-audit-20261125/source-pins.json'),
    'frozen_previous_action': PREVIOUS,
}
EXPECTED = {
    'pinned_inspection_code': 'cac564d055ea76520c5dbcc0e16e7d551a9b3243a4dba835e42ab7e4d208c14f',
    'pinned_extractor': '769173b9ab17f76229b0db79b4a7625ac1574427d2d40491be7d5b19d4d0ff25',
    'pinned_native_head': '9f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0',
    'historical_converter': 'ddd656810149c9a3e4b591f72a23efa91ed16a2e3406aa6f3e372b423260bab9',
    'historical_record_io': '145296d7086c4fa93f65ca1e2b8b71e74e1f1aeb9a669b9d63ec29b7b7c9b5b3',
}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def write(name, value):
    (OUT / name).write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def function(tree, name):
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(matches) == 1, (name, len(matches))
    return matches[0]


def location(n):
    return {'first_line': n.lineno, 'last_line': n.end_lineno}


def named_call(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name]


raw = {k: p.read_bytes() for k, p in INPUTS.items()}
pins = {k: {'path': str(INPUTS[k]), 'sha256': sha(b), 'bytes': len(b)} for k, b in raw.items()}
for k, h in EXPECTED.items():
    assert pins[k]['sha256'] == h, (k, pins[k]['sha256'])
previous = json.loads(raw['frozen_previous_action'])
assert previous['selected_plugin'] is None and previous['resource']['new_task_rows'] == 0
assert 'dependency' in previous['single_next_action']
inspection = json.loads(raw['existing_input_inspection'])
reference = json.loads(raw['canonical_reference'])
train = inspection['splits']['train']
assert inspection['dataset_identity_matches_reference']
assert train['records'] == reference['train']['records'] == 58465
assert train['sha256'] == reference['train']['source_sha256'] == 'f1051d574a44df3ecbd036c1de4fbbe0a372c768f6ec990330e2cea4a657269f'
assert train['matches_audited_reference_count'] and train['matches_audited_reference_sha256']
assert train['metadata_fields'] == ['docid', 'id', 'obj_type', 'source_format', 'subj_type']
assert reference['train']['source_sha256'] != reference['train']['output_sha256']

trees = {k: ast.parse(raw[k].decode('utf-8-sig')) for k in EXPECTED}
convert = function(trees['historical_converter'], '_tacred_object_to_record')
compact = function(trees['historical_converter'], '_compact_metadata')
calls = named_call(convert, '_compact_metadata')
assert len(calls) == 1
keep = next(ast.literal_eval(k.value) for k in calls[0].keywords if k.arg == 'keep')
assert keep == ('id', 'docid', 'sent_id', 'subj_type', 'obj_type', 'stanford_ner', 'source')
scalar_check = [c for c in named_call(compact, 'isinstance') if isinstance(c.args[0], ast.Name) and c.args[0].id == 'value']
assert len(scalar_check) == 1
types = [n.id for n in scalar_check[0].args[1].elts]
assert types == ['str', 'int', 'float', 'bool']
loader = function(trees['historical_record_io'], 'load_relation_jsonl')
writer = function(trees['historical_record_io'], 'write_relation_jsonl')
record_calls = named_call(loader, 'RelationRecord')
assert len(record_calls) == 1
loader_fields = [k.arg for k in record_calls[0].keywords]
assert loader_fields == ['tokens', 'subject', 'object', 'label', 'metadata']
write_fields = set()
for n in ast.walk(writer):
    if isinstance(n, ast.Dict):
        write_fields.update(k.value for k in n.keys if isinstance(k, ast.Constant))
    if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == 'item' and isinstance(n.slice, ast.Constant):
        write_fields.add(n.slice.value)
assert write_fields == set(loader_fields)
inspect_fn = function(trees['pinned_inspection_code'], 'inspect_split')
inspect_text = ast.get_source_segment(raw['pinned_inspection_code'].decode('utf-8-sig'), inspect_fn)
assert 'metadata_fields.update(meta.keys())' in inspect_text
extract = function(trees['pinned_extractor'], 'extract')
extract_text = ast.get_source_segment(raw['pinned_extractor'].decode('utf-8-sig'), extract)
assert 'truncation=False' in extract_text and 'No truncation permitted' in extract_text
assert 'for word in range(n)' in extract_text and 'Original word missing' in extract_text
assert 'vectors.append(hidden[i,pieces].mean(0))' in extract_text
assert all(s in extract_text for s in ('-h.npy', '-roles.npy', '-offsets.npy', '-targets.npy'))
assert not any(s in extract_text.lower() for s in ('stanford_head', 'stanford_deprel', 'dependency', 'adjacency'))
batch = function(trees['pinned_extractor'], 'batch')
forward = function(trees['pinned_native_head'], 'forward')
forward_args = [n.arg for n in forward.args.args]
assert forward_args == ['self', 'h', 'roles', 'mask', 'details']
canary = json.loads(raw['historical_synthetic_canary'])
assert [canary['data'][s]['source_count'] for s in ('train', 'valid', 'test')] == [12, 6, 6]
assert canary['data']['train']['source_sha256'] != train['sha256']

ledger = {
    'schema': 'qattn.dependency-interface-qualification.v1',
    'decision': 'NONE_CURRENT_GRAPH_INPUT_QUALIFICATION',
    'selected_plugin': None,
    'canonical_train_summary_reused': {
        'records': train['records'], 'sha256': train['sha256'], 'metadata_fields': train['metadata_fields'],
        'reference_source_sha256': reference['train']['source_sha256'],
        'different_baseline_converted_output_sha256': reference['train']['output_sha256'],
        'new_task_rows_read': 0,
        'scope_limit': 'Inspection counts keys inside metadata, not all top-level fields. No inference of raw-upstream dependency absence.'
    },
    'historical_converter': {
        'producer_binding_to_current_canonical_data': 'UNVERIFIED',
        'tacred_conversion': location(convert), 'keep_fields': keep,
        'compact_metadata': location(compact), 'accepted_value_types': types + ['None'],
        'array_fields_retained_by_this_function': False,
        'dependency_head_or_deprel_in_whitelist': False,
        'canonical_loader': location(loader), 'loader_constructor_fields': loader_fields,
        'canonical_writer': location(writer), 'writer_fields': sorted(write_fields),
        'synthetic_canary_counts': [12, 6, 6],
        'synthetic_canary_is_full_data_producer_receipt': False,
        'historical_git_commit': canary['git_revision'],
        'scope_limit': 'Local source SHA is verified; neither this canary nor current observations prove the exact producer of f1051d57.'
    },
    'pinned_current_interfaces': {
        'inspection': location(inspect_fn), 'metadata_key_union_only': True,
        'extract': location(extract), 'batch': location(batch), 'native_forward': location(forward),
        'native_forward_args': forward_args,
        'full_original_word_order_preserved_by_mean_wordpiece_mapping': True,
        'truncation_forbidden': True,
        'model_input_graph_tensor': False,
        'scope_limit': 'No raw data or feature arrays opened; code interface is not an upstream-field census.'
    },
    'qualified_original_word_graph': False,
    'novel_graph_interaction_selected': False,
    'kill_reason': 'Existing evidence does not establish an already pinned dependency graph and mapping. Frozen input/provenance gate fails; no parser, guessed indices, new census or fit.',
    'does_not_establish': ['Raw upstream has no graph', 'All graph mechanisms fail', 'Quantum/classical equivalence or impossibility', 'New L3 or L4-T'],
}
write('source-pins.json', {'files': pins, 'observed_at': datetime.now(timezone.utc).isoformat(), 'historical_converter_binding': 'UNVERIFIED'})
write('interface-ledger.json', ledger)
verification = {
    'status': 'PASS_MECHANICAL_INTERFACE_AUDIT', 'scientific_candidate_gate': 'NONE_INPUT_PROVENANCE_NOT_QUALIFIED',
    'checks': ['Exact three current-code pins', 'Two observed historical source hashes', 'Existing TRAIN summary/reference source identity',
               'Metadata-key-only inspection limitation', 'Converter whitelist and scalar AST', 'Loader/writer AST field sets',
               'Synthetic-canary separation from full-data producer', 'Original-word mapping and no-truncation code', 'Native forward signature'],
    'qualified_graph': False, 'new_method_selected': False, 'new_theory': False, 'independent_task_confirmation': False,
    'task_rows': 0, 'feature_checkpoint_reads': 0, 'head_forwards': 0, 'synthetic_circuit_executions': 0,
    'new_updates': 0, 'new_gpu_allocations': 0, 'server_writes': 0, 'COS': 0, 'model_API_calls': 0,
    'new_original_acquisitions': 0, 'goal': 'ACTIVE', 'newL3': 'NOT_ACCEPTED', 'newL4T': 'NOT_ACCEPTED',
    'scope': 'Static AST plus retained summary identity; not external research review or independently confirmed task behavior.',
}
write('verification.json', verification)
print(json.dumps({'status': verification['status'], 'decision': ledger['decision'], 'existing_train_records': train['records'],
                  'new_task_rows': 0, 'graph_qualified': False, 'producer_binding': 'UNVERIFIED', 'goal': 'ACTIVE'}, ensure_ascii=False))
