"""Independent verification of fixed selection, blinding and all64 judgments.

Private inputs and response reasons stay on server. Only nontext records leave.
"""
from pathlib import Path
from collections import Counter
import hashlib, json, numpy as np

root = Path('/hy-tmp/Q-Attention')
r = root / 'control/gen98-content32-20261125-cpu'
out = r / 'output'
base = root / 'inputs/native-baseline'
source = base / 'recovery-inputs/train.jsonl'
cache = base / 'discovery/cache'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(source) == 'f1051d574a44df3ecbd036c1de4fbbe0a372c768f6ec990330e2cea4a657269f'
assert sha(cache / 'metadata.json') == '9e6eac9ad582a4ef4e1a8a12190bfef3b8f0a18c5efbd9072f91f31563656c0e'
diag = root / 'control/distinct-token-evidence-20261125/output/diagnostic.private.npz'
assert sha(diag) == '3677d55f2e6d3c64df6c9789c76f7113b66a79ee1efbee7531bc9bf2bf5d1307'
metadata = json.loads((cache / 'metadata.json').read_bytes())['train']
labels = json.loads((base / 'recovery-inputs/labels.json').read_bytes())
assert sha(base / 'recovery-inputs/labels.json') == '6e822509a07c5c865130ebe315122a9d83e97b1b11496b9e86c6b2a600f8a429'
with np.load(diag, allow_pickle=False) as d:
    indices = d['assessment_indices'].copy()
    y = d['labels'][indices]
    q, dephased, g = [d[m + '_logits'].argmax(1) for m in (
        'quantum_distinct', 'dephased_distinct', 'gaussian_distinct')]
mask = {
    'q_only_positive': (y > 0) & (q == y) & (dephased != y) & (g != y),
    'q_only_null': (y == 0) & (q == y) & (dephased != y) & (g != y),
    'peer_only_positive': (y > 0) & (q != y) & (dephased == y) & (g == y),
    'peer_only_null': (y == 0) & (q != y) & (dephased == y) & (g == y)}
packet = json.loads((out / 'blind-packet.private.json').read_bytes())
truth = json.loads((out / 'truth.private.json').read_bytes())
selected = {}
for name, select in mask.items():
    pool = np.flatnonzero(select)
    assert len(pool) == {'q_only_positive':37, 'q_only_null':43,
                         'peer_only_positive':42, 'peer_only_null':103}[name]
    order = sorted(pool, key=lambda k: hashlib.sha256(
        ('gen98-content32-v1:' + metadata[int(indices[k])]['id_hash']).encode()).digest())
    for k in order[:8]:
        index = int(indices[k]); identity = metadata[index]['id_hash']
        case = hashlib.sha256(('blind-case:' + identity).encode()).hexdigest()[:12]
        flip = int(hashlib.sha256(('gen98-candidate-order-v1:' + identity).encode()).hexdigest()[-1],16) % 2
        pair = [int(q[k]), int(g[k])][::(-1 if flip else 1)]
        assert truth[case] == dict(group=name, index=index, id_hash=identity, gold=labels[int(y[k])],
            quantum_label=labels[int(q[k])], gaussian_label=labels[int(g[k])],
            gold_option='A' if pair[0] == y[k] else 'B')
        selected[case] = dict(identity=identity, index=index, pair=pair)
assert set(selected) == set(truth) and len(selected) == 32
ordered = sorted(selected, key=lambda case: hashlib.sha256(
    ('gen98-packet-order-v1:' + selected[case]['identity']).encode()).digest())
assert [x['case_id'] for x in packet['cases']] == ordered
source_rows = [json.loads(x) for x in source.read_bytes().splitlines() if x.strip()]
lookup = {hashlib.sha256(str(row['metadata']['id']).encode()).hexdigest(): row for row in source_rows}
assert len(lookup) == len(source_rows) == len(metadata) == 58465
for row in packet['cases']:
    assert set(row) == {'case_id', 'tokens', 'subject_span', 'object_span', 'subject_text', 'object_text', 'options'}
    k = selected[row['case_id']]; original = lookup[k['identity']]
    assert row['tokens'] == original['tokens']
    assert row['subject_span'] == original['subject'] and row['object_span'] == original['object']
    assert row['options'] == dict(A=labels[k['pair'][0]], B=labels[k['pair'][1]])
    assert row['subject_text'] == ' '.join(original['tokens'][slice(*original['subject'])])
    assert row['object_text'] == ' '.join(original['tokens'][slice(*original['object'])])
    assert original['label'] == truth[row['case_id']]['gold']

# Reconstruct the exact serialized reader input; original selected packet stays immutable.
packet['benchmark_relation_names'] = labels
packet['instructions'][1] = 'no_relation means the sentence does not establish a benchmark positive relation for this exact marked pair. If a different positive relation is established but neither option names it, choose neither. A relation of a different entity pair is insufficient.'
payload = json.dumps(packet, ensure_ascii=False).encode()
payload_hash = hashlib.sha256(payload).hexdigest()
sent = out / 'sent-packet.private.json'
if sent.exists():
    assert sent.read_bytes() == payload
else:
    sent.write_bytes(payload)
readers = [json.loads((out / f'reader{i}.private.json').read_bytes()) for i in (1,2)]
summary = json.loads((out / 'blind-summary.json').read_bytes())
responses = []
for i, reader in enumerate(readers, 1):
    assert reader['reader'] == i and reader['parse_error'] is None
    assert reader['model'] == reader['raw']['model'] == 'gpt-6.1-sol'
    assert reader['effort'] == reader['raw']['reasoning']['effort'] == 'medium'
    assert reader['packet_sha256'] == payload_hash
    claim = json.loads((out / f'reader{i}.dispatch.json').read_bytes())
    assert claim['packet_sha256'] == payload_hash and claim['max_inference_attempts'] == 1
    assert len(reader['answer']['rows']) == 32
    rows = {x['case_id']:x for x in reader['answer']['rows']}
    assert set(rows) == set(truth)
    responses.append(rows)
assert summary['packet_sha256'] == payload_hash
records = {x['case_id']: x for x in summary['case_records']}
groups = {}
for group in mask:
    cases = [case for case,t in truth.items() if t['group'] == group]
    counts = Counter(); by_reader = []
    for case in cases:
        t = truth[case]; a,b = [x[case] for x in responses]
        choices = [a['choice'], b['choice']]
        counts['both_support_gold'] += int(choices == [t['gold_option']]*2)
        counts['both_choose_other'] += int(choices[0] == choices[1] and choices[0] in ('A','B') and choices[0] != t['gold_option'])
        counts['choice_disagreement'] += int(choices[0] != choices[1])
        counts['any_uncertain_or_nonbinary'] += int(any(x not in ('A','B') for x in choices))
        assert records[case]['reader_choices'] == choices
        assert records[case]['both_support_gold'] == (choices == [t['gold_option']]*2)
    for key,n in counts.items():
        assert summary['groups'][group][key] == n
    for key in ('category', 'evidence_structure', 'distractor_relation'):
        agreed = Counter(responses[0][case][key] for case in cases
                         if responses[0][case][key] == responses[1][case][key])
        assert summary['groups'][group]['agreed'][key] == dict(agreed)
        for reader_index in range(2):
            counted = Counter(responses[reader_index][case][key] for case in cases)
            assert summary['groups'][group]['per_reader'][reader_index][key] == dict(counted)
        for case in cases:
            field = {'category':'reader_categories', 'evidence_structure':'reader_evidence_structure',
                     'distractor_relation':'reader_distractor_relation'}[key]
            assert records[case][field] == [responses[i][case][key] for i in range(2)]
    assert len(cases) == 8
    groups[group] = dict(counts)
legacy = set(json.loads((r / 'legacy-case-ids.json').read_bytes()))
assert len(set(truth).intersection(legacy)) == 0
result = dict(status='PASS', fixed_strata_and_salts_verified=True,
    selected_cases=32, total_valid_reader_judgments=64, exact_original_input_join=True,
    label_model_group_correctness_absent_from_packet=True,
    packet_sha256=sha(out / 'blind-packet.private.json'), sent_packet_sha256=payload_hash,
    truth_sha256=sha(out / 'truth.private.json'), overlap_with_legacy_cases=0,
    all_choices_and_group_counts_recomputed=True,
    all_categories_structures_distractors_recomputed=True, groups=groups,
    model='gpt-6.1-sol', reasoning_effort='medium', contexts=2, independent_humans=False,
    inference_redispatch=False, no_model_head_forwards=True, training_updates=0,
    new_valid=0, new_test=0, cos_writes=0, raw_text_exported=False,
    scope='Outcome-selected exploratory32; no population noise rate, causal mechanism, quantum utility or new theory certified.')
print(json.dumps(result))
