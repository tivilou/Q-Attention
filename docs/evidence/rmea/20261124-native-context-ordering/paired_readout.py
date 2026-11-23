"""R0 aggregation of existing TRAIN logits; no model loading or fitting."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import os
import resource
import sys
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

started = time.monotonic()
packet = json.load(sys.stdin)
cfg, allocation = packet['design'], packet['allocation']
permit_path = Path(allocation['permit_path'])
assert hashlib.sha256(permit_path.read_bytes()).hexdigest() == allocation['permit_sha256']
permit = json.loads(permit_path.read_bytes())
for field in ('allocation_id', 'generation', 'project_id', 'host_id', 'run_id', 'runner_attempt_id'):
    assert permit[field] == allocation[field]
resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
os.nice(10)
root = Path('/hy-tmp/Q-Attention')
for name, digest in cfg['input_hashes'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
labels = json.loads((root / cfg['labels']).read_bytes())
assert labels[0] == 'no_relation' and len(labels) == 40
metadata = json.loads((root / cfg['metadata']).read_bytes())['train']
index = {row['id_hash']: i for i, row in enumerate(metadata)}
assert len(index) == len(metadata) == cfg['rows']
targets = np.load(root / cfg['targets'], allow_pickle=False, mmap_mode='r')
logits = {mode: np.load(root / name, allow_pickle=False, mmap_mode='r') for mode, name in cfg['logits'].items()}
for z in logits.values():
    assert z.shape == (cfg['rows'], 40) and np.isfinite(z).all()
groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
seen = set()
with (root / cfg['source']).open('rb') as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        words, subject, obj, meta = tuple(row['tokens']), row['subject'], row['object'], row['metadata']
        key_hash = hashlib.sha256(str(meta['id']).encode()).hexdigest()
        assert key_hash in index and key_hash not in seen
        seen.add(key_hash)
        idx = index[key_hash]
        assert labels[int(targets[idx])] == row['label'] and len(words) == metadata[idx]['words']
        key = (words[subject[0]:subject[1]], words[obj[0]:obj[1]], meta['subj_type'], meta['obj_type'])
        sentence = hashlib.sha256(json.dumps(words, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        groups[key][row['label']][sentence].append(idx)
assert len(seen) == cfg['rows']
strata = []
contrasts = 0
for group in groups.values():
    null = group.get('no_relation', {})
    if not null:
        continue
    for relation, positive in group.items():
        if relation == 'no_relation':
            continue
        positive_keys, null_keys = list(positive), list(null)
        eligible = np.asarray(positive_keys, dtype=object)[:, None] != np.asarray(null_keys, dtype=object)[None, :]
        count = int(eligible.sum())
        if not count:
            continue
        relation_index = labels.index(relation)
        concordance = {}
        for mode, z in logits.items():
            def means(side, keys):
                return np.asarray([np.mean(z[side[key], relation_index].astype(np.float64) - z[side[key], 0].astype(np.float64)) for key in keys])
            difference = means(positive, positive_keys)[:, None] - means(null, null_keys)[None, :]
            wins, ties = int(((difference > 0) & eligible).sum()), int(((difference == 0) & eligible).sum())
            concordance[mode] = (wins + .5 * ties) / count
        strata.append((relation, count, concordance))
        contrasts += count
assert len(strata) == cfg['expected_strata'] and contrasts == cfg['expected_contrasts']
per_class = {}
for relation in sorted({row[0] for row in strata}):
    rows = [x for x in strata if x[0] == relation]
    per_class[relation] = {mode: float(np.mean([x[2][mode] for x in rows])) for mode in logits}
aggregate = {mode: float(np.mean([x[2][mode] for x in strata])) for mode in logits}
macro_class = {mode: float(np.mean([row[mode] for row in per_class.values()])) for mode in logits}
contrasts_to_q = {mode: aggregate['quantum'] - value for mode, value in aggregate.items() if mode != 'quantum'}
print(json.dumps(dict(
    status='COMPLETE', profile=cfg['profile'], rows=cfg['rows'], eligible_strata=len(strata),
    distinct_context_contrasts=contrasts, relation_coverage=len(per_class),
    concordance_equal_stratum=aggregate, secondary_equal_class_concordance=macro_class,
    q_minus_other_equal_stratum=contrasts_to_q,
    q_stratum_comparisons={mode: dict(positive=sum(x[2]['quantum'] > x[2][mode] for x in strata),
        negative=sum(x[2]['quantum'] < x[2][mode] for x in strata), equal=sum(x[2]['quantum'] == x[2][mode] for x in strata)) for mode in contrasts_to_q},
    per_class=per_class, seconds=time.monotonic() - started,
    new_model_forward_calls=0, optimizer_updates=0, gpu_operations=0, test_rows_read=0,
    same_checkpoint_phase_blind_replay='NOT_AVAILABLE_IN_EXISTING_LOGITS_NOT_RUN',
    formal_training_started=False, cohort=cfg['cohort'], limitations=cfg['limitations'],
)))
