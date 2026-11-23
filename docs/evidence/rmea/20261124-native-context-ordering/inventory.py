"""Read-only, logit-blind TRAIN support inventory; emits aggregate counts only."""
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import os
import resource
import sys
import time

v = json.load(sys.stdin)
cfg, allocation = v['design'], v['allocation']
started = time.monotonic()
permit_file = Path(allocation['permit_path'])
assert hashlib.sha256(permit_file.read_bytes()).hexdigest() == allocation['permit_sha256']
permit = json.loads(permit_file.read_bytes())
for name in ('allocation_id', 'generation', 'project_id', 'host_id', 'run_id', 'runner_attempt_id'):
    assert permit[name] == allocation[name]
resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
os.nice(10)
path = Path('/hy-tmp/Q-Attention') / cfg['input']
digest = hashlib.sha256()
groups = defaultdict(lambda: defaultdict(set))
ids = set()
rows = 0
with path.open('rb') as handle:
    for line in handle:
        digest.update(line)
        if not line.strip():
            continue
        row = json.loads(line)
        words, subject, obj, meta = tuple(row['tokens']), row['subject'], row['object'], row['metadata']
        ident = str(meta['id'])
        assert ident not in ids
        ids.add(ident)
        assert 0 <= subject[0] < subject[1] <= len(words)
        assert 0 <= obj[0] < obj[1] <= len(words)
        key = (words[subject[0]:subject[1]], words[obj[0]:obj[1]], meta['subj_type'], meta['obj_type'])
        sentence = hashlib.sha256(json.dumps(words, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        groups[key][row['label']].add(sentence)
        rows += 1
assert rows == cfg['rows'] and digest.hexdigest() == cfg['input_sha256']
strata = []
contrasts = eligible_keys = ambiguous_self = 0
null_contexts, positive_contexts = set(), set()
per_relation = Counter()
for key, labels in groups.items():
    null = labels.get('no_relation', set())
    any_eligible = False
    if not null:
        continue
    for relation, positive in labels.items():
        if relation == 'no_relation':
            continue
        self_edges = len(positive & null)
        edges = len(positive) * len(null) - self_edges
        ambiguous_self += self_edges
        if edges <= 0:
            continue
        strata.append((len(positive), len(null), edges))
        contrasts += edges
        per_relation[relation] += 1
        any_eligible = True
        positive_contexts.update(positive)
        null_contexts.update(null)
    eligible_keys += int(any_eligible)
print(json.dumps(dict(
    status='COMPLETE', rows=rows, input_sha256=digest.hexdigest(),
    unique_ordered_pair_type_groups=len(groups), eligible_pair_type_groups=eligible_keys,
    eligible_pair_relation_strata=len(strata), distinct_context_contrasts=contrasts,
    unique_positive_sentences=len(positive_contexts), unique_null_sentences=len(null_contexts),
    relation_coverage=len(per_relation), strata_per_relation=dict(sorted(per_relation.items())),
    ambiguous_same_sentence_edges_excluded=ambiguous_self,
    strata_support_histogram=dict(Counter('p' + str(min(p, 4)) + '_n' + str(min(n, 4)) for p, n, _ in strata)),
    max_stratum_contrasts=max((x[2] for x in strata), default=0), seconds=time.monotonic() - started,
    model_or_logit_reads=0, gpu_operations=0, formal_training_started=False,
    allocation_id=allocation['allocation_id'], generation=allocation['generation'], scope=cfg['scope'],
)))
