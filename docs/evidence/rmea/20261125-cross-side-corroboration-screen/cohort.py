"""Deterministic TRAIN-only prospective screen cohort; execute on the data host."""
import hashlib
import json
from pathlib import Path
import numpy as np


def build(cache):
    metadata = json.loads((cache/'metadata.json').read_bytes())['train']
    labels = np.load(cache/'train-targets.npy', allow_pickle=False)
    digest = lambda x: hashlib.sha256(x).hexdigest()
    source = cache.parent.parent/'recovery-inputs/train.jsonl'
    source_bytes = source.read_bytes()
    assert digest(source_bytes) == 'f1051d574a44df3ecbd036c1de4fbbe0a372c768f6ec990330e2cea4a657269f'
    original = [json.loads(x) for x in source_bytes.splitlines() if x.strip()]
    lookup = {digest(str(row['metadata']['id']).encode()): row for row in original}
    assert len(lookup) == len(original) == len(metadata) == len(labels) == 58465
    assert len({row['id_hash'] for row in metadata}) == len(metadata)
    records = [lookup[row['id_hash']] for row in metadata]
    label_names = json.loads((source.parent/'labels.json').read_bytes())
    assert all(row['label'] == label_names[int(y)] for row, y in zip(records, labels))
    assert all(len(row['tokens']) == m['words'] for row, m in zip(records, metadata))
    groups = [digest(json.dumps(row['tokens'], ensure_ascii=False, separators=(',', ':')).encode())
              for row in records]
    members = {}
    for i, group in enumerate(groups):
        members.setdefault(group, []).append(i)
    order = sorted(members, key=lambda g: digest(('staged-cohort-20261125-v1:'+g).encode()))
    assignment = {g: int(digest(('staged-side-v1:'+g).encode()), 16) % 2 for g in order}
    # Fix rare-label coverage by group only, before any model or outcome.
    group_labels = {g: set(map(int, labels[members[g]])) for g in order}
    for _ in range(80):
        missing = [(side, y) for side in (0, 1) for y in range(40)
                   if not any(assignment[g] == side and y in group_labels[g] for g in order)]
        if not missing:
            break
        side, y = missing[0]
        candidates = [g for g in order if assignment[g] != side and y in group_labels[g]
                      and all(sum(assignment[h] != side and z in group_labels[h] for h in order) > 1
                              for z in group_labels[g])]
        assert candidates, ('Insufficient independent rare-label groups', side, y)
        assignment[candidates[0]] = side
    assert not missing

    def sample(side, n):
        buckets = {y: [i for g in order if assignment[g] == side for i in members[g] if labels[i] == y]
                   for y in range(40)}
        counts = np.array([len(buckets[y]) for y in range(40)])
        expected = n*counts/counts.sum()
        quota = np.maximum(1, np.floor(expected).astype(int))
        while quota.sum() < n:
            available = np.flatnonzero(quota < counts)
            j = max(available, key=lambda j: (expected[j]-quota[j], -j))
            quota[j] += 1
        while quota.sum() > n:
            j = max(np.flatnonzero(quota > 1), key=lambda j: (quota[j]-expected[j], -j))
            quota[j] -= 1
        indices = []
        for y in range(40):
            bucket = sorted(buckets[y], key=lambda i: digest(('staged-row-v1:'+metadata[i]['id_hash']).encode()))
            indices.extend(bucket[:quota[y]])
        assert len(indices) == n
        return sorted(indices)

    fit, assessment = sample(0, 8192), sample(1, 4096)
    assert not set(fit) & set(assessment)
    assert not {groups[i] for i in fit} & {groups[i] for i in assessment}
    supports = {k: np.bincount(labels[idx], minlength=40).tolist()
                for k, idx in (('fit', fit), ('assessment', assessment))}
    assert all(min(v) > 0 for v in supports.values())
    return dict(schema='q-rmea.screen-cohort.v1', fit=fit, assessment=assessment,
                selected_groups={str(i): groups[i] for i in fit+assessment}, supports=supports,
                source_rows=58465, target_sha256=digest((cache/'train-targets.npy').read_bytes()),
                metadata_sha256=digest((cache/'metadata.json').read_bytes()),
                source_sha256=digest(source_bytes),
                rule='Fixed group hashing and label-stratified natural proportions, minimum one/class; no model/outcome selection',
                group_overlap=0, labels_used_for_stratification_only=True,
                valid_test_evaluated=False)
