"""TRAIN-only metadata census; no model, feature arrays, or held-out access."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time


def save(path, value):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def main():
    ap = argparse.ArgumentParser()
    for name in ('root', 'design', 'output', 'progress'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--attempt', required=True)
    a = ap.parse_args()
    started = time.monotonic()
    cfg = json.loads(a.design.read_bytes())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == cfg['code_sha256']
    source = a.root / cfg['source']
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == cfg['source_sha256']
    counts = defaultdict(Counter)
    sentences = defaultdict(lambda: defaultdict(set))
    n = 0
    for line in raw.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        relation = row['label']
        pair = (row['metadata']['subj_type'], row['metadata']['obj_type'])
        sid = hashlib.sha256(json.dumps(row['tokens'], ensure_ascii=False,
                                       separators=(',', ':')).encode()).hexdigest()
        counts[relation][pair] += 1
        sentences[relation][pair].add(sid)
        n += 1
    assert n == 58465 and len(counts) == 40 and 'no_relation' in counts
    per_class = []
    eligible = []
    for relation in sorted(counts):
        pairs = [dict(subject_type=pair[0], object_type=pair[1], rows=count,
                      distinct_sentences=len(sentences[relation][pair]))
                 for pair, count in sorted(counts[relation].items(),
                                           key=lambda item: (-item[1], item[0]))]
        supported = [p for p in pairs if p['distinct_sentences'] >= cfg['minimum_sentences_per_pair']]
        total = sum(counts[relation].values())
        per_class.append(dict(relation=relation, rows=total, type_pairs=pairs,
                              supported_type_pairs=len(supported),
                              majority_pair_fraction=pairs[0]['rows']/total))
        if relation != 'no_relation' and len(supported) >= 2:
            eligible.append(relation)
    assert sum(p['rows'] for p in per_class) == n
    passed = len(eligible) >= cfg['minimum_positive_relations']
    result = dict(profile=cfg['profile'], source_sha256=cfg['source_sha256'], rows=n,
                  classes=len(per_class), per_class=per_class,
                  eligible_positive_relations=eligible,
                  minimum_sentences_per_pair=cfg['minimum_sentences_per_pair'],
                  minimum_positive_relations=cfg['minimum_positive_relations'],
                  support_gate=passed,
                  disposition='SUPPORT_ONLY_NOT_A_MECHANISM' if passed else 'CLOSE_EXACT_TYPE_HETEROGENEITY_LEAD',
                  claim='Complete TRAIN support census only; no measured query conflict, model utility, novelty or quantum attribution.',
                  heldout_rows_accessed=0, model_calls=0, training_updates=0,
                  elapsed_seconds=time.monotonic()-started)
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    save(a.output/'summary.json', result)
    (a.output/'RUN_COMPLETE').write_text('TRAIN metadata census complete\n')
    save(a.progress, dict(schema='research.progress.v1', attempt=a.attempt,
                         phase='completed_train_metadata_census', current=n, total=n,
                         unit='rows', observed_at=time.time(), heartbeat_at=time.time()))
    print(json.dumps({k: result[k] for k in ('rows','classes','eligible_positive_relations',
                                           'support_gate','disposition','elapsed_seconds')}))


if __name__ == '__main__':
    main()
