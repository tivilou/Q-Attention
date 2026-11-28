"""Natural same-sentence, same-role single-endpoint contrast from sealed TRAIN cohorts."""

from collections import defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np


def read(pin):
    raw = Path(pin['path']).read_bytes()
    assert sha256(raw).hexdigest() == pin['sha256']
    return raw


def endpoint_pairs(edges):
    for a, b in combinations(sorted(edges), 2):
        s, o = edges[a]
        t, u = edges[b]
        if (s == t and o != u) or (o == u and s != t):
            yield (a,b)


def describe(groups, gold=None):
    pairs = 0
    candidate_groups = 0
    discordant_groups = set()
    discordant_edges = set()
    class_groups = defaultdict(set)
    shared_role = [0, 0]
    for key, edges in groups.items():
        if len(edges)<2:
            continue
        # Distinct raw examples can have the same token spans. Reject such
        # ambiguous identities before pairing; do not select among duplicates.
        by_pair = defaultdict(list)
        for i, edge in edges.items():
            by_pair[edge].append(i)
        clean = {members[0]: e for e, members in by_pair.items() if len(members)==1}
        links = list(endpoint_pairs(clean))
        candidate_groups += bool(links)
        pairs += len(links)
        if gold is None:
            continue
        for a, b in links:
            if not ((gold[a] == 0) ^ (gold[b] == 0)):
                continue
            p, n = (b, a) if gold[a] == 0 else (a, b)
            discordant_edges.add((p,n))
            discordant_groups.add(key)
            class_groups[int(gold[p])].add(key)
            shared_role[int(clean[a][0] != clean[b][0])] += 1
    counts = dict(rows=sum(map(len,groups.values())), sentence_groups=len(groups), same_role_distinct_pairs=pairs,
                  eligible_sentence_groups=candidate_groups, positive_null_labels_used=gold is not None)
    if gold is not None:
        counts.update(positive_null_pairs=len(discordant_edges), positive_null_sentence_groups=len(discordant_groups),
                      unique_positive_rows=len({p for p,n in discordant_edges}), unique_null_rows=len({n for p,n in discordant_edges}),
                      positive_classes=len(class_groups), positive_classes_with_at_least_2_groups=sum(len(v)>=2 for v in class_groups.values()),
                      positive_classes_with_at_least_5_groups=sum(len(v)>=5 for v in class_groups.values()),
                      shared_subject_pairs=shared_role[0], shared_object_pairs=shared_role[1])
        assert sum(shared_role)==len(discordant_edges)
    return counts,discordant_edges


def main(cfg):
    label_names = json.loads(read(cfg['label_contract']))
    assert len(label_names)==40 and label_names[0]=='no_relation'
    old_cohort = json.loads(read(cfg['pins']['cohort']))
    idx = np.asarray(old_cohort['assessment'],dtype=np.int64)
    meta = json.loads(read(cfg['metadata']))['train']
    assert len(meta)==len({m['id_hash'] for m in meta})==58465
    new = json.loads(read(cfg['new_cohort']))['indices']
    rosters = {'old_fit':old_cohort['fit'],'old_development':idx.tolist(),
               'fit':new['fit'],'selection':new['selection'],'qualification':new['qualification']}
    assert {k:len(v) for k,v in rosters.items()}==dict(old_fit=8192,old_development=4096,fit=4096,selection=1024,qualification=2048)
    assert not set(rosters['old_fit'])&set(rosters['old_development'])
    rows_needed = set().union(*(set(v) for v in rosters.values()))
    wanted = {meta[i]['id_hash']:i for i in rows_needed}
    records = {}
    fitgold = {}
    directions_done = set(new['fit'])
    label_lookup = {name:i for i,name in enumerate(label_names)}
    for line in read(cfg['train_source']).splitlines():
        if not line:
            continue
        row = json.loads(line)
        identity = sha256(str(row['metadata']['id']).encode()).hexdigest()
        if identity not in wanted:
            continue
        i=wanted[identity]
        tokens=row['tokens']
        assert i not in records and len(tokens)==meta[i]['words']
        key=sha256(json.dumps(tokens,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        s,o=tuple(row['subject']),tuple(row['object'])
        assert len(s)==len(o)==2 and s != o and 0<=s[0]<=s[1]<=len(tokens) and 0<=o[0]<=o[1]<=len(tokens)
        records[i]=(key,(s,o))
        if i in directions_done:
            fitgold[i]=label_lookup[row['label']]
    assert records.keys()==rows_needed and len(fitgold)==4096
    groups={}
    for k,idxs in rosters.items():
        g=defaultdict(dict)
        for i in idxs:
            key,edge=records[i]
            g[key][i]=edge
        groups[k]=g
    for a,b in combinations(('fit','selection','qualification'),2):
        assert not groups[a].keys()&groups[b].keys()
    target_path=Path(cfg['pins']['train_targets']['path'])
    assert sha256(target_path.read_bytes()).hexdigest()==cfg['pins']['train_targets']['sha256']
    y=np.load(target_path,allow_pickle=False)
    assert y.shape==(58465,) and y.dtype.kind=='i'
    # Form the strictly isolated cohort before indexing the historical labels.
    future_blind = set(new['selection'])|set(new['qualification'])
    avoid = set(groups['old_fit'])|set(groups['selection'])|set(groups['qualification'])
    strictly_separate = {k:v for k,v in groups['old_development'].items() if k not in avoid}
    stats={k:describe(v,fitgold if k=='fit' else None)[0] for k,v in groups.items()}
    rows=sorted(i for group in strictly_separate.values() for i in group)
    assert not set(rows)&future_blind
    stable=np.asarray(rows,dtype=np.int64)
    truth=y[stable]
    assert truth.min()==0 and truth.max()<=39
    labels_map=dict(zip(rows,map(int,truth)))
    isolated,pairs=describe(strictly_separate,labels_map)
    result={}
    old_positions={int(i):j for j,i in enumerate(idx)}
    selected_indices=np.array([old_positions[i] for i in rows],dtype=np.int64)
    position={i:j for j,i in enumerate(rows)}
    pi=sorted({position[p] for p,_ in pairs})
    ni=sorted({position[n] for _,n in pairs})
    for collection, models in [('distinct',('Q','D','G')),('reference',('Gcov',))]:
        pin=cfg['pins'][collection]
        assert sha256(Path(pin['path']).read_bytes()).hexdigest()==pin['sha256']
        with np.load(pin['path'],allow_pickle=False) as existing:
            assert np.array_equal(existing['assessment_indices'],idx)
            assert np.array_equal(existing['labels'][stable],truth)
            assert np.array_equal(np.flatnonzero(existing['calibration']),np.sort(rosters['old_fit']))
            for arm in models:
                archive=existing[cfg['keys'][arm]+'_logits']
                assert archive.shape==(4096,40)
                z=archive[selected_indices]
                assert z.shape==(len(rows),40) and np.isfinite(z).all()
                pred=z.argmax(1)
                pa=z[:,1:].argmax(1)+1
                result[arm]=dict(positive_correct=int((pred[pi]==truth[pi]).sum()),
                                 null_false_positive=int((pred[ni]!=0).sum()),
                                 positive_rank1=int((pa[pi]==truth[pi]).sum()))
    artificial={0:((0,1),(2,3)),1:((0,1),(4,5)),2:((4,5),(2,3)),3:((2,3),(0,1))}
    assert list(endpoint_pairs(artificial))==[(0,1),(0,2)]
    return dict(status='PASS_NATURAL_ORDERED_PAIR_ROSTER_AND_HISTORICAL_OUTCOME_LINK',
                rosters=stats,old_dev_sentence_groups_overlapping_old_fit=len(groups['old_development'].keys()&groups['old_fit'].keys()),
                old_dev_rows_overlapping_future_selection_qualification=len(set(idx)&future_blind),
                isolated_old_dev=isolated,isolated_same_input_historical_arms=result,
                new_roster_sentence_overlap=0,selection_or_qualification_labels_indexed_as_new_cohorts=0,
                old_exposed_score_rows=len(rows), protected_future_rows_were_filtered_before_gold_indexing=True,raw_rows_exported=0,optimizer_updates=0,task_forwards=0,gpu_allocations=0,
                source_pins={k:v['sha256'] for k,v in cfg['pins'].items()},
                metadata_sha256=cfg['metadata']['sha256'],train_source_sha256=cfg['train_source']['sha256'],
                new_cohort_sha256=cfg['new_cohort']['sha256'])


if __name__=='__main__':
    print(json.dumps(main(json.load(sys.stdin)),allow_nan=False))