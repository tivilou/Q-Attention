"""Only existing labeled TRAIN sentence groups; no changed model or inference input."""
from pathlib import Path
from collections import defaultdict
import hashlib,json
import numpy as np

def build(args,cfg,data,fit,assess):
    root=args.inputs;raw=(root/'recovery-inputs/train.jsonl').read_bytes();meta=(root/'discovery/cache/metadata.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==cfg['source_train_sha256']
    assert hashlib.sha256(meta).hexdigest()==cfg['metadata_sha256']
    original=[json.loads(line) for line in raw.splitlines() if line.strip()]
    metadata=json.loads(meta)['train']
    lookup={hashlib.sha256(str(row['metadata']['id']).encode()).hexdigest():row for row in original}
    assert len(lookup)==len(original)==len(metadata)==58465
    assert set(lookup)=={v['id_hash'] for v in metadata}
    rows=[lookup[v['id_hash']] for v in metadata]
    labels=json.loads((root/'recovery-inputs/labels.json').read_bytes());targets=data['targets']
    assert all(labels[int(y)]==row['label'] for y,row in zip(targets,rows))
    sentence=lambda row:hashlib.sha256(json.dumps(row['tokens'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    assert not {sentence(rows[i]) for i in fit}&{sentence(rows[i]) for i in assess}
    groups=defaultdict(list);seen=set()
    for i in sorted(map(int,fit),key=lambda j:metadata[j]['id_hash']):
        row=rows[i];identity=(tuple(row['tokens']),tuple(row['subject']),tuple(row['object']),row['label'])
        if identity in seen:continue
        seen.add(identity)
        key=(tuple(row['tokens']),tuple(row['subject']),row['metadata']['subj_type'],row['metadata']['obj_type'])
        groups[key].append(i)
    pairs=[]
    for indices in groups.values():
        for positive in indices:
            if targets[positive]==0:continue
            for negative in indices:
                if targets[negative]!=0:continue
                a,b=rows[positive]['object'],rows[negative]['object']
                if a[1]<=b[0] or b[1]<=a[0]:pairs.append((positive,negative))
    pairs=np.asarray(sorted(pairs),dtype=np.int64).reshape(-1,2)
    classes=sorted(set(int(targets[i]) for i,_ in pairs))
    if len(pairs)<cfg['minimum_fit_pairs'] or len(classes)<cfg['minimum_fit_pair_classes']:
        raise ValueError('Insufficient frozen fit-pair support: '+json.dumps(dict(pairs=len(pairs),classes=classes)))
    rng=np.random.default_rng(cfg['seed']+901);orders=[]
    while len(orders)<cfg['updates']*cfg['aux_pairs_per_batch']:orders.extend(rng.permutation(len(pairs)).tolist())
    batches=[pairs[orders[i:i+cfg['aux_pairs_per_batch']]].reshape(-1) for i in range(0,cfg['updates']*cfg['aux_pairs_per_batch'],cfg['aux_pairs_per_batch'])]
    record=dict(fit_pairs=len(pairs),positive_relations=classes,unique_sentences=len({sentence(rows[i]) for i,_ in pairs}),
                pair_table_sha256=hashlib.sha256(pairs.astype('<i8').tobytes()).hexdigest(),
                order_sha256=hashlib.sha256(b''.join(x.astype('<i8').tobytes() for x in batches)).hexdigest(),
                all_pairs_fit_only=True,assessment_sentence_disjoint=True,deduplicated_query_label=True)
    return dict(batches=batches,record=record)
