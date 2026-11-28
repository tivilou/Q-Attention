"""FIT-only roster for naturally matched, non-entity punctuation variants."""
from collections import defaultdict
from hashlib import sha256
import itertools
import json
from pathlib import Path
import sys


PUNCTUATION={'.',',',';',':','?','!'}


def read(pin):
    raw=Path(pin['path']).read_bytes()
    assert sha256(raw).hexdigest()==pin['sha256']
    return raw


def normalize(tokens,s,o):
    spans = [tuple(s),tuple(o)]
    assert all(0<=start<end<=len(tokens) for start,end in spans)
    assert s!=o and (s[1]<=o[0] or o[1]<=s[0])
    keep=[i for i,word in enumerate(tokens)
          if word not in PUNCTUATION or any(start<=i<end for start,end in spans)]
    positions={idx:new for new,idx in enumerate(keep)}
    mapped=tuple((positions[a],positions[b-1]+1) for a,b in spans)
    return tuple(tokens[i] for i in keep),mapped


def run(cfg):
    rosters=json.loads(read(cfg['cohort']))['indices']
    fit=rosters['fit']
    assert len(fit)==len(set(fit))==4096 and not set(fit)&(set(rosters['selection'])|set(rosters['qualification']))
    meta=json.loads(read(cfg['metadata']))['train']
    assert len(meta)==len({m['id_hash'] for m in meta})==58465
    label_names=json.loads(read(cfg['label_contract']))
    assert len(label_names)==40 and label_names[0]=='no_relation'
    labels=set(label_names)
    wanted={meta[i]['id_hash']:i for i in fit}
    groups=defaultdict(list)
    accounted=set()
    removed=[]
    for line in read(cfg['source']).splitlines():
        if not line.strip():
            continue
        r=json.loads(line)
        identifier=sha256(str(r['metadata']['id']).encode()).hexdigest()
        i=wanted.get(identifier)
        if i is None:
            continue
        assert i not in accounted
        accounted.add(i)
        t,s,o=tuple(r['tokens']),tuple(r['subject']),tuple(r['object'])
        assert len(t)==meta[i]['words'] and r['label'] in labels
        canon=normalize(t,s,o)
        old=(t,s,o)
        groups[canon].append((old,r['label']))
        removed.append(len(t)-len(canon[0]))
    assert accounted==set(fit)
    natural_groups=0
    pairs=[]
    for vals in groups.values():
        eligible=[(a,b) for a,b in itertools.combinations(vals,2) if a[0][0]!=b[0][0]]
        if eligible:
            natural_groups+=1
            pairs.extend(eligible)
    agreeing=sum(a[1]==b[1] for a,b in pairs)
    both_null=sum(a[1]==b[1]=='no_relation' for a,b in pairs)
    return dict(status='PASS_FIT_ONLY_NATURAL_PUNCTUATION_VARIANT_CENSUS', fit_rows=len(fit),
                rows_with_at_least_one_non_entity_punctuation_removed=sum(n>0 for n in removed),
                natural_groups=natural_groups,natural_variant_pairs=len(pairs),same_label_pairs=agreeing,
                same_positive_label_pairs=agreeing-both_null,distinct_label_pairs=len(pairs)-agreeing,
                scope='One metadata predicate in frozen FIT. Normalization does not certify semantic invariance or a feature-space error.',
                normalization='Drop isolated exact . , ; : ? ! tokens outside both marked entity spans; preserve all other tokens and remap ordered spans.',
                source_hashes={k:v['sha256'] for k,v in cfg.items()},
                selection_qualification_labels_indexed=0,new_model_forwards=0,new_updates=0,new_gpu_allocations=0)


if __name__=='__main__':
    print(json.dumps(run(json.load(sys.stdin)),allow_nan=False))