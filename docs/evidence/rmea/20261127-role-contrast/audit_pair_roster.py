"""Read-only TRAIN metadata availability for the already frozen 4096/1024/2048 roster."""
from collections import Counter,defaultdict
from pathlib import Path
import hashlib,json,sys,time

def sha(raw):return hashlib.sha256(raw).hexdigest()
def get(p,d):
    b=Path(p).read_bytes()
    if sha(b)!=d:raise ValueError('Input bytes differ from the pinned artifact')
    return b
def scan_rows(records,ids_by_row,roles,known_labels):
    inverse={ids_by_row[i]:(role,i) for role,rows in roles.items() for i in rows}
    expected=sum(map(len,roles.values()))
    if len(inverse)!=expected:raise AssertionError('Duplicate roster IDs')
    grouped={r:defaultdict(dict) for r in roles}
    seen=set()
    for row in records:
        key=sha(str(row['metadata']['id']).encode())
        if key not in inverse:continue
        assert key not in seen
        seen.add(key)
        role,i=inverse[key]
        group=sha(json.dumps(row['tokens'],ensure_ascii=False,separators=(',',':')).encode())
        s,o=tuple(row['subject']),tuple(row['object'])
        assert len(s)==len(o)==2 and s!=o and all(type(x) is int for x in s+o)
        assert 0<=s[0]<=s[1]<=len(row['tokens']) and 0<=o[0]<=o[1]<=len(row['tokens'])
        label=None
        if role=='fit':
            label=row['label']
            assert label in known_labels
        assert (s,o) not in grouped[role][group]
        grouped[role][group][(s,o)]=label
    assert len(seen)==expected
    for i,r in enumerate(roles):
        for t in list(roles)[i+1:]:assert not set(grouped[r])&set(grouped[t])
    return grouped

def summarize(grouped):
    results={}
    for role,groups in grouped.items():
        n_pairs=n_groups=0
        counts=Counter()
        type_groups=defaultdict(set)
        label_groups=defaultdict(set)
        distinct_groups=0
        for key,edges in groups.items():
            n=nd=0
            # (s,o) ordering uses geometry only. Both records must be present.
            for (s,o),label in edges.items():
                if not s<o or (o,s) not in edges:continue
                n+=1
                if role!='fit':continue
                y=edges[o,s]
                table=tuple(sorted((label,y)))
                counts[table]+=1
                type_groups[table].add(key)
                if label!=y:
                    nd+=1
                    label_groups[label].add(key)
                    label_groups[y].add(key)
            n_pairs+=n
            n_groups+=bool(n)
            distinct_groups+=bool(nd)
        x=dict(sentence_groups=len(groups),within_role_complete_reciprocal_pairs=n_pairs,
               within_role_reciprocal_groups=n_groups,labels_examined=(role=='fit'))
        if role=='fit':
            assert sum(counts.values())==n_pairs
            same=sum(v for (a,b),v in counts.items() if a==b)
            diff_non_null=sum(v for (a,b),v in counts.items() if a!=b and a!='no_relation' and b!='no_relation')
            diff_with_null=n_pairs-same-diff_non_null
            assert diff_with_null>=0
            x.update(same_label_pairs=same,distinct_label_pairs=n_pairs-same,distinct_label_groups=distinct_groups,
                distinct_both_positive=diff_non_null,distinct_with_null=diff_with_null,
                distinct_label_support={k:len(v) for k,v in sorted(label_groups.items())},
                fit_label_tables=[dict(labels=list(k),pairs=counts[k],groups=len(type_groups[k])) for k in sorted(counts)])
        results[role]=x
    return results

def main():
    t=time.monotonic()
    b=json.load(sys.stdin)
    roster=json.loads(get(b['cohort_path'],b['cohort_sha256']))
    m=json.loads(get(Path(b['cache'])/'metadata.json',b['metadata_sha256']))
    raw=get(b['train_source'],b['train_source_sha256'])
    jl=json.loads(get(Path(b['source'])/'docs/contracts/q_rmea_entity_labels.json',b['labels_sha256']))
    assert len(jl)==40 and jl[0]=='no_relation'
    roles=roster['indices']
    assert {k:len(v) for k,v in roles.items()}==dict(fit=4096,selection=1024,qualification=2048)
    keys=[x['id_hash'] for x in m['train']]
    assert len(keys)==len(set(keys))==58465
    groups=scan_rows((json.loads(l) for l in raw.splitlines() if l.strip()),keys,roles,set(jl))
    results=summarize(groups)
    expected={'fit':3582,'selection':884,'qualification':1724}
    assert {k:v['sentence_groups'] for k,v in results.items()}==expected
    assert time.monotonic()-t<45
    print(json.dumps(dict(status='PASS_FROZEN_ROLE_PAIR_ROSTER_METADATA_ONLY',cohort_sha256=b['cohort_sha256'],
        train_source_sha256=b['train_source_sha256'],metadata_sha256=b['metadata_sha256'],results=results,
        sentence_group_overlap=0,fit_rows_with_gold_label_inspected=4096,selection_gold_labels_indexed=0,qualification_gold_labels_indexed=0,
        valid_test_read=False,retained_train_jsonl_read=True,record_values_exported=False,remote_writes=0,
        new_task_forwards=0,updates=0,gpu_allocations=0,seconds=time.monotonic()-t),ensure_ascii=False))

if __name__=='__main__':main()
