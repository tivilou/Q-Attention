"""Aggregate-only exact-token/entity-query audit; no tokenizer or model import."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def aggregate(lines):
    digest=hashlib.sha256();groups={};rows=0;labels=Counter()
    for line in lines:
        digest.update(line);row=json.loads(line);rows+=1
        tokens=row['tokens'];subject=row['subject'];obj=row['object'];label=row['label']
        if not isinstance(tokens,list) or not tokens or not all(isinstance(t,str) for t in tokens):raise ValueError('invalid original tokens')
        for span in [subject,obj]:
            if not isinstance(span,list) or len(span)!=2 or not all(type(x) is int for x in span) or not 0<=span[0]<span[1]<=len(tokens):raise ValueError('invalid entity span')
        if not isinstance(label,str) or not label:raise ValueError('invalid label')
        q=(tuple(subject),tuple(obj));counts=groups.setdefault(tuple(tokens),{}).setdefault(q,Counter());counts[label]+=1;labels[label]+=1
    unique_rows=sum(len(c) for queries in groups.values() for c in queries.values())
    conflict_queries=sum(len(c)>1 for queries in groups.values() for c in queries.values())
    conflict_groups=sum(any(len(c)>1 for c in queries.values()) for queries in groups.values())
    multi=mixed=raw_mixed=unique_mixed=reversed_groups=0
    for queries in groups.values():
        if any(len(c)>1 for c in queries.values()):continue
        if len(queries)>=2:
            multi+=1
            if len({label for c in queries.values() for label in c})>=2:
                mixed+=1;raw_mixed+=sum(sum(c.values()) for c in queries.values());unique_mixed+=sum(len(c) for c in queries.values())
        if any(a!=b and (b,a) in queries for a,b in queries):reversed_groups+=1
    return dict(train_sha256=digest.hexdigest(),rows=rows,distinct_labels=len(labels),token_groups=len(groups),unique_token_query_label_rows=unique_rows,exact_duplicate_extra_rows=rows-unique_rows,distinct_token_queries=sum(len(x) for x in groups.values()),conflicting_queries=conflict_queries,token_groups_with_conflicts=conflict_groups,conflict_free_multi_query_groups=multi,conflict_free_multi_query_mixed_label_groups=mixed,raw_rows_in_primary_groups=raw_mixed,deduplicated_rows_in_primary_groups=unique_mixed,conflict_free_groups_with_reversed_pair=reversed_groups)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert not a.output.exists() or a.output.is_dir() and not any(a.output.iterdir()),'preserve nonempty output'
    raw=a.protocol.read_bytes();protocol=json.loads(raw)
    assert protocol['schema']=='qattn.l5.train-only-task-premise-audit.v1'
    assert a.input.name=='train.jsonl' and a.input.is_file() and a.input.stat().st_size<=64*1024**2
    with a.input.open('rb') as f:counts=aggregate(f)
    assert counts['train_sha256']==protocol['expected_sha256'] and counts['rows']==protocol['expected_rows'] and counts['distinct_labels']==protocol['expected_label_count'],'wrong data identity or population'
    result=dict(schema='qattn.l5.train-only-task-premise-result.v1',status='PASS',counts=counts,protocol_sha256=hashlib.sha256(raw).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),original_input_bytes=a.input.stat().st_size,claim_scope=protocol['interpretation'],new_model_forwards=0,new_training=0,valid_or_test_files_read=False,raw_examples_exported=False,l5_accepted=False)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'compact-report.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());(a.output/'RUN_COMPLETE').write_bytes(b'train-only task premise audit complete\n')
    print(json.dumps(dict(status='PASS',counts=counts,l5_accepted=False)))


if __name__=='__main__':main()
