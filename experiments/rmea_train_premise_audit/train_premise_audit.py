"""Aggregate-only exact-token/entity-query audit; no tokenizer or model import."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile


def aggregate(lines,directory=None):
    digest=hashlib.sha256();rows=0;labels=Counter();base=Path(directory or Path(__file__).resolve().parent).resolve()
    # Ephemeral disk tables bound memory. Canonical JSON is injective for the
    # validated token/span lists; BINARY comparison preserves exact strings.
    with tempfile.TemporaryDirectory(prefix='train-premise-',dir=base) as scratch:
        assert Path(scratch).resolve().parent==base
        db=sqlite3.connect(str(Path(scratch)/'counts.sqlite3'))
        try:
            db.execute('PRAGMA cache_size=-2048');db.execute('PRAGMA temp.cache_size=-2048');db.execute('PRAGMA temp_store=FILE');db.execute('PRAGMA journal_mode=OFF')
            db.execute('CREATE TABLE t(tokens TEXT COLLATE BINARY,query TEXT COLLATE BINARY,label TEXT COLLATE BINARY,n INTEGER,rq TEXT,PRIMARY KEY(tokens,query,label)) WITHOUT ROWID')
            def encoded(x):return json.dumps(x,ensure_ascii=True,separators=(',',':'))
            for line in lines:
                digest.update(line);row=json.loads(line);rows+=1
                tokens=row['tokens'];subject=row['subject'];obj=row['object'];label=row['label']
                if not isinstance(tokens,list) or not tokens or not all(isinstance(t,str) for t in tokens):raise ValueError('invalid original tokens')
                for span in [subject,obj]:
                    if not isinstance(span,list) or len(span)!=2 or not all(type(x) is int for x in span) or not 0<=span[0]<span[1]<=len(tokens):raise ValueError('invalid entity span')
                if not isinstance(label,str) or not label:raise ValueError('invalid label')
                db.execute('INSERT INTO t VALUES(?,?,?,?,?) ON CONFLICT(tokens,query,label) DO UPDATE SET n=n+1',(encoded(tokens),encoded([subject,obj]),label,1,encoded([obj,subject])));labels[label]+=1
            db.commit()
            def scalar(sql):return db.execute(sql).fetchone()[0]
            unique_rows=scalar('SELECT COUNT(*) FROM t');token_groups=scalar('SELECT COUNT(*) FROM (SELECT tokens FROM t GROUP BY tokens)');queries=scalar('SELECT COUNT(*) FROM (SELECT tokens,query FROM t GROUP BY tokens,query)')
            conflicts='SELECT tokens,query FROM t GROUP BY tokens,query HAVING COUNT(*)>1'
            conflict_queries=scalar('SELECT COUNT(*) FROM ('+conflicts+')');conflict_groups=scalar('SELECT COUNT(DISTINCT tokens) FROM ('+conflicts+')')
            db.execute('CREATE TEMP TABLE clean(tokens TEXT COLLATE BINARY,query TEXT COLLATE BINARY,label TEXT,n INTEGER,rq TEXT,PRIMARY KEY(tokens,query)) WITHOUT ROWID')
            db.execute('INSERT INTO clean SELECT tokens,query,label,n,rq FROM t WHERE tokens NOT IN (SELECT tokens FROM ('+conflicts+'))')
            multi=mixed=raw_mixed=unique_mixed=0
            for nqueries,nlabels,nraw in db.execute('SELECT COUNT(*),COUNT(DISTINCT label),SUM(n) FROM clean GROUP BY tokens HAVING COUNT(*)>=2'):
                multi+=1
                if nlabels>=2:mixed+=1;raw_mixed+=nraw;unique_mixed+=nqueries
            reversed_groups=scalar('SELECT COUNT(DISTINCT a.tokens) FROM clean a JOIN clean b ON a.tokens=b.tokens AND a.rq=b.query WHERE a.query<>a.rq')
            return dict(train_sha256=digest.hexdigest(),rows=rows,distinct_labels=len(labels),token_groups=token_groups,unique_token_query_label_rows=unique_rows,exact_duplicate_extra_rows=rows-unique_rows,distinct_token_queries=queries,conflicting_queries=conflict_queries,token_groups_with_conflicts=conflict_groups,conflict_free_multi_query_groups=multi,conflict_free_multi_query_mixed_label_groups=mixed,raw_rows_in_primary_groups=raw_mixed,deduplicated_rows_in_primary_groups=unique_mixed,conflict_free_groups_with_reversed_pair=reversed_groups)
        finally:db.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    assert not a.output.exists() or a.output.is_dir() and not any(a.output.iterdir()),'preserve nonempty output'
    raw=a.protocol.read_bytes();protocol=json.loads(raw)
    assert protocol['schema']=='qattn.l5.train-only-task-premise-audit.v1'
    assert a.input.name=='train.jsonl' and a.input.is_file() and a.input.stat().st_size<=64*1024**2
    with a.input.open('rb') as f:counts=aggregate(f,directory=a.output.parent)
    assert counts['train_sha256']==protocol['expected_sha256'] and counts['rows']==protocol['expected_rows'] and counts['distinct_labels']==protocol['expected_label_count'],'wrong data identity or population'
    try:
        import resource
        peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if sys.platform.startswith('linux') else None
    except ImportError:peak_kib=None
    result=dict(schema='qattn.l5.train-only-task-premise-result.v1',status='PASS',counts=counts,protocol_sha256=hashlib.sha256(raw).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),original_input_bytes=a.input.stat().st_size,sqlite_aggregation=True,peak_rss_kib_linux=peak_kib,claim_scope=protocol['interpretation'],new_model_forwards=0,new_training=0,valid_or_test_files_read=False,raw_examples_exported=False,l5_accepted=False)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'compact-report.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());(a.output/'RUN_COMPLETE').write_bytes(b'train-only task premise audit complete\n')
    print(json.dumps(dict(status='PASS',counts=counts,l5_accepted=False)))


if __name__=='__main__':main()
