"""Real child-launch canary with one completion-marker fault and exact retry."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();root=a.output.resolve();assert not root.exists();root.mkdir(parents=True)
    sandbox=root/'synthetic-only';sandbox.mkdir();data=sandbox/'train.jsonl';protocol=sandbox/'protocol.json';target=sandbox/'result'
    source=Path(__file__).resolve().with_name('train_premise_audit.py');source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    def row(t,s,o,y):return dict(tokens=t,subject=s,object=o,label=y)
    rows=[row(['a','b'],[0,1],[1,2],'x'),row(['a','b'],[0,1],[1,2],'x'),row(['a','b'],[1,2],[0,1],'y'),row(['c','d'],[0,1],[1,2],'x'),row(['c','d'],[0,1],[1,2],'y'),row(['c','d'],[1,2],[0,1],'x'),row(['e','f'],[0,1],[1,2],'x'),row(['e','f'],[1,2],[0,1],'x')]
    raw=b''.join((json.dumps(r)+'\n').encode() for r in rows);data.write_bytes(raw)
    p=dict(schema='qattn.l5.train-only-task-premise-audit.v1',expected_sha256=hashlib.sha256(raw).hexdigest(),expected_rows=8,expected_label_count=2,interpretation='Synthetic canary only, not real data or task evidence.');protocol.write_bytes((json.dumps(p)+'\n').encode())
    pinned={q:hashlib.sha256(q.read_bytes()).hexdigest() for q in [data,protocol,source]};logs=[]
    args=['--input',str(data),'--protocol',str(protocol),'--output',str(target)]
    # Fault is confined to this child: same source/data/protocol, fails only the
    # final marker write after a compact report was written.
    inject='''import pathlib,runpy,sys
original=pathlib.Path.write_bytes
def write(self,data):
 if self.name=='RUN_COMPLETE':raise OSError('injected completion marker failure')
 return original(self,data)
pathlib.Path.write_bytes=write
source=sys.argv[1];sys.argv=[source]+sys.argv[2:];runpy.run_path(source,run_name='__main__')
'''
    def call(cmd):
        r=subprocess.run(cmd,capture_output=True,text=True,encoding='utf8',timeout=20);logs.append(dict(command=cmd,stdout=r.stdout,stderr=r.stderr,exit_status=r.returncode));return r
    failed=call([sys.executable,'-B','-c',inject,str(source),*args]);assert failed.returncode!=0 and 'injected completion marker failure' in failed.stderr
    assert target.is_dir() and set(q.name for q in target.iterdir())=={'compact-report.json'} and not (target/'RUN_COMPLETE').exists()
    assert target.resolve().parent==sandbox.resolve() and set(q.name for q in target.iterdir())=={'compact-report.json'};shutil.rmtree(target)
    assert all(hashlib.sha256(q.read_bytes()).hexdigest()==h for q,h in pinned.items())
    ok=call([sys.executable,'-B',str(source),*args]);assert ok.returncode==0 and set(q.name for q in target.iterdir())=={'compact-report.json','RUN_COMPLETE'}
    result=json.loads((target/'compact-report.json').read_bytes());c=result['counts'];assert c['rows']==8 and c['exact_duplicate_extra_rows']==1 and c['conflicting_queries']==1 and c['conflict_free_multi_query_mixed_label_groups']==1 and c['raw_rows_in_primary_groups']==3 and c['deduplicated_rows_in_primary_groups']==2
    assert all(hashlib.sha256(q.read_bytes()).hexdigest()==h for q,h in pinned.items()) and source_hash==hashlib.sha256(source.read_bytes()).hexdigest()
    report=dict(schema='qattn.l5.train-premise-lifecycle-canary.v1',status='PASS',injected_failure_observed=True,partial_output_not_complete=True,partial_output_cleaned=True,retry_same_immutable_source_inputs=True,training=0,real_data_read=False,commands=logs,fixture_counts=c)
    assert sandbox.resolve().parent==root and set(q.name for q in sandbox.iterdir())=={'train.jsonl','protocol.json','result'};shutil.rmtree(sandbox)
    (root/'compact-report.json').write_bytes((json.dumps(report,indent=2)+'\n').encode());(root/'RUN_COMPLETE').write_bytes(b'synthetic lifecycle canary complete\n');print(json.dumps(dict(status='PASS',injected_failure_and_retry=True,real_data_read=False)))


if __name__=='__main__':main()
