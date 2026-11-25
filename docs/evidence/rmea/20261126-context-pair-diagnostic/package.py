from pathlib import Path
import ast,hashlib,json
r=Path(__file__).resolve().parent
read=lambda n:json.loads((r/n).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read('closeout.json')['status']=='PASS_GEN105_NEGATIVE_DIAGNOSTIC_CLOSEOUT'
assert read('verification.json')['status']=='PASS' and read('independent-verification.json')['status']=='PASS_REOPENED_PRIVATE_POPULATION_FRACTION_AUC'
assert read('execution-verification.json')['status']=='PASS_ACTUAL_EXECUTED_SOURCE_AND_CONFIG'
for n in ('prepare','probe','preflight','collect','release','closeout','verify_executed'):
 e=read(n+'-execution.json');assert e['exit_status']==0 and not e['stderr'] and e['code_sha256']==sha(r/(n+'.py'))
assert read('transport-failed/collect-execution.json')['exit_status']==1
for n,h in read('RUN_COMPLETE').items():assert sha(r/n)==h
for n,h in read('design.json')['code_hashes'].items():assert sha(r/n)==h
for p in read('next-direction.json')['source_pins']:assert sha(p['path'])==p['sha256']
assert read('resource-verification.json')['state']=='released'
names=['identity-spec.json','identity-probe.json','prepare.py','prepare-execution.json','probe_remote.py','probe.py','probe-execution.json',
 'runner.py','pair_math.py','pair_statistics.py','preflight.py','preflight-execution.json','local-preflight.json',
 'design.json','protocol.md','staged-design.json','staged-protocol.md','summary.json','verification.json','RUN_COMPLETE','execution.json',
 'verify_remote.py','collect.py','collect-execution.json','independent-verification.json','execution-verification.json','verify_executed.py','verify_executed-execution.json',
 'closeout.py','closeout-execution.json','closeout.json','resource-verification.json','allocation-closeout.json','next-direction.json','analysis.md',
 'collection-repair.diff','transport-failed/collect.py','transport-failed/collect-execution.json','execute.py']
raw={n:(r/n).read_bytes() for n in names}
for n,b in raw.items():
 if n.endswith('.json'):json.loads(b)
 if n.endswith('.py'):ast.parse(b.decode('utf8'))
assert sum(map(len,raw.values()))<1024*1024
assert all('.private.' not in n and not n.endswith(('.pdf','.pt','.npz','.npy','.html','.png')) for n in raw)
out=dict(status='PASS_GEN105_DIAGNOSTIC_EVIDENCE_PACKAGE',public_evidence_sha256={n:hashlib.sha256(b).hexdigest() for n,b in raw.items()},
 decision='STOP_THIS_EXACT_ALIASING_PREDICTION',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED',goal='ACTIVE')
(r/'package-verification.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=out['status'],files=len(raw),bytes=sum(map(len,raw.values())))))
