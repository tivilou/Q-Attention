from pathlib import Path
import ast,hashlib,json
r=Path(__file__).resolve().parent
read=lambda n:json.loads((r/n).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read('verification.json')['status']=='PASS_REOPENED_SOURCE_CLASS_AUDIT'
for before in ('diagnostic','centering'):
    e=read('audit-execution.before-'+before+'.json');assert e['exit_status']==1 and e['code_sha256']==sha(r/('audit.before-'+before+'.py'))
for n in ('audit','verify','closeout'):
    e=read(n+'-execution.json');assert e['exit_status']==0 and not e['stderr'] and e['code_sha256']==sha(r/(n+'.py'))
for pin in read('headroom-reuse-audit.json')['evidence_pins']+read('next-direction.json')['closed_route_pins']:assert sha(pin['path'])==pin['sha256']
names=['scope.json','prepare.py','derivation.md','independent_reference.py','fw_trace.py','audit.py','audit-execution.json','results.json',
 'audit.before-diagnostic.py','audit-execution.before-diagnostic.json','audit.before-centering.py','audit-execution.before-centering.json',
 'algebra-trace.json','distinct.values.json','repeated.values.json','sample-trace.v2.json','source/validate_sample_trace_v2.py','source/sample-trace.v2.schema.json',
 'verify.py','verify-execution.json','verification.json','closeout.py','closeout-execution.json','headroom-reuse-audit.json','next-direction.json','analysis.md','execute.py']
raw={n:(r/n).read_bytes() for n in names}
for n,b in raw.items():
    if n.endswith('.json'):json.loads(b)
    if n.endswith('.py'):ast.parse(b.decode('utf8'))
assert sum(map(len,raw.values()))<2*1024*1024
assert all('.private.' not in n and not n.endswith(('.pdf','.html','.png','.zip','.pt','.npz','.npy')) for n in raw)
out=dict(status='PASS_NATIVE_FW_CLASS_EVIDENCE_PACKAGE',public_evidence_sha256={n:hashlib.sha256(b).hexdigest() for n,b in raw.items()},
    decision='NO_QUALIFIED_NEW_METHOD_FROM_THIS_AUDIT',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED',goal='ACTIVE')
(r/'package-verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(status=out['status'],files=len(raw),bytes=sum(map(len,raw.values())))))
