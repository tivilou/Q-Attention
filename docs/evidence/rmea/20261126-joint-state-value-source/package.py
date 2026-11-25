from pathlib import Path
import ast,hashlib,json
r=Path(__file__).resolve().parent
read=lambda n:json.loads((r/n).read_bytes())
sha=lambda b:hashlib.sha256(b).hexdigest()
assert read('verification.json')['status']=='PASS_SOURCE_IDENTITY_SCOPE_AND_RETAINED_PINS'
for n in ('report','verify'):
 e=read(n+'-execution.json');assert e['exit_status']==0 and not e['stderr']
 assert e['code_sha256']==sha((r/(n+'.py')).read_bytes())
names=['analysis.md','source-request.json','source-resolution.json','fulltext-request.json',
 'reading-receipt.json','source-pins.json','comparison.json','summary.json','next-direction.json',
 'verification.json','report.py','report-execution.json','verify.py','verify-execution.json','resolve_source.py','execute.py']
raw={n:(r/n).read_bytes() for n in names}
for n,b in raw.items():
 if n.endswith('.json'):json.loads(b)
 if n.endswith('.py'):ast.parse(b)
assert sum(map(len,raw.values()))<150000
assert all('.private' not in n and not n.endswith(('.pdf','.html','.png','.pt','.npy','.npz')) for n in names)
out=dict(status='PASS_SCOPED_SOURCE_EVIDENCE_PACKAGE',public_evidence_sha256={n:sha(b) for n,b in raw.items()},
 decision='NONE_NEW_STRUCTURAL_CANDIDATE',goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
(r/'package-verification.json').write_bytes((json.dumps(out,indent=2)+'\n').encode())
print(json.dumps(dict(status=out['status'],files=len(raw),bytes=sum(map(len,raw.values())))))
