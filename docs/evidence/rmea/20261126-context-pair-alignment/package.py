from pathlib import Path
import ast,hashlib,json
r=Path(__file__).resolve().parent
read=lambda n:json.loads((r/n).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert read('verification.json')['status']=='PASS_REOPENED_SYNTHETIC_PAIRING_AND_SOURCE_EXCLUSION'
for n in ('prepare','select','audit','verify','closeout'):
 e=read(n+'-execution.json');assert e['exit_status']==0 and not e['stderr'] and e['code_sha256']==sha(r/(n+'.py'))
assert read('select-execution.before-syntax.json')['exit_status']==1
assert read('audit-execution.before-scope.json')['exit_status']==1
assert read('select-execution.before-syntax.json')['code_sha256']==sha(r/'select.before-syntax.py')
try:ast.parse((r/'select.before-syntax.py').read_text('utf8'))
except SyntaxError:pass
else:raise AssertionError('Preserved syntax failure missing')
for p in read('scope.json')['pins']:assert sha(p['path'])==p['sha256']
names=['scope.json','prepare.py','prepare-execution.json','select.py','select-execution.json',
 'source-exclusion.json','hypothesis.json','audit.py','audit-execution.json','algebra-trace.json',
 'results.json','verify.py','verify-execution.json','verification.json','derivation.md','analysis.md',
 'closeout.py','closeout-execution.json','next-direction.json','execute.py',
 'select.before-syntax.py','select-execution.before-syntax.json','audit-execution.before-scope.json']
raw={n:(r/n).read_bytes() for n in names}
for n,b in raw.items():
 if n.endswith('.json'):json.loads(b)
 if n.endswith('.py') and n!='select.before-syntax.py':ast.parse(b.decode('utf8'))
assert sum(map(len,raw.values()))<512*1024
assert all('.private.' not in n and not n.endswith(('.pdf','.pt','.npz','.npy','.html','.png')) for n in raw)
out=dict(status='PASS_PAIR_ALIGNMENT_SOURCE_EVIDENCE_PACKAGE',public_evidence_sha256={n:hashlib.sha256(b).hexdigest() for n,b in raw.items()},
 decision='ONE_EXPLORATORY_DIAGNOSTIC_ONLY_NO_METHOD_OR_TRAINING_ADMISSION',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED',goal='ACTIVE')
(r/'package-verification.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=out['status'],files=len(raw),bytes=sum(map(len,raw.values())))))
