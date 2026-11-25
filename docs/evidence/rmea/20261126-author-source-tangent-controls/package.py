from pathlib import Path
import ast,hashlib,json
r=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
def read(n):return json.loads((r/n).read_bytes())
v=read('verification.json');e=read('verify-execution.json')
assert v['status']=='PASS_AUTHOR_CHANNEL_AND_TANGENT_RECORDS' and e['exit_status']==0 and not e['stderr']
assert e['code_sha256']==sha((r/'verify.py').read_bytes()) and json.loads(e['stdout'])==v
names=['scope.json','prepare.py','prepare-execution.json','orcid_person.py','orcid_person-execution.json',
       'orcid-person-result.json','fetch_profile.py','request-ledger.json','wikipedia-author-result.json',
       'arxiv-author-result.json','arxiv-author-execution.json','index-inspection.json','source-summary.json',
       'tangent/derivation.md','tangent/prepare.py','tangent/prepare-execution.json','tangent/scope.json',
       'tangent/receipt.before-execution.json','tangent/decompose.py','tangent/decompose-execution.json',
       'tangent/trace.json','tangent/results.json','tangent/execute.py',
       'analysis.md','next-direction.json','build_closeout.py','build_closeout-execution.json','closeout-build.json',
       'verify.py','verification.json','verify-execution.json','execute.py']
raw={n:(r/n).read_bytes() for n in names}
for n,b in raw.items():
 if n.endswith('.json'):json.loads(b)
 if n.endswith('.py'):ast.parse(b.decode('utf8'))
assert sum(map(len,raw.values()))<512*1024
assert not any('.private.' in n or n.endswith(('.pdf','.pt','.npz','.npy','.zip','.html','.png')) for n in raw)
out=dict(status='PASS_AUTHOR_TANGENT_EVIDENCE_PACKAGE',public_evidence_sha256={n:sha(b) for n,b in raw.items()},
         original_source='UNRESOLVED_NOT_READ',new_theory=False,task_method_admitted=False,
         goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
(r/'package-verification.json').write_bytes((json.dumps(out,indent=2)+'\n').encode())
print(json.dumps(dict(status=out['status'],files=len(raw),bytes=sum(map(len,raw.values())))))
