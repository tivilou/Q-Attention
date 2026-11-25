from pathlib import Path
import ast
import hashlib
import json

r=Path(__file__).resolve().parent
v=json.loads((r/'verification.json').read_bytes())
q=json.loads((r/'verify-execution.json').read_bytes())
assert v['status']=='PASS_EFFECT_PREFLIGHT_AND_REVIEW_RECORDS' and q['exit_status']==0 and q['stderr']==''
assert q['code_sha256']==hashlib.sha256((r/'verify.py').read_bytes()).hexdigest()
assert json.loads(q['stdout'])==v
names=('scope.json','derivation.md','derivation-v2.md','wording-clarification.diff','preflight.py','independent.py',
       'results.json','independent-results.json','preflight-execution.json','independent-execution.json',
       'receipt.before-execution.json','source-pins.json','history-comparison.json','retained-phase-value-null.md',
       'contrast.md','contrast.py','contrast-results.json','contrast-execution.json',
       'contrast-receipt.before-execution.json','contrast-source-pins.json','review/ai-model-world.json',
       'review/selection.json','review/transport-preflight.json','review/packet.json','review/review.json',
       'review-response.md','next-direction.json','analysis.md','verify.py','verify-execution.json','verification.json')
raw={n:(r/n).read_bytes() for n in names}
for n,b in raw.items():
    if n.endswith('.json'): json.loads(b)
    if n.endswith('.py'): ast.parse(b.decode('utf8'))
assert sum(map(len,raw.values()))<512*1024
assert not any(n.endswith(('.npy','.npz','.pt','.pdf')) or '.private.' in n for n in raw)
out=dict(status='PASS_EFFECT_PREFLIGHT_EVIDENCE_PACKAGE',public_evidence_sha256={n:hashlib.sha256(b).hexdigest() for n,b in raw.items()},
         static_review='PASS_WITH_WORDING_CLARIFICATION_APPLIED',task_training_admitted=False,new_theory=False,
         goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
(r/'package-verification.json').write_bytes((json.dumps(out,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in out.items() if k!='public_evidence_sha256'}))
