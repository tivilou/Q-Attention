from pathlib import Path
import ast,hashlib,json
r=Path(__file__).resolve().parent
assert json.loads((r/'closeout-verification.json').read_bytes())['status']=='PASS_FROZEN_DIAGNOSTIC_CLOSEOUT'
assert json.loads((r/'verify-execution.json').read_bytes())['exit_status']==0
names=['design.json','protocol.md','runner.py','runner.diff','tangent_math.py','trace_io.py','independent_reference.py',
 'source/validate_sample_trace_v2.py','source/sample-trace.v2.schema.json','local-preflight.json','check_math.py',
 'implementation-review.json','metadata-contract-audit.json','metadata-mapping-audit.json','input-verification.json',
 'prepare.py','preparation.json','admit.py','start.py','allocation-summary.json','launch.json','observe.py','first-progress-observation.json',
 'collect.py','execution.json','summary.json','verification.json','RUN_COMPLETE','private-output-inventory.json',
 'audit_code.py','audit.py','audit-execution.json','output-audit.json','release.py','allocation-closeout.json','resource_verify.py','resource_verify-execution.json',
 'resource-verification.json','build_closeout.py','build_closeout-execution.json','next-direction.json','analysis.md','verify.py','verify-execution.json','closeout-verification.json','execute.py']
names += ['failed-v1/'+n for n in ('design.json','protocol.md','runner.py','execution.json','failure-review.json','metadata-mapping-audit.json','allocation-closeout.json','local-preflight.json')]
raw={n:(r/n).read_bytes() for n in names}
for n,b in raw.items():
    if n.endswith('.json'):json.loads(b)
    if n.endswith('.py'):ast.parse(b.decode('utf8'))
assert sum(map(len,raw.values()))<640*1024
assert not any('.private.' in n or n.endswith(('.pt','.npz','.npy','.pdf','.html')) for n in raw)
out=dict(status='PASS_TANGENT_LABEL_EVIDENCE_PACKAGE',public_evidence_sha256={n:hashlib.sha256(b).hexdigest() for n,b in raw.items()},
    goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED',decision='STOP_CURRENT_LABEL_DIRECTION')
(r/'package-verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(status=out['status'],files=len(raw),bytes=sum(map(len,raw.values())))))
