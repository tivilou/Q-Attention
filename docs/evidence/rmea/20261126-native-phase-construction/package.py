from pathlib import Path
import ast,hashlib,json
r=Path(__file__).resolve().parent
def sha(b):return hashlib.sha256(b).hexdigest()
def read(n):return json.loads((r/n).read_bytes())
v=read('verification.json');ex=read('verify-execution.json');b=read('closeout-build.json')
assert v['status']=='PASS_NATIVE_CONSTRUCTION_RECORDS' and ex['exit_status']==0 and not ex['stderr']
assert ex['code_sha256']==sha((r/'verify.py').read_bytes()) and json.loads(ex['stdout'])==v
assert b['analysis_sha256']==sha((r/'analysis.md').read_bytes())
assert b['next_direction_sha256']==sha((r/'next-direction.json').read_bytes())
build=read('build_closeout-execution.json');assert build['exit_status']==0 and not build['stderr']
assert build['code_sha256']==sha((r/'build_closeout.py').read_bytes())
n=read('next-direction.json');assert n['goal']=='ACTIVE' and n['collection']['max_ordinary_requests']==4
assert sha(Path(n['metadata_pin']['path']).read_bytes())==n['metadata_pin']['sha256']
for p in read('source-pins.json')[:2]:
    original=Path(p['path']);target=r/'native-source'/original.name;target.parent.mkdir(exist_ok=True)
    raw=original.read_bytes();assert sha(raw)==p['sha256']
    if target.exists():assert target.read_bytes()==raw
    else:target.write_bytes(raw)
names=['scope.json','source-pins.json','environment.json','receipt.before-execution.json',
       'prepare.py','prepare-execution.json','construct.py','independent_reference.py','construct-execution.json',
       'native-stage.json','trace.json','results.json','analytic-construction.md',
       'verify.py','verify-execution.json','verification.json','next-direction.json','analysis.md',
       'build_closeout.py','closeout-build.json','build_closeout-execution.json','execute.py',
       'native-source/rmea.py','native-source/frozen_feature_head.py']
raw={n:(r/n).read_bytes() for n in names}
for n,data in raw.items():
    if n.endswith('.json'):json.loads(data)
    if n.endswith('.py'):ast.parse(data.decode('utf8'))
assert sum(map(len,raw.values()))<512*1024
assert not any(n.endswith(('.pdf','.pt','.npz','.npy','.zip','.html','.png')) or '.private.' in n for n in raw)
out=dict(status='PASS_NATIVE_PHASE_EVIDENCE_PACKAGE',public_evidence_sha256={n:sha(data) for n,data in raw.items()},
         contents='Only explicit synthetic inputs/traces, native source and reports; no task data/features/checkpoint weights.',
         goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
(r/'package-verification.json').write_bytes((json.dumps(out,indent=2)+'\n').encode())
print(json.dumps(dict(status=out['status'],files=len(raw),bytes=sum(map(len,raw.values())))))
