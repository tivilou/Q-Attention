from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,uuid,time
r=Path(__file__).resolve().parent
p=Path('C:/Users/14630/AppData/Local/research-infrastructure/qattn-2080ti2-38434-20261122/tangent-label-20261126-v2')
cli='E:/Dir/ClaudeCode/.local/bin/researchctl.cmd';d=json.loads((r/'design.json').read_bytes())
assert json.loads((r/'local-preflight.json').read_bytes())['status']=='PASS_LOCAL_PREFLIGHT'
q=subprocess.run([cli,'health','--refresh'],capture_output=True,timeout=55)
(p/'fleet.private.json').write_bytes(q.stdout);assert q.returncode==0,q.stderr[-500:]
health=json.loads(q.stdout);node=next(v for v in health['servers'] if v['id']=='gpushare-i2-38434-2080tix2')
assert node['status']=='online' and time.time()-node['checked_at']<180
assert not (p/'request.json').exists(),'Reconcile exact existing request'
v=dict(request_id=d['profile'],project_id='q-attention',run_id=d['profile'],runner_attempt_id=uuid.uuid4().hex,
    preferred_host_ids=['gpushare-i2-38434-2080tix2'],require_preferred=True,resources=d['resources'])
(p/'request.json').write_text(json.dumps(v,indent=2)+'\n')
q=subprocess.run([cli,'allocation-request','--project','q-attention','--spec',str(p/'request.json')],capture_output=True,timeout=30)
(p/'request-response.private.json').write_bytes(q.stdout);assert q.returncode==0,q.stderr[-500:]
summary=dict(frozen_at=datetime.now(timezone.utc).isoformat(),design_sha256=hashlib.sha256((r/'design.json').read_bytes()).hexdigest(),
    protocol_sha256=hashlib.sha256((r/'protocol.md').read_bytes()).hexdigest(),request_id=v['request_id'],runner_attempt_id=v['runner_attempt_id'],
    input_verified=51,math_preflight='PASS',new_task_calls_before_request=0,
    selected_host=node['id'],selected_reason='Immutable pinned input locality; preferred3060 offline in whole-fleet view',
    fleet=[{k:x.get(k) for k in ('id','status','reason','checked_at','next_probe_at')} for x in health['servers'] if x['id'] in ('gpushare-i2-38434-2080tix2','gpushare-i2-45579-3060x3')])
(r/'preparation.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
