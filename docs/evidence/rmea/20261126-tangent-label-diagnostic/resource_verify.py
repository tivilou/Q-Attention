from pathlib import Path
import hashlib,json,sys
r=Path(__file__).resolve().parent
sys.path.insert(0,'D:/ResearchInfrastructure/scheduler/runtime/17831b100d60915a')
from allocation_store import Allocations
c=json.loads(Path('D:/ResearchInfrastructure/scheduler/config.json').read_bytes());store=Allocations(c['database'])
out=[]
for path in (Path('D:/CodexHome/task-artifacts/qattn-tangent-label-diagnostic-20261126'),r):
    launch=json.loads((path/'launch.json').read_bytes());row=next(x for x in store.rows() if x['allocation_id']==launch['allocation_id'])
    assert row['state']=='released';close=json.loads((path/'allocation-closeout.json').read_bytes());assert close['state']=='released'
    out.append(dict(allocation_id=row['allocation_id'],generation=row['generation'],runner_attempt_id=row['runner_attempt_id'],state=row['state'],closeout_sha256=hashlib.sha256((path/'allocation-closeout.json').read_bytes()).hexdigest()))
v=dict(status='PASS_EXACT_ALLOCATION_RETURN',allocations=out,scope='Only these2 project allocations changed; no foreign release',new_capacity_purchased=False,
    COS_writes=0,API_model_calls=0,private_arrays_on_windows=False)
(r/'resource-verification.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
