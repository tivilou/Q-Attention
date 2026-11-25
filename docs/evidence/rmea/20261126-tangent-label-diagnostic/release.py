"""Return only this settled allocation through installed trusted APIs."""
from pathlib import Path
from datetime import datetime
import json,sys,time
root=Path(__file__).resolve().parent
sys.path.insert(0,'D:/ResearchInfrastructure/scheduler/runtime/17831b100d60915a')
from allocation_control import rpc
from allocation_release import release_decision
from allocation_store import Allocations
config=json.loads(Path('D:/ResearchInfrastructure/scheduler/config.json').read_bytes())
store=Allocations(config['database']);identity=json.loads((root/'launch.json').read_bytes())['allocation_id']
a=next(a for a in store.rows() if a['allocation_id']==identity)
if a['state']=='released':
    result=dict(state='released',allocation_id=identity,generation=a['generation'],already_released=True)
else:
    observation=rpc(config,a['host_id'],dict(action='allocation_observe',allocation=a,ack=a['ack']))
    ev=observation['evidence'];assert ev
    delay=max(0,datetime.fromisoformat(ev['observed_at']).timestamp()+.15-time.time())
    assert delay < 3
    if delay:time.sleep(delay)
    decision=release_decision(a,ev,now=time.time());assert decision['release_allocation'],decision
    endpoint=rpc(config,a['host_id'],dict(action='allocation_release',allocation=a,evidence=ev))
    updated=store.release(a,ev)
    result=dict(state='released',allocation_id=identity,generation=a['generation'],decision=decision,
        trusted_observation=ev,endpoint=endpoint,updated=updated,clock_wait_seconds=delay,runtime_modified=False)
assert next(v for v in store.rows() if v['allocation_id']==identity)['state']=='released'
(root/'allocation-closeout.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
print(json.dumps({k:result[k] for k in ('state','allocation_id','generation')}))
