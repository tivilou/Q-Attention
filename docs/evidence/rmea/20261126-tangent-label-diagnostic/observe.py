"""Read only this exact allocation through its registered progress adapter."""
from pathlib import Path
import json
import sys
import time

r = Path(__file__).resolve().parent
sys.path.insert(0, 'D:/ResearchInfrastructure/scheduler/runtime/17831b100d60915a')
from allocation_control import rpc
from allocation_store import Allocations
c = json.loads(Path('D:/ResearchInfrastructure/scheduler/config.json').read_bytes())
store = Allocations(c['database'])
launch = json.loads((r / 'launch.json').read_bytes())
a = next(x for x in store.rows() if x['allocation_id'] == launch['allocation_id'])
obs = rpc(c, a['host_id'], {'action': 'allocation_observe', 'allocation': a, 'ack': a['ack']})
private = Path('C:/Users/14630/AppData/Local/research-infrastructure/qattn-2080ti2-38434-20261122/tangent-label-20261126-v2')
(private / 'observation.private.json').write_bytes((json.dumps(obs, indent=2) + '\n').encode())
out = {k: obs.get(k) for k in ('progress', 'evidence', 'owner_alive', 'reason', 'state')}
out['observed_locally_at'] = time.time()
(r / 'last-observation.json').write_bytes((json.dumps(out, indent=2) + '\n').encode())
print(json.dumps(out))
