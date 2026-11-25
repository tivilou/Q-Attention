"""Reconcile this request only; leave every foreign allocation untouched."""
from pathlib import Path
import json,sys,time
r=Path(__file__).resolve().parent
p=Path('C:/Users/14630/AppData/Local/research-infrastructure/qattn-2080ti2-38434-20261122/tangent-label-20261126-v2')
sys.path.insert(0,str(p.parent));import ops
sys.path.insert(0,'D:/ResearchInfrastructure/scheduler/runtime/17831b100d60915a')
from allocation_store import Allocations
from allocation_control import rpc
c=json.loads(Path('D:/ResearchInfrastructure/scheduler/config.json').read_bytes());store=Allocations(c['database'])
spec=json.loads((p/'request.json').read_bytes());assert spec['resources']['gpus']==1;request=store.requests(spec['request_id'])[0]
assert request['state'] in ('requested','allocated')
if request['allocation']:
 a=next(a for a in store.rows() if a['allocation_id']==request['allocation'])
else:
 occupied=[a for a in store.rows() if a['host_id']==ops.SERVER and a['state']!='released']
 if occupied:
  status=dict(state='queued',started=False,request_id=spec['request_id'],observed_at=time.time(),
   wait_reason='Exact data-locality host reserved by another project; no displacement or duplicate.')
  (r/'request-status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status));sys.exit(0)
 health=json.loads((p/'fleet.private.json').read_bytes())
 # prepare.py refreshes the whole fleet; endpoint admission below is fresh as well.
 match=next(v for v in health['servers'] if v['id']==ops.SERVER)
 assert match['status']=='online' and time.time()-match['checked_at']<180,'Refresh through prepare.py'
 node=c['allocation_hosts'][ops.SERVER];assert node['enabled'] and 'q-attention' in node['projects']
 probe=rpc(c,ops.SERVER,dict(action='allocation_probe',allocation={'allocation_id':'0'*32}))
 need=spec['resources'];gpus=[g['uuid'] for g in probe['gpus'] if g['free_mb']>=need['gpu_mb']]
 assert not probe['reservations'] and len(gpus)>=1 and probe['available_ram_mb']>=need['ram_mb'] and probe['available_disk_mb']>=need['disk_mb']
 a=store.reserve(request,ops.SERVER,gpus[:1]);assert a
if a['state']=='claiming':
 a=store.claimed(a,rpc(c,ops.SERVER,dict(action='allocation_claim',allocation=a,resources=spec['resources'],host_capacity=c['allocation_hosts'][ops.SERVER]['capacity'])))
assert a['state']=='allocated' and a['runner_attempt_id']==spec['runner_attempt_id']
(p/'allocation.private.json').write_text(json.dumps(a,indent=2)+'\n')
v={k:a[k] for k in ('allocation_id','generation','host_id','runner_attempt_id','state','gpu_uuids')}
(r/'allocation-summary.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
