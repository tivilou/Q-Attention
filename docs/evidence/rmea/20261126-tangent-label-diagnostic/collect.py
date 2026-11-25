"""Allowlisted reports and literal execution; task arrays remain private on host."""
from pathlib import Path
import hashlib,json,sys
r=Path(__file__).resolve().parent
sys.path.insert(0,'C:/Users/14630/AppData/Local/research-infrastructure/qattn-2080ti2-38434-20261122');import ops
launch=json.loads((r/'launch.json').read_bytes())
code='''from pathlib import Path
import json,hashlib,sys
r=Path('/hy-tmp/Q-Attention/control/tangent-label-diagnostic-20261126-v2');e=json.loads((r/'execution.json').read_bytes())
reports={n:(r/'output'/n).read_text() for n in ('summary.json','verification.json','RUN_COMPLETE') if (r/'output'/n).exists()}
inventory={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (r/'output').iterdir() if p.name.endswith(('.private.npz','.private.json'))}
print(json.dumps(dict(execution=e,reports=reports,private_inventory=inventory,owner_pid_present=Path('/proc',sys.argv[1]).exists())))
'''
v=json.loads(ops.remote([ops.PY,'-B','-c',code,str(launch['pid'])],timeout=30));assert not v['owner_pid_present']
for n,s in v['reports'].items():assert n in ('summary.json','verification.json','RUN_COMPLETE');(r/n).write_bytes(s.encode())
for n,x in [('execution.json',v['execution']),('private-output-inventory.json',v['private_inventory'])]:(r/n).write_text(json.dumps(x,indent=2)+'\n')
if v['execution']['exit_code']==0:
    receipt=json.loads(v['reports']['RUN_COMPLETE'])
    for n in ('summary','verification'):assert hashlib.sha256(v['reports'][n+'.json'].encode()).hexdigest()==receipt[n+'_sha256']
    s=json.loads(v['reports']['summary.json'])
    print(json.dumps(dict(status=s['status'],decision=s['decision'],gates=s['gates'],elapsed_seconds=s['elapsed_seconds'],native_head_forwards=s['native_head_forwards'],
        error_advantage=s['error_advantage'],excess_correct_harm=s['excess_correct_harm'],
        assessment={k:{g:row[g] for g in ('error','correct')} for k,row in s['metrics']['assessment'].items()},private_arrays_downloaded=False)))
else:print(json.dumps({k:v['execution'][k] for k in ('exit_code','stdout','stderr')}))
