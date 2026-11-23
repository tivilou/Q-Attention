"""Host-resident bounded project task under an existing nonexpiring allocation."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
r=Path(__file__).parent
v=json.loads((r/'launch.json').read_bytes());a=v['allocation']
def save(p,value):
 t=p.with_suffix('.pending');t.write_text(json.dumps(value,indent=2));os.replace(t,p)
permit=json.loads(Path(a['permit_path']).read_bytes())
for k in ('allocation_id','generation','project_id','host_id','run_id','runner_attempt_id'):assert permit[k]==a[k]
assert hashlib.sha256(Path(a['permit_path']).read_bytes()).hexdigest()==a['permit_sha256']
for n,h in v['file_hashes'].items():assert hashlib.sha256((r/n).read_bytes()).hexdigest()==h
start=time.time();pid=os.getpid();ticks=Path('/proc',str(pid),'stat').read_text().split(') ')[1].split()[19];boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a['gpu_uuids'][0],CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR='/hy-tmp')
save(r/'progress.json',dict(schema='research.progress.v1',attempt=a['runner_attempt_id'],phase='starting_geometry_probe',current=0,total=None,unit='stage',observed_at=time.time(),heartbeat_at=time.time()))
x=subprocess.run(v['argv'],cwd='/hy-tmp/Q-Attention/source',env=env,capture_output=True,text=True)
out=dict(argv=v['argv'],cwd='/hy-tmp/Q-Attention/source',inputs=v['input_hashes'],stdout=x.stdout,stderr=x.stderr,exit_code=x.returncode,started_at=start,finished_at=time.time(),runner_attempt_id=a['runner_attempt_id'],file_hashes=v['file_hashes'])
save(r/'execution.json',out)
status='completed' if x.returncode==0 and (r/'output/RUN_COMPLETE').is_file() else 'failed'
ev={k:a[k] for k in ('allocation_id','generation','project_id','host_id','run_id','runner_attempt_id')};ev.update(schema='research.resource-return.v1',event_sequence=1,observed_at=datetime.now(timezone.utc).isoformat(),campaign_status=status,all_jobs_terminal=True,quiescent=True,no_remaining_consumers=True,cleanup_status='not_needed',resources_reusable=True,owner_pid=pid,owner_pgid=os.getpgrp(),owner_start_ticks=ticks,boot_id=boot)
save(r/'resource-return.json',ev)
if status=='failed':save(r/'progress.json',dict(schema='research.progress.v1',attempt=a['runner_attempt_id'],phase='failed',current=0,total=None,unit='stage',observed_at=time.time(),heartbeat_at=time.time(),reason='See exact execution record; no automatic retry'))
print(json.dumps(dict(exit_code=x.returncode,status=status)),flush=True)
