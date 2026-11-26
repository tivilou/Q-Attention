from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,time
r=Path(__file__).resolve().parent
name=sys.argv[1];cmd=[sys.executable,str(r/name)]
env=os.environ.copy();env['PYTHONUTF8']='1';env['PYTHONDONTWRITEBYTECODE']='1'
env['MKL_THREADING_LAYER']='SEQUENTIAL';env['OMP_NUM_THREADS']='1';env['GIT_NO_LAZY_FETCH']='1'
start=time.perf_counter();q=subprocess.run(cmd,cwd=r,capture_output=True,env=env,timeout=600)
out=dict(command=cmd,code_sha256=hashlib.sha256((r/name).read_bytes()).hexdigest(),
         stdout=q.stdout.decode('utf8','replace'),stderr=q.stderr.decode('utf8','replace'),
         exit_status=q.returncode,elapsed_seconds=time.perf_counter()-start,recorded_at=datetime.now(timezone.utc).isoformat())
(r/(Path(name).stem+'-execution.json')).write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode())
print(out['stdout'],end='');print(out['stderr'],end='',file=sys.stderr)
raise SystemExit(q.returncode)
