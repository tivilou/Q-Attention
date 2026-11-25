from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,sys
r=Path(__file__).resolve().parent
command=[sys.executable,'-B',str(r/'verify.py')]
p=subprocess.run(command,cwd=r,capture_output=True,timeout=45)
out=dict(command=command,cwd=str(r),code_sha256=hashlib.sha256((r/'verify.py').read_bytes()).hexdigest(),
         recorded_at=datetime.now(timezone.utc).isoformat(),stdout=p.stdout.decode('utf8','replace'),
         stderr=p.stderr.decode('utf8','replace'),exit_status=p.returncode)
(r/'verification-execution.json').write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode())
print(out['stdout'],end='');print(out['stderr'],end='',file=sys.stderr)
raise SystemExit(p.returncode)
