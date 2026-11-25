from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
import os

r=Path(__file__).resolve().parent
helper='E:/Claude.Code/本地助手/ai-skills/research/paper-fulltext-acquisition/scripts/fulltext.py'
env=os.environ.copy();env['PYTHONUTF8']='1'

def sha(b):return hashlib.sha256(b).hexdigest()
def write(n,x):(r/n).write_bytes((json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode())
def run(name,cmd):
    q=subprocess.run(cmd,capture_output=True,env=env,cwd=r,timeout=40)
    out=dict(command=cmd,stdout=q.stdout.decode('utf8','replace'),stderr=q.stderr.decode('utf8','replace'),
             exit_status=q.returncode,recorded_at=datetime.now(timezone.utc).isoformat())
    if Path(cmd[1]).is_file():out['code_sha256']=sha(Path(cmd[1]).read_bytes())
    write(name,out)
    if q.returncode:raise RuntimeError(out['stderr'])
    print(out['stdout'],end='')

for key,version,evidence in [
    ('fulltexts/8fcb56c452e49c5e','conference_proceedings_2016','Official NIPS2016 directory and paper landing, title and five authors matched PDF first page; nine-page proceedings original inspected.'),
    ('arxiv-fulltexts/61034008b4c371d1','1610.06258v1','Official exact-ID API title/five authors match PDF first page; PDF explicitly arXiv:1610.06258v1 [stat.ML]20Oct2016. Appendix pages10-11 inspected; not whole-version equivalence.')]:
    p=r/key/'receipt.json';backup=p.with_name('receipt.before-identity-verification.json')
    assert not backup.exists();backup.write_bytes(p.read_bytes())
    run(key.split('/')[0]+'-identity-execution.json',[sys.executable,helper,'verify','--receipt',str(p),
        '--title','Using Fast Weights to Attend to the Recent Past','--version',version,'--evidence',evidence])
run('report-build-execution.json',[sys.executable,str(r/'build_report.py')])
record=r/'reading-record.json';h=sha(record.read_bytes())
for key,status in [('fulltexts/8fcb56c452e49c5e','read_for_scope'),('arxiv-fulltexts/61034008b4c371d1','partially_read_for_scope')]:
    p=r/key/'receipt.json';x=json.loads(p.read_bytes());x.update(reading=status,reading_record=str(record),reading_record_sha256=h)
    p.write_bytes((json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(dict(status='SOURCE_IDENTITY_AND_READING_RECORDS_SAVED',task_execution=0)))
