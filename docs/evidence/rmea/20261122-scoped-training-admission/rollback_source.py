"""Restore only these three admission files in an explicitly selected idle copy."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,tempfile
r=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def rollback(root):
 if (r/'source-snapshot.json').exists():
  snapshot=json.loads((r/'source-snapshot.json').read_bytes())
 else:
  git=lambda *a:subprocess.check_output(['git','-C',str(root),*a],stderr=subprocess.PIPE)
  snapshot=json.loads(git('show','1c2f39e032d81768a38be3d45f6de8221c338f52:docs/evidence/rmea/20261122-scoped-training-admission/scoped-admission.json'))
  snapshot=dict(staged=['baselines/retacred/'+n+'.py' for n in ('rmea_frozen_training','rmea_protected_context_protocol','rmea_scoped_admission')],source_hashes=snapshot['source_hashes'])
 root=Path(root).resolve()
 names=snapshot['staged']
 assert len(names)==3
 for name in names:
  p=root/name
  assert p.is_file() and sha(p.read_bytes())==snapshot['source_hashes'][name],name
 for name in names:
  p=root/name;b=r/'before-source'/name
  if name!='baselines/retacred/rmea_scoped_admission.py':
   data=b.read_bytes() if b.is_file() else subprocess.check_output(['git','-C',str(root),'show','84348e1cbe0d76fad9da8cdc6aa01d7a01d3b700:'+name],stderr=subprocess.PIPE)
   fd,tmp=tempfile.mkstemp(prefix=p.name+'.rollback-',dir=p.parent)
   try:
    with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
   finally:
    if os.path.exists(tmp):os.unlink(tmp)
   assert p.read_bytes()==data
  else:
   assert name=='baselines/retacred/rmea_scoped_admission.py'
   p.unlink();assert not p.exists()
 return dict(verified=True,restored_files=2,removed_added_file=1,root=str(root))
if __name__=='__main__':
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--root',required=True,type=Path)
 print(json.dumps(rollback(a.parse_args().root)))
