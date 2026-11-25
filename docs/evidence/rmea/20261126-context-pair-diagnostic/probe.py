from pathlib import Path
import hashlib,json,sys
r=Path(__file__).resolve().parent
sys.path.insert(0,'C:/Users/14630/AppData/Local/research-infrastructure/qattn-2080ti2-38434-20261122');import ops
spec=json.loads((r/'identity-spec.json').read_bytes())
assert hashlib.sha256(Path(spec['prediction']['path']).read_bytes()).hexdigest()==spec['prediction']['sha256']
for p in spec['local_pins']:assert hashlib.sha256(Path(p['path']).read_bytes()).hexdigest()==p['sha256']
# ops.resolve verifies concrete project-role, key authentication and strict known-host fingerprint.
remote=json.loads(ops.remote([ops.PY,'-B','-c',(r/'probe_remote.py').read_text('utf8')],data=json.dumps(spec).encode(),timeout=80))
(r/'identity-probe.json').write_text(json.dumps(remote,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=remote['status'],checked_files=len(remote['checked_files']),errors=remote['errors'],seconds=remote['seconds'],
 archive_fields={n:list(x) for n,x in remote['archives'].items()},large_input_full_hash_pending=remote['large_input_full_hash_pending'])))
assert remote['status']=='PASS_EXISTING_IDENTITY_PROBE'
