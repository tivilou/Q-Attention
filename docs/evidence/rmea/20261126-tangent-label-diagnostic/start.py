"""Stage one fixed diagnostic and launch the accepted host-resident wrapper."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

r = Path(__file__).resolve().parent
p = Path('C:/Users/14630/AppData/Local/research-infrastructure/qattn-2080ti2-38434-20261122/tangent-label-20261126-v2')
sys.path.insert(0, str(p.parent))
import ops


def sha(b):
    return hashlib.sha256(b).hexdigest()


def save(path, value):
    path.write_bytes((json.dumps(value, indent=2) + '\n').encode())


assert not (r / 'launch.json').exists(), 'Reconcile the exact existing attempt'
a = json.loads((p / 'allocation.private.json').read_bytes())
request = json.loads((p / 'request.json').read_bytes())
assert a['state'] == 'allocated' and a['host_id'] == ops.SERVER and len(a['gpu_uuids']) == 1
assert a['runner_attempt_id'] == request['runner_attempt_id']
review = json.loads((r / 'local-preflight.json').read_bytes())
assert review['status'] == 'PASS_LOCAL_PREFLIGHT' and not review['external_research_review']
for n, h in review['files'].items():
    assert sha((r / n).read_bytes()) == h
cfg = json.loads((r / 'design.json').read_bytes())
files = {n: (r / n).read_text('utf8') for n in tuple(cfg['code_hashes']) + ('design.json', 'protocol.md')}
for n, h in cfg['code_hashes'].items():
    assert sha(files[n].encode()) == h
wrapper = Path('E:/Claude.Code/Paper.Writing/Q-Attention/.codex-artifacts/relative-return-training-20261122/host_run.py').read_bytes()
assert sha(wrapper) == '238f7e68ea94250ecd786e64c269c8006c7225e40e638879515e545bf7795662'
files['host_run.py'] = wrapper.decode()
host = '/hy-tmp/Q-Attention/control/tangent-label-diagnostic-20261126-v2'
argv = ['/hy-tmp/Q-Attention/env/bin/python', '-B', host + '/runner.py', '--root', '/hy-tmp/Q-Attention',
        '--design', host + '/design.json', '--output', host + '/output', '--progress', host + '/progress.json',
        '--attempt', a['runner_attempt_id']]
launch = {'allocation': a, 'argv': argv, 'file_hashes': {n: sha(s.encode()) for n, s in files.items()},
          'input_hashes': cfg['input_hashes']}
files['launch.json'] = json.dumps(launch, indent=2) + '\n'
code = '''from pathlib import Path
import hashlib,json,sys
v=json.load(sys.stdin);r=Path(v['host']);a=v['allocation'];permit=Path(a['permit_path'])
assert hashlib.sha256(permit.read_bytes()).hexdigest()==a['permit_sha256']
q=json.loads(permit.read_bytes())
for k in ('allocation_id','generation','project_id','host_id','run_id','runner_attempt_id'):assert q[k]==a[k]
assert not r.exists();r.mkdir()
for n,s in v['files'].items():
 f=r/n;assert f.resolve().is_relative_to(r.resolve());f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(s.encode())
print(json.dumps({'staged':True}))
'''
assert json.loads(ops.remote([ops.PY, '-B', '-c', code], data=json.dumps({'host': host, 'allocation': a, 'files': files}).encode(), timeout=30))['staged']
ack = {k: a[k] for k in ('allocation_id', 'generation', 'project_id', 'host_id', 'run_id', 'runner_attempt_id')}
ack.update(permit_sha256=a['permit_sha256'], progress_path=host + '/progress.json', return_path=host + '/resource-return.json')
save(p / 'ack.json', ack)
q = subprocess.run(['E:/Dir/ClaudeCode/.local/bin/researchctl.cmd', 'allocation-ack', '--allocation', a['allocation_id'],
                    '--record', str(p / 'ack.json')], capture_output=True, timeout=30)
assert q.returncode == 0, q.stderr.decode()[-500:]
(p / 'ack-response.private.json').write_bytes(q.stdout)
code = '''from pathlib import Path
import json,subprocess,sys
r=Path(sys.argv[1]);assert not (r/'host.stdout.log').exists()
with (r/'host.stdout.log').open('xb') as out,(r/'host.stderr.log').open('xb') as err:
 p=subprocess.Popen(['/usr/local/miniconda3/bin/python','-B',str(r/'host_run.py')],cwd=r,stdout=out,stderr=err,start_new_session=True)
print(json.dumps({'pid':p.pid,'start_ticks':Path('/proc',str(p.pid),'stat').read_text().split(') ')[1].split()[19], 'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}))
'''
out = json.loads(ops.remote([ops.PY, '-B', '-c', code, host], timeout=30))
out.update(allocation_id=a['allocation_id'], generation=a['generation'], runner_attempt_id=a['runner_attempt_id'])
save(r / 'launch.json', out)
print(json.dumps(out))
