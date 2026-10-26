"""Owner control machine: authorized SSH RPC, reconciliation, then admission."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time

from queue_store import Store,review_gate

@contextmanager
def leader(path):
    path=Path(path)
    with open(path,'a+b') as stream:
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream,fcntl.LOCK_UN)

def serverctl_json(arguments,timeout=20):
    executable=shutil.which('serverctl')
    if not executable:raise ValueError('serverctl required')
    argv=[executable,*arguments,'--json']
    if os.name=='nt':
        quote=lambda x:"'"+x.replace("'","''")+"'"
        argv=['powershell.exe','-NoProfile','-NonInteractive','-Command','& '+' '.join(quote(x) for x in argv)]
    result=subprocess.run(argv,capture_output=True,timeout=timeout)
    if result.returncode:raise ValueError('project/node authorization unavailable')
    return json.loads(result.stdout)

def resolved(project,server_id):
    rows=serverctl_json(['resolve',project,'--role','development','--server',server_id])
    selected=[x for x in rows if x['id']==server_id and x.get('auth')=='key' and x.get('host_key_fingerprint','').startswith('SHA256:')]
    if len(selected)!=1:raise ValueError('explicit verified project node authorization required')
    return selected[0]

def fresh_health(row,max_age=180,now=None):
    now=time.time() if now is None else now
    checked=row.get('checked_at')
    return row.get('status')=='online' and isinstance(checked,(int,float)) and 0<=now-checked<=max_age

def fleet_status(config):
    if not config.get('health',{}).get('enabled'):return dict(enabled=False,servers=[])
    try:return dict(serverctl_json(['health','--max-age',str(config['health'].get('max_age_seconds',180))],timeout=10),enabled=True)
    except Exception:return dict(enabled=True,available=False,reason='health_cache_unavailable',servers=[])

class HealthMonitor:
    """Fleet network I/O must never delay active attempt lease renewal."""
    def __init__(self,config):
        self.config=config;self.pool=ThreadPoolExecutor(max_workers=1);self.future=None;self.next_refresh=0;self.rows={}
    def poll(self):
        if self.future is not None and self.future.done():
            try:
                result=self.future.result()
                self.rows={row['id']:row for row in result['servers']}
            except Exception:pass  # Old observations age out; never manufacture healthy state.
            self.future=None
        if self.future is None and time.monotonic()>=self.next_refresh:
            self.future=self.pool.submit(serverctl_json,['health','--refresh','--fleet'],120)
            self.next_refresh=time.monotonic()+max(10,self.config.get('health',{}).get('interval_seconds',60))
        return self.rows
    def close(self):self.pool.shutdown(wait=True)

def write_worker_health(store,rows):
    target=store.path.parent/'worker-health.json'
    fd,temporary=tempfile.mkstemp(prefix='worker-health-',dir=target.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            json.dump(dict(schema='research.worker-health.v1',nodes=rows),stream)
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,target)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)

class SSH:
    def __init__(self,config):self.config=config
    def __call__(self,node_id,message,project=None):
        node=self.config['nodes'][node_id]
        path=self.config['projects'][project]['path'] if project else node['control_project']
        server=resolved(path,node['server_id'])
        command=[node['python'],node['worker'],'rpc','--config',node['config']]
        argv=['ssh','-T','-i',server['identity_file'],'-p',str(server['port']),'-o','BatchMode=yes',
            '-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=8',server['user']+'@'+server['host'],shlex.join(command)]
        # This RPC deliberately uses stdin; -n would discard the JSON request.
        result=subprocess.run(argv,input=json.dumps(message).encode(),capture_output=True,timeout=20)
        if result.returncode:raise RuntimeError('worker RPC failed; execution may be uncertain')
        return json.loads(result.stdout)

def tick(store,config,rpc,health=None):
    store.heartbeat(os.getpid())
    cancelled={row['id'] for row in store.tasks() if row['state']=='cancel_requested'}
    for row in store.attempts():
        request=row['request']
        try:
            evidence=rpc(row['node'],dict(action='cancel' if row['task'] in cancelled else 'renew',attempt=row['id'],token=row['token']),request['project'])
            store.observe(request,evidence)
        except Exception:
            store.observe(request,dict(state='unknown',token=request['token']))
    probes={};observations={}
    for node_id,node in config['nodes'].items():
        if not node.get('enabled'):continue
        if config.get('health',{}).get('enabled') and not fresh_health((health or {}).get(node.get('server_id'),{}),config['health'].get('max_age_seconds',180)):
            observations[node_id]=dict(status='blocked_health',reason=(health or {}).get(node.get('server_id'),{}).get('reason','health_unknown'),checked_at=time.time())
            continue
        try:
            probes[node_id]=rpc(node_id,dict(action='describe'))
            observations[node_id]=dict(status='online',checked_at=time.time(),resources=probes[node_id])
        except Exception:
            observations[node_id]=dict(status='unavailable',reason='worker_rpc_failed',checked_at=time.time())
    write_worker_health(store,observations)
    used=[]
    while True:
        request=store.claim(config,probes,used)
        if request is None:break
        used.append(request['node'])
        try:
            evidence=rpc(request['node'],dict(action='start',request=request),request['project'])
            store.observe(request,evidence)
        except Exception:
            store.observe(request,dict(state='unknown',token=request['token']))
    store.heartbeat(os.getpid())

def serve(config,once=False):
    review_gate()
    store=Store(config['database']);directory=Path(config['database']).parent
    with leader(directory/'controller.lock'):
        stop=directory/'stop'
        if stop.exists():stop.unlink()
        rpc=SSH(config)
        monitor=HealthMonitor(config) if config.get('health',{}).get('enabled') else None
        try:
            while True:
                tick(store,config,rpc,monitor.poll() if monitor else None)
                if once or stop.exists():break
                # Small stop-aware waits; no resource release or automatic adoption on exit.
                until=time.monotonic()+config.get('poll_seconds',5)
                while time.monotonic()<until and not stop.exists():time.sleep(.2)
        finally:
            if monitor:monitor.close()

