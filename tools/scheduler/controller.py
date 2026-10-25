"""Owner control machine: authorized SSH RPC, reconciliation, then admission."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
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

def resolved(project,server_id):
    executable=shutil.which('serverctl')
    if not executable:raise ValueError('serverctl required')
    argv=[executable,'resolve',project,'--role','development','--json']
    if os.name=='nt':
        quote=lambda x:"'"+x.replace("'","''")+"'"
        argv=['powershell.exe','-NoProfile','-NonInteractive','-Command','& '+' '.join(quote(x) for x in argv)]
    result=subprocess.run(argv,capture_output=True,timeout=20)
    if result.returncode:raise ValueError('project/node authorization unavailable')
    rows=json.loads(result.stdout)
    selected=[x for x in rows if x['id']==server_id and x.get('auth')=='key' and x.get('host_key_fingerprint','').startswith('SHA256:')]
    if len(selected)!=1:raise ValueError('explicit verified project node authorization required')
    return selected[0]

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

def tick(store,config,rpc):
    store.heartbeat(os.getpid())
    cancelled={row['id'] for row in store.tasks() if row['state']=='cancel_requested'}
    for row in store.attempts():
        request=row['request']
        try:
            evidence=rpc(row['node'],dict(action='cancel' if row['task'] in cancelled else 'renew',attempt=row['id'],token=row['token']),request['project'])
            store.observe(request,evidence)
        except Exception:
            store.observe(request,dict(state='unknown',token=request['token']))
    probes={}
    for node_id,node in config['nodes'].items():
        if not node.get('enabled'):continue
        try:probes[node_id]=rpc(node_id,dict(action='describe'))
        except Exception:pass
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
        while True:
            tick(store,config,rpc)
            if once or stop.exists():break
            # Small stop-aware waits; no resource release or automatic adoption on exit.
            until=time.monotonic()+config.get('poll_seconds',5)
            while time.monotonic()<until and not stop.exists():time.sleep(.2)

