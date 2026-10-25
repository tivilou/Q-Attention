"""POSIX worker RPC and independent expiring-lease subprocess supervisor."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from queue_store import ACTIVE,recipe_check,review_gate

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name)
    try:
        with os.fdopen(fd,'w') as f:json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)
@contextmanager
def lock(path,nonblocking=False):
    import fcntl
    with open(path,'a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|(fcntl.LOCK_NB if nonblocking else 0))
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)
def clean_env():
    blocked=('SECRET','TOKEN','PASSWORD','CREDENTIAL','API_KEY','AUTH','PRIVATE_KEY','SESSION')
    env={k:v for k,v in os.environ.items() if not any(x in k.upper() for x in blocked) and not k.startswith(('AWS_','TENCENTCLOUD_','ALIPAN_'))}
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    return env
def boot_id():return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args],env=clean_env(),stderr=subprocess.DEVNULL,timeout=15,text=True).rstrip('\n')
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def ram_available():
    info=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    available=int(info['MemAvailable'].split()[0])*1024
    maximum=Path('/sys/fs/cgroup/memory.max');current=maximum.with_name('memory.current')
    if maximum.is_file() and maximum.read_text().strip()!='max':available=min(available,max(0,int(maximum.read_text())-int(current.read_text())))
    return max(0,available//(1024**2)-32)

class Node:
    def __init__(self,config):
        if os.name!='posix':raise ValueError('workers require POSIX')
        self.file=Path(config).resolve(strict=True);self.config=read(self.file)
        self.root=Path(self.config['state_root']).resolve();self.root.mkdir(parents=True,exist_ok=True)
    def directory(self,attempt):
        import re
        if not isinstance(attempt,str) or not re.fullmatch('[0-9a-f]{32}',attempt):raise ValueError('invalid attempt')
        path=self.root/attempt
        if path.is_symlink():raise ValueError('symlinked attempt')
        return path
    def describe(self):
        import shutil
        gpus=[]
        try:
            result=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
            if result.returncode==0:
                for line in result.stdout.splitlines():
                    gpu,free=line.split(',');gpus.append(dict(uuid=gpu.strip(),free_mb=int(free.strip())))
        except (OSError,ValueError,subprocess.TimeoutExpired):pass
        return dict(schema='research.node.v1',node=self.config['id'],boot_id=boot_id(),
            available_ram_mb=ram_available(),available_disk_mb=shutil.disk_usage(self.root).free//(1024**2),gpus=gpus)
    def terminal(self,request,state,**extra):
        return dict(schema='research.receipt.v1',attempt=request['attempt'],task=request['task'],token=request['token'],
            node=self.config['id'],revision=request['revision'],state=state,quiescent=True,finished=time.time(),**extra)
    def status(self,attempt,token):
        directory=self.directory(attempt)
        if not (directory/'request.json').exists():return dict(state='unknown',token=token,reason='no durable launch record')
        req=read(directory/'request.json')
        if req['token']!=token:raise ValueError('stale token')
        state=read(directory/'state.json')
        if state['state'] in ACTIVE and req['boot_id']!=boot_id():
            state=self.terminal(req,'node_rebooted');atomic(directory/'state.json',state)
        if state['state'] in ('launching','running') and time.time()-state.get('heartbeat',state.get('created',0))>2*self.config['lease_seconds']:
            return dict(state='unknown',token=token,reason='supervisor heartbeat stale')
        return state
    def renew(self,attempt,token):
        directory=self.directory(attempt)
        with lock(self.root/'.node.lock'):
            state=self.status(attempt,token)
            if state['state'] in ('launching','running'):
                atomic(directory/'lease.json',dict(token=token,until=time.monotonic()+self.config['lease_seconds']))
            return state
    def cancel(self,attempt,token):
        directory=self.directory(attempt)
        with lock(self.root/'.node.lock'):
            state=self.status(attempt,token)
            if state['state'] in ACTIVE:atomic(directory/'cancel.json',dict(token=token))
            return state
    def start(self,request):
        directory=self.directory(request['attempt'])
        with lock(self.root/'.node.lock'):
            if directory.exists():
                if read(directory/'request.json')!=request:raise ValueError('attempt identity conflict')
                return self.status(request['attempt'],request['token'])
            if request['node']!=self.config['id'] or request['boot_id']!=boot_id():raise ValueError('node changed before dispatch')
            binding=self.config['projects'][request['project']]
            recipe=recipe_check(request['recipe'])
            if recipe!=binding['recipes'][request['recipe_id']] or recipe['kind'] not in binding['allowed_kinds']:raise ValueError('unregistered worker recipe/policy')
            # Independent admission prevents another controller or launch race from oversubscribing this node.
            active=[]
            for file in self.root.glob('*/request.json'):
                req=read(file);state=self.status(req['attempt'],req['token'])
                if state['state'] in ACTIVE:active.append(req)
            cap=self.config['capacity'];r=recipe['resources'];probe=self.describe()
            if any(sum(a['recipe']['resources'][k] for a in active)+r[k]>cap[k] for k in ('cpu','ram_mb')):raise ValueError('node capacity reserved')
            if r['ram_mb']>probe['available_ram_mb'] or r['disk_mb']>probe['available_disk_mb']:raise ValueError('node memory/disk pressure')
            busy={g for a in active for g in a['gpu_ids']};free={g['uuid']:g['free_mb'] for g in probe['gpus']}
            ids=request['gpu_ids']
            if len(set(ids))!=len(ids) or len(ids)!=r['gpus'] or any(g in busy or free.get(g,-1)<r['gpu_mb'] for g in ids):raise ValueError('GPU admission failed')
            directory.mkdir(exist_ok=False)
            atomic(directory/'request.json',request)
            atomic(directory/'lease.json',dict(token=request['token'],until=time.monotonic()+self.config['lease_seconds']))
            atomic(directory/'state.json',dict(state='launching',token=request['token'],created=time.time()))
            # Durable identity precedes launch. Lost acknowledgement can only query this same attempt.
            with open(os.devnull,'wb') as log:
                subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run','--config',str(self.file),'--attempt',request['attempt']],
                    stdin=subprocess.DEVNULL,stdout=log,stderr=log,env=clean_env(),start_new_session=True,close_fds=True)
            return self.status(request['attempt'],request['token'])
    def run(self,attempt):
        directory=self.directory(attempt);request=read(directory/'request.json')
        with lock(directory/'.runner.lock',True):
            if read(directory/'state.json')['state'] not in ACTIVE:return
            binding=self.config['projects'][request['project']];repo=Path(binding['repo']).resolve(strict=True)
            recipe=request['recipe'];out=directory/'output';out.mkdir(exist_ok=False)
            child=None;state='failed';code=None;outputs={};error=None
            try:
                if request['boot_id']!=boot_id() or git(repo,'rev-parse','HEAD')!=request['revision'] or git(repo,'status','--porcelain'):raise ValueError('source/boot differs')
                argv=[x.replace('{python}',binding['python']).replace('{output}',str(out)) for x in recipe['command']]
                env=clean_env();env['CUDA_VISIBLE_DEVICES']=','.join(request['gpu_ids']);env['RESEARCH_OUTPUT_DIR']=str(out)
                atomic(directory/'command.json',dict(argv=argv,cwd=str(repo),revision=request['revision'],gpu_ids=request['gpu_ids']))
                start=time.monotonic()
                with open(directory/'stdout.log','xb') as stdout,open(directory/'stderr.log','xb') as stderr:
                    child=subprocess.Popen(argv,cwd=repo,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
                    while True:
                        lease=read(directory/'lease.json')
                        if (directory/'cancel.json').exists() and read(directory/'cancel.json')['token']==request['token']:state='cancelled';break
                        if lease['token']!=request['token'] or time.monotonic()>lease['until']:state='lease_expired';break
                        if time.monotonic()-start>recipe['timeout']:state='timed_out';break
                        code=child.poll()
                        if code is not None:state='complete' if code==0 else 'failed';break
                        atomic(directory/'state.json',dict(state='running',token=request['token'],heartbeat=time.time(),pid=child.pid,boot_id=boot_id()))
                        time.sleep(.2)
            except Exception as exc:state='failed';error=type(exc).__name__
            finally:
                if child:
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        atomic(directory/'state.json',dict(state='unknown',token=request['token'],reason='child termination unconfirmed'));return
                    try:os.killpg(child.pid,0)
                    except ProcessLookupError:pass
                    else:
                        atomic(directory/'state.json',dict(state='unknown',token=request['token'],reason='descendant group not quiescent'));return
                if state=='complete':
                    try:
                        if git(repo,'rev-parse','HEAD')!=request['revision'] or git(repo,'status','--porcelain'):raise ValueError('job modified source')
                        for name in recipe['outputs']:
                            file=out/name
                            if not file.is_file() or file.is_symlink() or not file.resolve().is_relative_to(out.resolve()):raise ValueError('required output missing/nonregular')
                            outputs[name]=dict(sha256=sha(file),bytes=file.stat().st_size)
                    except Exception as exc:state='failed';error=type(exc).__name__
                atomic(directory/'state.json',self.terminal(request,state,returncode=code,outputs=outputs,error_class=error))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['rpc','run']);parser.add_argument('--config',required=True);parser.add_argument('--attempt')
    args=parser.parse_args();review_gate();os.umask(0o077);node=Node(args.config)
    if args.action=='run':node.run(args.attempt);return
    value=json.load(sys.stdin);action=value['action']
    if action=='describe':result=node.describe()
    elif action=='start':result=node.start(value['request'])
    elif action=='status':result=node.status(value['attempt'],value['token'])
    elif action=='renew':result=node.renew(value['attempt'],value['token'])
    elif action=='cancel':result=node.cancel(value['attempt'],value['token'])
    else:raise ValueError('unknown RPC')
    print(json.dumps(result))
if __name__=='__main__':main()
