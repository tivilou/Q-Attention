"""Owner-only, run-scoped COS publisher. Reads private config, never exports it.

The worker waits at each sealed request. Only this process has the COS key and
private-index Git identity. Payloads travel worker->COS, not to owner disk.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import time


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''): h.update(chunk)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path); temporary=path.with_suffix('.pending')
    with temporary.open('w',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


class Publisher:
    def __init__(self,config):
        from qcloud_cos import CosConfig,CosS3Client
        self.config=config; c=json.loads(Path(config['credential_file']).read_text(encoding='utf-8-sig'))
        self.client=CosS3Client(CosConfig(Region=config['region'],SecretId=c['SecretId'],SecretKey=c['SecretKey'],Scheme='https',VerifySSL=True,Timeout=90))
        self.index=Path(config['index_worktree']);self.verified=Path(config['status_file']).with_name('verified-objects.json');self.root=PurePosixPath(config['remote_run'])
        if not self.root.is_absolute() or '..' in self.root.parts: raise ValueError('Absolute run identity required')
        if self.root.name != config['run_id']: raise ValueError('Run identity differs')
        if not config['prefix'].startswith('projects/q-attention/formal-runs/'): raise ValueError('Project prefix required')

    def remote(self,code,payload=None,timeout=120):
        command=shlex.join([self.config['remote_python'],'-B','-c',code])
        r=subprocess.run(self.config['ssh_argv']+[command],input=json.dumps(payload).encode() if payload is not None else b'',capture_output=True,timeout=timeout)
        if r.returncode: raise RuntimeError('Worker publisher RPC failed: exit '+str(r.returncode))
        return json.loads(r.stdout)

    def requests(self):
        return self.remote("""import json,pathlib,sys
r=pathlib.Path(json.load(sys.stdin));out=[]
for f in sorted((r/'cos-channel').glob('*/request.json')):
 if not (f.parent/'ack.json').exists():out.append(dict(relative=f.relative_to(r).as_posix(),sha256=__import__('hashlib').sha256(f.read_bytes()).hexdigest(),request=json.loads(f.read_text())))
print(json.dumps(out))""",str(self.root))

    def validate(self,request):
        if (request.get('schema')!='q-rmea.cos-request.v1' or request['run_id']!=self.config['run_id']
                or request['identity']!=self.config['identity'] or request['total']!=self.config['total']
                or not isinstance(request['done'],int) or not 0<=request['done']<=request['total']):
            raise ValueError('Run request contract differs')
        generation=request['generation']
        if generation not in ['000-preparation']+['%03d-progress'%n for n in range(10,101,10)]:
            raise ValueError('Undeclared milestone')
        files=request['files']
        if not files or len(files)>48: raise ValueError('Object inventory bound exceeded')
        total=0
        for rel,row in files.items():
            path=PurePosixPath(rel)
            if path.is_absolute() or '..' in path.parts or path.parent!=PurePosixPath('cos-channel')/generation:
                raise ValueError('Request path escapes generation')
            if generation=='000-preparation':
                import re
                if not re.fullmatch(r'preparation\.\d{3}\.tarpart',path.name):raise ValueError('Unexpected preparation member')
            elif path.name not in ('latest.pt','best.pt'):raise ValueError('Unexpected checkpoint role')
            if not isinstance(row['bytes'],int) or not 0<row['bytes']<=512*1024**2:
                raise ValueError('Object byte bound exceeded')
            if len(row['sha256'])!=64 or any(c not in '0123456789abcdef' for c in row['sha256']):raise ValueError('Invalid digest')
            total+=row['bytes']
        if generation!='000-preparation' and (len(files)!=2 or total>16*1024**2):raise ValueError('Checkpoint inventory bound exceeded')
        if total>21*1024**3:raise ValueError('Preparation budget exceeded')

    def object(self,rel,row):
        from qcloud_cos.cos_exception import CosServiceError
        key=self.config['prefix'].rstrip('/')+'/'+rel
        bucket=self.config['bucket'];head=None
        try: head=self.client.head_object(Bucket=bucket,Key=key)
        except CosServiceError as e:
            if e.get_status_code()!=404:raise
        if head is None:
            headers={'x-cos-server-side-encryption':'AES256','x-cos-meta-sha256':row['sha256']}
            url=self.client.get_presigned_url(Bucket=bucket,Key=key,Method='PUT',Expired=1800,Params={},Headers=headers)
            result=self.remote("""import hashlib,http.client,json,pathlib,ssl,sys,urllib.parse
x=json.load(sys.stdin);p=pathlib.Path(x['path'])
if p.is_symlink() or not p.is_file() or p.stat().st_size!=x['bytes']:raise ValueError('Sealed source differs')
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1048576),b''):h.update(b)
if h.hexdigest()!=x['sha256']:raise ValueError('Sealed source hash differs')
u=urllib.parse.urlsplit(x['url'])
if u.scheme!='https':raise ValueError('HTTPS required')
c=http.client.HTTPSConnection(u.hostname,u.port or 443,timeout=300,context=ssl.create_default_context())
try:
 with p.open('rb') as f:c.request('PUT',u.path+'?'+u.query,body=f,headers=dict(x['headers'],**{'Content-Length':str(x['bytes'])}))
 response=c.getresponse();response.read(4096)
 if response.status not in (200,201):raise RuntimeError('COS upload HTTP '+str(response.status))
 print(json.dumps(dict(status=response.status)))
finally:c.close()""",dict(path=str(self.root/rel),url=url,headers=headers,**row),timeout=900)
            if result['status'] not in (200,201):raise ValueError('Upload failed')
            head=self.client.head_object(Bucket=bucket,Key=key)
        if (int(head.get('Content-Length',-1))!=row['bytes'] or head.get('x-cos-meta-sha256')!=row['sha256']
                or not head.get('x-cos-version-id') or head['x-cos-version-id']=='null'):
            raise ValueError('Version-pinned uploaded identity differs')
        version=head['x-cos-version-id']
        record=dict(key=key,version_id=version,sha256=row['sha256'],bytes=row['bytes'])
        ledger=json.loads(self.verified.read_text(encoding='utf8')) if self.verified.exists() else {}
        if ledger.get(key)==record:return record
        response=self.client.get_object(Bucket=bucket,Key=key,VersionId=version)
        h=hashlib.sha256();count=0;body=response['Body'].get_raw_stream()
        try:
            for chunk in iter(lambda:body.read(1024**2),b''):
                count+=len(chunk)
                if count>row['bytes']:raise ValueError('Readback byte ceiling exceeded')
                h.update(chunk)
        finally:body.close()
        if count!=row['bytes'] or h.hexdigest()!=row['sha256']:raise ValueError('Pinned readback differs')
        ledger[key]=record;atomic_json(self.verified,ledger)
        return record

    def index_receipt(self,request,request_sha,objects):
        relative=Path('projects/q-attention/formal-runs')/self.config['run_id']/(request['generation']+'.json')
        path=self.index/relative;path.parent.mkdir(parents=True,exist_ok=True)
        value=dict(schema='q-rmea.cos-generation.v1',run_id=self.config['run_id'],generation=request['generation'],
            identity=request['identity'],done=request['done'],total=request['total'],request_sha256=request_sha,
            objects=objects,restoration='Restore pinned preparation parts, install pinned Git/environment/assets; restore latest/best and receipts; explicit --owner-managed --resume. Never automatically replay an unknown live attempt.')
        def git(*args):
            return subprocess.check_output(['git',*args],cwd=self.index,text=True,stderr=subprocess.DEVNULL).strip()
        # This run has a separate clean linked worktree in the one shared index.
        status=git('status','--porcelain','--untracked-files=all')
        if status and any(line[3:]!=relative.as_posix() for line in status.splitlines()):
            raise ValueError('Private index has unrelated changes')
        if path.exists():
            if json.loads(path.read_text(encoding='utf8'))!=value:raise ValueError('Index generation already differs')
        else:
            git('fetch','origin','main');git('rebase','origin/main');atomic_json(path,value)
        git('add','--',relative.as_posix())
        changed=git('diff','--cached','--name-only')
        if changed:
            if changed!=relative.as_posix():raise ValueError('Index staging differs')
            git('commit','-m','recovery: q-attention '+self.config['run_id']+' '+request['generation'])
        if json.loads(git('show','HEAD:'+relative.as_posix()))!=value:raise ValueError('Committed index bytes differ')
        try:git('push','origin','HEAD:refs/heads/main')
        except subprocess.CalledProcessError:
            git('fetch','origin','main');git('rebase','origin/main');git('push','origin','HEAD:refs/heads/main')
        commit=git('rev-parse','HEAD');git('fetch','origin','main')
        if json.loads(git('show','origin/main:'+relative.as_posix()))!=value:raise ValueError('Index remote readback differs')
        return commit

    def publish(self,observation):
        request=observation['request'];self.validate(request)
        objects={rel:self.object(rel,row) for rel,row in sorted(request['files'].items())}
        commit=self.index_receipt(request,observation['sha256'],objects)
        ack=dict(request_sha256=observation['sha256'],done=request['done'],objects=objects,index_commit=commit)
        if self.config.get('canary_restore'):
            ack['restore_urls']={rel:self.client.get_presigned_url(Bucket=self.config['bucket'],Key=row['key'],
                Method='GET',Expired=900,Params={'versionId':row['version_id']}) for rel,row in objects.items()}
        self.remote("""import hashlib,json,os,pathlib,sys
x=json.load(sys.stdin);root=pathlib.Path(x['root']);q=root/x['request']
if hashlib.sha256(q.read_bytes()).hexdigest()!=x['sha256']:raise ValueError('Request changed')
out=q.parent/'ack.json';tmp=out.with_suffix('.pending')
with tmp.open('w') as f:json.dump(x['ack'],f);f.flush();os.fsync(f.fileno())
os.replace(tmp,out);print(json.dumps({'acknowledged':True}))""",dict(root=str(self.root),request=observation['relative'],sha256=observation['sha256'],ack=ack))
        return dict(generation=request['generation'],done=request['done'],index_commit=commit,objects=len(objects))


    def terminal(self):
        if not self.config.get('attempt_root'):return None
        return self.remote("""import json,pathlib,sys
+p=pathlib.Path(json.load(sys.stdin))/'state.json'
+s=json.loads(p.read_text()) if p.exists() else {}
+print(json.dumps({k:s.get(k) for k in ('state','returncode','quiescent','revision','outputs')}))""".replace('\n+','\n'),self.config['attempt_root'])

    def deliver(self,terminal):
        if terminal['state']!='complete' or terminal['returncode']!=0 or terminal['quiescent'] is not True:
            raise ValueError('Execution failed or is not quiescent')
        if terminal['revision']!=self.config['revision']:raise ValueError('Execution revision differs')
        names=['summary.json','manifest.json','trace_validation.json','receipt.json','run_summary.json','data.sha256','data_counts.txt']
        report=str(self.root.parent/(self.root.name+'-safe-report'))
        files=self.remote("""import hashlib,json,pathlib,sys
+x=json.load(sys.stdin);r=pathlib.Path(x['report']);out={}
+if {f.name for f in r.iterdir()}!=set(x['names']):raise ValueError('Whitelist differs')
+for name in x['names']:
+ p=r/name
+ if p.is_symlink() or p.stat().st_size>2097152:raise ValueError('Unsafe report')
+ out[name]=p.read_text()
+print(json.dumps(out))""".replace('\n+','\n'),dict(report=report,names=names))
        for name,text in files.items():
            relative=PurePosixPath(report,name).relative_to(PurePosixPath(self.config['attempt_root'])/'output').as_posix()
            row=terminal['outputs'].get(relative,{})
            if row.get('sha256')!=hashlib.sha256(text.encode()).hexdigest():raise ValueError('Fenced report hash differs')
        code="""import hashlib,json,pathlib,sys
+x=json.load(sys.stdin);r=pathlib.Path(x['worktree']);sys.path.insert(0,str(r))
+from baselines.retacred import rmea_token_transport_protocol as p
+from baselines.retacred import rmea_token_transport_handoff as h
+assert p.ROOT.resolve()==r.resolve();p.review_gate(formal=True)
+target=r/'reports/q_rmea_token_transport'/x['run_id'];relative=target.relative_to(r).as_posix()
+from baselines.retacred.rmea_token_transport_recovery import run_lock
+lock=pathlib.Path(p.git('rev-parse','--git-path','q-rmea-owner-publish.lock'))
+if not lock.is_absolute():lock=r/lock
+with run_lock(lock):
+ h.branch_gate(relative)
+ target.mkdir(parents=True,exist_ok=True)
+ for name,text in x['files'].items():
+  assert name in p.REPORT_FILES
+  f=target/name
+  if f.exists():assert f.read_bytes()==text.encode()
+  else:f.write_bytes(text.encode())
+ record=h.validate_report(target);assert record['git_revision']==x['revision']
+ h._commit_report_and_push(target)
+ print(json.dumps({'published':True,'commit':p.git('rev-parse','HEAD')}))""".replace('\n+','\n')
        argv=self.config['owner_ssh_argv']+[shlex.join([self.config['owner_python'],'-B','-c',code])]
        payload=dict(worktree=self.config['owner_worktree'],run_id=self.config['run_id'],revision=self.config['revision'],files=files)
        response=subprocess.run(argv,input=json.dumps(payload).encode(),capture_output=True,timeout=300)
        if response.returncode:raise RuntimeError('Owner report-only publication failed; immutable report retained')
        return json.loads(response.stdout.splitlines()[-1])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();config=json.loads(Path(args.config).read_text(encoding='utf8'))
    publisher=Publisher(config);status=Path(config['status_file']);deadline=time.monotonic()+config.get('max_seconds',172800)
    failures=0
    while time.monotonic()<deadline:
        try:
            for request in publisher.requests():
                result=publisher.publish(request);atomic_json(status,dict(state='verified',**result));print(json.dumps(result),flush=True)
            terminal=publisher.terminal()
            if terminal and terminal['state'] in ('complete','failed','lost','cancelled'):
                result=publisher.deliver(terminal)
                atomic_json(status,dict(state='complete',delivery=result));return 0
            failures=0
        except Exception as e:
            failures+=1;atomic_json(status,dict(state='protection_pending',error_class=type(e).__name__,failures=failures))
            print(json.dumps(dict(error_class=type(e).__name__,failures=failures)),flush=True)
            if failures>=3:return 1
        if config.get('once'):return 0
        time.sleep(3)
    return 2


if __name__=='__main__':sys.exit(main())
