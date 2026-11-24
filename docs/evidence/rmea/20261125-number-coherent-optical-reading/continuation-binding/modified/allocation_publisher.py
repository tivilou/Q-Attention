"""Allocation-bound owner publisher using the previously exercised COS transport."""
import argparse,hashlib,importlib.util,json,msvcrt,os,re,sqlite3,subprocess,sys,time
from pathlib import Path
from contextlib import closing
from datetime import datetime

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)

INPUT_REQUEST_SHA='a2903267be99693d2a5e248eff4fed899d31d0c3615fd09a0cd7bc1911b996cf'
INPUT_ACK_ID='e6c468164fb9358b0f262c9a60aeed370f60a182f6e3a8cdbaa1a8465fd0e4e0'
DESTINATION_SHA256='37612869c45e2591511481fd6baa175a84468d1d9915746ce7416229d48e315f'

def destination_guard(cfg):
    binding=Path(cfg['destination_binding_file'])
    require(cfg['destination_binding_sha256']==DESTINATION_SHA256 and sha(binding)==DESTINATION_SHA256,'Accepted destination binding changed')
    value=json.loads(binding.read_bytes())
    require(value['project_id']=='q-attention' and all(cfg[k]==value[k] for k in ('bucket','region')),'Publication destination differs')
    origin=subprocess.check_output(['git','-C',str(cfg['index_worktree']),'remote','get-url','origin'],text=True).strip()
    require(origin==value['index_origin'],'Private index origin differs')
    push_urls=subprocess.check_output(['git','-C',str(cfg['index_worktree']),'remote','get-url','--push','--all','origin'],text=True).splitlines()
    require(push_urls==[value['index_origin']],'Private index push destinations differ')

def optical_binding(experiment,run_id,control_root):
    expected_run='number-coherent-optical-seed13-20261125-v1'
    expected_root='/hy-tmp/Q-Attention/control/number-coherent-optical-full-20261125/continuation-20261125'
    if experiment=='number-coherent-optical' or run_id.startswith('number-coherent-optical-') or 'number-coherent-optical' in str(control_root):
        require(experiment=='number-coherent-optical' and run_id==expected_run and str(control_root)==expected_root,
            'Optical experiment/run/control binding differs')

def validate_attempt_paths(cfg,a):
    """Retain the legacy binding; add only this reviewed fair-reference namespace."""
    require(a['project_id']=='q-attention' and cfg['run_id']==a['run_id'],
            'Project/run binding differs')
    output='/hy-tmp/Q-Attention/owner/attempts/'+a['runner_attempt_id']+'/output'
    experiment=cfg.get('experiment','relative-return')
    optical_binding(experiment,a['run_id'],cfg['control_root'])
    require(not a['run_id'].startswith('fair-reference-seed13-') or experiment=='fair-reference',
            'Fair-reference requires explicit experiment binding')
    if experiment=='relative-return':
        expected=output+'/'+a['run_id']
        require(cfg['control_root']=='/hy-tmp/Q-Attention/control/relative-return-full-20261122',
                'Legacy control root differs')
    elif experiment=='number-coherent-optical':
        expected=output+'/'+a['run_id']
        require(a['runner_attempt_id'] != '5698113c7a6a407797c6d2e7b146be06' and a['generation'] > 83, 'Fresh continuation allocation required')
        require(not cfg.get('publication_lineage'), 'Existing authenticated generation index must be reused')
        require(a['run_id']=='number-coherent-optical-seed13-20261125-v1'
            and cfg['control_root']=='/hy-tmp/Q-Attention/control/number-coherent-optical-full-20261125/continuation-20261125',
                'Optical-fock exact run/control binding differs')
    elif experiment=='entity-query':
        expected=output+'/'+a['run_id']
        require(a['run_id']=='entity-query-seed13-20261124-v1'
            and cfg['control_root']=='/hy-tmp/Q-Attention/control/entity-query-full-20261124',
                'Entity-query exact run/control binding differs')
    else:
        require(experiment=='fair-reference'
            and re.fullmatch(r'[0-9a-f]{32}',a['runner_attempt_id']) is not None
            and re.fullmatch(r'fair-reference-seed13-[A-Za-z0-9_-]+',a['run_id']) is not None,
                'Declared fair-reference attempt required')
        report_id=cfg.get('report_id')
        require(isinstance(report_id,str) and re.fullmatch(r'[0-9]{8}T[0-9]{6}Z_seed13',report_id) is not None,
                'One startup timestamp_seed13 report identity required')
        datetime.strptime(report_id,'%Y%m%dT%H%M%SZ_seed13')
        expected=output+'/'+report_id+'/'+a['run_id']
        require(cfg['control_root']=='/hy-tmp/Q-Attention/control/full-protocol-fair-reference-20261124/formal-'+a['runner_attempt_id'],
                'Fair-reference control path differs')
    require(cfg['remote_run']==expected,'Authenticated run path differs')
    return (('number_coherent02', 'number_mixed02', 'number_mixed02_purity_normalized', 'curvature_matched_gaussian02') if experiment=='number-coherent-optical' else ('quantum_context','dephased_context','real12_context','softmax12_context','real_circuit_context') if experiment=='entity-query' else ('uniform','quantum','dephased','softmax12','real12') if experiment=='fair-reference'
            else ('return_coherent','return_dephased','native_square'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--config-sha256',required=True)
    args=parser.parse_args()
    require(not sys.flags.optimize and sha(args.config)==args.config_sha256,'Publisher configuration differs')
    cfg=json.loads(args.config.read_bytes());a=cfg['allocation'];destination_guard(cfg)
    require(Path(cfg['queue_db']).resolve()==Path(json.loads(Path(cfg['authority_config']).read_bytes())['database']).resolve(),'Queue authority mismatch')
    conditions=validate_attempt_paths(cfg,a)
    require(not cfg.get('canary_restore'), 'Formal publisher has no restoration URL export')
    require(a['project_id']=='q-attention' and cfg['run_id']==a['run_id']
            and cfg['prefix']=='projects/q-attention/formal-runs/'+a['run_id'],'Project/run binding differs')
    require(sha(cfg['implementation_file'])==cfg['implementation_sha256'],'Accepted publisher implementation differs')
    spec=importlib.util.spec_from_file_location('accepted_rmea_cos_publisher',cfg['implementation_file'])
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    lock=args.config.with_name('publisher.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:raise RuntimeError('Existing publisher retains this exact run')
    try:
        atomic_json=base.atomic_json
        class AllocationPublisher(base.Publisher):
            def object(self, rel, row):
                from qcloud_cos.cos_exception import CosServiceError
                key = self.config['prefix'].rstrip('/') + '/' + rel
                bucket = self.config['bucket']
                head = None
                try:
                    head = self.client.head_object(Bucket=bucket, Key=key)
                except CosServiceError as e:
                    if e.get_status_code() != 404:
                        raise
                if head is None:
                    headers = {'x-cos-server-side-encryption': 'AES256', 'x-cos-meta-sha256': row['sha256']}
                    url = self.client.get_presigned_url(Bucket=bucket, Key=key, Method='PUT', Expired=1800, Params={}, Headers=headers)
                    result = self.remote("import hashlib,http.client,json,pathlib,ssl,sys,urllib.parse\nimport fcntl\nfrom pathlib import Path\nx=json.load(sys.stdin)\nroot=Path(x['transfer_root'])\nwith (root/'transfers.lock').open('a+b') as gate:\n fcntl.flock(gate.fileno(),fcntl.LOCK_SH)\n if (root/'transfers.closed').exists():raise ValueError('Attempt transfers are closed')\n p=pathlib.Path(x['path'])\n if p.is_symlink() or not p.is_file() or p.stat().st_size!=x['bytes']:raise ValueError('Sealed source differs')\n h=hashlib.sha256()\n with p.open('rb') as f:\n  for b in iter(lambda:f.read(1048576),b''):h.update(b)\n if h.hexdigest()!=x['sha256']:raise ValueError('Sealed source hash differs')\n u=urllib.parse.urlsplit(x['url'])\n if u.scheme!='https':raise ValueError('HTTPS required')\n c=http.client.HTTPSConnection(u.hostname,u.port or 443,timeout=300,context=ssl.create_default_context())\n try:\n  with p.open('rb') as f:c.request('PUT',u.path+'?'+u.query,body=f,headers=dict(x['headers'],**{'Content-Length':str(x['bytes'])}))\n  response=c.getresponse();response.read(4096)\n  if response.status not in (200,201):raise RuntimeError('COS upload HTTP '+str(response.status))\n  print(json.dumps(dict(status=response.status)))\n finally:c.close()", dict(transfer_root=cfg['control_root'],path=str(self.root / rel), url=url, headers=headers, **row), timeout=900)
                    if result['status'] not in (200, 201):
                        raise ValueError('Upload failed')
                    head = self.client.head_object(Bucket=bucket, Key=key)
                if int(head.get('Content-Length', -1)) != row['bytes'] or head.get('x-cos-meta-sha256') != row['sha256'] or (not head.get('x-cos-version-id')) or (head['x-cos-version-id'] == 'null'):
                    raise ValueError('Version-pinned uploaded identity differs')
                version = head['x-cos-version-id']
                record = dict(bucket=bucket, region=cfg['region'], key=key, version_id=version, sha256=row['sha256'], bytes=row['bytes'])
                ledger = json.loads(self.verified.read_text(encoding='utf8')) if self.verified.exists() else {}
                if ledger.get(key) == record:
                    return record
                url = self.client.get_presigned_url(Bucket=bucket, Key=key, Method='GET', Expired=1800, Params={'versionId': version})
                result = self.remote("import hashlib,json,ssl,sys,urllib.request\nimport fcntl\nfrom pathlib import Path\nx=json.load(sys.stdin)\nroot=Path(x['transfer_root'])\nwith (root/'transfers.lock').open('a+b') as gate:\n fcntl.flock(gate.fileno(),fcntl.LOCK_SH)\n if (root/'transfers.closed').exists():raise ValueError('Attempt transfers are closed')\n op=urllib.request.build_opener(urllib.request.ProxyHandler({}))\n h=hashlib.sha256();count=0\n with op.open(x['url'],timeout=300) as response:\n  if response.headers.get('x-cos-version-id')!=x['version_id']:raise ValueError('Readback version differs')\n  for chunk in iter(lambda:response.read(1048576),b''):\n   count+=len(chunk);h.update(chunk)\n print(json.dumps(dict(bytes=count,sha256=h.hexdigest(),version_id=x['version_id'])))", dict(transfer_root=cfg['control_root'],url=url, version_id=version), timeout=900)
                if result['bytes'] != row['bytes'] or result['sha256'] != row['sha256'] or result['version_id'] != version:
                    raise ValueError('Managed-host pinned readback differs')
                ledger[key] = record
                atomic_json(self.verified, ledger)
                return record
            def publish(self, observation):
                allocation_guard()
                request = observation['request']
                self.validate(request)
                generation=request['generation'];names=['000-preparation']+['%03d-progress'%x for x in range(10,101,10)]
                require(generation in names and observation.get('relative')=='cos-channel/'+generation+'/request.json','Noncanonical request generation path')
                done=request['done']
                require(type(done) is int,'Milestone cursor must be integer')
                if generation!='000-preparation':require(set(request['files'])=={'cos-channel/'+generation+'/'+n for n in ('latest.pt','best.pt')},'Noncanonical checkpoint inventory')
                else:require(set(request['files'])=={'cos-channel/000-preparation/preparation.000.tarpart'},'Noncanonical preparation inventory')
                require((done==0 if generation=='000-preparation' else (cfg['total']*int(generation[:3])+99)//100<=done<=cfg['total']),'Premature milestone cursor')
                prefix=self.predecessor(request)
                existing = self.index / self.relative_index(request)
                recorded = self.published_receipt(request,prefix)
                self.verify_input_dependencies()
                if prefix:self.verify_pinned_versions(prefix)
                local_only=recorded is None and existing.exists()
                if local_only:recorded=json.loads(existing.read_bytes())
                verified_local=json.loads(self.verified.read_bytes()) if local_only and self.verified.exists() else {}
                if recorded is not None:
                    if recorded['request_sha256'] != observation['sha256'] or recorded['identity'] != request['identity'] or recorded['run_id'] != request['run_id'] or (recorded['done'] != request['done']) or (recorded['total'] != request['total']) or (set(recorded['objects']) != set(request['files'])):
                        raise ValueError('Indexed request differs')
                    objects = recorded['objects']
                    for (rel, row) in request['files'].items():
                        obj = objects[rel]
                        if obj.get('bucket')!=self.config['bucket'] or obj.get('region')!=self.config['region'] or obj['key'] != self.config['prefix'].rstrip('/') + '/' + rel or (not isinstance(obj.get('version_id'),str) or obj['version_id'] in ('','null')) or any((obj[k] != row[k] for k in ('bytes', 'sha256'))):
                            raise ValueError('Indexed object scope differs')
                        if local_only:require(verified_local.get(obj['key'])==obj,'Local-only object lacks exact verified readback')
                        head = self.client.head_object(Bucket=self.config['bucket'], Key=obj['key'], VersionId=obj['version_id'])
                        if int(head['Content-Length']) != row['bytes'] or head.get('x-cos-meta-sha256') != row['sha256'] or head.get('x-cos-version-id') != obj['version_id']:
                            raise ValueError('Indexed immutable version differs')
                else:
                    objects = {rel: self.object(rel, row) for (rel, row) in sorted(request['files'].items())}
                commit = self.index_receipt(request, observation['sha256'], objects,prefix)
                ack = dict(request_sha256=observation['sha256'], done=request['done'], objects=objects, index_commit=commit)
                if self.config.get('canary_restore'):
                    ack['restore_urls'] = {rel: self.client.get_presigned_url(Bucket=self.config['bucket'], Key=row['key'], Method='GET', Expired=900, Params={'versionId': row['version_id']}) for (rel, row) in objects.items()}
                installed=self.acknowledge(observation, ack)
                if installed is not None:commit=installed['index_commit']
                return dict(generation=request['generation'], done=request['done'], index_commit=commit, objects=len(objects))
            def predecessor(self,request):
                names=['000-preparation']+['%03d-progress'%x for x in range(10,101,10)]
                position=names.index(request['generation'])
                if position==0:return []
                code="""import hashlib,json,pathlib,sys
x=json.load(sys.stdin);request=x['request'];history=[];previous=0
for name in x['names']:
 f=pathlib.Path(x['root'])/'cos-channel'/name/'request.json';ack=f.with_name('ack.json')
 if f.is_symlink() or ack.is_symlink() or not f.is_file() or not ack.is_file():raise ValueError('Acknowledged predecessor missing')
 q=json.loads(f.read_bytes());a=json.loads(ack.read_bytes());done=q.get('done');expected={'cos-channel/'+name+'/'+n for n in (('preparation.000.tarpart',) if name=='000-preparation' else ('latest.pt','best.pt'))}
 if q.get('schema')!='q-rmea.cos-request.v1' or q.get('generation')!=name or q.get('run_id')!=request['run_id'] or q.get('identity')!=request['identity'] or q.get('total')!=request['total'] or type(done) is not int or not previous<=done<=request['done'] or a.get('done')!=done or a.get('request_sha256')!=hashlib.sha256(f.read_bytes()).hexdigest() or not a.get('index_commit') or set(q.get('files',{}))!=expected or set(a.get('objects',{}))!=expected:raise ValueError('Acknowledged predecessor cursor or identity differs')
 if name=='000-preparation':
  if done!=0:raise ValueError('Predecessor preparation cursor differs')
 elif done<(request['total']*int(name[:3])+99)//100:raise ValueError('Predecessor threshold differs')
 for key,pin in q['files'].items():
  obj=a['objects'][key]
  if obj.get('bucket')!=x['bucket'] or obj.get('region')!=x['region'] or obj.get('key')!=x['prefix']+'/'+key or (not isinstance(obj.get('version_id'),str) or obj['version_id'] in ('','null')) or any(obj.get(k)!=pin.get(k) for k in ('bytes','sha256')):raise ValueError('Predecessor object differs')
 previous=done;history.append(dict(request=q,ack=a))
print(json.dumps(history))"""
                result=self.remote(code,dict(root=str(self.root),request=request,names=names[:position],bucket=cfg['bucket'],region=cfg['region'],prefix=cfg['prefix']),timeout=15)
                require([row['request']['generation'] for row in result]==names[:position],'Predecessor verification failed')
                return result
            def prefix_at(self,tip,rows):
                def git(*argv):return subprocess.check_output(['git','-C',str(self.index),*argv],text=True,stderr=subprocess.DEVNULL,timeout=60).strip()
                for row in rows:
                    req,ack=row['request'],row['ack'];relative=self.relative_index(req).as_posix();commit=ack['index_commit']
                    require(len(commit)==40 and all(c in '0123456789abcdef' for c in commit),'Invalid predecessor index commit')
                    git('merge-base','--is-ancestor',commit,tip)
                    expected=dict(schema='q-rmea.cos-generation.v1',run_id=cfg['run_id'],generation=req['generation'],identity=cfg['identity'],done=req['done'],total=req['total'],request_sha256=ack['request_sha256'],objects=ack['objects'])
                    for revision in (commit,tip):
                        record=json.loads(git('show',revision+':'+relative))
                        require(all(record.get(k)==value for k,value in expected.items()),'Acknowledged prefix index differs')
            def published_receipt(self,request,prefix=()):
                with base.index_lock(self.index.parent/(self.index.name+'.publisher.lock')):
                    destination_guard(cfg)
                    def git(*argv):return subprocess.check_output(['git','-C',str(self.index),*argv],text=True,stderr=subprocess.DEVNULL).strip()
                    git('fetch','--no-tags','origin','refs/heads/main');tip=git('rev-parse','FETCH_HEAD')
                    self.prefix_at(tip,prefix)
                    relative=self.relative_index(request).as_posix()
                    if not git('ls-tree',tip,'--',relative):return None
                    return json.loads(git('show',tip+':'+relative))
            def index_receipt(self,request,request_sha,objects,prefix=()):
                import tempfile
                relative=self.relative_index(request).as_posix()
                value=dict(schema='q-rmea.cos-generation.v1',run_id=cfg['run_id'],generation=request['generation'],
                    identity=request['identity'],done=request['done'],total=request['total'],request_sha256=request_sha,
                    objects=objects,restoration=cfg.get('restoration','Restore pinned preparation parts, install pinned Git/environment/assets; restore latest/best and receipts; explicit --owner-managed --resume. Never automatically replay an unknown live attempt.'))
                content=(json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
                def git(*argv,input=None,env=None):
                    return subprocess.check_output(['git','-C',str(self.index),*argv],input=input,env=env,stderr=subprocess.DEVNULL).decode().strip()
                def at(commit,path):
                    if not git('ls-tree',commit,'--',path):return None
                    return json.loads(git('show',commit+':'+path))
                with base.index_lock(self.index.parent/(self.index.name+'.publisher.lock')):
                    require(not git('status','--porcelain','--untracked-files=all'),'Private index worktree must be clean')
                    for attempt in range(3):
                        destination_guard(cfg);git('fetch','--no-tags','origin','refs/heads/main');tip=git('rev-parse','FETCH_HEAD');head=git('rev-parse','HEAD')
                        self.prefix_at(tip,prefix)
                        ancestor=git('merge-base',head,tip)
                        if head!=ancestor:
                            changed=git('diff','--name-only',ancestor,head).splitlines()
                            require(git('rev-list','--count',ancestor+'..'+head)=='1'
                                and git('rev-list','--parents','-n','1',head).split()==[head,ancestor]
                                and len(changed)==1,'Unrelated unpublished index commit')
                            pending=changed[0]
                            require(pending.startswith('projects/q-attention/formal-runs/'+cfg['run_id']+'/') and pending.endswith('.json'),'Unrelated unpublished index commit')
                            pending_value=at(head,pending)
                            owned_current=pending==relative and pending_value==value
                            already_published=pending.startswith('projects/q-attention/formal-runs/'+cfg['run_id']+'/') and pending_value==at(tip,pending)
                            require(owned_current or already_published,'Unrelated unpublished index commit')
                        existing=at(tip,relative)
                        if existing is not None:
                            require(existing==value,'Published index generation already differs')
                            return tip
                        fd,name=tempfile.mkstemp(prefix=self.index.name+'.receipt-index-',dir=self.index.parent);os.close(fd);os.unlink(name)
                        environment=dict(os.environ,GIT_INDEX_FILE=name)
                        try:
                            git('read-tree',tip,env=environment)
                            blob=git('hash-object','-w','--stdin',input=content)
                            git('update-index','--add','--cacheinfo','100644',blob,relative,env=environment)
                            tree=git('write-tree',env=environment)
                            commit=git('commit-tree',tree,'-p',tip,'-m','recovery: q-attention '+cfg['run_id']+' '+request['generation'])
                            require(git('diff','--name-only',tip,commit)==relative and at(commit,relative)==value,'Receipt-only commit differs')
                            destination_guard(cfg)
                            try:git('-c','push.followTags=false','push','--no-follow-tags','--force-with-lease=refs/heads/main:'+tip,'origin',commit+':refs/heads/main')
                            except subprocess.CalledProcessError:
                                if attempt==2:raise
                                continue
                            git('fetch','--no-tags','origin','refs/heads/main');published_tip=git('rev-parse','FETCH_HEAD')
                            git('merge-base','--is-ancestor',commit,published_tip)
                            self.prefix_at(published_tip,prefix)
                            require(at(published_tip,relative)==value,'Published index readback differs')
                            return commit
                        finally:
                            for candidate in (Path(name),Path(name+'.lock')):
                                if candidate.exists():candidate.unlink()
                    raise RuntimeError('Index publication did not settle')
            def acknowledge(self, observation, ack):
                with closing(sqlite3.connect(cfg['queue_db'], timeout=5, isolation_level=None)) as database:
                    database.execute('BEGIN IMMEDIATE')
                    try:
                        row = database.execute('select generation,state,payload from allocations where id=?', (a['allocation_id'],)).fetchone()
                        require(row is not None and row[0] == a['generation'] and (row[1] in ('allocated', 'running')), 'Authority changed before ack')
                        current = json.loads(row[2])
                        for key in ('allocation_id', 'generation', 'project_id', 'host_id', 'run_id', 'runner_attempt_id', 'gpu_uuids'):
                            require(current[key] == a[key], 'Authority payload changed before ack')
                        result=self.remote("import hashlib,json,os,pathlib,sys\nx=json.load(sys.stdin);a=x['allocation'];permit=pathlib.Path(a['permit_path'])\nhost=pathlib.Path(x['host_config']);sys.path.insert(0,str(host.parent/'runtime'))\nfrom resource_guard import admission\nwith admission(json.loads(host.read_bytes())):\n state=json.loads((permit.parent/'state.json').read_bytes())\n if state.get('state')!='unknown' or state.get('allocation_only') is not True:raise ValueError('Host claim returned before ack')\n if hashlib.sha256(permit.read_bytes()).hexdigest()!=a['permit_sha256']:raise ValueError('Permit changed before ack')\n p=json.loads(permit.read_bytes())\n for key in ('allocation_id','generation','project_id','host_id','run_id','runner_attempt_id','gpu_uuids'):\n  if p[key]!=a[key]:raise ValueError('Permit generation changed')\n control=pathlib.Path(x['control_root'])\n with (control/'transfers.lock').open('a+b') as transfer_gate:\n  import fcntl\n  fcntl.flock(transfer_gate.fileno(),fcntl.LOCK_SH)\n  if (control/'transfers.closed').exists():raise ValueError('Attempt transfers are closed before ack')\n  root=pathlib.Path(x['root']);request=root/x['request']\n  if hashlib.sha256(request.read_bytes()).hexdigest()!=x['sha256']:raise ValueError('Request changed before ack')\n  out=request.parent/'ack.json';temporary=out.with_suffix('.owner-pending')\n  def installed():\n   old=json.loads(out.read_bytes())\n   if any(old.get(k)!=x['ack'][k] for k in ('request_sha256','done','objects')) or not old.get('index_commit'):raise ValueError('Existing ACK differs')\n   return old\n  if not out.exists():\n   with temporary.open('wb') as f:f.write(json.dumps(x['ack']).encode());f.flush();os.fsync(f.fileno())\n   try:os.link(temporary,out)\n   except FileExistsError:pass\n   finally:temporary.unlink(missing_ok=True)\n  print(json.dumps({'acknowledged':True,'ack':installed()}))", dict(allocation=a, host_config=cfg['host_config'], control_root=cfg['control_root'], root=str(self.root), request=observation['relative'], sha256=observation['sha256'], ack=ack), timeout=15)
                    finally:
                        database.rollback()
                installed=result['ack']
                if installed['index_commit']!=ack['index_commit']:
                    self.verify_terminal_index(observation['request'],installed)
                return installed
            def exact_terminal(self):
                result = self.remote("import hashlib,json,pathlib,sys\nx=json.load(sys.stdin);a=x['allocation'];control=pathlib.Path(x['control_root'])\nfiles=[control/'execution.json',control/'resource-return.json']\nif not all(f.is_file() for f in files):print('null');sys.exit()\nexecution,event=[json.loads(f.read_bytes()) for f in files]\nfor record in (execution,event):\n for key in ('allocation_id','generation','project_id','host_id','run_id','runner_attempt_id'):\n  if record[key]!=a[key]:raise ValueError('Terminal belongs to another allocation')\nif execution['revision']!=x['revision'] or execution['identity']!=x['identity']:raise ValueError('Terminal source differs')\nif event['campaign_status']=='failed':\n print(json.dumps(dict(failed=True,exit_code=execution['exit_code'],wrapper_error_class=execution.get('wrapper_error_class'),scope_empty=execution['scope_empty'],reported_quiescence=event['quiescent'])));sys.exit()\nif execution['exit_code']!=0 or not execution['scope_empty'] or event['campaign_status']!='completed' or event['quiescent'] is not True:\n raise ValueError('Exact training scope failed or is not empty')\nrun=pathlib.Path(x['remote_run']);request=run/'cos-channel/100-progress/request.json';ack=request.with_name('ack.json')\nif not request.is_file() or not ack.is_file():raise ValueError('Final acknowledged generation missing')\nq=json.loads(request.read_bytes());receipt=json.loads(ack.read_bytes())\nif q['run_id']!=a['run_id'] or q['generation']!='100-progress' or q['identity']!=x['identity'] or q['done']!=x['total'] or q['total']!=x['total']:\n raise ValueError('Final generation identity differs')\nif receipt['done']!=x['total'] or receipt['request_sha256']!=hashlib.sha256(request.read_bytes()).hexdigest() or not receipt['index_commit'] or set(receipt['objects'])!=set(q['files']):\n raise ValueError('Final authenticated acknowledgement differs')\nfor rel,pin in q['files'].items():\n obj=receipt['objects'][rel]\n if obj.get('bucket')!=x['bucket'] or obj.get('region')!=x['region'] or obj.get('sha256')!=pin['sha256'] or obj.get('bytes')!=pin['bytes'] or (not isinstance(obj.get('version_id'),str) or obj['version_id'] in ('','null')) or obj.get('key')!=x['prefix']+'/'+rel:\n  raise ValueError('Final object version/payload identity differs')\nproof_file=control/'delivery-verification.json'\nif not proof_file.is_file() or hashlib.sha256(proof_file.read_bytes()).hexdigest()!=execution.get('delivery_verification_sha256'):raise ValueError('Pre-return delivery verification missing')\nproof=json.loads(proof_file.read_bytes())\nfor key in ('allocation_id','generation','project_id','host_id','run_id','runner_attempt_id'):\n if proof.get(key)!=a[key]:raise ValueError('Delivery verification attempt differs')\nif proof.get('identity')!=x['identity'] or proof.get('revision')!=x['revision'] or proof.get('run_root')!=str(run) or proof.get('verified_before_return') is not True or not proof.get('seal_sha256') or not proof.get('delivery_sha256') or set(proof.get('checkpoints',{}))!=set(x['conditions']) or set(proof.get('report_files',{}))!={'summary.json','manifest.json','analysis-readiness.json','data.sha256','data_counts.txt','run_summary.json','receipt.json'}:raise ValueError('Delivery verification scope differs')\nhistory=[];previous=0\nfor percent in range(0,101,10):\n name='000-preparation' if percent==0 else '%03d-progress'%percent\n f=run/'cos-channel'/name/'request.json';ack=f.with_name('ack.json')\n if not f.is_file() or f.is_symlink() or not ack.is_file() or ack.is_symlink():raise ValueError('Acknowledged recovery prefix missing')\n req=json.loads(f.read_bytes());a2=json.loads(ack.read_bytes());done=req.get('done')\n if req.get('schema')!='q-rmea.cos-request.v1' or req.get('run_id')!=a['run_id'] or req.get('generation')!=name or req.get('identity')!=x['identity'] or req.get('total')!=x['total'] or type(done) is not int or not previous<=done<=x['total'] or a2.get('done')!=done or a2.get('request_sha256')!=hashlib.sha256(f.read_bytes()).hexdigest() or not a2.get('index_commit') or not req.get('files') or set(a2.get('objects',{}))!=set(req['files']):raise ValueError('Acknowledged recovery prefix differs')\n if percent==0:\n  if done!=0 or set(req['files'])!={'cos-channel/000-preparation/preparation.000.tarpart'}:raise ValueError('Preparation prefix differs')\n else:\n  if done<(x['total']*percent+99)//100 or set(req['files'])!={'cos-channel/'+name+'/'+n for n in ('latest.pt','best.pt')}:raise ValueError('Progress prefix differs')\n for n,pin in req['files'].items():\n  obj=a2['objects'][n]\n  if obj.get('bucket')!=x['bucket'] or obj.get('region')!=x['region'] or obj.get('key')!=x['prefix']+'/'+n or (not isinstance(obj.get('version_id'),str) or obj['version_id'] in ('','null')) or any(obj.get(k)!=pin.get(k) for k in ('bytes','sha256')):raise ValueError('Recovery prefix object differs')\n previous=done;history.append(dict(request=req,ack=a2))\nprint(json.dumps({'complete':True,'request':q,'ack':receipt,'history':history}))", dict(allocation=a,control_root=cfg['control_root'],revision=cfg['revision'],identity=cfg['identity'],remote_run=cfg['remote_run'],total=cfg['total'],prefix=cfg['prefix'],bucket=cfg['bucket'],region=cfg['region'],conditions=conditions),timeout=30)
                if result is None:return None
                if result.get('failed'):return result
                self.verify_input_dependencies()
                self.verify_terminal_index(result['request'],result['ack'],result['history'])
                self.verify_pinned_versions(result['history'])
                return dict(complete=True)
            def verify_input_dependencies(self):
                destination_guard(cfg)
                dependency=self.remote("from pathlib import Path\nimport json,sys\nf=Path(json.load(sys.stdin)['root'])/'recovery-inputs/input-recovery.json'\nif f.is_symlink() or not f.is_file():raise ValueError('Input dependency metadata missing')\nprint(json.dumps(json.loads(f.read_bytes())['dependency']))",dict(root=str(self.root)),timeout=15)
                request,ack=dependency['request'],dependency['ack']
                raw=(json.dumps(request,ensure_ascii=False,indent=2)+'\n').encode()
                ack_hash=hashlib.sha256(json.dumps(ack,sort_keys=True,separators=(',',':')).encode()).hexdigest()
                require(hashlib.sha256(raw).hexdigest()==INPUT_REQUEST_SHA and ack.get('request_sha256')==INPUT_REQUEST_SHA and ack_hash==INPUT_ACK_ID,'Original input dependency authentication differs')
                require(len(request['files'])==13 and set(request['files'])==set(ack['objects']) and sum(v['bytes'] for v in request['files'].values())==6947440640,'Original input dependency inventory differs')
                resolved={}
                for name,pin in request['files'].items():
                    obj=ack['objects'][name]
                    require(all(obj.get(k)==pin[k] for k in ('sha256','bytes')),'Original input object differs')
                    require(all(k not in obj or obj[k]==cfg[k] for k in ('bucket','region')),'Original input destination differs')
                    resolved[name]=dict(obj,bucket=cfg['bucket'],region=cfg['region'])
                self.verify_pinned_versions([dict(ack=dict(objects=resolved))])
            def verify_pinned_versions(self,history):
                seen=set()
                for row in history:
                    for obj in row['ack']['objects'].values():
                        require(obj.get('bucket')==cfg['bucket'] and obj.get('region')==cfg['region'],'Pinned destination differs')
                        require(isinstance(obj.get('version_id'),str) and obj['version_id'] not in ('','null'),'Concrete COS version required')
                        identity=(obj['bucket'],obj['key'],obj['version_id'])
                        if identity in seen:continue
                        head=self.client.head_object(Bucket=obj['bucket'],Key=obj['key'],VersionId=obj['version_id'])
                        require(head.get('x-cos-version-id')==obj['version_id'] and int(head.get('Content-Length',-1))==obj['bytes'] and head.get('x-cos-meta-sha256')==obj['sha256'],'Pinned recovery version missing or changed')
                        seen.add(identity)
            def verify_terminal_index(self,request,receipt,history=None):
                destination_guard(cfg)
                def git(*argv):return subprocess.check_output(['git','-C',str(self.index),*argv],text=True,stderr=subprocess.DEVNULL,timeout=60).strip()
                rows=history if history is not None else [dict(request=request,ack=receipt)]
                with base.index_lock(self.index.parent/(self.index.name+'.publisher.lock')):
                    git('fetch','--no-tags','origin','refs/heads/main');tip=git('rev-parse','FETCH_HEAD')
                    for row in rows:
                        req,ack=row['request'],row['ack'];relative=self.relative_index(req).as_posix();commit=ack['index_commit']
                        require(len(commit)==40 and all(c in '0123456789abcdef' for c in commit),'Invalid index commit')
                        git('merge-base','--is-ancestor',commit,tip)
                        records=[json.loads(git('show',revision+':'+relative)) for revision in (commit,tip)]
                        expected=dict(schema='q-rmea.cos-generation.v1',run_id=cfg['run_id'],generation=req['generation'],
                            identity=cfg['identity'],done=req['done'],total=req['total'],request_sha256=ack['request_sha256'],objects=ack['objects'])
                        require(all(all(record.get(k)==value for k,value in expected.items()) for record in records),'Final owner index differs')


        publisher=AllocationPublisher(cfg)
        def allocation_guard():
            destination_guard(cfg)
            require(sha(args.config)==args.config_sha256,'Publisher configuration changed')
            x=subprocess.run([cfg['researchctl'],'allocations'],capture_output=True,text=True,encoding='utf-8',timeout=30)
            require(x.returncode==0,'Allocation authority read failed')
            current=next(z for z in json.loads(x.stdout)['allocations'] if z['allocation_id']==a['allocation_id'])
            require(current['state'] in ('allocated','running'),'Allocation is no longer active')
            for key in ('generation','project_id','host_id','run_id','runner_attempt_id','permit_sha256','gpu_uuids'):
                require(current[key]==a[key],'Allocation identity differs')
            verified=publisher.remote("""import hashlib,json,pathlib,subprocess,sys
x=json.load(sys.stdin);permit=pathlib.Path(x['permit_path'])
if hashlib.sha256(permit.read_bytes()).hexdigest()!=x['permit_sha256']:raise ValueError('Permit differs')
p=json.loads(permit.read_bytes())
for key in ('allocation_id','generation','project_id','host_id','run_id','runner_attempt_id'):
 if p[key]!=x[key]:raise ValueError('Permit binding differs')
root=pathlib.Path(x['source']);head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
if head!=x['revision'] or subprocess.check_output(['git','-C',str(root),'status','--porcelain']):raise ValueError('Current source changed')
m=pathlib.Path(x['remote_run'])/'discovery/manifest.json'
if not m.is_file():raise ValueError('Current manifest required')
v=json.loads(m.read_bytes())
if v['source_sha256']!=x['identity']['source'] or v['config_sha256']!=x['identity']['config']:raise ValueError('Manifest identity differs')
print(json.dumps({'verified':True}))""",dict(a,source=cfg['source'],revision=cfg['revision'],remote_run=cfg['remote_run'],identity=cfg['identity']),timeout=45)
            require(verified.get('verified') is True,'Remote binding failed')
        status=Path(cfg['status_file']);failures=0
        import ctypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.GetCurrentProcess.restype=ctypes.c_void_p
        values=[ctypes.c_ulonglong() for _ in range(4)]
        kernel.GetProcessTimes.argtypes=[ctypes.c_void_p,*([ctypes.c_void_p]*4)]
        require(bool(kernel.GetProcessTimes(kernel.GetCurrentProcess(),*[ctypes.byref(x) for x in values])),'Creation identity unavailable')
        attachment=dict(attempt=a['runner_attempt_id'],pid=os.getpid(),process_start=values[0].value,
            allocation=a,config_sha256=args.config_sha256,code_sha256=sha(__file__))
        def heartbeat(**fields):
            nonlocal safe
            safe=dict(safe,**fields,**attachment,heartbeat_at=time.time());base.atomic_json(status,safe)
        safe=dict(state='attached');heartbeat()
        def terminal_result(terminal):
            if terminal.get('failed'):
                heartbeat(state='failed',trainer_exit_code=terminal['exit_code'],wrapper_error_class=terminal['wrapper_error_class'],
                    scope_empty=terminal['scope_empty'],reported_quiescence=terminal['reported_quiescence'])
                print(json.dumps(dict(state='failed',attempt=a['runner_attempt_id'],trainer_exit_code=terminal['exit_code'],wrapper_error_class=terminal['wrapper_error_class'],reported_quiescence=terminal['reported_quiescence'])),flush=True)
                return 1
            require(terminal.get('complete') is True,'Exact terminal incomplete')
            require(not publisher.requests(),'Unacknowledged final request remains')
            heartbeat(state='complete');print(json.dumps(dict(state='complete',attempt=a['runner_attempt_id'])),flush=True)
            return 0
        while True:
            try:
                terminal=publisher.exact_terminal()
                if terminal is not None:return terminal_result(terminal)
                pending=publisher.requests()
                for request in pending:
                    result=publisher.publish(request)
                    safe=dict(state='verified',generation=result['generation'],done=result['done'],
                              total=cfg['total'],attempt=a['runner_attempt_id'],pid=os.getpid(),heartbeat_at=time.time())
                    heartbeat();print(json.dumps({k:safe[k] for k in ('state','generation','done','total','attempt')}),flush=True)
                terminal=publisher.exact_terminal()
                if terminal is not None:return terminal_result(terminal)
                failures=0;heartbeat(state='verified' if safe.get('generation') else 'attached')
            except Exception as error:
                failures+=1
                heartbeat(state='protection_pending',error_class=type(error).__name__,failures=failures)
                print(json.dumps(dict(state='protection_pending',error_class=type(error).__name__,failures=failures)),flush=True)
                if failures>=3:return 1
            time.sleep(3)
    except KeyboardInterrupt:
        if 'heartbeat' in locals():heartbeat(state='stopped')
        return 130
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()

if __name__=='__main__':sys.exit(main())
