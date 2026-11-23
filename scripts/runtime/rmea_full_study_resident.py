"""Resident full-study supervision; no lease or browser connection controls training."""
import argparse,ctypes,hashlib,json,os,signal,subprocess,sys,tempfile,time
from datetime import datetime,timezone
from pathlib import Path

def require(ok,message):
    if not ok:raise ValueError(message)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    temp=path.with_suffix('.pending');temp.write_bytes((json.dumps(value,indent=2)+'\n').encode());os.replace(temp,path)
def children():
    return [int(x) for x in Path('/proc/self/task',str(os.getpid()),'children').read_text().split()]
def reap():
    while True:
        try:pid,_=os.waitpid(-1,os.WNOHANG)
        except ChildProcessError:return
        if pid==0:return
def run_scope(argv,*,cwd,env,stdout,stderr,monitor,interval=2,pending=None):
    libc=ctypes.CDLL(None,use_errno=True)
    require(libc.prctl(36,1,0,0,0)==0,'Subreaper unavailable')
    termination=pending if pending is not None else []
    owner=os.getpid()
    def parent_fence():
        if libc.prctl(1,signal.SIGKILL,0,0,0)!=0 or os.getppid()!=owner:os._exit(72)
    try:
        proc=subprocess.Popen(argv,cwd=cwd,env=env,stdout=stdout,stderr=stderr,
                              start_new_session=True,preexec_fn=parent_fence)
    except OSError as error:
        reap()
        return dict(exit_code=127,trainer_pid=None,scope_empty=children()==[],
                    wrapper_error_class=type(error).__name__,monitor_errors=0)
    error=None;monitor_errors=0
    def interrupted(signum,_):
        termination.append(signum)
    prior={sig:signal.signal(sig,interrupted) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP)}
    try:
        while proc.poll() is None:
            if termination:raise InterruptedError('Wrapper interrupted')
            try:monitor(proc)
            except Exception:
                if termination:raise InterruptedError('Wrapper interrupted during monitor')
                monitor_errors+=1
            if termination:raise InterruptedError('Wrapper interrupted during monitor')
            time.sleep(interval)
    except BaseException as exc:error=type(exc).__name__
    finally:
        for sig,handler in prior.items():signal.signal(sig,handler)
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=60)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        else:proc.wait()
        # Kernel subreaper adoption gives a positive process-scope membership test.
        deadline=time.monotonic()+2
        while children() and time.monotonic()<deadline:reap();time.sleep(.05)
        # Each killed adopted child may expose another generation. Keep ownership
        # until all descendants have become visible, been stopped and reaped.
        while children():
            for pid in children():
                try:os.kill(pid,signal.SIGKILL)
                except ProcessLookupError:pass
            reap();time.sleep(.05)
        reap();empty=children()==[]
    code=proc.returncode if proc.returncode else (1 if error else 0)
    return dict(exit_code=code,trainer_pid=proc.pid,scope_empty=empty,
                wrapper_error_class=error,monitor_errors=monitor_errors)

def self_test(output):
    output.mkdir(exist_ok=True);rows=[];calls=[0]
    def faulty_monitor(proc):
        calls[0]+=1
        if calls[0]==1:raise ValueError('Injected monitor error')
    with (output/'dummy.stdout').open('wb') as stdout,(output/'dummy.stderr').open('wb') as stderr:
        row=run_scope([sys.executable,'-B','-c','import time,sys;time.sleep(.15);sys.exit(17)'],
            cwd=output,env=os.environ.copy(),stdout=stdout,stderr=stderr,monitor=lambda _:None,interval=.03)
        require(row['exit_code']==17 and row['scope_empty'],'Dummy failure lost');rows.append(dict(case='trainer_exit17',**row))
        row=run_scope([sys.executable,'-B','-c','import time;time.sleep(.2)'],
            cwd=output,env=os.environ.copy(),stdout=stdout,stderr=stderr,monitor=faulty_monitor,interval=.03)
        require(row['exit_code']==0 and row['scope_empty'] and row['monitor_errors']==1 and calls[0]>1,'Monitor error orphaned trainer')
        rows.append(dict(case='monitor_error_preserves_supervision',**row))
    save(output/'scope-test.json',dict(status='PASS',cases=rows,scientific_updates=0,cos_operations=0))
    print(json.dumps(dict(status='PASS',cases=[x['case'] for x in rows])))

def close_transfers(root):
    import fcntl
    save(root/'transfers.closing',dict(closing_at=time.time()))
    with (root/'transfers.lock').open('a+b') as gate:
        fcntl.flock(gate.fileno(),fcntl.LOCK_EX)
        save(root/'transfers.closed',dict(closed_at=time.time()))
    # All shared-lock payload consumers have exited; future consumers reject closure.
    return True

def verify_delivery(control,v):
    run=Path(v['run_root']);x=dict(identity=v['identity'],revision=v['revision'])
    if not (run/'discovery/RUN_COMPLETE').is_file() or not (run.parent/'run-complete.json').is_file():raise ValueError('Final export incomplete')
    study=run/'discovery';delivery=json.loads((run.parent/'run-complete.json').read_bytes());seal=json.loads((study/'RUN_COMPLETE').read_bytes())
    if delivery.get('run')!=str(study) or delivery.get('source_sha256')!=x['identity']['source']:raise ValueError('Delivery run/source differs')
    if set(seal.get('artifacts',{}))!={'manifest.json','summary.json','analysis-readiness.json'}:raise ValueError('Study seal differs')
    for name,pin in seal['artifacts'].items():
     f=study/name
     if f.is_symlink() or not f.is_file() or f.stat().st_size!=pin['bytes'] or hashlib.sha256(f.read_bytes()).hexdigest()!=pin['sha256']:raise ValueError('Sealed study member differs')
    m=json.loads((study/'manifest.json').read_bytes())
    if m['source_sha256']!=x['identity']['source'] or m['config_sha256']!=x['identity']['config'] or m['git_revision']!=x['revision']:raise ValueError('Study provenance differs')
    ready=json.loads((study/'analysis-readiness.json').read_bytes());summary=json.loads((study/'summary.json').read_bytes())
    conditions=({'uniform','quantum','dephased','softmax12','real12'} if v['experiment']=='fair-reference'
                else {'return_coherent','return_dephased','native_square'})
    if ready.get('trace_materialized') is not False or set(ready.get('checkpoints',{}))!=conditions or set(summary.get('results',{}))!=conditions:raise ValueError('Final checkpoint inventory differs')
    for kind,row in ready['checkpoints'].items():
     rel='checkpoints/'+kind+'-final.pt';f=study/rel
     if row.get('path')!=rel or f.is_symlink() or not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=row['sha256'] or row['sha256']!=summary['results'][kind]['checkpoint_sha256']:raise ValueError('Final checkpoint differs')
    report=run.parent/(v['experiment']+'-safe-report');members={'summary.json','manifest.json','analysis-readiness.json','data.sha256','data_counts.txt','run_summary.json','receipt.json'}
    if report.is_symlink() or not report.is_dir() or {f.name for f in report.iterdir()}!=members or delivery.get('report',{}).get('path')!=str(report) or set(delivery['report'].get('files',{}))!=members:raise ValueError('Seven-file report coverage differs')
    for name,expected in delivery['report']['files'].items():
     f=report/name
     if f.is_symlink() or not f.is_file() or f.stat().st_size>2*1024**2 or hashlib.sha256(f.read_bytes()).hexdigest()!=expected:raise ValueError('Delivered report member differs')
    for name in seal['artifacts']:
     if hashlib.sha256((report/name).read_bytes()).hexdigest()!=seal['artifacts'][name]['sha256']:raise ValueError('Report differs from sealed study')
    r=json.loads((report/'receipt.json').read_bytes())
    if r.get('completed') is not True or r.get('source_sha256')!=x['identity']['source'] or r.get('git_revision')!=x['revision'] or set(r.get('whitelist',[]))!=members or set(r.get('files',{}))!=members-{'receipt.json'}:raise ValueError('Report receipt identity differs')
    if any(r['files'][n]!=delivery['report']['files'][n] for n in r['files']):raise ValueError('Report receipt hash differs')
    a=v['allocation'];identity={k:a[k] for k in ('allocation_id','generation','project_id','host_id','run_id','runner_attempt_id')}
    return dict(identity,revision=v['revision'],identity=v['identity'],run_root=str(run),verified_before_return=True,
        seal_sha256=digest(study/'RUN_COMPLETE'),delivery_sha256=digest(run.parent/'run-complete.json'),
        checkpoints={k:row['sha256'] for k,row in ready['checkpoints'].items()},report_files=delivery['report']['files'],verified_at=time.time())

def failed_start(root,v,error):
    a=v['allocation'];close_transfers(root);reap();empty=children()==[]
    identity={k:a[k] for k in ('allocation_id','generation','project_id','host_id','run_id','runner_attempt_id')}
    pid=os.getpid();ticks=Path('/proc',str(pid),'stat').read_text().split(') ')[1].split()[19];boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    save(root/'execution.json',dict(identity,argv=v['argv'],cwd=v['source'],revision=v['revision'],identity=v['identity'],
        exit_code=127,trainer_pid=None,scope_empty=empty,wrapper_error_class=type(error).__name__,
        resident_spawn_failed=True,finished_at=time.time()))
    save(root/'resource-return.json',dict(identity,schema='research.resource-return.v1',event_sequence=1,
        observed_at=datetime.now(timezone.utc).isoformat(),campaign_status='failed',all_jobs_terminal=empty,
        quiescent=empty,no_remaining_consumers=empty,cleanup_status='not_needed',resources_reusable=empty,
        owner_pid=pid,owner_pgid=os.getpgrp(),owner_start_ticks=ticks,boot_id=boot))
    save(root/'progress.json',dict(schema='research.progress.v1',attempt=a['runner_attempt_id'],project=a['project_id'],
        run_id=a['run_id'],phase='failed',current=0,total=v['total'],unit='update',heartbeat_at=time.time(),observed_at=time.time()))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--launch-sha256')
    parser.add_argument('--self-test',type=Path);args=parser.parse_args()
    if args.self_test:self_test(args.self_test);return 0
    root=Path(__file__).parent;launch=root/'full-launch.json'
    import fcntl
    resident_lock=(root/'resident.lock').open('a+b')
    fcntl.flock(resident_lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    require(not sys.flags.optimize and args.launch_sha256 and digest(launch)==args.launch_sha256,'Independent launch digest differs')
    v=json.loads(launch.read_bytes());a=v['allocation'];permit=Path(a['permit_path'])
    require(digest(permit)==a['permit_sha256'],'Permit changed');p=json.loads(permit.read_bytes())
    for key in ('allocation_id','generation','project_id','host_id','run_id','runner_attempt_id','gpu_uuids'):
        require(p[key]==a[key],'Permit resource binding differs')
    require(a['project_id']=='q-attention' and len(a['gpu_uuids'])==2,'Expected exact two-card project allocation')
    source=Path(v['source']);out=Path(v['output']);run=Path(v['run_root'])
    fair=v['experiment']=='fair-reference'
    if fair:
        import re
        require(re.fullmatch(r'[0-9]{8}T[0-9]{6}Z_seed13',out.name) is not None,'Startup report identity differs')
        datetime.strptime(out.name,'%Y%m%dT%H%M%SZ_seed13')
        require(root==Path('/hy-tmp/Q-Attention/control/full-protocol-fair-reference-20261124')/('formal-'+a['runner_attempt_id'])
                and out==Path('/hy-tmp/Q-Attention/owner/attempts')/a['runner_attempt_id']/'output'/out.name
                and run==out/a['run_id'],'Fair-reference attempt/control/output differs')
    else:
        require(out.parent.name==a['runner_attempt_id'] and run==out/a['run_id'],'Attempt/run namespace differs')
    require(not any((root/n).exists() for n in ('resident-start.json','execution.json','resource-return.json','training.stdout.log','training.stderr.log')),
            'Attempt already consumed; preserve first startup and terminal records')
    with (root/'resident-claim.json').open('x') as claim:
        json.dump(dict(attempt=a['runner_attempt_id'],launch_sha256=args.launch_sha256),claim)
        claim.flush();os.fsync(claim.fileno())
    pid=os.getpid();ticks=Path('/proc',str(pid),'stat').read_text().split(') ')[1].split()[19];boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    started=time.time();previous=[started,0]
    interrupted_signals=[]
    def defer_termination(signum,_):interrupted_signals.append(signum)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,defer_termination)
    try:
        if fair:
            expected=[v['python'],'-B','-m','baselines.retacred.rmea_fair_reference_handoff',
                      'run','--binding','/etc/q-attention/owner-handoffs/fair-reference.json','--resident-child']
            if v.get('resume') is True:expected.append('--resume')
            if v.get('restore_pin') is True:
                require(v.get('resume') is True,'Restore requires explicit resume');expected.append('--restore-pin')
            require(v['argv']==expected,'Fair-reference resident child entry differs')
        else:
            require(v['experiment']=='relative-return' and v['argv'][:4]==[v['python'],'-B',str(source/'experiments/q_rmea_train.py'),'--experiment'],'Trainer entry differs')
            require(v['argv'][4:6]==['relative-return','run'],'Experiment action differs')
            require('--resume' not in v['argv'] and '--restore-pin' not in v['argv'],'Fresh launch cannot imply resume')
            for flag,value in (('--output',str(out)),('--run-id',a['run_id']),('--admission-sha256',v['admission_sha256'])):
                require(v['argv'][v['argv'].index(flag)+1]==value,'Launch argument differs')
        for name,h in v['file_hashes'].items():require(digest(root/name)==h,'Reviewed control bytes changed')
        require(subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==v['revision'],'Current revision differs')
        require(not subprocess.check_output(['git','-C',str(source),'status','--porcelain']),'Current source dirty')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(p['gpu_uuids']),CUBLAS_WORKSPACE_CONFIG=':4096:8',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',TMPDIR='/hy-tmp',
            RESEARCH_OUTPUT_DIR=str(out.parent if fair else out),
            QATTN_RESIDENT_ATTEMPT=a['runner_attempt_id'],
            QATTN_PROGRESS_PATH=str(out.parent/'progress.json' if fair else root/'training-progress.json'))
        save(root/'resident-start.json',dict(pid=pid,attempt=a['runner_attempt_id'],started_at=started,launch_sha256=args.launch_sha256,owner_start_ticks=ticks,boot_id=boot))
        def monitor(proc):
            try:
                f=(out.parent/'progress.json' if fair else root/'training-progress.json');raw=json.loads(f.read_bytes()) if f.exists() else {}
                require(not raw or raw['attempt']==a['runner_attempt_id'],'Cross-attempt progress')
                now=time.time();done=raw.get('completed',0);rate=(done-previous[1])/(now-previous[0]) if done>previous[1] else None
                if done>previous[1]:previous[:]=[now,done]
                save(root/'progress.json',dict(schema='research.progress.v1',attempt=a['runner_attempt_id'],
                    project='q-attention',run_id=a['run_id'],phase=raw.get('stage','preparing_full_study'),
                    current=done,total=raw.get('total',v['total']),unit='update',rate=rate,jobs=raw.get('jobs',{}),
                    cloud_verified_updates=raw.get('cloud_verified_updates',0),observed_at=now,heartbeat_at=now))
            except Exception as error:
                save(root/'progress.json',dict(schema='research.progress.v1',attempt=a['runner_attempt_id'],
                    project='q-attention',run_id=a['run_id'],phase='monitor_unavailable',
                    current=previous[1],total=v['total'],unit='update',error_class=type(error).__name__,
                    observed_at=time.time(),heartbeat_at=time.time()))
                raise
        try:
            with (root/'training.stdout.log').open('xb') as stdout,(root/'training.stderr.log').open('xb') as stderr:
                if interrupted_signals:raise InterruptedError('Interrupted before trainer launch')
                result=run_scope(v['argv'],cwd=source,env=env,stdout=stdout,stderr=stderr,monitor=monitor,pending=interrupted_signals)
        except Exception as error:
            reap()
            result=dict(exit_code=127,trainer_pid=None,scope_empty=children()==[],
                        wrapper_error_class=type(error).__name__,monitor_errors=0)
    except Exception as error:
        reap();result=dict(exit_code=127,trainer_pid=None,scope_empty=children()==[],wrapper_error_class=type(error).__name__,monitor_errors=0)
    if result['exit_code']==0 and result['scope_empty']:
        try:
            proof=verify_delivery(root,v);save(root/'delivery-verification.json',proof)
            result['delivery_verification_sha256']=digest(root/'delivery-verification.json')
        except Exception as error:
            result['exit_code']=1;result['wrapper_error_class']=type(error).__name__
    transfers_empty=close_transfers(root)
    result['interrupted_signals']=interrupted_signals
    if interrupted_signals and result['exit_code']==0:result['exit_code']=1;result['wrapper_error_class']='InterruptedError'
    complete=result['exit_code']==0 and (run/'discovery/RUN_COMPLETE').is_file() and (out/'run-complete.json').is_file()
    identity={k:a[k] for k in ('allocation_id','generation','project_id','host_id','run_id','runner_attempt_id')}
    save(root/'execution.json',dict(identity,argv=v['argv'],cwd=str(source),revision=v['revision'],
        identity=v['identity'],started_at=started,finished_at=time.time(),**result))
    status='completed' if complete and result['scope_empty'] else 'failed'
    event=dict(identity,schema='research.resource-return.v1',event_sequence=1,
        observed_at=datetime.now(timezone.utc).isoformat(),campaign_status=status,
        all_jobs_terminal=result['scope_empty'],quiescent=result['scope_empty'],
        no_remaining_consumers=result['scope_empty'] and transfers_empty,cleanup_status='not_needed',
        resources_reusable=result['scope_empty'] and transfers_empty,owner_pid=pid,owner_pgid=os.getpgrp(),
        owner_start_ticks=ticks,boot_id=boot)
    save(root/'resource-return.json',event)
    save(root/'progress.json',dict(schema='research.progress.v1',attempt=a['runner_attempt_id'],
        project='q-attention',run_id=a['run_id'],phase=status,current=v['total'] if complete else previous[1],
        total=v['total'],unit='update',observed_at=time.time(),heartbeat_at=time.time()))
    print(json.dumps(dict(status=status,exit_code=result['exit_code'],quiescent=result['scope_empty'])),flush=True)
    return 0 if status=='completed' else (result['exit_code'] or 1)
if __name__=='__main__':sys.exit(main())
