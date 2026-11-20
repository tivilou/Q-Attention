"""One admitted technical canary through the actual collaborator parent path."""
import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baselines.retacred import rmea_token_transport_protocol as p


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture-cache',type=Path);parser.add_argument('--unit-only',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args(); output=a.output.absolute();attempt=output.parent
    if a.unit_only==bool(a.fixture_cache):
        raise ValueError('Choose the pure mechanism units or the pinned full technical canary')
    request=p.read(attempt/'request.json');command=p.read(attempt/'command.json')
    expected=['{python}','-B','experiments/q_rmea_token_transport_canary.py']+(
        ['--unit-only'] if a.unit_only else ['--fixture-cache',str(a.fixture_cache)])+['--output','{output}']
    if (os.environ.get('RESEARCH_OUTPUT_DIR')!=str(output) or request['project']!='q-attention'
            or request['attempt']!=attempt.name or request['recipe']['command']!=expected
            or request['recipe']['outputs']!=['compact-report.json','RUN_COMPLETE']
            or request['recipe']['timeout']!=1200 or request['recipe']['max_attempts']!=1 or request['recipe']['replay_safe'] is not False
            or request['recipe']['resources']!=dict(cpu=1,ram_mb=4096,disk_mb=4096,gpus=1,gpu_mb=8192)
            or request['boot_id']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            or request['revision']!=p.git('rev-parse','HEAD') or command['cwd']!=str(p.ROOT)
            or command['revision']!=request['revision']
            or command['argv']!=[x.replace('{python}',sys.executable).replace('{output}',str(output)) for x in expected]
            or os.environ.get('CUDA_VISIBLE_DEVICES')!=','.join(request['gpu_ids']) or len(request['gpu_ids'])!=1):
        raise ValueError('Exact scheduler admission required')
    if os.getpgrp()!=os.getpid() or sys.platform!='linux':
        raise ValueError('Worker must own this Linux job process group')

    def terminate_group(*unused):
        os.killpg(os.getpgrp(),signal.SIGKILL)

    # The parent remains Torch-free. Its ordinary timeout/cancellation/supervisor
    # death kills this complete process group; the worker remains the authority
    # for terminal/quiescent receipts, including an unexpected kill of this parent.
    signal.signal(signal.SIGTERM,terminate_group);signal.signal(signal.SIGALRM,terminate_group)
    parent=os.getppid();lib=ctypes.CDLL(None,use_errno=True)
    if parent<=1 or lib.prctl(1,signal.SIGTERM,0,0,0)!=0 or os.getppid()!=parent:
        terminate_group()
    base=time.monotonic();wall=time.time();age=wall-(attempt/'request.json').stat().st_mtime
    if not -5<=age<1200:raise ValueError('Admission expired')
    deadline=base+1200-max(0,age)
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))

    def check():
        lease=p.read(attempt/'lease.json')
        if lease['token']!=request['token'] or time.monotonic()>=lease['until'] or (attempt/'cancel.json').exists():
            terminate_group()

    p.review_gate();check()
    if output.name!='output' or not output.is_dir() or list(output.iterdir()):
        raise ValueError('Fresh assigned output required')
    if not a.unit_only:
        from baselines.retacred.rmea_entity_study import FIXTURE_SHA
        if p.digest(a.fixture_cache)!=FIXTURE_SHA:raise ValueError('Pinned tiny historical fixture required')
    work=attempt/'scientific-work';work.mkdir();env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')

    def execute(name,argv,expected_code):
        check();begin=time.monotonic()
        with (work/(name+'.stdout')).open('xb') as out,(work/(name+'.stderr')).open('xb') as err:
            try:
                done=subprocess.run(argv,cwd=p.ROOT,env=env,stdout=out,stderr=err,
                    timeout=max(.001,deadline-time.monotonic()-5))
            except BaseException:
                terminate_group();raise
        check()
        if done.returncode!=expected_code:
            raise RuntimeError(name+' unexpected exit '+str(done.returncode)+'; preserve private logs, no scientific retry')
        return dict(exit_status=done.returncode,seconds=time.monotonic()-begin,
                    stdout_sha256=p.digest(work/(name+'.stdout')),stderr_sha256=p.digest(work/(name+'.stderr')))

    tests=execute('unit',[sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_rmea_token_transport.py','-v'],0)
    if a.unit_only:
        check()
        record=dict(status='MECHANISM_UNITS_PASS',source_revision=request['revision'],
            source_sha256=p.identity(p.sources()),tests=tests,formal_training=False,
            full_parent_canary=False,release_ready=False,L5='NOT_ACCEPTED')
        p.atomic_json(output/'compact-report.json',record)
        p.atomic_json(output/'RUN_COMPLETE',dict(schema='q-rmea.token-transport.units.closed.v1',
            files={'compact-report.json':p.digest(output/'compact-report.json')},formal_training=False))
        print(json.dumps(dict(status=record['status'],release_ready=False,L5='NOT_ACCEPTED')))
        return
    root=work/(dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_seed13')
    common=[sys.executable,'-B','experiments/q_rmea_token_transport_handoff.py']
    parent_run=execute('parent',common+['run','--development-canary','--canary-only','--gpu','auto',
        '--development-cache',str(a.fixture_cache),'--run-dir',str(root),'--inject-export-failure'],1)
    if 'Injected report copy failure' not in (work/'parent.stderr').read_text(errors='replace'):
        raise ValueError('Parent failed before the intended exporter injection')
    manifest,summary=p.audit(root/'canary');before=p.inventory(root/'canary')
    target=root.parent/(root.name+'-safe-report')
    if target.exists() or list(target.parent.glob('.'+target.name+'-*')):
        raise ValueError('Injected exporter left a final or partial report')
    retry=execute('export-retry',common+['export','--development-canary','--run-dir',str(root)],0)
    from baselines.retacred.rmea_token_transport_handoff import validate_report
    validate_report(target)
    if p.inventory(root/'canary')!=before or list(target.parent.glob('.'+target.name+'-*')):
        raise ValueError('Export retry mutated source or left partials')
    delivery=execute('git-delivery',[sys.executable,'-B','experiments/check_rmea_token_transport_delivery.py','--run-dir',str(root)],0)
    handoff=p.read(root/'handoff.json')
    if handoff['parent_imported_torch_before_children'] or handoff['parent_imported_torch_after_children']:
        raise ValueError('Actual handoff parent retained Torch')
    record=dict(status='TECHNICAL_CANARY_PASS',source_revision=request['revision'],source_sha256=manifest['source_sha256'],
        source_run_complete_sha256=p.digest(root/'canary/RUN_COMPLETE'),tests=tests,parent=parent_run,export_retry=retry,git_delivery=delivery,
        trace_validation=p.read(root/'canary/trace_validation.json'),
        conditions={k:dict(updates=v['updates'],parameters=v['parameters']) for k,v in summary['results'].items()},
        report_sha256={n:p.digest(target/n) for n in p.REPORT_FILES},
        injected_failure_cleanup=True,immutable_export_retry=True,retry_count=1,
        control_wrapper_imported_torch='torch' in sys.modules,
        actual_handoff_parent_imported_torch=False,formal_training=False,L5='NOT_ACCEPTED')
    if record['control_wrapper_imported_torch']:raise ValueError('Control wrapper retained Torch')
    check();p.atomic_json(output/'compact-report.json',record)
    p.atomic_json(output/'RUN_COMPLETE',dict(schema='q-rmea.token-transport.canary.closed.v1',
        files={'compact-report.json':p.digest(output/'compact-report.json')},formal_training=False))
    print(json.dumps(dict(status=record['status'],formal_training=False,L5='NOT_ACCEPTED')))


if __name__=='__main__':
    main()
