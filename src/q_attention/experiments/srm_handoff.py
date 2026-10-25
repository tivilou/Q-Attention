"""Collaborator-only bounded Q-SRM screen. No full-data experiment entry."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from q_attention.experiments import srm_protocol as p

EXPERIMENT='q_srm_screen'
EXTRA=('src/q_attention/experiments/srm_handoff.py','experiments/q_srm_handoff.py',
       'scripts/run_q_srm_handoff.sh','scripts/check_q_srm_handoff.sh',
       'scripts/export_q_srm_handoff.sh','tests/test_srm_handoff.py',
       'docs/current/q_srm_screen_handoff_zh.md')
REPORT_FILES={'summary.json','manifest.json','trace_validation.json','receipt.json',
              'run_summary.json','data.sha256','data_counts.txt'}


def source_contract():
    return dict(p.source_contract(),**{name:p.sha(p.ROOT/name) for name in EXTRA})


def git(*args):
    return subprocess.check_output(['git',*args],cwd=p.ROOT,text=True).rstrip('\n')


def release_review():
    p.validate_review()
    review=p.read(p.ROOT/'docs/validation/q_srm_handoff_review.json')
    if (review.get('review_scope')!='both' or review.get('verdict')!='PASS' or review.get('must_fix')
        or review.get('source_sha256')!=p.identity(source_contract())):
        raise ValueError('handoff requires exact-source scope-both PASS review')
    return review


def branch_gate(allowed=None):
    if git('branch','--show-current')!='1.1':raise ValueError('use collaborator branch 1.1')
    for ref in ('origin/main','origin/1.1'):
        if subprocess.run(['git','merge-base','--is-ancestor',ref,'HEAD'],cwd=p.ROOT).returncode:
            raise ValueError('fetch and integrate '+ref+' before continuing')
    # NUL output avoids quote/space ambiguity. Only a fresh final report can be dirty.
    entries=subprocess.check_output(['git','status','--porcelain=v1','-z','--untracked-files=all'],cwd=p.ROOT).split(b'\0')
    for entry in filter(None,entries):
        status=entry[:2].decode();name=entry[3:].decode()
        if status not in ('??','A ') or not allowed or not name.startswith(allowed+'/'):
            raise ValueError('worktree must be clean apart from this audited report: '+name)
    # A symlinked code file could otherwise retain a trusted digest and change later.
    if any((p.ROOT/name).is_symlink() for name in source_contract()):raise ValueError('symlinked source refused')


def runtime():
    import torch,numpy
    if sys.version_info[:2]!=(3,10) or torch.__version__.split('+')[0]!='2.7.1' or numpy.__version__!='1.26.4':
        raise ValueError('required Python3.10 / Torch2.7.1 / NumPy1.26.4; report mismatch, do not silently proceed')


def gpu_environment(choice,canary_only):
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if choice=='cpu':
        if not canary_only:raise ValueError('CPU is diagnostic canary-only in this handoff')
        env['CUDA_VISIBLE_DEVICES']=''
        return env,dict(device='cpu',execution_profile='low_memory_serial',selected_visible_index=None)
    import torch
    if not torch.cuda.is_available():raise ValueError('CUDA unavailable; no automatic CPU fallback')
    indices=range(torch.cuda.device_count()) if choice=='auto' else [int(choice)]
    candidates=[]
    for index in indices:
        free,_=torch.cuda.mem_get_info(index)
        if free>=8*1024**3:candidates.append((free,index))
    if not candidates:raise ValueError('no visible GPU with >=8GiB free; wait for a GPU')
    free,index=max(candidates)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    env['CUDA_VISIBLE_DEVICES']=visible.split(',')[index].strip() if visible is not None else str(index)
    return env,dict(device='cuda',execution_profile='low_memory_serial',selected_visible_index=index,cuda_visible_devices=env['CUDA_VISIBLE_DEVICES'],
                    gpu_name=torch.cuda.get_device_name(index),free_bytes_at_selection=free)


def preflight(args):
    release_review();branch_gate();runtime()
    env,selection=gpu_environment(args.gpu,args.canary_only)
    if not args.canary_only:
        if not args.data_dir:raise ValueError('--data-dir is required')
        _,_,_,_,data=p.load_data(Path(args.data_dir).resolve(),p.config('screen'))
        print(json.dumps({'data':{s:{k:v for k,v in data[s].items() if k in ('source_sha256','source_count','selected_count')} for s in ('train','valid','test')}}))
    print(json.dumps(dict(preflight='passed',source_sha256=p.identity(source_contract()),selection=selection)))
    return env,selection


def isolated_preflight(args):
    # Release the validation process (and its Torch/CUDA allocations) before training.
    # Never serialize the inherited environment: it can contain owner credentials.
    code=('import argparse,json,sys; sys.path.insert(0,sys.argv[1]); '
          'from q_attention.experiments.srm_handoff import preflight; '
          'env,selection=preflight(argparse.Namespace(**json.loads(sys.argv[2]))); '
          'print("QSRM_PROBE="+json.dumps(dict(selection=selection,cuda_visible_devices=env["CUDA_VISIBLE_DEVICES"])))')
    result=subprocess.run([sys.executable,'-c',code,str(p.ROOT/'src'),json.dumps(vars(args))],
                          cwd=p.ROOT,text=True,capture_output=True)
    if result.returncode:
        print(result.stdout,end='');print(result.stderr,file=sys.stderr,end='')
        raise RuntimeError('isolated preflight failed; no worker launched')
    lines=result.stdout.splitlines()
    if not lines or not lines[-1].startswith('QSRM_PROBE='):raise ValueError('preflight result missing')
    for line in lines[:-1]:print(line,flush=True)
    probe=json.loads(lines[-1].split('=',1)[1])
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
             CUDA_VISIBLE_DEVICES=probe['cuda_visible_devices'])
    return env,probe['selection']


def resolve_report(root,override=None):
    # The private batch is timestamp_seed13 although screen includes five frozen seeds.
    if not re.fullmatch(r'\d{8}T\d{6}Z_seed13',root.name):raise ValueError('run basename must be timestamp_seed13')
    reports=p.ROOT/'reports'/EXPERIMENT
    if any(x.is_symlink() for x in (p.ROOT/'reports',reports)):raise ValueError('symlinked reports root')
    target=Path(override) if override else reports/root.name
    if not target.is_absolute():target=p.ROOT/target
    if target.is_symlink() or target.parent.resolve()!=reports.resolve():raise ValueError('report must be an immediate child of reports/'+EXPERIMENT)
    return target


def validate_report(target):
    if target.is_symlink() or {x.name for x in target.iterdir()}!=REPORT_FILES:raise ValueError('report file allowlist differs')
    if any(not f.is_file() or f.is_symlink() for f in target.iterdir()):raise ValueError('nonregular report entry')
    record=p.read(target/'run_summary.json')
    if not record.get('data') or record.get('schema_version')!='q-srm.handoff-report.v1':raise ValueError('run_summary.data missing')
    if set(record['sha256'])!=REPORT_FILES-{'run_summary.json'}:raise ValueError('report hash inventory differs')
    if any(p.sha(target/name)!=digest for name,digest in record['sha256'].items()):raise ValueError('report hash mismatch')
    if p.read(target/'manifest.json')['fingerprint']!=record['fingerprint']:raise ValueError('report identity differs')
    return record


def export_report(root,override=None,inject_failure=False):
    from q_attention.experiments import srm_runner as runner
    root=Path(root).resolve();target=resolve_report(root,override)
    if root.is_relative_to(p.ROOT):raise ValueError('private run must stay outside checkout')
    relative=target.relative_to(p.ROOT).as_posix()
    release_review();branch_gate(relative)
    # Both the child completion/trace checks and the release provenance must agree.
    for phase in ('canary',) if p.read(root/'handoff.json')['canary_only'] else ('canary','screen'):
        manifest,_=runner.audit_run(root/phase)
        expected=p.config('canary' if phase=='canary' else 'screen')
        if manifest['config']!=expected or manifest['source_sha256']!=p.identity(p.source_contract()):raise ValueError('wrong completed source/config')
    handoff=p.read(root/'handoff.json')
    if handoff['source_sha256']!=p.identity(source_contract()):raise ValueError('release changed; export with the same source')
    phase=root/('canary' if handoff['canary_only'] else 'screen')
    manifest=p.read(phase/'run_manifest.json')
    if not handoff['canary_only'] and (manifest['device']!='cuda' or manifest['environment']!=p.read(root/'canary/run_manifest.json')['environment']):
        raise ValueError('screen lacks same-environment CUDA canary')
    if inject_failure and not handoff['canary_only']:raise ValueError('fault injection is synthetic-only')
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.'+target.name+'-',dir=target.parent))
    try:
        runner.export(phase,temporary/'safe')
        for f in (temporary/'safe').iterdir():f.rename(temporary/f.name)
        (temporary/'safe').rmdir()
        if inject_failure:raise RuntimeError('injected report copy failure')
        data={s:{k:v for k,v in manifest['data'][s].items() if k in ('source_sha256','source_count','selected_count')} for s in ('train','valid','test')}
        (temporary/'data.sha256').write_text(''.join(f'{data[s]["source_sha256"]}  {s}.jsonl\n' for s in data),encoding='utf-8')
        (temporary/'data_counts.txt').write_text(''.join(f'{s} source={data[s]["source_count"]} selected={data[s]["selected_count"]}\n' for s in data),encoding='utf-8')
        record=dict(schema_version='q-srm.handoff-report.v1',data=data,fingerprint=manifest['fingerprint'],run_id=root.name,
            stage='preflight' if handoff['canary_only'] else 'bounded_screen',formal=False,
            source_sha256=handoff['source_sha256'],git_revision=handoff['git_revision'],
            selection=handoff['selection'],canary_fingerprint=p.read(root/'canary/run_manifest.json')['fingerprint'],
            sha256={f.name:p.sha(f) for f in temporary.iterdir()})
        p.atomic_json(temporary/'run_summary.json',record);validate_report(temporary)
        if target.exists():
            validate_report(target)
            if {f.name:p.sha(f) for f in target.iterdir()}!={f.name:p.sha(f) for f in temporary.iterdir()}:raise ValueError('existing final report differs')
        else:temporary.rename(target)
        print(json.dumps(dict(report=relative,fingerprint=record['fingerprint'])))
        return target
    finally:
        if temporary.exists():shutil.rmtree(temporary)


def run(args):
    env,selection=isolated_preflight(args)
    root=Path(args.run_dir).resolve() if args.run_dir else p.ROOT.parent/'q-srm-private/handoff'/ (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_seed13')
    resolve_report(root,args.report_dir)
    if root.is_relative_to(p.ROOT):raise ValueError('raw output must stay outside checkout')
    if root.exists()!=args.resume:raise ValueError('fresh output must not exist; existing run requires --resume')
    root.mkdir(parents=True,exist_ok=True)
    with p.exclusive(root/'.handoff.lock'):
        metadata=dict(source_sha256=p.identity(source_contract()),git_revision=p.revision(),canary_only=args.canary_only,
                      selection=selection,data_dir=str(Path(args.data_dir).resolve()) if args.data_dir else None)
        if (root/'handoff.json').exists():
            old=p.read(root/'handoff.json')
            # Free-memory observations change; the selected mask, model/device and environment must not.
            if old['source_sha256']!=metadata['source_sha256'] or old['canary_only']!=args.canary_only or old['data_dir']!=metadata['data_dir']:
                raise ValueError('handoff contract differs')
            if {k:v for k,v in old['selection'].items() if k!='free_bytes_at_selection'}!={k:v for k,v in selection.items() if k!='free_bytes_at_selection'}:
                raise ValueError('GPU selection differs; resume with the original --gpu visible index and CUDA_VISIBLE_DEVICES')
            metadata=old
        else:p.atomic_json(root/'handoff.json',metadata)
        print('PRIVATE_RUN='+str(root),flush=True)
        for phase in ('canary',) if args.canary_only else ('canary','screen'):
            directory=root/phase
            argv=[sys.executable,str(p.ROOT/'experiments/run_srm_screen.py'),'--profile',phase,'--device',selection['device'],'--output-dir',str(directory)]
            if phase=='screen':argv+=['--data-dir',metadata['data_dir']]
            if directory.exists():argv+=['--resume']
            subprocess.run(argv,cwd=p.ROOT,env=env,check=True)
        return export_report(root,args.report_dir,args.inject_export_failure)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','run','export','check-staged'))
    parser.add_argument('--data-dir');parser.add_argument('--gpu',default='auto')
    parser.add_argument('--canary-only',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--run-dir');parser.add_argument('--report-dir')
    parser.add_argument('--inject-export-failure',action='store_true')
    args=parser.parse_args(argv)
    if args.action=='check':preflight(args)
    elif args.action=='run':run(args)
    elif args.action=='export':
        if not args.run_dir:parser.error('export requires --run-dir')
        export_report(args.run_dir,args.report_dir,args.inject_export_failure)
    else:
        if not args.report_dir or not args.run_dir:parser.error('check-staged requires --report-dir and immutable --run-dir')
        target=Path(args.report_dir)
        if not target.is_absolute():target=p.ROOT/target
        target=resolve_report(Path(target.name),str(target))
        relative=target.relative_to(p.ROOT).as_posix();branch_gate(relative);release_review();validate_report(target)
        export_report(args.run_dir,str(target))
        paths=set(filter(None,git('diff','--cached','--name-only').splitlines()))
        expected={relative+'/'+n for n in REPORT_FILES}
        if paths!=expected:raise ValueError('stage exactly the seven audited report files; no other changes')
        print('STAGED_REPORT_OK')
    return 0
