"""Portable collaborator entry and audited report-only Git delivery."""
import argparse
import datetime as dt
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from . import rmea_token_transport_protocol as p

REPORT_FILES=p.REPORT_FILES

def branch_gate(allowed=None):
    if p.git('branch','--show-current')!='1.1':raise ValueError('Use collaborator branch1.1')
    for ref in ('origin/1.1','origin/main'):
        if subprocess.run(['git','merge-base','--is-ancestor',ref,'HEAD'],cwd=p.ROOT).returncode:raise ValueError('Fetch/integrate '+ref)
    entries=subprocess.check_output(['git','status','--porcelain=v1','-z','--untracked-files=all'],cwd=p.ROOT).split(b'\0')
    for entry in filter(None,entries):
        status=entry[:2].decode();name=entry[3:].decode()
        if not allowed or status not in ('??','A ') or not name.startswith(allowed+'/'):raise ValueError('Clean worktree required: '+name)

def probe(args):
    p.review_gate(formal=not args.development_canary)
    if args.owner_managed and args.development_canary: raise ValueError('Owner-managed full run and development canary are distinct')
    if args.development_canary:
        if not args.canary_only:raise ValueError('Development mode is canary-only')
        if p.git('status','--porcelain'):raise ValueError('Clean development worktree required')
    else:
        branch_gate()

    import importlib.metadata as md
    import torch
    allowed={('2.4.1','1.24.3'),('2.5.1','1.26.4'),('2.7.1','1.26.4')}
    if sys.version_info[:2]!=(3,10) or (torch.__version__.split('+')[0],md.version('numpy')) not in allowed:
        raise ValueError('Requires Python3.10 and an explicit Torch/NumPy pair:2.4.1/1.24.3,2.5.1/1.26.4,2.7.1/1.26.4; no silent install')
    if args.development_cache:
        if not (args.development_canary and args.canary_only):raise ValueError('Pinned fixture is bounded development canary only')
    elif md.version('transformers')!='4.44.2':raise ValueError('Requires Transformers4.44.2')
    memory=Path('/sys/fs/cgroup/memory.max')
    budget=4*1024**3 if not args.canary_only else 2*1024**3
    if memory.is_file() and memory.read_text().strip()!='max' and int(memory.read_text())<budget:
        raise ValueError('Container memory limit too small; keep scientific budget fixed')
    space=Path(args.run_dir).absolute().parent if args.run_dir else p.ROOT.parent
    while not space.exists():space=space.parent
    if shutil.disk_usage(space).free < (12*1024**3 if not args.canary_only else 1024**3):
        raise ValueError('Insufficient private run disk space')
    if not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    indices=range(torch.cuda.device_count()) if args.gpu=='auto' else [int(args.gpu)]
    minimum=8*1024**3
    candidates=[(torch.cuda.mem_get_info(i)[0],i) for i in indices if torch.cuda.mem_get_info(i)[0]>=minimum]
    if not candidates:raise ValueError('No eligible free GPU; wait, do not alter scientific budget')
    free,index=max(candidates);visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    mask=visible.split(',')[index].strip() if visible is not None else str(index)
    return dict(cuda_visible_devices=mask,gpu_name=torch.cuda.get_device_name(index),free_bytes_at_selection=free,execution_profile='low_memory_serial')

def isolated_probe(args):
    code='import argparse,json;from baselines.retacred.rmea_token_transport_handoff import probe;print(json.dumps(probe(argparse.Namespace(**json.loads(__import__("sys").argv[1])))))'
    result=subprocess.run([sys.executable,'-B','-c',code,p.json.dumps(vars(args))],cwd=p.ROOT,text=True,capture_output=True)
    if result.returncode:
        print(result.stderr,file=sys.stderr,end='');raise ValueError('Preflight failed before training')
    return p.json.loads(result.stdout)

def validate_report(target):
    if target.is_symlink() or {f.name for f in target.iterdir()}!=REPORT_FILES:raise ValueError('Report whitelist differs')
    if any(f.is_symlink() or not f.is_file() for f in target.iterdir()):raise ValueError('Nonregular report file')
    record=p.read(target/'run_summary.json')
    if record.get('schema')!='q-rmea.handoff-report.v1' or not record.get('data') or set(record['sha256'])!=REPORT_FILES-{'run_summary.json'}:raise ValueError('Report schema/inventory differs')
    if any(p.digest(target/k)!=v for k,v in record['sha256'].items()):raise ValueError('Report hash mismatch')
    return record

def export(root,target,inject_failure=False):
    root=Path(root).resolve();target=Path(target).absolute()
    if root.is_relative_to(p.ROOT) or target.is_symlink():raise ValueError('Private source must stay outside checkout; regular report destination')
    handoff=p.read(root/'handoff.json');phase='canary' if handoff['canary_only'] else 'discovery'
    # Audit both actual parent phases; export retry never imports Torch or trains.
    for name in ('canary',) if phase=='canary' else ('canary','discovery'):
        audited,_=p.audit(root/name)
        if audited['config']['profile']!=name or audited['git_revision']!=handoff['git_revision']:
            raise ValueError('Audited phase/revision differs from parent')
        if phase=='discovery' and audited['environment'].get('fixture_sha256'):
            raise ValueError('Formal handoff requires a live nonfixture canary and discovery')
    manifest,summary=p.audit(root/phase)
    if p.read(root/'canary/manifest.json')['environment']!=manifest['environment']:raise ValueError('Canary and discovery environments differ')
    if manifest['git_revision']!=handoff['git_revision']:raise ValueError('Run revision differs')
    if inject_failure and phase!='canary':raise ValueError('Fault injection is canary-only')
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.'+target.name+'-',dir=target.parent))
    try:
        for name in ('summary.json','manifest.json','trace_validation.json'):shutil.copyfile(root/phase/name,temporary/name)
        if inject_failure:raise RuntimeError('Injected report copy failure; export-only retry from same completed source')
        p.atomic_json(temporary/'receipt.json',dict(closed_inventory_sha256=p.digest(root/phase/'RUN_COMPLETE'),source_sha256=manifest['source_sha256'],raw_artifacts_committed=False,private_artifacts_policy='retained on collaborator; no automatic transfer'))
        (temporary/'data.sha256').write_text(''.join(f'{v["source_sha256"]}  {s}.jsonl\n' for s,v in manifest['data'].items()),encoding='utf8')
        (temporary/'data_counts.txt').write_text(''.join(f'{s} source={v["source_count"]} selected={v["selected_count"]}\n' for s,v in manifest['data'].items()),encoding='utf8')
        p.atomic_json(temporary/'run_summary.json',dict(schema='q-rmea.handoff-report.v1',data=manifest['data'],source_sha256=manifest['source_sha256'],
            git_revision=manifest['git_revision'],run_id=root.name,profile=manifest['config']['profile'],formal=manifest['config']['profile']=='discovery',
            sha256={f.name:p.digest(f) for f in temporary.iterdir()}))
        validate_report(temporary)
        if target.exists():
            validate_report(target)
            if {f.name:p.digest(f) for f in temporary.iterdir()}!={f.name:p.digest(f) for f in target.iterdir()}:raise ValueError('Existing final report differs')
        else:temporary.rename(target)
        return target
    finally:
        if temporary.exists():shutil.rmtree(temporary)

def _commit_report_and_push(target):
    """Shared delivery mechanics; the caller owns publication admission and lock."""
    relative=target.relative_to(p.ROOT).as_posix()
    if target.parent!=p.ROOT/'reports/q_rmea_token_transport' or not re.fullmatch(r'\d{8}T\d{6}Z_seed13',target.name):
        raise ValueError('Unexpected report destination')
    branch_gate(relative);record=validate_report(target)
    baseline=record['git_revision'];head=p.git('rev-parse','HEAD');expected={relative+'/'+n for n in REPORT_FILES}
    hashes={name:p.digest(p.ROOT/name) for name in expected}
    if head==baseline:
        p.git('add','--',*sorted(expected))
        if set(p.git('diff','--cached','--name-only').splitlines())!=expected:raise ValueError('Unexpected staged changes')
        p.git('diff','--cached','--check');p.git('commit','-m','reports: Q-RMEA '+record['profile']+' '+target.name);head=p.git('rev-parse','HEAD')
    if p.git('rev-list','--parents','-n','1',head).split()!=[head,baseline]:raise ValueError('Only the single report commit may follow run revision')
    if set(p.git('diff','--name-only',baseline,head).splitlines())!=expected or set(p.git('diff','--name-only','--diff-filter=A',baseline,head).splitlines())!=expected:raise ValueError('Commit must add exactly audited report files')
    for name,sha in hashes.items():
        raw=subprocess.check_output(['git','show',head+':'+name],cwd=p.ROOT)
        if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Committed bytes differ')
    branch_gate();validate_report(target)
    if p.git('rev-parse','HEAD')!=head:raise ValueError('HEAD changed before push')
    p.git('push','origin',head+':refs/heads/1.1')
    print(p.json.dumps(dict(published_report=relative,commit=head,branch='1.1')),flush=True)

def publish(root):
    root=Path(root).resolve()
    if not re.fullmatch(r'\d{8}T\d{6}Z_seed13',root.name):raise ValueError('Expected timestamp_seed13 run basename')
    parent=p.ROOT/'reports/q_rmea_token_transport';target=parent/root.name;relative=target.relative_to(p.ROOT).as_posix()
    if any(x.is_symlink() for x in (p.ROOT/'reports',parent,target)):raise ValueError('Symlinked report directory')
    p.review_gate(formal=True)
    handoff=p.read(root/'handoff.json');canary,_=p.audit(root/'canary')
    if canary['config']['profile']!='canary' or canary['git_revision']!=handoff['git_revision']:
        raise ValueError('Audited canary differs from parent')
    if canary['environment'].get('fixture_sha256'):
        raise ValueError('Ordinary publication requires a live nonfixture canary')
    lock=Path(p.git('rev-parse','--git-path','q-rmea-publish.lock'))
    if not lock.is_absolute():lock=p.ROOT/lock
    with p.lock(lock):
        p.git('fetch','origin','+refs/heads/main:refs/remotes/origin/main','+refs/heads/1.1:refs/remotes/origin/1.1')
        branch_gate(relative)
        export(root,target)
        _commit_report_and_push(target)

def run(args):
    root=Path(args.run_dir).resolve() if args.run_dir else p.ROOT.parent/'q-rmea-token-transport-private'/ (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_seed13')
    if not args.development_canary and not re.fullmatch(r'\d{8}T\d{6}Z_seed13',root.name):
        raise ValueError('Expected timestamp_seed13 run basename before training')
    if root.is_relative_to(p.ROOT) or (root.exists() and not args.resume):raise ValueError('Fresh private run or explicit resume required')
    if args.resume and ((not args.owner_managed and not args.development_canary) or not root.exists()):raise ValueError('Explicit owner-managed existing full-state resume required')
    selection=isolated_probe(args)
    root.mkdir(parents=True,exist_ok=args.resume)
    previous=p.read(root/'handoff.json') if args.resume else None
    handoff=dict(git_revision=p.git('rev-parse','HEAD'),canary_only=args.canary_only,selection=selection,
                 parent_imported_torch_before_children='torch' in sys.modules)
    if previous:
        if previous['git_revision']!=handoff['git_revision'] or previous['canary_only']!=args.canary_only:
            raise ValueError('Resume parent/source differs')
        handoff=previous
    p.atomic_json(root/'handoff.json',handoff)
    print('PRIVATE_RUN='+str(root),flush=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=selection['cuda_visible_devices'],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
    env['QATTN_PROGRESS_PATH']=str(root.parent/'progress.json')
    if not args.canary_only:env['QATTN_OWNER_MANAGED' if args.owner_managed else 'QATTN_COLLABORATOR_FORMAL']='1'
    for profile in ('canary',) if args.canary_only else ('canary','discovery'):
        if args.resume and profile=='canary' and (root/profile/'RUN_COMPLETE').exists():
            p.audit(root/profile);continue
        command=[sys.executable,'-B','-m','baselines.retacred.rmea_token_transport_study','--profile',profile,'--data-dir',args.data_dir,'--assets',args.assets,'--output',str(root/profile)]
        if args.development_cache:command+=['--fixture-cache',args.development_cache]
        if args.owner_managed and profile=='discovery':command+=['--cos']
        if args.resume and (root/profile/'manifest.json').exists():command+=['--resume']
        if args.pause_after and not args.resume:command+=['--pause-after',str(args.pause_after)]
        subprocess.run(command,cwd=p.ROOT,env=env,check=True)
    handoff['parent_imported_torch_after_children']='torch' in sys.modules
    if handoff['parent_imported_torch_before_children'] or handoff['parent_imported_torch_after_children']:
        raise ValueError('Handoff control parent retained Torch')
    p.atomic_json(root/'handoff.json',handoff)
    if args.development_canary or args.owner_managed:
        target=root.parent/(root.name+'-safe-report');export(root,target,args.inject_export_failure)
        print(p.json.dumps(dict(development_canary=True,report=str(target))),flush=True)
    else:publish(root)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','run','export'))
    parser.add_argument('--data-dir',default='data/relation/retacred');parser.add_argument('--assets',default='.external/assets/722cf37b1afa9454edce342e7895e588b6ff1d59/model');parser.add_argument('--development-cache');parser.add_argument('--gpu',default='auto')
    parser.add_argument('--run-dir');parser.add_argument('--canary-only',action='store_true');parser.add_argument('--development-canary',action='store_true')
    parser.add_argument('--inject-export-failure',action='store_true')
    parser.add_argument('--owner-managed',action='store_true',help='Explicit owner seed13 exception; requires COS supervisor')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--pause-after',type=int,default=0);args=parser.parse_args(argv)
    if args.pause_after and not args.canary_only:parser.error('Fault injection is canary-only')
    if args.action in ('check','run') and not all((args.data_dir,args.assets)):parser.error('--data-dir and --assets required')
    if args.action=='check':print(p.json.dumps(isolated_probe(args)))
    elif args.action=='run':run(args)
    elif not args.run_dir:parser.error('export requires --run-dir')
    elif args.development_canary or args.owner_managed:
        root=Path(args.run_dir).resolve()
        if args.development_canary and not p.read(root/'handoff.json')['canary_only']:raise ValueError('Development export requires canary source')
        export(root,root.parent/(root.name+'-safe-report'),args.inject_export_failure)
    else:publish(args.run_dir)

if __name__=='__main__':main()
