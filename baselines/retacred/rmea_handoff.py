"""Portable collaborator entry and audited report-only Git delivery."""
import argparse
import datetime as dt
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from . import rmea_protocol as p

REPORT_FILES={'summary.json','manifest.json','trace_validation.json','receipt.json','run_summary.json','data.sha256','data_counts.txt'}

def branch_gate(allowed=None):
    if p.git('branch','--show-current')!='1.1':raise ValueError('Use collaborator branch1.1')
    for ref in ('origin/1.1','origin/main'):
        if subprocess.run(['git','merge-base','--is-ancestor',ref,'HEAD'],cwd=p.ROOT).returncode:raise ValueError('Fetch/integrate '+ref)
    entries=subprocess.check_output(['git','status','--porcelain=v1','-z','--untracked-files=all'],cwd=p.ROOT).split(b'\0')
    for entry in filter(None,entries):
        status=entry[:2].decode();name=entry[3:].decode()
        if not allowed or status not in ('??','A ') or not name.startswith(allowed+'/'):raise ValueError('Clean worktree required: '+name)

def probe(args):
    p.review_gate()
    if args.development_canary:
        if not args.canary_only:raise ValueError('Development mode is canary-only')
        if p.git('status','--porcelain'):raise ValueError('Clean development worktree required')
    else:
        branch_gate()
        from .rmea_exchange import preflight
        preflight()
    import importlib.metadata as md
    import torch
    if sys.version_info[:2]!=(3,10) or torch.__version__.split('+')[0] not in ('2.5.1','2.7.1') or md.version('numpy')!='1.26.4' or md.version('transformers')!='4.44.2':raise ValueError('Requires Python3.10, Torch2.5.1/2.7.1, NumPy1.26.4, Transformers4.44.2; do not silently change environment')
    if not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    indices=range(torch.cuda.device_count()) if args.gpu=='auto' else [int(args.gpu)]
    minimum=(7000*1024**2 if args.development_canary else 8*1024**3)
    candidates=[(torch.cuda.mem_get_info(i)[0],i) for i in indices if torch.cuda.mem_get_info(i)[0]>=minimum]
    if not candidates:raise ValueError('No eligible free GPU; wait, do not alter scientific budget')
    free,index=max(candidates);visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    mask=visible.split(',')[index].strip() if visible is not None else str(index)
    return dict(cuda_visible_devices=mask,gpu_name=torch.cuda.get_device_name(index),free_bytes_at_selection=free,execution_profile='low_memory_serial')

def isolated_probe(args):
    code='import argparse,json;from baselines.retacred.rmea_handoff import probe;print(json.dumps(probe(argparse.Namespace(**json.loads(__import__("sys").argv[1])))))'
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

def export(root,target,inject_failure=False,exchange_receipt=None):
    root=Path(root).resolve();target=Path(target).absolute()
    if root.is_relative_to(p.ROOT) or target.is_symlink():raise ValueError('Private source must stay outside checkout; regular report destination')
    handoff=p.read(root/'handoff.json');phase='canary' if handoff['canary_only'] else 'discovery'
    # Audit both actual parent phases; export retry never imports Torch or trains.
    for name in ('canary',) if phase=='canary' else ('canary','discovery'):
        p.audit(root/name)
    manifest,summary=p.audit(root/phase)
    if phase=='discovery' and (not exchange_receipt or exchange_receipt.get('status')!='uploaded' or exchange_receipt.get('closed_sha256')!=p.digest(root/phase/'RUN_COMPLETE') or not exchange_receipt.get('parts') or not all(r.get('verified') for r in exchange_receipt['parts'])):raise ValueError('Verified full private evidence exchange required before formal report')
    if p.read(root/'canary/manifest.json')['environment']!=manifest['environment']:raise ValueError('Canary and discovery environments differ')
    if manifest['git_revision']!=handoff['git_revision']:raise ValueError('Run revision differs')
    if inject_failure and phase!='canary':raise ValueError('Fault injection is canary-only')
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.'+target.name+'-',dir=target.parent))
    try:
        for name in ('summary.json','manifest.json','trace_validation.json'):shutil.copyfile(root/phase/name,temporary/name)
        if inject_failure:raise RuntimeError('Injected report copy failure; export-only retry from same completed source')
        p.atomic_json(temporary/'receipt.json',dict(closed_inventory_sha256=p.digest(root/phase/'RUN_COMPLETE'),source_sha256=manifest['source_sha256'],raw_artifacts_committed=False,private_exchange=exchange_receipt))
        (temporary/'data.sha256').write_text(''.join(f'{v["source_sha256"]}  {s}.jsonl\n' for s,v in manifest['data'].items()),encoding='utf8')
        (temporary/'data_counts.txt').write_text(''.join(f'{s} source={v["source_count"]} selected={v["selected_count"]}\n' for s,v in manifest['data'].items()),encoding='utf8')
        p.atomic_json(temporary/'run_summary.json',dict(schema='q-rmea.handoff-report.v1',data=manifest['data'],source_sha256=manifest['source_sha256'],
            git_revision=handoff['git_revision'],run_id=root.name,profile=phase,formal=phase=='discovery',
            sha256={f.name:p.digest(f) for f in temporary.iterdir()}))
        validate_report(temporary)
        if target.exists():
            validate_report(target)
            if {f.name:p.digest(f) for f in temporary.iterdir()}!={f.name:p.digest(f) for f in target.iterdir()}:raise ValueError('Existing final report differs')
        else:temporary.rename(target)
        return target
    finally:
        if temporary.exists():shutil.rmtree(temporary)

def publish(root):
    root=Path(root).resolve()
    if not re.fullmatch(r'\d{8}T\d{6}Z_seed13',root.name):raise ValueError('Expected timestamp_seed13 run basename')
    parent=p.ROOT/'reports/q_rmea';target=parent/root.name;relative=target.relative_to(p.ROOT).as_posix()
    if any(x.is_symlink() for x in (p.ROOT/'reports',parent,target)):raise ValueError('Symlinked report directory')
    p.review_gate()
    lock=Path(p.git('rev-parse','--git-path','q-rmea-publish.lock'))
    if not lock.is_absolute():lock=p.ROOT/lock
    with p.lock(lock):
        p.git('fetch','origin','+refs/heads/main:refs/remotes/origin/main','+refs/heads/1.1:refs/remotes/origin/1.1')
        branch_gate(relative)
        phase='canary' if p.read(root/'handoff.json')['canary_only'] else 'discovery'
        from .rmea_exchange import upload
        private_receipt=upload(root,root/phase)
        export(root,target,exchange_receipt=private_receipt);record=validate_report(target)
        baseline=record['git_revision'];head=p.git('rev-parse','HEAD');expected={relative+'/'+n for n in REPORT_FILES}
        hashes={name:p.digest(p.ROOT/name) for name in expected}
        if head==baseline:
            p.git('add','--',*sorted(expected))
            if set(p.git('diff','--cached','--name-only').splitlines())!=expected:raise ValueError('Unexpected staged changes')
            p.git('diff','--cached','--check');p.git('commit','-m','reports: Q-RMEA '+record['profile']+' '+root.name);head=p.git('rev-parse','HEAD')
        if p.git('rev-list','--parents','-n','1',head).split()!=[head,baseline]:raise ValueError('Only the single report commit may follow run revision')
        if set(p.git('diff','--name-only',baseline,head).splitlines())!=expected or set(p.git('diff','--name-only','--diff-filter=A',baseline,head).splitlines())!=expected:raise ValueError('Commit must add exactly audited report files')
        for name,sha in hashes.items():
            raw=subprocess.check_output(['git','show',head+':'+name],cwd=p.ROOT)
            if p.hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Committed bytes differ')
        branch_gate();validate_report(target)
        if p.git('rev-parse','HEAD')!=head:raise ValueError('HEAD changed before push')
        p.git('push','origin',head+':refs/heads/1.1')
        print(p.json.dumps(dict(published_report=relative,commit=head,branch='1.1')),flush=True)

def run(args):
    selection=isolated_probe(args)
    root=Path(args.run_dir).resolve() if args.run_dir else p.ROOT.parent/'q-rmea-private'/ (dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_seed13')
    if root.is_relative_to(p.ROOT) or root.exists():raise ValueError('Fresh private run required; interrupted training is not automatically resumed')
    root.mkdir(parents=True)
    p.atomic_json(root/'handoff.json',dict(git_revision=p.git('rev-parse','HEAD'),canary_only=args.canary_only,selection=selection))
    print('PRIVATE_RUN='+str(root),flush=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=selection['cuda_visible_devices'],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
    if not args.canary_only:env['QATTN_COLLABORATOR_FORMAL']='1'
    for profile in ('canary',) if args.canary_only else ('canary','discovery'):
        subprocess.run([sys.executable,'-B','-m','baselines.retacred.rmea_run','--profile',profile,'--data-dir',args.data_dir,'--assets',args.assets,'--output',str(root/profile)],cwd=p.ROOT,env=env,check=True)
    if args.development_canary:
        target=root.parent/(root.name+'-safe-report');export(root,target,args.inject_export_failure)
        print(p.json.dumps(dict(development_canary=True,report=str(target))),flush=True)
    else:publish(root)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','run','export'))
    parser.add_argument('--data-dir');parser.add_argument('--assets');parser.add_argument('--gpu',default='auto')
    parser.add_argument('--run-dir');parser.add_argument('--canary-only',action='store_true');parser.add_argument('--development-canary',action='store_true')
    parser.add_argument('--inject-export-failure',action='store_true');args=parser.parse_args(argv)
    if args.action in ('check','run') and not all((args.data_dir,args.assets)):parser.error('--data-dir and --assets required')
    if args.action=='check':print(p.json.dumps(isolated_probe(args)))
    elif args.action=='run':run(args)
    elif not args.run_dir:parser.error('export requires --run-dir')
    elif args.development_canary:
        root=Path(args.run_dir).resolve()
        if not p.read(root/'handoff.json')['canary_only']:raise ValueError('Development export requires canary source')
        export(root,root.parent/(root.name+'-safe-report'),args.inject_export_failure)
    else:publish(args.run_dir)

if __name__=='__main__':main()
