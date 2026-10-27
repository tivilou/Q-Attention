"""Five fresh seeds, one portable command, immutable export-only recovery."""
import argparse
import datetime as dt
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from . import rmea_confirmation_protocol as p
from .rmea_handoff import branch_gate

REPORT_FILES={'summary.json','manifest.json','trace_validation.json','receipt.json','run_summary.json','data.sha256','data_counts.txt'}

def phases(handoff):
    return [('canary',29)] if handoff['canary_only'] else [('canary',29)]+[('seed'+str(s),s) for s in p.SEEDS]

def probe(args):
    p.review_gate()
    from .rmea_handoff import probe as old_probe
    result=old_probe(args)
    result['confirmation_contract_sha256']=p.identity(p.contract())
    return result

def isolated_probe(args):
    code='import argparse,json,sys;from baselines.retacred.rmea_confirmation_handoff import probe;print(json.dumps(probe(argparse.Namespace(**json.loads(sys.argv[1])))))'
    r=subprocess.run([sys.executable,'-B','-c',code,p.legacy.json.dumps(vars(args))],cwd=p.ROOT,text=True,capture_output=True)
    if r.returncode:
        print(r.stderr,file=sys.stderr,end='');raise ValueError('Confirmation preflight failed before training')
    return p.legacy.json.loads(r.stdout)

def audit_batch(root):
    root=Path(root).resolve()
    if root.is_relative_to(p.ROOT) or root.is_symlink():raise ValueError('Private source outside checkout required')
    handoff=p.read(root/'handoff.json')
    if handoff.get('contract_sha256')!=p.identity(p.contract()) or handoff.get('source_sha256')!=p.identity(p.sources()):raise ValueError('Confirmation handoff contract/source differs')
    runs={};env=None
    for name,seed in phases(handoff):
        m,s=p.audit(root/name)
        if m['config']!=p.config('canary' if name=='canary' else 'discovery',seed) or m['git_revision']!=handoff['git_revision']:raise ValueError('Phase/seed/revision differs')
        if env is not None and m['environment']!=env:raise ValueError('Seed/canary environments differ')
        env=m['environment'];runs[name]=(m,s)
    return handoff,runs

def summary(runs,canary):
    selected=['canary'] if canary else ['seed'+str(s) for s in p.SEEDS]
    result=dict(schema='q-rmea.confirmation.results.v1',formal=not canary,scope='technical_only' if canary else 'independent_confirmation_pending_owner_audit',
      contract=p.contract(),per_seed={str(runs[n][0]['config']['seed']):runs[n][1] for n in selected},discovery_seed13_included=False,
      evidence_tier='pending_owner_audit',automatic_promotion=False)
    if not canary:
        ds=[runs[n][1]['results'] for n in selected]
        utility=p.paired([d['quantum']['test']['macro_f1']-d['uniform']['test']['macro_f1'] for d in ds])
        mechanism=p.paired([d['quantum']['test']['diagnosis']['shuffle_minus_actual_ce'] for d in ds])
        invariance=all(abs(d['uniform']['test']['diagnosis']['shuffle_minus_actual_ce'])<=2e-5 and all(v['test']['diagnosis']['max_identity_residual']<=2e-5 for v in d.values()) for d in ds)
        result.update(utility_statistics=utility,mechanism_statistics=mechanism,identity_and_uniform_invariance=invariance,
          candidate_gate_outcomes=dict(reproducible_utility=utility['paired_t_gate'],mechanism_consistency=mechanism['paired_t_gate'] and invariance),
          attribution_contrasts={m:p.paired([d['quantum']['test']['macro_f1']-d[m]['test']['macro_f1'] for d in ds]) for m in ('relation_softmax','real_overlap','dephased')})
    return result

def validate_report(target):
    if target.is_symlink() or not target.is_dir() or {f.name for f in target.iterdir()}!=REPORT_FILES:raise ValueError('Report whitelist differs')
    if any(f.is_symlink() or not f.is_file() for f in target.iterdir()):raise ValueError('Nonregular report')
    r=p.read(target/'run_summary.json')
    if r.get('schema')!='q-rmea.confirmation-report.v1' or r.get('contract_sha256')!=p.identity(p.contract()) or set(r['sha256'])!=REPORT_FILES-{'run_summary.json'}:raise ValueError('Report contract differs')
    if any(p.digest(target/n)!=sha for n,sha in r['sha256'].items()):raise ValueError('Report hash differs')
    return r

def export(root,target,inject_failure=False,exchange_receipts=None):
    root=Path(root).resolve();target=Path(target).absolute();handoff,runs=audit_batch(root)
    if target.is_symlink():raise ValueError('Symlinked report')
    canary=handoff['canary_only']
    if inject_failure and not canary:raise ValueError('Canary-only fault injection')
    if not canary:
        for name,_ in phases(handoff):
            receipt=(exchange_receipts or {}).get(name)
            if not receipt or receipt.get('status')!='uploaded' or receipt.get('closed_sha256')!=p.digest(root/name/'RUN_COMPLETE') or not receipt.get('parts') or not all(v.get('verified') for v in receipt['parts']):raise ValueError('Every closed private phase requires verified exchange')
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.'+target.name+'-',dir=target.parent))
    try:
        p.atomic_json(temporary/'summary.json',summary(runs,canary))
        p.atomic_json(temporary/'manifest.json',dict(schema='q-rmea.confirmation.manifest.v1',handoff=handoff,phases={k:v[0] for k,v in runs.items()}))
        if inject_failure:raise RuntimeError('Injected confirmation report copy failure; export-only retry')
        p.atomic_json(temporary/'trace_validation.json',{k:p.read(root/k/'trace_validation.json') for k in runs})
        p.atomic_json(temporary/'receipt.json',dict(closed_inventories={k:p.digest(root/k/'RUN_COMPLETE') for k in runs},private_exchange=exchange_receipts,raw_artifacts_committed=False))
        (temporary/'data.sha256').write_text(''.join(f'{v[1]}  {k}.jsonl\n' for k,v in p.EXPECTED.items()),encoding='utf8')
        (temporary/'data_counts.txt').write_text(''.join(f'{k} source={v[0]} formal_selected={v[0]}\n' for k,v in p.EXPECTED.items()),encoding='utf8')
        p.atomic_json(temporary/'run_summary.json',dict(schema='q-rmea.confirmation-report.v1',run_id=root.name,formal=not canary,seeds=[29] if canary else list(p.SEEDS),
          git_revision=handoff['git_revision'],source_sha256=handoff['source_sha256'],contract_sha256=handoff['contract_sha256'],
          sha256={f.name:p.digest(f) for f in temporary.iterdir()}))
        validate_report(temporary)
        if target.exists():
            validate_report(target)
            if {f.name:p.digest(f) for f in target.iterdir()}!={f.name:p.digest(f) for f in temporary.iterdir()}:raise ValueError('Existing report differs')
        else:temporary.rename(target)
        return target
    finally:
        if temporary.exists():shutil.rmtree(temporary)

def publish(root):
    root=Path(root).resolve()
    if not re.fullmatch(r'\d{8}T\d{6}Z_fresh5',root.name):raise ValueError('Expected timestamp_fresh5 basename')
    parent=p.ROOT/'reports/q_rmea_confirmation';target=parent/root.name;relative=target.relative_to(p.ROOT).as_posix()
    if any(x.is_symlink() for x in (p.ROOT/'reports',parent,target)):raise ValueError('Symlinked report path')
    p.review_gate();handoff,runs=audit_batch(root)
    lock=Path(p.git('rev-parse','--git-path','q-rmea-confirmation-publish.lock'))
    if not lock.is_absolute():lock=p.ROOT/lock
    with p.lock(lock):
        p.git('fetch','origin','+refs/heads/main:refs/remotes/origin/main','+refs/heads/1.1:refs/remotes/origin/1.1');branch_gate(relative)
        from .rmea_confirmation_exchange import upload
        receipts={name:upload(root,root/name) for name in runs}
        export(root,target,exchange_receipts=receipts);record=validate_report(target)
        baseline=record['git_revision'];head=p.git('rev-parse','HEAD');expected={relative+'/'+n for n in REPORT_FILES};hashes={n:p.digest(p.ROOT/n) for n in expected}
        if head==baseline:
            p.git('add','--',*sorted(expected))
            if set(p.git('diff','--cached','--name-only').splitlines())!=expected:raise ValueError('Unexpected staged changes')
            p.git('diff','--cached','--check');p.git('commit','-m','reports: Q-RMEA independent confirmation '+root.name);head=p.git('rev-parse','HEAD')
        if p.git('rev-list','--parents','-n','1',head).split()!=[head,baseline]:raise ValueError('Exactly one report commit after executed revision required')
        if set(p.git('diff','--name-only',baseline,head).splitlines())!=expected or set(p.git('diff','--name-only','--diff-filter=A',baseline,head).splitlines())!=expected:raise ValueError('Report-only additions required')
        for n,sha in hashes.items():
            if p.legacy.hashlib.sha256(subprocess.check_output(['git','show',head+':'+n],cwd=p.ROOT)).hexdigest()!=sha:raise ValueError('Committed report differs')
        branch_gate();validate_report(target)
        if p.git('rev-parse','HEAD')!=head:raise ValueError('HEAD changed before push')
        p.git('push','origin',head+':refs/heads/1.1')
        print(p.legacy.json.dumps(dict(published_report=relative,commit=head,branch='1.1')),flush=True)

def run(args):
    selection=isolated_probe(args)
    root=Path(args.run_dir).resolve() if args.run_dir else p.ROOT.parent/'q-rmea-confirmation-private'/(dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_fresh5')
    if root.is_relative_to(p.ROOT) or root.exists():raise ValueError('Fresh private batch required; partial training is never automatically resumed')
    root.mkdir(parents=True)
    handoff=dict(git_revision=p.git('rev-parse','HEAD'),canary_only=args.canary_only,selection=selection,source_sha256=p.identity(p.sources()),contract_sha256=p.identity(p.contract()))
    p.atomic_json(root/'handoff.json',handoff);print('PRIVATE_RUN='+str(root),flush=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=selection['cuda_visible_devices'],OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1')
    if not args.canary_only:env['QATTN_COLLABORATOR_FORMAL']='1'
    for name,seed in phases(handoff):
        subprocess.run([sys.executable,'-B','-m','baselines.retacred.rmea_confirmation_run','--profile','canary' if name=='canary' else 'discovery',
          '--seed',str(seed),'--data-dir',args.data_dir,'--assets',args.assets,'--output',str(root/name)],cwd=p.ROOT,env=env,check=True)
    if args.development_canary:
        export(root,root.parent/(root.name+'-safe-report'),args.inject_export_failure)
    else:publish(root)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('check','run','export'))
    parser.add_argument('--data-dir');parser.add_argument('--assets');parser.add_argument('--gpu',default='auto');parser.add_argument('--run-dir')
    parser.add_argument('--canary-only',action='store_true');parser.add_argument('--development-canary',action='store_true');parser.add_argument('--inject-export-failure',action='store_true')
    args=parser.parse_args(argv)
    if args.action in ('check','run') and not all((args.data_dir,args.assets)):parser.error('--data-dir and --assets required')
    if args.action=='check':print(p.legacy.json.dumps(isolated_probe(args)))
    elif args.action=='run':run(args)
    elif not args.run_dir:parser.error('export requires --run-dir')
    elif args.development_canary:
        root=Path(args.run_dir).resolve()
        if not p.read(root/'handoff.json')['canary_only']:raise ValueError('Development export requires canary source')
        export(root,root.parent/(root.name+'-safe-report'),args.inject_export_failure)
    else:publish(args.run_dir)

if __name__=='__main__':main()
