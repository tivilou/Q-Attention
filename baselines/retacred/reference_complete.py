"""Portable complete reference queue. Formal data runs are collaborator-only."""
from __future__ import annotations
import argparse,json,os,sys,subprocess,signal,time,shutil,tempfile
from datetime import datetime,timezone
from pathlib import Path
from collections import deque
from .common import ROOT,PACKAGE,atomic_json,digest,fingerprint,cohort,git
from .reference_recipes import recipe,inventory

REVIEW_FILES=('reference_complete.py','reference_recipes.py','reference_models.py','reference_mechanisms.py',
 'reference_storage.py','reference_archive.py',
 'reference_preflight.py','reference_worker.py','reference_nli.py','reference_trace.py','reference_resources.py','reference_resources.lock.json',
 'ctl_drp_runner.py','ctl_drp_task.py','ctl_drp_neural.py','ctl_drp_trace.py','ctl_drp_pretrained.py',
 'assets.py','luke_assets.py','mnli_assets.py','train.py','preflight.py','cohort.json','requirements-reference-complete.txt','../../scripts/run_retacred_all_references.sh','../../scripts/check_retacred_all_references.sh',
 'assets.lock.json','luke_assets.lock.json','metaentail_assets.lock.json','metaentail_paper.lock.json','sequence_routing_resources.lock.json',
 'metaentail.py','metaentail_neural.py','luke.py','ptr.py','improved.py','report.py','common.py','orchestrate.py')

def identity():return {name:digest(PACKAGE/name) for name in REVIEW_FILES}

def require_review(path):
    value=json.loads(Path(path).read_text())
    if value.get('schema')!='retacred-reference-independent-review.v1' or value.get('status')!='accepted' or value.get('reviewer_role')!='independent' or not value.get('reviewer') or value.get('source_sha256')!=identity():
        raise ValueError('accepted independent mechanism+lifecycle review must bind current source')
    return digest(path)

def model_ids(raw):
    names=[r['id'] for r in cohort()['models'] if r.get('run_eligible',True)] if raw=='all' else cohort()['default_models'] if raw=='ready' else raw.split(',')
    allowed={r['id'] for r in cohort()['models'] if r.get('run_eligible',True)}
    if not names or len(set(names))!=len(names) or any(n not in allowed for n in names):raise ValueError('unknown, duplicate or excluded model')
    return names

def finish_summary(run,model,seed,r,splits,updates,traces,data_dir,assets,*,diagnostic=False):
    manifest=json.loads((Path(data_dir)/'data_manifest.json').read_text())
    summary={'schema':'retacred-reference-result.v2','status':'complete','model':model,'seed':seed,'updates':updates,
        'splits':splits,'contract':json.loads((run/'contract.json').read_text()),'data_manifest':manifest,
        'adaptations':r.get('adaptations',[]),'fidelity':r['fidelity'],'environment':{'torch':__import__('torch').__version__,'transformers':__import__('transformers').__version__},
        'case_study':{'status':'complete_local','traces':traces},'diagnostic':diagnostic,
        'paper_score_reproduced':False,'recipe':r,'assets_identity':assets.get('resource_lock_sha256')}
    atomic_json(run/'run_summary.json',summary);atomic_json(run/'metrics.json',splits)
    (run/('DIAGNOSTIC_COMPLETE' if diagnostic else 'RUN_COMPLETE')).touch()

def prepare_data(source,run):
    prepared=run/'prepared_data'
    if (prepared/'dataset_qualification.json').exists():
        q=json.loads((prepared/'dataset_qualification.json').read_text());m=json.loads((prepared/'data_manifest.json').read_text())
        for s,f in [('train','train.json'),('valid','dev.json'),('test','test.json')]:
            if digest(prepared/f)!=q['splits'][s]['output_sha256'] or digest(source/(s+'.jsonl'))!=q['splits'][s]['source_sha256']:raise ValueError('qualified data changed')
        return prepared
    if prepared.exists():raise ValueError('incomplete prepared cache preserved; inspect before replacing')
    command=[sys.executable,str(ROOT/'experiments/prepare_retacred_reference_improved_baseline.py')]
    for split in ('train','valid','test'):command+=['--'+split+'-path',str(source/(split+'.jsonl'))]
    command+=['--output-dir',str(prepared)]
    subprocess.run(command,cwd=ROOT,check=True)
    (prepared/'valid.json').rename(prepared/'dev.json');manifest=json.loads((prepared/'data_manifest.json').read_text())
    for split,name in [('train','train.json'),('valid','dev.json'),('test','test.json')]:
        path=prepared/name;rows=json.loads(path.read_text());seen=set()
        for i,row in enumerate(rows):
            if 'id' not in row:row['id']=f'reference:{split}:{i}:{fingerprint(row)}'
            if not isinstance(row['id'],str) or not row['id'] or row['id'] in seen:raise ValueError('invalid/duplicate source row ID')
            seen.add(row['id'])
        atomic_json(path,rows);manifest['splits'][split]['output_path']=str(path);manifest['splits'][split]['output_sha256']=digest(path)
    if [manifest['splits'][s]['records'] for s in ('train','valid','test')]!=[58465,19584,13418]:raise ValueError('complete Re-TACRED split counts required')
    manifest['interface_adaptations']={'valid_to_dev':True,'missing_ids':'split_ordinal_content_sha256_no_label_change'}
    atomic_json(prepared/'data_manifest.json',manifest)
    atomic_json(prepared/'dataset_qualification.json',{'status':'qualified','task':'supervised_sentence_entity_pair_relation_classification',
        'labels':40,'null_relation':'no_relation','splits':manifest['splits'],
        'verification':'converter_validates_labels_entity_spans_types_and_full_split_counts'})
    return prepared

def validate_task(path):
    path=Path(path)
    if not (path/'RUN_COMPLETE').is_file():raise ValueError('worker completion missing')
    s=json.loads((path/'run_summary.json').read_text())
    from .reference_archive import load as load_archive
    archived=load_archive(path)
    if archived and archived['status']=='complete' and (archived.get('summary_sha256')!=digest(path/'run_summary.json') or archived.get('validated_trace_identity')!=fingerprint(s['case_study']['traces'])):
        raise ValueError('sealed completed archive/result identity drift')
    if s['status']!='complete' or s.get('diagnostic') or s['case_study']['status']!='complete_local':raise ValueError('not a completed formal task')
    from .reference_trace import validate
    if s['model'].startswith('ctl_drp_'):
        from .ctl_drp_trace import validate as ctl_validate
        for item in s['case_study']['traces']:ctl_validate(path/'case_study'/item['path'],layers=24)
    else:validate(path,s['case_study']['traces'])
    contract=s['contract']
    if s['schema']!='retacred-reference-result.v2' or s['recipe']!=recipe(s['model']) or s['seed']!=contract.get('seed',contract.get('task',{}).get('seed')):
        raise ValueError('completed task recipe or seed drift')
    budget=contract.get('budget',contract.get('budget_updates'))
    if budget is None and s['model'].startswith('ctl_drp_'):budget=1150
    if s['updates']!=budget:raise ValueError('completed task update budget mismatch')
    for split,expected in [('valid',19584),('test',13418)]:
        if s['splits'][split]['items']!=expected:raise ValueError('incomplete evaluation denominator')
    return s

def command(job,args,assets,prepared):
    module='baselines.retacred.reference_complete' if job['model'].startswith('ctl_drp_') else 'baselines.retacred.reference_worker'
    argv=[sys.executable,'-u','-m',module,'--model',job['model'],'--seed',str(job['seed']),
        '--data-dir',str(prepared),'--assets',str(assets),'--run-dir',str(job['directory']),'--micro-batch',str(args.micro_batch)]
    if module.endswith('reference_complete'):argv+=['--ctl-worker','--review-receipt',str(args.review_receipt)]
    if (job['directory']/'checkpoints/latest.pt').exists() or (job['directory']/'contract.json').exists():argv.append('--resume')
    return argv

def queue(jobs,devices,args,assets,prepared):
    pending=deque(jobs);active={};paused=[False];failures=[];done=[]
    handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    for s in handlers:signal.signal(s,lambda *_:paused.__setitem__(0,True))
    try:
        while pending or active:
            if paused[0]:break
            for gpu in devices:
                if gpu in active or not pending:continue
                if os.environ.get('QATTN_REFERENCE_STORAGE_ROOT'):
                    from .reference_storage import task_peak, owned_bytes, FLOOR, durable_json
                    free=shutil.disk_usage(args.run_dir).free
                    candidate=pending[0]
                    committed=sum(max(0,task_peak(j['model'])-owned_bytes(j['directory'])) for _,j,_ in active.values())
                    additional=max(0,task_peak(candidate['model'])-owned_bytes(candidate['directory']))
                    if free<additional+committed+FLOOR:
                        if active:continue
                        durable_json(args.run_dir/'STORAGE_PAUSED.json',{'status':'paused','task':candidate['id'],
                            'free_bytes':free,'additional_estimate_bytes':additional,'recovery':'free space, then resume same suite'})
                        paused[0]=True;break
                job=pending.popleft();log=args.run_dir/'logs'/f"{job['id']}.log";log.parent.mkdir(parents=True,exist_ok=True)
                handle=log.open('a');argv=command(job,args,assets,prepared)
                atomic_json(log.with_suffix('.command.json'),{'argv':argv,'cwd':str(ROOT),'gpu':gpu})
                env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'CUDA_DEVICE_ORDER':'PCI_BUS_ID','PYTHONDONTWRITEBYTECODE':'1',
                    'USE_TF':'0','USE_FLAX':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
                process=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                active[gpu]=(process,job,handle)
            for gpu,(process,job,handle) in list(active.items()):
                rc=process.poll()
                if rc is None:continue
                handle.close();del active[gpu]
                if rc==0:
                    try:
                        validate_task(job['directory'])
                        if os.environ.get('QATTN_REFERENCE_STORAGE_ROOT'):
                            from .reference_archive import archive_completed
                            archive_completed(job['directory'])
                        done.append(job['id'])
                    except Exception as error:failures.append({'task':job['id'],'error':str(error)})
                elif rc==75:paused[0]=True
                else:failures.append({'task':job['id'],'exit_code':rc})
            if failures:break
            rows=[]
            for gpu,(_,job,_) in active.items():
                hb=job['directory']/'heartbeat.json'
                try:value=json.loads(hb.read_text()) if hb.exists() else {}
                except json.JSONDecodeError:value={}
                rows.append({'gpu':gpu,'task':job['id'],**value})
            atomic_json(args.run_dir/'queue_status.json',{'completed':done,'active':rows,'pending':[j['id'] for j in pending]})
            print(f"[reference cohort] {len(done)}/{len(jobs)} | running {len(active)} | pending {len(pending)}",flush=True)
            for row in rows:print(f"  GPU {row['gpu']} | {row['task']} | {row.get('phase','initializing')} | {row.get('batch','-')}/{row.get('batches','-')} | {row.get('batches_per_second',0):.2f} batch/s | VRAM {row.get('memory_allocated_bytes',0)/1024**3:.2f}GiB",flush=True)
            time.sleep(10 if active else .1)
    finally:
        for process,_,_ in active.values():
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for process,_,handle in active.values():
            try:process.wait(timeout=60)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            handle.close()
        for sig,handler in handlers.items():signal.signal(sig,handler)
    if failures or paused[0]:
        atomic_json(args.run_dir/'RUN_FAILED.json',{'failures':failures,'paused':paused[0],'training_restarts':False})
        return 75 if paused[0] else 1
    return 0

def ctl_worker(args):
    require_review(args.review_receipt)
    if (args.run_dir/'RUN_COMPLETE').exists():validate_task(args.run_dir);return 0
    from .ctl_drp_qualification import materialize
    from .ctl_drp_task import CTLDRPTask
    from .ctl_drp_runner import run
    assets=json.loads(args.assets.read_text());variant='sup' if args.models=='ctl_drp_supervised' else 'unsup'
    q=materialize(args.data_dir/'dataset_qualification.json',prepared=args.data_dir,output=args.run_dir.with_name(args.run_dir.name+'.physical'))
    adapter=CTLDRPTask.from_assets(q,assets['ctl_cache'],assets['ctl_paper'],assets['ctl_'+variant],variant=variant,seed=args.seed,micro_batch=args.micro_batch,device='cuda:0')
    # Normalize only after native compute validation. An interrupted normalization
    # never sends the generic schema back into native checkpoint validation.
    if (args.run_dir/'native_run_summary.json').exists() and not (args.run_dir/'RUN_COMPLETE').exists():
        atomic_json(args.run_dir/'run_summary.json',json.loads((args.run_dir/'native_run_summary.json').read_text()))
    result=run(adapter,args.run_dir,resume=args.resume is not None,formal_review_receipt=args.review_receipt)
    if result['status']=='paused':return 75
    traces=[dict(t,path=t['name']) for t in result['case_study']['traces']]
    splits={'valid':result['splits']['dev'],'test':result['splits']['test']}
    atomic_json(args.run_dir/'native_run_summary.json',result)
    finish_summary(args.run_dir,args.models,args.seed,recipe(args.models),splits,result['updates'],traces,args.data_dir,assets)
    return 0

def _main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models','--model',default='all');p.add_argument('--seeds',default='13');p.add_argument('--seed',type=int,default=13)
    p.add_argument('--gpu',default='auto');p.add_argument('--micro-batch',type=int,default=1)
    p.add_argument('--run-dir',type=Path);p.add_argument('--resume',type=Path,nargs='?',const=Path('.'))
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/relation/retacred');p.add_argument('--assets',type=Path)
    p.add_argument('--review-receipt',type=Path,default=PACKAGE/'reference_review.json')
    p.add_argument('--dry-run','--describe',action='store_true');p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--export-only',action='store_true');p.add_argument('--no-push',action='store_true');p.add_argument('--ctl-worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    if args.ctl_worker:return ctl_worker(args)
    names=model_ids(args.models);seeds=[int(x) for x in args.seeds.split(',')]
    if args.micro_batch<1 or not seeds or len(seeds)!=len(set(seeds)) or min(seeds)<0:raise ValueError('invalid seeds/physical batch')
    if any(n.startswith('ctl_drp_') for n in names) and any(s not in (13,29,53) for s in seeds):raise ValueError('CTL frozen seed set is13,29,53')
    if args.dry_run:
        print(json.dumps({'models':names,'configurations':len(names),'seeds':seeds,'recipes':[recipe(n) for n in names],
            'not_a_certified_top15':True,'formal_runs_started':0,'excluded':[r['id'] for r in cohort()['models'] if not r.get('run_eligible',True)]},indent=2));return 0
    require_review(args.review_receipt)
    if args.export_only and not args.resume:raise ValueError('export-only requires completed suite resume directory')
    args.run_dir=(args.resume or args.run_dir or ROOT/'runs/retacred_reference_complete'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
    if not args.run_dir.is_relative_to(ROOT/'runs'):raise ValueError('suite directory must stay inside runs')
    from .reference_worker import exclusive
    with exclusive(args.run_dir):return suite(args,names,seeds)

def suite(args,names,seeds):
    from .report import publication_preflight
    report=ROOT/'reports/retacred_reference_suite'/args.run_dir.name
    publication_preflight(no_push=args.no_push,allowed_report=report if args.export_only else None)
    contract={'schema':'retacred-reference-suite.v2','models':names,'seeds':seeds,'source':identity(),
        'physical_micro':args.micro_batch,'data':{s:digest(args.data_dir/(s+'.jsonl')) for s in ('train','valid','test')}}
    manifest=args.run_dir/'run_manifest.json'
    if args.resume:
        if json.loads(manifest.read_text())!=contract:raise ValueError('immutable suite mismatch; only GPU assignment may change')
    elif args.run_dir.exists():raise ValueError('existing suite requires resume')
    args.run_dir.mkdir(parents=True,exist_ok=True);atomic_json(manifest,contract)
    from .reference_storage import configure
    configure(args.run_dir)
    jobs=[{'id':f'{n}_seed{s}','model':n,'seed':s,'directory':args.run_dir/'tasks'/f'{n}_seed{s}'} for n in names for s in seeds]
    if not args.export_only:
        from .reference_preflight import check
        check(names,seeds,args.data_dir,args.gpu,review_receipt=args.review_receipt,resume=args.run_dir if args.resume else None)
        from .reference_resources import prepare
        from .orchestrate import gpu_ids
        prepared=prepare_data(args.data_dir,args.run_dir);assets=prepare(names);asset_path=args.run_dir/'assets.private.json';atomic_json(asset_path,assets)
        if args.prepare_only:print('[references] inputs and assets prepared; no training started');return 0
        devices=gpu_ids(args.gpu);remaining=[]
        for job in jobs:
            if (job['directory']/'RUN_COMPLETE').exists():
                validate_task(job['directory'])
                from .reference_archive import archive_completed
                archive_completed(job['directory'])
            else:remaining.append(job)
        result=queue(remaining,devices,args,asset_path,prepared)
        if result:return result
    for job in jobs:validate_task(job['directory'])
    (args.run_dir/'RUN_COMPLETE').touch()
    from .report import export_and_publish
    # The exporter only sees completed normalized worker contracts, never raw
    # checkpoints, datasets or predictions. It requires private numeric upload.
    export_and_publish(args.run_dir,jobs,no_push=args.no_push)
    return 0

def main(argv=None):
    from .reference_storage import pause_cli, ENV
    before=os.environ.get(ENV)
    try:return pause_cli(lambda:_main(argv))
    finally:
        if before is None:os.environ.pop(ENV,None)
        else:os.environ[ENV]=before

if __name__=='__main__':raise SystemExit(main())
