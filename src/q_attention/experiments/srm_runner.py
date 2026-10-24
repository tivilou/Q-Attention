"""Serial parent/worker bounded experiment with committed optimizer cursors."""
from __future__ import annotations
import argparse
import copy
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile

from q_attention.experiments import srm_protocol as p


def completed(directory,manifest,seed,mode):
    directory=Path(directory);marker=directory/'RUN_COMPLETE.json'
    if not marker.exists():return False
    record=p.read(marker)
    if record['fingerprint']!=manifest['fingerprint'] or record['seed']!=seed or record['mode']!=mode:
        raise ValueError('completed phase identity differs')
    required={'summary.json','best.pt','latest.json'}|({'sample_trace.json'} if mode!='warmup' else set())
    if set(record['files'])!=required:raise ValueError('completed artifact inventory differs')
    for name,digest in record['files'].items():
        if name not in ('summary.json','best.pt','sample_trace.json','latest.json') or p.sha(directory/name)!=digest:
            raise ValueError('completed artifact changed: '+name)
    from q_attention.experiments import cfra_checkpoint as cp
    state=cp.load(directory)
    if (state['fingerprint']!=manifest['fingerprint'] or state['next_update']!=record['updates']
        or state['schema_version']!='srm-worker.v1' or state['seed']!=seed or state['mode']!=mode):
        raise ValueError('completed checkpoint identity/cursor differs')
    if mode!='warmup':
        from q_attention.analysis.srm_trace import validate_trace
        validate_trace(directory/'sample_trace.json')
    return True


class Worker:
    def __init__(self,root,seed,mode,*,stop_after=None):
        import torch
        from q_attention.models.relation_transformer import RelationExtractionModel,RelationTransformerConfig
        from q_attention.adapters.q_srm import SRMModel
        from q_attention.analysis.srm_trace import Trace,state_sha
        from q_attention.experiments import cfra_checkpoint as cp
        from q_attention.experiments.ecpa_protocol import seed_all,restore_rng
        self.root,self.seed,self.mode=Path(root),seed,mode
        self.manifest=p.read(self.root/'run_manifest.json');self.cfg=self.manifest['config']
        expected=p.config('canary' if self.cfg['profile']=='synthetic_canary' else 'screen')
        if self.cfg!=expected or mode not in ('warmup',*p.MODES) or seed not in self.cfg['seeds']:
            raise ValueError('worker config/seed/mode outside frozen contract')
        p.validate_review()
        self.device=self.manifest['device']
        if p.environment(self.device)!=self.manifest['environment']:raise ValueError('runtime environment changed')
        self.rows,self.datasets,self.vocab,self.labels,data=p.load_data(self.manifest['data_dir'],self.cfg)
        identity=dict(config=self.cfg,data=data,source=p.identity(p.source_contract()),environment=self.manifest['environment'])
        if p.identity(identity)!=self.manifest['fingerprint']:raise ValueError('source/data/config identity differs')
        if stop_after is not None and self.cfg['profile']!='synthetic_canary':raise ValueError('stop injection is canary-only')
        self.stop_after=stop_after
        self.directory=self.root/f'seed{seed}'/mode;self.directory.mkdir(parents=True,exist_ok=True)
        seed_all(seed)
        base=RelationExtractionModel(RelationTransformerConfig(vocab_size=len(self.vocab),num_labels=len(self.labels),
            max_length=self.cfg['max_length'],**self.cfg['model']))
        self.base_hash=None
        if mode=='warmup':self.model=base
        else:
            warm=self.root/f'seed{seed}/warmup'
            if not completed(warm,self.manifest,seed,'warmup'):raise ValueError('shared warm-start incomplete')
            state=torch.load(warm/'best.pt',map_location='cpu',weights_only=False)
            base.load_state_dict(state['model']);self.base_hash=state_sha(base)
            seed_all(seed+1000);self.model=SRMModel(base,mode)
        self.model.to(self.device)
        self.budget=self.cfg['warmup' if mode=='warmup' else 'screen']
        self.params=[x for x in self.model.parameters() if x.requires_grad]
        self.optimizer=torch.optim.Adam(self.params,lr=self.budget['lr'])
        self.per_epoch=math.ceil(len(self.rows['train'])/self.budget['batch_size'])
        self.total=self.per_epoch*self.budget['epochs']
        self.update,self.selected,self.best,self.best_metric,self.best_update=0,0,None,-math.inf,0
        self.writer=None
        if mode!='warmup':
            names=[dict(name=n,shape=list(v.shape),numel=v.numel()) for n,v in self.model.named_parameters() if v.requires_grad]
            self.writer=Trace(self.directory,mode,dict(run_id=self.root.name,seed=seed,model_identity=mode,
                profile=self.cfg['profile'],config_sha256=p.identity(self.cfg),dataset_identity=p.identity(data),
                code_revision=self.manifest['source_sha256'],dataset='synthetic canary' if self.cfg['profile']=='synthetic_canary' else 'bounded Re-TACRED subset'),self.labels,names)
        if cp.exists(self.directory):
            state=cp.load(self.directory)
            if (state['schema_version']!='srm-worker.v1' or state['fingerprint']!=self.manifest['fingerprint']
                or state['seed']!=seed or state['mode']!=mode or state['base_state_sha256']!=self.base_hash):
                raise ValueError('resume phase identity differs')
            self.model.load_state_dict(state['model']);self.optimizer.load_state_dict(state['optimizer'])
            self.update,self.selected,self.best,self.best_metric,self.best_update=(state[k] for k in
                ('next_update','selected_through','best_model','best_metric','best_update'))
            if not 0<=self.update<=self.total or state['next_indexes']!=self.indexes():raise ValueError('resume sampler cursor differs')
            if self.writer and {s['split'] for s in self.writer.payload['samples'] if s['checkpoint']=='initial'}!={'train','valid','test'}:
                raise ValueError('committed checkpoint lacks initial traces')
            restore_rng(state['rng'])
        else:
            if self.writer:self.snapshot('initial',0)
            self.save()

    def indexes(self):return p.update_indexes(len(self.rows['train']),self.budget['batch_size'],self.update,self.seed)

    def save(self):
        from q_attention.experiments import cfra_checkpoint as cp
        from q_attention.experiments.ecpa_protocol import rng_state
        cp.save(self.directory,dict(schema_version='srm-worker.v1',fingerprint=self.manifest['fingerprint'],seed=self.seed,mode=self.mode,
            base_state_sha256=self.base_hash,next_update=self.update,selected_through=self.selected,next_indexes=self.indexes(),
            best_model=self.best,best_metric=self.best_metric,best_update=self.best_update,
            model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),rng=rng_state()))

    def batch(self,split,indexes):
        from q_attention.tasks.relation import collate_relation_batch
        return {k:v.to(self.device) for k,v in collate_relation_batch([self.datasets[split][i] for i in indexes]).items()}

    @staticmethod
    def inputs(batch):return {k:batch[k] for k in ('input_ids','attention_mask','subject_mask','object_mask')}

    def evaluate(self,split):
        import torch
        from q_attention.experiments.ecpa_protocol import metrics
        self.model.eval();predicted=[];gold=[];loss=0.
        with torch.no_grad():
            size=self.cfg['runtime']['evaluation_batch']
            for start in range(0,len(self.rows[split]),size):
                batch=self.batch(split,range(start,min(start+size,len(self.rows[split]))))
                logits=self.model(**self.inputs(batch))
                loss+=float(torch.nn.functional.cross_entropy(logits,batch['labels'],reduction='sum'))
                predicted.extend(logits.argmax(-1).cpu().tolist());gold.extend(batch['labels'].cpu().tolist())
        return metrics(predicted,gold,self.labels,loss)

    def snapshot(self,checkpoint,updates):
        from q_attention.adapters.q_srm import objective
        from q_attention.analysis.srm_trace import vector,state_sha,values
        self.model.eval();identity=state_sha(self.model)
        for split in ('train','valid','test'):
            self.model.zero_grad(set_to_none=True);batch=self.batch(split,[0]);before=vector(self.model)
            loss,_=objective(self.model(**self.inputs(batch),capture=True),batch['labels']);loss.backward()
            self.writer.add(self.rows[split][0],split,checkpoint,'replay',
                values(self.model,batch['labels'],loss,before,vector(self.model,True),before),updates,identity)
        self.model.zero_grad(set_to_none=True)

    def train_update(self):
        import torch
        from q_attention.analysis.srm_trace import vector,state_sha,values
        self.model.train();batch=self.batch('train',self.indexes());count=len(batch['labels'])
        self.optimizer.zero_grad(set_to_none=True);capture=self.update==0 and self.writer is not None
        before=vector(self.model) if capture else None;identity=state_sha(self.model) if capture else None
        observed=None;total_loss=torch.zeros((),device=self.device)
        micro=self.cfg['runtime']['micro_batch']
        for start in range(0,count,micro):
            item={k:v[start:start+micro] for k,v in batch.items()}
            kwargs=dict(capture=capture and start==0) if self.writer else {}
            logits=self.model(**self.inputs(item),**kwargs)
            loss=torch.nn.functional.cross_entropy(logits,item['labels'],reduction='sum')/count
            if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
            loss.backward();total_loss+=loss.detach()
            if capture and start==0:observed={k:v.detach().clone() for k,v in self.model.observed.items()}
        if any(x.grad is not None and not torch.isfinite(x.grad).all() for x in self.params):raise ValueError('nonfinite training gradient')
        gradient=vector(self.model,True) if capture else None
        self.optimizer.step()
        if capture:
            self.model.observed=observed
            self.writer.add(self.rows['train'][0],'train','update0','training',
                values(self.model,batch['labels'],total_loss,before,gradient,vector(self.model)),0,identity)
        self.update+=1;self.save()

    def select(self):
        score=self.evaluate('valid')['metrics']['macro_f1']
        if score>self.best_metric:
            self.best={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
            self.best_metric,self.best_update=score,self.update
        self.selected=self.update;self.save()
        p.atomic_json(self.directory/'heartbeat.json',dict(seed=self.seed,mode=self.mode,update=self.update,total=self.total,valid_macro_f1=score))
        print(dict(seed=self.seed,mode=self.mode,update=self.update,valid_macro_f1=score),flush=True)

    def run(self):
        from q_attention.experiments.ecpa_protocol import save_state
        from q_attention.analysis.srm_trace import validate_trace
        if self.update and self.update%self.per_epoch==0 and self.selected<self.update:self.select()
        while self.update<self.total:
            self.train_update()
            if self.update%self.per_epoch==0:self.select()
            if self.stop_after is not None and self.update>=self.stop_after and self.update<self.total:return p.PAUSED_EXIT
        if self.best is None or self.selected!=self.total:raise ValueError('missing validation selection')
        save_state(self.directory/'best.pt',dict(model=self.best,update=self.best_update,fingerprint=self.manifest['fingerprint']))
        files=['best.pt','latest.json'];readiness=None
        if self.writer:
            self.snapshot('final',self.update);self.model.load_state_dict(self.best);self.snapshot('best',self.best_update)
            self.writer.payload['trace_completeness']='complete';self.writer.save()
            readiness=validate_trace(self.directory/'sample_trace.json');files.append('sample_trace.json')
        else:self.model.load_state_dict(self.best)
        report=dict(seed=self.seed,mode=self.mode,fingerprint=self.manifest['fingerprint'],updates=self.update,best_update=self.best_update,
            base_state_sha256=self.base_hash,trainable_parameters=sum(x.numel() for x in self.params),
            plugin_parameters=sum(x.numel() for x in self.model.kernel.parameters()) if self.writer and self.model.kernel else 0,
            valid=self.evaluate('valid'),trace=readiness,test_used_for_training_or_selection=False)
        if self.writer:report['test']=self.evaluate('test')
        p.atomic_json(self.directory/'summary.json',report);files.append('summary.json')
        p.atomic_json(self.directory/'RUN_COMPLETE.json',dict(fingerprint=self.manifest['fingerprint'],seed=self.seed,mode=self.mode,
            updates=self.update,files={name:p.sha(self.directory/name) for name in files}))
        return 0


def aggregate(summaries,cfg):
    if cfg['profile']=='synthetic_canary':return dict(classification='canary_only',utility_claim=False)
    scores={(s['seed'],s['mode']):s['test']['metrics']['macro_f1'] for s in summaries}
    delta=[scores[seed,'quantum']-scores[seed,'disabled'] for seed in cfg['seeds']]
    mean=statistics.mean(delta);sd=statistics.stdev(delta)
    base_sd=statistics.stdev(scores[s,'disabled'] for s in cfg['seeds'])
    mismatch=statistics.mean(scores[s,'quantum']-scores[s,'key_mismatch'] for s in cfg['seeds'])
    parity=mean>0 and mismatch<=cfg['gates']['mismatch_parity_tolerance']
    passed=mean>0 and sum(x>0 for x in delta)>=cfg['gates']['positive_seeds'] and sd<=base_sd
    return dict(classification='mechanism_inconclusive' if parity else 'promote_to_formal_preparation' if passed else 'inconclusive',
        mean_delta=mean,paired_deltas=delta,delta_sample_sd=sd,disabled_sample_sd=base_sd,positive_seeds=sum(x>0 for x in delta),
        mismatch_parity=parity,quantum_attribution='not established',stage='bounded reused development subset, not formal evidence')


def audit_run(root):
    root=Path(root);manifest=p.read(root/'run_manifest.json');cfg=manifest['config'];results=[]
    for seed in cfg['seeds']:
        for mode in ('warmup',*p.MODES):
            directory=root/f'seed{seed}'/mode
            if not completed(directory,manifest,seed,mode):raise ValueError('incomplete phase: '+str(directory))
            if mode!='warmup':results.append(p.read(directory/'summary.json'))
    expected_updates=math.ceil(cfg['expected_selected']['train']/cfg['screen']['batch_size'])*cfg['screen']['epochs']
    if any(r['updates']!=expected_updates for r in results):raise ValueError('unequal declared updates')
    if any(r['plugin_parameters']!=(0 if r['mode']=='disabled' else 144) for r in results):raise ValueError('unexpected plugin budget')
    for seed in cfg['seeds']:
        if len({r['base_state_sha256'] for r in results if r['seed']==seed})!=1:raise ValueError('different warm-starts')
    return manifest,dict(fingerprint=manifest['fingerprint'],profile=cfg['profile'],decision=aggregate(results,cfg),conditions=results)


def export(root,destination,*,inject_failure=False):
    root=Path(root);destination=Path(destination)
    manifest,summary=audit_run(root)
    marker=p.read(root/'RUN_COMPLETE.json')
    if (marker['fingerprint']!=manifest['fingerprint'] or marker['summary_sha256']!=p.sha(root/'summary.json')
        or p.read(root/'summary.json')!=summary):raise ValueError('root summary changed')
    if inject_failure and manifest['config']['profile']!='synthetic_canary':raise ValueError('export injection canary-only')
    data={split:{k:v for k,v in manifest['data'][split].items() if k in ('source_sha256','source_count','selected_count')} for split in ('train','valid','test')}
    payloads={'summary.json':summary,'manifest.json':dict(run_id=root.name,fingerprint=manifest['fingerprint'],config=manifest['config'],data=data,
        environment=manifest['environment'],source_sha256=manifest['source_sha256'],git_revision=manifest['git_revision']),
        'trace_validation.json':dict(status='passed',conditions=[dict(seed=r['seed'],mode=r['mode'],**r['trace']) for r in summary['conditions']])}
    if destination.exists():
        if destination.is_symlink() or {x.name for x in destination.iterdir()}!=set(payloads)|{'receipt.json'}:raise ValueError('unexpected existing export')
        receipt=p.read(destination/'receipt.json')
        if receipt['fingerprint']!=manifest['fingerprint']:raise ValueError('export identity differs')
        for name,obj in payloads.items():
            if p.read(destination/name)!=obj or p.sha(destination/name)!=receipt['sha256'][name]:raise ValueError('existing export changed')
        return receipt
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.'+destination.name+'-',dir=destination.parent))
    try:
        for index,(name,obj) in enumerate(payloads.items()):
            p.atomic_json(temporary/name,obj)
            if inject_failure and index==0:raise RuntimeError('injected export failure')
        receipt=dict(fingerprint=manifest['fingerprint'],sha256={name:p.sha(temporary/name) for name in payloads},raw_artifacts_included=False)
        p.atomic_json(temporary/'receipt.json',receipt)
        temporary.rename(destination)
        return receipt
    finally:
        if temporary.exists():shutil.rmtree(temporary)


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--output-dir',required=True)
    result.add_argument('--profile',choices=('canary','screen'),default='canary')
    result.add_argument('--data-dir')
    result.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    result.add_argument('--resume',action='store_true')
    result.add_argument('--export-only',action='store_true')
    result.add_argument('--report-dir')
    result.add_argument('--stop-after',type=int)
    result.add_argument('--inject-export-failure',action='store_true')
    result.add_argument('--worker',choices=('warmup',*p.MODES),help=argparse.SUPPRESS)
    result.add_argument('--seed',type=int,help=argparse.SUPPRESS)
    return result


def child_spec(root,seed,mode,stop_after=None):
    argv=[sys.executable,str(p.ROOT/'experiments/run_srm_screen.py'),'--output-dir',str(root),'--worker',mode,'--seed',str(seed)]
    if stop_after is not None:argv+=['--stop-after',str(stop_after)]
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return dict(argv=argv,cwd=str(p.ROOT),env=env)


def main(argv=None):
    args=parser().parse_args(argv)
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    p.validate_review()
    if args.worker:
        if args.seed is None:raise ValueError('worker needs a seed')
        lock=Path(args.output_dir)/f'seed{args.seed}'/args.worker/'.worker.lock'
        with p.exclusive(lock):
            worker=Worker(args.output_dir,args.seed,args.worker,stop_after=args.stop_after)
            if completed(worker.directory,worker.manifest,args.seed,args.worker):return 0
            return worker.run()
    if args.seed is not None:raise ValueError('parent seeds are frozen; no --seed override')
    if (args.stop_after is not None or args.inject_export_failure) and args.profile!='canary':raise ValueError('fault injection is canary-only')
    if args.stop_after is not None and args.stop_after<1:raise ValueError('stop cursor must be positive')
    if args.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable; no silent CPU fallback')
    root=Path(args.output_dir).resolve()
    if root.is_relative_to(p.ROOT):raise ValueError('raw experiment output must stay outside the checkout')
    if root.exists() and not args.resume:raise ValueError('output exists; use --resume with the same contract')
    if not root.exists() and (args.resume or args.export_only):raise ValueError('resume/export source absent')
    root.mkdir(parents=True,exist_ok=True)
    with p.exclusive(root/'.run.lock'):
        cfg=p.config(args.profile)
        directory=root/'synthetic_data' if args.profile=='canary' else Path(args.data_dir).resolve() if args.data_dir else None
        if directory is None:raise ValueError('bounded screen requires --data-dir')
        _,_,_,_,data=p.load_data(directory,cfg)
        source=p.identity(p.source_contract());environment=p.environment(args.device)
        fingerprint=p.identity(dict(config=cfg,data=data,source=source,environment=environment))
        manifest=dict(config=cfg,data=data,data_dir=str(directory),source_sha256=source,environment=environment,
            device=args.device,fingerprint=fingerprint,git_revision=p.revision())
        path=root/'run_manifest.json'
        if path.exists():
            old=p.read(path)
            if old['fingerprint']!=fingerprint or old['data_dir']!=str(directory):raise ValueError('resume source/data/config/environment differs')
            manifest=old
        else:p.atomic_json(path,manifest)
        if not args.export_only:
            for seed in cfg['seeds']:
                for mode in ('warmup',*p.MODES):
                    if completed(root/f'seed{seed}'/mode,manifest,seed,mode):continue
                    spec=child_spec(root,seed,mode,args.stop_after if mode=='quantum' else None)
                    result=subprocess.run(spec['argv'],cwd=spec['cwd'],env=spec['env'])
                    if result.returncode==p.PAUSED_EXIT:return p.PAUSED_EXIT
                    if result.returncode:raise RuntimeError(f'worker failed: {seed}/{mode}, exit {result.returncode}')
            _,summary=audit_run(root)
            if (root/'RUN_COMPLETE.json').exists():
                old=p.read(root/'RUN_COMPLETE.json')
                if p.read(root/'summary.json')!=summary or old['summary_sha256']!=p.sha(root/'summary.json'):raise ValueError('completed root changed')
            else:
                p.atomic_json(root/'summary.json',summary)
                p.atomic_json(root/'RUN_COMPLETE.json',dict(fingerprint=fingerprint,summary_sha256=p.sha(root/'summary.json')))
        destination=Path(args.report_dir).resolve() if args.report_dir else root/'export'
        export(root,destination,inject_failure=args.inject_export_failure)
    return 0
