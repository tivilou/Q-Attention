"""Single bounded-memory warm-start/condition worker with committed batch cursors."""
from __future__ import annotations

import math
from pathlib import Path
import signal
import time
import torch
from torch.nn import functional as F

from q_attention.adapters.q_cfra import CFRAModel,objective
from q_attention.adapters.ecpa_formal import ECPAFormalModel
from q_attention.analysis.cfra_trace import Trace,values,vector,state_sha,validate_trace
from q_attention.analysis.ecpa_formal_trace import FormalTrace,gradients,state_identity,validate as validate_baseline
from q_attention.models.relation_transformer import RelationExtractionModel,RelationTransformerConfig
from q_attention.tasks import cfra_data as d
from q_attention.experiments import cfra_protocol as p,cfra_checkpoint as cp
from q_attention.experiments.ecpa_protocol import save_state


def validate_job(path,mode,config):
    path=Path(path)
    report=validate_baseline(path,mode,config) if mode=='baseline' else validate_trace(path)
    payload=p.read(path)
    if payload['experiment']['dataset_identity'] is None:raise ValueError('missing data identity')
    if mode!='baseline':
        if payload['sample_selection']['population_scope']!=config['profile']:
            raise ValueError('incorrect formal trace population')
        if payload['sample_selection']['rule']!='Official source record0 per split, fixed before training.':
            raise ValueError('unfrozen case selection')
    return report


class Worker:
    def __init__(self,run_dir,mode,*,device='cuda',stop_after=None,recover_previous=False):
        if mode not in ('baseline',*p.MODES):raise ValueError('unknown worker')
        self.root,self.mode=Path(run_dir),mode
        self.config,self.contract=p.read(self.root/'run_config.json'),p.read(self.root/'run_manifest.json')
        if self.config!=p.load_config(self.config['profile']=='synthetic_canary'):
            raise ValueError('worker config differs from frozen release')
        if p.identity(dict(config=self.config,data=self.contract['data'],source_sha256=self.contract['source_sha256']))!=self.contract['fingerprint']:
            raise ValueError('worker scientific fingerprint differs')
        self.device=torch.device(device);self.paused=False;self.stop_after=stop_after
        if self.config['profile']=='formal_seed13' and self.device.type!='cuda':
            raise ValueError('formal worker requires CUDA; CPU is synthetic-canary only')
        self.directory=self.root/'jobs'/mode;self.directory.mkdir(parents=True,exist_ok=True)
        self.started=time.monotonic();self.update=0
        if stop_after is not None and self.config['profile']!='synthetic_canary':
            raise ValueError('stop injection only in synthetic canary')
        self.rows,self.datasets,self.vocab,self.labels,maximum,manifest=p.load_data(self.contract['data_dir'],self.config)
        if manifest!=self.contract['data'] or p.identity(p.source_contract())!=self.contract['source_sha256']:
            raise ValueError('worker source/data differs from immutable contract')
        p.seed_all(self.config['seed'])
        base=RelationExtractionModel(RelationTransformerConfig(vocab_size=len(self.vocab),num_labels=len(self.labels),
            max_length=maximum,**self.config['model']))
        self.base_state=None
        if mode=='baseline':
            self.model=ECPAFormalModel(base,'baseline',self.config,train_base=True)
        else:
            upstream=self.root/'jobs/baseline'
            complete=p.read(upstream/'RUN_COMPLETE.json')
            if complete['fingerprint']!=self.contract['fingerprint'] or complete['best_sha256']!=p.sha(upstream/'best.pt'):
                raise ValueError('shared warm-start binding differs')
            state=torch.load(upstream/'best.pt',map_location='cpu',weights_only=False)
            if state['fingerprint']!=self.contract['fingerprint']:raise ValueError('wrong upstream state')
            base.load_state_dict({name.removeprefix('base.'):value for name,value in state['model'].items()})
            d.initialize_neutral_embeddings(base,self.rows['train'],self.vocab)
            self.base_state=state_sha(base)
            p.seed_all(self.config['seed']+1000)
            self.model=CFRAModel(base,mode,self.config['channel'])
        self.model.to(self.device)
        self.budget=self.config['baseline' if mode=='baseline' else 'kernel']
        self.parameters=[v for v in self.model.parameters() if v.requires_grad]
        self.optimizer=torch.optim.Adam(self.parameters,lr=self.budget['lr'])
        self.per_epoch=math.ceil(len(self.rows['train'])/self.budget['batch_size'])
        self.total=self.per_epoch*self.budget['epochs']
        self.micro=min(self.contract['execution']['micro_batch'],self.budget['batch_size'])
        self.best,self.best_metric,self.best_update,self.selection_update=None,-math.inf,0,0
        self.sampler_epoch,self.sampler_order=-1,[]
        if cp.exists(self.directory):
            state=cp.load(self.directory,recover_previous=recover_previous)
            if (state['schema_version']!='cfra-worker-checkpoint.v1' or state['mode']!=mode or
                state['fingerprint']!=self.contract['fingerprint'] or state['base_state_sha256']!=self.base_state or
                state['logical_batch']!=self.budget['batch_size']):raise ValueError('incompatible resume state')
            repair=self.directory/'completed_checkpoint_recovery.json'
            if repair.exists() and (state['next_update']!=self.total or state['selection_update']!=self.total or state['best_model'] is None):
                raise ValueError('completed repair must not repeat optimizer updates or selection')
            self.model.load_state_dict(state['model']);self.optimizer.load_state_dict(state['optimizer'])
            self.update,self.best,self.best_metric,self.best_update,self.selection_update=(state[k] for k in
                ('next_update','best_model','best_metric','best_update','selection_update'))
            self.sampler_epoch,self.sampler_order=state['sampler']['epoch'],state['sampler']['order']
            expected_epoch,next_batch=divmod(self.update,self.per_epoch)
            if (state['sampler']['next_batch']!=next_batch or self.sampler_epoch!=expected_epoch or
                self.sampler_order!=p.epoch_order(len(self.rows['train']),self.config['seed'],expected_epoch)):
                raise ValueError('sampler permutation/cursor differs')
            p.restore_rng(state['rng'])
        elif recover_previous:raise ValueError('no previous state to recover')
        self.saved_update,self.saved_at=self.update,time.monotonic()
        if mode=='baseline':
            self.writer=FormalTrace(self.directory,mode,self.config,self.contract,self.labels)
            self.writer.trace['sample_selection'].update(rule='Official source record0 per split, fixed before training.',population_scope=self.config['profile'])
            self.writer.trace['task_contract']['checkpoint_policy']='Official record0 per split at initial/best/final; actual update0 train lane. No target reaches intervention.'
            self.writer.trace['parameter_layout']=[dict(name=n,shape=list(v.shape),elements=v.numel())
                for n,v in self.model.named_parameters() if v.requires_grad]
        else:
            names=[dict(name=n,shape=list(v.shape),numel=v.numel()) for n,v in self.model.named_parameters() if v.requires_grad]
            self.writer=Trace(self.directory,mode,dict(run_id=self.contract['run_id'],seed=self.config['seed'],
                model_identity=mode,config_sha256=p.identity(self.config),dataset='Re-TACRED complete official splits',
                code_revision=self.contract['source_sha256'],dataset_identity=p.identity(manifest)),self.labels,names)
            self.writer.payload['sample_selection'].update(rule='Official source record0 per split, fixed before training.',population_scope=self.config['profile'])
        if not cp.exists(self.directory):self.save()

    def indexes(self):
        epoch,position=divmod(self.update,self.per_epoch)
        if epoch!=self.sampler_epoch:
            self.sampler_epoch=epoch;self.sampler_order=p.epoch_order(len(self.rows['train']),self.config['seed'],epoch)
        logical=self.budget['batch_size']
        return self.sampler_order[position*logical:(position+1)*logical]

    def save(self):
        self.indexes()
        cp.save(self.directory,dict(schema_version='cfra-worker-checkpoint.v1',fingerprint=self.contract['fingerprint'],
            mode=self.mode,base_state_sha256=self.base_state,model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),
            next_update=self.update,best_model=self.best,best_metric=self.best_metric,best_update=self.best_update,
            selection_update=self.selection_update,rng=p.rng_state(),logical_batch=self.budget['batch_size'],
            sampler=dict(epoch=self.sampler_epoch,next_batch=self.update%self.per_epoch,order=self.sampler_order),
            execution=dict(micro=self.micro,device=str(self.device))))
        self.saved_update,self.saved_at=self.update,time.monotonic()

    def heartbeat(self,phase,done=None,total=None):
        now=time.monotonic()
        if phase==getattr(self,'last_phase',None) and now-getattr(self,'last_heartbeat',0)<1:return
        done,total=(self.update,self.total) if done is None else (done,total)
        if phase!=getattr(self,'rate_phase',None):self.rate_phase,self.rate_start,self.rate_done=phase,now,done
        event=dict(phase=phase,done=done,total=total,rate=(done-self.rate_done)/max(now-self.rate_start,1e-9),
            elapsed=now-self.started,micro=self.micro,checkpoint_update=self.saved_update,checkpoint_age=now-self.saved_at)
        if self.device.type=='cuda':event.update(allocated_gib=torch.cuda.memory_allocated()/2**30,peak_gib=torch.cuda.max_memory_allocated()/2**30)
        p.atomic_json(self.directory/'heartbeat.json',event);self.last_phase,self.last_heartbeat=phase,now

    def batch(self,split,indexes):
        return d.batch(self.rows[split],self.datasets[split],self.vocab,indexes)

    def inputs(self,value):
        names=('input_ids','attention_mask','subject_mask','object_mask')+(('neutral_ids',) if self.mode!='baseline' else ())
        return {name:value[name] for name in names}

    def model_sha(self):
        return state_identity(self.model) if self.mode=='baseline' else state_sha(self.model)

    def evaluate(self,split):
        self.model.eval();predicted,gold,loss=[],[],0.
        with torch.no_grad():
            for start in range(0,len(self.rows[split]),self.micro):
                value={k:v.to(self.device) for k,v in self.batch(split,range(start,min(start+self.micro,len(self.rows[split])))).items()}
                logits=self.model(**self.inputs(value))
                if self.mode!='baseline':logits=logits[:,0]
                loss+=float(F.cross_entropy(logits,value['labels'],reduction='sum'))
                predicted.extend(logits.argmax(-1).cpu().tolist());gold.extend(value['labels'].cpu().tolist())
                self.heartbeat('evaluate_'+split,len(gold),len(self.rows[split]))
                if self.paused:raise InterruptedError('safe pause during evaluation')
        return p.metrics(predicted,gold,self.labels,loss)

    def snapshot(self,checkpoint,update):
        self.model.eval();identity=self.model_sha()
        for split in ('train','valid','test'):
            value={k:v.to(self.device) for k,v in self.batch(split,[0]).items()}
            self.model.zero_grad(set_to_none=True);before=vector(self.model)
            logits=self.model(**self.inputs(value),capture=True)
            if self.mode=='baseline':
                loss=F.cross_entropy(logits,value['labels']);loss.backward()
                captured=dict(self.model.observed,loss=loss.detach().cpu(),parameter_gradient=gradients(self.model))
                self.writer.add(self.rows[split][0],split,0,checkpoint,'replay',captured,update,identity)
            else:
                loss,parts=objective(logits,self.model.attention,value['attention_mask'],value['labels'],**self.config['loss'])
                loss.backward()
                self.writer.add(self.rows[split][0],split,checkpoint,'replay',
                    values(self.model,value['labels'],parts,before,vector(self.model,True),before),update,identity)
            self.model.zero_grad(set_to_none=True);self.heartbeat('trace_'+checkpoint)
            if self.paused:raise InterruptedError('safe pause during replay')

    def train_update(self):
        ids=self.indexes();logical=self.batch('train',ids)
        self.optimizer.zero_grad(set_to_none=True);observed=None;components=None
        before=vector(self.model) if self.update==0 else None
        identity=self.model_sha() if self.update==0 else None
        queries=int(logical['attention_mask'].sum())
        for offset in range(0,len(ids),self.micro):
            micro={k:v[offset:offset+self.micro].to(self.device) for k,v in logical.items()}
            capture=self.update==0 and offset==0
            logits=self.model(**self.inputs(micro),capture=capture)
            if self.mode=='baseline':
                loss=F.cross_entropy(logits,micro['labels'],reduction='sum')/len(ids)
                detached=loss.detach().cpu()
            else:
                _,parts=objective(logits,self.model.attention,micro['attention_mask'],micro['labels'],**self.config['loss'])
                weighted=torch.stack((parts[0]*len(micro['labels'])/len(ids),parts[1]*len(micro['labels'])/len(ids),
                    parts[2]*micro['attention_mask'].sum()/queries))
                loss=weighted[0]+self.config['loss']['prediction_js']*weighted[1]+self.config['loss']['attention_js']*weighted[2]
                detached=torch.cat((weighted,loss.reshape(1))).detach().cpu()
            if not torch.isfinite(loss):raise ValueError('nonfinite objective')
            loss.backward();components=detached if components is None else components+detached
            if capture:observed={k:v.detach().clone() for k,v in self.model.observed.items()}
            self.heartbeat('training')
        if any(not torch.isfinite(v.grad).all() for v in self.parameters if v.grad is not None):raise ValueError('nonfinite gradient')
        gradient=vector(self.model,True) if observed is not None else None
        self.optimizer.step()
        if observed is not None:
            self.model.observed=observed
            if self.mode=='baseline':
                self.writer.add(self.rows['train'][0],'train',0,'update-0','training',
                    dict(observed,loss=components,parameter_gradient=gradient),0,identity)
            else:
                self.writer.add(self.rows['train'][0],'train','update0','training',
                    values(self.model,logical['labels'],components,before,gradient,vector(self.model)),0,identity)
        self.update+=1

    def select(self):
        self.save()  # validation is interruptible; persist optimizer state first
        score=self.evaluate('valid')['metrics'][self.config['primary']]
        if score>self.best_metric:
            self.best={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
            self.best_metric,self.best_update=score,self.update
        self.selection_update=self.update;self.save()

    def run(self):
        if self.update==0:self.snapshot('initial',0)
        else:
            payload=self.writer.trace if self.mode=='baseline' else self.writer.payload
            if sum(s['checkpoint']=='initial' for s in payload['samples'])!=3:raise ValueError('committed cursor lacks initial cases')
        if self.update and self.update%self.per_epoch==0 and self.selection_update<self.update:self.select()
        while self.update<self.total:
            self.model.train();self.train_update()
            if self.paused:
                self.save();return p.PAUSED_EXIT
            if self.update%self.per_epoch==0:self.select()
            cadence=self.config['checkpoint']
            if self.update-self.saved_update>=cadence['every_updates'] or time.monotonic()-self.saved_at>=cadence['every_seconds']:self.save()
            if self.stop_after is not None and self.update>=self.stop_after:
                self.save();return p.PAUSED_EXIT
        self.save()
        p.atomic_json(self.directory/'TRAIN_COMPLETE.json',dict(fingerprint=self.contract['fingerprint'],updates=self.update))
        save_state(self.directory/'best.pt',dict(model=self.best,update=self.best_update,metric=self.best_metric,fingerprint=self.contract['fingerprint']))
        final={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
        self.snapshot('final',self.update);self.model.load_state_dict(self.best);self.snapshot('best',self.best_update)
        reports={s:self.evaluate(s) for s in ('valid','test')}
        if self.mode=='baseline':self.writer.trace['trace_completeness']='complete';self.writer.complete()
        else:self.writer.payload['trace_completeness']='complete';self.writer.save()
        readiness=validate_job(self.directory/'sample_trace.json',self.mode,self.config)
        p.atomic_json(self.directory/'trace_validation.json',readiness)
        self.model.load_state_dict(final)
        p.atomic_json(self.directory/'metrics.json',dict(mode=self.mode,fingerprint=self.contract['fingerprint'],
            base_state_sha256=self.base_state,reports=reports,updates=self.update,best_update=self.best_update,
            trainable_parameters=sum(v.numel() for v in self.parameters),plugin_parameters=0 if self.mode=='baseline' or self.model.kernel is None else sum(v.numel() for v in self.model.kernel.parameters()),
            micro=self.micro,trace=readiness,test_used_for_training_or_selection=False))
        repair=self.directory/'completed_checkpoint_recovery.json'
        if repair.exists():
            old=p.read(repair)['old_completion']
            if p.sha(self.directory/'metrics.json')!=old['metrics_sha256'] or p.sha(self.directory/'sample_trace.json')!=old['trace_sha256']:
                raise ValueError('completed repair changed frozen metrics/cases; stop for owner audit')
        p.atomic_json(self.directory/'RUN_COMPLETE.json',dict(fingerprint=self.contract['fingerprint'],metrics_sha256=p.sha(self.directory/'metrics.json'),
            trace_sha256=p.sha(self.directory/'sample_trace.json'),best_sha256=p.sha(self.directory/'best.pt'),
            checkpoint_sha256=p.read(self.directory/'latest.json')['latest']['sha256']))
        self.heartbeat('complete',self.total,self.total)
        return 0


def main(run,mode,**kwargs):
    worker=None
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        worker=Worker(run,mode,**kwargs)
        def pause(signum,frame):worker.paused=True
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,pause)
        return worker.run()
    except InterruptedError:return p.PAUSED_EXIT
    except torch.cuda.OutOfMemoryError:
        p.atomic_json(Path(run)/'jobs'/mode/'OOM.json',dict(trigger='actual_cuda_oom',policy='stop; retain checkpoint; no automatic retry or scientific adaptation'))
        raise
