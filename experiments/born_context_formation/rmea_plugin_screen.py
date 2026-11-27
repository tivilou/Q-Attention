"""Shared closed-registry scientific screen; accepted micro4/fixed-order core."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_all40(metric,labels,supports):
    """Independently derive the gated all40 mean from the complete count table."""
    cm=metric['confusion_matrix']
    assert len(cm)==40 and all(len(row)==40 for row in cm)
    assert [sum(row) for row in cm]==supports
    assert [x['label'] for x in metric['per_class']]==labels
    f1=[]
    for i in range(40):
        denominator=sum(cm[i])+sum(row[i] for row in cm)
        f1.append(2*cm[i][i]/denominator if denominator else 0.)
    value=sum(f1)/40
    assert abs(metric['macro_f1']-value)<1e-12
    assert all(abs(x['f1']-y)<1e-12 for x,y in zip(metric['per_class'],f1))
    return value


def main():
    if sys.flags.optimize:
        raise RuntimeError('Unoptimized interpreter required')
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ['spec','output','progress']:
        ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--spec-sha256',required=True)
    ap.add_argument('--attempt',required=True)
    ap.add_argument('--check-only',action='store_true')
    a=ap.parse_args();here=Path(__file__).resolve().parent
    assert sha(a.spec)==a.spec_sha256
    cfg=json.loads(a.spec.read_bytes())
    assert cfg['scope']=='ONE_FIXED_PLUGIN_DEVELOPMENT_SCREEN'
    assert cfg['execution']=='micro4' and cfg['seed']==13 and cfg['updates']==1024 and cfg['logical_batch']==128
    assert cfg['runner_attempt_id']==a.attempt
    signal.alarm(cfg['campaign_seconds'])
    assert set(cfg['code_hashes'])=={'rmea_plugin_screen.py','rmea_context_formation.py','rmea_plugin_registry.py'}
    for n,pin in cfg['code_hashes'].items():assert Path(n).name==n and sha(here/n)==pin
    assert sha(here/'binding.json')==cfg['binding_sha256']
    assert sha(here/'reference-audit.json')==cfg['reference_audit_sha256']
    assert not a.output.exists(), 'Observe existing scientific attempt; no automatic retry'
    old=cfg['retained_guard_contract']
    guard=old['argv']+['--check-only']
    p=subprocess.run(guard,cwd=here,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},capture_output=True,timeout=90)
    assert p.returncode==0,p.stderr.decode()
    guarded=json.loads(p.stdout);assert guarded['status']=='PASS_SEALED_CANARY_INPUT_GUARDS'
    b=json.loads((here/'binding.json').read_bytes())
    assert sha(b['strong_reference']['summary_path'])==b['strong_reference']['summary_sha256']
    for n,pin in cfg['borrowed_hashes'].items():assert sha(Path(cfg['borrowed_directory'])/n)==pin
    sys.path[:0]=[str(here),cfg['borrowed_directory'],b['source']]
    import numpy as np
    import torch
    import fixed_order_fit as core
    from rmea_plugin_registry import resolve_plugin
    plugin=resolve_plugin(cfg['plugin_id'])
    import sequence_task as task
    from baselines.retacred.rmea import RMEAHead
    from baselines.retacred.rmea_token_transport_study import state_identity
    from baselines.retacred.rmea_token_transport_recovery import atomic_torch,cpu_tree
    from baselines.retacred.common import metrics
    from baselines.retacred.rmea_task_runtime import score_batch
    assert sha(plugin.__file__)==cfg['code_hashes'][cfg['plugin_file']]
    for module in [core,task]:assert sha(module.__file__)==cfg['borrowed_hashes'][Path(module.__file__).name]
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    count=torch.cuda.device_count()
    if not a.check_only:assert count==len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==cfg['allocated_gpu_count']==2
    devices=['cuda:'+str(i) for i in range(count)]
    data=task.FrozenWords(b['cache'],b['cohort_path'])
    assert (len(data.cohort['fit']),len(data.cohort['assessment']))==(8192,4096)
    rng=np.random.default_rng(13);batches=[]
    while len(batches)<1024:
        order=rng.permutation(data.cohort['fit'])
        batches.extend(order[i:i+128] for i in range(0,len(order),128))
    batches=batches[:1024]
    order_sha=hashlib.sha256(b''.join(x.astype('<i8').tobytes() for x in batches)).hexdigest()
    assert order_sha==b['order_sha256']==guarded['order_sha256']
    random.seed(13);np.random.seed(13);torch.manual_seed(13)
    initial=RMEAHead('quantum').float().state_dict()
    assert state_identity(initial)==cfg['native_initial_raw_sha256']
    models=plugin.make_models(13,initial);modes=tuple(models)
    assert modes==tuple(cfg['modes'])
    assert {k:sum(p.numel() for p in v.parameters()) for k,v in models.items()}==cfg['parameters']
    assert {k:state_identity(v.state_dict()) for k,v in models.items()}==cfg['initial_sha256']
    label_contract=Path(b['source'])/cfg['label_contract']['source_path']
    assert sha(label_contract)==cfg['label_contract']['sha256']
    labels=cfg['label_contract']['labels_in_target_id_order']
    assert json.loads(label_contract.read_bytes())==labels
    assert len(labels)==40 and labels[0]=='no_relation'
    if a.check_only:
        assert not torch.cuda.is_initialized()
        print(json.dumps(dict(status='PASS_FULL_FIELD_SCIENTIFIC_PREFLIGHT',initial_sha256=cfg['initial_sha256'],order_sha256=order_sha,model_forwards=0,optimizer_updates=0,cuda_initialized=False)),flush=True)
        return
    a.output.mkdir();(a.output/'fits').mkdir()
    core.atomic_json(a.output/'input-guard-execution.json',dict(argv=guard,stdout=p.stdout.decode(),stderr=p.stderr.decode(),exit_code=p.returncode))
    core.atomic_json(a.output/'manifest.json',dict(spec=cfg,spec_sha256=a.spec_sha256,
        argv=sys.argv,guard_result=guarded,torch=torch.__version__,numpy=np.__version__,
        selection='Final update1024 only, all fits before any assessment; best checkpoints never evaluated.',
        original_L4_retained=True,newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED'))
    def check_peak(device):
        peak=torch.cuda.max_memory_allocated(device)
        if peak>cfg['max_peak_gpu_bytes']:raise MemoryError('Frozen Torch peak memory cap exceeded')
        return peak
    def check_gradients(model):
        task.check_gradients(model);check_peak(next(model.parameters()).device)
    adapter=SimpleNamespace(batch=data.batch,optimizer=task.optimizer,backward=task.backward,check_gradients=check_gradients)
    cursors={k:0 for k in modes};roster={k:dict(state='pending') for k in modes};lock=threading.Lock();phase='training'
    def progress(mode,state,**values):
        with lock:
            if 'step' in values:cursors[mode]=values['step']
            roster[mode].update(state=state,**values)
            core.atomic_json(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,phase=phase,
                current=sum(cursors.values()),total=len(modes)*cfg['updates'],unit='optimizer_update',observed_at=time.time(),heartbeat_at=time.time(),children=roster))
    def train(mode,device,stopped):
        call=lambda:core.fit(models[mode],adapter,batches,a.output/'fits'/mode,mode=mode,device=device,
            contract_sha=a.spec_sha256,order_sha=order_sha,progress=progress,stopped=stopped,
            atomic_torch=atomic_torch,cpu_tree=cpu_tree,state_identity=state_identity,checkpoint_every=32)
        fit,seconds=core.bounded(call,cfg['arm_active_seconds']);fit['outer_active_seconds']=seconds
        assert fit['peak_allocated_bytes']<=cfg['max_peak_gpu_bytes']
        assert fit['updates']==1024 and fit['sample_exposures']==131072 and fit['base_unchanged']
        return fit
    started=time.time();trained=core.lanes(modes,devices,train,progress)
    # Complete barrier. Check actual final/latest identity before opening assessment.
    for mode in modes:
        f=a.output/'fits'/mode/'final.pt'
        assert sha(f)==trained[mode]['final_sha256']
        state=torch.load(f,map_location='cpu',weights_only=True)
        latest=torch.load(f.parent/'latest.pt',map_location='cpu',weights_only=True)
        assert latest['next_update']==1024 and latest['contract_sha']==a.spec_sha256 and latest['order_sha']==order_sha
        assert state_identity(state)==trained[mode]['final_state_sha256']==state_identity(latest['model'])
        models[mode].load_state_dict(state)
    # Construct once before parallel evaluation; independently trained D remains unchanged.
    fixed_dephased=copy.deepcopy(models[cfg['dephased_mode']]).cpu()
    fixed_dephased.load_state_dict(models[cfg['quantum_mode']].state_dict(),strict=True)
    barrier=time.time();phase='assessment'
    def evaluate(mode,device,stopped):
        def call():
            model=models[mode].to(device).eval();zs=[];ds=[]
            fixed_d=copy.deepcopy(fixed_dephased).to(device).eval() if mode==cfg['quantum_mode'] else None
            with torch.cuda.device(device),torch.no_grad():
                for start in range(0,4096,128):
                    core.require_running(stopped)
                    inputs,_=data.batch(data.cohort['assessment'][start:start+128],device)
                    z=score_batch(model,inputs);assert z.shape==(128,40) and torch.isfinite(z).all();zs.append(z.cpu().numpy())
                    if fixed_d is not None:
                        d=score_batch(fixed_d,inputs)
                        assert d.shape==(128,40) and torch.isfinite(d).all();ds.append(d.cpu().numpy())
                    check_peak(device)
                    progress(mode,'evaluating',assessment_rows=start+128,device=device)
            peak=check_peak(device);model.cpu()
            if fixed_d is not None:fixed_d.cpu()
            return dict(logits=np.concatenate(zs),dephased=np.concatenate(ds) if ds else None,peak_allocated_bytes=peak)
        result,seconds=core.bounded(call,cfg['arm_active_seconds']-trained[mode]['outer_active_seconds'])
        result['seconds']=seconds;return result
    evaluated=core.lanes(modes,devices,evaluate,progress)
    y=np.asarray(data.data['targets'][data.cohort['assessment']],dtype=np.int64)
    assert len(y)==4096 and [int((y==i).sum()) for i in range(40)]==b['supports']['assessment']
    def ce(z):
        z=z.astype(np.float64);m=z.max(-1)
        return float((np.log(np.exp(z-m[:,None]).sum(-1))+m-z[np.arange(len(y)),y]).mean())
    rows={}
    for mode,v in evaluated.items():
        loss=ce(v['logits']);m=metrics(y.tolist(),v['logits'].argmax(-1).tolist(),labels,loss)
        all40=checked_all40(m,labels,b['supports']['assessment'])
        rows[mode]=dict(all40_macro_f1=all40,ce=loss,confusion_matrix=m['confusion_matrix'],
            supports=[x['support'] for x in m['per_class']],updates=trained[mode]['updates'],
            parameters=trained[mode]['registered_parameters'],initial_sha256=trained[mode]['initial_sha'],
            final_sha256=trained[mode]['final_sha256'],final_state_sha256=trained[mode]['final_state_sha256'],
            train_seconds=trained[mode]['outer_active_seconds'],assessment_seconds=v['seconds'],peak_allocated_bytes=max(trained[mode]['peak_allocated_bytes'],v['peak_allocated_bytes']))
    q=rows[cfg['quantum_mode']]['all40_macro_f1'];dce=ce(evaluated[cfg['quantum_mode']]['dephased'])
    margin=cfg['minimum_macro_margin']
    gates={'q_margin_over_'+k:q>=rows[k]['all40_macro_f1']+margin for k in modes if k!=cfg['quantum_mode']}
    gates.update({'q_margin_over_native_'+k:q>=r['all40_macro_f1']+margin for k,r in b['native_references'].items() if k=='quantum'})
    gates.update(q_margin_over_strong_gcov=q>=b['strong_reference']['all40_macro_f1']+margin,
        fixed_q_dephasing_ce_worse=dce>=rows[cfg['quantum_mode']]['ce']+cfg['minimum_dephasing_ce_gap'])
    assert len(gates)==cfg['critical_gate_count']
    np.savez_compressed(a.output/'predictions.private.npz',assessment_indices=np.asarray(data.cohort['assessment']),
        targets=y,q_same_state_dephased_logits=evaluated[cfg['quantum_mode']]['dephased'],
        **{k+'_logits':v['logits'] for k,v in evaluated.items()})
    summary=dict(status='COMPLETE_FIXED_PLUGIN_SCREEN',results=rows,gates=gates,gate=all(gates.values()),
        decision='PROMISING_PREPARE_FULL_SINGLE_SEED' if all(gates.values()) else 'STOP_EXACT_PLUGIN_PROFILE_RETURN_PARENT',
        q_same_state_dephased_ce=dce,minimum_macro_margin=margin,updates=len(modes)*cfg['updates'],seconds=time.time()-started,fit_barrier=barrier,
        latest_fit_finished=max(v['finished_at'] for v in trained.values()),
        cohort_sha256=b['cohort_sha256'],order_sha256=order_sha,binding_sha256=cfg['binding_sha256'],
        spec_sha256=a.spec_sha256,code_hashes=cfg['code_hashes'],execution='micro4',
        all_fits_before_assessment=True,valid_read=False,test_read=False,
        strong_reference=b['strong_reference'],native_references=b['native_references'],
        original_scoped_L4_retained=True,newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED',
        reference_audit_sha256=cfg['reference_audit_sha256'],max_peak_gpu_bytes=cfg['max_peak_gpu_bytes'],limits=cfg['limits'])
    core.atomic_json(a.output/'summary.json',summary)
    core.atomic_json(a.output/'RUN_COMPLETE',dict(summary_sha256=sha(a.output/'summary.json'),manifest_sha256=sha(a.output/'manifest.json')))
    phase='complete'
    for mode in modes:progress(mode,'complete',step=1024)
    print(json.dumps(dict(status=summary['status'],gate=summary['gate'],decision=summary['decision'],gates=gates,seconds=summary['seconds'])),flush=True)


if __name__=='__main__':
    main()
