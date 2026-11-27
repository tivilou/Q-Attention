"""Original Re-TACRED frozen-head screen adapter; reuse pinned fit and task.

This entry point requires an exact separately reviewed and measured admission
seal. It does not allocate resources, change the core or retry an attempt.
"""
import argparse,hashlib,json,math,os,random,signal,sys,threading,time
from pathlib import Path
from types import SimpleNamespace
import goal_aligned_stage as stage
import numpy as np

if sys.flags.optimize:
    raise RuntimeError('Unoptimized Python required; evidence assertions must be active')

TAGS=('quantum_subsequence','path_dephased_subsequence','signed_real16_subsequence','gaussian_covariance')
SIZES=dict(fit=4096,selection=1024,qualification=2048)
COHORT='2f358dda3c74666ec98a1394a558709e14279e6fa868d224eaf79776acc70fb2'
INIT='f79575534288a8e57fef61deeba24ba568af7e683e4c1ebc984dcc7cfb6e0781'
LABEL='6e822509a07c5c865130ebe315122a9d83e97b1b11496b9e86c6b2a600f8a429'
TRAIN_SOURCE='f1051d574a44df3ecbd036c1de4fbbe0a372c768f6ec990330e2cea4a657269f'
PROFILE='0b2da30bf1c0ca6b8809220c2fa72b598835c6e37c84b40c3c610ee4281e7e25'
CAP=3537.678
CAMPAIGN='ordered-distinct-subsequence-k3-seed13-screen-v1'
CLAIM_ROOT=Path('/hy-tmp/Q-Attention/control/ordered-subsequence-goal-aligned-claims')
BORROWED={'fixed_order_fit.py':'f63d3e0c4c5ce076abd67f97eab86c2f41ae35e4233bc6e7247785534710658c','sequence_task.py':'6f6a2dc86284ee1c4a2412cab734bdfa8a4b8ffb0ad8c1567d7ccc071aeb6e3d'}

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def parsed(p):return json.loads(Path(p).read_bytes())
def pred_sha(a):return hashlib.sha256(np.asarray(a,dtype='<f4').tobytes()).hexdigest()

def read_cohort(binding,profile,*,group_audit=True):
    p=Path(binding['cohort_path']);assert digest(p)==COHORT==profile['data']['cohort_sha256']
    j=parsed(p);assert j['schema']=='q-rmea.goal-aligned-small-cohort.v1' and j['sampler_revision']==2
    assert j['profile_id']=='q-rmea-small-screen-v1' and j['source_train_sha256']==TRAIN_SOURCE
    assert j['supports']==profile['data']['supports'] and j['train_class_counts']==profile['data']['train_class_counts']
    assert not j['valid_test_read'] and j['group_overlap']==0
    indices=j['indices'];assert set(indices)==set(SIZES)
    for k,n in SIZES.items():
        assert len(indices[k])==n and all(type(i) is int and 0<=i<58465 for i in indices[k])
    if group_audit:
        raw=Path(binding['train_source']).read_bytes();assert hashlib.sha256(raw).hexdigest()==TRAIN_SOURCE
        lookup={}
        for line in raw.splitlines():
            if not line.strip():continue
            row=json.loads(line);id_hash=hashlib.sha256(str(row['metadata']['id']).encode()).hexdigest()
            assert id_hash not in lookup
            lookup[id_hash]=hashlib.sha256(json.dumps(row['tokens'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        metadata=parsed(Path(binding['cache'])/'metadata.json')['train']
        assert len(lookup)==len(metadata)==58465 and len({m['id_hash'] for m in metadata})==58465
        groups=[lookup[m['id_hash']] for m in metadata];assert len(set(groups))==j['full_sentence_groups']==33477
        stage.validate_roles(indices,groups)
    else:groups=None
    order,order_hash=stage.batches(indices['fit'])
    return j,groups,order,order_hash

def checked_models(plugin,torch,RMEAHead,initialize_head,state_identity):
    random.seed(13);np.random.seed(13);torch.manual_seed(13)
    initial=RMEAHead('quantum').float().state_dict();assert state_identity(initial)==INIT
    models=plugin.make_models(13,initial);assert tuple(models)==TAGS
    assert [plugin.real_parameters(m) for m in models.values()]==[81028,81028,85748,82308]
    states=[m.state_dict() for m in models.values()]
    for s in states[:2]:assert set(s)==set(initial) and all(torch.equal(s[k],initial[k]) for k in initial)
    for s in states:
        assert all(torch.equal(s[k],initial[k]) for k in plugin.COMMON)
    for k in ('prepare.weight','prepare.bias'):assert torch.equal(states[2][k],initial[k])
    gcov=initialize_head('real12',initial,dict(seed=13,training_dtype='float32')).head.state_dict()
    assert set(states[3])==set(gcov)|{'moment_decoder'}
    assert all(torch.equal(states[3][k],gcov[k]) for k in gcov)
    assert torch.count_nonzero(states[3]['moment_decoder']).item()==0
    pointers=[x.untyped_storage().data_ptr() for s in [initial,*states] for x in s.values()]
    assert len(pointers)==len(set(pointers)),'Initial tensors alias'
    return models,{k:state_identity(m.state_dict()) for k,m in models.items()}

def gate_decision(interval,curves,q,controls,gaps):
    gates={f'margin_over_{c}':interval['point_deltas'][c]>=.01 for c in controls}
    gates['positive_paired_lower']=interval['minimum_delta_quantiles_05_95'][0]>0
    gates['both_final_checkpoints']=stage.stable(curves,q,controls)
    gates.update({f'{name}_gap':gaps[name]>=.005 for name in ('path_D','terminal_D')})
    if all(gates.values()):decision='promote_full_single_seed'
    elif not gates['both_final_checkpoints']:decision='defer_uncertain_or_cost'
    elif interval['minimum_delta_quantiles_05_95'][1]<.01:decision='stop_exact_configuration'
    elif all(gates[f'margin_over_{c}'] for c in controls) and gates['positive_paired_lower'] and not all(gates[f'{name}_gap' ] for name in ('path_D','terminal_D')):decision='stop_exact_configuration'
    else:decision='defer_uncertain_or_cost'
    return gates,decision

def require_device_count(record,spec_count,devices,actual_cuda_count=None):
    expected=record['existing_gpu_count']
    if (type(expected) is not int or type(spec_count) is not int or not 1<=expected<=2
        or spec_count!=expected or len(devices)!=expected or len(set(devices))!=expected or not all(devices)):
        raise ValueError('Admission, spec and distinct visible-device counts must agree')
    if actual_cuda_count is not None and actual_cuda_count!=expected:
        raise ValueError('CUDA enumerated lane count differs from admitted device count')
    return expected

def require_admission(record,binding,*,spec_sha,stage_bundle_sha,attempt,devices,spec_count,now):
    assert record['status']=='ADMITTED_MEASURED_GOAL_ALIGNED_SCREEN' and record['attempt']==attempt
    assert record['spec_sha256']==spec_sha and record['stage_bundle_sha256']==stage_bundle_sha
    assert record['cohort_sha256']==COHORT and record['order_sha256']==binding['order_sha256']
    assert record['technical_updates']==4 and record['previous_campaign_technical_updates']==4
    assert record['source_review_verdict']=='PASS' and record['semantic_preflight']=='PASS'
    assert record['native_observability_preflight']=='PASS' and record['measured_forecast']=='PASS'
    require_device_count(record,spec_count,devices)
    assert all(math.isfinite(record[k]) for k in ('first_request_at','science_request_at','technical_allocated_gpu_seconds','max_science_elapsed_seconds','predicted_science_gpu_seconds','predicted_science_elapsed_seconds','reserved_closeout_seconds'))
    assert record['first_request_at']<=record['science_request_at']<=now
    assert (0<record['technical_allocated_gpu_seconds']<CAP
            and 0<record['predicted_science_gpu_seconds']<=CAP-record['technical_allocated_gpu_seconds']
            and 0<record['predicted_science_elapsed_seconds']<=2400-(record['science_request_at']-record['first_request_at'])
            and 0<record['max_science_elapsed_seconds']<=2400 and record['reserved_closeout_seconds']>=10)
    usable_wall=min(record['first_request_at']+2400,
                    record['science_request_at']+record['max_science_elapsed_seconds'])-now
    queue_seconds=now-record['science_request_at']
    available_gpu=CAP-record['technical_allocated_gpu_seconds']-len(devices)*queue_seconds
    closeout=record['reserved_closeout_seconds']
    usable=math.floor(min(usable_wall,available_gpu/len(devices))-closeout)
    # Forecast fields measure the remaining work (including tail uncertainty,
    # excluding this separately reserved closeout). Charge idle allocated cards.
    work=max(record['predicted_science_elapsed_seconds'],record['predicted_science_gpu_seconds']/len(devices))
    if usable<1 or work>usable:
        raise TimeoutError('Measured remaining workload exceeds usable wall/GPU budget after queue and closeout')
    return usable

def claim_once(claim_root,record):
    """Consume the campaign even after a partial failure; never unlink on exit."""
    p=Path(claim_root);p.mkdir(exist_ok=True)
    claim=p/(CAMPAIGN+'.json')
    with claim.open('x',encoding='utf-8') as f:
        json.dump(record,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    if os.name=='posix':
        d=os.open(str(p),os.O_RDONLY)
        try:os.fsync(d)
        finally:os.close(d)
    return digest(claim)

def complete_chain(out,admission_sha,claim_sha):
    names=('summary.json','manifest.json','all_fits_selection_frozen.json','qualification.private.npz')
    pins={n:digest(out/n) for n in names}
    metrics=parsed(out/'summary.json');manifest=parsed(out/'manifest.json')
    assert metrics['admission_sha256']==manifest['admission_sha256']==admission_sha
    assert manifest['campaign_claim_sha256']==claim_sha
    assert metrics['qualification_sha256']==pins['qualification.private.npz']
    return dict(admission_sha256=admission_sha,campaign_claim_sha256=claim_sha,artifacts_sha256=pins)

def preflight(here,cfg):
    required={'goal_aligned_runner.py','goal_aligned_stage.py','rmea_ordered_subsequence.py','ordered_subsequence.py','rmea_plugin_registry.py','rmea_history_posterior.py'}
    assert set(cfg['code_hashes'])==required
    for n,p in cfg['code_hashes'].items():assert digest(here/n)==p
    profile_path=here/'profile.json';assert digest(profile_path)==PROFILE
    profile=parsed(profile_path)
    b=parsed(here/'binding.json');assert digest(here/'binding.json')==cfg['binding_sha256']
    for n,p in b['cache_pins'].items():
        f=Path(b['cache'])/n;assert n in ('train-h.npy','train-roles.npy','train-offsets.npy','train-targets.npy','metadata.json')
        assert f.stat().st_size==p['bytes']
        assert digest(f)==p['sha256']
    assert set(b['cache_pins'])=={'train-h.npy','train-roles.npy','train-offsets.npy','train-targets.npy','metadata.json'}
    for n,p in BORROWED.items():assert digest(Path(b['borrowed_directory'])/n)==p
    label=Path(b['source'])/'docs/contracts/q_rmea_entity_labels.json';assert digest(label)==LABEL
    labels=parsed(label);assert len(set(labels))==40 and labels[0]=='no_relation'
    cohort,groups,order,order_hash=read_cohort(b,profile);assert order_hash==b['order_sha256']
    return b,cohort,groups,order,labels

def run_science(a,cfg,b,cohort,groups,order,labels,claim_sha,admitted_device_count):
    # The stage-only adapter uses the same existing worker/core/task stack.
    here=Path(__file__).resolve().parent
    sys.path[:0]=[str(here),b['borrowed_directory'],b['source']]
    import torch, fixed_order_fit as core, sequence_task as task
    for m in (core,task):
        p=Path(m.__file__).resolve();assert p.parent==Path(b['borrowed_directory']).resolve() and digest(p)==BORROWED[p.name]
    from baselines.retacred.rmea import RMEAHead
    from baselines.retacred.rmea_matched_routers import initialize_head
    from baselines.retacred.rmea_token_transport_study import state_identity
    from baselines.retacred.rmea_token_transport_recovery import atomic_torch,cpu_tree
    from rmea_plugin_registry import resolve_plugin
    assert torch.__version__=='2.4.1+cu121' and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    devices=['cuda:'+str(i) for i in range(torch.cuda.device_count())]
    if len(devices)!=admitted_device_count or cfg['allocated_gpu_count']!=admitted_device_count:
        raise ValueError('Actual lane count changed after the admitted check')
    for d in devices:torch.cuda.set_per_process_memory_fraction(4294967296/torch.cuda.get_device_properties(d).total_memory,d)
    plugin=resolve_plugin('ordered_subsequence');assert Path(plugin.__file__).resolve()==here/'rmea_ordered_subsequence.py'
    models,identities=checked_models(plugin,torch,RMEAHead,initialize_head,state_identity);assert identities==cfg['initial_sha256']
    data=task.FrozenWords(b['cache'],b['cohort_path'])
    q,*controls=TAGS;indices=cohort['indices'];weights={};labels_by_role={}
    def open_role(role):
        y=np.asarray(data.data['targets'][indices[role]],dtype=np.int64)
        w=stage.row_weights(y,cohort['train_class_counts'],cohort['supports'][role])
        assert np.array_equal(w,np.asarray(cohort['assessment_row_weights'][role])[y])
        weights[role]=w;labels_by_role[role]=y
    open_role('selection')
    a.output.mkdir();(a.output/'fits').mkdir();folders={k:a.output/'fits'/k for k in TAGS}
    core.atomic_json(a.output/'manifest.json',dict(attempt=a.attempt,spec_sha256=a.spec_sha256,admission_sha256=a.admission_sha256,campaign_claim_sha256=claim_sha,source_hashes=cfg['code_hashes'],order_sha256=b['order_sha256'],stages=['fit+selection','frozen-all-arms barrier','qualification'],labels=labels))
    peak={}
    def check_peak(d):
        p=torch.cuda.max_memory_allocated(d);assert p<=4294967296
        return p
    def evaluate_rows(role,model,methods,stopped=None):
        assert role in weights
        device=next(model.parameters()).device;parts={k:[] for k in methods}
        with torch.no_grad():
            for start in range(0,SIZES[role],128):
                if stopped is not None:core.require_running(stopped)
                inp,y=data.batch(indices[role][start:start+128],device)
                assert np.array_equal(y.cpu().numpy(),labels_by_role[role][start:start+128])
                for name,fn in methods.items():
                    out=task.score_batch(fn,inp);assert out.shape==(128,40) and torch.isfinite(out).all()
                    parts[name].append(out.detach().cpu().numpy())
                check_peak(device)
        return {k:np.concatenate(v) for k,v in parts.items()}
    lock=threading.Lock();roster={k:dict(state='pending',step=0) for k in TAGS};phase='fit_selection'
    def progress(mode,state,**values):
        with lock:
            roster[mode].update(state=state,**values)
            core.atomic_json(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,phase=phase,current=sum(x['step'] for x in roster.values()),total=2560,unit='optimizer_update',observed_at=time.time(),heartbeat_at=time.time(),children=roster))
    def score_selection(model,mode,step):
        z=evaluate_rows('selection',model,{'logits':model},stop_signals[mode])['logits']
        np.save(folders[mode]/f'selection-{step}.private.npy',z,allow_pickle=False)
        return stage.evaluate(z,labels_by_role['selection'],weights['selection']),pred_sha(z)
    hook=stage.attach_selection_progress(progress,models=models,score_selection=score_selection,folders=folders,contract_sha=a.spec_sha256,atomic_torch=atomic_torch,cpu_tree=cpu_tree,state_identity=state_identity)
    def gradients(model):
        task.check_gradients(model);check_peak(next(model.parameters()).device)
    task_adapter=SimpleNamespace(batch=data.batch,optimizer=task.optimizer,backward=task.backward,check_gradients=gradients)
    stop_signals={}
    def train(mode,device,stopped):
        stop_signals[mode]=stopped
        return core.fit(models[mode],task_adapter,order,folders[mode],mode=mode,device=device,contract_sha=a.spec_sha256,order_sha=b['order_sha256'],progress=hook,stopped=stopped,atomic_torch=atomic_torch,cpu_tree=cpu_tree,state_identity=state_identity,checkpoint_every=32)
    trained=core.lanes(TAGS,devices,train,progress)
    curves={};selected={}
    for mode in TAGS:
        f=trained[mode];assert f['updates']==640 and f['sample_exposures']==81920 and f['base_unchanged']
        latest=torch.load(folders[mode]/'latest.pt',map_location='cpu',weights_only=True)
        assert latest['next_update']==640 and latest['order_sha']==b['order_sha256'] and latest['contract_sha']==a.spec_sha256
        assert state_identity(latest['model'])==f['final_state_sha256'] and digest(folders[mode]/'final.pt')==f['final_sha256']
        journal=stage.SelectionJournal(folders[mode],mode,a.spec_sha256)
        rows=journal.receipts(require_complete=True)
        curves[mode]={r['step']:r['weighted_macro_f1'] for r in rows};winner=stage.choose(rows);selected[mode]=winner
        chosen=folders[mode]/f"selection-{winner['step']}.private.pt";assert digest(chosen)==winner['checkpoint_sha']
        models[mode].load_state_dict(torch.load(chosen,map_location='cpu',weights_only=True),strict=True)
    core.atomic_json(a.output/'all_fits_selection_frozen.json',dict(attempt=a.attempt,selected=selected,curves=curves,last_fit_finished=max(x['finished_at'] for x in trained.values()),frozen_at=time.time()))
    open_role('qualification')
    phase='qualification'
    def qualify(mode,device,stopped):
        model=models[mode].to(device).eval();torch.cuda.reset_peak_memory_stats(device)
        methods={'logits':model}
        if mode==q:methods.update(path_D=model.same_state_dephased_logits,terminal_D=model.same_state_terminal_dephased_logits)
        before=state_identity(model.state_dict())
        result=evaluate_rows('qualification',model,methods,stopped)
        assert state_identity(model.state_dict())==before
        peak[mode]=check_peak(device);model.cpu();return result
    z=core.lanes(TAGS,devices,qualify,progress)
    y,w=labels_by_role['qualification'],weights['qualification']
    metrics={k:stage.evaluate(z[k]['logits'],y,w) for k in TAGS}
    preds={k:z[k]['logits'].argmax(-1) for k in TAGS}
    paired=stage.grouped_bootstrap(y,preds,w,[groups[i] for i in indices['qualification']],q,controls)
    gaps={k:stage.weighted_ce(z[q][k],y,w)-metrics[q]['weighted_ce'] for k in ('path_D','terminal_D')}
    gates,decision=gate_decision(paired,curves,q,controls,gaps)
    np.savez_compressed(a.output/'qualification.private.npz',targets=y,indices=np.asarray(indices['qualification']),**{k:v['logits'] for k,v in z.items()},path_D=z[q]['path_D'],terminal_D=z[q]['terminal_D'])
    summary=dict(status='COMPLETE_GOAL_ALIGNED_SCREEN',decision=decision,gates=gates,metrics=metrics,paired=paired,mechanism_weighted_ce_gaps=gaps,selected=selected,curves=curves,updates=2560,exposures=327680,peak_bytes=peak,assessment_row_forwards=24576,additional_row_forwards=4096,cohort_sha256=COHORT,order_sha256=b['order_sha256'],spec_sha256=a.spec_sha256,admission_sha256=a.admission_sha256,qualification_sha256=digest(a.output/'qualification.private.npz'),valid_test_read=False,new_L3='NOT_ACCEPTED',new_L4_T='NOT_ACCEPTED')
    core.atomic_json(a.output/'summary.json',summary)
    core.atomic_json(a.output/'RUN_COMPLETE',complete_chain(a.output,a.admission_sha256,claim_sha))
    phase='complete'
    for mode in TAGS:progress(mode,'complete')
    print(json.dumps(dict(status=summary['status'],decision=decision,gates=gates)))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('spec','admission','output','progress'):ap.add_argument('--'+key,type=Path,required=True)
    for key in ('spec-sha256','admission-sha256','attempt'):ap.add_argument('--'+key,required=True)
    a=ap.parse_args();here=Path(__file__).resolve().parent
    assert digest(a.spec)==a.spec_sha256 and digest(a.admission)==a.admission_sha256 and not a.output.exists()
    cfg=parsed(a.spec);assert cfg['scope']=='ONE_GOAL_ALIGNED_SUBSEQUENCE_SCREEN' and cfg['attempt']==a.attempt
    assert cfg['seed']==13 and cfg['execution']=='micro4' and cfg['updates_per_arm']==640 and cfg['logical_batch']==128
    # Wall cap is armed before expensive source/cohort checks.
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',');assert all(devices)
    b=parsed(here/'binding.json')
    receipt=parsed(a.admission)
    admission_args=dict(spec_sha=a.spec_sha256,stage_bundle_sha=hashlib.sha256(json.dumps(cfg['code_hashes'],sort_keys=True).encode()).hexdigest(),attempt=a.attempt,devices=devices,spec_count=cfg['allocated_gpu_count'])
    duration=require_admission(receipt,b,**admission_args,now=time.time())
    signal.alarm(duration)
    b,cohort,groups,order,labels=preflight(here,cfg)
    import torch
    assert torch.__version__=='2.4.1+cu121'
    count=require_device_count(receipt,cfg['allocated_gpu_count'],devices,torch.cuda.device_count())
    signal.alarm(require_admission(receipt,b,**admission_args,now=time.time()))
    claim_sha=claim_once(CLAIM_ROOT,dict(campaign=CAMPAIGN,attempt=a.attempt,spec_sha256=a.spec_sha256,admission_sha256=a.admission_sha256,output=str(a.output.resolve())))
    run_science(a,cfg,b,cohort,groups,order,labels,claim_sha,count)

if __name__=='__main__':main()
