"""Shared allowlisted plugin canary: one TRAIN128 clone update per condition; no assessment."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('spec','output','progress'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--spec-sha256',required=True)
    parser.add_argument('--attempt',required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    assert not sys.flags.optimize
    assert sha(args.spec)==args.spec_sha256
    cfg = json.loads(args.spec.read_bytes())
    assert cfg['scope']=='ONE_TRAIN128_PLUGIN_TECHNICAL_UPDATE_ONLY'
    assert cfg['runner_attempt_id']==args.attempt and cfg['seed']==13
    signal.alarm(cfg['wall_seconds'])
    for name,pin in cfg['code_hashes'].items():
        assert Path(name).name==name and sha(here/name)==pin
    assert sha(here/'binding.json')==cfg['binding_sha256']
    binding = json.loads((here/'binding.json').read_bytes())
    # Reuse old zero-task input guard, never the old scientific attempt.
    old = cfg['retained_guard_contract']
    argv = old['argv']+['--check-only']
    guard = subprocess.run(argv,capture_output=True,text=True,timeout=120)
    assert guard.returncode==0,guard.stderr
    guarded = json.loads(guard.stdout)
    assert guarded['status']=='PASS_SEALED_CANARY_INPUT_GUARDS'
    assert guarded['order_sha256']==binding['order_sha256']
    for name,pin in cfg['borrowed_hashes'].items():
        assert sha(Path(cfg['borrowed_directory'])/name)==pin
    assert not args.output.exists(),'Inspect the existing attempt, no retry overwrite'
    args.output.mkdir()
    sys.path[:0] = [str(here),cfg['borrowed_directory'],binding['source']]
    import numpy as np
    import torch
    from rmea_plugin_registry import resolve_plugin
    plugin = resolve_plugin(cfg['plugin_id'])
    from baselines.retacred.rmea import RMEAHead
    import copy
    import sequence_task as task
    from baselines.retacred.rmea_token_transport_study import state_identity
    from baselines.retacred.rmea_token_transport_recovery import atomic_torch,cpu_tree
    from baselines.retacred.rmea_task_runtime import chunks
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.device_count()==1
    device = torch.device('cuda:0')
    data = task.FrozenWords(binding['cache'],binding['cohort_path'])
    ids = np.random.default_rng(13).permutation(data.cohort['fit'])[:128]
    assert hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()==guarded['first_batch_sha256']
    inputs,targets = data.batch(ids,device)
    assert len(list(chunks(inputs)))==32
    torch.manual_seed(13)
    initial = RMEAHead('quantum').float().state_dict()
    assert state_identity(initial)=='f79575534288a8e57fef61deeba24ba568af7e683e4c1ebc984dcc7cfb6e0781'
    models = plugin.make_models(13, initial)
    assert list(models)==cfg['modes']
    assert {k:plugin.real_parameters(m) for k,m in models.items()}==cfg['real_parameters']
    assert {k:state_identity(m.state_dict()) for k,m in models.items()}==cfg['initial_model_identities']
    for model in models.values():
        for key in plugin.COMMON:
            assert torch.equal(model.state_dict()[key],initial[key])
    atomic_torch(args.output/'initial.private.pt',{k:cpu_tree(m.state_dict()) for k,m in models.items()})
    rows = {}
    start = time.monotonic()
    for i,(name,model) in enumerate(models.items()):
        args.progress.write_bytes((json.dumps(dict(schema='research.progress.v1',attempt=args.attempt,phase=name,
            current=i,total=len(models),unit='technical_condition',heartbeat_at=time.time(),observed_at=time.time()))+'\n').encode())
        model.to(device).train()
        first = tuple(x[:4] for x in inputs)
        with torch.no_grad():
            gpu_before = model(*first).cpu()
            cpu_reference = copy.deepcopy(model).cpu().double()
            cpu_reference.load_state_dict(cpu_tree(model.state_dict()))
            cpu_before = cpu_reference(*(x.cpu().double() if x.dtype!=torch.bool else x.cpu() for x in first))
            # Probabilities are invariant to a common class logit offset.
            centered_gpu = gpu_before.double()-gpu_before.double().mean(-1,keepdim=True)
            centered_cpu = cpu_before-cpu_before.mean(-1,keepdim=True)
            parity_error = float((centered_gpu-centered_cpu).abs().max())
            assert parity_error<=cfg['centered_logit_tolerance'],parity_error
        del cpu_reference
        before = state_identity(model.state_dict())
        opt = task.optimizer(model)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        tick = time.monotonic()
        opt.zero_grad(set_to_none=True)
        loss = task.backward(model,inputs,targets)
        task.check_gradients(model)
        gradients = {n:dict(norm=float(p.grad.norm()),nonzero=int(torch.count_nonzero(p.grad)),elements=p.numel(),complex=p.is_complex()) for n,p in model.named_parameters()}
        assert all(v['nonzero']>0 for v in gradients.values())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        torch.cuda.synchronize(device)
        seconds = time.monotonic()-tick
        assert all(torch.isfinite(p).all() for p in model.parameters())
        after = state_identity(model.state_dict())
        assert after!=before
        with torch.no_grad():
            logits = task.score_batch(model,inputs)
            assert logits.shape==(128,40) and torch.isfinite(logits).all()
            first = tuple(x[:4] for x in inputs)
            pad = (torch.nn.functional.pad(first[0],(0,0,0,2),value=float('nan')),
                   torch.nn.functional.pad(first[1],(0,0,0,2),value=float('nan')),
                   torch.nn.functional.pad(first[2],(0,2)))
            padding_error=float((model(*first)-model(*pad)).abs().max())
            assert padding_error<=cfg['centered_logit_tolerance']
        checkpoint = args.output/(name+'-technical.private.pt')
        atomic_torch(checkpoint,dict(model=cpu_tree(model.state_dict()),optimizer=cpu_tree(opt.state_dict())))
        saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert state_identity(saved['model'])==after
        assert len(saved['optimizer']['state'])==len(list(model.parameters()))
        rows[name] = dict(raw_real_parameters=plugin.real_parameters(model),registered_elements=sum(p.numel() for p in model.parameters()),
            initial_sha256=before,final_sha256=after,updates=1,logical_batch=128,microbatch=4,
            train_ce=float(loss),update_seconds=seconds,peak_gpu_bytes=torch.cuda.max_memory_allocated(device),
            initial_cpu64_centered_logit_error=parity_error,padding_error=padding_error,gradients=gradients,state_roundtrip=True)
        (args.output/'partial-rows.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
        assert seconds<=cfg['max_one_update_seconds'],dict(condition=name,seconds=seconds,ceiling=cfg['max_one_update_seconds'])
        assert rows[name]['peak_gpu_bytes']<=cfg['max_peak_gpu_bytes']
        assert sum(v['update_seconds'] for v in rows.values())<=cfg['max_total_update_seconds']
        model.cpu()
        del opt,saved
        torch.cuda.empty_cache()
    summary = dict(status='PASS_REAL_TRAIN128_PLUGIN_TECHNICAL',attempt=args.attempt,results=rows,
        binding_sha256=cfg['binding_sha256'],spec_sha256=args.spec_sha256,code_hashes=cfg['code_hashes'],
        cpu64_tolerance=cfg['centered_logit_tolerance'],word_length_summary=dict(min=int(inputs[2].sum(-1).min()),max=int(inputs[2].sum(-1).max()),
            mean=float(inputs[2].sum(-1).float().mean())),first_batch_sha256=guarded['first_batch_sha256'],
        torch=torch.__version__,numpy=np.__version__,seconds=time.monotonic()-start,
        task_updates=len(models),task_examples=128,assessment_examples=0,valid_examples=0,test_examples=0,
        guards=dict(argv=argv,exit_code=guard.returncode,status=guarded['status']),
        limits=['First TRAIN batch technical check, not efficacy or independent confirmation.',
            'One cold update per arm; extrapolation is a cost lead, not a full-cohort runtime bound.',
            'No automatic utility admission, seed/budget/state/basis rescue, or old profile restart.'])
    (args.output/'summary.json').write_bytes((json.dumps(summary,indent=2,allow_nan=False)+'\n').encode())
    (args.output/'RUN_COMPLETE').write_bytes((json.dumps(dict(summary_sha256=sha(args.output/'summary.json')))+'\n').encode())
    args.progress.write_bytes((json.dumps(dict(schema='research.progress.v1',attempt=args.attempt,phase='complete',
        current=len(models),total=len(models),unit='technical_condition',heartbeat_at=time.time(),observed_at=time.time()))+'\n').encode())
    print(json.dumps(dict(status=summary['status'],seconds=summary['seconds'],updates=len(models),
        update_seconds={n:v['update_seconds'] for n,v in rows.items()})))


if __name__=='__main__':
    main()
