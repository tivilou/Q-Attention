"""TRAIN-only bounded comparison; reuse accepted microbatch/metric/save helpers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    p = path.with_suffix('.pending')
    p.write_bytes((json.dumps(value, indent=2, allow_nan=False) + '\n').encode())
    p.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'inputs', 'reference', 'design', 'output', 'progress', 'allocation'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--attempt', required=True)
    args = ap.parse_args(); cfg = json.loads(args.design.read_bytes())
    allocation = json.loads(args.allocation.read_bytes())['allocation']
    assert allocation['runner_attempt_id'] == args.attempt and len(set(allocation['gpu_uuids'])) == 2
    # Expand only to both GPUs in this same confirmed campaign allocation.
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(allocation['gpu_uuids'])
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    for n, sha in cfg['code_hashes'].items():
        assert digest(Path(__file__).parent / n) == sha
    for n, sha in cfg['source_hashes'].items():
        assert digest(args.source / n) == sha
    assert digest(args.inputs/'discovery/manifest.json') == cfg['native_manifest_sha256']
    prior = args.reference/'output/diagnostic.private.npz'
    assert digest(prior) == cfg['split_reference_sha256']
    assert digest(args.reference/'diagnose.py') == cfg['bootstrap_code_sha256']
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output/'started.json').exists(), 'No scientific retry or overwrite'
    write(args.output/'started.json', dict(attempt=args.attempt, design_sha256=digest(args.design)))
    signal.alarm(cfg['timeout_seconds'])
    import torch
    from torch.nn import functional as F
    sys.path.insert(0, str(args.source))
    sys.path.insert(0, str(Path(__file__).parent))
    spec = importlib.util.spec_from_file_location('reviewed_shared_null', Path(__file__).parent/cfg.get('plugin_module', 'differential.py'))
    plugin = importlib.util.module_from_spec(spec); spec.loader.exec_module(plugin)
    from baselines.retacred.rmea import RMEAHead
    from baselines.retacred.rmea_task_runtime import backward, score_batch
    from baselines.retacred.rmea_token_transport_study import state_identity
    from baselines.retacred.rmea_token_transport_recovery import atomic_torch, cpu_tree
    from baselines.retacred.common import metrics
    dtype_name=cfg.get('training_dtype','float64');assert dtype_name in ('float32','float64')
    dtype=getattr(torch,dtype_name)
    if cfg.get('dense_batches',False):
        from baselines.retacred.rmea_fair_reference_runtime import backward,score_batch
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.device_count() == 2
    spec = importlib.util.spec_from_file_location('reviewed_bootstrap', args.reference/'diagnose.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    cache = args.inputs/'discovery/cache'
    data = {k: np.load(cache/('train-'+k+'.npy'), mmap_mode='r', allow_pickle=False)
            for k in ('h', 'roles', 'offsets', 'targets')}
    if cfg.get('cohort_file'):
        cohort_path = args.design.parent/cfg['cohort_file']
        assert cohort_path.parent == args.design.parent and digest(cohort_path) == cfg['cohort_sha256']
        cohort = json.loads(cohort_path.read_bytes())
        assert digest(cache/'train-targets.npy') == cohort['target_sha256']
        assert digest(cache/'metadata.json') == cohort['metadata_sha256']
        fit = np.asarray(cohort['fit'], dtype=np.int64)
        assess = np.asarray(cohort['assessment'], dtype=np.int64)
        assert (len(fit), len(assess)) == (cfg['fit'], cfg['assessment'])
        selected = np.concatenate((fit, assess))
        assert len(set(map(int, selected))) == len(selected) and selected.min() >= 0 and selected.max() < len(data['targets'])
        targets = np.array(data['targets'])
        groups = np.full(len(targets), '', dtype='<U64')
        for i in selected: groups[i] = cohort['selected_groups'][str(i)]
        assert all(groups[i] for i in selected)
        assert not set(groups[fit]) & set(groups[assess])
        assert np.bincount(targets[fit], minlength=40).tolist() == cohort['supports']['fit']
        assert np.bincount(targets[assess], minlength=40).tolist() == cohort['supports']['assessment']
        assert all(min(v) > 0 for v in cohort['supports'].values())
        calibration = np.zeros(len(targets), dtype=bool); calibration[fit] = True
    else:
        with np.load(prior, allow_pickle=False) as q:
            calibration = q['calibration'].copy(); groups = q['groups'].copy(); targets = q['labels'].copy()
        assert calibration.shape == targets.shape == groups.shape == (8192,)
        fit = np.flatnonzero(calibration); assess = np.flatnonzero(~calibration)
        assert (len(fit), len(assess)) == (4192, 4000)
        assert not set(groups[fit]) & set(groups[assess])
        assert np.array_equal(targets, data['targets'][:8192])
    rng = np.random.default_rng(cfg['seed']); batches = []
    while len(batches) < cfg['updates']:
        order = rng.permutation(fit)
        batches.extend(order[i:i+cfg['batch_size']] for i in range(0, len(order), cfg['batch_size']))
    batches = batches[:cfg['updates']]
    order_sha = hashlib.sha256(b''.join(x.astype('<i8').tobytes() for x in batches)).hexdigest()
    torch.manual_seed(cfg['seed'])
    initial = {k: v.clone() for k, v in RMEAHead('quantum').to(dtype=dtype).state_dict().items()}
    initial_sha = state_identity(initial)
    models = plugin.initialize(initial, cfg['seed'])
    assert tuple(models) == tuple(cfg['modes'])
    shared = {k: v for k,v in initial.items() if k in ('value.weight','value.bias','decoder','bias')}
    shared_sha = state_identity(shared)
    for model in models.values():
        assert state_identity({k: model.head.state_dict()[k] for k in shared}) == shared_sha
    started = time.time(); lock = threading.Lock(); cursors = {k: 0 for k in cfg['modes']}
    # Scientific per-condition compute budget, independent of scheduler leases.
    budget_lock=threading.Lock();active_deadlines={};budget_finished=threading.Event()
    def enter_budget(mode,remaining):
        assert remaining>0
        now=time.monotonic()
        with budget_lock:active_deadlines[mode]=(now,now+remaining)
    def leave_budget(mode):
        with budget_lock:
            now=time.monotonic();start,deadline=active_deadlines.pop(mode)
            assert now<deadline,'Per-arm compute deadline exhausted'
            return now-start
    def budget_watch():
        while not budget_finished.wait(.05):
            with budget_lock:
                expired=any(time.monotonic()>=deadline for start,deadline in active_deadlines.values())
            if expired:
                # Same default signal as the existing whole-campaign alarm.
                # Fail the complete scientific attempt, including in-flight work.
                os.kill(os.getpid(),signal.SIGALRM)
                return
    threading.Thread(target=budget_watch,daemon=True).start()

    def progress(mode, update):
        with lock:
            cursors[mode] = update
            write(args.progress, dict(schema='research.progress.v1', attempt=args.attempt,
                phase='train_'+mode, current=sum(cursors.values()), total=cfg['updates']*len(cfg['modes']),
                unit='optimizer_update', observed_at=time.time(), heartbeat_at=time.time(),
                children={k: dict(current=v, total=cfg['updates']) for k, v in cursors.items()}))

    def batch(indices, device):
        lengths = [int(data['offsets'][i+1]-data['offsets'][i]) for i in indices]
        h = np.zeros((len(indices), max(lengths), 1024), getattr(np,dtype_name))
        roles = np.zeros(h.shape[:2]+(4,), getattr(np,dtype_name)); mask = np.zeros(h.shape[:2], bool)
        for j, i in enumerate(indices):
            a, b = map(int, data['offsets'][i:i+2]); n = b-a
            h[j, :n] = data['h'][a:b]; roles[j, :n] = data['roles'][a:b]; mask[j, :n] = True
        inputs = tuple(torch.from_numpy(x).to(device) for x in (h, roles, mask))
        return inputs, torch.tensor(np.array(data['targets'][indices]), dtype=torch.long, device=device)

    def worker(mode, gpu):
        device = torch.device('cuda', gpu)
        with torch.cuda.device(device):
            arm_started=time.monotonic()
            enter_budget(mode,cfg.get('arm_timeout_seconds',float('inf')))
            model = models[mode]
            arm_initial_sha = state_identity(model.state_dict())
            model.to(device)
            parameter_count = sum(p.numel() for p in model.parameters())
            optimizer = (plugin.make_optimizer(model,cfg) if hasattr(plugin,'make_optimizer') else
                torch.optim.AdamW(model.parameters(), lr=cfg['lr'],
                    betas=tuple(cfg['betas']), eps=cfg['optimizer_eps'], weight_decay=cfg['weight_decay']))
            folder = args.output/mode; folder.mkdir()
            losses = []; first = None; retained_best = float('inf')
            for step, ids in enumerate(batches, 1):
                assert time.monotonic()-arm_started<cfg.get('arm_timeout_seconds',float('inf')),'Per-arm budget exhausted'
                if hasattr(model, 'set_training_progress'):
                    model.set_training_progress(step, cfg['updates'])
                model.train(); inputs, y = batch(ids, device); optimizer.zero_grad(set_to_none=True)
                loss = backward(model, inputs, y)
                assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                norms = {k: float(v.grad.norm()) for k, v in model.named_parameters()} if step == 1 else None
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_norm'], error_if_nonfinite=True)
                optimizer.step(); assert all(torch.isfinite(p).all() for p in model.parameters())
                losses.append(float(loss))
                if step == 1:
                    first = dict(ce=float(loss), finite_gradient_norms=norms,
                        state_changed=state_identity(model.state_dict()) != arm_initial_sha)
                    assert first['state_changed']
                # Existing atomic CPU-state serializer, no COS or resume claim.
                state = dict(head=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()),
                    update=step, initial_sha256=arm_initial_sha, order_sha256=order_sha, shared_value_sha256=shared_sha,
                    design_sha256=digest(args.design), selection='Final update only; local best is training-batch-loss retention only')
                atomic_torch(folder/'latest.pt', state)
                if float(loss) < retained_best:
                    atomic_torch(folder/'best.pt', state); retained_best = float(loss)
                progress(mode, step)
            atomic_torch(folder/'final.pt', cpu_tree(model.state_dict()))
            result = dict(updates=cfg['updates'], parameters=parameter_count, initial_sha256=arm_initial_sha,
                shared_value_sha256=shared_sha,
                order_sha256=order_sha, final_sha256=digest(folder/'final.pt'), first_update=first,
                train_ce_history=losses, training_finished_at=time.time())
            result['train_seconds']=leave_budget(mode)
            write(folder/'result.json', result)
            return result

    def evaluate(mode, gpu, trained):
        assessment_compute_started=time.monotonic()
        enter_budget(mode,cfg.get('arm_timeout_seconds',float('inf'))-trained[mode]['train_seconds'])
        device = torch.device('cuda', gpu); folder = args.output/mode
        # No assessment prediction until EVERY fit's immutable final state exists.
        assert all(row['updates'] == cfg['updates'] for row in trained.values())
        assert all(digest(args.output/k/'final.pt') == row['final_sha256'] for k,row in trained.items())
        latest = torch.load(folder/'latest.pt', map_location='cpu', weights_only=True)
        assert latest['update'] == cfg['updates']
        with torch.cuda.device(device):
            model = (plugin.make_head(mode) if hasattr(plugin, 'make_head') else plugin.SharedNullHead(mode)).to(dtype=dtype)
            model.load_state_dict(torch.load(folder/'final.pt', map_location='cpu', weights_only=True))
            assert state_identity(model.state_dict()) == state_identity(latest['head'])
            model.to(device); model.eval(); logits = []
            assessment_started_at = time.time()
            assert assessment_started_at >= max(row['training_finished_at'] for row in trained.values())
            with torch.no_grad():
                for lo in range(0, len(assess), cfg['batch_size']):
                    inputs, _ = batch(assess[lo:lo+cfg['batch_size']], device)
                    z = score_batch(model, inputs); assert torch.isfinite(z).all()
                    logits.append(z.cpu().numpy())
            logits = np.concatenate(logits); y = targets[assess]
            ce = F.cross_entropy(torch.tensor(logits), torch.tensor(y), reduction='none').numpy()
            row = metrics(y.tolist(), logits.argmax(-1).tolist(), ['no_relation']+['class_'+str(i) for i in range(1,40)], float(ce.mean()))
            macro = float(np.mean([q['f1'] for q in row['per_class'][1:]]))
            result = dict(trained[mode], ce=float(ce.mean()), positive39_macro_f1=macro,
                metrics=row, assessment_started_at=assessment_started_at)
            result['compute_seconds']=trained[mode]['train_seconds']+leave_budget(mode)
            assert result['compute_seconds']<cfg.get('arm_timeout_seconds',float('inf'))
            write(folder/'result.json', result)
            return result, logits, ce

    # Validate only the new readout; no training/resume/COS capability matrix.
    with torch.cuda.device(0):
        inputs, fixture_labels = batch(fit[:8], torch.device('cuda', 0))
        checks = plugin.fixtures(models,inputs,fixture_labels) if cfg.get('fixture_real_labels',False) else plugin.fixtures(models,inputs)
    write(args.output/'fixtures.json', checks)

    from queue import Queue, Empty
    ready = Queue()
    for mode in cfg['modes']:
        ready.put(mode)

    def lane(gpu):
        result = {}
        while True:
            try:
                mode = ready.get_nowait()
            except Empty:
                return result
            result[mode] = worker(mode, gpu)

    # One campaign, two fixed device lanes. Both settle before host return.
    with ThreadPoolExecutor(max_workers=2) as pool:
        left = pool.submit(lane, 0)
        right = pool.submit(lane, 1)
        trained = dict(left.result(), **right.result())
    assert set(trained) == set(cfg['modes'])
    # Scoring is cheap; serialize it after both training lanes settle.
    results = {mode: evaluate(mode, 1 if mode == 'diagonal' else 0, trained) for mode in cfg['modes']}
    rows = {k:{n:v[0][n] for n in ('ce','positive39_macro_f1','updates','parameters',
        'initial_sha256','order_sha256','final_sha256','training_finished_at','assessment_started_at','compute_seconds')}
        for k,v in results.items()}
    checks['first_updates'] = {k:dict(finite_gradients=True, state_changed=v[0]['first_update']['state_changed']) for k,v in results.items()}
    write(args.output/'fixtures.json', checks)
    if hasattr(plugin, 'summarize'):
        y = targets[assess]
        for mode in cfg['modes']:
            pred = results[mode][1].argmax(-1)
            rows[mode]['positive_to_null'] = int(((y > 0) & (pred == 0)).sum())
            rows[mode]['null_false_positive'] = int(((y == 0) & (pred != 0)).sum())
            rows[mode]['assessment_correct'] = int((y == pred).sum())
        np.savez(args.output/'diagnostic.private.npz', assessment_indices=assess,
            calibration=calibration, groups=groups, labels=targets,
            **{k+'_logits': v[1] for k, v in results.items()})
        summary = plugin.summarize(rows, {k:v[1] for k,v in results.items()}, y, cfg)
        summary.update(status='completed', attempt=args.attempt, seconds=time.time()-started,
            train_items=len(fit), assessment_items=len(assess), valid_read=False, test_read=False,
            code_hashes=cfg['code_hashes'], source_hashes=cfg['source_hashes'],
            design_sha256=digest(args.design), private_diagnostic_sha256=digest(args.output/'diagnostic.private.npz'), cos_writes=0)
        write(args.output/'summary.json', summary)
        write(args.output/'RUN_COMPLETE', {n:digest(args.output/n) for n in ('summary.json','fixtures.json')})
        write(args.progress, dict(schema='research.progress.v1', attempt=args.attempt, phase='completed',
            current=cfg['updates']*len(cfg['modes']), total=cfg['updates']*len(cfg['modes']), unit='optimizer_update', heartbeat_at=time.time(), observed_at=time.time()))
        print(json.dumps(dict(gate=summary['gate'], seconds=summary['seconds'], results={k:dict(ce=v['ce'],all40_macro_f1=v['all40_macro_f1']) for k,v in rows.items()})), flush=True)
        return
    comparators = tuple(k for k in cfg['modes'] if k != 'coherent')
    differences = np.column_stack([results[k][2]-results['coherent'][2] for k in comparators])
    boot = helper.bootstrap(differences, groups[assess], cfg['bootstrap_replicates'], cfg['bootstrap_seed'])
    lower = np.quantile(boot, .025, axis=0)
    y = targets[assess]
    for mode in cfg['modes']:
        prediction = results[mode][1].argmax(-1)
        rows[mode]['positive_to_null'] = int(((y != 0) & (prediction == 0)).sum())
        rows[mode]['null_false_positive'] = int(((y == 0) & (prediction != 0)).sum())
        rows[mode]['assessment_correct'] = int((y == prediction).sum())
    gates = {}
    for j, name in enumerate(comparators):
        gates['ce_beats_'+name] = rows['coherent']['ce'] < rows[name]['ce']
        gates['paired_lower_positive_'+name] = bool(lower[j] > 0)
        gates['macro_not_below_'+name] = rows['coherent']['positive39_macro_f1'] >= rows[name]['positive39_macro_f1']
    task_gates = dict(fewer_positive_to_null=rows['coherent']['positive_to_null'] < rows['native']['positive_to_null'],
                      no_extra_null_fp=rows['coherent']['null_false_positive'] <= rows['native']['null_false_positive'])
    utility_gate = all(v for k,v in gates.items() if not k.endswith('_diagonal')) and all(task_gates.values())
    attribution_gate = all(v for k,v in gates.items() if k.endswith('_diagonal'))
    np.savez(args.output/'diagnostic.private.npz', assessment_indices=assess, calibration=calibration,
        groups=groups, labels=targets, bootstrap=boot, differences=differences,
        **{k+'_logits': v[1] for k, v in results.items()})
    summary = dict(status='completed', attempt=args.attempt, seconds=time.time()-started,
        train_items=len(fit), assessment_items=len(assess), valid_read=False, test_read=False,
        results=rows, paired_ce_comparator_minus_coherent=dict(zip(comparators, map(float, differences.mean(0)))),
        paired_bootstrap_lower_025=dict(zip(comparators, map(float, lower))), gates=gates, task_gates=task_gates, attribution_gate=attribution_gate,
        gate='PASS_FOR_INDEPENDENT_STUDY_PREPARATION_ONLY' if utility_gate else 'NO_PROMOTION_EXACT_SCREEN',
        code_hashes=cfg['code_hashes'], source_hashes=cfg['source_hashes'], design_sha256=digest(args.design),
        private_diagnostic_sha256=digest(args.output/'diagnostic.private.npz'), cos_writes=0,
        limitations='Exposed TRAIN planning split; one initialization, fixed2048 updates/arm. Approximate sentence bootstrap, not independent utility/novelty/quantumnecessity or L5. Short-screen miss is not family impossibility.', L5='NOT_ACCEPTED')
    write(args.output/'summary.json', summary)
    write(args.output/'RUN_COMPLETE', {n:digest(args.output/n) for n in ('summary.json','fixtures.json')})
    write(args.progress, dict(schema='research.progress.v1', attempt=args.attempt, phase='completed',
        current=cfg['updates']*len(cfg['modes']), total=cfg['updates']*len(cfg['modes']), unit='optimizer_update', heartbeat_at=time.time(), observed_at=time.time()))
    print(json.dumps(dict(gate=summary['gate'], seconds=summary['seconds'], results={k:dict(ce=v['ce'],macro=v['positive39_macro_f1']) for k,v in rows.items()})), flush=True)



if __name__ == '__main__':
    main()
