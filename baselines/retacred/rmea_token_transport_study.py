"""Four matched final-state heads; full-data execution is collaborator-only."""
import json
import random
import argparse
import gc
import hashlib
import os
from pathlib import Path
import sys
from . import rmea_token_transport_protocol as p

RECOVERED_FIXTURE_SHA = '176cfe2369436d340ca618e1a6e43f4743755a2cded1dcfe60221b2e85588d83'


def fixture_identity(archive):
    """Keep the historical pin; separately identify the verified COS cache repack."""
    from .rmea_entity_study import FIXTURE_SHA
    value = p.digest(archive)
    if value not in (FIXTURE_SHA, RECOVERED_FIXTURE_SHA):
        raise ValueError('Pinned tiny fixture required')
    return value


def load_fixture(archive, output):
    import io
    import tarfile
    from .rmea_entity_study import fixture, FIXTURE_SHA
    value = fixture_identity(archive)
    if value == FIXTURE_SHA:
        fixture(archive, output)
        return value
    # Same archived feature bytes, not a reconstruction of the old tar identity.
    # This new pin is technical-canary-only; the discovery path cannot use it.
    expected = {'canary/manifest.json', 'canary/cache/metadata.json'} | {
        f'canary/cache/{split}-{field}.npy' for split in ('train', 'valid', 'test')
        for field in ('encoder-0', 'encoder-1', 'h', 'roles', 'offsets', 'targets')}
    with Path(archive).open('rb') as src:
        snapshot = src.read(3055802)
    if len(snapshot) != 3055801 or hashlib.sha256(snapshot).hexdigest() != RECOVERED_FIXTURE_SHA:
        raise ValueError('Recovered fixture snapshot differs')
    with tarfile.open(fileobj=io.BytesIO(snapshot), mode='r:gz') as tar:
        members = tar.getmembers()
        if (len(members) != 20 or {v.name for v in members} != expected
                or any(not v.isfile() or not 0 < v.size <= 2**21 for v in members)
                or sum(v.size for v in members) != 4118454):
            raise ValueError('Recovered fixture inventory differs')
        manifest = json.load(tar.extractfile('canary/manifest.json'))
        if (manifest['config']['counts'] != {'train': 16, 'valid': 8, 'test': 8}
                or manifest['config']['encoder_revision'] != p.config('canary')['encoder_revision']):
            raise ValueError('Recovered tiny cache contract differs')
        for member in members:
            if member.name.startswith('canary/cache/'):
                target = output / Path(member.name).relative_to('canary')
                target.parent.mkdir(exist_ok=True)
                with target.open('xb') as dst:
                    dst.write(tar.extractfile(member).read())
    return value


def state_identity(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        a = value.detach().cpu().numpy()
        h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def train(cfg, output, labels, *, data_dir=None, assets=None, recovery=None, modes=None, finalize=True, runtime=None):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from .common import metrics
    from .rmea import RMEAHead
    from .rmea_token_transport import TrainableTokenTransportHead, route
    from .rmea_token_transport_diagnostics import inspect_unit, aggregate
    from .rmea_token_transport_trace import capture, close
    p = runtime.protocol if runtime is not None else globals()['p']
    from .rmea_training_options import batch_size, per_epoch, dtype_name, selection_metric, selection_description, reported_checkpoint
    logical_batch = batch_size(cfg); tensor_dtype = getattr(torch, dtype_name(cfg))
    array_dtype = getattr(np, dtype_name(cfg)); metric_name = selection_metric(cfg)
    report_policy = reported_checkpoint(cfg)
    if report_policy=='validation_best' and (runtime is None or not hasattr(runtime,'capture_selected')):
        raise ValueError('Validation-best reporting needs explicit selected-checkpoint capture callback')
    if len(labels) != 40 or labels[0] != 'no_relation':
        raise ValueError('Fixed train-derived label mapping differs')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    cache = output / 'cache'; metadata = p.read(cache / 'metadata.json')
    data = {s: {k: np.load(cache / f'{s}-{k}.npy', mmap_mode='r', allow_pickle=False)
                for k in ('h', 'roles', 'offsets', 'targets')} for s in cfg['counts']}
    for split, count in cfg['counts'].items():
        d = data[split]
        if (len(d['targets']) != count or len(d['offsets']) != count + 1 or len(metadata[split]) != count
                or d['offsets'][0] != 0 or np.any(np.diff(d['offsets']) <= 0)
                or d['offsets'][-1] != len(d['h']) or d['h'].shape[1:] != (1024,)
                or d['roles'].shape != (len(d['h']), 4)
                or not np.isin(d['targets'], np.arange(40)).all()):
            raise ValueError('Invalid fixed feature cache: ' + split)

    def batch(split, indices):
        d = data[split]; lengths = [int(d['offsets'][i+1] - d['offsets'][i]) for i in indices]
        h = np.zeros((len(indices), max(lengths), 1024), array_dtype)
        r = np.zeros((len(indices), max(lengths), 4), array_dtype); mask = np.zeros(h.shape[:2], bool)
        for j, (i, length) in enumerate(zip(indices, lengths)):
            a, b = map(int, d['offsets'][i:i+2]); h[j, :length] = d['h'][a:b]
            r[j, :length] = d['roles'][a:b]; mask[j, :length] = True
        inputs = tuple(torch.from_numpy(a).to('cuda') for a in (h, r, mask))
        y = torch.from_numpy(np.array(d['targets'][indices], dtype=np.int64)).to('cuda')
        return inputs, y

    @torch.no_grad()
    def evaluate(head, split, routing_kind=None):
        head.eval(); gold = []; predicted = []; loss = 0.
        evaluation_batch = cfg.get('eval_batch_size', 32)
        if type(evaluation_batch) is not int or evaluation_batch < 1: raise ValueError('Invalid evaluation batch')
        for start in range(0, cfg['counts'][split], evaluation_batch):
            inputs, y = batch(split, list(range(start, min(start+evaluation_batch, cfg['counts'][split]))))
            if routing_kind is None:
                logits = runtime.score_batch(head, inputs) if runtime is not None else head(*inputs)[0]
            else:
                d = head(*inputs, details=True)
                alpha = route(d['psi'], d['phi'], inputs[2], kind=routing_kind)['alpha']
                logits = (alpha * d['utilities']).sum(-1) + head.bias
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite final logits')
            loss += float(F.cross_entropy(logits, y, reduction='sum'))
            gold += y.tolist(); predicted += logits.argmax(-1).tolist()
        row = metrics(gold, predicted, labels, loss / len(gold))
        row['positive39_macro_f1'] = sum(r['f1'] for r in row['per_class'][1:]) / 39
        row['positive39_convention'] = 'fixed39labels_excluding_no_relation_zero_division0'
        row['ce'] = row['loss']
        return row

    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    initial = {k: v.clone() for k, v in RMEAHead('quantum').to(dtype=tensor_dtype).state_dict().items()}
    initial_sha = state_identity(initial)
    generator = cfg.get('order_generator', 'numpy_default_rng')
    if generator == 'torch_epoch_seed':
        orders = [torch.randperm(cfg['counts']['train'], generator=torch.Generator().manual_seed(cfg['seed']+e)).numpy() for e in range(cfg['epochs'])]
    elif generator == 'numpy_default_rng':
        rng = np.random.default_rng(cfg['seed'])
        orders = [rng.permutation(cfg['counts']['train']) for _ in range(cfg['epochs'])]
    else: raise ValueError('Unknown epoch permutation generator')
    order_sha = hashlib.sha256(b''.join(o.astype('<i8').tobytes() for o in orders)).hexdigest()
    from .rmea_token_transport_recovery import atomic_torch, cpu_tree, restore_rng
    checkpoints = output / 'checkpoints'; checkpoints.mkdir(exist_ok=True)
    resumed = recovery.state if recovery is not None else None
    if resumed and (resumed['initial_sha'] != initial_sha or resumed['order_sha'] != order_sha):
        raise ValueError('Resume initialization/order differs')
    completed = dict(resumed['completed']) if resumed else {}
    best = dict(resumed['best']) if resumed else {}
    if recovery is not None: atomic_torch(recovery.best, best)
    if recovery is not None:
        if recovery.state is None:
            recovery.save(mode=0,epoch=0,next_start=0,history=[],loss_sum=0.,head=None,optimizer=None,
                completed=completed,best=best,initial_sha=initial_sha,order_sha=order_sha)
        if cfg['profile']=='discovery':
            if runtime is None: recovery.prepare(labels,data_dir)
            else: runtime.prepare(recovery,labels,data_dir)
        recovery.drain_pending()
    results = {}; samples = []; diagnostics = None
    for mode_index, kind in enumerate(tuple(modes or p.MODES)):
        boundary=(resumed is not None and mode_index==resumed['cursor']['mode'] and resumed['active'] is None)
        if boundary:
            cursor=resumed['cursor']
            if cursor['epoch'] or cursor['next_start'] or cursor['history'] or cursor['loss_sum'] or resumed['optimizer'] is not None:
                raise ValueError('Fresh-head boundary state differs')
            restore_rng(resumed['rng'])
        if runtime is not None and hasattr(runtime, 'initialize_head'):
            head = runtime.initialize_head(kind, initial, cfg).to(dtype=tensor_dtype)
            runtime.validate_initialized_head(head, kind, initial, cfg)
        else:
            head = (runtime.make_head(kind) if runtime is not None else TrainableTokenTransportHead(kind)).to(dtype=tensor_dtype); head.load_state_dict(initial)
            if state_identity(head.state_dict()) != initial_sha or sum(v.numel() for v in head.parameters()) != 81028:
                raise ValueError('Initialization or parameter matching differs')
        arm_initial_sha = state_identity(head.state_dict())
        head.to('cuda'); optimizer = torch.optim.AdamW(head.parameters(), lr=cfg['lr'],
            weight_decay=cfg['weight_decay'], betas=tuple(cfg['betas']), eps=cfg['optimizer_eps'])
        history = []; steps = 0; first_epoch = 0; first_start = 0; saved_loss = 0.
        if kind in completed:
            row = completed[kind]; head.load_state_dict(row['state'])
            history = row['history']; steps = row['steps']; first_epoch = cfg['epochs']
        elif resumed and mode_index == resumed['cursor']['mode'] and not boundary:
            cursor = resumed['cursor']; first_epoch = cursor['epoch']; first_start = cursor['next_start']
            history = list(cursor['history']); saved_loss = cursor['loss_sum']
            steps = first_epoch*per_epoch(cfg)+(first_start+logical_batch-1)//logical_batch
            if resumed['active'] is None or resumed['optimizer'] is None:
                raise ValueError('Active model/optimizer missing')
            head.load_state_dict(resumed['active']); optimizer.load_state_dict(resumed['optimizer'])
            restore_rng(resumed['rng'])
        def save(epoch_index, next_start, loss_sum, *, finished=False):
            if recovery is not None:
                recovery.save(mode=mode_index+int(finished), epoch=0 if finished else epoch_index,
                    next_start=0 if finished else next_start, history=[] if finished else history,
                    loss_sum=0. if finished else loss_sum,
                    head=None if finished else head, optimizer=None if finished else optimizer,
                    completed=completed, best=best, initial_sha=initial_sha, order_sha=order_sha)
        for epoch_index in range(first_epoch, cfg['epochs']):
            epoch = epoch_index+1; order = orders[epoch_index]
            head.train(); loss_sum = saved_loss if epoch_index == first_epoch else 0.
            start_at = first_start if epoch_index == first_epoch else 0
            for start in range(start_at, len(order), logical_batch):
                if recovery is not None and hasattr(recovery,'before_update'): recovery.before_update()
                ids = order[start:start+logical_batch]; inputs, y = batch('train', ids)
                optimizer.zero_grad(set_to_none=True)
                if runtime is None:
                    loss = F.cross_entropy(head(*inputs)[0], y)
                    if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                    loss.backward()
                else:
                    loss = runtime.backward(head,inputs,y)
                    if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                if not all(v.grad is not None and torch.isfinite(v.grad).all() for v in head.parameters()):
                    raise ValueError('Missing or nonfinite gradient')
                torch.nn.utils.clip_grad_norm_(head.parameters(), cfg['clip_norm'], error_if_nonfinite=True)
                optimizer.step(); steps += 1; loss_sum += float(loss.detach()) * len(ids)
                next_start = min(start+logical_batch,len(order)); save(epoch_index,next_start,loss_sum)
                if recovery is not None and next_start < len(order): recovery.milestone()
            if not all(torch.isfinite(v).all() for v in head.parameters()):
                raise ValueError('Nonfinite final parameters')
            history.append(dict(epoch=epoch, train_ce=loss_sum / len(order)))
            # Commit latest (which embeds best) before updating the derived best file.
            validation = evaluate(head,'valid')[metric_name]
            if kind not in best or validation > best[kind]['validation_'+metric_name]:
                best[kind] = dict(state=cpu_tree(head.state_dict()), epoch=epoch, selection=selection_description(cfg))
                best[kind]['validation_'+metric_name] = validation
            save(epoch_index+1,0,0.)
            if recovery is not None: atomic_torch(recovery.best,best)
            if recovery is not None: recovery.milestone()
            print(json.dumps(dict(stage='train', condition=kind, epoch=epoch, updates=steps)), flush=True)
        head.eval(); final_state = cpu_tree(head.state_dict())
        cp = checkpoints / (kind + '-final.pt')
        if cp.exists():
            if state_identity(torch.load(cp,map_location='cpu',weights_only=True))!=state_identity(final_state):
                raise ValueError('Retained final differs; preserve evidence')
        else:atomic_torch(cp, final_state)
        if cp.stat().st_size > 1024**2: raise ValueError('Tiny final checkpoint bound exceeded')
        if kind not in completed:
            completed[kind] = dict(state=final_state,history=history,steps=steps,optimizer=cpu_tree(optimizer.state_dict()))
            save(0,0,0.,finished=True)
            if recovery is not None: recovery.milestone(final=mode_index==len(p.MODES)-1)
        loaded = torch.load(cp, map_location='cpu', weights_only=True)
        if state_identity(loaded) != state_identity(final_state):
            raise ValueError('Final checkpoint serialization differs')
        head.load_state_dict(loaded)
        evaluation_state = None
        if report_policy == 'validation_best':
            chosen = best[kind]; head.load_state_dict(chosen['state'])
            selected = checkpoints / (kind + '-selected.pt')
            if not selected.exists():atomic_torch(selected, chosen['state'])
            selected_state=torch.load(selected,map_location='cpu',weights_only=True)
            if state_identity(selected_state)!=state_identity(chosen['state']):
                raise ValueError('Selected serialization differs from committed best')
            head.load_state_dict(selected_state)
            evaluation_state = dict(path='checkpoints/'+selected.name, sha256=p.digest(selected), state_sha256=state_identity(selected_state), epoch=chosen['epoch'])
        result = dict(selected_epoch=best[kind]['epoch'] if report_policy == 'validation_best' else cfg['epochs'], selection=cfg['selection'], parameters=sum(v.numel() for v in head.parameters()),
            updates=steps, history=history, initial_state_sha256=initial_sha,
            final_state_sha256=state_identity(loaded), checkpoint_sha256=p.digest(cp),
            valid=evaluate(head, 'valid'), test=evaluate(head, 'test'))
        if evaluation_state is not None: result['evaluation_checkpoint'] = evaluation_state
        if runtime is not None and hasattr(runtime, 'initialize_head'): result['arm_initial_state_sha256'] = arm_initial_sha
        results[kind] = result
        for split in cfg['counts']:
            if evaluation_state is not None:
                samples += runtime.capture_selected(output,head,kind,split,batch,metadata,evaluation_state)
                continue
            samples += (runtime.capture(output,head,kind,split,batch,metadata) if runtime is not None
                        else capture(output,head,kind,split,batch,metadata))
        if kind == 'cubic_coherent':
            frozen = {k: evaluate(head, 'test', k) for k in p.MODES}
            if frozen[kind] != result['test']:
                raise ValueError('Actual and cubic replay metrics differ')
            units = []
            for i in range(min(cfg['diagnostic_samples'], cfg['counts']['test'])):
                inputs, y = batch('test', [i])
                with torch.no_grad():
                    d = head(*inputs, details=True)
                row = inspect_unit(d['psi'][0], d['phi'], d['utilities'][0], head.bias, int(y[0]))
                row['unit_sha256'] = metadata['test'][i]['id_hash']; units.append(row)
            p.atomic_json(output / 'private-geometry.json', dict(selection='first32 hash-ordered testunits, before outcomes', units=units))
            diagnostics = dict(frozen_cubic_route_metrics=frozen, geometry=aggregate(units))
        del head, optimizer, final_state, loaded; gc.collect(); torch.cuda.empty_cache()
    if recovery is not None: recovery.milestone(final=True)
    part = dict(results=results, samples=samples, diagnostics=diagnostics,
                initial_sha=initial_sha, order_sha=order_sha)
    if finalize:
        if runtime is None: finish(cfg, output, [part], data_dir=data_dir, assets=assets)
        else: runtime.finish(cfg,output,[part],data_dir=data_dir,assets=assets)
    return part


def finish(cfg, output, parts, *, data_dir=None, assets=None):
    from .rmea_token_transport_trace import close
    results = {}; samples = []; diagnostics = None
    if len({row['initial_sha'] for row in parts}) != 1 or len({row['order_sha'] for row in parts}) != 1:
        raise ValueError('Independent jobs changed common initialization/order')
    initial_sha, order_sha = parts[0]['initial_sha'], parts[0]['order_sha']
    for row in parts:
        if set(results) & set(row['results']): raise ValueError('Duplicate completed condition')
        results.update(row['results']); samples += row['samples']
        if row['diagnostics'] is not None:
            if diagnostics is not None: raise ValueError('Duplicate cubic diagnostics')
            diagnostics = row['diagnostics']
    if set(results) != set(p.MODES): raise ValueError('Missing condition; no summary seal')
    metadata = p.read(output/'cache/metadata.json')
    trace = close(output, samples, cfg, metadata, data_dir=data_dir, assets=assets)
    cubic = results['cubic_coherent']['test']; native = results['native']['test']
    contrasts = {k: dict(positive39_macro_f1=cubic['positive39_macro_f1']-results[k]['test']['positive39_macro_f1'],
                        ce=cubic['ce']-results[k]['test']['ce']) for k in p.MODES if k != 'cubic_coherent'}
    gate = (all(v['positive39_macro_f1'] > 0 for v in contrasts.values()) and cubic['ce'] <= native['ce'])
    delta = cubic['positive39_macro_f1'] - native['positive39_macro_f1']
    summary = dict(schema='q-rmea.token-transport.results.v1', results=results, coherent_contrasts=contrasts,
        common_initialization_equal=len({r['initial_state_sha256'] for r in results.values()}) == 1,
        common_initial_state_sha256=initial_sha, common_epoch_order_sha256=order_sha,
        discovery_gate=gate, decision=('SINGLE_SEED_GATE_PASS_PENDING_OWNER_AUDIT' if gate else 'NO_AUTOMATIC_PROMOTION')
            if cfg['profile']=='discovery' else 'TECHNICAL_CANARY_ONLY',
        versus_native_absolute_delta=delta,
        versus_native_relative_delta=delta/abs(native['positive39_macro_f1']) if native['positive39_macro_f1'] else None,
        full_data_training=cfg['profile']=='discovery', test_selection=False, seed=13,
        diagnostics=diagnostics, trace=trace, L5='NOT_ACCEPTED',
        limitations=['One seed on previously exposed official splits; no population/seed-stability inference.',
            'Geometry does not establish useful routing, coherence necessity or total-gain mediation.',
            'Native is a new same-budget control, not a rerun of historical baselines.',
            'No novelty, quantum advantage or inherited L4 claim. No automatic rescue or replication.'])
    p.atomic_json(output / 'summary.json', summary)


def run(args):
    import torch
    cfg = p.config(args.profile); p.review_gate(formal=args.profile=='discovery')
    if p.git('status', '--porcelain'):
        raise ValueError('Clean reviewed source required')
    if args.profile=='discovery' and (not any(os.environ.get(k)=='1' for k in
            ('QATTN_COLLABORATOR_FORMAL','QATTN_OWNER_MANAGED')) or p.git('branch','--show-current')!='1.1'):
        raise ValueError('Full formal run requires the reviewed parent and branch1.1')
    if args.fixture_cache and args.profile != 'canary':
        raise ValueError('Historical fixture is bounded canary only')
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count() < 1 or (args.execution=='serial' and torch.cuda.device_count()!=1):
        raise ValueError('Execution must stay inside the declared GPU allocation')
    allocation=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(allocation)!=torch.cuda.device_count() or len(set(allocation))!=len(allocation):raise ValueError('Invalid CUDA allocation map')
    output = args.output.resolve()
    if output.is_relative_to(p.ROOT) or (output.exists() and not args.resume):
        raise ValueError('Fresh private output or explicit full-state resume required')
    if args.resume and (not output.exists() or (output/'RUN_COMPLETE').exists()):
        raise ValueError('Resume requires an interrupted, unsealed existing run')
    output.mkdir(parents=True,exist_ok=args.resume)
    from .rmea_token_transport_recovery import run_lock
    with run_lock(output / '.run.lock'):
        preparing=output/'PREPARING.json'
        prep_identity=dict(source=p.identity(p.sources()),config=cfg)
        if args.resume and not (output/'manifest.json').exists():
            import shutil
            if ((preparing.exists() and p.read(preparing)!=prep_identity) or
                    (not preparing.exists() and any(x.name!='.run.lock' for x in output.iterdir())) or (output/'checkpoints').exists()
                    or (output/'.workers').exists() or (output/'summary.json').exists()):
                raise ValueError('Unverified interrupted preparation; preserve it for inspection')
            cache=output/'cache'
            if cache.is_symlink():raise ValueError('Symlinked incomplete cache')
            if cache.exists():shutil.rmtree(cache)
            args.resume=False
        if not args.resume:p.atomic_json(preparing,prep_identity)
        if args.resume:
            previous_manifest=p.read(output/'manifest.json')
            fixture_sha=previous_manifest['environment'].get('fixture_sha256')
            if args.fixture_cache and fixture_identity(args.fixture_cache)!=fixture_sha: raise ValueError('Fixture resume pin differs')
            labels = sorted(p.read(p.ROOT/'docs/contracts/q_rmea_entity_labels.json'))
            recorded=output.parent/'recovery-inputs/labels.json'
            if recorded.exists() and p.read(recorded)!=labels:raise ValueError('Resume label identity differs')
        elif args.fixture_cache:
            fixture_sha = load_fixture(args.fixture_cache, output)
            labels = sorted(p.read(p.ROOT/'docs/contracts/q_rmea_entity_labels.json'))
        else:
            from .rmea_run import extract
            p.verify_loader_directory(args.assets)
            labels, _ = extract(args, cfg, output)
        import importlib.metadata as md
        environment = dict(python=sys.version.split()[0], torch=torch.__version__, numpy=md.version('numpy'),
            gpu=torch.cuda.get_device_name(0), fixture_sha256=fixture_sha if args.fixture_cache or args.resume and fixture_sha else None,
            transformers=md.version('transformers') if not (args.fixture_cache or args.resume and fixture_sha) else 'not used; pinned cached canary')
        manifest = dict(schema='q-rmea.token-transport.run.v1', execution=args.execution, config=cfg, source_sha256=p.identity(p.sources()),
            git_revision=p.git('rev-parse', 'HEAD'), environment=environment,
            data={s: dict(source_sha256=p.EXPECTED[s][1], source_count=p.EXPECTED[s][0], selected_count=n) for s, n in cfg['counts'].items()})
        if args.resume:
            previous=p.read(output/'manifest.json')
            if any(previous[k]!=manifest[k] for k in ('config','source_sha256','git_revision','data','execution')):
                raise ValueError('Resume run provenance differs')
            if any(previous['environment'][k]!=environment[k] for k in ('python','torch','numpy','transformers')):
                raise ValueError('Resume software differs')
            p.atomic_json(output/'resume-environment.json',environment)
        else:
            from .rmea_token_transport_recovery import cache_inventory
            manifest['cache']=cache_inventory(output/'cache')
            p.atomic_json(output/'manifest.json',manifest)
            preparing.unlink(missing_ok=True)
        (output.parent/'recovery-inputs').mkdir(exist_ok=True)
        p.atomic_json(output.parent/'recovery-inputs/labels.json',labels)
        from .rmea_token_transport_recovery import Recovery
        if args.execution=='auto':
            from .rmea_token_transport_parallel import parallel_train
            parallel_train(cfg,output,labels,os.environ['CUDA_VISIBLE_DEVICES'].split(','),
                data_dir=None if environment['fixture_sha256'] else args.data_dir,assets=None if environment['fixture_sha256'] else args.assets,
                cos=args.cos and args.profile=='discovery',resume=args.resume,pause_after=args.pause_after,restore_pin=args.restore_pin)
        else:
            recovery=Recovery(output,cfg,enabled=args.cos and args.profile=='discovery',resume=args.resume,
                pause_after=args.pause_after)
            train(cfg, output, labels, data_dir=None if environment['fixture_sha256'] else args.data_dir,
                  assets=None if environment['fixture_sha256'] else args.assets,recovery=recovery)
        (output/'RUN_PAUSED').unlink(missing_ok=True)
        if not environment['fixture_sha256']:
            for split in cfg['counts']:
                if p.digest(args.data_dir/(split+'.jsonl')) != p.EXPECTED[split][1]:
                    raise ValueError('Data changed during run')
        p.seal(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('canary','discovery'),required=True)
    parser.add_argument('--data-dir',type=Path,required=True); parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--fixture-cache',type=Path)
    parser.add_argument('--execution',choices=('auto','serial'),default='auto')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--cos',action='store_true')
    parser.add_argument('--restore-pin',action='store_true')
    parser.add_argument('--parent-pid',type=int)
    parser.add_argument('--pause-after',type=int,default=0,help='Bounded canary only')
    args=parser.parse_args()
    if args.pause_after and args.profile!='canary': parser.error('Fault injection is canary-only')
    if args.restore_pin and not (args.resume and args.execution=='auto'): parser.error('Pinned restore requires parallel resume')
    if args.parent_pid:
        import ctypes,signal
        if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGKILL,0,0,0)!=0:raise OSError('Parent-death fencing failed')
        if os.getppid()!=args.parent_pid:os.kill(os.getpid(),signal.SIGKILL)
    run(args)


if __name__=='__main__':
    main()
