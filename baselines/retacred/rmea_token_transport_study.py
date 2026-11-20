"""Four matched final-state heads; full-data execution is collaborator-only."""
import argparse
import gc
import hashlib
import os
from pathlib import Path
import sys
from . import rmea_token_transport_protocol as p


def state_identity(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        a = value.detach().cpu().numpy()
        h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def train(cfg, output, labels):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from .common import metrics
    from .rmea import RMEAHead
    from .rmea_token_transport import TrainableTokenTransportHead, route
    from .rmea_token_transport_diagnostics import inspect_unit, aggregate
    from .rmea_token_transport_trace import capture, close
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
        h = np.zeros((len(indices), max(lengths), 1024), np.float64)
        r = np.zeros((len(indices), max(lengths), 4), np.float64); mask = np.zeros(h.shape[:2], bool)
        for j, (i, length) in enumerate(zip(indices, lengths)):
            a, b = map(int, d['offsets'][i:i+2]); h[j, :length] = d['h'][a:b]
            r[j, :length] = d['roles'][a:b]; mask[j, :length] = True
        inputs = tuple(torch.from_numpy(a).to('cuda') for a in (h, r, mask))
        y = torch.from_numpy(np.array(d['targets'][indices], dtype=np.int64)).to('cuda')
        return inputs, y

    @torch.no_grad()
    def evaluate(head, split, routing_kind=None):
        head.eval(); gold = []; predicted = []; loss = 0.
        for start in range(0, cfg['counts'][split], 32):
            inputs, y = batch(split, list(range(start, min(start+32, cfg['counts'][split]))))
            if routing_kind is None:
                logits = head(*inputs)[0]
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

    torch.manual_seed(cfg['seed'])
    initial = {k: v.clone() for k, v in RMEAHead('quantum').double().state_dict().items()}
    initial_sha = state_identity(initial)
    rng = np.random.default_rng(cfg['seed'])
    orders = [rng.permutation(cfg['counts']['train']) for _ in range(cfg['epochs'])]
    order_sha = hashlib.sha256(b''.join(o.astype('<i8').tobytes() for o in orders)).hexdigest()
    checkpoints = output / 'checkpoints'; checkpoints.mkdir()
    results = {}; samples = []; diagnostics = None
    for kind in p.MODES:
        head = TrainableTokenTransportHead(kind).double(); head.load_state_dict(initial)
        if state_identity(head.state_dict()) != initial_sha or sum(v.numel() for v in head.parameters()) != 81028:
            raise ValueError('Initialization or parameter matching differs')
        head.to('cuda'); optimizer = torch.optim.AdamW(head.parameters(), lr=cfg['lr'],
            weight_decay=cfg['weight_decay'], betas=tuple(cfg['betas']), eps=cfg['optimizer_eps'])
        history = []; steps = 0
        for epoch, order in enumerate(orders, 1):
            head.train(); loss_sum = 0.
            for start in range(0, len(order), 32):
                ids = order[start:start+32]; inputs, y = batch('train', ids)
                optimizer.zero_grad(set_to_none=True); loss = F.cross_entropy(head(*inputs)[0], y)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                if not all(v.grad is not None and torch.isfinite(v.grad).all() for v in head.parameters()):
                    raise ValueError('Missing or nonfinite gradient')
                torch.nn.utils.clip_grad_norm_(head.parameters(), cfg['clip_norm'], error_if_nonfinite=True)
                optimizer.step(); steps += 1; loss_sum += float(loss.detach()) * len(ids)
            if not all(torch.isfinite(v).all() for v in head.parameters()):
                raise ValueError('Nonfinite final parameters')
            history.append(dict(epoch=epoch, train_ce=loss_sum / len(order)))
            print(p.json.dumps(dict(stage='train', condition=kind, epoch=epoch, updates=steps)), flush=True)
        head.eval(); final_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
        cp = checkpoints / (kind + '-final.pt'); temporary = cp.with_suffix('.pending')
        with temporary.open('xb') as f:
            torch.save(final_state, f); f.flush(); os.fsync(f.fileno())
        if temporary.stat().st_size > 1024**2:
            raise ValueError('Tiny final checkpoint bound exceeded')
        temporary.rename(cp)
        loaded = torch.load(cp, map_location='cpu', weights_only=True)
        if state_identity(loaded) != state_identity(final_state):
            raise ValueError('Final checkpoint serialization differs')
        head.load_state_dict(loaded)
        result = dict(selected_epoch=cfg['epochs'], selection=cfg['selection'], parameters=81028,
            updates=steps, history=history, initial_state_sha256=initial_sha,
            final_state_sha256=state_identity(loaded), checkpoint_sha256=p.digest(cp),
            valid=evaluate(head, 'valid'), test=evaluate(head, 'test'))
        results[kind] = result
        for split in cfg['counts']:
            samples += capture(output, head, kind, split, batch, metadata)
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
    trace = close(output, samples, cfg, metadata)
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
    if args.profile=='discovery' and (os.environ.get('QATTN_COLLABORATOR_FORMAL') != '1'
                                    or p.git('branch', '--show-current') != '1.1'):
        raise ValueError('Full formal run is collaborator-only via parent')
    if args.fixture_cache and args.profile != 'canary':
        raise ValueError('Historical fixture is bounded canary only')
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count() != 1:
        raise ValueError('Exactly one assigned GPU required')
    output = args.output.resolve()
    if output.is_relative_to(p.ROOT) or output.exists():
        raise ValueError('Fresh private output outside checkout required; no automatic training restart')
    output.mkdir(parents=True)
    with p.lock(output / '.run.lock'):
        if args.fixture_cache:
            from .rmea_entity_study import fixture, FIXTURE_SHA
            fixture(args.fixture_cache, output)
            labels = sorted(p.read(p.ROOT/'docs/contracts/q_rmea_entity_labels.json'))
        else:
            from .rmea_run import extract
            labels, _ = extract(args, cfg, output)
        import importlib.metadata as md
        environment = dict(python=sys.version.split()[0], torch=torch.__version__, numpy=md.version('numpy'),
            gpu=torch.cuda.get_device_name(0), fixture_sha256=FIXTURE_SHA if args.fixture_cache else None,
            transformers=md.version('transformers') if not args.fixture_cache else 'not used; pinned cached canary')
        manifest = dict(schema='q-rmea.token-transport.run.v1', config=cfg, source_sha256=p.identity(p.sources()),
            git_revision=p.git('rev-parse', 'HEAD'), environment=environment,
            data={s: dict(source_sha256=p.EXPECTED[s][1], source_count=p.EXPECTED[s][0], selected_count=n) for s, n in cfg['counts'].items()})
        p.atomic_json(output / 'manifest.json', manifest); train(cfg, output, labels)
        if not args.fixture_cache:
            for split in cfg['counts']:
                if p.digest(args.data_dir/(split+'.jsonl')) != p.EXPECTED[split][1]:
                    raise ValueError('Data changed during run')
        p.seal(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('canary','discovery'),required=True)
    parser.add_argument('--data-dir',type=Path,required=True); parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--fixture-cache',type=Path)
    run(parser.parse_args())


if __name__=='__main__':
    main()
