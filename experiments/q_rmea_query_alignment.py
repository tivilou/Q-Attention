"""Fixed linear checkpoint: aligned, deranged and training-mean query rotations."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.retacred import rmea_token_transport_protocol as p
from experiments.q_rmea_mass_direction import macro

ARMS = ('aligned', 'deranged', 'training_mean')
N = 19584
TRAIN = 58465
ORIGINAL = 'd9837319bb368e6b43939596802a40e05d1f276cbe5e26c27f34947ceb6f3ee7'


def design():
    return dict(schema='q-rmea.query-alignment.design.v1', checkpoint='linear_coherent-final',
        counts=dict(train_inputs=TRAIN, valid=N), arms=list(ARMS), tau=.25,
        static='Arithmetic mean over all training sentences of C_i/c_i; no training targets read',
        permutation='One deterministic label-blind hash-order cycle, no fixed point; valid contexts reused bijectively',
        primary='Aligned positive39Macro exceeds both controls with both paired95% bootstrap lower bounds>0; CE no worse than either control',
        bootstrap=dict(draws=1000, seed=20261121, unit='validation example, fixed cohort'),
        exclusions='No fitting, test access, seed/tau tuning, cloud writes or L5/novelty claim',
        interpretation='Fixed-checkpoint context-to-query-rotation alignment only; not total training-gain mediation',
        trace='First2 immutable valid cache units per arm; partial numeric probe, not sample-trace.v2',
        execution='Prepare once then three independent workers within one scheduler GPU-group allocation',
        recovery='Restart bounded inference from original pinned inputs; no new checkpoint generation')


def derangement(n):
    if n < 2:
        raise ValueError('Need at least two contexts')
    order = sorted(range(n), key=lambda i: hashlib.sha256(('query-alignment-v1:'+str(i)).encode()).digest())
    result = [0]*n
    for j, target in enumerate(order):
        result[target] = order[(j+1) % n]
    return result


def generator(psi, mask):
    import torch
    x = psi.conj().masked_fill(~mask[..., None], 0)
    k = x @ x.mH
    graph = k - torch.diag_embed(k.diagonal(dim1=-2, dim2=-1))
    scale = torch.linalg.eigvalsh(graph).abs().amax(-1)
    if ((mask.sum(-1) > 1) & (scale == 0)).any():
        raise ValueError('Undefined original graph scale')
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    # The omitted -I/c is a sentencewise global phase, invisible to Born scores.
    return (x.mH @ x) / safe[:, None, None]


def distribution(psi, phi, mask, g):
    import torch
    from baselines.retacred.rmea import EPS
    r = torch.matrix_exp(-.25j*g)
    x = psi.conj().masked_fill(~mask[..., None], 0)
    a = torch.einsum('btd,brd->brt', x, torch.einsum('bij,rj->bri', r, phi))
    raw = (a.abs().square()+EPS)*mask[:, None, :]
    return raw/raw.sum(-1, keepdim=True), r


def data_batches(data, count):
    import numpy as np
    import torch
    for start in range(0, count, 32):
        ids = list(range(start, min(start+32, count)))
        lengths = [int(data['offsets'][i+1]-data['offsets'][i]) for i in ids]
        h = np.zeros((len(ids), max(lengths), 1024), np.float64)
        roles = np.zeros((len(ids), max(lengths), 4), np.float64)
        mask = np.zeros(h.shape[:2], bool)
        for j, (i, length) in enumerate(zip(ids, lengths)):
            a, b = map(int, data['offsets'][i:i+2])
            h[j, :length] = data['h'][a:b]
            roles[j, :length] = data['roles'][a:b]
            mask[j, :length] = True
        yield start, tuple(torch.from_numpy(x).to('cuda') for x in (h, roles, mask))


def dataset(run, split):
    import numpy as np
    data = {k: np.load(run/'cache'/(split+'-'+k+'.npy'), mmap_mode='r', allow_pickle=False)
            for k in ('h', 'roles', 'offsets')}
    count = TRAIN if split == 'train' else N
    if len(data['offsets']) != count+1 or data['offsets'][-1] != len(data['h']):
        raise ValueError('Cache coverage differs')
    return data


def event(out, phase, current, total, **fields):
    p.atomic_json(out/'progress.json', dict(schema='research.progress.v1',
        attempt=Path(os.environ['RESEARCH_OUTPUT_DIR']).parent.name,
        phase=phase, stage=phase, unit='sample', current=current, completed=current,
        total=total, observed_at=time.time(), batch_id='rmea-query-alignment-valid-20261121', **fields))


def validate_inputs(run):
    manifest, summary, seal = (p.read(run/name) for name in ('manifest.json', 'summary.json', 'RUN_COMPLETE'))
    if manifest['source_sha256'] != ORIGINAL or manifest['config'] != p.config('discovery'):
        raise ValueError('Original run identity differs')
    for name in ('manifest.json', 'summary.json'):
        if seal['artifacts'][name]['sha256'] != p.digest(run/name):
            raise ValueError('Original run seal differs')
    checkpoint = run/'checkpoints/linear_coherent-final.pt'
    if p.digest(checkpoint) != summary['results']['linear_coherent']['checkpoint_sha256']:
        raise ValueError('Frozen linear checkpoint differs')
    names = ['metadata.json', 'valid-targets.npy'] + [s+'-'+k+'.npy'
             for s in ('train', 'valid') for k in ('h', 'roles', 'offsets')]
    for name in names:
        path = run/'cache'/name
        row = manifest['cache'][name]
        if path.is_symlink() or path.stat().st_size != row['bytes'] or p.digest(path) != row['sha256']:
            raise ValueError('Input seal differs: '+name)
    return dict(checkpoint_sha256=p.digest(checkpoint),
        input_hashes={n:manifest['cache'][n] for n in names}, original_source=ORIGINAL)


def child_init(args):
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Parent-death fencing failed')
    if os.getppid() != args.parent_pid:
        raise RuntimeError('Parent identity changed')
    import torch
    from baselines.retacred.rmea import RMEAHead
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.device_count() != 1 or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Exactly one allocated GPU per child')
    head = RMEAHead('quantum').double()
    head.load_state_dict(torch.load(args.run/'checkpoints/linear_coherent-final.pt',
                                  map_location='cpu', weights_only=True))
    return head.to('cuda').eval()


def prepare(args):
    import numpy as np
    import torch
    head = child_init(args)
    prep = args.output/'prepared'
    prep.mkdir()
    summed = torch.zeros(8, 8, dtype=torch.complex128, device='cuda')
    valid = np.empty((N, 8, 8), np.complex128)
    done = 0
    with torch.no_grad():
        for split, count in (('train', TRAIN), ('valid', N)):
            data = dataset(args.run, split)
            for start, inputs in data_batches(data, count):
                d = head(*inputs, details=True)
                g = generator(d['psi'], inputs[2])
                if not torch.isfinite(g).all():
                    raise ValueError('Nonfinite generator')
                if split == 'train':
                    for row in g:
                        summed += row
                else:
                    valid[start:start+len(g)] = g.cpu().numpy()
                done += len(g)
                if start % 256 == 0 or start+32 >= count:
                    event(prep, 'preparation', done, TRAIN+N, split=split)
    np.save(prep/'generators.npy', valid, allow_pickle=False)
    np.save(prep/'training-mean.npy', (summed/TRAIN).cpu().numpy(), allow_pickle=False)
    np.save(prep/'permutation.npy', np.array(derangement(N), np.int64), allow_pickle=False)
    p.atomic_json(prep/'PREPARED', dict(files={f:p.digest(prep/f)
        for f in ('generators.npy', 'training-mean.npy', 'permutation.npy')}))


def worker(args):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from baselines.retacred.rmea_token_transport import route
    head = child_init(args)
    out = args.output/args.condition
    out.mkdir()
    prep = args.output/'prepared'
    for name, sha in p.read(prep/'PREPARED')['files'].items():
        if p.digest(prep/name) != sha:
            raise ValueError('Prepared input changed')
    generators = np.load(prep/'generators.npy', mmap_mode='r', allow_pickle=False)
    mean = np.load(prep/'training-mean.npy', allow_pickle=False)
    permutation = np.load(prep/'permutation.npy', allow_pickle=False)
    if sorted(permutation.tolist()) != list(range(N)) or np.any(permutation == np.arange(N)):
        raise ValueError('Expected bijective no-self control')
    data = dataset(args.run, 'valid')
    # Targets first enter after all label-blind controls are sealed.
    target = np.load(args.run/'cache/valid-targets.npy', mmap_mode='r', allow_pickle=False)
    if len(target) != N:
        raise ValueError('Target coverage differs')
    pred, losses = [], []
    with torch.no_grad():
        for start, inputs in data_batches(data, N):
            stop = min(N, start+32)
            d = head(*inputs, details=True)
            if args.condition == 'aligned':
                g = np.asarray(generators[start:stop]).copy()
            elif args.condition == 'deranged':
                g = np.asarray(generators[permutation[start:stop]]).copy()
            else:
                g = np.broadcast_to(mean, (stop-start, 8, 8)).copy()
            g = torch.from_numpy(g).to('cuda')
            alpha, rotation = distribution(d['psi'], d['phi'], inputs[2], g)
            logits = (alpha*d['utilities']).sum(-1)+head.bias
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite scores')
            if start == 0:
                original = route(d['psi'], d['phi'], inputs[2], kind='linear_coherent')['alpha']
                identity, _ = distribution(d['psi'], d['phi'], inputs[2], generator(d['psi'], inputs[2]))
                if not torch.allclose(identity, original, atol=2e-12, rtol=2e-12):
                    raise ValueError('Eight-dimensional identity differs from original route')
                arrays = dict(psi=d['psi'][:2].cpu().numpy(), phi=d['phi'].cpu().numpy(),
                    mask=inputs[2][:2].cpu().numpy(), generator=g[:2].cpu().numpy(),
                    rotation=rotation[:2].cpu().numpy(), alpha=alpha[:2].cpu().numpy(),
                    utilities=d['utilities'][:2].cpu().numpy(), bias=head.bias.cpu().numpy(),
                    logits=logits[:2].cpu().numpy())
                np.savez_compressed(out/'first2.npz', **arrays)
                p.atomic_json(out/'trace-metadata.json', dict(path='first2.npz', sha256=p.digest(out/'first2.npz'),
                    status='partial numeric probe, not full sample-trace.v2',
                    arrays={k:dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in arrays.items()}))
            pred.extend(logits.argmax(-1).tolist())
            y = torch.tensor(np.asarray(target[start:stop], np.int64), device='cuda')
            losses.extend(F.cross_entropy(logits, y, reduction='none').tolist())
            if start % 256 == 0 or stop == N:
                event(out, 'frozen_validation', stop, N, condition=args.condition)
    np.save(out/'predictions.npy', np.array(pred, np.int64), allow_pickle=False)
    row = dict(positive39_macro_f1=macro(np.asarray(target, np.int64), np.array(pred, np.int64)),
        ce=float(np.mean(losses)), count=N, physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        prediction_sha256=p.digest(out/'predictions.npy'), condition=args.condition)
    if args.condition == 'aligned':
        old = p.read(args.run/'summary.json')['results']['linear_coherent']['valid']
        if any(abs(row[k]-old[k]) > 1e-10 for k in ('positive39_macro_f1', 'ce')):
            raise ValueError('Original complete validation does not replay')
    p.atomic_json(out/'summary.json', row)
    p.atomic_json(out/'WORKER_COMPLETE', dict(summary_sha256=p.digest(out/'summary.json')))


def command(args, action, condition=None):
    argv = [sys.executable, '-B', str(Path(__file__).resolve()), action,
            '--run', str(args.run), '--output', str(args.output), '--parent-pid', str(os.getpid())]
    if condition:
        argv += ['--condition', condition]
    return argv


def execute(jobs, out, phase):
    """Own only these no-grandchild processes; the scheduler owns their process group."""
    active = []
    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    saved = {s:signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    for s in saved:
        signal.signal(s, stop)
    try:
        for name, argv, gpu in jobs:
            if stopping:
                raise InterruptedError('Batch interrupted before child launch')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            log = (out/(name+'.log')).open('wb')
            try:
                proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            except BaseException:
                log.close()
                raise
            active.append((name, proc, log, gpu))
            if stopping:
                raise InterruptedError('Batch interrupted during child launch')
        p.atomic_json(out/(phase+'-commands.json'), dict(commands=[
            dict(name=n, argv=a, cwd=str(ROOT), physical_gpu=g, pid=pr.pid)
            for (n,a,g), (_,pr,_,_) in zip(jobs, active)]))
        while any(proc.poll() is None for _,proc,_,_ in active):
            if stopping:
                raise InterruptedError('Batch interrupted')
            if any(proc.poll() not in (None, 0) for _,proc,_,_ in active):
                raise RuntimeError('Child failed; inspect bounded logs')
            states = []
            for name, proc, _, gpu in active:
                path = out/('prepared' if phase == 'prepare' else name)/'progress.json'
                row = p.read(path) if path.exists() else dict(current=0)
                states.append(dict(condition=name, physical_gpu=gpu, pid=proc.pid,
                    current=row.get('current',0), total=row.get('total'), status='running' if proc.poll() is None else 'complete'))
            total = TRAIN+N if phase == 'prepare' else N*len(jobs)
            current = sum(x['current'] for x in states)
            event(out, phase, current, total, jobs=states)
            time.sleep(1)
        if stopping:
            raise InterruptedError('Batch interrupted')
        if any(proc.returncode != 0 for _,proc,_,_ in active):
            raise RuntimeError('Child exited nonzero')
    finally:
        for _, proc, _, _ in active:
            if proc.poll() is None:
                proc.terminate()
        deadline = time.monotonic()+15
        for _, proc, log, _ in active:
            try:
                proc.wait(timeout=max(.01, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log.close()
        for s, handler in saved.items():
            signal.signal(s, handler)
    if stopping:
        raise InterruptedError('Batch interrupted during cleanup')


def run(args):
    import numpy as np
    from baselines.retacred.orchestrate import gpu_ids
    devices = gpu_ids('auto')
    if len(devices) != 3:
        raise ValueError('This frozen recipe requires exactly three reserved GPUs')
    if p.git('status', '--porcelain'):
        raise ValueError('Clean source required')
    review = p.read(ROOT/'docs/evidence/rmea/20261121-query-alignment/review.json')
    if (review.get('verdict') != 'PASS' or review.get('must_fix') or review.get('review_scope') != 'both'
            or review.get('script_sha256') != p.digest(Path(__file__))
            or review.get('design_sha256') != p.identity(design())
            or review.get('core_sha256') != p.identity(p.sources())):
        raise ValueError('Exact mechanism/lifecycle review required')
    args.run = args.run.resolve()
    args.output = args.output.resolve()
    if (args.output.is_relative_to(ROOT) or args.output.is_relative_to(args.run)
            or args.run.is_relative_to(args.output)):
        raise ValueError('Independent diagnostic output required')
    if list(args.output.iterdir()):
        raise ValueError('Preserve existing attempt output')
    p.atomic_json(args.output/'design.json', design())
    started = time.time()
    inputs = validate_inputs(args.run)
    p.atomic_json(args.output/'inputs.json', inputs)
    execute([('prepare', command(args,'prepare'), devices[0])], args.output, 'prepare')
    execute([(arm, command(args,'worker',arm),gpu) for arm,gpu in zip(ARMS,devices)], args.output, 'validation')
    rows, pred, traces = {}, {}, {}
    for arm in ARMS:
        directory = args.output/arm
        if p.read(directory/'WORKER_COMPLETE')['summary_sha256'] != p.digest(directory/'summary.json'):
            raise ValueError('Worker summary seal differs')
        rows[arm] = p.read(directory/'summary.json')
        if p.digest(directory/'predictions.npy') != rows[arm]['prediction_sha256']:
            raise ValueError('Prediction seal differs')
        pred[arm] = np.load(directory/'predictions.npy', allow_pickle=False)
        traces[arm] = p.read(directory/'trace-metadata.json')
    gold = np.load(args.run/'cache/valid-targets.npy', allow_pickle=False)
    draws = {a:[] for a in ARMS[1:]}
    rng = np.random.default_rng(20261121)
    for i in range(1000):
        ids = rng.integers(0,N,size=N)
        aligned = macro(gold[ids],pred['aligned'][ids])
        for arm in draws:
            draws[arm].append(aligned-macro(gold[ids],pred[arm][ids]))
        if i % 100 == 0:
            event(args.output,'paired_bootstrap',N*3,N*3,resamples=i)
    contrasts = {a:dict(delta=rows['aligned']['positive39_macro_f1']-rows[a]['positive39_macro_f1'],
        paired95_percentile=np.quantile(v,[.025,.975]).tolist()) for a,v in draws.items()}
    gate = all(v['delta']>0 and v['paired95_percentile'][0]>0 for v in contrasts.values())
    gate = gate and all(rows['aligned']['ce']<=rows[a]['ce'] for a in ARMS[1:])
    result = dict(schema='q-rmea.query-alignment.result.v1', results=rows, contrasts=contrasts,
        gate=gate, source=p.git('rev-parse','HEAD'), script_sha256=p.digest(Path(__file__)),
        design_sha256=p.identity(design()), input_identity=inputs, seconds=time.time()-started,
        scope='Fixed validation/checkpoint diagnostic, not independent replication, new method or L5',
        physical_gpus=devices, source_core_sha256=p.identity(p.sources()))
    p.atomic_json(args.output/'trace-metadata.json', traces)
    p.atomic_json(args.output/'summary.json', result)
    p.atomic_json(args.output/'RUN_COMPLETE', dict(artifacts={n:p.digest(args.output/n)
        for n in ('design.json','summary.json','trace-metadata.json','inputs.json')}))
    event(args.output,'complete',N*3,N*3)
    print(json.dumps(dict(gate=gate,seconds=result['seconds'])),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','prepare','worker'))
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parent-pid',type=int)
    parser.add_argument('--condition',choices=ARMS)
    args = parser.parse_args()
    if args.action != 'run' and not args.parent_pid:
        parser.error('Parent identity required')
    if args.action == 'worker' and not args.condition:
        parser.error('Condition required')
    dict(run=run,prepare=prepare,worker=worker)[args.action](args)


if __name__ == '__main__':
    main()
