"""Same-forward, representation-explicit short training-credit fork."""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.retacred import rmea_token_transport_protocol as p
from experiments import q_rmea_query_alignment as qa

ARMS = ('full', 'detach_feature', 'detach_token')
STEPS = 256
START = 46816
CHECKPOINT_SHA = 'd2620148df8c1879b6bf0679ef4386ee265991c1b1d15825b1b9290ad7bc3832'
EVIDENCE = 'docs/evidence/rmea/20261121-backward-credit'


def design():
    return dict(schema='q-rmea.backward-credit.design.v1', arms=list(ARMS),
        input_checkpoint_sha256=CHECKPOINT_SHA, start=dict(mode=3, epoch=0, next_start=START),
        steps=STEPS, seed=13, batch_size=32, train_count=qa.TRAIN, valid_count=qa.N,
        optimizer=p.config('discovery'),
        forward='All training arms have identical aligned linear-coherent forward at matched weights; tau=.25, exact original spectral scale',
        gradients='full: differentiate 8D rotation and scale; detach_feature: freeze whole 8D rotation; detach_token: freeze whole original token unitary',
        normalization='All arms explicitly normalize the masked rotated Born numerator. Full matches original native-denominator values AND gradients; detach_feature uses this normalized graph, not the historical native-denominator detach graph.',
        static='One arithmetic mean of C_i/c_i from ALL training inputs at the immutable fork weights, frozen for all evaluations; no labels in construction',
        primary='After exactly256 updates, full static-rotation validation Macro exceeds both detached arms with both paired95% bootstrap lower bounds>0 and CE no worse. Development diagnostic only.',
        bootstrap=dict(draws=1000, seed=20261121, unit='fixed validation example'),
        probe='First2 examples in the checkpoint-owned next minibatch; original/full gradient parity, equal forward and readout gradients, prepare-gradient differences; partial numeric trace',
        exclusions='No test access, checkpoint/epoch selection, new seeds/tau, L5/novelty claim, cloud writes or automatic full-run promotion',
        recovery='Cheap bounded restart from original COS-acknowledged input; atomic local latest per update and initial/final validation-best; no new COS generations',
        interpretation='A null short fork is insufficient to reject a training-dependent family; detach interventions depend on representation and normalization',
        execution='One scheduler allocation of3GPUs; once-only preparation, then one independent worker per arm')


def alpha(psi, phi, mask, arm):
    import torch
    from baselines.retacred.rmea import EPS
    from baselines.retacred.rmea_token_transport import route
    x = psi.conj().masked_fill(~mask[..., None], 0)
    if arm == 'original':
        return route(psi, phi, mask, kind='linear_coherent')['alpha']
    if arm == 'detach_token':
        with torch.no_grad():
            u = route(psi, phi, mask, kind='linear_coherent')['unitary']
        a = torch.einsum('btd,rd->brt', x, phi)
        moved = a @ u.transpose(-2, -1)
    else:
        if arm == 'full':
            r = torch.matrix_exp(-.25j*qa.generator(psi, mask))
        elif arm == 'detach_feature':
            with torch.no_grad():
                r = torch.matrix_exp(-.25j*qa.generator(psi, mask))
        else:
            raise ValueError('Unknown backward intervention')
        moved = torch.einsum('btd,brd->brt', x, torch.einsum('bij,rj->bri', r, phi))
    raw = (moved.abs().square()+EPS)*mask[:, None, :]
    return raw/raw.sum(-1, keepdim=True)


def batch(data, indices, device='cuda'):
    import numpy as np
    import torch
    ids = list(map(int, indices))
    lengths = [int(data['offsets'][i+1]-data['offsets'][i]) for i in ids]
    h = np.zeros((len(ids), max(lengths), 1024), np.float64)
    roles = np.zeros(h.shape[:2]+(4,), np.float64)
    mask = np.zeros(h.shape[:2], bool)
    for j, (i, length) in enumerate(zip(ids, lengths)):
        a, b = map(int, data['offsets'][i:i+2])
        h[j, :length] = data['h'][a:b]
        roles[j, :length] = data['roles'][a:b]
        mask[j, :length] = True
    inputs = tuple(torch.from_numpy(v).to(device) for v in (h, roles, mask))
    targets = torch.tensor([int(data['targets'][i]) for i in ids], device=device)
    return inputs, targets


def orders():
    import numpy as np
    rng = np.random.default_rng(13)
    return [rng.permutation(qa.TRAIN) for _ in range(4)]


def checkpoint(run):
    import torch
    from baselines.retacred.rmea_token_transport_recovery import Recovery
    generation = run.parent/'cos-channel/080-progress'
    path = generation/'latest.pt'
    if path.is_symlink() or p.digest(path) != CHECKPOINT_SHA:
        raise ValueError('Immutable input checkpoint differs')
    request, ack = p.read(generation/'request.json'), p.read(generation/'ack.json')
    spec = request['files']['cos-channel/080-progress/latest.pt']
    if (request['generation'] != '080-progress' or spec != dict(bytes=path.stat().st_size, sha256=CHECKPOINT_SHA)
            or ack['request_sha256'] != p.digest(generation/'request.json') or ack['done'] != request['done']):
        raise ValueError('Historical cloud acknowledgement differs')
    state = torch.load(path, map_location='cpu', weights_only=True)
    cfg = p.config('discovery')
    if (state.get('schema') != 'q-rmea.resume.v1' or state['identity'] != dict(config=p.identity(cfg), source=qa.ORIGINAL)
            or request['identity'] != state['identity'] or request['done'] != state['done']
            or state['total'] != 29248 or state['done'] != 23399
            or state['cursor']['mode'] != 3 or state['cursor']['epoch'] != 0
            or state['cursor']['next_start'] != START or state['cursor']['history']):
        raise ValueError('Original checkpoint-owned cursor differs')
    # Historical completed serial arms have no optimizer. Validate only the
    # retained ACTIVE linear arm using the stricter current single-arm checker.
    validator = Recovery.__new__(Recovery)
    validator.cfg, validator.modes = cfg, ('linear_coherent',)
    validator.total = 4*math.ceil(qa.TRAIN/32)
    scoped = dict(state, cursor=dict(state['cursor'], mode=0), completed={}, best={},
                  total=validator.total, done=state['done']-3*validator.total)
    validator._validate_cursor(scoped)
    if hashlib.sha256(b''.join(o.astype('<i8').tobytes() for o in orders())).hexdigest() != state['order_sha']:
        raise ValueError('Original minibatch order differs')
    return state


def probe(head, inputs, y, directory=None):
    import numpy as np
    import torch
    from torch.nn import functional as F
    names = [n for n, _ in head.named_parameters()]
    grads, logits, distributions = {}, {}, {}
    arrays = {}
    for arm in ('original',)+ARMS:
        d = head(*inputs, details=True)
        a = alpha(d['psi'], d['phi'], inputs[2], arm)
        z = (a*d['utilities']).sum(-1)+head.bias
        gs = torch.autograd.grad(F.cross_entropy(z, y), tuple(head.parameters()))
        if not all(torch.isfinite(g).all() for g in gs):
            raise ValueError('Nonfinite probe gradient')
        grads[arm] = dict(zip(names, [g.detach() for g in gs]))
        logits[arm], distributions[arm] = z.detach(), a.detach()
        arrays[arm+'__logits'] = z.detach().cpu().numpy()
        arrays[arm+'__alpha'] = a.detach().cpu().numpy()
        for n, g in grads[arm].items():
            arrays[arm+'__grad__'+n] = g.cpu().numpy()
    for arm in ARMS:
        if not torch.allclose(distributions[arm], distributions['original'], atol=2e-10, rtol=2e-10):
            raise ValueError('Same-forward token distribution parity failed')
        if not torch.allclose(logits[arm], logits['original'], atol=2e-10, rtol=2e-10):
            raise ValueError('Same-forward graph parity failed')
        if not torch.allclose(distributions[arm].sum(-1), torch.ones_like(distributions[arm].sum(-1)), atol=2e-12, rtol=0):
            raise ValueError('Intervention is not normalized')
        for n in names:
            if arm == 'full' or not n.startswith('prepare.'):
                if not torch.allclose(grads[arm][n], grads['original'][n], atol=2e-9, rtol=1e-7):
                    raise ValueError('Full/readout gradient parity failed: '+arm+'/'+n)
    prep = lambda arm: torch.cat([g.flatten() for n,g in grads[arm].items() if n.startswith('prepare.')])
    v = prep('full')
    row = dict(full_gradient_max_abs=max(float((grads['full'][n]-grads['original'][n]).abs().max()) for n in names),
        forward_max_abs=max(float((logits[a]-logits['original']).abs().max()) for a in ARMS),
        prepare={a:dict(norm=float(prep(a).norm()), difference_from_full=float((prep(a)-v).norm()),
            cosine_to_full=float(torch.nn.functional.cosine_similarity(prep(a),v,dim=0))) for a in ARMS},
        detach_difference=float((prep('detach_feature')-prep('detach_token')).norm()),
        status='partial numeric gradient probe; not sample-trace.v2')
    if directory is not None:
        np.savez_compressed(directory/'gradient-probe.npz', **arrays)
        row['private_probe_sha256'] = p.digest(directory/'gradient-probe.npz')
    return row


def init(args):
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'Parent-death fencing failed')
    if os.getppid() != args.parent_pid:
        raise RuntimeError('Parent identity changed')
    import torch
    from baselines.retacred.rmea import RMEAHead
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.device_count() != 1 or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('One allocated GPU required per child')
    state = checkpoint(args.run)
    head = RMEAHead('quantum').double().to('cuda')
    head.load_state_dict(state['active'])
    return head, state


def event(out, phase, current, total, **fields):
    p.atomic_json(out/'progress.json', dict(schema='research.progress.v1',
        attempt=Path(os.environ['RESEARCH_OUTPUT_DIR']).parent.name,
        phase=phase, stage=phase, unit='work_item', current=current, completed=current,
        total=total, observed_at=time.time(), batch_id='rmea-backward-credit-20261121', **fields))


def evaluate(head, data, rotation, out, phase, offset=0):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from baselines.retacred.rmea import EPS
    predicted, losses = [], []
    with torch.no_grad():
        for start in range(0, qa.N, 32):
            inputs, y = batch(data, range(start,min(start+32,qa.N)))
            d = head(*inputs, details=True)
            moved = torch.einsum('btd,rd->brt', d['psi'].conj().masked_fill(~inputs[2][...,None],0),
                                torch.einsum('ij,rj->ri', rotation, d['phi']))
            raw = (moved.abs().square()+EPS)*inputs[2][:,None,:]
            z = (raw/raw.sum(-1,keepdim=True)*d['utilities']).sum(-1)+head.bias
            if not torch.isfinite(z).all():
                raise ValueError('Nonfinite static evaluation')
            predicted.extend(z.argmax(-1).tolist())
            losses.extend(F.cross_entropy(z,y,reduction='none').tolist())
            if start%256 == 0 or start+32>=qa.N:
                event(out,phase,offset+len(predicted),offset+qa.N)
    pred = np.asarray(predicted,np.int64)
    row = dict(positive39_macro_f1=qa.macro(np.asarray(data['targets'],np.int64),pred),ce=float(np.mean(losses)),count=qa.N)
    return row, pred


def prepare(args):
    import numpy as np
    import torch
    head, state = init(args)
    out = args.output/'prepared'
    out.mkdir()
    inputs, y = batch(load(args.run,'train'),orders()[0][START:START+2])
    p.atomic_json(out/'gradient-probe.json',probe(head,inputs,y,out))
    summed = torch.zeros(8,8,dtype=torch.complex128,device='cuda')
    with torch.no_grad():
        for start, inputs in qa.data_batches(qa.dataset(args.run,'train'),qa.TRAIN):
            d = head(*inputs,details=True)
            for g in qa.generator(d['psi'],inputs[2]):
                summed += g
            if start%256==0 or start+32>=qa.TRAIN:
                event(out,'training_input_mean',min(start+32,qa.TRAIN),qa.TRAIN+qa.N)
        rotation = torch.matrix_exp(-.25j*summed/qa.TRAIN)
    np.save(out/'static-rotation.npy',rotation.cpu().numpy(),allow_pickle=False)
    baseline, pred = evaluate(head,load(args.run,'valid'),rotation,out,'initial_validation',qa.TRAIN)
    np.save(out/'predictions.npy',pred,allow_pickle=False)
    p.atomic_json(out/'initial-summary.json',baseline)
    p.atomic_json(out/'PREPARED',dict(files={n:p.digest(out/n) for n in
        ('static-rotation.npy','predictions.npy','initial-summary.json','gradient-probe.json','gradient-probe.npz')}))


def load(run, split):
    import numpy as np
    d = qa.dataset(run,split)
    d['targets'] = np.load(run/'cache'/(split+'-targets.npy'),mmap_mode='r',allow_pickle=False)
    count = qa.TRAIN if split=='train' else qa.N
    if len(d['targets'])!=count or not np.isin(d['targets'],np.arange(40)).all():
        raise ValueError('Training/validation target coverage differs')
    return d


def worker(args):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from baselines.retacred.rmea_token_transport_recovery import atomic_torch,cpu_tree,restore_rng,rng_state
    from baselines.retacred.rmea_token_transport_study import state_identity
    head, state = init(args)
    out = args.output/args.condition
    out.mkdir()
    prepared = args.output/'prepared'
    for n,sha in p.read(prepared/'PREPARED')['files'].items():
        if p.digest(prepared/n)!=sha:
            raise ValueError('Prepared sealed input differs')
    rotation = torch.from_numpy(np.load(prepared/'static-rotation.npy',allow_pickle=False)).to('cuda')
    cfg = p.config('discovery')
    optimizer = torch.optim.AdamW(head.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'],betas=tuple(cfg['betas']),eps=cfg['optimizer_eps'])
    optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'])
    initial = p.read(prepared/'initial-summary.json')
    atomic_torch(out/'best.pt',dict(state=cpu_tree(head.state_dict()),updates=0,metric=initial['positive39_macro_f1'],selection='initial/final fixed-static validation Macro; first tie'))
    data, order = load(args.run,'train'), orders()[0]
    losses = []
    for step in range(STEPS):
        start = START+32*step
        ids = order[start:start+32]
        if len(ids)!=32:
            raise ValueError('Frozen short budget crossed epoch boundary')
        inputs,y = batch(data,ids)
        optimizer.zero_grad(set_to_none=True)
        d = head(*inputs,details=True)
        a = alpha(d['psi'],d['phi'],inputs[2],args.condition)
        z = (a*d['utilities']).sum(-1)+head.bias
        loss = F.cross_entropy(z,y)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        if not all(v.grad is not None and torch.isfinite(v.grad).all() for v in head.parameters()):
            raise ValueError('Missing/nonfinite training gradient')
        torch.nn.utils.clip_grad_norm_(head.parameters(),cfg['clip_norm'],error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        atomic_torch(out/'latest.pt',dict(schema='q-rmea.backward-credit.local.v1',condition=args.condition,
            input_checkpoint_sha256=CHECKPOINT_SHA,design_sha256=p.identity(design()),
            updates=step+1,epoch=0,next_start=start+32,active=cpu_tree(head.state_dict()),
            optimizer=cpu_tree(optimizer.state_dict()),rng=rng_state()))
        event(out,'short_training',step+1,STEPS+qa.N,condition=args.condition,loss=losses[-1],checkpoint_cursor=step+1)
    row,pred = evaluate(head,load(args.run,'valid'),rotation,out,'static_validation',STEPS)
    if row['positive39_macro_f1']>initial['positive39_macro_f1']:
        atomic_torch(out/'best.pt',dict(state=cpu_tree(head.state_dict()),updates=STEPS,metric=row['positive39_macro_f1'],selection='initial/final fixed-static validation Macro; first tie'))
    np.save(out/'predictions.npy',pred,allow_pickle=False)
    row.update(condition=args.condition,steps=STEPS,initial=initial,first_loss=losses[0],last_loss=losses[-1],
        mean_train_ce=float(np.mean(losses)),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        model_sha256=state_identity(head.state_dict()),latest_sha256=p.digest(out/'latest.pt'),
        prediction_sha256=p.digest(out/'predictions.npy'))
    p.atomic_json(out/'summary.json',row)
    p.atomic_json(out/'WORKER_COMPLETE',dict(summary_sha256=p.digest(out/'summary.json')))


def command(args, action, condition=None):
    argv = [sys.executable,'-B',str(Path(__file__).resolve()),action,'--run',str(args.run),
            '--output',str(args.output),'--parent-pid',str(os.getpid())]
    return argv+(['--condition',condition] if condition else [])


def report(out, phase, states):
    total = qa.TRAIN+qa.N if phase=='prepare' else (STEPS+qa.N)*len(ARMS)
    event(out,phase,sum(v['current'] for v in states),total,jobs=states)


def run(args):
    import numpy as np
    from baselines.retacred.orchestrate import gpu_ids
    devices = gpu_ids('auto')
    if len(devices)!=3 or p.git('status','--porcelain'):
        raise ValueError('Clean exact three-GPU allocation required')
    review = p.read(ROOT/EVIDENCE/'review.json')
    names = ('experiments/q_rmea_backward_credit.py','experiments/check_rmea_backward_credit.py','experiments/q_rmea_query_alignment.py')
    if (review.get('verdict')!='PASS' or review.get('review_scope')!='both' or review.get('must_fix')
            or review.get('hashes')!={n:p.digest(ROOT/n) for n in names}
            or review.get('design_sha256')!=p.identity(design()) or review.get('core_sha256')!=p.identity(p.sources())):
        raise ValueError('Exact scoped mechanism/lifecycle review required')
    args.run,args.output = args.run.resolve(),args.output.resolve()
    if args.output.is_relative_to(ROOT) or args.output.is_relative_to(args.run.parent) or args.run.is_relative_to(args.output):
        raise ValueError('Isolated new attempt output required')
    qa.fresh_output(args.output)
    p.atomic_json(args.output/'design.json',design())
    started = time.time()
    inputs = qa.validate_inputs(args.run)
    train_target = args.run/'cache/train-targets.npy'
    if p.digest(train_target)!=p.read(args.run/'manifest.json')['cache']['train-targets.npy']['sha256']:
        raise ValueError('Original training targets differ')
    state = checkpoint(args.run)
    inputs.update(training_targets_sha256=p.digest(train_target),input_checkpoint_sha256=CHECKPOINT_SHA,
                  original_cursor=state['cursor'],active_scope_AdamW_validated=True,order_sha256=state['order_sha'])
    p.atomic_json(args.output/'inputs.json',inputs)
    qa.execute([('prepare',command(args,'prepare'),devices[0])],args.output,'prepare',report=report)
    qa.execute([(a,command(args,'worker',a),g) for a,g in zip(ARMS,devices)],args.output,'fork',report=report)
    rows,pred = {},{}
    for arm in ARMS:
        out = args.output/arm
        if p.read(out/'WORKER_COMPLETE')['summary_sha256']!=p.digest(out/'summary.json'):
            raise ValueError('Worker seal differs')
        rows[arm] = p.read(out/'summary.json')
        if p.digest(out/'predictions.npy')!=rows[arm]['prediction_sha256'] or p.digest(out/'latest.pt')!=rows[arm]['latest_sha256']:
            raise ValueError('Prediction/model seal differs')
        pred[arm] = np.load(out/'predictions.npy',allow_pickle=False)
    if max(v['first_loss'] for v in rows.values())-min(v['first_loss'] for v in rows.values())>1e-10:
        raise ValueError('Matched first training forward differs')
    gold = np.load(args.run/'cache/valid-targets.npy',allow_pickle=False)
    draws = {a:[] for a in ARMS[1:]}
    rng = np.random.default_rng(20261121)
    for i in range(1000):
        ids = rng.integers(0,qa.N,size=qa.N)
        reference = qa.macro(gold[ids],pred['full'][ids])
        for a in draws:
            draws[a].append(reference-qa.macro(gold[ids],pred[a][ids]))
        if i%100==0:
            event(args.output,'paired_bootstrap',(STEPS+qa.N)*3,(STEPS+qa.N)*3,resamples=i)
    contrasts = {a:dict(delta=rows['full']['positive39_macro_f1']-rows[a]['positive39_macro_f1'],
        paired95_percentile=np.quantile(v,[.025,.975]).tolist()) for a,v in draws.items()}
    gate = all(v['delta']>0 and v['paired95_percentile'][0]>0 and rows['full']['ce']<=rows[a]['ce'] for a,v in contrasts.items())
    result = dict(schema='q-rmea.backward-credit.result.v1',results=rows,contrasts=contrasts,gate=gate,
        probe=p.read(args.output/'prepared/gradient-probe.json'),seconds=time.time()-started,
        source=p.git('rev-parse','HEAD'),script_sha256=p.digest(Path(__file__)),design_sha256=p.identity(design()),
        core_sha256=p.identity(p.sources()),physical_gpus=devices,
        scope='Exploratory short256update train-dependent diagnostic; no independent replication or L5; null does not close family')
    p.atomic_json(args.output/'summary.json',result)
    p.atomic_json(args.output/'RUN_COMPLETE',dict(artifacts={n:p.digest(args.output/n) for n in ('design.json','inputs.json','summary.json')}))
    event(args.output,'complete',(STEPS+qa.N)*3,(STEPS+qa.N)*3)
    print(json.dumps(dict(gate=gate,seconds=result['seconds'])),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','prepare','worker'))
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parent-pid',type=int)
    parser.add_argument('--condition',choices=ARMS)
    args = parser.parse_args()
    if args.action!='run' and not args.parent_pid:
        parser.error('Exact parent required')
    if args.action=='worker' and not args.condition:
        parser.error('Condition required')
    dict(run=run,prepare=prepare,worker=worker)[args.action](args)


if __name__=='__main__':
    main()
