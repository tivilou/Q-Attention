"""Bounded readiness check, not a task-utility experiment or formal training.

One three-GPU allocation runs three independent, matched input checks. No
optimizer step, validation/test access, output tensor dump or checkpoint save.
"""
import argparse
import ctypes
import hashlib
import io
import json
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

ARMS = ('complex', 'real_magnitude', 'linear_reference')
COUNT = 32
INPUT_RUN = '/hy-tmp/Q-Attention/worker/attempts/6a2bf35b8d9d4ac4a6b4fe2acb6bc116/output/20261121T034306Z_seed13/discovery'
CHECKPOINT = '23691c168f8900001f94a2eba44aa4dde217b703fa4b06a4af228293dc4ab6c8'
MANIFEST = 'dfb347aef35657c5463486e1043aca0617ccc11d3aca1eea851a59af15ec5f49'
REVIEW = 'docs/evidence/rmea/20261122-entity-scattering/implementation-review.json'
SOURCES = ('baselines/retacred/rmea_entity_scattering.py', 'experiments/q_rmea_entity_scattering.py',
           'experiments/q_rmea_query_alignment.py', 'baselines/retacred/rmea_token_transport.py')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def design():
    return dict(schema='q-rmea.entity-scattering.readiness.v1',
        constants=dict(hopping=1., energy=0., damping=.1, terminal_lead=1., epsilon=1e-12, tau=.25),
        onsite='tanh(original relation-token utility), with all upstream gradients retained',
        anchors='First valid word with role[2]==1 and role[3]==1 respectively; nonempty disjoint binary entity flags required',
        arms=list(ARMS), control='Real-magnitude final hopping |G| with identical q/e/anchors/constants; upstream q still phase-sensitive',
        score='log(gamma^2 |R_subject,object|^2+epsilon)+original bias; not attention-weighted utility pooling',
        inputs='Protected manifest '+MANIFEST+' and linear_coherent-final checkpoint; train units0:32 only, microbatch1',
        computation='One forward/backward per training unit, clear gradients each unit, no optimizer/update/fitting',
        gates='Independent NumPy/direct-vs-8D amplitude; gradient parity; fixed-input signed triangle law; real reciprocity; gauge; padding/anchors; finite full-head gradients',
        novelty='Known Green/chiral/Woodbury tools; theoretical PASS is not task utility or L5',
        exclusion='No validation/test data or targets, model selection, hyperparameter/grid/rescue, formal training, new model/cache/COS',
        recovery='Cheap restart from pinned Git/checkpoint/cache, no automatic retry, preserve failed outputs',
        interpretation='Implementation readiness only. Reusing trained linear weights does not test coadapted scattering task utility')


def hashes():
    return {n:p.digest(ROOT/n) for n in SOURCES}


def reviewed():
    r = p.read(ROOT/REVIEW)
    require(r.get('verdict') == 'PASS' and r.get('review_scope') == 'both' and r.get('must_fix') == [], 'Exact both-scope review required')
    require(r.get('hashes') == hashes() and r.get('design_sha256') == p.identity(design())
            and r.get('core_sha256') == p.identity(p.sources()), 'Reviewed revision or contract differs')


def event(out, phase, current, total, **kw):
    p.atomic_json(out/'progress.json', dict(schema='research.progress.v1',
        attempt=Path(os.environ['RESEARCH_OUTPUT_DIR']).parent.name,
        phase=phase, stage=phase, unit='sample', current=current, completed=current,
        total=total, observed_at=time.time(), batch_id='rmea-entity-scattering-readiness-20261122', **kw))


def unit(out):
    import numpy as np
    import torch
    from baselines.retacred.rmea import circuit
    from baselines.retacred.rmea_token_transport import route
    from baselines.retacred.rmea_entity_scattering import scattering, anchors, ETA, GAMMA, READOUT_EPS
    require(not os.environ.get('CUDA_VISIBLE_DEVICES'), 'CPU-only recipe required')
    torch.set_num_threads(1)
    torch.manual_seed(20261122)
    qa.fresh_output(out)
    raw = torch.randn(2, 5, 8, 2, dtype=torch.float64, requires_grad=True)
    logits_q = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    raw_e = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    psi = torch.view_as_complex(raw)
    psi = psi/psi.norm(dim=-1, keepdim=True)
    mask = torch.tensor([[1,1,1,1,1],[1,1,1,0,0]], dtype=torch.bool)
    q = logits_q.masked_fill(~mask[:,None,:], -torch.inf).softmax(-1)
    e = raw_e.tanh()
    s, o = torch.tensor([0,0]), torch.tensor([2,2])
    a = scattering(psi, q, e, mask, s, o)
    b = scattering(psi, q, e, mask, s, o, dense=True)
    err = float((a['amplitude']-b['amplitude']).abs().max().detach())
    require(err < 1e-11, 'Dense/8D amplitude differs')
    ga = torch.autograd.grad(a['logits'].sum(), (raw, logits_q, raw_e), retain_graph=True)
    gb = torch.autograd.grad(b['logits'].sum(), (raw, logits_q, raw_e))
    graderr = max(float((x-y).abs().max()) for x,y in zip(ga, gb))
    require(all(torch.isfinite(x).all() for x in ga+gb) and graderr < 1e-8, 'Dense/8D full gradient differs')
    pn, qn, en = (x.detach().numpy() for x in (psi, q, e))
    nerr = 0.
    for i in range(2):
        for r in range(3):
            keep = mask[i].numpy()
            v = pn[i][keep]
            w = np.sqrt(qn[i,r,keep])
            h = w[:,None]*(v.conj()@v.T-np.eye(len(v)))*w[None,:]+np.diag(en[i,r,keep])
            damp = np.full(len(v), ETA)
            damp[0] += GAMMA/2
            damp[2] += GAMMA/2
            inverse = np.linalg.inv(-h+1j*np.diag(damp))
            nerr = max(nerr, abs(inverse[0,2]-complex(a['amplitude'][i,r].detach())))
    require(nerr < 1e-11, 'Independent NumPy inverse differs')
    angle = torch.zeros(1,3,2,3,2, dtype=torch.float64)
    angle[0,1,1,0,0] = torch.pi/2
    angle[0,2,1,0,:] = torch.pi/2
    psi = circuit(angle)
    phi = circuit(torch.zeros(40,2,3,2, dtype=torch.float64))
    mask = torch.ones(1,3, dtype=torch.bool)
    q = route(psi, phi, mask, kind='linear_coherent')['alpha'][:,:1]
    e = torch.zeros_like(q, requires_grad=True)
    s, o = torch.tensor([0]), torch.tensor([1])
    forward = scattering(psi, q, e, mask, s, o)
    reverse = scattering(psi, q, e, mask, o, s)
    gram = psi.conj()@psi.transpose(-2,-1)
    root = q[0,0].sqrt()
    h = root[:,None]*(gram[0]-torch.eye(3))*root[None,:]
    damp = torch.tensor([ETA+GAMMA/2, ETA+GAMMA/2, ETA], dtype=torch.float64)
    inverse = torch.linalg.inv(-h+1j*torch.diag(damp))
    determinant = torch.linalg.det(-h+1j*torch.diag(damp))
    predicted = -4*GAMMA**2*ETA*(h[0,1]*h[1,2]*h[2,0]).imag/determinant.abs().square()
    delta = (forward['transmission']-reverse['transmission']).squeeze()
    require(float(delta) < 0 and abs(float(delta-predicted)) < 1e-11, 'Circuit triangle signed prediction failed')
    real_f = scattering(psi, q, e, mask, s, o, kind='real_magnitude')
    real_r = scattering(psi, q, e, mask, o, s, kind='real_magnitude')
    require(float((real_f['transmission']-real_r['transmission']).abs().max()) < 1e-12, 'Real control reciprocity failed')
    conjugate = psi.conj()
    cq = route(conjugate, phi, mask, kind='linear_coherent')['alpha'][:,:1]
    cd = scattering(conjugate,cq,e,mask,s,o)['transmission']-scattering(conjugate,cq,e,mask,o,s)['transmission']
    require(float(cd) > 0, 'Conjugated cycle directional sign failed')
    phase = torch.exp(1j*torch.tensor([[.31,-.53,.83]], dtype=torch.float64))
    gauge_psi = psi*phase[...,None]
    gauge_q = route(gauge_psi,phi,mask,kind='linear_coherent')['alpha'][:,:1]
    gauge = scattering(gauge_psi,gauge_q,e,mask,s,o)
    gaugeerr = float((gauge['transmission']-forward['transmission']).abs().max().detach())
    require(torch.allclose(gauge_q,q,atol=1e-12,rtol=0) and gaugeerr < 1e-12, 'End-to-end fixed-value gauge failed')
    derivative = torch.autograd.grad(forward['logits'].sum(), e)[0][0,0]
    expected = 2*GAMMA**2*(inverse[0,1].conj()*inverse[0,:]*inverse[:,1]).real/(forward['transmission'].detach().squeeze()+READOUT_EPS)
    require(float((derivative-expected).abs().max()) < 1e-9, 'Onsite response identity failed')
    roles = torch.zeros(1,3,4, dtype=torch.float64)
    roles[0,0,2] = roles[0,1,3] = 1
    aa = anchors(roles,mask)
    require(aa[0].item()==0 and aa[1].item()==1, 'Anchor convention differs')
    rejected = 0
    for bad in (torch.zeros_like(roles), roles.clone()):
        if bad.any():
            bad[0,0,3] = 1
        try:
            anchors(bad,mask)
        except ValueError:
            rejected += 1
    require(rejected == 2, 'Missing/overlap anchors were accepted')
    p.atomic_json(out/'unit-validation.json', dict(gate=True, source_hashes=hashes(),
        design_sha256=p.identity(design()),core_sha256=p.identity(p.sources()),dense_amplitude_error=err,
        full_gradient_error=graderr, numpy_error=nerr, triangle_delta=float(delta.detach()),
        triangle_prediction=float(predicted.detach()), conjugated_sign_delta=float(cd.detach()),
        gauge_error=gaugeerr, scope='Synthetic implementation identities only'))
    p.atomic_json(out/'RUN_COMPLETE',dict(artifacts={'unit-validation.json':p.digest(out/'unit-validation.json')}))


def worker(args):
    import numpy as np
    import torch
    from baselines.retacred.rmea_entity_scattering import EntityScatteringHead
    from baselines.retacred.rmea_token_transport import TrainableTokenTransportHead
    require(ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGKILL,0,0,0)==0, 'Parent-death fencing failed')
    require(os.getppid()==args.parent_pid and torch.cuda.device_count()==1 and bool(os.environ.get('CUDA_VISIBLE_DEVICES')), 'Parent/GPU identity differs')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    blob = (args.run/'checkpoints/linear_coherent-final.pt').read_bytes()
    require(hashlib.sha256(blob).hexdigest()==CHECKPOINT, 'Checkpoint differs')
    head = (TrainableTokenTransportHead('linear_coherent') if args.arm=='linear_reference' else EntityScatteringHead(args.arm)).double()
    head.load_state_dict(torch.load(io.BytesIO(blob),map_location='cpu',weights_only=True))
    head = head.to('cuda').train()
    require(sum(x.numel() for x in head.parameters())==81028, 'Parameter budget differs')
    data = {k:np.load(args.run/'cache'/('train-'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('h','roles','offsets','targets')}
    out = args.output/args.arm
    out.mkdir()
    losses, norms = [], []
    started = time.monotonic()
    for i in range(COUNT):
        lo, hi = map(int,data['offsets'][i:i+2])
        inputs = tuple(torch.from_numpy(np.array(data[k][lo:hi],dtype=np.float64,copy=True))[None].to('cuda') for k in ('h','roles'))
        mask = torch.ones(1,hi-lo,dtype=torch.bool,device='cuda')
        head.zero_grad(set_to_none=True)
        d = head(*inputs,mask,details=True)
        z = d['logits']
        loss = torch.nn.functional.cross_entropy(z,torch.tensor([int(data['targets'][i])],device='cuda'))
        loss.backward()
        grads = [x.grad for x in head.parameters()]
        require(torch.isfinite(z).all() and torch.isfinite(loss) and all(g is not None and torch.isfinite(g).all() for g in grads), 'Nonfinite/missing full-head gradient')
        losses.append(float(loss.detach()))
        norms.append({n:float(x.grad.norm()) for n,x in head.named_parameters()})
        event(out,'backward_check',i+1,COUNT,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'])
    row = dict(arm=args.arm, count=COUNT, checkpoint_sha256=CHECKPOINT, parameters=81028,
        mean_train_fixture_ce=sum(losses)/COUNT, gradient_norm_max={k:max(v[k] for v in norms) for k in norms[0]},
        seconds=time.monotonic()-started, physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        optimizer_steps=0, scope='Finite forward/backward readiness, not training or utility evidence')
    p.atomic_json(out/'summary.json',row)
    p.atomic_json(out/'WORKER_COMPLETE',dict(summary_sha256=p.digest(out/'summary.json')))


def run(args):
    from baselines.retacred.orchestrate import gpu_ids
    require(args.unit_output is not None, 'Explicit completed unit output required')
    unit_output = args.unit_output.resolve()
    ur = p.read(unit_output/'unit-validation.json')
    us = p.read(unit_output/'RUN_COMPLETE')
    require(us['artifacts']['unit-validation.json']==p.digest(unit_output/'unit-validation.json')
        and ur.get('gate') is True and ur.get('source_hashes')==hashes()
        and ur.get('design_sha256')==p.identity(design()) and ur.get('core_sha256')==p.identity(p.sources()), 'Unit success does not bind this reviewed contract')
    devices = gpu_ids('auto')
    require(len(devices)==3, 'One exact three-GPU allocation required')
    args.run, args.output = args.run.resolve(), args.output.resolve()
    require(not(args.output.is_relative_to(ROOT) or args.output.is_relative_to(args.run) or args.run.is_relative_to(args.output)), 'Separate output required')
    qa.fresh_output(args.output)
    require(p.digest(args.run/'manifest.json')==MANIFEST, 'Protected manifest digest differs')
    manifest, seal = p.read(args.run/'manifest.json'), p.read(args.run/'RUN_COMPLETE')
    require(manifest['source_sha256']==qa.ORIGINAL and manifest['config']==p.config('discovery'), 'Original run identity differs')
    require(seal['artifacts']['manifest.json']['sha256']==p.digest(args.run/'manifest.json'), 'Original manifest seal differs')
    cp = args.run/'checkpoints/linear_coherent-final.pt'
    require(not cp.is_symlink() and p.digest(cp)==CHECKPOINT, 'Pinned checkpoint differs')
    cache = {}
    for k in ('h','roles','offsets','targets'):
        name = 'train-'+k+'.npy'
        f = args.run/'cache'/name
        row = manifest['cache'][name]
        require(not f.is_symlink() and f.stat().st_size==row['bytes'] and p.digest(f)==row['sha256'], 'Training input differs: '+name)
        cache[name] = row
    p.atomic_json(args.output/'inputs.json',dict(checkpoint_sha256=CHECKPOINT,manifest_sha256=MANIFEST,
        unit_validation_sha256=p.digest(unit_output/'unit-validation.json'),cache=cache,training_units=[0,COUNT],no_validation_or_test_access=True))
    p.atomic_json(args.output/'design.json',design())
    jobs = [(arm,[sys.executable,'-B',str(Path(__file__).resolve()),'worker','--run',str(args.run),
        '--output',str(args.output),'--parent-pid',str(os.getpid()),'--arm',arm],gpu) for arm,gpu in zip(ARMS,devices)]
    def report(out,phase,states):
        event(out,phase,sum(x.get('current',0) for x in states),COUNT*len(ARMS),jobs=states)
    started = time.monotonic()
    qa.execute(jobs,args.output,'readiness',report=report)
    rows = []
    for arm in ARMS:
        out = args.output/arm
        require(p.read(out/'WORKER_COMPLETE')['summary_sha256']==p.digest(out/'summary.json'), 'Worker seal differs')
        row = p.read(out/'summary.json')
        require(row['count']==COUNT and row['optimizer_steps']==0 and row['checkpoint_sha256']==CHECKPOINT, 'Worker contract differs')
        rows.append(row)
    require(len({x['physical_gpu'] for x in rows})==3, 'Distinct reserved GPU assignment required')
    p.atomic_json(args.output/'summary.json',dict(schema='q-rmea.entity-scattering.readiness.result.v1',
        gate=True, arms=rows, seconds=time.monotonic()-started, source=p.git('rev-parse','HEAD'),
        source_hashes=hashes(),design_sha256=p.identity(design()),core_sha256=p.identity(p.sources()),
        scope='Implementation readiness only; no task-utility, independent-confirmation or L5 claim'))
    event(args.output,'complete',COUNT*len(ARMS),COUNT*len(ARMS))
    p.atomic_json(args.output/'RUN_COMPLETE',dict(artifacts={n:p.digest(args.output/n) for n in ('summary.json','design.json','inputs.json')}))


def main():
    a = argparse.ArgumentParser()
    a.add_argument('action',choices=('unit','run','worker'))
    a.add_argument('--run',type=Path,default=Path(INPUT_RUN))
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--arm',choices=ARMS)
    a.add_argument('--parent-pid',type=int)
    a.add_argument('--unit-output',type=Path)
    args = a.parse_args()
    reviewed()
    if args.action=='unit':
        unit(args.output)
    elif args.action=='worker':
        worker(args)
    else:
        run(args)


if __name__=='__main__':
    main()
