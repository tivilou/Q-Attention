"""One fixed training batch: paired context credit versus a matched shuffled step."""
import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.retacred import rmea_token_transport_protocol as p
from experiments import q_rmea_backward_credit as b

EVIDENCE = 'docs/evidence/rmea/20261121-credit-pairing'
ARM_NAMES = ('full', 'direct', 'matched_shuffle')
ANCHOR_SOURCE = 'f4b730e71f310e9034470ac4bc1f81050ebb36f6'
STATIC_SHA = 'beb177fd74d3cb4af3573ee28f5badc092f673ba850b21b8f31e22d072a1147e'


def design():
    return dict(schema='q-rmea.credit-pairing.design.v1', input_checkpoint_sha256=b.CHECKPOINT_SHA,
        batch='The original checkpoint-owned next32 training examples, seed13 epoch0 order positions46816:46848; exactly one fixed batch',
        gradient='Right-trivialize each8x8 rotation loss cotangent: K=skew(R^H V),remove trace. Own cotangent R_i K_i and shuffled R_i K_perm(i) pull back through each original context Jacobian; direct full-minus-ownVJP checked against explicit detach-feature differentiation.',
        frame='Fixed8D feature representation, right-trivialized SU(8) cotangent; invariant to per-example global phase, not claimed invariant to arbitrary factorization or metric choice',
        permutation='One SHA256(credit-cotangent-v1:batch_position) ordered cycle, no fixed point; no labels/outcomes in construction',
        optimizer='Original active1463step AdamW moments/hyperparameters; each proposed one-step displacement computed from the same immutable state and global clip1',
        matching='Control displacement matches full norm and projection onto the direct-only displacement; preserve shuffled orthogonal direction. Matching uses training steps only.',
        degeneracy='Zero direct displacement or full/shuffled orthogonal residual<=1e-12 of its displacement norm means uninformative; stop with no substitute permutation or axis.',
        validation='All19584 validation units, common static rotation reused byte-for-byte from completed backward-credit prep at original fork weights. No test or validation-dependent update choice.',
        static_rotation_sha256=STATIC_SHA,
        primary='Informative matching, -<gradient L_valid, full_step-matched_step> >0 AND mean(CE_matched-CE_full)>0 with paired1000bootstrap95% lower bound>0',
        secondary='Initial/direct/full/matched CE and actual full-step loss change; a local directional result is not full-training utility or independent replication',
        numeric='Explicit VJP chain rule, unchanged optimizer input, norm/projection matching, first2 immutable validation units central difference of the fixed contrast; partial probe only',
        execution='Prepare train gradient once; one shared3GPU allocation, three disjoint contiguous6528-example validation shards, identical frozen proposed steps',
        recovery='Cheap restart from original COS-protected input and sealed prior static rotation; no iterative training/checkpoint generation or cloud writes',
        exclusions='No checkpoint/batch/step/permutation selection, further seed/tau, automatic promotion, quantum/L5 novelty claim; oldL4 and negative results retained')


def permutation(n):
    if n < 2:
        raise ValueError('At least two examples required')
    order = sorted(range(n),key=lambda i:hashlib.sha256(('credit-cotangent-v1:'+str(i)).encode()).digest())
    result = [0]*n
    for j,i in enumerate(order):
        result[i] = order[(j+1)%n]
    return result


def logits(head, inputs, rotation):
    import torch
    from baselines.retacred.rmea import EPS
    d = head(*inputs,details=True)
    r = rotation.expand(len(inputs[0]),-1,-1) if rotation.ndim==2 else rotation
    x = d['psi'].conj().masked_fill(~inputs[2][...,None],0)
    a = torch.einsum('btd,brd->brt',x,torch.einsum('bij,rj->bri',r,d['phi']))
    raw = (a.abs().square()+EPS)*inputs[2][:,None,:]
    return (raw/raw.sum(-1,keepdim=True)*d['utilities']).sum(-1)+head.bias


def transported_cotangents(rotation,cotangent,perm):
    import torch
    with torch.no_grad():
        a=rotation.mH@cotangent
        k=.5*(a-a.mH)
        trace=k.diagonal(dim1=-2,dim2=-1).sum(-1)
        k=k-trace[:,None,None]/k.shape[-1]*torch.eye(k.shape[-1],device=k.device,dtype=k.dtype)
        return rotation@k,rotation@k[perm],k,trace


def gradient_packet(head, inputs, y):
    import torch
    from torch.nn import functional as F
    d = head(*inputs,details=True)
    r = torch.matrix_exp(-.25j*b.qa.generator(d['psi'],inputs[2]))
    # Reuse these exact states; the only extra path is the context rotation.
    from baselines.retacred.rmea import EPS
    a = torch.einsum('btd,brd->brt',d['psi'].conj().masked_fill(~inputs[2][...,None],0),
                     torch.einsum('bij,rj->bri',r,d['phi']))
    raw = (a.abs().square()+EPS)*inputs[2][:,None,:]
    z = (raw/raw.sum(-1,keepdim=True)*d['utilities']).sum(-1)+head.bias
    loss = F.cross_entropy(z,y)
    named = tuple(head.named_parameters())
    params = tuple(v for _,v in named)
    cotangent = torch.autograd.grad(loss,r,retain_graph=True)[0].detach()
    full = torch.autograd.grad(loss,params,retain_graph=True)
    perm = permutation(len(y))
    own_cot,shuffled_cot,k,phase_trace=transported_cotangents(r.detach(),cotangent,perm)
    own = torch.autograd.grad(r,params,grad_outputs=own_cot,retain_graph=True,allow_unused=True)
    shuffled = torch.autograd.grad(r,params,grad_outputs=shuffled_cot,allow_unused=True)
    gradients = {k:{} for k in ('full','direct','shuffle')}
    for (name,param),f,o,s in zip(named,full,own,shuffled):
        o = torch.zeros_like(param) if o is None else o
        s = torch.zeros_like(param) if s is None else s
        gradients['full'][name] = f.detach()
        gradients['direct'][name] = (f-o).detach()
        gradients['shuffle'][name] = (f-o+s).detach()
    direct_loss = F.cross_entropy(logits(head,inputs,r.detach()),y)
    explicit = torch.autograd.grad(direct_loss,params)
    error = max(float((g-gradients['direct'][n]).abs().max()) for (n,_),g in zip(named,explicit))
    if not all(torch.allclose(g,gradients['direct'][n],atol=2e-9,rtol=1e-7) for (n,_),g in zip(named,explicit)):
        raise ValueError('Rotation cotangent VJP decomposition differs')
    if not all(torch.isfinite(v).all() for row in gradients.values() for v in row.values()):
        raise ValueError('Nonfinite gradient packet')
    metadata = dict(loss=float(loss.detach()),permutation=perm,direct_VJP_max_abs=error,
        discarded_phase_response_max=float(phase_trace.abs().max()),
        gradient_norms={a:float(flat(v).norm()) for a,v in gradients.items()},
        credit_norm=float((flat(gradients['full'])-flat(gradients['direct'])).norm()),
        shuffled_credit_norm=float((flat(gradients['shuffle'])-flat(gradients['direct'])).norm()))
    return gradients, metadata, dict(rotation=r.detach(),cotangent=cotangent,right_cotangent=k,permutation=torch.tensor(perm))


def flat(row):
    import torch
    return torch.cat([v.reshape(-1) for v in row.values()])


def shaped(vector, template):
    result = {}; start=0
    for n,tensor in template.items():
        result[n] = vector[start:start+tensor.numel()].reshape_as(tensor).clone()
        start += tensor.numel()
    if start != vector.numel():
        raise ValueError('Parameter inventory differs')
    return result


def proposed_step(head, optimizer_state, gradient):
    import torch
    from baselines.retacred.rmea import RMEAHead
    with torch.random.fork_rng(devices=[]):
        clone = RMEAHead('quantum').double().to(next(head.parameters()).device)
    clone.load_state_dict(head.state_dict())
    cfg = p.config('discovery')
    optimizer = torch.optim.AdamW(clone.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'],
        betas=tuple(cfg['betas']),eps=cfg['optimizer_eps'])
    optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    for name,param in clone.named_parameters():
        param.grad = gradient[name].clone()
    clipped_from = float(torch.nn.utils.clip_grad_norm_(clone.parameters(),cfg['clip_norm'],error_if_nonfinite=True))
    optimizer.step()
    before = dict(head.named_parameters())
    step = {name:(param-before[name]).detach() for name,param in clone.named_parameters()}
    if not all(torch.isfinite(x).all() for x in step.values()):
        raise ValueError('Nonfinite proposed update')
    return step,clipped_from


class Uninformative(ValueError):
    pass


def match(full,direct,shuffle):
    import torch
    f,d,q = flat(full),flat(direct),flat(shuffle)
    dn = d.norm()
    if not torch.isfinite(dn) or dn==0:
        raise Uninformative('zero direct displacement')
    u = d/dn; a=torch.dot(f,u)
    residual=f-a*u; other=q-torch.dot(q,u)*u
    if residual.norm()<=1e-12*f.norm():
        raise Uninformative('full displacement collinear with direct')
    if other.norm()<=1e-12*q.norm():
        raise Uninformative('shuffled displacement collinear with direct')
    m = a*u+residual.norm()*other/other.norm()
    if (not torch.allclose(m.norm(),f.norm(),atol=1e-12,rtol=1e-10)
            or not torch.allclose(torch.dot(m,d),torch.dot(f,d),atol=1e-12,rtol=1e-10)):
        raise ValueError('Matched-control constraints failed')
    return shaped(m,full),dict(full_norm=float(f.norm()),direct_norm=float(dn),shuffle_norm=float(q.norm()),
        matched_norm=float(m.norm()),projection_full=float(torch.dot(f,d)),projection_matched=float(torch.dot(m,d)),
        full_residual_norm=float(residual.norm()),shuffle_residual_norm=float(other.norm()),
        contrast_norm=float((f-m).norm()))


def moved(head,step,scale=1.):
    import torch
    from baselines.retacred.rmea import RMEAHead
    with torch.random.fork_rng(devices=[]):
        clone = RMEAHead('quantum').double().to(next(head.parameters()).device)
    clone.load_state_dict(head.state_dict())
    with torch.no_grad():
        for n,t in clone.named_parameters():
            t.add_(step[n],alpha=scale)
    return clone


def finite_check(head,inputs,y,rotation,contrast):
    import torch
    from torch.nn import functional as F
    loss = F.cross_entropy(logits(head,inputs,rotation),y)
    gradients = torch.autograd.grad(loss,tuple(head.parameters()))
    derivative = sum(float((g*contrast[n]).sum()) for (n,_),g in zip(head.named_parameters(),gradients))
    with torch.no_grad():
        plus = float(F.cross_entropy(logits(moved(head,contrast,1e-3),inputs,rotation),y))
        minus = float(F.cross_entropy(logits(moved(head,contrast,-1e-3),inputs,rotation),y))
    numeric = (plus-minus)/2e-3
    if abs(derivative-numeric)>1e-8+1e-4*abs(derivative):
        raise ValueError('Fixed direction derivative failed central difference')
    return dict(analytic=derivative,central_difference=numeric,error=abs(derivative-numeric),spacing=1e-3)


def event(out,phase,current,total,**fields):
    p.atomic_json(out/'progress.json',dict(schema='research.progress.v1',
        attempt=Path(os.environ['RESEARCH_OUTPUT_DIR']).parent.name,phase=phase,stage=phase,
        unit='sample',current=current,completed=current,total=total,observed_at=time.time(),
        batch_id='rmea-credit-pairing-20261121',**fields))


def anchor(args):
    import numpy as np
    import torch
    for n,sha in p.read(args.anchor/'RUN_COMPLETE')['artifacts'].items():
        if p.digest(args.anchor/n)!=sha:
            raise ValueError('Prior completed anchor seal differs')
    old = p.read(args.anchor/'summary.json')
    inputs = p.read(args.anchor/'inputs.json')
    if old['source']!=ANCHOR_SOURCE or inputs['input_checkpoint_sha256']!=b.CHECKPOINT_SHA:
        raise ValueError('Prior static rotation provenance differs')
    name='static-rotation.npy';prep=args.anchor/'prepared'
    snapshot=(prep/name).read_bytes()
    if len(snapshot)!=1152 or hashlib.sha256(snapshot).hexdigest()!=STATIC_SHA or STATIC_SHA!=p.read(prep/'PREPARED')['files'][name]:
        raise ValueError('Prior static rotation bytes differ')
    return torch.from_numpy(np.load(io.BytesIO(snapshot),allow_pickle=False)).to('cuda')


def prepare(args):
    import torch
    from baselines.retacred.rmea_token_transport_recovery import atomic_torch,cpu_tree
    head,state=b.init(args);out=args.output/'prepared';out.mkdir()
    event(out,'fixed_batch_gradient',0,1)
    inputs,y=b.batch(b.load(args.run,'train'),b.orders()[0][b.START:b.START+32])
    gradients,meta,trace=gradient_packet(head,inputs,y)
    steps={};clips={}
    for name in gradients:
        steps[name],clips[name]=proposed_step(head,state['optimizer'],gradients[name])
    rotation=anchor(args)
    info=dict(informative=True,gradient=meta,preclip_norms=clips,
        trace_status='32 fixed train rotations/cotangents, aggregate VJPs and2valid derivative probe; partial,not sample-trace.v2')
    try:
        matched,matching=match(steps['full'],steps['direct'],steps['shuffle'])
    except Uninformative as exc:
        info.update(informative=False,reason=str(exc))
    else:
        proposed=dict(full=steps['full'],direct=steps['direct'],matched_shuffle=matched)
        contrast={n:steps['full'][n]-matched[n] for n in matched}
        vi,vy=b.batch(b.load(args.run,'valid'),[0,1])
        info.update(matching=matching,finite_probe=finite_check(head,vi,vy,rotation,contrast))
        atomic_torch(out/'steps.pt',cpu_tree(proposed))
    atomic_torch(out/'numeric-probe.pt',cpu_tree(trace))
    p.atomic_json(out/'calibration.json',info)
    files=['calibration.json','numeric-probe.pt']+(['steps.pt'] if info['informative'] else [])
    p.atomic_json(out/'PREPARED',dict(files={n:p.digest(out/n) for n in files}))
    event(out,'fixed_batch_gradient',1,1)


def worker(args):
    import numpy as np
    import torch
    from torch.nn import functional as F
    head,_=b.init(args);out=args.output/('shard'+str(args.shard));out.mkdir()
    prep=args.output/'prepared'
    for n,sha in p.read(prep/'PREPARED')['files'].items():
        if p.digest(prep/n)!=sha:
            raise ValueError('Prepared step seal differs')
    steps=torch.load(prep/'steps.pt',map_location='cuda',weights_only=True)
    models={a:moved(head,steps[a]) for a in ARM_NAMES}
    contrast={n:steps['full'][n]-steps['matched_shuffle'][n] for n in steps['full']}
    rotation=anchor(args);data=b.load(args.run,'valid')
    start0,stop=(b.qa.N//3)*args.shard,(b.qa.N//3)*(args.shard+1)
    loss_arrays=[];direction=0.
    for start in range(start0,stop,32):
        inputs,y=b.batch(data,range(start,min(start+32,stop)))
        z=logits(head,inputs,rotation)
        losses=F.cross_entropy(z,y,reduction='none')
        gradients=torch.autograd.grad(losses.sum(),tuple(head.parameters()))
        direction-=sum(float((g*contrast[n]).sum()) for (n,_),g in zip(head.named_parameters(),gradients))/b.qa.N
        values=[losses.detach().cpu().numpy()]
        with torch.no_grad():
            for a in ARM_NAMES:
                values.append(F.cross_entropy(logits(models[a],inputs,rotation),y,reduction='none').cpu().numpy())
        loss_arrays.append(np.stack(values,axis=1))
        event(out,'validation',min(start+32,stop)-start0,stop-start0,shard=args.shard)
    losses=np.concatenate(loss_arrays)
    if not np.isfinite(losses).all() or not np.isfinite(direction):
        raise ValueError('Nonfinite local validation')
    np.save(out/'losses.npy',losses,allow_pickle=False)
    row=dict(shard=args.shard,start=start0,stop=stop,count=stop-start0,columns=['initial',*ARM_NAMES],
        ce=losses.mean(0).tolist(),directional_contrast=direction,losses_sha256=p.digest(out/'losses.npy'))
    p.atomic_json(out/'summary.json',row)
    p.atomic_json(out/'WORKER_COMPLETE',dict(summary_sha256=p.digest(out/'summary.json')))


def command(args,action,shard=None):
    argv=[sys.executable,'-B',str(Path(__file__).resolve()),action,'--run',str(args.run),
        '--output',str(args.output),'--anchor',str(args.anchor),'--parent-pid',str(os.getpid())]
    return argv+(['--shard',str(shard)] if shard is not None else [])


def report(out,phase,states):
    event(out,phase,sum(x['current'] for x in states),1 if phase=='prepare' else b.qa.N,jobs=states)


def run(args):
    import numpy as np
    from baselines.retacred.orchestrate import gpu_ids
    devices=gpu_ids('auto')
    if len(devices)!=3 or p.git('status','--porcelain'):
        raise ValueError('Clean exact3GPU allocation required')
    review=p.read(ROOT/EVIDENCE/'review.json')
    names=['experiments/q_rmea_credit_pairing.py','experiments/check_rmea_credit_pairing.py',
           'experiments/q_rmea_backward_credit.py','experiments/q_rmea_query_alignment.py']
    if (review.get('verdict')!='PASS' or review.get('review_scope')!='both' or review.get('must_fix')
            or review.get('hashes')!={n:p.digest(ROOT/n) for n in names}
            or review.get('design_sha256')!=p.identity(design()) or review.get('core_sha256')!=p.identity(p.sources())):
        raise ValueError('Exact scoped mechanism/lifecycle review required')
    args.run,args.output,args.anchor=args.run.resolve(),args.output.resolve(),args.anchor.resolve()
    if any(args.output.is_relative_to(d) or d.is_relative_to(args.output) for d in (ROOT,args.run.parent,args.anchor)):
        raise ValueError('Independent new attempt output required')
    b.qa.fresh_output(args.output);started=time.time()
    p.atomic_json(args.output/'design.json',design())
    inputs=b.qa.validate_inputs(args.run);state=b.checkpoint(args.run)
    target=args.run/'cache/train-targets.npy'
    if p.digest(target)!=p.read(args.run/'manifest.json')['cache']['train-targets.npy']['sha256']:
        raise ValueError('Original training targets differ')
    inputs.update(input_checkpoint_sha256=b.CHECKPOINT_SHA,order_sha256=state['order_sha'],
        training_targets_sha256=p.digest(target),anchor_summary_sha256=p.digest(args.anchor/'summary.json'),static_rotation_sha256=STATIC_SHA)
    p.atomic_json(args.output/'inputs.json',inputs)
    b.qa.execute([('prepare',command(args,'prepare'),devices[0])],args.output,'prepare',report=report)
    calibration=p.read(args.output/'prepared/calibration.json')
    result=dict(schema='q-rmea.credit-pairing.result.v1',source=p.git('rev-parse','HEAD'),calibration=calibration,
        gate=False,scope='One fixed training batch/update and full development-validation localdirection;not independent utility or L5',
        source_core_sha256=p.identity(p.sources()),design_sha256=p.identity(design()))
    if calibration['informative']:
        jobs=[('shard'+str(i),command(args,'worker',i),gpu) for i,gpu in enumerate(devices)]
        b.qa.execute(jobs,args.output,'validation',report=report)
        parts=[];rows=[]
        for i in range(3):
            out=args.output/('shard'+str(i));row=p.read(out/'summary.json')
            if p.read(out/'WORKER_COMPLETE')['summary_sha256']!=p.digest(out/'summary.json') or p.digest(out/'losses.npy')!=row['losses_sha256']:
                raise ValueError('Validation shard seal differs')
            if row['start']!=(b.qa.N//3)*i or row['stop']!=(b.qa.N//3)*(i+1) or row['count']!=b.qa.N//3:
                raise ValueError('Validation shard coverage differs')
            parts.append(np.load(out/'losses.npy',allow_pickle=False));rows.append(row)
        losses=np.concatenate(parts)
        if losses.shape!=(b.qa.N,4):
            raise ValueError('Full validation loss inventory differs')
        paired=losses[:,3]-losses[:,1]
        rng=np.random.default_rng(20261121)
        draws=[float(paired[rng.integers(0,b.qa.N,size=b.qa.N)].mean()) for _ in range(1000)]
        interval=np.quantile(draws,[.025,.975]).tolist()
        direction=sum(row['directional_contrast'] for row in rows)
        result.update(ce=dict(zip(['initial',*ARM_NAMES],losses.mean(0).tolist())),
            directional_contrast=direction,matched_minus_full_ce=float(paired.mean()),paired95_percentile=interval,
            gate=direction>0 and float(paired.mean())>0 and interval[0]>0,validation_count=b.qa.N,shards=rows)
    else:
        result.update(validation_count=0,status='uninformative_matching')
    result['seconds']=time.time()-started
    p.atomic_json(args.output/'summary.json',result)
    p.atomic_json(args.output/'RUN_COMPLETE',dict(artifacts={n:p.digest(args.output/n) for n in ('design.json','inputs.json','summary.json')}))
    event(args.output,'complete',result['validation_count'],result['validation_count'])
    print(json.dumps(dict(gate=result['gate'],seconds=result['seconds'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','prepare','worker'))
    for name in ('run','output','anchor'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--parent-pid',type=int)
    parser.add_argument('--shard',type=int,choices=range(3))
    args=parser.parse_args()
    if args.action!='run' and not args.parent_pid:
        parser.error('Exact parent required')
    if args.action=='worker' and args.shard is None:
        parser.error('Shard required')
    dict(run=run,prepare=prepare,worker=worker)[args.action](args)


if __name__=='__main__':
    main()
