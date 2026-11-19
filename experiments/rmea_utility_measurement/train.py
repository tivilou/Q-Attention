"""Matched bounded training diagnostic, with one common native initialization."""
import copy
import hashlib
import json
import math
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.rmea_context_geometry.legacy.rmea import RMEAHead
from experiments.rmea_ecqp_screen import metrics, runner
from experiments.rmea_original_scores.common import save, sha
from experiments.rmea_utility_measurement.head import measured_attention

CONDITIONS = ['native', 'full', 'simplex']


class TrainHead(RMEAHead):
    """No trainable additions. The same head is optimized through each readout."""
    def __init__(self, kind):
        assert kind in CONDITIONS
        super().__init__('quantum')
        self.kind = kind

    def forward(self, h, roles, mask):
        d = super().forward(h, roles, mask, details=True)
        if self.kind == 'native':
            return d['logits'], d['alpha']
        u = d['utilities']
        if self.kind == 'full':
            alpha = measured_attention(d['psi'], d['phi'], u, mask)['alpha']
        else:
            centered = u-(d['alpha']*u).sum(-1,keepdim=True)
            centered = torch.where(mask[:,None],centered,torch.zeros_like(centered))
            scale = centered.abs().amax(-1,keepdim=True)
            scale = torch.where(scale>0,scale,torch.ones_like(scale))
            alpha = d['alpha']*torch.exp(centered/scale)
            alpha = alpha/alpha.sum(-1,keepdim=True)
        return (alpha*u).sum(-1)+self.bias, alpha


def evaluation(model, data, ids, batch_size):
    model.eval();pred=[];ce=0.;first=None
    with torch.no_grad():
        for start in range(0,len(ids),batch_size):
            h,r,mask,y,donor=runner.batch(data,ids[start:start+batch_size],'eval',False)
            assert donor is None
            z=model(h.double(),r.double(),mask)[0]
            assert torch.isfinite(z).all()
            if first is None:first=z.clone()
            pred.extend(z.argmax(-1).tolist());ce+=float(F.cross_entropy(z,y,reduction='sum'))
    row=metrics.from_confusion(metrics.confusion(np.asarray(data['targets'][ids],dtype=np.int64),np.asarray(pred,dtype=np.int64)))
    row['ce']=ce/len(ids)
    return row,first


def run(a,design):
    started=time.monotonic();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    identity=design['data'];settings=design['training']
    data=runner.read_training_prefix(a.assets,a.receipt,a.cohort,identity['cohort_sha256'],a.donors,identity['donors_sha256'],
        expected_receipt_sha256=identity['prefix_receipt_sha256'],expected_plan_sha256=identity['plan_sha256'])
    fit,ev=data['splits']['fit'],data['splits']['eval'];assert len(fit)==3124 and len(ev)==972
    assert len({data['cohort'][int(i)]['group'] for i in fit})==2714 and len({data['cohort'][int(i)]['group'] for i in ev})==866
    torch.manual_seed(settings['seed']);native=RMEAHead('quantum').double()
    initial=copy.deepcopy(native.state_dict());initial_sha=runner.state_fingerprint(initial)
    rng=np.random.default_rng(settings['seed']);orders=[rng.permutation(fit) for _ in range(settings['epochs'])]
    order_sha=hashlib.sha256(b''.join(o.astype('<i8').tobytes() for o in orders)).hexdigest()
    checkpoints=a.output.parent/'scientific-work/checkpoints';assert not checkpoints.exists();checkpoints.mkdir(mode=0o700)
    results={};checkpoint_files={}
    for kind in CONDITIONS:
        model=TrainHead(kind).double();model.load_state_dict(initial);model.train()
        assert runner.state_fingerprint(model.state_dict())==initial_sha and sum(p.numel() for p in model.parameters())==81028
        # Actual fit-input first-batch forward/backward, no optimizer step, then
        # discard gradients. It cannot inspect evaluation labels for gradients.
        h,r,mask,y,_=runner.batch(data,orders[0][:settings['batch_size']],'fit',False)
        first_loss=F.cross_entropy(model(h.double(),r.double(),mask)[0],y)
        assert torch.isfinite(first_loss);first_loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        model.zero_grad(set_to_none=True);assert runner.state_fingerprint(model.state_dict())==initial_sha
        optimizer=torch.optim.AdamW(model.parameters(),lr=settings['lr'],weight_decay=settings['weight_decay'],
            betas=tuple(settings['betas']),eps=settings['eps'])
        epochs=[];steps=0;gradient_max={n:0. for n,_ in model.named_parameters()}
        for epoch,order in enumerate(orders,1):
            model.train();loss_sum=0.
            for start in range(0,len(order),settings['batch_size']):
                chosen=order[start:start+settings['batch_size']]
                h,r,mask,y,donor=runner.batch(data,chosen,'fit',False);assert donor is None
                optimizer.zero_grad(set_to_none=True)
                z=model(h.double(),r.double(),mask)[0];loss=F.cross_entropy(z,y)
                assert torch.isfinite(loss);loss.backward()
                for n,p in model.named_parameters():
                    assert p.grad is not None and torch.isfinite(p.grad).all()
                    gradient_max[n]=max(gradient_max[n],float(p.grad.abs().max()))
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['clip_norm']);assert torch.isfinite(norm)
                optimizer.step();assert all(torch.isfinite(p).all() for p in model.parameters())
                loss_sum+=float(loss.detach())*len(chosen);steps+=1
            epochs.append(dict(epoch=epoch,fit_ce=loss_sum/len(fit)))
            print(json.dumps(dict(progress='EPOCH_COMPLETE',condition=kind,epoch=epoch,optimizer_steps=steps)),file=sys.stderr,flush=True)
        expected_steps=settings['epochs']*math.ceil(len(fit)/settings['batch_size']);assert steps==expected_steps==392
        measured,first=evaluation(model,data,ev,settings['batch_size'])
        final_sha=runner.state_fingerprint(model.state_dict());filename=kind+'-final.pt';path=checkpoints/filename
        assert not path.exists();torch.save(model.state_dict(),path);path.chmod(0o600)
        restored=TrainHead(kind).double();restored.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));restored.eval()
        assert runner.state_fingerprint(restored.state_dict())==final_sha
        h,r,mask,_,_=runner.batch(data,ev[:settings['batch_size']],'eval',False)
        with torch.no_grad():replay=restored(h.double(),r.double(),mask)[0]
        assert torch.equal(replay,first),'Same-current-checkpoint first-batch replay failed'
        checkpoint_files[filename]=dict(sha256=sha(path),bytes=path.stat().st_size)
        results[kind]=dict(metrics=measured,epochs=epochs,parameters=81028,initial_state_sha256=initial_sha,final_state_sha256=final_sha,
            checkpoint=checkpoint_files[filename],optimizer_steps=steps,first_batch_finite_gradient=True,checkpoint_reload_first_batch_exact=True,gradient_max_abs=gradient_max)
        del optimizer,restored,model
    assert sum(v['optimizer_steps'] for v in results.values())==1176
    assert set(p.name for p in checkpoints.iterdir())==set(checkpoint_files)
    save(checkpoints/'RUN_COMPLETE',dict(schema='qattn.utility-training.checkpoints.v1',files={n:v['sha256'] for n,v in checkpoint_files.items()},training=True))
    assert sum(p.stat().st_size for p in checkpoints.iterdir())<=design['budget']['private_checkpoint_bytes_max']
    base=results['native']['metrics'];deltas={}
    for kind in CONDITIONS[1:]:
        m=results[kind]['metrics'];delta=m['macro_f1']-base['macro_f1']
        deltas[kind]=dict(macro_f1=delta,macro_relative=delta/abs(base['macro_f1']) if base['macro_f1'] else None,ce=m['ce']-base['ce'])
    full=deltas['full'];decision='DISCOVERY_MACRO_POINT_POSITIVE' if full['macro_f1']>0 else 'NO_POSITIVE_JOINT_TRAINING_SCREEN'
    return dict(status='TRAIN_ONLY_UTILITY_JOINT_OPTIMIZATION_COMPLETE',decision=decision,results=results,deltas_vs_native=deltas,
        joint_macro_ce_positive=full['macro_f1']>0 and full['ce']<=0,
        full_vs_simplex=dict(macro_f1=results['full']['metrics']['macro_f1']-results['simplex']['metrics']['macro_f1'],ce=results['full']['metrics']['ce']-results['simplex']['metrics']['ce']),
        conditions=CONDITIONS,fit_examples=3124,evaluation_examples=972,fit_groups=2714,evaluation_groups=866,
        training=settings,common_initial_state_sha256=initial_sha,common_epoch_order_sha256=order_sha,optimizer_steps=1176,
        private_checkpoints=checkpoint_files,private_checkpoint_marker_sha256=sha(checkpoints/'RUN_COMPLETE'),
        dtype='float64/complex128',task_training=True,full_formal_run=False,independent_confirmation=False,
        COS_requests=0,cloud_writes=0,new_checkpoints=3,formal_handoff=False,L5='NOT_ACCEPTED',
        retention='Private final checkpoints for exact reload audit and potential post-training analysis; no COS protection claimed; stopped-result retention follows project policy.',
        scope='Single common initialization on repeatedly explored original-training group partition; not original full-data evidence or a confirmation.',
        elapsed_seconds=time.monotonic()-started)
