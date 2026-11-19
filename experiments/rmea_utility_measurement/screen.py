"""Fixed postfit intervention screen; no fitting or new evidence-tier claim."""
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.rmea_context_geometry.legacy.rmea import RMEAHead
from experiments.rmea_ecqp_screen import metrics, runner
from experiments.rmea_utility_measurement.head import measured_attention, utility_permutation

CONDITIONS = ['native', 'full', 'simplex', 'pinched', 'permuted']


def run(a, design):
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(design['seed'])
    identity = design['data']
    data = runner.read_training_prefix(a.assets, a.receipt, a.cohort,
        identity['cohort_sha256'], a.donors, identity['donors_sha256'],
        expected_receipt_sha256=identity['prefix_receipt_sha256'],
        expected_plan_sha256=identity['plan_sha256'])
    ids = data['splits']['eval']
    assert len(ids) == 972 and len({data['cohort'][int(i)]['group'] for i in ids}) == 866
    state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    assert all(n.startswith('head.') for n in state)
    native = RMEAHead('quantum')
    native.load_state_dict({n[len('head.'):]: v for n,v in state.items()})
    assert sum(p.numel() for p in native.parameters()) == 81028
    native.double().eval().requires_grad_(False)
    before = runner.state_fingerprint(native.state_dict())
    predictions = {n: [] for n in CONDITIONS}
    losses = {n: 0. for n in CONDITIONS}
    changes = {n: dict(prediction_changes=0, alpha_l1_sum=0., alpha_l1_max=0.) for n in CONDITIONS}
    comparisons = {n: dict(prediction_disagreements=0, absolute_logit_sum=0., absolute_logit_max=0.) for n in CONDITIONS if n != 'full'}
    zero_errors = dict(alpha=0., logits=0.)
    own_gain_min = None
    batches = 0
    with torch.no_grad():
        for start in range(0, len(ids), design['batch_size']):
            selected = ids[start:start+design['batch_size']]
            h, roles, mask, y, donor = runner.batch(data, selected, 'eval', False)
            assert donor is None
            h, roles = h.double(), roles.double()
            d = native(h, roles, mask, details=True)
            psi, phi, u = d['psi'], d['phi'], d['utilities']
            zero = measured_attention(psi, phi, u, mask, eta=0)
            zero_logits = zero['aggregate'] + native.bias
            for name,left,right in [('alpha',zero['alpha'],d['alpha']),('logits',zero_logits,d['logits'])]:
                error=float((left-right).abs().max());zero_errors[name]=max(zero_errors[name],error)
                assert torch.allclose(left,right,atol=design['tolerances']['zero_update_absolute'],rtol=0)
            full = measured_attention(psi, phi, u, mask)
            pinched = measured_attention(psi, phi, u, mask, kind='pinched')
            permutation = utility_permutation([data['metadata'][int(i)]['id_hash'] for i in selected],mask)
            construction = u.gather(-1,permutation[:,None].expand_as(u))
            wrong = measured_attention(psi, phi, u, mask, construction_utilities=construction)
            center = u-(d['alpha']*u).sum(-1,keepdim=True)
            center = torch.where(mask[:,None],center,torch.zeros_like(center))
            magnitude = center.abs().masked_fill(~mask[:,None],0).amax(-1,keepdim=True)
            magnitude = torch.where(magnitude>0,magnitude,torch.ones_like(magnitude))
            beta = d['alpha']*torch.exp(center/magnitude)
            beta = beta/beta.sum(-1,keepdim=True)
            alphas = dict(native=d['alpha'],full=full['alpha'],simplex=beta,
                pinched=pinched['alpha'],permuted=wrong['alpha'])
            logits = {n:(alpha*u).sum(-1)+native.bias for n,alpha in alphas.items()}
            assert torch.equal(logits['native'],d['logits'])
            gain = full['aggregate']-(d['alpha']*u).sum(-1)
            minimum=float(gain.min());own_gain_min=minimum if own_gain_min is None else min(own_gain_min,minimum)
            assert minimum >= -design['tolerances']['own_gain_absolute']
            for n,z in logits.items():
                assert torch.isfinite(z).all() and torch.isfinite(alphas[n]).all()
                assert torch.allclose(alphas[n].sum(-1),torch.ones_like(z),atol=2e-10,rtol=0)
                assert (alphas[n].masked_select(~mask[:,None])==0).all()
                pred=z.argmax(-1);predictions[n].extend(pred.tolist())
                losses[n]+=float(F.cross_entropy(z,y,reduction='sum'))
                l1=(alphas[n]-d['alpha']).abs().sum(-1)
                changes[n]['prediction_changes']+=int((pred!=logits['native'].argmax(-1)).sum())
                changes[n]['alpha_l1_sum']+=float(l1.sum())
                changes[n]['alpha_l1_max']=max(changes[n]['alpha_l1_max'],float(l1.max()))
            for n,row in comparisons.items():
                diff=(logits['full']-logits[n]).abs()
                row['prediction_disagreements']+=int((logits['full'].argmax(-1)!=logits[n].argmax(-1)).sum())
                row['absolute_logit_sum']+=float(diff.sum());row['absolute_logit_max']=max(row['absolute_logit_max'],float(diff.max()))
            batches+=1
    assert runner.state_fingerprint(native.state_dict())==before
    gold=np.asarray(data['targets'][ids],dtype=np.int64)
    measured={n:metrics.from_confusion(metrics.confusion(gold,np.asarray(pred,dtype=np.int64)))|dict(ce=losses[n]/len(ids)) for n,pred in predictions.items()}
    deltas={n:dict(macro_f1=measured[n]['macro_f1']-measured['native']['macro_f1'],ce=measured[n]['ce']-measured['native']['ce']) for n in CONDITIONS if n!='native'}
    for row in changes.values():row['mean_class_alpha_l1']=row.pop('alpha_l1_sum')/(len(ids)*40)
    for row in comparisons.values():row['mean_absolute_logit']=row.pop('absolute_logit_sum')/(len(ids)*40)
    point_positive=deltas['full']['macro_f1']>0 and deltas['full']['ce']<=0
    # This closes only the fixed-checkpoint inference route if it does not pass.
    # It does not test a separately trained architecture or certify a new tier.
    decision='DISCOVERY_POINT_POSITIVE_NOT_CONFIRMATORY' if point_positive else 'NO_POSITIVE_FIXED_CHECKPOINT_SCREEN'
    return dict(status='TRAIN_ONLY_UTILITY_MEASUREMENT_SCREEN_COMPLETE',decision=decision,
        evaluation_examples=len(ids),groups=866,batches=batches,conditions=CONDITIONS,
        metrics=measured,deltas_vs_native=deltas,changes_vs_native=changes,full_vs_controls=comparisons,
        zero_update_max_errors=zero_errors,minimum_own_class_gain=own_gain_min,
        parameters=81028,checkpoint_unchanged=True,dtype='float64/complex128',seed=design['seed'],
        scope='Repeatedly used original-training development partition, frozen existing final4 weights; descriptive postfit inference intervention only.',
        independent_confirmation=False,trained_candidate_tested=False,formal_handoff=False,L5='NOT_ACCEPTED',
        task_training=False,optimizer_steps=0,new_checkpoints=0,COS_requests=0,cloud_writes=0,
        elapsed_seconds=time.monotonic()-started)
