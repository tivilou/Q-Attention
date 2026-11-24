"""Same stable heads, CE on identical48 exposures, one optional pair loss."""
from pathlib import Path
import torch
from torch.nn import functional as F
from baselines.retacred.rmea_matched_routers import Head,initialize_head
from baselines.retacred.rmea_task_runtime import backward as native_backward
from objective import pair_loss,reference_checks
import task_adapter

KINDS=dict(quantum_ce='quantum',quantum_pair='quantum',dephased_pair='dephased',softmax12_pair='softmax12',real12_pair='real12',uniform_pair='uniform')

def initialize(initial,seed):
    models={name:initialize_head(kind,initial,dict(seed=seed,training_dtype='float64')) for name,kind in KINDS.items()}
    for name,model in models.items():model.use_pair=name!='quantum_ce'
    return models

def make_head(name):return Head(KINDS[name])

def build_task_context(args,cfg,data,fit,assess):return task_adapter.build(args,cfg,data,fit,assess)

def backward(model,inputs,y,context,step,device,batch):
    # All6arms see identical main32 and auxiliary16 labels/rows. CE is their
    # item-weighted48-example mean; ranking weight1 only on declared pair arms.
    base=native_backward(model,inputs,y)
    for p in model.parameters():
        if p.grad is not None:p.grad.mul_(2/3)
    total=base*(2/3)
    aux,labels=batch(context['batches'][step-1],device)
    for lo in range(0,len(labels),4):
        z=model(*(v[lo:lo+4] for v in aux));yy=labels[lo:lo+4]
        rank=pair_loss(z,yy)
        loss=F.cross_entropy(z,yy)*(len(yy)/48)
        if model.use_pair:loss=loss+rank*(len(yy)/16)
        loss.backward();total=total+loss.detach()
    return total

def fixtures(models,inputs):
    checks=reference_checks()
    # The runner prepares CUDA inputs before assigning heads to its two lanes.
    # Keep this scoped parity check on CPU; workers retain placement ownership.
    inputs=tuple(x.cpu() for x in inputs)
    a=models['quantum_ce'];b=models['quantum_pair'];assert all(torch.equal(x,y) for x,y in zip(a.state_dict().values(),b.state_dict().values()))
    with torch.no_grad():error=float((a(*inputs)-b(*inputs)).abs().max())
    assert error==0
    return dict(status='PASS',objective_checks=checks,identical_quantum_forward_error=error,
                scope='New objective formula/gradient/shift invariance and Qreference parity; stable head/runner/resume/COS untouched.')

def summarize(rows,logits,labels,cfg):
    q=rows['quantum_pair'];deltas={name:q['all40_macro_f1']-row['all40_macro_f1'] for name,row in rows.items() if name!='quantum_pair'}
    return dict(results=rows,all40_macro_deltas=deltas,gates={k:v>0 for k,v in deltas.items()},
                gate='PASS_FOR_NEW_FULL_PROTOCOL_DESIGN_ONLY' if all(v>0 for v in deltas.values()) else 'NO_PROMOTION_EXACT_SCREEN',
                limitations='Fixed exposed TRAIN screen; known CE+pairwise logistic auxiliary objective, not independent utility, new theory or quantum necessity. Do not tune coefficients, seeds, controls or budget to rescue.',
                L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED')
