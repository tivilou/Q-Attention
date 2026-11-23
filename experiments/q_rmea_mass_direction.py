"""Frozen-checkpoint validation-only 2x2 routing diagnostic; no optimization."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.retacred import rmea_token_transport_protocol as p

ARMS = ('native', 'mass_only', 'direction_only', 'full')
HEADS = ('native', 'cubic_coherent', 'linear_coherent')
ORIGINAL_SOURCE = 'd9837319bb368e6b43939596802a40e05d1f276cbe5e26c27f34947ceb6f3ee7'


def design():
    return dict(schema='q-rmea.mass-direction.design.v1', split='valid', count=19584,
        seed=20261121, bootstrap=1000, heads=list(HEADS), arms=list(ARMS),
        primary_head='cubic_coherent', primary_metric='positive39_macro_f1',
        success='direction_only exceeds full and native; paired95% bootstrap lower bounds both >0; CE no worse than native',
        secondary='Other heads are descriptive adaptation checks; no promotion by selecting a different head',
        pairing='Same19584 validation units, same1000 bootstrap resamples across heads and arms',
        smoothing='Add same Born epsilon after each raw score; normalize over valid tokens for each arm',
        trace='First2 immutable cache-order valid units per head; bounded numeric mechanism trace, not full sample-trace.v2',
        zero_mass='At z<=1e-12 stop; no guessed direction or clipping',
        exclusions='No test access, training, seed/tau changes, new COS objects or model/data downloads',
        scope='Exploratory frozen-checkpoint diagnostic, not independent confirmation or L5 novelty')


def arms(psi, phi, mask):
    import torch
    from baselines.retacred.rmea_token_transport import route
    from baselines.retacred.rmea import EPS
    routing = route(psi, phi, mask, kind='cubic_coherent')
    x = psi.conj().masked_fill(~mask[..., None], 0)
    y = routing['unitary'] @ x
    z = y.abs().square().sum(-1)
    if (z[mask] <= 1e-12).any():
        raise ValueError('Degenerate routed row; direction not identifiable')
    native = routing['amplitudes'].abs().square()
    routed = torch.einsum('btd,rd->brt', y, phi).abs().square()
    direction = routed / torch.where(mask, z, torch.ones_like(z))[:, None, :]
    raw = dict(native=native, mass_only=native*z[:, None, :], direction_only=direction, full=routed)
    alpha = {}
    for name, score in raw.items():
        score = (score+EPS)*mask[:, None, :]
        alpha[name] = score/score.sum(-1, keepdim=True)
        if not torch.isfinite(alpha[name]).all():
            raise ValueError('Nonfinite diagnostic distribution')
    if not torch.allclose(alpha['full'], routing['alpha'], atol=2e-12, rtol=2e-12):
        raise ValueError('Factorized full route differs from original implementation')
    comm = routing['generator'] @ routing['gram'] - routing['gram'] @ routing['generator']
    if comm.diagonal(dim1=-2, dim2=-1).abs().max() > 2e-12:
        raise ValueError('Cubic first-order mass cancellation failed')
    return alpha, dict(z=z, native_raw=native, routed_raw=routed, direction_raw=direction,
        first_mass_derivative=(-1j*comm.diagonal(dim1=-2, dim2=-1)).real)


def macro(gold, pred):
    import numpy as np
    c = np.bincount(gold*40+pred, minlength=1600).reshape(40,40)
    den = c.sum(0)+c.sum(1)
    return float(np.divide(2*c.diagonal(),den,out=np.zeros(40,dtype=float),where=den>0)[1:].mean())


def run(args):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from baselines.retacred.rmea_token_transport import TrainableTokenTransportHead
    from baselines.retacred.common import metrics
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count()!=1:
        raise ValueError('Exactly one scheduler-allocated device per frozen head')
    if p.git('status','--porcelain'):
        raise ValueError('Clean immutable diagnostic source required')
    review = p.read(ROOT/'docs/evidence/rmea/20261121-mass-direction/review.json')
    if (review.get('verdict')!='PASS' or review.get('must_fix')
            or review.get('script_sha256')!=p.digest(Path(__file__))
            or review.get('design_sha256')!=p.identity(design())
            or review.get('core_sha256')!=p.identity(p.sources())):
        raise ValueError('Exact diagnostic review required')
    source = args.run.resolve()
    manifest = p.read(source/'manifest.json')
    summary = p.read(source/'summary.json')
    if manifest['source_sha256']!=ORIGINAL_SOURCE or manifest['config']!=p.config('discovery'):
        raise ValueError('Original frozen experiment identity differs')
    complete = p.read(source/'RUN_COMPLETE')
    for name in ('manifest.json','summary.json'):
        row = complete['artifacts'][name]
        if row['sha256']!=p.digest(source/name):
            raise ValueError('Original source manifest/summary seal changed')
    checkpoint = source/'checkpoints'/(args.head+'-final.pt')
    if p.digest(checkpoint)!=summary['results'][args.head]['checkpoint_sha256']:
        raise ValueError('Frozen final checkpoint differs')
    cache = source/'cache'
    for name in ['metadata.json']+['valid-'+key+'.npy' for key in ('h','roles','offsets','targets')]:
        row=manifest['cache'][name]
        if (cache/name).is_symlink() or (cache/name).stat().st_size!=row['bytes'] or p.digest(cache/name)!=row['sha256']:
            raise ValueError('Validation input identity differs: '+name)
    data={k:np.load(cache/('valid-'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('h','roles','offsets','targets')}
    n=design()['count']
    if len(data['targets'])!=n or len(data['offsets'])!=n+1 or data['offsets'][-1]!=len(data['h']):
        raise ValueError('Validation cache coverage differs')
    out=args.output.resolve()
    if out.is_relative_to(ROOT) or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('Independent diagnostic output required')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists() or (out/'RUN_COMPLETE').exists():
        raise ValueError('Preserve existing diagnostic result')
    p.atomic_json(out/'design.json',design())
    started=time.time()
    attempt=Path(os.environ['RESEARCH_OUTPUT_DIR']).parent.name
    def progress(stage,done,total,unit):
        p.atomic_json(out/'progress.json',dict(schema='research.progress.v1',attempt=attempt,
            stage=stage,phase=stage,unit=unit,completed=done,current=done,total=total,observed_at=time.time(),
            batch_id='rmea-mass-direction-valid-20261121',head=args.head))
    head=TrainableTokenTransportHead('cubic_coherent').double()
    head.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    head.to('cuda').eval()
    labels=sorted(p.read(ROOT/'docs/contracts/q_rmea_entity_labels.json'))
    if len(labels)!=40 or labels[0]!='no_relation':raise ValueError('Fixed label mapping differs')
    pred={k:[] for k in ARMS};loss={k:[] for k in ARMS};gold=[]
    max_gap=0.;trace=[]
    with torch.no_grad():
        for start in range(0,n,32):
            indices=list(range(start,min(start+32,n)))
            lengths=[int(data['offsets'][i+1]-data['offsets'][i]) for i in indices]
            h=np.zeros((len(indices),max(lengths),1024),np.float64)
            roles=np.zeros((len(indices),max(lengths),4),np.float64)
            mask=np.zeros(h.shape[:2],bool)
            for j,(i,length) in enumerate(zip(indices,lengths)):
                a,b=map(int,data['offsets'][i:i+2]);h[j,:length]=data['h'][a:b];roles[j,:length]=data['roles'][a:b];mask[j,:length]=True
            inputs=tuple(torch.from_numpy(x).to('cuda') for x in (h,roles,mask))
            y=torch.tensor(np.asarray(data['targets'][indices],np.int64),device='cuda')
            d=head(*inputs,details=True)
            distributions,values=arms(d['psi'],d['phi'],inputs[2])
            max_gap=max(max_gap,float((values['z'][inputs[2]]-1).abs().max()))
            for name in ARMS:
                logits=(distributions[name]*d['utilities']).sum(-1)+head.bias
                if not torch.isfinite(logits).all():raise ValueError('Nonfinite logits')
                pred[name].extend(logits.argmax(-1).tolist())
                loss[name].extend(F.cross_entropy(logits,y,reduction='none').tolist())
            gold.extend(y.tolist())
            if start==0:
                arrays={k:v[:2].cpu().numpy() for k,v in values.items()}
                arrays.update(psi=d['psi'][:2].cpu().numpy(),phi=d['phi'].cpu().numpy(),mask=mask[:2],
                    utilities=d['utilities'][:2].cpu().numpy(),bias=head.bias.cpu().numpy(),target=y[:2].cpu().numpy())
                arrays.update({'alpha_'+k:v[:2].cpu().numpy() for k,v in distributions.items()})
                path=out/'fixed-valid-first2.npz';np.savez_compressed(path,**arrays)
                trace=[dict(path=path.name,sha256=p.digest(path),arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()},
                            sample_rule='immutable hash-ordered valid cache indices0,1; chosen before diagnostic',coverage='bounded mechanism values; no full encoder/training trajectory claim')]
            if start%256==0 or start+32>=n:progress('frozen_validation',min(n,start+32),n,'sample')
    gold=np.asarray(gold,np.int64);pred={k:np.asarray(v,np.int64) for k,v in pred.items()}
    rows={k:metrics(gold.tolist(),pred[k].tolist(),labels,float(np.mean(loss[k]))) for k in ARMS}
    for k in ARMS:rows[k]['positive39_macro_f1']=macro(gold,pred[k]);rows[k]['ce']=rows[k]['loss']
    if args.head=='cubic_coherent':
        for metric in ('positive39_macro_f1','ce'):
            if abs(rows['full'][metric]-summary['results'][args.head]['valid'][metric])>1e-10:
                raise ValueError('Full validation replay differs from original report')
    draws={k:[] for k in ('full','native')};rng=np.random.default_rng(design()['seed'])
    for i in range(design()['bootstrap']):
        ids=rng.integers(0,n,size=n);candidate=macro(gold[ids],pred['direction_only'][ids])
        for k in draws:draws[k].append(candidate-macro(gold[ids],pred[k][ids]))
        if i%100==0:progress('paired_bootstrap',i,design()['bootstrap'],'resample')
    contrasts={k:dict(delta=rows['direction_only']['positive39_macro_f1']-rows[k]['positive39_macro_f1'],
                    paired95_percentile=np.quantile(v,[.025,.975]).tolist()) for k,v in draws.items()}
    gate=all(v['delta']>0 and v['paired95_percentile'][0]>0 for v in contrasts.values()) and rows['direction_only']['ce']<=rows['native']['ce']
    p.atomic_json(out/'trace-metadata.json',dict(artifacts=trace,trace_status='partial bounded probe; not full sample-trace.v2'))
    result=dict(schema='q-rmea.mass-direction.result.v1',head=args.head,split='valid',count=n,
        results=rows,contrasts=contrasts,gate=gate,primary_gate=gate if args.head=='cubic_coherent' else None,
        max_row_mass_gap=max_gap,source=p.git('rev-parse','HEAD'),script_sha256=p.digest(Path(__file__)),
        original_source=ORIGINAL_SOURCE,checkpoint_sha256=p.digest(checkpoint),design_sha256=p.identity(design()),
        seconds=time.time()-started,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        limitations=['Frozen-head exploratory validation diagnostic, not independent seed confirmation','No novel-theory/L5 claim','Other heads cannot rescue failed primary','1000 paired example bootstraps condition on this fixed cohort, not seed variability'])
    p.atomic_json(out/'summary.json',result)
    p.atomic_json(out/'RUN_COMPLETE',dict(summary_sha256=p.digest(out/'summary.json'),design_sha256=p.digest(out/'design.json'),trace_sha256=p.digest(out/'trace-metadata.json')))
    progress('complete',n,n,'sample')
    print(json.dumps(dict(head=args.head,primary_gate=result['primary_gate'],seconds=result['seconds'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--head',choices=HEADS,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())


if __name__=='__main__':main()
