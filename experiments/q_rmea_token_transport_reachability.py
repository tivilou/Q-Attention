"""Bounded, label-blind mixed-query support certificates on sealed canary traces."""
import argparse
import hashlib
import io
import math
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baselines.retacred import rmea_token_transport_protocol as p

DESIGN='docs/evidence/rmea/20261120-l5-token-graph-separation/reachability-design.json'
THEORY='docs/evidence/rmea/20261120-l5-token-graph-separation/full-vector-reachability.md'
SELF='experiments/q_rmea_token_transport_reachability.py'

def source_identity():
    return p.identity(dict(p.sources(),**{n:p.digest(p.ROOT/n) for n in (SELF,DESIGN,THEORY,'docs/evidence/rmea/20261120-l5-token-graph-separation/nuisance-stability.md')}))

def snapshot(path,expected,limit):
    if path.is_symlink():raise ValueError('Symlinked input')
    with path.open('rb') as f:data=f.read(limit+1)
    if len(data)>limit or hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Pinned input differs')
    return data

def density_projection(matrix,torch):
    values,vectors=torch.linalg.eigh((matrix+matrix.mH)/2)
    descending=values.flip(-1);ranks=torch.arange(1,9,device=values.device,dtype=values.dtype)
    # Torch2.4 CUDA cumsum rejects deterministic mode. Eight explicit ordered
    # additions compute the same prefix sums without that unsupported kernel.
    partial=torch.zeros_like(descending[...,0]);prefix=[]
    for i in range(8):
        partial=partial+descending[...,i];prefix.append(partial)
    offsets=(torch.stack(prefix,-1)-1)/ranks
    count=(descending>offsets).sum(-1,keepdim=True)
    theta=offsets.gather(-1,count-1)
    return (vectors*(values-theta).clamp_min(0)[...,None,:])@vectors.mH

def certificates(effects,alpha,torch,iterations):
    """An approximate primal search proposes w; the spectral bound decides."""
    r=alpha.shape[0];eye=torch.eye(8,dtype=effects.dtype,device=effects.device)
    upper=torch.triu_indices(8,8,1,device=effects.device)
    features=torch.cat((effects.diagonal(dim1=-2,dim2=-1).real,
        2**.5*effects[:,upper[0],upper[1]].real,
        2**.5*effects[:,upper[0],upper[1]].imag),-1)
    lipschitz=torch.linalg.svdvals(features)[0].square()
    rho=eye.expand(r,-1,-1)/8;y=rho.clone();momentum=1.
    for _ in range(iterations):
        prediction=torch.einsum('tij,rji->rt',effects,y).real
        gradient=torch.einsum('rt,tij->rij',(prediction-alpha).to(effects.dtype),effects)
        nxt=density_projection(y-gradient/lipschitz,torch)
        value=(1+(1+4*momentum*momentum)**.5)/2
        y=nxt+(momentum-1)/value*(nxt-rho);rho=nxt;momentum=value
    prediction=torch.einsum('tij,rji->rt',effects,rho).real
    residual=alpha-prediction;norm=residual.norm(dim=-1,keepdim=True)
    w=residual/torch.where(norm>1e-14,norm,torch.ones_like(norm))
    h=torch.einsum('rt,tij->rij',w.to(effects.dtype),effects);h=(h+h.mH)/2
    vals,vecs=torch.linalg.eigh(h)
    backward=(h@vecs-vecs*vals[:,None,:]).norm(dim=(-2,-1))
    margin=(alpha*w).sum(-1)-vals[:,-1]-backward-2e-14
    return margin,float(norm.max()),float(backward.max())

def normalized_effects(psi,unitary,torch,eps):
    eye=torch.eye(8,dtype=psi.dtype,device=psi.device)
    projectors=psi[:,:,None]*psi.conj()[:,None,:]
    denominator=projectors.sum(0)+len(psi)*eps*eye
    inverse=torch.linalg.inv(torch.linalg.cholesky(denominator))
    native=inverse@(projectors+eps*eye)@inverse.mH
    dephased=torch.einsum('ij,jab->iab',unitary.abs().square().to(psi.dtype),native)
    for effects in (native,dephased):
        if (effects.sum(0)-eye).abs().max()>1e-8:raise ValueError('POVM normalization differs')
    return native,dephased

def nuisance_measurement(psi,phi,roles,utilities,bias,torch,eps,route,kinds):
    """One fixed index/role mask; targets are never read."""
    n=psi.shape[1];indices=torch.arange(n,device=psi.device)
    keep=(indices.remainder(4)!=3)|(roles[0,:,2:].abs().sum(-1)>0)
    r=int(keep.sum());m=n-r
    if r<2 or m<1:raise ValueError('Frozen mask requires retained graph and omitted tokens')
    valid=torch.ones(1,n,dtype=torch.bool,device=psi.device)
    subvalid=torch.ones(1,r,dtype=torch.bool,device=psi.device)
    full={k:route(psi,phi,valid,kind=k) for k in kinds}
    sub={k:route(psi[:,keep],phi,subvalid,kind=k) for k in kinds}
    delta=full['native']['alpha'][0,:,~keep].sum(-1)
    u=utilities[0];width=u.max(-1).values-u.min(-1).values
    gram=full['cubic_coherent']['gram'][0]
    offdiag=~torch.eye(n,dtype=torch.bool,device=psi.device)
    touches=(~keep[:,None]|~keep[None,:])&offdiag
    eta=float(gram.abs()[touches].max())
    rg=gram[keep][:,keep].abs();rg=rg-torch.diag_embed(rg.diagonal())
    rho=float(rg.max());rows={};qmetrics={}
    for k in kinds:
        if k=='native':e=lam=relative=0.;term=delta
        else:
            graph=full[k]['graph'][0];small=sub[k]['graph'][0]
            embedded=torch.zeros_like(graph);ix=keep.nonzero().flatten()
            embedded[ix[:,None],ix[None,:]]=small
            e=float(torch.linalg.eigvalsh(graph-embedded).abs().max())
            lam=float(sub[k]['spectral_scale'][0])
            if lam<=0:raise ValueError('Undefined retained normalized graph')
            relative=e/lam;term=delta+2*.25*relative
        bound=width*term.clamp(max=1)
        z=(full[k]['alpha'][0]*u).sum(-1)+bias
        zr=(sub[k]['alpha'][0]*u[:,keep]).sum(-1)+bias
        difference=(z-zr).abs();residual=float((difference-bound).max())
        if not torch.isfinite(bound).all() or residual>1e-9:raise ValueError('Nuisance logit bound violated')
        ce_difference=((z.logsumexp(0)-zr.logsumexp(0))-(z-zr)).abs()
        ce_bounds=bound+bound.max()
        if float((ce_difference-ce_bounds).max())>1e-9:raise ValueError('Gold-free every-target CE bound violated')
        winner=int(zr.argmax());rivals=zr.clone();rivals[winner]=-torch.inf
        margin=float(zr[winner]-rivals.max());other=bound.clone();other[winner]=-torch.inf
        certified=margin>float(bound[winner]+other.max())
        if certified and int(z.argmax())!=winner:raise ValueError('Certified prediction changed')
        nonconstant=width>1e-12
        rows[k]=dict(coordinates=len(width),nonconstant_coordinates=int(nonconstant.sum()),
            nonvacuous_coordinates=int(((term<1)&nonconstant).sum()),
            mean_absolute_logit_shift=float(difference.mean()),maximum_absolute_logit_shift=float(difference.max()),
            mean_bound=float(bound.mean()),maximum_bound=float(bound.max()),maximum_bound_violation=residual,
            maximum_all_target_ce_shift=float(ce_difference.max()),maximum_ce_bound=float(ce_bounds.max()),
            retained_winner_margin=margin,prediction_certified=certified,prediction_unchanged=int(z.argmax())==winner)
        if k in ('linear_coherent','cubic_coherent'):
            q=1 if k=='linear_coherent' else 3
            qmetrics[str(q)]=dict(retained_graph_norm=lam,perturbation_norm=e,relative_perturbation=relative,
                affinity_upper_envelope=(math.sqrt(r*m)+(m-1))*(eta/rho)**q)
    return dict(tokens=n,retained=r,omitted=m,eta=eta,rho=rho,strict_affinity_contrast=eta<rho,
        native_omitted_mass=dict(minimum=float(delta.min()),mean=float(delta.mean()),maximum=float(delta.max())),
        graph=qmetrics,routing=rows,
        cubic_lower_observed_mean_shift=rows['cubic_coherent']['mean_absolute_logit_shift']<rows['linear_coherent']['mean_absolute_logit_shift'],
        cubic_lower_relative_graph_perturbation=qmetrics['3']['relative_perturbation']<qmetrics['1']['relative_perturbation'])


def run_nuisance(design,args,closed,request,output,torch,np,route,kinds,eps):
    rows=[];replay_error=0.
    for row in design['inputs']:
        if closed['artifacts'][row['path']]!=dict(sha256=row['sha256'],bytes=row['bytes']):raise ValueError('Sealed trace reference differs')
        data=snapshot(args.run/row['path'],row['sha256'],1048576)
        with np.load(io.BytesIO(data),allow_pickle=False) as arrays:
            state={k:torch.as_tensor(arrays[k].copy(),device='cuda') for k in ('psi','phi','mask','roles','utilities','bias','route_swap_alpha')}
        if list(state['psi'].shape)!=row['psi_shape'] or not state['mask'].all():raise ValueError('Frozen complete input differs')
        replay=torch.stack([route(state['psi'],state['phi'],state['mask'],kind=k)['alpha'] for k in kinds])
        error=float((replay-state['route_swap_alpha']).abs().max())
        if error>1e-12:raise ValueError('Frozen four-route replay differs')
        replay_error=max(replay_error,error)
        rows.append(nuisance_measurement(*(state[k] for k in ('psi','phi','roles','utilities','bias')),torch,eps,route,kinds))
    summary=dict(units=len(rows),query_coordinates=40*len(rows),maximum_route_replay_error=replay_error,
        strict_affinity_contrast_units=sum(r['strict_affinity_contrast'] for r in rows),
        cubic_lower_observed_mean_shift_units=sum(r['cubic_lower_observed_mean_shift'] for r in rows),
        cubic_lower_relative_graph_perturbation_units=sum(r['cubic_lower_relative_graph_perturbation'] for r in rows),
        routing={k:dict(nonconstant_coordinates=sum(r['routing'][k]['nonconstant_coordinates'] for r in rows),
            nonvacuous_coordinates=sum(r['routing'][k]['nonvacuous_coordinates'] for r in rows),
            certified_predictions=sum(r['routing'][k]['prediction_certified'] for r in rows),
            unchanged_predictions=sum(r['routing'][k]['prediction_unchanged'] for r in rows),
            maximum_bound_violation=max(r['routing'][k]['maximum_bound_violation'] for r in rows),
            mean_absolute_logit_shift=sum(r['routing'][k]['mean_absolute_logit_shift'] for r in rows)/len(rows)) for k in kinds})
    record=dict(status='NUISANCE_STABILITY_DIAGNOSTIC_COMPLETE',source_revision=request['revision'],source_sha256=source_identity(),
        contract_sha256=p.identity(design),summary=summary,units=rows,formal_training=False,L5='NOT_ACCEPTED',
        scope='Six immutable technical-canary examples after one update; fixed role/index head mask, frozen features and values, no target reads. Bounds/informativeness and paired sensitivities only, not semantic invariance, learned utility or coherence necessity.')
    p.atomic_json(output/'compact-report.json',record)
    p.atomic_json(output/'RUN_COMPLETE',dict(schema='q-rmea.nuisance-stability.closed.v1',files={'compact-report.json':p.digest(output/'compact-report.json')}))
    print(json.dumps(dict(status=record['status'],summary=summary)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    design=p.read(p.ROOT/DESIGN);output=args.output.absolute();attempt=output.parent
    request=p.read(attempt/'request.json');command=p.read(attempt/'command.json')
    expected=['{python}','-B',SELF,'--run',design['run_path'],'--output','{output}']
    if (str(args.run)!=design['run_path'] or os.environ.get('RESEARCH_OUTPUT_DIR')!=str(output) or request['project']!='q-attention'
            or request['attempt']!=attempt.name or request['recipe']['command']!=expected
            or request['recipe']['outputs']!=['compact-report.json','RUN_COMPLETE']
            or request['recipe']['timeout']!=1200 or request['recipe']['max_attempts']!=1
            or request['recipe']['replay_safe'] is not False or request['recipe']['kind']!='diagnostic'
            or request['recipe']['resources']!=dict(cpu=1,ram_mb=4096,disk_mb=4096,gpus=1,gpu_mb=8192)
            or request['revision']!=p.git('rev-parse','HEAD') or command['cwd']!=str(p.ROOT)
            or command['revision']!=request['revision']
            or command['argv']!=[v.replace('{python}',sys.executable).replace('{output}',str(output)) for v in expected]
            or request['boot_id']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            or os.environ.get('CUDA_VISIBLE_DEVICES')!=','.join(request['gpu_ids']) or len(request['gpu_ids'])!=1):
        raise ValueError('Exact scheduler-only diagnostic admission required')
    review=p.read(p.ROOT/'docs/validation/q_rmea_token_transport_reachability_review.json')
    if (review.get('verdict') not in ('PASS','PASS_WITH_RISK') or review.get('must_fix')
            or review.get('review_scope')!='both' or review.get('source_sha256')!=source_identity()
            or review.get('contract_sha256')!=p.identity(design)):
        raise ValueError('Exact bounded mathematics and execution review required')
    if output.name!='output' or not output.is_dir() or list(output.iterdir()):raise ValueError('Fresh assigned output required')
    closed=json.loads(snapshot(args.run/'RUN_COMPLETE',design['run_complete_sha256'],262144))
    manifest=json.loads(snapshot(args.run/'manifest.json',design['manifest_sha256'],262144))
    if manifest['source_sha256']!=design['producer_source_sha256']:raise ValueError('Producer identity differs')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import numpy as np
    import torch
    from baselines.retacred.rmea import EPS
    from baselines.retacred.rmea_token_transport import route,KINDS
    torch.set_num_threads(1);torch.set_grad_enabled(False);torch.use_deterministic_algorithms(True)
    if design.get('analysis')=='nuisance_stability_v1':
        run_nuisance(design,args,closed,request,output,torch,np,route,KINDS,EPS);return
    device='cuda';rows=[];replay_error=0.;null_max=-1.;iterations=design['iterations'];tolerance=design['tolerance']
    # Independent analytic depth-two toy must yield a strict dual certificate.
    e=torch.eye(8,dtype=torch.complex128,device=device)
    psi=torch.stack((e[0],3**.5/2*e[0]+.5*e[1],e[1]));query=e[0,None].expand(40,-1)
    mask=torch.ones(1,3,dtype=torch.bool,device=device)
    toy=route(psi[None],query,mask,kind='cubic_coherent');n,d=normalized_effects(psi,toy['unitary'][0],torch,EPS)
    # A point mass at middle token exceeds the independently known native ceiling.
    target=torch.zeros(1,3,dtype=torch.float64,device=device);target[:,1]=1
    toy_margin,*_=certificates(n,target,torch,iterations)
    if toy_margin.min()<.1:raise ValueError('Known infeasible probability vector missed')
    for row in design['inputs']:
        if closed['artifacts'][row['path']]!=dict(sha256=row['sha256'],bytes=row['bytes']):raise ValueError('Sealed trace reference differs')
        data=snapshot(args.run/row['path'],row['sha256'],1048576)
        with np.load(io.BytesIO(data),allow_pickle=False) as arrays:
            psi=torch.as_tensor(arrays['psi'].copy(),device=device)
            phi=torch.as_tensor(arrays['phi'].copy(),device=device)
            mask=torch.as_tensor(arrays['mask'].copy(),device=device)
            recorded=torch.as_tensor(arrays['route_swap_alpha'].copy(),device=device)
        if list(psi.shape)!=row['psi_shape'] or not mask.all():raise ValueError('Frozen complete unpadded sample differs')
        routes={k:route(psi,phi,mask,kind=k) for k in KINDS};alpha={k:v['alpha'][0] for k,v in routes.items()}
        error=float((torch.stack([alpha[k] for k in KINDS])[:,None]-recorded).abs().max())
        if error>1e-12:raise ValueError('Frozen four-route probability replay differs')
        replay_error=max(replay_error,error);native,dephased=normalized_effects(psi[0],routes['cubic_coherent']['unitary'][0],torch,EPS)
        results={};backward=0.
        for kind,effects,label in [('native',native,'native_null'),('linear_coherent',native,'linear_null'),
                ('cubic_initial_dephased',dephased,'dephased_null'),('cubic_coherent',native,'cubic_vs_native'),
                ('cubic_coherent',dephased,'cubic_vs_dephased')]:
            margins,residual,back=certificates(effects,alpha[kind],torch,iterations);backward=max(backward,back)
            results[label]=margins
            if label.endswith('_null'):
                null_max=max(null_max,float(margins.max()))
                if (margins>tolerance).any():raise ValueError('Feasible control falsely separated')
        rows.append(dict(tokens=len(psi[0]),query_count=40,maximum_eigen_backward_error=backward,
            results={k:dict(witnesses=int((v>tolerance).sum()),maximum_margin=float(v.max())) for k,v in results.items()},
            simultaneous_queries=int(((results['cubic_vs_native']>tolerance)&(results['cubic_vs_dephased']>tolerance)).sum())))
    summary=dict(units=len(rows),queries=40*len(rows),token_range=[min(r['tokens'] for r in rows),max(r['tokens'] for r in rows)],
        simultaneous_queries=sum(r['simultaneous_queries'] for r in rows),maximum_route_replay_error=replay_error,
        maximum_null_margin=null_max,results={k:dict(witnesses=sum(r['results'][k]['witnesses'] for r in rows),
            maximum_margin=max(r['results'][k]['maximum_margin'] for r in rows)) for k in rows[0]['results']})
    record=dict(status='FULL_VECTOR_REACHABILITY_DIAGNOSTIC_COMPLETE',source_revision=request['revision'],source_sha256=source_identity(),
        contract_sha256=p.identity(design),summary=summary,units=rows,iterations=iterations,tolerance=tolerance,
        toy_margin=float(toy_margin.min()),formal_training=False,L5='NOT_ACCEPTED',
        scope='Six fixed technical-canary samples after one update; label-blind numerical mixed-query support witnesses. Null is inconclusive. No task utility, novelty or formal-evidence claim.')
    p.atomic_json(output/'compact-report.json',record)
    p.atomic_json(output/'RUN_COMPLETE',dict(schema='q-rmea.reachability.closed.v1',files={'compact-report.json':p.digest(output/'compact-report.json')}))
    print(json.dumps(dict(status=record['status'],summary=summary)),flush=True)

if __name__=='__main__':main()
