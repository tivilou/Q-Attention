"""One fixed TRAIN32 geometry/gradient/cost check; no fit or held-out use."""
import argparse
import hashlib
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from normalized_mass import NormalizedMassSteering


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('source', 'inputs', 'checkpoint', 'design', 'population', 'output', 'progress'):
        p.add_argument('--' + n, type=Path, required=True)
    p.add_argument('--attempt', required=True)
    args = p.parse_args()
    cfg = json.loads(args.design.read_bytes())
    signal.alarm(cfg['timeout_seconds'])
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'RUN_COMPLETE').exists()
    def progress(phase, current, total):
        save(args.progress, dict(schema='research.progress.v1', attempt=args.attempt, phase=phase,
                                current=current, total=total, unit='case' if phase=='geometry' else 'stage',
                                observed_at=time.time(), heartbeat_at=time.time()))
    assert digest(args.population) == cfg['population_sha256']
    assert digest(args.checkpoint) == cfg['checkpoint_sha256']
    assert digest(args.inputs / 'discovery/manifest.json') == cfg['native_manifest_sha256']
    for n, h in cfg['source_hashes'].items():
        assert digest(args.source / n) == h
    for n in ('normalized_mass.py', 'probe.py'):
        assert digest(Path(__file__).parent / n) == cfg['code_hashes'][n]
    population = json.loads(args.population.read_bytes())
    assert len(population['cases']) == 32 and population['train_total'] == 58465
    sys.path.insert(0, str(args.source))
    from baselines.retacred.rmea import RMEAHead, EPS, circuit
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(13)
    device = torch.device('cuda:0')
    assert torch.cuda.is_available()
    native = RMEAHead('quantum').double().to(device).eval()
    native.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    torch.cuda.synchronize()
    start = time.perf_counter()
    model = NormalizedMassSteering(native, circuit, EPS, cfg['smax'])
    torch.cuda.synchronize()
    cache_seconds = time.perf_counter() - start
    cache = args.inputs / 'discovery/cache'
    meta = json.loads((cache / 'metadata.json').read_bytes())['train']
    data = {k: np.load(cache / ('train-' + k + '.npy'), mmap_mode='r', allow_pickle=False) for k in ('h', 'roles', 'offsets')}
    def batch(cases):
        lengths = [int(data['offsets'][v['index'] + 1] - data['offsets'][v['index']]) for v in cases]
        h = torch.zeros(len(cases), max(lengths), 1024, dtype=torch.float64, device=device)
        roles = torch.zeros(len(cases), max(lengths), 4, dtype=torch.float64, device=device)
        mask = torch.zeros(len(cases), max(lengths), dtype=torch.bool, device=device)
        for k, (v, n) in enumerate(zip(cases, lengths)):
            i=v['index']; assert meta[i]['id_hash']==v['id_hash']
            lo, hi = map(int, data['offsets'][i:i+2])
            for name, target in [('h',h),('roles',roles)]:
                target[k,:n] = torch.from_numpy(np.array(data[name][lo:hi], dtype=np.float64, copy=True)).to(device)
            mask[k,:n] = True
        return h, roles, mask
    rows=[]; progress('geometry',0,32)
    with torch.no_grad():
        for i,case in enumerate(population['cases']):
            inputs=batch([case]); record=dict(index=case['index'],id_hash=case['id_hash'],mention_tokens=case['mention_tokens'],signed=[])
            for sign in (-1,1):
                model.beta.fill_(sign*math.atanh(.5)); result=model(*inputs,details=True)
                admitted=bool(result['admitted'][0,0]);delta=float(result['logits'][0,0]-result['native_logits'][0,0]);pred=sign*.5*cfg['smax']*float(result['kappa'][0,0]);error=float(result['first_order_error'][0,0])
                record['signed'].append(dict(sign=sign,eligible=bool(result['eligible'][0,0]),admitted=admitted,kappa=float(result['kappa'][0,0]),logit_delta=delta,predicted_delta=pred,first_order_error=error,raw_error=float(result['raw_error'][0,0]),attention_error=float(result['attention_error'][0,0]),mass_error=float(result['mass_error'][0,0]),positive=admitted and abs(delta)>=1e-8 and abs(error)<=max(1e-8,.05*abs(pred))))
            record['positive']=all(x['positive'] for x in record['signed']);rows.append(record);progress('geometry',i+1,32)
    # Fixed first8 cases, all40 relations. Test the actual beta derivative.
    inputs=batch(population['cases'][:8]);model.beta.data.zero_();model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
    result=model(*inputs,details=True);loss=result['logits'].square().mean();loss.backward();torch.cuda.synchronize();seconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()
    zero_beta_error=float((result['logits']-result['native_logits']).abs().max())
    gradient=model.beta.grad.detach().clone();assert torch.isfinite(gradient).all()
    checks=[];eps=cfg['finite_difference_step']
    with torch.no_grad():
        for k in (0,1,39):
            model.beta.zero_();model.beta[k]=eps;plus=float(model(*inputs).square().mean());model.beta[k]=-eps;minus=float(model(*inputs).square().mean());fd=(plus-minus)/(2*eps);automatic=float(gradient[k]);error=abs(fd-automatic)
            checks.append(dict(coordinate=k,autograd=automatic,finite_difference=fd,error=error,passed=error<=1e-7+.001*abs(fd)))
        model.beta.zero_()
    positive=sum(x['positive'] for x in rows)
    gates=dict(native_identity=zero_beta_error<=1e-12,nonzero_geometry=positive>0,finite_gradients=bool(torch.isfinite(gradient).all()),nonzero_gradient=bool((gradient.abs()>1e-12).any()),finite_difference=all(x['passed'] for x in checks),bounded_cost=seconds<=30 and peak<8*1024**3)
    summary=dict(status='complete',profile=cfg['profile'],design_sha256=digest(args.design),selected_count=32,positive_count=positive,rows=rows,gradient_checks=checks,gradient_abs_max=float(gradient.abs().max()),batch=dict(examples=8,relations=40,forward_backward_seconds=seconds,peak_cuda_bytes=peak,admitted_queries=int(result['admitted'].sum()),total_queries=320),geometry_cache_seconds=cache_seconds,zero_beta_error=zero_beta_error,trainable_parameters=sum(x.numel() for x in model.parameters() if x.requires_grad),gates=gates,local_gate=all(gates.values()),labels_used=False,optimizer_updates=0,cos_writes=0,claim_scope='Exposed TRAIN engineering check only. No task utility, independent confirmation, global convergence or novelty.',L5='NOT_ACCEPTED')
    assert summary['trainable_parameters']==40
    save(args.output/'summary.json',summary);save(args.output/'RUN_COMPLETE',dict(artifacts={'summary.json':digest(args.output/'summary.json')}));progress('complete',1,1)
    print(json.dumps({k:summary[k] for k in ('status','positive_count','batch','gates','local_gate','L5')}),flush=True)


if __name__=='__main__':
    main()
