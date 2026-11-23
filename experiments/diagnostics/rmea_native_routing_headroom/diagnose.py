"""One unchanged native forward on predeclared exposed TRAIN8192; no fit."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import numpy as np
from bounds import fixtures, intervals, source_logits, summarize


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    pending = p.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    pending.replace(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for n in ('source','inputs','reference','checkpoint','design','output','progress','allocation'):
        ap.add_argument('--'+n, required=True, type=Path)
    ap.add_argument('--attempt', required=True); args = ap.parse_args()
    cfg = json.loads(args.design.read_bytes()); launch = json.loads(args.allocation.read_bytes())
    assert launch['allocation']['runner_attempt_id'] == args.attempt
    assert len(launch['allocation']['gpu_uuids']) >= 1
    for n, sha in cfg['code_hashes'].items():
        assert digest(Path(__file__).parent/n) == sha
    for n, sha in cfg['source_hashes'].items():
        assert digest(args.source/n) == sha
    assert digest(args.checkpoint) == cfg['checkpoint_sha256']
    assert digest(args.inputs/'discovery/manifest.json') == cfg['native_manifest_sha256']
    prior = args.reference/'output/diagnostic.private.npz'
    assert digest(prior) == cfg['split_reference_sha256']
    args.output.mkdir(exist_ok=True); assert not (args.output/'started.json').exists()
    write(args.output/'started.json', dict(attempt=args.attempt, design_sha256=digest(args.design)))
    signal.alarm(cfg['timeout_seconds']); started = time.time()
    import torch
    sys.path.insert(0, str(args.source))
    from baselines.retacred.rmea import RMEAHead, EPS
    assert EPS == cfg['eps'] == 1e-6
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    assert torch.cuda.device_count() == 1
    model = RMEAHead('quantum').double()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
    model.to('cuda'); model.eval(); assert sum(p.numel() for p in model.parameters()) == 81028
    with np.load(prior, allow_pickle=False) as q:
        calibration=q['calibration'].copy(); groups=q['groups'].copy(); y=q['labels'].copy()
        prior_logits=q['logits'].copy()
    assert len(y) == 8192 and int(calibration.sum()) == 4192
    assert not set(groups[calibration]) & set(groups[~calibration])
    cache=args.inputs/'discovery/cache'
    data={k:np.load(cache/('train-'+k+'.npy'),mmap_mode='r',allow_pickle=False)
          for k in ('h','roles','offsets','targets')}
    assert np.array_equal(data['targets'][:8192],y)
    check=fixtures(EPS); bias=model.bias.detach().cpu().numpy()
    z=[]; spectral=[]; replay=[]; convex=[]; tolerances=[]; traces=[]
    maxima=dict(native_rayleigh=0.,native_source=0.,prior_logits=0.,extremizer=0.,
                normalization=0.,spectral_outside_convex=0.,native_outside_spectral=0.)
    # No optimizer and no storage of token-level representations or states.
    with torch.inference_mode():
        for lo in range(0,len(y),cfg['microbatch_size']):
            ids=list(range(lo,min(lo+cfg['microbatch_size'],len(y))))
            lengths=[int(data['offsets'][i+1]-data['offsets'][i]) for i in ids]
            h=np.zeros((len(ids),max(lengths),1024),np.float64)
            roles=np.zeros(h.shape[:2]+(4,),np.float64); mask=np.zeros(h.shape[:2],bool)
            for j,i in enumerate(ids):
                a,b=map(int,data['offsets'][i:i+2]);n=b-a
                h[j,:n]=data['h'][a:b];roles[j,:n]=data['roles'][a:b];mask[j,:n]=True
            detail=model(*(torch.from_numpy(x).to('cuda') for x in (h,roles,mask)),details=True)
            states=detail['psi'].cpu().numpy(); u=detail['utilities'].cpu().numpy()
            phi=detail['phi'].cpu().numpy(); logits=detail['logits'].cpu().numpy()
            for j,(i,n) in enumerate(zip(ids,lengths)):
                ps=states[j,:n]; util=u[j,:,:n]; q=intervals(ps,util,bias,EPS)
                tolerance=cfg['relative_tolerance']*max(1.,float(np.max(np.abs(util))+np.max(np.abs(bias))))
                norm_error=max(float(np.max(np.abs(np.linalg.norm(ps,axis=-1)-1))),
                               float(np.max(np.abs(np.linalg.norm(phi,axis=-1)-1))))
                assert norm_error < 1e-10
                assert int(logits[j].argmax()) == int(prior_logits[i].argmax()), 'Native winner changed'
                direct=source_logits(ps,util,phi,bias,EPS)
                quotient=np.einsum('rd,rde,re->r',phi.conj(),q['b'],phi)
                quotient/=np.einsum('rd,de,re->r',phi.conj(),q['s'],phi)
                errors=dict(native_rayleigh=float(np.max(np.abs(logits[j]-bias-quotient))),
                            native_source=float(np.max(np.abs(logits[j]-direct))),
                            prior_logits=float(np.max(np.abs(logits[j]-prior_logits[i]))),
                            extremizer=float(np.max(np.abs(q['spectral']-q['replay']))),
                            normalization=norm_error,
                            native_outside_spectral=max(0.,float(np.max(q['spectral'][:,0]-logits[j])),
                                float(np.max(logits[j]-q['spectral'][:,1]))),
                            spectral_outside_convex=max(0.,float(np.max(q['convex'][:,0]-q['spectral'][:,0])),
                                float(np.max(q['spectral'][:,1]-q['convex'][:,1]))))
                assert all(v < tolerance for k,v in errors.items() if k!='normalization'), errors
                for k,v in errors.items():maxima[k]=max(maxima[k],v)
                z.append(logits[j]);spectral.append(q['spectral']);replay.append(q['replay']);convex.append(q['convex']);tolerances.append(tolerance)
                # Eight predeclared first TRAIN rows, private algebra replay only.
                if i<cfg['fixture_rows']:
                    traces.append(dict(index=i,psi=ps,utilities=util,phi=phi,
                        extremizer_phi=q['phi'],logits=logits[j],bias=bias))
            if lo%128==0 or lo+len(ids)==len(y):
                write(args.progress,dict(schema='research.progress.v1',attempt=args.attempt,
                    phase='native_routing_headroom',current=lo+len(ids),total=len(y),unit='example',
                    observed_at=time.time(),heartbeat_at=time.time()))
    arrays=dict(logits=np.stack(z),spectral=np.stack(spectral),replay=np.stack(replay),
                convex=np.stack(convex),labels=y,calibration=calibration,groups=groups,
                tolerance=np.array(tolerances))
    private=args.output/'diagnostic.private.npz';np.savez_compressed(private,**arrays)
    # Fixed first8 raw numeric fixtures stay on the GPU host and never enter Git.
    fixture_arrays={f'{i}_{k}':v for i,t in enumerate(traces) for k,v in t.items()}
    np.savez_compressed(args.output/'algebra-fixtures.private.npz',**fixture_arrays)
    report=summarize(arrays['logits'],arrays['spectral'],arrays['replay'],arrays['convex'],y,calibration,arrays['tolerance'])
    report.update(profile=cfg['profile'],examples=len(y),seconds=time.time()-started,
                  optimizer_updates=0,source_sha256=cfg['source_hashes'],checkpoint_sha256=digest(args.checkpoint),
                  design_sha256=digest(args.design),diagnostic_sha256=digest(private),
                  private_fixtures_sha256=digest(args.output/'algebra-fixtures.private.npz'),
                  numerical_checks=maxima,numerical_tolerance=cfg['relative_tolerance'],
                  claim_scope=cfg['claim_scope'])
    check['real_native_max_errors']=maxima
    write(args.output/'fixtures.json',check);write(args.output/'summary.json',report)
    write(args.output/'RUN_COMPLETE',dict(files={n:digest(args.output/n) for n in
        ('summary.json','fixtures.json','diagnostic.private.npz','algebra-fixtures.private.npz')}))
    print(json.dumps(dict(status='complete',seconds=report['seconds'],assessment=report['partitions']['assessment']['native_errors'])))


if __name__=='__main__':main()
