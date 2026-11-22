"""A frozen native acceptance-rate contrast, using the existing diagnostic helpers."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import argparse
import hashlib
import importlib.util
import json
import signal
import sys
import time
from pathlib import Path
import numpy as np


def density_contrast(psi, phi, eps=1e-6):
    rho = np.einsum('ti,tj->ij', psi, psi.conj())/len(psi)
    p = np.abs(phi.conj() @ psi.T)**2
    p0 = np.abs(phi)**2 @ (np.abs(psi)**2).T
    d = p.mean(1)+eps; d0 = p0.mean(1)+eps
    dense = np.einsum('ri,ij,rj->r', phi.conj(), rho, phi).real+eps
    dense0 = np.abs(phi)**2 @ rho.diagonal().real+eps
    error = max(float(np.max(abs(d-dense))), float(np.max(abs(d0-dense0))))
    off = rho-np.diag(rho.diagonal())
    return np.log(d)-np.log(d0), d0, float(np.linalg.norm(off)), error


def fixtures():
    rng = np.random.default_rng(61124)
    psi = rng.normal(size=(5,8))+1j*rng.normal(size=(5,8)); psi/=np.linalg.norm(psi,axis=1,keepdims=True)
    phi = rng.normal(size=(4,8))+1j*rng.normal(size=(4,8)); phi/=np.linalg.norm(phi,axis=1,keepdims=True)
    _, _, _, error = density_contrast(psi,phi)
    diagonal = density_contrast(np.eye(8,dtype=complex),phi)[0]
    basis = density_contrast(psi,np.eye(8,dtype=complex))[0]
    zero_error=max(float(np.max(abs(diagonal))),float(np.max(abs(basis))))
    phase = np.exp(1j*.49)
    invariant_error=float(np.max(abs(density_contrast(psi*phase,phi*phase)[0]-density_contrast(psi,phi)[0])))
    assert max(error,zero_error,invariant_error)<1e-12
    return dict(status='PASS',density_scalar_error=error,diagonal_state_or_effect_zero_error=zero_error,global_phase_error=invariant_error)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','inputs','checkpoint','reference','design','output','progress'):
        p.add_argument('--'+n,required=True,type=Path)
    p.add_argument('--attempt',required=True);args=p.parse_args()
    cfg=json.loads(args.design.read_bytes())
    assert hashlib.sha256((args.reference/'diagnose.py').read_bytes()).hexdigest()==cfg['reference_code_sha256']
    # Import already-reviewed calibration, RMS and bootstrap code without executing its main.
    spec=importlib.util.spec_from_file_location('native_diagnostic_reference',args.reference/'diagnose.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    cfg=json.loads(args.design.read_bytes());args.output.mkdir(parents=True,exist_ok=True)
    assert not (args.output/'started.json').exists()
    for path,sha in [(Path(__file__),cfg['code_sha256']),(args.checkpoint,cfg['checkpoint_sha256']),
                     (args.inputs/'discovery/manifest.json',cfg['native_manifest_sha256']),
                     (args.reference/'diagnose.py',cfg['reference_code_sha256']),
                     (args.reference/'output/diagnostic.private.npz',cfg['reference_diagnostic_sha256'])]:
        assert ref.digest(path)==sha
    for n,h in cfg['source_hashes'].items():assert ref.digest(args.source/n)==h
    old=np.load(args.reference/'output/diagnostic.private.npz',allow_pickle=False)
    calibration=old['calibration'];assessment=~calibration;groups=old['groups'];labels=old['labels']
    assert len(labels)==8192 and calibration.sum()==4192 and assessment.sum()==4000
    order=sorted(range(40),key=lambda r:hashlib.sha256(('native-mass-evidence-v1:'+str(r)).encode()).digest())
    perm=np.empty(40,dtype=int)
    for i,r in enumerate(order):perm[r]=order[(i+1)%40]
    assert (perm!=np.arange(40)).all() and len(set(perm))==40
    ref.write(args.output/'started.json',dict(attempt=args.attempt,design_sha256=ref.digest(args.design),prior_split_sha256=cfg['prior_split_sha256'],relation_derangement=perm.tolist()))
    signal.alarm(240);checks=fixtures();started=time.time()
    import torch
    sys.path.insert(0,str(args.source));from baselines.retacred.rmea import RMEAHead,EPS
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    native=RMEAHead('quantum').double().cuda().eval();native.load_state_dict(torch.load(args.checkpoint,map_location='cuda',weights_only=True));native.requires_grad_(False)
    cache=args.inputs/'discovery/cache';data={k:np.load(cache/('train-'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('h','roles','offsets')}
    arrays={k:[] for k in ('feature','dephased_log_mass','mean_utility','coherence','logits','lengths')};parity=0.;density_error=0.
    with torch.inference_mode():
        for lo in range(0,8192,16):
            lengths=[int(data['offsets'][i+1]-data['offsets'][i]) for i in range(lo,lo+16)]
            h=torch.zeros(16,max(lengths),1024,dtype=torch.float64,device='cuda');roles=torch.zeros(16,max(lengths),4,dtype=torch.float64,device='cuda');mask=torch.zeros(16,max(lengths),dtype=torch.bool,device='cuda')
            for j,n in enumerate(lengths):
                aa,bb=map(int,data['offsets'][lo+j:lo+j+2]);h[j,:n]=torch.from_numpy(np.array(data['h'][aa:bb],dtype=np.float64,copy=True)).cuda();roles[j,:n]=torch.from_numpy(np.array(data['roles'][aa:bb],dtype=np.float64,copy=True)).cuda();mask[j,:n]=True
            detail=native(h,roles,mask,details=True);phi=detail['phi'].cpu().numpy();z=detail['logits'].cpu().numpy()
            for j,n in enumerate(lengths):
                psi=detail['psi'][j,:n].cpu().numpy();f,d0,c,e=density_contrast(psi,phi,EPS);density_error=max(density_error,e)
                dn=detail['normalizer'][j,:,0].cpu().numpy()/n
                direct=(np.abs(phi.conj()@psi.T)**2).mean(1)+EPS
                parity=max(parity,float(np.max(abs(dn-direct))),float(np.max(abs(z[j]-old['logits'][lo+j]))))
                for k,v in [('feature',f),('dephased_log_mass',np.log(d0)),('mean_utility',detail['utilities'][j,:,:n].cpu().numpy().mean(1)),('coherence',c),('logits',z[j]),('lengths',n)]:arrays[k].append(v)
            ref.write(args.progress,dict(schema='research.progress.v1',attempt=args.attempt,phase='native_acceptance_evidence',current=lo+16,total=8192,unit='example',heartbeat_at=time.time(),observed_at=time.time()))
    assert max(parity,density_error)<1e-8
    arrays={k:np.asarray(v) for k,v in arrays.items()};z=arrays['logits'];feature=arrays['feature'];broken=feature[:,perm]
    entity=np.stack([arrays['mean_utility'],np.broadcast_to(arrays['coherence'][:,None],feature.shape)],axis=-1)
    f,fit,ranks=ref.residualize(feature,z,arrays['dephased_log_mass'],entity,arrays['lengths'],calibration)
    cf,cfit,cranks=ref.residualize(broken,z,arrays['dephased_log_mass'],entity,arrays['lengths'],calibration)
    f,rms=ref.unit_rms(f);cf,crms=ref.unit_rms(cf);eligible=(rms>1e-12)&(crms>1e-12);f[~eligible]=0;cf[~eligible]=0
    assert np.max(abs(np.sqrt((f*f).mean(1))-np.sqrt((cf*cf).mean(1))))<1e-10
    g=np.exp(z-z.max(1,keepdims=True));g/=g.sum(1,keepdims=True);g[np.arange(8192),labels]-=1
    d=(g*f).sum(1);cd=(g*cf).sum(1);orientation=-float(np.sign(d[calibration].mean()));co=-float(np.sign(cd[calibration].mean()));d*=orientation;cd*=co
    raw,_=ref.unit_rms(feature);raw_d=(g*raw).sum(1)*orientation
    boot=ref.bootstrap(np.column_stack([d[assessment],(d-cd)[assessment]]),groups[assessment],cfg['bootstrap_replicates'],cfg['bootstrap_seed']);ci=np.quantile(boot,[.025,.975],axis=0)
    gates=dict(nonzero_orientations=orientation!=0 and co!=0,matched_feature_nondegenerate=bool(eligible[calibration].mean()>=.99 and eligible[assessment].mean()>=.99),raw_direction_consistent=float(raw_d[assessment].mean())<0,adjusted_negative_upper=float(ci[1,0])<0,paired_beats_deranged_upper=float(ci[1,1])<0)
    arrays.update(calibration=calibration,groups=groups,labels=labels,nuisance_fit=fit,deranged_nuisance_fit=cfit,normalized_intact=f,normalized_deranged=cf,derivative=d,deranged_derivative=cd,raw_derivative=raw_d,bootstrap=boot,eligible=eligible,relation_derangement=perm)
    np.savez(args.output/'diagnostic.private.npz',**arrays);ref.write(args.output/'fixtures.json',checks)
    summary=dict(status='completed',train_items=8192,relations=40,calibration=4192,assessment=4000,valid_read=False,test_read=False,optimizer_updates=0,cos_writes=0,seconds=time.time()-started,native_replay_max_error=parity,density_form_max_error=density_error,
        orientations=dict(intact=orientation,deranged=co),assessment_derivatives=dict(intact=float(d[assessment].mean()),deranged=float(cd[assessment].mean()),paired=float((d-cd)[assessment].mean()),raw_intact=float(raw_d[assessment].mean())),bootstrap_percentile_intervals=dict(intact=list(map(float,ci[:,0])),paired=list(map(float,ci[:,1]))),eligible_counts=dict(calibration=int(eligible[calibration].sum()),assessment=int(eligible[assessment].sum())),assessment_class_contributions=list(map(float,(g*f*orientation)[assessment].mean(0))),nuisance_rank_range=[min(ranks+cranks),max(ranks+cranks)],gates=gates,gate='PASS_FOR_PLUGIN_PREPARATION_ONLY' if all(gates.values()) else 'FAIL_CLOSE_EXACT_PROFILE',warning='Adaptive exploratory exposedTRAIN; exactsentence groups only; no independent utility, guaranteedbootstrapcoverage, quantumnecessity, novelty orL5',code_sha256=cfg['code_sha256'],design_sha256=ref.digest(args.design),checkpoint_sha256=cfg['checkpoint_sha256'],source_hashes=cfg['source_hashes'],attempt=args.attempt,private_diagnostic_sha256=ref.digest(args.output/'diagnostic.private.npz'),L5='NOT_ACCEPTED')
    ref.write(args.output/'summary.json',summary);ref.write(args.output/'RUN_COMPLETE',{n:ref.digest(args.output/n) for n in ('summary.json','fixtures.json')})
    ref.write(args.progress,dict(schema='research.progress.v1',attempt=args.attempt,phase='completed',current=8192,total=8192,unit='example',heartbeat_at=time.time(),observed_at=time.time()));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
