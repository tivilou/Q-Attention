"""One same-gate phase-disabled training audit, not a new formal study."""
import argparse,gc,hashlib,importlib.util,json,os,resource,subprocess,sys,time
from pathlib import Path
SRC=Path(__file__).resolve().parent;REPO=SRC.parents[1];BASE=REPO/'experiments/rmea_entity_fusion'
sys.path[:0]=[str(REPO),str(BASE)]
import numpy as np
import torch
from fusion import FusionHead
from coverage_data import load_overlay,sha
spec=importlib.util.spec_from_file_location('frozen_round6_runner',BASE/'runner.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
assert Path(sys.modules['fusion'].__file__).resolve()==(BASE/'fusion.py').resolve()
assert Path(sys.modules['coverage_data'].__file__).resolve()==(BASE/'coverage_data.py').resolve()
assert Path(sys.modules['unit'].__file__).resolve()==(BASE/'unit.py').resolve()

def save(p,x):p.write_bytes((json.dumps(x,indent=2)+'\n').encode())
def twin(m):
    other=FusionHead('mixture').to(dtype=m.bias.dtype)
    missing=other.load_state_dict({k:v for k,v in m.state_dict().items() if k!='gates'},strict=False)
    assert missing.missing_keys==['gates'] and not missing.unexpected_keys
    k,l=m.coefficients();assert (k.abs()<1).all()
    with torch.no_grad():other.gates[:,0].copy_(torch.atanh(k))
    return other

def unit():
    torch.manual_seed(619);m=FusionHead('fusion').double()
    with torch.no_grad():m.gates[:,0].copy_(torch.linspace(-.3,.3,40,dtype=torch.float64))
    h=torch.randn(2,5,1024,dtype=torch.float64);r=torch.zeros(2,5,4,dtype=torch.float64);mask=torch.tensor([[1]*5,[1,1,1,0,0]],dtype=torch.bool)
    for row,n in enumerate([5,3]):
        pos=torch.arange(5,dtype=torch.float64);r[row,:,0]=pos/32;r[row,:,1]=(pos-n+1)/32;r[row,0,2]=1;r[row,n-1,3]=1
    z,a=m(h,r,mask,phase_off=True);other=twin(m);zz,aa=other(h,r,mask)
    torch.testing.assert_close(z,zz,atol=2e-12,rtol=2e-12);torch.testing.assert_close(a,aa,atol=2e-12,rtol=2e-12)
    torch.nn.functional.cross_entropy(z,torch.tensor([1,2])).backward()
    assert torch.equal(m.gates[:,1],torch.zeros(40,dtype=torch.float64)) and torch.equal(m.gates.grad[:,1],torch.zeros(40,dtype=torch.float64))
    assert m.gates.grad[:,0].abs().max()>0 and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    return dict(status='PASS',phase_free_function_twin_max_logit_error=float((z-zz).abs().max()),disabled_gate_and_gradient_exact_zero=True,nonzero_bias_gate_gradient=True,stored_parameters=sum(p.numel() for p in m.parameters()),inactive_coordinates=40,scope='Numerical graph/mapping check; elementary reparameterization, not L5 or equal expressive capacity.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['run']);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();start=time.perf_counter()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert torch.__version__=='2.5.1+cpu' and np.__version__=='1.26.4' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    cfg=json.loads((SRC/'contract.json').read_bytes());csha=sha(SRC/'contract.json');git=lambda *a:subprocess.check_output(['git','-C',str(REPO),*a],stderr=subprocess.DEVNULL).decode().strip()
    assert not git('status','--porcelain=v1');revision=git('rev-parse','HEAD')
    for n,h in cfg['repo_dependency_files'].items():assert sha(REPO/n)==h,n
    prior=json.loads((REPO/cfg['prior_summary']).read_bytes());identity=json.loads((REPO/cfg['prior_input_identity']).read_bytes())
    assert prior['seed']==cfg['seed']==307 and prior['epochs']==cfg['epochs']==4 and prior['source_revision']==cfg['prior_scientific_revision']
    assert prior['results']['fusion']['parameters']==80628 and not prior['test_loaded']
    out=args.output;assert out.is_dir() and not any(out.iterdir());private=out/'private';private.mkdir();save(out/'unit.json',unit())
    d,overlay=load_overlay(cfg['base_cache'],cfg['overlay'],cfg['overlay_complete_sha256'])
    observed=dict(base_features_sha256=sha(Path(cfg['base_cache'])/'features.npz'),overlay_complete_sha256=sha(Path(cfg['overlay'])/'RUN_COMPLETE'),counts=dict(train=len(d['train_targets']),valid=len(d['valid_targets'])),valid_ids_sha256=hashlib.sha256(b''.join(d['valid_ids'])).hexdigest(),train_ids_sha256=hashlib.sha256(b''.join(d['train_ids'])).hexdigest(),test_loaded=False,cpu_rebuilt_cache_not_original_gpu_bitwise=True)
    assert observed==identity and observed['counts']==dict(train=2361,valid=1024);save(out/'input-identity.json',observed)
    rng=np.random.default_rng(cfg['seed']);orders=[rng.permutation(2361).tolist() for _ in range(4)]
    torch.manual_seed(cfg['seed']);model=FusionHead('fusion');common=hashlib.sha256()
    for n,p in model.named_parameters():
        if not n.startswith(('effect','gates')):common.update(n.encode()+p.detach().numpy().tobytes())
    assert common.hexdigest()==prior['results']['fusion']['shared_initialization_sha256']
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);history=[];best=-1.;clamps=0
    for epoch,order in enumerate(orders):
        model.train();loss_sum=0.
        for offset in range(0,2361,cfg['batch_size']):
            ix=order[offset:offset+cfg['batch_size']];h,r,mask,y=base.batch(d,'train',ix);optimizer.zero_grad(set_to_none=True)
            detail=model(h,r,mask,details=True,phase_off=True);loss=torch.nn.functional.cross_entropy(detail['logits'],y);assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()) and torch.equal(model.gates.grad[:,1],torch.zeros_like(model.gates[:,1]))
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip']);optimizer.step();assert torch.equal(model.gates[:,1],torch.zeros_like(model.gates[:,1]));loss_sum+=float(loss.detach())*len(ix);clamps+=detail['clamped']
        z,met,c=base.evaluate(model,d,cfg,phase_off=True);history.append(dict(epoch=epoch+1,train_mean_ce=loss_sum/2361,valid=met,valid_roundoff_clamps=c))
        if met['macro_f1']>best:
            best=met['macro_f1'];selected=epoch+1;selected_metrics=met;torch.save(dict(state=model.state_dict(),epoch=selected,source_revision=revision,contract_sha256=csha),private/'control.pt');np.save(private/'control-valid-logits.npy',z,allow_pickle=False)
        print(json.dumps(dict(event='epoch',variant='same_gate_phase_disabled_training',epoch=epoch+1,valid_macro=met['macro_f1'])),flush=True)
    ck=torch.load(private/'control.pt',map_location='cpu',weights_only=True);assert ck['source_revision']==revision and ck['contract_sha256']==csha and ck['epoch']==selected
    model.load_state_dict(ck['state']);z,again,c=base.evaluate(model,d,cfg,phase_off=True);assert again==selected_metrics and np.array_equal(z,np.load(private/'control-valid-logits.npy',allow_pickle=False))
    alias=twin(model);zz,alias_metrics,cc=base.evaluate(alias,d,cfg);error=float(np.max(np.abs(z-zz)));assert error<2e-5 and np.array_equal(z.argmax(-1),zz.argmax(-1))
    delta=prior['results']['fusion']['valid']['macro_f1']-selected_metrics['macro_f1'];ce_delta=selected_metrics['mean_ce']-prior['results']['fusion']['valid']['mean_ce']
    assert sha(Path(cfg['base_cache'])/'features.npz')==identity['base_features_sha256'] and sha(Path(cfg['overlay'])/'RUN_COMPLETE')==identity['overlay_complete_sha256'] and not git('status','--porcelain=v1')
    save(out/'epoch-metrics.json',history);save(out/'source-validation.json',dict(status='PASS',revision=revision,contract_sha256=csha,repo_dependency_files=cfg['repo_dependency_files'],shared_initialization_sha256=common.hexdigest(),source_clean=True,inputs_unchanged=True,selected_checkpoint_roundtrip_equal=True,ordinary_mixture_twin_max_logit_error=error,ordinary_mixture_twin_predictions_equal=True,gate2_and_gradients_exact_zero_every_update=True))
    summary=dict(status='PASS',schema='qattn.phase-training-audit.v1',source_revision=revision,contract_sha256=csha,seed=307,epochs=4,selected_epoch=selected,valid=selected_metrics,prior_fusion_valid=prior['results']['fusion']['valid'],prior_selected_phase_off_valid=prior['results']['fusion']['phase_off_fixed_selected']['valid'],fusion_minus_same_gate_control_macro=delta,control_minus_fusion_ce=ce_delta,screen_disposition=('RETAIN_EXPLORATORY_TRAINING_COMPONENT_SIGNAL' if delta>0 else 'NO_TRAINING_PHASE_UTILITY_OBSERVED_IN_SCREEN'),stored_parameters=80628,inactive_coordinates=40,train_roundoff_clamps=clamps,head_training_runs=1,repeated_prior_training_runs=0,formal_runs=0,new_encoder_forward=0,new_cos_reads=0,test_loaded=False,l5_accepted=False,elapsed_seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='One later control on exposed development. Differences include the intended disabled training channel, optimizer trajectory and selection. No independent confirmation, capacity-matched phase attribution, mature performance, novel theory, inherited L4 or L5.')
    save(out/'summary.json',summary);files={p.relative_to(out).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()};assert sum(v['bytes'] for v in files.values())<4*1024**2
    save(out/'RUN_COMPLETE',dict(schema='qattn.phase-training.closed.v1',revision=revision,files=files));print(json.dumps(dict(status='PASS',disposition=summary['screen_disposition'],elapsed_seconds=summary['elapsed_seconds'])),flush=True)
if __name__=='__main__':main()
