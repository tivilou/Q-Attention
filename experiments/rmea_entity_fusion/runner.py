"""One restartable four-condition bounded development screen; never formal/test."""
import argparse,gc,hashlib,json,os,resource,subprocess,sys,time
from pathlib import Path
SRC=Path(__file__).resolve().parent;REPO=SRC.parents[1];sys.path.insert(0,str(REPO))
import numpy as np
import torch
from fusion import FusionHead,VARIANTS
from unit import run as unit_run
from coverage_data import load_overlay,sha

def save(p,x):p.write_bytes((json.dumps(x,indent=2)+'\n').encode())
def metrics(z,y):
    z=np.asarray(z,np.float64);pred=z.argmax(-1);cm=np.zeros((40,40),np.int64);np.add.at(cm,(y,pred),1)
    tp=np.diag(cm);den=cm.sum(0)+cm.sum(1);f=np.divide(2*tp,den,out=np.zeros(40),where=den>0);m=z.max(-1)
    loss=m+np.log(np.exp(z-m[:,None]).sum(-1))-z[np.arange(len(y)),y]
    return dict(macro_f1=float(f.mean()),accuracy=float((pred==y).mean()),mean_ce=float(loss.mean()),n=len(y),per_class_f1=f.tolist(),confusion=cm.tolist())

def batch(data,split,indices):
    off=data[split+'_offsets'];length=max(int(off[i+1]-off[i]) for i in indices)
    h=torch.zeros(len(indices),length,1024);r=torch.zeros(len(indices),length,4);mask=torch.zeros(len(indices),length,dtype=torch.bool)
    for j,i in enumerate(indices):
        a,b=map(int,off[i:i+2]);n=b-a;h[j,:n]=torch.from_numpy(data[split+'_features'][a:b].astype(np.float32));r[j,:n]=torch.from_numpy(data[split+'_roles'][a:b]);mask[j,:n]=True
    return h,r,mask,torch.from_numpy(data[split+'_targets'][indices])

def evaluate(model,data,cfg,phase_off=False):
    model.eval();rows=[];clamped=0
    with torch.inference_mode():
        for a in range(0,1024,cfg['batch_size']):
            h,r,mask,_=batch(data,'valid',list(range(a,min(a+cfg['batch_size'],1024))))
            out=model(h,r,mask,details=True,phase_off=phase_off);rows.append(out['logits'].numpy().copy());clamped+=out['clamped']
    z=np.concatenate(rows);assert z.shape==(1024,40) and np.isfinite(z).all()
    return z,metrics(z,data['valid_targets']),clamped

def main():
    args=argparse.ArgumentParser();args.add_argument('action',choices=['run']);args.add_argument('--output',type=Path,required=True);args=args.parse_args()
    start=time.perf_counter();torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert torch.__version__=='2.5.1+cpu' and np.__version__=='1.26.4' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    cfg=json.loads((SRC/'contract.json').read_bytes());contract_sha=sha(SRC/'contract.json')
    git=lambda *a:subprocess.check_output(['git','-C',str(REPO),*a],stderr=subprocess.DEVNULL).decode().strip()
    assert not git('status','--porcelain=v1');revision=git('rev-parse','HEAD')
    for n,h in cfg['repo_dependency_files'].items():assert sha(REPO/n)==h,n
    out=args.output;assert out.is_dir() and not any(out.iterdir()),'Fresh scheduler output required'
    private=out/'private';private.mkdir();u=unit_run();save(out/'unit.json',u)
    d,overlay=load_overlay(cfg['base_cache'],cfg['overlay'],cfg['overlay_complete_sha256'])
    assert len(d['train_targets'])==2361 and len(d['valid_targets'])==1024 and min(np.bincount(d['train_targets'],minlength=40))==6
    identity=dict(base_features_sha256=sha(Path(cfg['base_cache'])/'features.npz'),overlay_complete_sha256=sha(Path(cfg['overlay'])/'RUN_COMPLETE'),counts=dict(train=2361,valid=1024),valid_ids_sha256=hashlib.sha256(b''.join(d['valid_ids'])).hexdigest(),train_ids_sha256=hashlib.sha256(b''.join(d['train_ids'])).hexdigest(),test_loaded=False,cpu_rebuilt_cache_not_original_gpu_bitwise=True)
    save(out/'input-identity.json',identity)
    rng=np.random.default_rng(cfg['seed']);orders=[rng.permutation(2361).tolist() for _ in range(cfg['epochs'])]
    results={};history={};shared_hash=None
    for variant in VARIANTS:
        torch.manual_seed(cfg['seed']);model=FusionHead(variant)
        common=hashlib.sha256()
        for name,p in model.named_parameters():
            if not name.startswith(('effect','gates')):common.update(name.encode()+p.detach().numpy().tobytes())
        if shared_hash is None:shared_hash=common.hexdigest()
        else:assert common.hexdigest()==shared_hash
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);best=-1.;hist=[];clamps=0
        for epoch,order in enumerate(orders):
            model.train();loss_sum=0.
            for offset in range(0,2361,cfg['batch_size']):
                ix=order[offset:offset+cfg['batch_size']];h,r,mask,y=batch(d,'train',ix)
                optimizer.zero_grad(set_to_none=True);detail=model(h,r,mask,details=True);loss=torch.nn.functional.cross_entropy(detail['logits'],y)
                assert torch.isfinite(loss);loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip']);optimizer.step();loss_sum+=float(loss.detach())*len(ix);clamps+=detail['clamped']
            z,met,c=evaluate(model,d,cfg);hist.append(dict(epoch=epoch+1,train_mean_ce=loss_sum/2361,valid=met,valid_roundoff_clamps=c))
            if met['macro_f1']>best:
                best=met['macro_f1'];selected=epoch+1;best_metrics=met;torch.save(dict(state=model.state_dict(),variant=variant,epoch=selected,source_revision=revision,contract_sha256=contract_sha),private/(variant+'.pt'));np.save(private/(variant+'-valid-logits.npy'),z,allow_pickle=False)
            print(json.dumps(dict(event='epoch',variant=variant,epoch=epoch+1,valid_macro=met['macro_f1'])),flush=True)
        checkpoint=torch.load(private/(variant+'.pt'),map_location='cpu',weights_only=True);assert checkpoint['variant']==variant and checkpoint['source_revision']==revision and checkpoint['epoch']==selected
        model.load_state_dict(checkpoint['state']);z,again,c=evaluate(model,d,cfg);assert again==best_metrics and np.array_equal(z,np.load(private/(variant+'-valid-logits.npy'),allow_pickle=False))
        k,l=model.coefficients();record=dict(selected_epoch=selected,valid=best_metrics,parameters=sum(p.numel() for p in model.parameters()),updates=cfg['epochs']*((2361+cfg['batch_size']-1)//cfg['batch_size']),shared_initialization_sha256=common.hexdigest(),kappa=k.detach().numpy().tolist(),lam=l.detach().numpy().tolist(),train_roundoff_clamps=clamps)
        if variant=='fusion':
            zz,off,cc=evaluate(model,d,cfg,phase_off=True);record['phase_off_fixed_selected']=dict(valid=off,valid_roundoff_clamps=cc);np.save(private/'fusion-phase-off-valid-logits.npy',zz,allow_pickle=False)
        results[variant]=record;history[variant]=hist;del optimizer,model;gc.collect()
    delta_mix=results['fusion']['valid']['macro_f1']-results['mixture']['valid']['macro_f1'];delta_global=results['fusion']['valid']['macro_f1']-results['global']['valid']['macro_f1']
    assert sha(Path(cfg['base_cache'])/'features.npz')==identity['base_features_sha256'] and sha(Path(cfg['overlay'])/'RUN_COMPLETE')==identity['overlay_complete_sha256']
    save(out/'epoch-metrics.json',history);save(out/'source-validation.json',dict(status='PASS',revision=revision,contract_sha256=contract_sha,repo_dependency_files=cfg['repo_dependency_files'],shared_initialization_sha256=shared_hash,source_clean=not git('status','--porcelain=v1'),all_inputs_unchanged=True,all_selected_roundtrips_equal=True))
    summary=dict(status='PASS',schema='qattn.entity-fusion.screen.v1',source_revision=revision,contract_sha256=contract_sha,seed=cfg['seed'],epochs=cfg['epochs'],results=results,fusion_minus_mixture_macro=delta_mix,fusion_minus_global_macro=delta_global,screen_disposition=('RETAIN_EXPLORATORY_UTILITY_LEAD' if delta_mix>0 and delta_global>0 else 'HOLD_NO_PROMOTION'),test_loaded=False,formal_runs=0,head_training_runs=4,new_encoder_forward=0,new_cos_reads=0,l5_accepted=False,elapsed_seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Exposed2361/1024 development screen with validation selection, no independent utility confirmation or inherited L4. Known partial-swap operator; novelty and nontrivial same-contribution theory unresolved.')
    save(out/'summary.json',summary)
    names=sorted(p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file());files={n:dict(sha256=sha(out/n),bytes=(out/n).stat().st_size) for n in names}
    assert sum(v['bytes'] for v in files.values())<16*1024**2
    save(out/'RUN_COMPLETE',dict(schema='qattn.entity-fusion.closed.v1',revision=revision,files=files));print(json.dumps(dict(status='PASS',disposition=summary['screen_disposition'],elapsed_seconds=summary['elapsed_seconds'])),flush=True)
if __name__=='__main__':main()
