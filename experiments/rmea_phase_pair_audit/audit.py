"""Forward-free paired audit of existing selected development logits only."""
import argparse,hashlib,json,os,resource,subprocess,time
from pathlib import Path
import numpy as np
SRC=Path(__file__).resolve().parent;REPO=SRC.parents[1]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def save(p,x):p.write_bytes((json.dumps(x,indent=2)+'\n').encode())
def metrics(z,y):
    z=np.asarray(z,np.float64);pred=z.argmax(1);cm=np.zeros((40,40),np.int64);np.add.at(cm,(y,pred),1);tp=np.diag(cm);den=cm.sum(0)+cm.sum(1);f=np.divide(2*tp,den,out=np.zeros(40),where=den>0)
    loss=np.log(np.exp(z-z.max(1)[:,None]).sum(1))+z.max(1)-z[np.arange(len(y)),y]
    other=z.copy();other[np.arange(len(y)),y]=-np.inf;margin=z[np.arange(len(y)),y]-other.max(1)
    return dict(n=len(y),macro_f1=float(f.mean()),accuracy=float((pred==y).mean()),mean_ce=float(loss.mean()),per_class_f1=f.tolist(),confusion=cm.tolist()),pred,loss,margin
def pair(a,b,y,mask):
    _,pa,la,ma=a;_,pb,lb,mb=b;n=int(mask.sum());assert mask.dtype==np.bool_
    if n==0:return dict(n=0,status='empty')
    aa=pa[mask]==y[mask];bb=pb[mask]==y[mask];d=(la-lb)[mask];md=(ma-mb)[mask]
    return dict(n=n,prediction_changes=int((pa[mask]!=pb[mask]).sum()),both_correct=int((aa&bb).sum()),a_only_correct=int((aa&~bb).sum()),b_only_correct=int((~aa&bb).sum()),both_wrong=int((~aa&~bb).sum()),ce_a_minus_b_mean=float(d.mean()),ce_a_better=int((d<0).sum()),ce_b_better=int((d>0).sum()),ce_equal=int((d==0).sum()),true_margin_a_minus_b_mean=float(md.mean()),true_margin_a_better=int((md>0).sum()),true_margin_b_better=int((md<0).sum()),true_margin_equal=int((md==0).sum()))
def main():
    arg=argparse.ArgumentParser();arg.add_argument('action',choices=['run']);arg.add_argument('--output',required=True,type=Path);a=arg.parse_args();start=time.perf_counter()
    assert np.__version__=='1.26.4' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    cfg=json.loads((SRC/'contract.json').read_bytes());contract_sha=sha(SRC/'contract.json');git=lambda *args:subprocess.check_output(['git','-C',str(REPO),*args],stderr=subprocess.DEVNULL).decode().strip();assert not git('status','--porcelain=v1');revision=git('rev-parse','HEAD')
    for n,h in cfg['repo_dependency_files'].items():assert sha(REPO/n)==h,n
    out=a.output;assert out.is_dir() and not any(out.iterdir())
    for n,h in cfg['immutable_inputs'].items():assert Path(n).is_file() and not Path(n).is_symlink() and sha(n)==h,n
    identity=json.loads((REPO/cfg['prior_identity']).read_bytes());base=Path(cfg['base_cache'])/'features.npz';assert sha(base)==identity['base_features_sha256']
    with np.load(base,allow_pickle=False) as f:y=f['valid_targets'];ids=f['valid_ids'];roles=f['valid_roles'];off=f['valid_offsets'];train=f['train_targets']
    assert y.shape==ids.shape==(1024,) and hashlib.sha256(b''.join(ids)).hexdigest()==identity['valid_ids_sha256']
    topup=Path(cfg['overlay'])/'topup-features.npz';closed=json.loads((Path(cfg['overlay'])/'RUN_COMPLETE').read_bytes());assert sha(topup)==closed['files']['topup-features.npz']
    with np.load(topup,allow_pickle=False) as f:extra=f['targets']
    train=np.concatenate([train,extra]);assert train.shape==(2361,);support=np.bincount(train,minlength=40)
    zs={};allmet={}
    for key,path in cfg['logits'].items():
        z=np.load(path,allow_pickle=False);assert z.shape==(1024,40) and np.isfinite(z).all();zs[key]=z;allmet[key]=metrics(z,y)
        expected=json.loads((REPO/cfg['expected_summaries'][key]['path']).read_bytes())
        for k in cfg['expected_summaries'][key]['keys']:expected=expected[k]
        assert allmet[key][0]==expected,key
    orientation=[];distance=[]
    for i in range(1024):
        r=roles[int(off[i]):int(off[i+1])];sp=np.flatnonzero(r[:,2]==1);op=np.flatnonzero(r[:,3]==1);assert len(sp) and len(op)
        d=(float(sp[0]+sp[-1])-float(op[0]+op[-1]))/2;orientation.append(np.sign(d));distance.append(abs(d))
    orientation=np.asarray(orientation);distance=np.asarray(distance);one=np.ones(1024,dtype=bool)
    strata={'all':one,'subject_before_object':orientation<0,'same_midpoint':orientation==0,'subject_after_object':orientation>0,'midpoint_gap_le32':distance<=32,'midpoint_gap_gt32':distance>32,'training_support_le16':support[y]<=16,'training_support_gt16':support[y]>16}
    assert sum(int(strata[k].sum()) for k in ['subject_before_object','same_midpoint','subject_after_object'])==1024
    comparisons={}
    for name,ka,kb in [('full_vs_phase_free','full','phase_free'),('phase_free_vs_control','phase_free','control')]:
        comparisons[name]={'strata':{k:pair(allmet[ka],allmet[kb],y,mask) for k,mask in strata.items()},'all40classes':[dict(class_index=k,train_support=int(support[k]),valid_support=int((y==k).sum()),a_f1=allmet[ka][0]['per_class_f1'][k],b_f1=allmet[kb][0]['per_class_f1'][k],pair=pair(allmet[ka],allmet[kb],y,y==k)) for k in range(40)]}
        for p in comparisons[name]['strata'].values():
            if p['n']:assert p['both_correct']+p['a_only_correct']+p['b_only_correct']+p['both_wrong']==p['n'] and p['a_only_correct']+p['b_only_correct']<=p['prediction_changes']
    save(out/'pair-metrics.json',dict(scope='All exposed validation units and predeclared disjoint marginal task strata; no outcome-selected cases or confirmatory population inference.',comparisons=comparisons))
    save(out/'input-identity.json',dict(prior_identity=identity,immutable_inputs=cfg['immutable_inputs'],valid_count=1024,train_count=2361,raw_rows_or_predictions_exported=False,test_loaded=False))
    same=np.array_equal(allmet['full'][1],allmet['phase_free'][1]);summary=dict(status='PASS',source_revision=revision,contract_sha256=contract_sha,full_and_phase_free_predictions_equal=same,full_phase_free_max_abs_logit_difference=float(np.abs(zs['full']-zs['phase_free']).max()),full_vs_phase_free=comparisons['full_vs_phase_free']['strata']['all'],phase_free_vs_control=comparisons['phase_free_vs_control']['strata']['all'],macro_phase_free_minus_control=allmet['phase_free'][0]['macro_f1']-allmet['control'][0]['macro_f1'],head_training_runs=0,repeated_prior_training_runs=0,formal_runs=0,new_encoder_forward=0,new_model_forward=0,new_cos_reads=0,test_loaded=False,l5_accepted=False,elapsed_seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Forward-free descriptive audit of three immutable selected logits arrays on already exposed validation. No paired significance test, phase necessity, optimization acceleration theorem, independent confirmation, original-L4 inheritance or L5.')
    save(out/'summary.json',summary)
    assert not git('status','--porcelain=v1')
    for n,h in cfg['immutable_inputs'].items():assert sha(n)==h,n
    save(out/'source-validation.json',dict(status='PASS',revision=revision,contract_sha256=contract_sha,clean_source=True,inputs_unchanged=True,all_three_published_confusions_and_metrics_reproduced=True,predeclared_strata=True,raw_predictions_exported=False,torch_imported=False))
    files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in out.iterdir()};assert sum(x['bytes'] for x in files.values())<1024**2;save(out/'RUN_COMPLETE',dict(schema='qattn.phase-pair.closed.v1',revision=revision,files=files));print(json.dumps(dict(status='PASS',head_training_runs=0,new_model_forward=0,l5_accepted=False,elapsed_seconds=summary['elapsed_seconds'])),flush=True)
if __name__=='__main__':main()
