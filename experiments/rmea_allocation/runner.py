"""Matched, descriptive allocation/value audit. No optimizer or test split.

The only intervention replaces a recipient's attention with a donor's attention
on the identical frozen features, retaining the recipient utilities and bias.
It is an intermediate-variable replay, not a new trained model.
"""
import argparse,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
from legacy.rmea import RMEAHead,MODES
import allocation_trace as traces
import hybrid_trace

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def save(p,v):p.write_bytes((json.dumps(v,indent=2)+'\n').encode())
def ce(z,y):
    a=np.asarray(z,np.float64);m=a.max(-1)
    return m+np.log(np.exp(a-m[:,None]).sum(-1))-a[np.arange(len(a)),y]
def metrics(z,y):
    pred=z.argmax(-1);cm=np.zeros((40,40),np.int64);np.add.at(cm,(y,pred),1)
    tp=np.diag(cm);den=cm.sum(0)+cm.sum(1);f=np.divide(2*tp,den,out=np.zeros(40),where=den>0)
    return dict(macro_f1=float(f.mean()),accuracy=float((pred==y).mean()),mean_ce=float(ce(z,y).mean()),n=len(y))
def compose(alpha,utilities,bias):
    if alpha.shape!=utilities.shape or alpha.ndim!=3 or bias.shape!=(40,):raise ValueError('Replay shapes differ')
    if not np.isfinite(alpha).all() or not np.isfinite(utilities).all():raise ValueError('Nonfinite replay input')
    if alpha.min()<-1e-7 or not np.allclose(alpha.sum(-1),1,atol=2e-6):raise ValueError('Invalid attention')
    return (np.ascontiguousarray(alpha,dtype=np.float64)*np.ascontiguousarray(utilities,dtype=np.float64)).sum(-1)+bias
def unit():
    a=np.zeros((1,40,3));a[:,:,0]=1;b=np.zeros_like(a);b[:,:,1]=1
    u=np.tile([1,2,3],(1,40,1));v=np.tile([4,-1,2],(1,40,1));bias=np.zeros(40)
    aa=compose(a,u,bias);ba=compose(b,u,bias);ab=compose(a,v,bias);bb=compose(b,v,bias)
    assert np.array_equal(aa,np.ones((1,40))) and np.array_equal(ba,np.full((1,40),2))
    allocation=(aa-ba+ab-bb)/2;value=(aa-ab+ba-bb)/2
    assert np.array_equal(aa-bb,allocation+value)
    assert np.array_equal(compose(a,u,bias),compose(a.copy(),u,bias))
    try:compose(a*2,u,bias)
    except ValueError:pass
    else:raise AssertionError('Invalid normalization accepted')
    assert np.allclose(ce(np.zeros((1,40)),np.array([0])),np.log(40))
    # A no-op transplant must share the original estimand's reduction path.
    # fp32 head sums can reverse an extremely close tie versus compose64.
    tie=np.full((1,40,3),np.float32(1/3));tu=np.zeros((1,40,3),np.float32)
    tu[0,1]=[1,1e-7,-1];tb=np.full(40,-1.,np.float32);tb[0]=3.1e-8;tb[1]=0
    native=(tie*tu).sum(-1)+tb;common=compose(tie,tu,tb)
    assert native.argmax()==0 and common.argmax()==1
    assert np.array_equal(common,compose(tie.copy(),tu,tb))
    rng=np.random.default_rng(20261115);sa=rng.uniform(size=(1,40,37));sa/=sa.sum(-1,keepdims=True);su=rng.normal(size=sa.shape)
    assert np.array_equal(compose(sa,su,bias),compose(np.asfortranarray(sa),np.asfortranarray(su),bias))
    return dict(status='PASS',cases=7,scope='Replay direction, Shapley identity, normalization, identity, CE, near-tie no-op and strided-storage arithmetic')

def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    run=sub.add_parser('run');run.add_argument('--output',type=Path,required=True)
    value=sub.add_parser('value');value.add_argument('--root',type=Path,required=True);value.add_argument('--trace-kind',choices=['base','intervention'],default='base');value.add_argument('--sample',required=True);value.add_argument('--representation',required=True);value.add_argument('--coords',required=True)
    args=parser.parse_args()
    if args.action=='value':
        payload=json.loads((args.root/('sample-trace.json' if args.trace_kind=='base' else 'intervention-trace.json')).read_bytes());s=next(x for x in payload['samples'] if x['sample_id']==args.sample)
        print(json.dumps(traces.read_value(args.root,s,args.representation,json.loads(args.coords))));return
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    assert torch.__version__=='2.5.1+cpu' and np.__version__=='1.26.4' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    src=Path(__file__).resolve().parent;repo=src.parents[1];cfg=json.loads((src/'contract.json').read_bytes())
    git=lambda *a:subprocess.check_output(['git','-C',str(repo),*a],stderr=subprocess.DEVNULL).decode().strip()
    assert not git('status','--porcelain=v1')
    out=args.output.resolve();assert not out.is_relative_to(repo) and not out.is_symlink()
    out.mkdir(parents=True,exist_ok=True);assert not any(out.iterdir())
    began=time.monotonic();checks=unit();save(out/'unit.json',checks)
    for x in cfg['inputs'].values():assert Path(x['path']).is_file() and not Path(x['path']).is_symlink() and sha(x['path'])==x['sha256']
    m=json.loads(Path(cfg['inputs']['cache_manifest']['path']).read_bytes())
    assert m['counts']=={'train':2048,'valid':1024} and m['encoder_revision']==cfg['encoder_revision']
    assert m['features']['sha256']==cfg['inputs']['cache']['sha256'] and not m['test_loaded']
    inputs=Path(cfg['head_directory']);models={};identities={}
    for mode in MODES:
        row=cfg['heads'][mode];p=inputs/(mode+'-selected.pt');assert p.is_file() and not p.is_symlink() and sha(p)==row['sha256']
        h=RMEAHead(mode);h.load_state_dict(torch.load(p,map_location='cpu',weights_only=True),strict=True);h.eval()
        assert sum(t.numel() for t in h.parameters())==row['parameters']
        models[mode]=h;identities[mode]=dict(sha256=row['sha256'],parameters=row['parameters'])
    # Original two selected-Q validation anchors use their original saved
    # features; this verifies head replay, NOT CPU/GPU encoder equivalence.
    anchor_residual=0.
    with torch.inference_mode():
        for x in cfg['anchors']:
            p=Path(x['path']);assert sha(p)==x['sha256']
            with np.load(p,allow_pickle=False) as f:d={k:f[k] for k in ['h','roles','mask','logits']}
            actual=models['quantum'](*(torch.from_numpy(d[k]) for k in ('h','roles','mask')))[0].numpy()
            np.testing.assert_allclose(actual,d['logits'],atol=5e-5,rtol=3e-4)
            anchor_residual=max(anchor_residual,float(np.max(np.abs(actual-d['logits']))))
    with np.load(cfg['inputs']['cache']['path'],allow_pickle=False) as f:
        assert not any(k.startswith('test') for k in f.files)
        data={k:f['valid_'+k].copy() for k in ['features','roles','offsets','targets','ids']}
    offsets=data['offsets'];y=data['targets'];ids=[x.decode('ascii') for x in data['ids']]
    assert len(y)==1024 and len(set(ids))==1024 and len(offsets)==1025 and offsets[0]==0 and offsets[-1]==len(data['features'])
    assert data['features'].dtype==np.float16 and data['roles'].dtype==np.float32 and np.all(np.diff(offsets)>0)
    assert y.min()>=0 and y.max()<40
    Z={m:[] for m in MODES};native_Z={m:[] for m in MODES};Q_by_donor={m:[] for m in MODES if m!='quantum'};D_by_Q={m:[] for m in Q_by_donor}
    distances={m:[] for m in Q_by_donor};maximum_identity=0.;maximum_shapley=0.
    with torch.inference_mode():
        for i in range(1024):
            lo,hi=map(int,offsets[i:i+2]);h=torch.from_numpy(data['features'][lo:hi].astype(np.float32))[None]
            roles=torch.from_numpy(data['roles'][lo:hi])[None];mask=torch.ones((1,hi-lo),dtype=torch.bool)
            # No target or outcome enters any head or intervention.
            d={m:models[m](h,roles,mask,details=True) for m in MODES}
            a={m:v['alpha'].numpy() for m,v in d.items()};u={m:v['utilities'].numpy() for m,v in d.items()}
            z={m:v['logits'].numpy() for m,v in d.items()};b={m:models[m].bias.detach().numpy() for m in MODES}
            for mode in MODES:
                residual=float(np.max(np.abs(compose(a[mode],u[mode],b[mode])-z[mode])))
                assert residual<2e-4;maximum_identity=max(maximum_identity,residual)
                Z[mode].append(compose(a[mode],u[mode],b[mode])[0]);native_Z[mode].append(z[mode][0])
            for mode in Q_by_donor:
                cq=compose(a[mode],u['quantum'],b['quantum']);qc=compose(a['quantum'],u[mode],b[mode])
                Q_by_donor[mode].append(cq[0]);D_by_Q[mode].append(qc[0])
                qq=compose(a['quantum'],u['quantum'],b['quantum']);cc=compose(a[mode],u[mode],b[mode])
                alloc=(qq-cq+qc-cc)/2;val=(qq-qc+cq-cc)/2
                residual=float(np.max(np.abs(qq-cc-alloc-val)));assert residual<1e-11;maximum_shapley=max(maximum_shapley,residual)
                distances[mode].append(float(np.abs(a['quantum']-a[mode]).sum()/40))
    Z={m:np.array(v) for m,v in Z.items()};Q_by_donor={m:np.array(v) for m,v in Q_by_donor.items()};D_by_Q={m:np.array(v) for m,v in D_by_Q.items()}
    native_Z={m:np.array(v) for m,v in native_Z.items()}
    numeric_parity={m:dict(native_metrics=metrics(native_Z[m],y),common_metrics=metrics(Z[m],y),
        argmax_disagreements=int((native_Z[m].argmax(-1)!=Z[m].argmax(-1)).sum()),
        native_min_top2_margin=float(np.diff(np.sort(native_Z[m],axis=-1)[:,-2:],axis=-1).min()),
        common_min_top2_margin=float(np.diff(np.sort(Z[m],axis=-1)[:,-2:],axis=-1).min())) for m in MODES}
    correct={m:Z[m].argmax(-1)==y for m in MODES};q=correct['quantum'];summary={};selected=set()
    for m in Q_by_donor:
        c=correct[m];qi=Q_by_donor[m].argmax(-1)==y;ci=D_by_Q[m].argmax(-1)==y
        cells={}
        for qok,cok in [(False,True),(True,False),(True,True),(False,False)]:
            eligible=[i for i in range(1024) if q[i]==qok and c[i]==cok]
            key='q'+str(int(qok))+'_control'+str(int(cok));cells[key]=len(eligible)
            if eligible:selected.add(min(eligible,key=lambda i:ids[i]))
        summary[m]=dict(original=metrics(Z[m],y),q_values_donor_allocation=metrics(Q_by_donor[m],y),control_values_q_allocation=metrics(D_by_Q[m],y),
            correctness_cells=cells,q_rescued=int((~q&qi).sum()),q_harmed=int((q&~qi).sum()),
            control_rescued=int((~c&ci).sum()),control_harmed=int((c&~ci).sum()),
            q_errors_with_correct_control=int((~q&c).sum()),q_errors_with_correct_control_rescued=int((~q&c&qi).sum()),
            allocation_L1_mean=float(np.mean(distances[m])),q_ce_change=float(np.mean(ce(Q_by_donor[m],y)-ce(Z['quantum'],y))))
    selected=sorted(selected,key=lambda i:ids[i]);assert len(selected)<=16
    package=out/'private';package.mkdir();(package/'checkpoints').mkdir()
    for mode in MODES:shutil.copyfile(inputs/(mode+'-selected.pt'),package/'checkpoints'/(mode+'-selected.pt'))
    length=max(int(offsets[i+1]-offsets[i]) for i in selected);h=np.zeros((len(selected),length,1024),np.float32);r=np.zeros((len(selected),length,4),np.float32);mask=np.zeros((len(selected),length),bool)
    for j,i in enumerate(selected):
        lo,hi=map(int,offsets[i:i+2]);h[j,:hi-lo]=data['features'][lo:hi];r[j,:hi-lo]=data['roles'][lo:hi];mask[j,:hi-lo]=True
    np.savez_compressed(package/'inputs.npz',h=h,roles=r,mask=mask)
    manifest=dict(schema='q-rmea.allocation.selected-replay.v1',lane='replay',scope=cfg['scope'],source_revision=git('rev-parse','HEAD'),source_files=traces.source_files(),
        sample_selection_rule=cfg['trace_selection'],samples=[dict(sample_id=ids[i],split='valid') for i in selected],inputs=traces.ref(package,package/'inputs.npz'),
        models=[dict(variant=m,selection_rule=cfg['head_selection'],checkpoint=traces.ref(package,package/'checkpoints'/(m+'-selected.pt'))) for m in MODES])
    save(package/'replay-manifest.json',manifest);payload,warnings=traces.replay(package,manifest);validation=traces.audit(package,manifest,payload)
    expected={}
    for mode in Q_by_donor:
        for i in selected:
            expected[(mode,'quantum',ids[i])]=Q_by_donor[mode][i]
            expected[('quantum',mode,ids[i])]=D_by_Q[mode][i]
    hybrid_validation=hybrid_trace.capture(package,payload,expected)
    validation['interventions']=hybrid_validation
    # Save target/outcome context separately from producer head inputs.
    save(package/'posthoc-case-selection.json',dict(rule=cfg['trace_selection'],sample_indices=selected,target_by_id={ids[i]:int(y[i]) for i in selected},comparison=summary))
    save(out/'trace-validation.json',validation)
    result=dict(schema='qattn.allocation-audit.descriptive.v1',status='PASS',scope=cfg['scope'],quantum=metrics(Z['quantum'],y),comparisons=summary,
        source_revision=git('rev-parse','HEAD'),contract_sha256=sha(src/'contract.json'),input_identities=identities,
        cache_sha256=cfg['inputs']['cache']['sha256'],sample_ids_sha256=hashlib.sha256(b''.join(data['ids'])).hexdigest(),
        anchor_max_logit_residual=anchor_residual,maximum_logit_identity_residual=maximum_identity,maximum_shapley_identity_residual=maximum_shapley,
        arithmetic='All original/intervention estimands use identical float64 compose; native fp32 head metrics separately reported.',numeric_parity=numeric_parity,
        trace_cases=len(selected),trace_model_cases=len(payload['samples']),trace_validation='PASS',elapsed_seconds=time.monotonic()-began,
        training_runs=0,formal_runs=0,test_loaded=False,cos_reads=0,
        limitation='Posthoc descriptive CPU-rebuilt validation cohort. Cross-head swaps are off-family intermediate interventions with coadapted utilities, not training mediation, task benefit, novelty or L5 confirmation. No p-values or formal pass threshold.')
    save(out/'summary.json',result)
    files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()};assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<128*1024**2
    save(out/'RUN_COMPLETE',dict(schema='qattn.allocation-audit.closed.v1',files=files))
    print(json.dumps(dict(status='COMPLETE',cases=1024,trace_cases=len(selected),training_runs=0,formal_runs=0)))

if __name__=='__main__':main()
