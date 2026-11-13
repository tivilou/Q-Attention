"""CPU train/valid-only joint-head diagnostic; never a formal experiment."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from baselines.retacred.common import atomic_json,digest

def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()

def train(args):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from baselines.retacred.common import metrics
    from baselines.retacred.rmea_joint import VARIANTS,make_head
    if git('status','--porcelain') or os.environ.get('CUDA_VISIBLE_DEVICES','')!='':raise ValueError('Clean CPU-only source required')
    counts=dict(train=4,valid=4) if args.profile=='canary' else dict(train=2048,valid=1024)
    epochs=1 if args.profile=='canary' else 4
    cache=args.cache.resolve();manifest=json.loads((cache/'cache-manifest.json').read_text());validation=json.loads((cache/'cache-validation.json').read_text());closed=json.loads((cache/'RUN_COMPLETE').read_text())
    if digest(cache/'cache-manifest.json')!=args.cache_manifest_sha256:raise ValueError('Pinned feature manifest differs')
    if args.cache.is_symlink() or any((cache/n).is_symlink() or not (cache/n).is_file() for n in ('features.npz','cache-manifest.json','cache-validation.json','RUN_COMPLETE')):raise ValueError('Regular closed cache files required')
    if digest(cache/'features.npz')!=args.cache_sha256:raise ValueError('Pinned feature cache differs')
    if manifest['features']!=dict(sha256=args.cache_sha256,bytes=(cache/'features.npz').stat().st_size):raise ValueError('Feature reference differs')
    audit=ROOT/'docs/evidence/rmea/20261026-readiness/input-audit.json'
    if digest(audit)!='c21d23017140d89a3632853d0f33d9e76c5cb7d574d4a9108b63d1ed23bf0996' or manifest['labels']!=json.loads(audit.read_text())['train_derived_labels']:raise ValueError('Train-derived label order differs')
    if closed!=dict(schema='qattn.rebuilt-cache.closed.v1',files={n:digest(cache/n) for n in ('features.npz','cache-manifest.json')}):raise ValueError('Cache not closed')
    if validation['status']!='passed' or validation['manifest_sha256']!=digest(cache/'cache-manifest.json') or validation['features_sha256']!=args.cache_sha256:raise ValueError('Cache audit mismatch')
    if manifest['counts']!=counts or manifest['test_loaded'] or manifest['head_training'] or manifest['source_revision']!='0e25cf2309cfb1cc406e6b96521c2a29671a1048':raise ValueError('Input workload/provenance differs')
    if manifest['schema']!='qattn.rmea.rebuilt-features.v1' or manifest['profile']!=('preflight' if args.profile=='canary' else 'rebuild') or manifest['producer_sha256']!='b6ccef2e3634af7e3dbe1c7f2e97914c1a5c40bbb837b2d45a060c850b811402':raise ValueError('Producer/profile differs')
    if validation['source_revision']!=manifest['source_revision'] or validation['counts']!=counts or validation['profile']!=manifest['profile'] or validation['test_loaded'] or validation['head_training']:raise ValueError('Audit workload differs')
    expected_ids=dict(train='033bfcc598a349b1522b0734c7ceb230a1aefd598c072fe69629cc24107decda',valid='d3c434f423952896da3dfd84f79acc7aa76d78edf60e4552dbd8909527e1b9c1') if args.profile=='canary' else dict(train='f2df684905140be85bcff9dc3556864bfe89b52ab8c566865e86bf0c91db96c5',valid='473ccfbb31f9831a3fe5413a4a0147630a756cdbfd8edc3a73acceadf9820b28')
    if manifest['sample_id_hashes']!=expected_ids or manifest['formal_runs']!=0 or manifest['device']!='cpu' or manifest['attention']!='eager':raise ValueError('Exact cache membership/device differs')
    if manifest['source_data']!=dict(train='f1051d574a44df3ecbd036c1de4fbbe0a372c768f6ec990330e2cea4a657269f',valid='9d6965f984d7485e2e81d380a0b0de2922d3da1e2685393ea3e46869805c2b55'):raise ValueError('Raw input identities differ')
    if torch.__version__!='2.5.1+cpu' or np.__version__!='1.26.4':raise ValueError('Pinned numerical runtime differs')
    if manifest['dtype']!='float32 CPU encoder and word-piece mean; saved features float16; roles float32':raise ValueError('CPU profile differs')
    out=args.run_dir.resolve()
    if out.exists() or out.is_relative_to(ROOT):raise ValueError('Fresh private output outside source required')
    out.mkdir(parents=True);(out/'checkpoints').mkdir();started=time.monotonic()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    seed=307;labels=manifest['labels']
    if len(labels)!=40 or len(set(labels))!=40:raise ValueError('Forty labels required')
    with np.load(cache/'features.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
    if set(data)!={s+'_'+k for s in counts for k in ('features','roles','offsets','targets','ids')}:raise ValueError('Exact input inventory required')
    for split,n in counts.items():
        h,roles,offset,y,ids=[data[split+'_'+k] for k in ('features','roles','offsets','targets','ids')]
        if h.dtype!=np.float16 or roles.dtype!=np.float32 or offset.dtype!=np.int64 or y.dtype!=np.int64 or ids.dtype!=np.dtype('S64'):raise ValueError('Cache dtype mismatch')
        if offset.shape!=(n+1,) or offset[0]!=0 or offset[-1]!=len(h) or not (np.diff(offset)>0).all() or h.shape!=(len(h),1024) or roles.shape!=(len(h),4) or y.shape!=(n,) or ids.shape!=(n,):raise ValueError('Cache shape mismatch')
        if not np.isfinite(h).all() or not np.isfinite(roles).all() or not ((y>=0)&(y<40)).all():raise ValueError('Nonfinite input or invalid target')
        if hashlib.sha256(b''.join(ids)).hexdigest()!=manifest['sample_id_hashes'][split]:raise ValueError('Sample IDs differ')

    def batch(split,indices,targets=True):
        offsets=data[split+'_offsets'];lengths=[int(offsets[i+1]-offsets[i]) for i in indices];size=max(lengths)
        h=np.zeros((len(indices),size,1024),np.float32);roles=np.zeros((len(indices),size,4),np.float32);mask=np.zeros((len(indices),size),bool)
        for j,(i,n) in enumerate(zip(indices,lengths)):
            a,b=map(int,offsets[i:i+2]);h[j,:n]=data[split+'_features'][a:b];roles[j,:n]=data[split+'_roles'][a:b];mask[j,:n]=True
        return tuple(torch.from_numpy(x) for x in (h,roles,mask)),torch.from_numpy(data[split+'_targets'][indices].copy()) if targets else None

    def batches(split):
        for start in range(0,counts[split],32):yield list(range(start,min(start+32,counts[split])))

    def evaluate(head):
        head.eval();gold=[];predicted=[];loss=0.
        with torch.no_grad():
            for indices in batches('valid'):
                inputs,y=batch('valid',indices);logits,alpha=head(*inputs)
                if not torch.isfinite(logits).all() or not torch.isfinite(alpha).all() or not torch.allclose(alpha.sum(-1),torch.ones_like(alpha.sum(-1)),atol=2e-5,rtol=0):raise ValueError('Invalid forward/attention')
                loss+=F.cross_entropy(logits,y,reduction='sum').item();gold+=y.tolist();predicted+=logits.argmax(-1).tolist()
        return metrics(gold,predicted,labels,loss/len(gold))

    results={};common={}
    for variant in VARIANTS:
        torch.manual_seed(seed);head=make_head(variant)
        if variant=='no_content':head.fit_role_means((batch('train',i,False)[0] for i in batches('train')),split='train')
        common[variant]=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in (head.value.weight,head.value.bias,head.decoder,head.bias))).hexdigest()
        optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
        history=[];best=None;selected_epoch=None
        for epoch in range(epochs):
            head.train();order=torch.randperm(counts['train'],generator=torch.Generator().manual_seed(seed+epoch)).tolist()
            for start in range(0,len(order),32):
                inputs,y=batch('train',order[start:start+32]);optimizer.zero_grad(set_to_none=True)
                loss=F.cross_entropy(head(*inputs)[0],y)
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            valid=evaluate(head);history.append(valid)
            if best is None or valid['macro_f1']>best['macro_f1']:
                best=valid;selected_epoch=epoch+1;torch.save(head.state_dict(),out/'checkpoints'/f'{variant}-selected.pt')
            # Latest closed epoch is separate from the immutable selected evaluation model.
            latest=out/'checkpoints'/f'{variant}-latest.pt';tmp=latest.with_suffix('.pending')
            torch.save(dict(model=head.state_dict(),optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),epoch=epoch+1,seed=seed,counts=counts,cache_sha256=args.cache_sha256,source_revision=git('rev-parse','HEAD'),selected_epoch=selected_epoch,selected_checkpoint_sha256=digest(out/'checkpoints'/f'{variant}-selected.pt')),tmp);os.replace(tmp,latest)
            print(json.dumps(dict(stage='joint_diagnostic',variant=variant,epoch=epoch+1,valid_macro_f1=valid['macro_f1'])),flush=True)
        head.load_state_dict(torch.load(out/'checkpoints'/f'{variant}-selected.pt',weights_only=True));selected=evaluate(head)
        if selected!=best:raise ValueError('Reloaded selected model metrics differ')
        results[variant]=dict(history=history,selected_epoch=selected_epoch,selected_valid=selected,selected_checkpoint_sha256=digest(out/'checkpoints'/f'{variant}-selected.pt'),parameter_count=sum(v.numel() for v in head.parameters()),updates=epochs*((counts['train']+31)//32))
        del head,optimizer
    if len(set(common.values()))!=1:raise ValueError('Common initialization differs')
    # Fixed first hash-ranked train/valid samples; selected after no result inspection.
    samples=[(s,0) for s in ('train','valid')];max_words=max(int(data[s+'_offsets'][i+1]-data[s+'_offsets'][i]) for s,i in samples)
    h=np.zeros((2,max_words,1024),np.float32);roles=np.zeros((2,max_words,4),np.float32);mask=np.zeros((2,max_words),bool)
    for j,(s,i) in enumerate(samples):
        a,b=map(int,data[s+'_offsets'][i:i+2]);h[j,:b-a]=data[s+'_features'][a:b];roles[j,:b-a]=data[s+'_roles'][a:b];mask[j,:b-a]=True
    np.savez_compressed(out/'inputs.npz',h=h,roles=roles,mask=mask)
    from baselines.retacred import rmea_joint_trace as trace
    replay=dict(schema='q-rmea.joint.selected-replay.v1',lane='replay',scope='bounded train/valid diagnostic selected checkpoints; no held-out test or formal utility claim',source_revision=git('rev-parse','HEAD'),source_files=trace.source_files(),sample_selection_rule='first readiness-v1 hash-ranked train and valid examples, fixed before training',samples=[dict(sample_id=s+':'+data[s+'_ids'][i].decode(),split=s) for s,i in samples],inputs=trace.ref(out,out/'inputs.npz'),models=[dict(variant=v,checkpoint=trace.ref(out,out/'checkpoints'/f'{v}-selected.pt'),selection_rule='strict greatest validation full40 macro-F1; earliest tie') for v in VARIANTS])
    atomic_json(out/'replay-manifest.json',replay)
    summary=dict(schema='qattn.joint.bounded-diagnostic.v1',scope='technical_train_valid_diagnostic_only',profile=args.profile,seed=seed,counts=counts,epochs=epochs,batch_size=32,optimizer=dict(name='AdamW',lr=.001,weight_decay=.01,clip_norm=1.),source_revision=git('rev-parse','HEAD'),producer_sha256=digest(Path(__file__)),cache_sha256=args.cache_sha256,cache_manifest_sha256=digest(cache/'cache-manifest.json'),labels=labels,results=results,common_initialization=common,elapsed_seconds=time.monotonic()-started,test_loaded=False,formal_runs=0,cos_protection=False,exact_continuation_implemented=False,training_recovery='Restart bounded diagnostic from pinned inputs; latest files are observations, not certified resume',limitations=['Not L3 or formal task utility evidence','Single seed and validation selection; no held-out test','Static eight-state versus joint four-state word register differs','Nominal parameter counts do not imply equal active capacity','CPU rebuilt feature cache is a new identity; no historical GPU equivalence'])
    atomic_json(out/'summary.json',summary)

def main():
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='4'
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['run','_train']);p.add_argument('--profile',choices=['canary','diagnostic'],required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--cache-sha256',required=True);p.add_argument('--cache-manifest-sha256',required=True);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args()
    if a.action=='_train':train(a);return
    # No Torch in the control parent; producer exits before independent replay.
    subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'_train',*sys.argv[2:]],check=True)
    subprocess.run([sys.executable,'-B',str(ROOT/'experiments/q_rmea_joint_replay.py'),'materialize','--run-dir',str(a.run_dir)],check=True)
    report=json.loads((a.run_dir/'trace-validation.json').read_text())
    if report['status']!='passed':raise ValueError('Selected-checkpoint replay failed')
    atomic_json(a.run_dir/'RUN_COMPLETE',dict(schema='qattn.joint.bounded-diagnostic.closed.v1',files={f.relative_to(a.run_dir).as_posix():digest(f) for f in sorted(a.run_dir.rglob('*')) if f.is_file()}))

if __name__=='__main__':main()
