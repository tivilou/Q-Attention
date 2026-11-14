"""One finite CPU development diagnostic, never a complete-data formal run."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path

def save(p,x):
    tmp=p.with_suffix('.pending');tmp.write_text(json.dumps(x,separators=(',',':'))+'\n',encoding='utf8');os.replace(tmp,p)

def main():
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--exclusions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    import numpy as np
    import torch
    from data import load_cache,batch,sha
    from spectrum import SpectrumHead,VARIANTS
    from unit_checks import run_checks
    spec=json.loads((Path(__file__).parent/'contract.json').read_text())
    assert spec['seed']==419 and spec['epochs']==20 and spec['batch_size']==32 and spec['variants']==list(VARIANTS)
    repo=Path(__file__).resolve().parents[2]
    git=lambda *a:subprocess.check_output(['git','-C',str(repo),*a],text=True).strip()
    assert not git('status','--porcelain=v1') and os.environ.get('CUDA_VISIBLE_DEVICES','')==''
    assert torch.__version__=='2.5.1+cpu' and np.__version__=='1.26.4'
    out=args.output.resolve();assert not out.is_relative_to(repo) and not out.is_symlink()
    out.mkdir(parents=True,exist_ok=True);assert not any(out.iterdir())
    torch.use_deterministic_algorithms(True);started=time.monotonic()
    unit=run_checks();save(out/'unit.json',unit)
    d,indices,cohort=load_cache(args.cache,args.exclusions,spec['exclusions_sha256']);save(out/'cohort.json',cohort)
    torch.set_num_threads(4);(out/'checkpoints').mkdir()
    seed=419;epochs=20;size=32;results={};initialization={}

    def evaluate(head):
        head.eval();cm=np.zeros((40,40),np.int64);total_loss=0.
        with torch.no_grad():
            for start in range(0,len(indices['valid']),size):
                ind=indices['valid'][start:start+size];inputs,y=batch(d,'valid',ind);logits,alpha=head(*inputs)
                assert torch.isfinite(logits).all() and torch.isfinite(alpha).all()
                assert torch.allclose(alpha.sum(-1),torch.ones_like(alpha.sum(-1)),atol=2e-5,rtol=0)
                total_loss+=torch.nn.functional.cross_entropy(logits,y,reduction='sum').item()
                for a,b in zip(y.tolist(),logits.argmax(-1).tolist()):cm[a,b]+=1
        denom=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),denom,out=np.zeros(40),where=denom>0)
        return dict(macro_f1=float(f1.mean()),accuracy=float(np.trace(cm)/cm.sum()),loss=total_loss/cm.sum(),confusion=cm.tolist())

    for variant in VARIANTS:
        torch.manual_seed(seed);head=SpectrumHead(variant)
        initialization[variant]=hashlib.sha256(b''.join(t.detach().numpy().tobytes() for n,t in head.named_parameters() if n!='spectrum_logits')).hexdigest()
        optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
        history=[];best=None;selected_epoch=None;spectral_gradient_max=0.
        for epoch in range(epochs):
            head.train();order=torch.randperm(len(indices['train']),generator=torch.Generator().manual_seed(seed+epoch)).numpy();train_loss=0.
            for start in range(0,len(order),size):
                ind=indices['train'][order[start:start+size]];inputs,y=batch(d,'train',ind)
                optimizer.zero_grad(set_to_none=True);logits,_=head(*inputs)
                loss=torch.nn.functional.cross_entropy(logits,y);assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
                if variant!='static':spectral_gradient_max=max(spectral_gradient_max,float(head.spectrum_logits.grad.abs().max()))
                optimizer.step();train_loss+=float(loss.detach())*len(ind)
            valid=evaluate(head);history.append(dict(epoch=epoch+1,train_loss=train_loss/len(order),valid=valid))
            if best is None or valid['macro_f1']>best['macro_f1']:
                best=valid;selected_epoch=epoch+1
                tmp=out/'checkpoints'/f'{variant}.pending';torch.save(head.state_dict(),tmp);os.replace(tmp,out/'checkpoints'/f'{variant}-selected.pt')
            print(json.dumps(dict(variant=variant,epoch=epoch+1,valid_macro=valid['macro_f1'],train_loss=train_loss/len(order))),flush=True)
        checkpoint=out/'checkpoints'/f'{variant}-selected.pt'
        head.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        selected=evaluate(head);assert selected['confusion']==best['confusion'] and abs(selected['loss']-best['loss'])<1e-10
        spectrum={}
        if variant!='static':
            weights=head.spectrum_logits.softmax(-1).detach().numpy()
            spectrum=dict(distribution_kind='Bernstein coefficients' if variant=='scalar_calibration' else 'measurement eigenweights',mean_entropy=float((-weights*np.log(weights)).sum(-1).mean()),mean_max_weight=float(weights.max(-1).mean()),gradient_max=spectral_gradient_max)
        results[variant]=dict(history=history,selected_epoch=selected_epoch,selected=selected,selected_checkpoint_sha256=sha(checkpoint),parameter_count=sum(t.numel() for t in head.parameters()),updates=epochs*((len(order)+size-1)//size),spectrum=spectrum)
        del head,optimizer
    assert len(set(initialization.values()))==1
    main=results['spectral']['selected']['macro_f1'];base=results['static']['selected']['macro_f1'];cal=results['scalar_calibration']['selected']['macro_f1']
    gate=main>base and main>cal
    summary=dict(schema='qattn.spectral.bounded-development.v1',status='PASS',seed=seed,epochs=epochs,batch_size=size,
        variants=list(VARIANTS),results=results,common_initialization=initialization,cohort_sha256=sha(out/'cohort.json'),
        source_revision=git('rev-parse','HEAD'),source_sha256=sha(Path(__file__)),contract_sha256=sha(Path(__file__).parent/'contract.json'),
        optimizer=dict(name='AdamW',lr=.001,weight_decay=.01,clip_norm=1.),elapsed_seconds=time.monotonic()-started,
        gate=dict(status='PASS' if gate else 'NOT_PASSED',spectral_minus_static=main-base,spectral_minus_scalar=main-cal,
            scope='Positive development deltas permit formal single-seed PREPARATION only; negative/short null stops this attempt priority, not trained-family disproof.'),
        test_loaded=False,formal_runs=0,cos_bytes=0,training_recovery='Restart from pinned inputs only; no certified exact resume or automatic retry.',
        limitations=['Single-seed reused development cache; no independent confirmatory test or utility evidence tier','Nominal/softmax degrees match spectral and scalar but functional classes differ','Dephased readout still allows phase-dependent basis construction; not dephasing every operation','Early selected epochs or end-budget improvement must be considered before diagnosing convergence','No quantum-exclusive attribution or hardware advantage'])
    save(out/'summary.json',summary)
    closed=dict(schema='qattn.spectral.bounded.closed.v1',source_revision=summary['source_revision'],files={str(f.relative_to(out)).replace('\\','/'):sha(f) for f in sorted(out.rglob('*')) if f.is_file()})
    save(out/'RUN_COMPLETE',closed)
    print(json.dumps(dict(status='COMPLETE',gate=summary['gate'],elapsed_seconds=summary['elapsed_seconds'])))

if __name__=='__main__':main()
