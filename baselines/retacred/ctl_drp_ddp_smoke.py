"""Explicit <=2-update private CUDA canary; no upload or formal comparison."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def install_fault(adapter,rank):
    from . import ctl_drp_ddp
    import torch
    used=[False]
    def hook(*_):
        context=ctl_drp_ddp.CURRENT
        if rank==context.world_size-1 and context.updates_seen==2 and not used[0]:
            used[0]=True
            torch.empty(torch.cuda.get_device_properties(rank).total_memory*2,dtype=torch.uint8,device=f'cuda:{rank}')
    adapter.model.register_forward_pre_hook(hook)


def main(argv=None):
    import torch
    from . import ctl_drp_ddp as distributed
    from .ctl_drp_runner import run,synthetic_receipt,seed_runtime
    from . import ctl_drp_runner as runner
    from .ctl_drp_task import CTLDRPTask
    from .common import atomic_json
    from .common import digest
    runner.git=lambda *_:'bounded-diagnostic-source-sha256:'+digest(Path(__file__).with_name('ctl_drp_ddp.py'))
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True);p.add_argument('--paper',required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--mode',choices=('numeric','lifecycle','oom','adaptive','pretrained','observer-error','completed-retry'),default='numeric')
    p.add_argument('--weight');p.add_argument('--variant',choices=('sup','unsup'),default='sup')
    a=p.parse_args(argv)
    ctx=distributed.CURRENT
    if ctx is None:raise ValueError('launch with ctl_drp_ddp --entry-module baselines.retacred.ctl_drp_ddp_smoke')
    a.run_dir.mkdir(parents=True,exist_ok=True)
    factory=dict(tiny=True,receipt=str(synthetic_receipt(a.run_dir/'inputs')),cache=a.cache,paper=a.paper,
                 seed=13,micro=1,dropout=0. if a.mode=='numeric' else .1)
    if a.mode=='pretrained':
        if not a.weight:p.error('pretrained mode requires a pinned weight')
        factory.update(tiny=False,bounded_pretrained=True,weight=a.weight,variant=a.variant)
    def bind(f):
        adapter=distributed.make_adapter(f,0)
        ctx.bind(adapter,f)
        return adapter
    if a.mode=='adaptive':
        from types import SimpleNamespace
        from .reference_execution import adaptive_cli,attach
        options=SimpleNamespace(run_dir=a.run_dir,micro_batch='2',seed=13,checkpoint_seconds=0.)
        def operation():
            f=dict(factory,micro=options.micro_batch,test_oom=options.execution_state['index']==0)
            adapter=bind(f)
            import time
            stamp=a.run_dir/'ddp_starts.json'
            times=json.loads(stamp.read_text()) if stamp.exists() else []
            atomic_json(stamp,times+[time.time()])
            attach(adapter,options)
            adapter.contract.pop('physical_micro_batch_sentences')
            adapter.contract.pop('execution_memory')
            native=a.run_dir/'recover'
            result=run(adapter,native,resume=(native/'contract.json').exists())
            assert result['updates']==2
            atomic_json(a.run_dir/'receipt.json',dict(status='passed',mode=a.mode,world_size=ctx.world_size,
                actual_cuda_oom=True,tier=options.execution_state['index'],complete_train_test_values=True,updates=2))
            return 0
        return adaptive_cli(options,'ctl_drp_supervised',512,operation)
    if a.mode=='observer-error':
        from contextlib import nullcontext
        adapter=bind(factory);optimizer=adapter.make_optimizer()
        class Observer:
            index=0
            def context(self,*_):return nullcontext()
            def forward(self,*_):pass
            def post_update(self):raise ValueError('injected post-update trace failure')
        try:adapter.train_update([0,1,2],optimizer,observer=Observer())
        except RuntimeError as error:
            if isinstance(error,torch.cuda.OutOfMemoryError) or 'injected post-update trace failure' not in str(error):raise
        else:raise AssertionError('coordinated non-OOM trace error not reported')
        atomic_json(a.run_dir/'receipt.json',dict(status='passed',mode=a.mode,world_size=ctx.world_size,
            non_oom_error_coordinated=True))
        return 0
    if a.mode=='numeric':
        errors=[]
        for indices in ([0,1,2],[0]):
            baseline=distributed.make_adapter(factory,0)
            adapter=bind(factory)
            opt=adapter.make_optimizer();ref_opt=baseline.make_optimizer()
            reference=baseline.train_update(indices,ref_opt)
            actual=adapter.train_update(indices,opt)
            assert abs(actual['loss']-reference['loss'])<2e-6,(actual,reference)
            maximum=0.;grad_max=0.
            for (name,param),(other,ref) in zip(adapter.model.named_parameters(),baseline.model.named_parameters()):
                assert name==other
                maximum=max(maximum,float((param-ref).abs().max()))
                grad_max=max(grad_max,float((param.grad-ref.grad).abs().max()))
                assert torch.allclose(param,ref,atol=2e-6,rtol=2e-5),name
                assert torch.allclose(param.grad,ref.grad,atol=2e-6,rtol=2e-4),name
            errors.append(dict(sentences=len(indices),max_weight_error=maximum,max_gradient_error=grad_max))
        result=dict(status='passed',mode=a.mode,world_size=ctx.world_size,comparisons=errors)
    elif a.mode in ('lifecycle','pretrained','completed-retry'):
        adapter=bind(factory)
        result=run(adapter,a.run_dir/'full',diagnostic_budget=2)
        assert result['updates']==2
        if a.mode=='lifecycle':
            partial=bind(factory)
            paused=run(partial,a.run_dir/'resume',pause_after=1)
            assert paused['status']=='paused'
            restored=bind(factory)
            resumed=run(restored,a.run_dir/'resume',resume=True)
            def state(root):
                pointer=json.loads((root/'latest.json').read_text())
                return torch.load(root/'checkpoints'/pointer['generation']/'state.pt',weights_only=True,map_location='cpu')
            full=state(a.run_dir/'full');continued=state(a.run_dir/'resume')
            from .ctl_drp_cuda_smoke import state_hash
            for key in ('model','optimizer','rank_rng'):
                assert state_hash(full[key])==state_hash(continued[key]),key
            assert result['splits']==resumed['splits']
            result=dict(status='passed',mode=a.mode,world_size=ctx.world_size,
                        complete_train_test_values=True,exact_resume=True,updates=2)
        elif a.mode=='completed-retry':
            before={p.relative_to(a.run_dir/'full').as_posix():digest(p)
                for p in (a.run_dir/'full').rglob('*') if p.is_file()}
            fresh=bind(factory)
            def forbidden(*_a,**_k):raise AssertionError('completed retry repeated model work')
            fresh.make_optimizer=forbidden;fresh.train_update=forbidden;fresh.evaluate=forbidden
            repeated=run(fresh,a.run_dir/'full',resume=True)
            after={p.relative_to(a.run_dir/'full').as_posix():digest(p)
                for p in (a.run_dir/'full').rglob('*') if p.is_file()}
            assert repeated==result and before==after
            assert not (a.run_dir/'full/orphaned_trace_attempts').exists()
            result=dict(status='passed',mode=a.mode,world_size=ctx.world_size,updates=2,
                complete_train_test_values=True,completed_retry_without_model_work=True,committed_trace_hashes_unchanged=True)
        else:
            result=dict(status='passed',mode=a.mode,world_size=ctx.world_size,
                pretrained_identity=adapter.initialization,updates=2,complete_train_test_values=True)
    else:
        faulty=dict(factory,test_oom=True)
        adapter=bind(faulty)
        try:
            run(adapter,a.run_dir/'recover')
        except torch.cuda.OutOfMemoryError as exc:
            assert 'rank' in str(exc)
        else:raise AssertionError('actual rank-local OOM was not propagated')
        pointer=json.loads((a.run_dir/'recover/latest.json').read_text())
        latest=torch.load(a.run_dir/'recover/checkpoints'/pointer['generation']/'state.pt',map_location='cpu',weights_only=True)
        assert latest['cursor']==1
        adapter=bind(factory)
        recovered=run(adapter,a.run_dir/'recover',resume=True)
        assert recovered['updates']==2
        result=dict(status='passed',mode=a.mode,world_size=ctx.world_size,replayed_from_cursor=1,
                    actual_cuda_oom=True,complete_train_test_values=True)
    atomic_json(a.run_dir/'receipt.json',result)
    print(json.dumps(result),flush=True)
    return 0
