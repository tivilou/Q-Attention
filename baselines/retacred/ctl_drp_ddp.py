"""Sentence-preserving CTL DDP, rank-complete recovery and bounded launcher.

Accumulation runs under DDP.no_sync: ragged/tail ranks may do different numbers
of forwards. After every rank reports success on the CPU control group, one
SUM reduction of globally normalized accumulated gradients precedes AdamW. In
particular, a local OOM cannot leave peers inside a backward gradient collective.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import signal
import tempfile

CURRENT = None


def partition(indices, world_size, selected=None):
    if not indices or len(indices) != len(set(indices)) or not 1 <= world_size <= 6:
        raise ValueError('nonempty unique sentence indices and world size1..6 required')
    parts = [list(indices[len(indices)*r//world_size:len(indices)*(r+1)//world_size]) for r in range(world_size)]
    if selected in indices and selected not in parts[0]:
        source = next(p for p in parts if selected in p)
        # Keep the sampled event on the coordinator without changing its
        # candidates, the global batch, or any sentence's objective weight.
        source.remove(selected)
        if parts[0]:
            source.append(parts[0].pop())
        parts[0].append(selected)
    return parts


def make_adapter(factory, rank):
    from .ctl_drp_task import CTLDRPTask
    if factory.get('tiny'):
        from .ctl_drp_neural import tiny_model, tokenizer_from_cache, descriptions_from_native
        from .ctl_drp_task import PhysicalSplits
        model = tiny_model(seed=factory['seed'], dropout=factory.get('dropout', .1))
        adapter=CTLDRPTask(PhysicalSplits(factory['receipt']), tokenizer_from_cache(factory['cache']),
            descriptions_from_native(factory['paper']), model, seed=factory['seed'],
            micro_batch=factory['micro'], device=f'cuda:{rank}')
        if factory.get('test_oom'):
            from .ctl_drp_ddp_smoke import install_fault
            install_fault(adapter,rank)
        return adapter
    if factory.get('bounded_pretrained'):
        from .ctl_drp_neural import tokenizer_from_cache, descriptions_from_native, joint_model
        from .ctl_drp_pretrained import load_encoder
        from .ctl_drp_task import PhysicalSplits
        data=PhysicalSplits(factory['receipt'])
        if not data.identity['synthetic'] or len(data.records['train'])>16:
            raise ValueError('pretrained canary requires explicitly synthetic <=16 sentence fixture')
        encoder,identity=load_encoder(factory['weight'],factory['cache'],variant=factory['variant'],marker_seed=factory['seed'])
        return CTLDRPTask(data,tokenizer_from_cache(factory['cache'],variant=factory['variant']),descriptions_from_native(factory['paper']),
            joint_model(encoder,head_seed=factory['seed'],memory_efficient=False),seed=factory['seed'],micro_batch=factory['micro'],
            device=f'cuda:{rank}',initialization=identity,diagnostic=True)
    return CTLDRPTask.from_assets(factory['receipt'], factory['cache'], factory['paper'], factory['weight'],
        variant=factory['variant'], seed=factory['seed'], micro_batch=factory['micro'], device=f'cuda:{rank}',
        memory_efficient=factory['checkpointing'], audit_cache=Path(factory['audit_cache']))


class Context:
    def __init__(self, rank, world_size, rendezvous):
        import torch
        import torch.distributed as dist
        self.rank, self.world_size = rank, world_size
        torch.cuda.set_device(rank)
        from .ctl_drp_runner import seed_runtime
        seed_runtime(13+rank)
        self.device = f'cuda:{rank}'
        dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=world_size,
                                timeout=timedelta(hours=24))
        self.grad_group = dist.new_group(backend='nccl', timeout=timedelta(minutes=30))
        self.adapter = self.optimizer = None
        self.closed = False

    def command(self, operation, **payload):
        import torch.distributed as dist
        if self.rank != 0:
            raise ValueError('only the coordinator issues commands')
        dist.broadcast_object_list([{'operation': operation, **payload}], src=0)

    def agree(self, error):
        import torch.distributed as dist
        errors = [None] * self.world_size
        dist.all_gather_object(errors, error)
        if any(errors):
            import torch
            text = '; '.join(f'rank{r}: {e[1]}' for r, e in enumerate(errors) if e)
            if any(e and e[0] == 'cuda_oom' for e in errors):
                raise torch.cuda.OutOfMemoryError(text)
            raise RuntimeError(text)

    @staticmethod
    def error(exc):
        import torch
        return ('cuda_oom' if isinstance(exc, torch.cuda.OutOfMemoryError) else type(exc).__name__, str(exc))

    def bind(self, adapter, factory):
        if self.rank != 0:
            raise ValueError('coordinator bind required')
        self.command('initialize', factory=factory)
        self._bind(adapter)

    def _bind(self, adapter):
        import torch
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel
        self.adapter = adapter
        self.updates_seen=0
        adapter.distributed = self
        self.original_optimizer = adapter.make_optimizer
        adapter.make_optimizer = self.make_optimizer
        adapter.train_update = self.train_update
        self.ddp = DistributedDataParallel(adapter.model, device_ids=[self.rank], output_device=self.rank,
            process_group=self.grad_group, broadcast_buffers=False, find_unused_parameters=False,
            gradient_as_bucket_view=True)
        p = torch.cuda.get_device_properties(self.rank)
        info = dict(device=p.name, uuid=str(getattr(p, 'uuid', 'not_exposed')),
                    capability=[p.major, p.minor], total_memory=p.total_memory)
        ranks = [None] * self.world_size
        dist.all_gather_object(ranks, info)
        self.runtime = dict(schema='ctl-ddp-runtime.v1', world_size=self.world_size, ranks=ranks,
            visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), torch=str(torch.__version__),
            cuda=torch.version.cuda, precision='FP32_no_autocast',
            reduction='DDP_no_sync_then_sum_global_normalized_gradients_before_AdamW', elastic_world_size=False)

    def make_optimizer(self):
        if self.rank == 0:
            self.command('optimizer')
        self.optimizer = self.original_optimizer()
        return self.optimizer

    def train_update(self, indices, optimizer, *, observer=None):
        selected = observer.index if observer is not None else None
        if self.rank == 0:
            self.command('update', indices=indices, selected=selected)
        return self._update(indices, selected, observer)

    def _update(self, indices, selected, observer=None):
        import torch
        import torch.distributed as dist
        from .ctl_drp_neural import collate_groups
        adapter, optimizer = self.adapter, self.optimizer
        self.updates_seen+=1
        if not 1 <= len(indices) <= 512:
            raise ValueError('global logical batch must remain1..512 sentences')
        local = partition(indices, self.world_size, selected)[self.rank]
        optimizer.zero_grad(set_to_none=True)
        adapter.model.train()
        value = 0.
        error = None
        try:
            groups = adapter.encode('train', local) if local else []
            with self.ddp.no_sync():
                for start in range(0, len(groups), adapter.micro_batch):
                    part = groups[start:start+adapter.micro_batch]
                    batch = collate_groups(part, device=self.device)
                    scale = len(part) / len(indices)
                    context = observer.context('train', part, batch, scale) if observer else nullcontext()
                    with context:
                        result = self.ddp(**batch, gold=[g['gold'] for g in part])
                        if observer:
                            observer.forward(result)
                        scaled = result['loss'] * scale
                        if not bool(torch.isfinite(scaled)):
                            raise ValueError('nonfinite CTL loss')
                    scaled.backward()
                    value += float(scaled.detach())
                    del result, scaled, batch
        except Exception as exc:
            error = self.error(exc)
        try:
            self.agree(error)
        except Exception:
            optimizer.zero_grad(set_to_none=True)
            raise
        # Equal tensor order on every rank, including ranks with an empty tail.
        # no_sync avoids automatic bucket reduction during a partial update.
        error = None
        try:
            for parameter in adapter.model.parameters():
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
        except Exception as exc:
            error = self.error(exc)
        self.agree(error)
        for parameter in adapter.model.parameters():
            dist.all_reduce(parameter.grad, group=self.grad_group)
        error = None
        try:
            optimizer.step()
        except Exception as exc:
            error = self.error(exc)
        self.agree(error)
        error = None
        try:
            if observer:
                observer.post_update()
        except Exception as exc:
            error = self.error(exc)
        self.agree(error)
        values = [None] * self.world_size
        dist.all_gather_object(values, dict(loss=value, sentences=len(local),
            allocated=torch.cuda.memory_allocated(self.rank), reserved=torch.cuda.memory_reserved(self.rank)))
        self.last_memory = values
        return dict(loss=sum(v['loss'] for v in values), sentences=len(indices), optimizer_updates=1,
                    world_size=self.world_size, rank_sentences=[v['sentences'] for v in values])

    def rng(self):
        import torch
        from .ctl_drp_cuda_checkpoint import numpy_rng
        return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(self.rank),
                    python=random.getstate(), numpy=numpy_rng())

    def save(self, path, optimizer, *, contract, cursor):
        import torch
        import torch.distributed as dist
        from .ctl_drp_cuda_checkpoint import cpu
        if self.rank == 0:
            self.command('save', cursor=cursor)
        if cursor == 0 and self.rank:
            from .ctl_drp_runner import seed_runtime
            seed_runtime(self.adapter.seed + self.rank)
        states = [None] * self.world_size
        dist.all_gather_object(states, self.rng())
        if self.rank:
            return
        payload = dict(schema='ctl-ddp-fp32.v1', runtime=self.runtime, contract=contract, cursor=cursor,
            phase='post_optimizer_update', model=cpu(self.adapter.model.state_dict()),
            optimizer=cpu(optimizer.state_dict()), training=self.adapter.model.training, rank_rng=states)
        with Path(path).open('xb') as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())

    def restore(self, path, optimizer, *, contract, cursor):
        import torch
        import numpy as np
        from .ctl_drp_cuda_checkpoint import decode_numpy_rng
        if self.rank == 0:
            self.command('restore', path=str(path), contract=contract, cursor=cursor)
        error = None
        try:
            payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            if (payload.get('schema') != 'ctl-ddp-fp32.v1' or payload.get('contract') != contract
                    or payload.get('runtime') != self.runtime or payload.get('cursor') != cursor
                    or payload.get('phase') != 'post_optimizer_update'
                    or type(payload.get('training')) is not bool
                    or len(payload.get('rank_rng', [])) != self.world_size):
                raise ValueError('DDP checkpoint runtime/contract/world size/cursor mismatch')
            expected = self.adapter.model.state_dict()
            if (set(payload['model']) != set(expected) or any(t.shape != expected[k].shape
                    or t.dtype != expected[k].dtype or not torch.isfinite(t).all() for k,t in payload['model'].items())):
                raise ValueError('DDP checkpoint architecture or numeric drift')
            current = optimizer.state_dict()
            if payload['optimizer']['param_groups'] != current['param_groups']:
                raise ValueError('DDP optimizer settings drift')
            params = [p for g in optimizer.param_groups for p in g['params']]
            ids = [i for g in payload['optimizer']['param_groups'] for i in g['params']]
            if set(payload['optimizer']['state']) != (set(ids) if cursor else set()):
                raise ValueError('DDP optimizer slots incomplete')
            for i,p in zip(ids,params):
                if not cursor:
                    continue
                slot=payload['optimizer']['state'][i]
                if set(slot) != {'step','exp_avg','exp_avg_sq'}:
                    raise ValueError('DDP AdamW slot fields drift')
                for name,t in slot.items():
                    if (t.dtype != torch.float32 or not torch.isfinite(t).all()
                            or name == 'step' and (t.numel()!=1 or t.item()!=cursor)
                            or name != 'step' and t.shape!=p.shape):
                        raise ValueError('DDP AdamW slot shape/cursor drift')
            rng=payload['rank_rng'][self.rank]
            if (rng['torch'].dtype!=torch.uint8 or rng['torch'].shape!=torch.get_rng_state().shape
                    or rng['cuda'].dtype!=torch.uint8 or rng['cuda'].shape!=torch.cuda.get_rng_state(self.rank).shape):
                raise ValueError('DDP rank RNG drift')
            torch.Generator().set_state(rng['torch'])
            random.Random().setstate(rng['python'])
            numpy_state=decode_numpy_rng(rng['numpy'])
        except Exception as exc:
            error=self.error(exc)
        self.agree(error)
        error = None
        try:
            self.adapter.model.load_state_dict(payload['model'],strict=True)
            optimizer.load_state_dict(payload['optimizer'])
            self.adapter.model.train(payload['training'])
            torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'],self.rank)
            random.setstate(rng['python']);np.random.set_state(numpy_state)
        except Exception as exc:
            error = self.error(exc)
        self.agree(error)
        self.updates_seen=cursor
        return cursor

    def workers(self):
        import torch.distributed as dist
        while True:
            message=[None]
            dist.broadcast_object_list(message,src=0)
            msg=message[0];op=msg['operation']
            if op=='quit':
                return
            try:
                if op=='initialize': self._bind(make_adapter(msg['factory'],self.rank))
                elif op=='optimizer': self.make_optimizer()
                elif op=='update': self._update(msg['indices'],msg['selected'])
                elif op=='save': self.save(None,self.optimizer,contract=None,cursor=msg['cursor'])
                elif op=='restore': self.restore(msg['path'],self.optimizer,contract=msg['contract'],cursor=msg['cursor'])
                else: raise ValueError('unknown coordinator operation')
            except (RuntimeError,ValueError):
                # Coordinated update/restore errors were already reported to
                # rank0. Wait for its quit; do not start a different collective.
                if op not in ('update','restore'):
                    raise

    def close(self):
        import torch.distributed as dist
        if self.closed:
            return
        self.closed=True
        if self.rank==0:
            self.command('quit')
        dist.destroy_process_group(self.grad_group)
        dist.destroy_process_group()


def rank_entry(rank, world_size, rendezvous, argv, entry_module):
    global CURRENT
    import importlib
    import torch
    torch.set_num_threads(1)
    # Only rank0 owns lifecycle/signals/checkpoints. Other ranks remain alive
    # long enough to complete its update and post-update safe-pause checkpoint.
    if rank:
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
            signal.signal(sig,signal.SIG_IGN)
    CURRENT=Context(rank,world_size,rendezvous)
    try:
        if rank:
            CURRENT.workers();code=0
        else:
            code=importlib.import_module(entry_module).main(argv)
    finally:
        CURRENT.close()
    if code:
        raise SystemExit(code)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world-size',type=int,required=True)
    parser.add_argument('--entry-module',default='baselines.retacred.reference_complete',help=argparse.SUPPRESS)
    parser.add_argument('worker_args',nargs=argparse.REMAINDER)
    args=parser.parse_args(argv)
    import torch
    import torch.multiprocessing as mp
    if not 2<=args.world_size<=6 or torch.cuda.device_count()!=args.world_size:
        parser.error('world-size2..6 must equal the isolated visible GPU group')
    worker_args=args.worker_args[1:] if args.worker_args[:1]==['--'] else args.worker_args
    with tempfile.TemporaryDirectory(prefix='qattn-ctl-ddp-') as temporary:
        rendezvous='file://'+str(Path(temporary)/'control')
        handlers={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
        for sig in handlers:signal.signal(sig,signal.SIG_IGN)
        try:
            mp.spawn(rank_entry,args=(args.world_size,rendezvous,worker_args,args.entry_module),nprocs=args.world_size,join=True)
        except mp.ProcessExitedException as exc:
            return exc.exit_code if exc.exit_code in (75,76) else 1
        finally:
            for sig,handler in handlers.items():signal.signal(sig,handler)
    return 0


if __name__=='__main__':
    # Spawn must pickle the canonical module's function, not __mp_main__;
    # CURRENT is read by the ordinary worker module in each rank.
    from baselines.retacred.ctl_drp_ddp import main as canonical_main
    raise SystemExit(canonical_main())
