"""Execution-only OOM tiers, periodic durable saves and bounded archive I/O."""
from __future__ import annotations

import copy
import json
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import atomic_json

OOM_RETRY = 76
_archives = None


def tiers(model, logical, requested='auto'):
    if requested != 'auto':
        width = int(requested)
        if not 1 <= width <= logical:
            raise ValueError('physical micro-batch must be within the logical batch')
        if model == 'sequence_routing_h3' and width != 1:
            raise ValueError('H3 requires micro-batch1: upstream routing does not mask padding')
        widths = [width]
        while widths[-1] > 1:
            widths.append(max(1, widths[-1] // 2))
    elif model == 'sequence_routing_h3':
        # Upstream H3 routes padded positions: grouping sentences changes its
        # active mechanism. Keep its original one-sentence partition.
        widths = [1]
    else:
        widths = [logical]
        while widths[-1] > 1:
            widths.append(max(1, widths[-1] // 2))
    return [dict(micro=n, checkpointing=False) for n in widths] + [
        dict(micro=widths[-1], checkpointing=True)]


def ledger_path(run):
    run = Path(run)
    return run.with_name('.' + run.name + '.execution.json')


def prepare(args, model, logical):
    requested = str(args.micro_batch)
    seconds = getattr(args, 'checkpoint_seconds', 120.)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('checkpoint seconds must be finite and nonnegative')
    policy = dict(schema='reference-execution-tiers.v1', model=model,
                  seed=args.seed, requested_micro=requested,
                  tiers=tiers(model, logical, requested), checkpoint_seconds=seconds,
                  guarantee='full_state_restore; tier_changes_not_bitwise_identical')
    path = ledger_path(args.run_dir)
    if path.is_symlink():
        raise ValueError('symlinked execution ledger')
    if path.exists():
        state = json.loads(path.read_text())
        if state['policy'] != policy:
            raise ValueError('execution policy drift')
        index = state['index']
        if type(index) is not int or not 0 <= index < len(policy['tiers']):
            raise ValueError('invalid OOM tier index')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = dict(policy=policy, index=0, events=[])
        atomic_json(path, state)
    args.execution_policy = policy
    args.execution_state = state
    args.micro_batch = policy['tiers'][state['index']]['micro']
    args.activation_checkpointing = policy['tiers'][state['index']]['checkpointing']
    return state


def attach(adapter, args):
    if hasattr(args,'execution_policy'):
        adapter.execution_policy = args.execution_policy
        adapter.execution_state = args.execution_state


def metadata(adapter, cursor, checkpoint_cursor):
    state = getattr(adapter, 'execution_state', None)
    if state is None:
        return {}
    tier = state['policy']['tiers'][state['index']]
    return dict(execution_tier=state['index'], oom_retries=len(state['events']),
                physical_micro=tier['micro'], activation_checkpointing=tier['checkpointing'],
                checkpoint_batch=checkpoint_cursor, replay_updates=cursor-checkpoint_cursor)


class Cadence:
    def __init__(self, policy=None, *, cursor=0, clock=time.monotonic):
        self.seconds = policy['checkpoint_seconds'] if policy else 0.
        self.clock = clock
        self.last = clock()
        self.cursor = cursor

    def due(self):
        return self.seconds == 0 or self.clock()-self.last >= self.seconds

    def saved(self, cursor):
        self.cursor = cursor
        self.last = self.clock()


class ArchiveQueue:
    """A single writer; queued items contain paths/metadata, never GPU tensors."""
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='reference-archive')
        self.pending = deque()

    def poll(self):
        while self.pending and self.pending[0].done():
            self.pending.popleft().result()

    def submit(self, operation):
        self.poll()
        # Bound retained local payloads and queued work. Backpressure is safe;
        # network failures propagate instead of losing selected case values.
        if len(self.pending) >= 2:
            self.pending.popleft().result()
        self.pending.append(self.pool.submit(operation))

    def close(self):
        try:
            while self.pending:
                self.pending.popleft().result()
        finally:
            self.pool.shutdown(wait=True)


def schedule_archive(operation):
    if _archives is None:
        return operation()
    _archives.submit(operation)


def poll_archives():
    if _archives is not None:
        _archives.poll()


def drain_archives():
    if _archives is not None:
        while _archives.pending:
            _archives.pending.popleft().result()


def adaptive_cli(args, model, logical, operation):
    """Only CUDA OOM advances one tier. Other failures keep their real status."""
    import torch
    global _archives
    prepare(args, model, logical)
    _archives = ArchiveQueue()
    try:
        try:
            return operation()
        finally:
            _archives.close()
    except torch.cuda.OutOfMemoryError as error:
        state = args.execution_state
        old = state['index']
        event = dict(from_tier=old, to_tier=old+1, trigger='cuda_oom',
                     error=str(error), timestamp=time.time())
        if torch.cuda.is_available():
            event.update(allocated_bytes=torch.cuda.memory_allocated(),
                         reserved_bytes=torch.cuda.memory_reserved(),
                         peak_bytes=torch.cuda.max_memory_allocated())
        if old+1 >= len(state['policy']['tiers']):
            atomic_json(Path(args.run_dir).with_name('.'+Path(args.run_dir).name+'.OOM_EXHAUSTED.json'), event)
            raise
        state['events'].append(event)
        state['index'] = old+1
        from .reference_storage import durable_json
        durable_json(ledger_path(args.run_dir), state)
        print(f'[OOM] tier {old} -> {old+1}; restart from last durable optimizer update', flush=True)
        return OOM_RETRY
    finally:
        _archives = None
