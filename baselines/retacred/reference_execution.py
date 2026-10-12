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


def tiers(model, logical, requested='auto', *, linear=True):
    if requested != 'auto':
        width = int(requested)
        if not 1 <= width <= logical:
            raise ValueError('physical micro-batch must be within the logical batch')
        if model == 'sequence_routing_h3' and width != 1:
            raise ValueError('H3 requires micro-batch1: upstream routing does not mask padding')
        widths = [width]
    elif model == 'sequence_routing_h3':
        # Upstream H3 routes padded positions: grouping sentences changes its
        # active mechanism. Keep its original one-sentence partition.
        widths = [1]
    else:
        widths = [logical]
    # Absolute reductions, never a proportion of the current width. Use unit
    # steps near the useful small-batch range; do not skip 15..2. This bounds the
    # number of large-model process reloads during initial allocation discovery.
    decrement=max(1, math.ceil(widths[0]/32))
    while widths[-1]>1:
        current=widths[-1]
        widths.append(max(16,current-decrement) if linear and current>16
                      else current-1 if linear else max(1,current//2))
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
    cap = getattr(args,'physical_sentence_cap',logical)
    if type(cap) is not int or not 1<=cap<=logical:
        raise ValueError('invalid per-rank sentence cap')
    tier_request='auto' if requested=='auto' else str(min(int(requested),cap))
    policy = dict(schema='reference-execution-tiers.v2', model=model,
                  seed=args.seed, requested_micro=requested,
                  tiers=tiers(model, cap, tier_request), checkpoint_seconds=seconds,
                  guarantee='full_state_restore; tier_changes_not_bitwise_identical')
    if cap!=logical:
        policy.update(logical_batch=logical,per_rank_sentence_cap=cap)
    path = ledger_path(args.run_dir)
    if path.is_symlink():
        raise ValueError('symlinked execution ledger')
    if path.exists():
        state = json.loads(path.read_text())
        if state['policy']==policy and 'linear_upgrade' in state:
            raise ValueError('unexpected historical controls on a new execution policy')
        if state['policy'] != policy:
            from .execution_compat import active, verify_release
            legacy={**policy,'schema':'reference-execution-tiers.v1',
                    'tiers':tiers(model,cap,tier_request,linear=False)}
            if not active() or state['policy']!=legacy:
                raise ValueError('execution policy drift')
            verify_release()
            if 'linear_upgrade' not in state:
                index=state['index']
                if type(index) is not int or not 0<=index<len(legacy['tiers']):
                    raise ValueError('invalid historical OOM tier index')
                previous=legacy['tiers'][index]
                # One small probe once per source upgrade. Explicit micro caps,
                # H3 padding semantics and an already enabled recomputation tier
                # are retained. Failed probes restore the last complete update.
                start=previous['micro']
                if requested=='auto' and model!='sequence_routing_h3' and not previous['checkpointing']:
                    start=min(cap,start+1)
                controls=([dict(micro=start,checkpointing=True)] if previous['checkpointing']
                          else tiers(model,cap,str(start)))
                state['linear_upgrade']=dict(schema='reference-linear-controls.v1',
                    tiers=controls,cursor=0,origin_index=index,origin_micro=previous['micro'])
                from .reference_storage import durable_json
                durable_json(path,state)
                print(f'[EXECUTION UPGRADE] micro {previous["micro"]} -> {start}; '
                      'restore the original full-state checkpoint; linear OOM descent',flush=True)
            upgrade=state['linear_upgrade'];origin=upgrade['origin_index']
            if type(origin) is not int or not 0<=origin<len(legacy['tiers']):
                raise ValueError('invalid historical origin tier')
            previous=legacy['tiers'][origin];start=previous['micro']
            if requested=='auto' and model!='sequence_routing_h3' and not previous['checkpointing']:
                start=min(cap,start+1)
            expected=([dict(micro=start,checkpointing=True)] if previous['checkpointing']
                      else tiers(model,cap,str(start)))
            if (upgrade.get('schema')!='reference-linear-controls.v1' or upgrade['tiers']!=expected
                    or upgrade['origin_micro']!=previous['micro']):
                raise ValueError('historical linear controls drift')
            policy=legacy  # Preserve all existing checkpoint contract hashes.
        _current(state)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        state = dict(policy=policy, index=0, events=[])
        atomic_json(path, state)
    args.execution_policy = policy
    args.execution_state = state
    args.micro_batch = _current(state)['micro']
    args.activation_checkpointing = _current(state)['checkpointing']
    return state


def _current(state):
    upgrade=state.get('linear_upgrade')
    table=upgrade['tiers'] if upgrade else state['policy']['tiers']
    cursor=upgrade['cursor'] if upgrade else state['index']
    if type(state['index']) is not int or type(cursor) is not int or not 0<=cursor<len(table):
        raise ValueError('invalid OOM tier index')
    if upgrade and state['index']!=upgrade['origin_index']+cursor:
        raise ValueError('historical OOM transition count drift')
    if (len(state['events'])!=state['index'] or any(e.get('from_tier')!=i or e.get('to_tier')!=i+1
            for i,e in enumerate(state['events']))):
        raise ValueError('OOM event history drift')
    return table[cursor]


def reset_update_peak(device):
    import torch
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)


def memory_metadata(device):
    import torch
    if not str(device).startswith('cuda'):return {}
    return dict(memory_allocated_bytes=torch.cuda.memory_allocated(device),
                memory_reserved_bytes=torch.cuda.memory_reserved(device),
                memory_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                memory_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                memory_total_bytes=torch.cuda.get_device_properties(device).total_memory)


def attach(adapter, args):
    if hasattr(args,'execution_policy'):
        adapter.execution_policy = args.execution_policy
        adapter.execution_state = args.execution_state


def metadata(adapter, cursor, checkpoint_cursor):
    state = getattr(adapter, 'execution_state', None)
    if state is None:
        return {}
    tier = _current(state)
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
        upgrade=state.get('linear_upgrade')
        cursor=upgrade['cursor'] if upgrade else old
        table=upgrade['tiers'] if upgrade else state['policy']['tiers']
        if cursor+1 >= len(table):
            atomic_json(Path(args.run_dir).with_name('.'+Path(args.run_dir).name+'.OOM_EXHAUSTED.json'), event)
            raise
        state['events'].append(event)
        state['index'] = old+1
        if upgrade:upgrade['cursor']=cursor+1
        event.update(from_micro=table[cursor]['micro'],to_micro=table[cursor+1]['micro'],
                     to_checkpointing=table[cursor+1]['checkpointing'])
        from .reference_storage import durable_json
        durable_json(ledger_path(args.run_dir), state)
        print(f'[OOM] tier {old} -> {old+1}; micro {event["from_micro"]} -> {event["to_micro"]}; '
              'restart from last durable optimizer update', flush=True)
        return OOM_RETRY
    finally:
        _archives = None
