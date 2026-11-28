"""Exact Linux journal timing and length-only padding census. No ML runtime."""
import ast
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

import numpy as np


def bound(raw, n):
    functions = [x for x in ast.parse(raw).body if isinstance(x, ast.FunctionDef) and x.name == n]
    if len(functions) != 1:
        raise ValueError(f'Missing identity for {n}')
    scope = {'json':json,'os':os,'Path':Path}
    exec(compile(ast.Module(body=functions,type_ignores=[]), '<hashed source>', 'exec'),scope)
    return scope[n]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def census(lengths, rows, times=1):
    a = np.asarray(rows, dtype=np.int64)
    logical = lengths[a].reshape(-1,128)
    maxes = logical.reshape(-1,32,4).max(axis=-1)
    padded = 4*maxes.sum(axis=-1)
    return {'logical_batches':len(logical)*times, 'rows':int(logical.size*times),
            'max_token_length':int(logical.max()),'padded_token_sites':int(padded.sum()*times),
            'mean_padded_sites_per_batch':float(padded.mean()),
            'padded_sites_p10_p50_p90':[float(v) for v in np.quantile(padded,[.1,.5,.9])],
            'batch_largest_sites':int(padded.max()),'valid_tokens':int(logical.sum()*times)}


def run(cfg):
    t = time.monotonic()
    sources = {k:Path(p).read_bytes() for k,p in cfg['paths'].items() if k in ('core','cohort','stage','offsets','terminal')}
    assert all(sha(v)==cfg['hashes'][k] for k,v in sources.items()) and sha(cfg['canary_source'].encode())==cfg['hashes']['canary']
    save = bound(cfg['canary_source'],'save')
    atomic = bound(sources['core'].decode(),'atomic_json')
    bound_order = ast.parse(sources['stage'].decode())
    codes = [x for x in bound_order.body if isinstance(x,ast.FunctionDef) and x.name=='batches']
    assert len(codes)==1
    scope={'np':np,'hashlib':hashlib}
    exec(compile(ast.Module(body=codes,type_ignores=[]),'<hashed order source>','exec'),scope)
    cohort=json.loads(sources['cohort']); ii=cohort['indices']
    order, identity=scope['batches'](ii['fit']); assert identity==cfg['order_hash']
    from io import BytesIO
    offsets=np.load(BytesIO(sources['offsets']),allow_pickle=False)
    assert offsets.shape==(58466,) and offsets.dtype.kind in 'iu' and (np.diff(offsets)>0).all()
    length=np.diff(offsets)
    summaries={'first_fit':census(length,order[0]),'fit':census(length,np.concatenate(order)),
               'selection':census(length,ii['selection'],times=4),'qualification':census(length,ii['qualification'])}

    root=Path(cfg['paths']['temporary_parent']);real=Path(cfg['paths']['terminal']).parent
    assert root.stat().st_dev==real.stat().st_dev
    counts=[int(x) for x in cfg['json_sizes']]
    assert len(counts)==4 and min(counts)>0 and max(counts)<65536
    # Exact serialization/fsync functions, synthetic contents matching four approximate sizes.
    # No timing or labels mined from archived aggregate numbers.
    payloads=[{'data':'a'*(n-len(json.dumps({'data':''},indent=2).encode())-1)} for n in counts]
    timings=[]
    # A timed canary arm writes 32 in-flight records, one complete-backward
    # record and one pre-step record. It records the confirmed step after timing.
    for rep in range(7):
        arms=[]
        with tempfile.TemporaryDirectory(prefix='qattn-phase-journal-cost-',dir=root) as tmp:
            lane=Path(tmp)
            assert lane.stat().st_dev==real.stat().st_dev
            for size,payload in zip(counts,payloads):
                values={}
                methods=[('crash_safe_34',save,34),('atomic_34',atomic,34),('atomic_1',atomic,1)]
                if rep%2:methods.reverse()
                for name,fn,n in methods:
                    file=lane/(name+'.json'); tick=time.perf_counter()
                    for _ in range(n):fn(file,payload)
                    values[name]=time.perf_counter()-tick
                    assert file.stat().st_size==size
                arms.append(values)
        timings.append(arms)
    total={name:[sum(v[name] for v in arm) for arm in timings] for name in ('crash_safe_34','atomic_34','atomic_1')}
    exact_placement_delta=[a-b for a,b in zip(total['crash_safe_34'],total['atomic_1'])]
    return dict(status='PASS_NATIVE_SAME_VOLUME_SYNTHETIC_JOURNAL_AND_LENGTH_ONLY',
                elapsed_seconds=time.monotonic()-t,source_hashes=cfg['hashes'],cohort_order_sha256=identity,
                pinned_profile=cfg['profile'],same_archive_filesystem=True,temporary_files_cleaned=True,
                parsed_offset_count=int(len(offsets)),payload_bytes=counts,scope='synthetic writes only; original paths read only; no CUDA/library import, no features, targets, logits, training or allocation',
                fit_and_score_shapes=summaries,alternating_repetitions=timings,totals_seconds=total,
                delta_seconds={'all':exact_placement_delta,'median':statistics.median(exact_placement_delta),
                               'min':min(exact_placement_delta),'max':max(exact_placement_delta)},
                gpu_synchronization_latency_measured=False,new_task_forwards=0,new_optimizer_updates=0,new_gpu_allocations=0,
                official_splits_read=False,targets_or_model_inputs_read=False)


if __name__=='__main__':
    print(json.dumps(run(json.load(sys.stdin)),allow_nan=False))
