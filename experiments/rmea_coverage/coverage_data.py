"""Deterministic training-only coverage selection and immutable base reader."""
import collections,hashlib,heapq,json
from pathlib import Path
import numpy as np
BASE_SHA='c1279284a9a5e9d59e3303c591e657cc2f4f340b510f96d3ace727b89dcd6511'
MANIFEST_SHA='8e5c6dec3ae31b2bbb4bdb45fa6b80b2be74275cea4d6afb87da612ad80392d8'
TRAIN_SHA='f1051d574a44df3ecbd036c1de4fbbe0a372c768f6ec990330e2cea4a657269f'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def sid(row):return hashlib.sha256(str(row['metadata']['id']).encode()).hexdigest()
def rank(row):return hashlib.sha256(('rmea-coverage-topup-v1:'+str(row['metadata']['id'])).encode()).hexdigest()
def roles_for(row):
    p=np.arange(len(row['tokens']),dtype=np.float32);s,o=row['subject'],row['object']
    return np.stack([np.clip((p-(s[0]+s[1]-1)/2)/32,-1,1),np.clip((p-(o[0]+o[1]-1)/2)/32,-1,1),((p>=s[0])&(p<s[1])).astype(np.float32),((p>=o[0])&(p<o[1])).astype(np.float32)],axis=-1)
def array_sha(a):return hashlib.sha256(a.tobytes(order='C')).hexdigest()
def load_base(root):
    root=Path(root)
    for name,h in [('features.npz',BASE_SHA),('cache-manifest.json',MANIFEST_SHA),('cache-validation.json','b5b264dd6888dad5bb048568a39f2e46a08c450056b52184042ebe2881fafcbb')]:
        assert not (root/name).is_symlink() and sha(root/name)==h
    m=json.loads((root/'cache-manifest.json').read_text())
    assert m['counts']=={'train':2048,'valid':1024} and not m['test_loaded'] and not m['head_training']
    assert m['source_data']['train']==TRAIN_SHA and m['dtype']=='float32 CPU encoder and word-piece mean; saved features float16; roles float32'
    assert m['source_revision']=='0e25cf2309cfb1cc406e6b96521c2a29671a1048'
    with np.load(root/'features.npz',allow_pickle=False) as f:d={n:f[n] for n in f.files}
    assert set(d)=={s+'_'+k for s in ['train','valid'] for k in ['features','roles','offsets','targets','ids']}
    for s,n in m['counts'].items():
        h,r,o,y,ids=[d[s+'_'+k] for k in ['features','roles','offsets','targets','ids']]
        assert h.dtype==np.float16 and r.dtype==np.float32 and o.dtype==y.dtype==np.int64 and ids.dtype==np.dtype('S64')
        assert h.shape==(len(h),1024) and r.shape==(len(h),4) and o.shape==(n+1,) and y.shape==ids.shape==(n,)
        assert o[0]==0 and o[-1]==len(h) and (np.diff(o)>0).all() and np.isfinite(h).all() and np.isfinite(r).all()
        assert len(set(ids))==n and ((y>=0)&(y<40)).all()
        assert hashlib.sha256(b''.join(ids)).hexdigest()==m['sample_id_hashes'][s]
    assert not set(d['train_ids'])&set(d['valid_ids'])
    return d,m
def select(rows,old,labels,old_counts,cap=16):
    """Keep at most cap candidates/class; no held-out rows/metrics accepted."""
    existing=set(old);assert len(existing)==len(old)
    heaps={k:[] for k in labels};counts=collections.Counter();seen=set();old_rows={}
    for row in rows:
        key=sid(row);assert key not in seen;seen.add(key)
        label=row['label'];assert label in heaps;counts[label]+=1
        if key in existing:old_rows[key]=row;continue
        need=max(0,cap-old_counts[label])
        if need:
            item=(-int(rank(row),16),-int(key,16),row)
            if len(heaps[label])<need:heapq.heappush(heaps[label],item)
            elif item[:2]>heaps[label][0][:2]:heapq.heapreplace(heaps[label],item)
    assert set(old_rows)==existing
    actual_old=collections.Counter(r['label'] for r in old_rows.values())
    assert all(actual_old[k]==old_counts[k] for k in labels)
    result=[]
    for label in labels:
        want=max(0,min(cap,counts[label])-old_counts[label])
        assert len(heaps[label])==want
        result.extend(sorted((x[2] for x in heaps[label]),key=lambda r:(rank(r),sid(r))))
    assert len({sid(r) for r in result})==len(result) and not {sid(r) for r in result}&existing
    return result,old_rows,dict(counts)

def validate_arrays(d,rows,labels):
    assert set(d)=={'features','roles','offsets','targets','ids'}
    h,r,o,y,ids=[d[k] for k in ['features','roles','offsets','targets','ids']]
    assert h.dtype==np.float16 and r.dtype==np.float32 and o.dtype==y.dtype==np.int64 and ids.dtype==np.dtype('S64')
    assert h.shape==(len(h),1024) and r.shape==(len(h),4) and y.shape==ids.shape==(len(rows),)
    assert np.isfinite(h).all() and np.isfinite(r).all()
    assert np.array_equal(o,np.cumsum([0]+[len(x['tokens']) for x in rows],dtype=np.int64))
    assert ids.astype(str).tolist()==[sid(x) for x in rows] and y.tolist()==[labels.index(x['label']) for x in rows]
    assert np.array_equal(r,np.concatenate([roles_for(x) for x in rows]))

def load_overlay(base,root,expected_complete_sha):
    """Consumer recipe; concatenates in RAM only, never rewrites base NPZ."""
    root=Path(root);assert sha(root/'RUN_COMPLETE')==expected_complete_sha
    d,m=load_base(base);closed=json.loads((root/'RUN_COMPLETE').read_text())
    assert closed['schema']=='qattn.coverage.closed.v1'
    assert set(closed['files'])=={'topup-features.npz','selection.json','overlay-manifest.json','validation.json','summary.json','unit.json'}
    for n,h in closed['files'].items():assert not (root/n).is_symlink() and sha(root/n)==h
    manifest=json.loads((root/'overlay-manifest.json').read_text())
    assert manifest['base_features_sha256']==BASE_SHA and manifest['base_manifest_sha256']==MANIFEST_SHA and manifest['new_records']==313
    with np.load(root/'topup-features.npz',allow_pickle=False) as f:extra={n:f[n] for n in f.files}
    assert {n:array_sha(a) for n,a in extra.items()}==manifest['extra_array_sha256']
    assert extra['ids'].shape==(313,) and not set(extra['ids'])&(set(d['train_ids'])|set(d['valid_ids']))
    oldwords=len(d['train_features'])
    for k in ['features','roles','targets','ids']:d['train_'+k]=np.concatenate([d['train_'+k],extra[k]])
    d['train_offsets']=np.concatenate([d['train_offsets'],extra['offsets'][1:]+oldwords])
    assert d['train_targets'].shape==(2361,)
    assert np.bincount(d['train_targets'],minlength=40).tolist()==manifest['combined_train_support']
    return d,manifest
