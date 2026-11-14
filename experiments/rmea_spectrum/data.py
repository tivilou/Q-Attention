"""Pinned cached train/validation representations; no encoder or test reader."""
import hashlib,json
from pathlib import Path
import numpy as np

FEATURE_SHA='c1279284a9a5e9d59e3303c591e657cc2f4f340b510f96d3ace727b89dcd6511'
MANIFEST_SHA='8e5c6dec3ae31b2bbb4bdb45fa6b80b2be74275cea4d6afb87da612ad80392d8'
ID_SHA={'train':'f2df684905140be85bcff9dc3556864bfe89b52ab8c566865e86bf0c91db96c5','valid':'473ccfbb31f9831a3fe5413a4a0147630a756cdbfd8edc3a73acceadf9820b28'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def load_cache(root,exclusions,expected_exclusion_sha):
    root=Path(root);exclusions=Path(exclusions)
    for n in ('features.npz','cache-manifest.json','cache-validation.json','RUN_COMPLETE'):
        assert (root/n).is_file() and not (root/n).is_symlink()
    assert sha(root/'features.npz')==FEATURE_SHA and sha(root/'cache-manifest.json')==MANIFEST_SHA
    m=json.loads((root/'cache-manifest.json').read_text());v=json.loads((root/'cache-validation.json').read_text())
    closed=json.loads((root/'RUN_COMPLETE').read_text())
    assert closed==dict(schema='qattn.rebuilt-cache.closed.v1',files={n:sha(root/n) for n in ('features.npz','cache-manifest.json')})
    assert m['counts']==dict(train=2048,valid=1024) and not m['test_loaded'] and not m['head_training']
    assert m['source_revision']=='0e25cf2309cfb1cc406e6b96521c2a29671a1048' and m['sample_id_hashes']==ID_SHA
    assert m['schema']=='qattn.rmea.rebuilt-features.v1' and m['profile']=='rebuild'
    assert m['features']==dict(sha256=FEATURE_SHA,bytes=(root/'features.npz').stat().st_size)
    assert m['source_data']==dict(train='f1051d574a44df3ecbd036c1de4fbbe0a372c768f6ec990330e2cea4a657269f',valid='9d6965f984d7485e2e81d380a0b0de2922d3da1e2685393ea3e46869805c2b55')
    assert v['status']=='passed' and v['manifest_sha256']==MANIFEST_SHA and v['features_sha256']==FEATURE_SHA
    assert sha(exclusions)==expected_exclusion_sha and not exclusions.is_symlink()
    excluded=json.loads(exclusions.read_text())['id_hashes'];assert 240<=len(excluded)<=242 and len(excluded)==len(set(excluded))
    assert all(len(s)==64 and all(c in '0123456789abcdef' for c in s) for s in excluded)
    with np.load(root/'features.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
    assert set(d)=={s+'_'+k for s in ('train','valid') for k in ('features','roles','offsets','targets','ids')}
    for s,n in m['counts'].items():
        h,r,o,y,ids=[d[s+'_'+k] for k in ('features','roles','offsets','targets','ids')]
        assert h.dtype==np.float16 and r.dtype==np.float32 and o.dtype==np.int64 and y.dtype==np.int64 and ids.dtype==np.dtype('S64')
        assert h.shape==(len(h),1024) and r.shape==(len(h),4) and y.shape==(n,) and ids.shape==(n,)
        assert o.shape==(n+1,) and o[0]==0 and o[-1]==len(h) and (np.diff(o)>0).all()
        assert np.isfinite(h).all() and np.isfinite(r).all() and ((y>=0)&(y<40)).all()
        assert hashlib.sha256(b''.join(ids)).hexdigest()==ID_SHA[s] and len(set(ids))==n
    assert not (set(d['train_ids']) & set(d['valid_ids']))
    exclude=set(excluded);indices={'train':np.arange(2048),
        'valid':np.array([i for i,x in enumerate(d['valid_ids']) if x.decode('ascii') not in exclude],dtype=np.int64)}
    assert len(indices['valid'])>=782
    assert not any(d['valid_ids'][i].decode('ascii') in exclude for i in indices['valid'])
    labels=m['labels'];assert len(labels)==len(set(labels))==40 and labels[0]=='no_relation'
    cohort=dict(status='PASS',cache_sha256=FEATURE_SHA,cache_manifest_sha256=MANIFEST_SHA,
        counts={s:len(i) for s,i in indices.items()},original_valid=1024,excluded_from_valid=1024-len(indices['valid']),
        exclusion_inventory_count=len(excluded),exclusions_sha256=expected_exclusion_sha,
        selected_id_sequence_sha256={s:hashlib.sha256(b''.join(d[s+'_ids'][i] for i in ind)).hexdigest() for s,ind in indices.items()},
        class_support={s:np.bincount(d[s+'_targets'][ind],minlength=40).tolist() for s,ind in indices.items()},
        labels=labels,test_loaded=False,selection='Existing readiness-v1 cache, all train2048; valid1024 minus ALL previously explored multiclass/parity id_hashes, no prediction-based selection.',
        limitation='This development cache was used in prior joint diagnostics. Excluding the adaptive240 does not create an independent confirmatory population.')
    return d,indices,cohort

def batch(d,split,indices):
    import torch
    offsets=d[split+'_offsets'];lengths=[int(offsets[i+1]-offsets[i]) for i in indices];n=max(lengths)
    h=np.zeros((len(indices),n,1024),np.float32);roles=np.zeros((len(indices),n,4),np.float32);mask=np.zeros((len(indices),n),bool)
    for j,(i,t) in enumerate(zip(indices,lengths)):
        lo,hi=map(int,offsets[i:i+2]);h[j,:t]=d[split+'_features'][lo:hi];roles[j,:t]=d[split+'_roles'][lo:hi];mask[j,:t]=True
    return tuple(torch.from_numpy(x) for x in (h,roles,mask)),torch.from_numpy(d[split+'_targets'][indices].copy())
