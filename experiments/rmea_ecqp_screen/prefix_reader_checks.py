"""Full-shape synthetic prefix receipt/isolation checks, never real data."""
import hashlib
import json
from pathlib import Path

import numpy as np


def checks(runner,folder):
    folder=Path(folder);folder.mkdir();assets=folder/'assets';assets.mkdir()
    lengths=np.full(4096,37,dtype=np.int64);lengths[:19]+=1
    offsets=np.concatenate([np.zeros(1,dtype=np.int64),lengths.cumsum()])
    assert offsets[-1]==151571
    h=np.lib.format.open_memmap(assets/'train-h.npy',mode='w+',dtype='<f2',shape=(151571,1024))
    roles=np.zeros((151571,4),dtype='<f4')
    for i in range(4096):
        lo=int(offsets[i]);h[lo,0]=(i%11)/16;h[lo+1,1]=(i%7)/16
        roles[lo,2]=1;roles[lo+1,3]=1
    h.flush();del h
    targets=np.arange(4096,dtype='<i8')%40
    for name,array in [('train-offsets.npy',offsets),('train-roles.npy',roles),('train-targets.npy',targets)]:
        np.save(assets/name,array,allow_pickle=False)
    metadata=[dict(id_hash=f'synthetic-{i}',words=int(lengths[i])) for i in range(4096)]
    (assets/'metadata.json').write_text(json.dumps(dict(train=metadata)),encoding='utf8')
    cohort=[dict(id_hash=r['id_hash'],source_index=i,group=f'synthetic-group-{i}',split='fit' if i<3124 else 'eval')
        for i,r in enumerate(metadata)]
    donors={}
    for split in ['fit','eval']:
        selected=[r for r in cohort if r['split']==split]
        donors[split]=runner.cohort_derangement([r['id_hash'] for r in selected],[r['group'] for r in selected],split)
    cohort_path=folder/'cohort.json';donors_path=folder/'donors.json'
    cohort_path.write_text(json.dumps(dict(rows=cohort)),encoding='utf8')
    donors_path.write_text(json.dumps(donors),encoding='utf8')
    cohort_original=cohort_path.read_bytes();donors_original=donors_path.read_bytes()
    receipt=folder/'prefix-acceptance.json';plan_hash=hashlib.sha256(b'synthetic-not-real-plan').hexdigest()
    receipt.write_text(json.dumps(dict(status='PASS_TRAIN_PREFIX_RECOVERY',prefix_examples=4096,
        full_original_H_hash_recomputed=False,sidecar_original_hashes_verified=True,plan_sha256=plan_hash,
        files={p.name:dict(bytes=p.stat().st_size,sha256=runner.digest(p)) for p in assets.iterdir()})),encoding='utf8')
    receipt_hash=runner.digest(receipt)
    def load(**override):
        kwargs=dict(expected_receipt_sha256=receipt_hash,expected_plan_sha256=plan_hash);kwargs.update(override)
        return runner.read_training_prefix(assets,receipt,cohort_path,runner.digest(cohort_path),
            donors_path,runner.digest(donors_path),**kwargs)
    def rejects(fn):
        try:fn()
        except AssertionError:return
        raise AssertionError('Invalid prefix chain unexpectedly accepted')
    data=load();assert len(data['splits']['fit'])==3124 and len(data['splits']['eval'])==972
    assert data['summaries'][3,0]==np.float32(3/32) and data['summaries'][3,1]==np.float32(3/32)
    del data
    rejects(lambda:load(expected_receipt_sha256='0'*64))
    rejects(lambda:load(expected_plan_sha256='0'*64))
    cohort_path.write_text(json.dumps(dict(rows=cohort[::-1])),encoding='utf8')
    rejects(load);cohort_path.write_bytes(cohort_original)
    donors['fit'][cohort[0]['id_hash']]=cohort[-1]['id_hash']
    donors_path.write_text(json.dumps(donors),encoding='utf8')
    rejects(load);donors_path.write_bytes(donors_original)
    target=assets/'train-targets.npy';original=target.read_bytes();target.write_bytes(original+b'x')
    rejects(load);target.write_bytes(original)
    return ['full_shape_synthetic_prefix_receipt_plan_chain_and_torch_entity_pooling',
        'wrong_receipt_or_plan_hash_rejected_before_array_use',
        'reordered_cohort_and_cross_split_donor_rejected','tampered_declared_asset_rejected']
