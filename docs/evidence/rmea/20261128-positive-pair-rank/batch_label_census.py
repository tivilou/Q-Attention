"""Reconcile exact FIT-only class presence with the unchanged 640-batch order."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(pin):
    z = Path(pin['path']).read_bytes()
    assert sha(z) == pin['sha256']
    return z


def audit(cfg):
    code = cfg['stage_source']
    assert sha(code.encode()) == '67ea88a03ceb8aa92579c992ec933c27e641622e8cb5a1878e9ca21923a16a6b'
    frozen = {}
    exec(compile(code, '<pinned-stage>', 'exec'), frozen)
    cohort = json.loads(read(cfg['cohort']))
    roles = cohort['indices']
    fit = np.asarray(roles['fit'],dtype=np.int64)
    assert not set(fit)&(set(roles['selection'])|set(roles['qualification']))
    raw = Path(cfg['targets']['path'])
    assert read(cfg['targets'])
    targets = np.load(raw, mmap_mode='r', allow_pickle=False)
    assert targets.shape == (58465,) and targets.dtype.kind == 'i'
    y = targets[fit]
    supports = np.bincount(y,minlength=40)
    assert y.min()==0 and y.max()==39 and supports.tolist()==cohort['supports']['fit']==cfg['fit_support_counts']
    order, order_hash = frozen['batches'](fit)
    assert order_hash==cfg['order_sha256']
    counts = np.stack([np.bincount(targets[part],minlength=40) for part in order])
    assert counts.shape==(640,40) and np.all(counts.sum(1)==128)
    assert np.all(counts.reshape((20,32,40)).sum(axis=1)==supports)
    absent=(counts==0).sum(0)
    zeros_per_batch=(counts==0).sum(1)
    assert np.all(absent >= np.maximum(0,640-20*supports))
    return dict(status='PASS_FROZEN_FIT_CLASS_PRESENCE_CENSUS',
                fit_rows=len(y),fit_label_supports=supports.tolist(),logical_batch_size=128,
                batches=640,epochs=20,cohort_sha256=cfg['cohort']['sha256'],
                targets_sha256=cfg['targets']['sha256'], order_sha256=order_hash,
                objective_meaning='Absent gold label makes that class own soft F1 term identically zero within the batch.',
                per_class_absent_batches=absent.tolist(),
                per_batch_absent_classes=dict(min=int(min(zeros_per_batch)), median=float(np.median(zeros_per_batch)),
                                              max=int(max(zeros_per_batch))),
                absence_over_all_label_batch_cells=int(absent.sum()),
                classes_absent_in_more_than_half_of_batches=int((absent>320).sum()),
                classes_always_present=int((absent==0).sum()),
                fit_positive_label_count_only=True,selection_qualification_labels_indexed=0,
                official_splits_read=0,task_forwards=0,updates=0,gpu_allocations=0)


if __name__=='__main__':
    print(json.dumps(audit(json.load(sys.stdin)),allow_nan=False))