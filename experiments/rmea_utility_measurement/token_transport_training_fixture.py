"""Pre-update CUDA gate for numerical ranges and checkpoint artifact failures."""
import hashlib
from pathlib import Path

import numpy as np
import torch

from experiments.rmea_ecqp_screen import runner
from experiments.rmea_original_scores.common import read, sha
from experiments.rmea_utility_measurement.token_transport_screen import KINDS, batch, range_diagnostics, seal_checkpoints, stage_checkpoint
from experiments.rmea_utility_measurement.token_transport_trainable import TrainableTokenTransportHead


@torch.no_grad()
def run(a, *,serialization=True):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261120)
    rng = np.random.default_rng(20261120)
    data = dict(h=rng.normal(0,.2,(8,1024)).astype('float32'),
        roles=np.zeros((8,4),dtype='float32'),offsets=np.asarray([0,3,8],dtype='int64'),
        targets=np.asarray([2,17],dtype='int64'),splits=dict(eval=np.asarray([0,1],dtype='int64')),
        metadata=[dict(id_hash=hashlib.sha256(('training-fixture:'+str(i)).encode()).hexdigest()) for i in range(2)])
    data['roles'][0,2] = data['roles'][1,3] = data['roles'][3,2] = data['roles'][4,3] = 1
    model = TrainableTokenTransportHead('cubic_coherent').double().to('cuda').eval()
    h,r,mask,_ = batch(data,data['splits']['eval'],'eval')
    original = runner.state_fingerprint(model.state_dict())
    d = model(h,r,mask,details=True)
    assert d['psi'].imag.abs().max()>1e-5
    assert d['utilities'].min()<0 and d['utilities'].max()>0
    generic = range_diagnostics(model,data)
    assert runner.state_fingerprint(model.state_dict()) == original
    state = {n:t.detach().cpu() for n,t in model.state_dict().items()}
    if not serialization:
        restored = TrainableTokenTransportHead('cubic_coherent').double().to('cuda').eval()
        restored.load_state_dict(state)
        assert runner.state_fingerprint(restored.state_dict()) == original
        assert torch.equal(restored(h,r,mask)[0],d['logits'])
        model.prepare.weight.zero_();model.prepare.bias.fill_(.13)
        degenerate = range_diagnostics(model,data)
        rows=[v for report in (generic,degenerate) for u in report['units'] for v in u['routing'].values()]
        assert len(rows)==16
        return dict(status='PASS',range_rows=16,optimizer_steps=0,real_dataset_reads=0,
            maxima={k:max(v[k] for v in rows) for k in ('direct_quadratic_score_residual','direct_quadratic_readout_residual','masked_unpadded_score_residual','own_readout_range_residual')},
            state_replay_current_gpu_logits_exact=True,checkpoint_files=0,
            scope='Same-child complex/masked/signed/rank-one numerical fixture and RAM state replay;no filesystem serialization or scientific utility claim.')
    root = a.output.parent/'scientific-work/fixture'
    assert not root.exists()
    root.mkdir(mode=0o700)
    failed = root/'failed'
    failed.mkdir()
    # This is an artifact-write retry from one immutable state, never retraining.
    rejected = False
    try:
        stage_checkpoint(failed/'fixture-final.pt',state,maximum=16)
    except (OSError,RuntimeError):
        rejected = True
    assert rejected and not (failed/'fixture-final.pt').exists() and not (failed/'RUN_COMPLETE').exists()
    assert all(p.stat().st_size<=16 for p in failed.iterdir())
    good = root/'verified'
    good.mkdir()
    path = good/'fixture-final.pt'
    pending = stage_checkpoint(path,state)
    restored = TrainableTokenTransportHead('cubic_coherent').double().to('cuda').eval()
    restored.load_state_dict(torch.load(pending,map_location='cpu',weights_only=True))
    assert runner.state_fingerprint(restored.state_dict()) == original
    assert torch.equal(restored(h,r,mask)[0],d['logits'])
    pending.replace(path)
    files={path.name:dict(sha256=sha(path),bytes=path.stat().st_size)}
    # A failed size acceptance must leave no completion marker.
    rejected = False
    try:
        seal_checkpoints(good,files,maximum=1)
    except AssertionError:
        rejected = True
    assert rejected and not (good/'RUN_COMPLETE').exists()
    seal_checkpoints(good,files,maximum=1024**2)
    assert read(good/'RUN_COMPLETE')['files']=={path.name:sha(path)}
    with torch.no_grad():
        model.prepare.weight.zero_()
        model.prepare.bias.fill_(.13)
    degenerate = range_diagnostics(model,data)
    rows=[v for report in (generic,degenerate) for u in report['units'] for v in u['routing'].values()]
    assert len(rows)==16
    maxima={k:max(v[k] for v in rows) for k in ('direct_quadratic_score_residual',
        'direct_quadratic_readout_residual','masked_unpadded_score_residual','own_readout_range_residual')}
    assert set(generic['units'][0]['routing'])==set(KINDS)
    return dict(status='PASS',cases=['complex-padded-signed-utilities','rank-one-key-conditioning',
        'write-limit-rejection-before-final','reload-fingerprint-and-current-GPU-logits-exact',
        'size-failure-leaves-no-marker','same-state-artifact-retry-seals-exact-hashes'],
        range_rows=16,maxima=maxima,optimizer_steps=0,real_dataset_reads=0,
        immutable_fixture_state_sha256=original,fixture_checkpoint_sha256=sha(path),
        fixture_checkpoint_bytes=path.stat().st_size,
        scope='Bounded synthetic source/current-GPU replay and private artifact rejection;not training resume,crash recovery or task evidence.')
