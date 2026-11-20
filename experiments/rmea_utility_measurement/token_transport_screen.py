"""Fixed train-derived discovery protocol; execution requires exact admission.

Profiling makes no optimizer updates and reads no evaluation outcomes. Training
has a separate prospective contract; implementing it does not admit it.
"""
import copy
import hashlib
import math
import json
import os
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.rmea_context_geometry.legacy.rmea import EPS, RMEAHead
from experiments.rmea_ecqp_screen import metrics, runner
from experiments.rmea_original_scores.common import save, sha
from experiments.rmea_utility_measurement.token_transport_trainable import KINDS, TrainableTokenTransportHead, route


SETTINGS = dict(seed=20261118, epochs=4, batch_size=32, lr=.001,
                weight_decay=.01, betas=[.9, .999], eps=1e-8, clip_norm=1.)


def setup(a, design):
    assert design['conditions'] == list(KINDS) and design['training'] == SETTINGS
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    identity = design['data']
    data = runner.read_training_prefix(a.assets, a.receipt, a.cohort,
        identity['cohort_sha256'], a.donors, identity['donors_sha256'],
        expected_receipt_sha256=identity['prefix_receipt_sha256'],
        expected_plan_sha256=identity['plan_sha256'])
    fit, ev = data['splits']['fit'], data['splits']['eval']
    assert len(fit) == 3124 and len(ev) == 972
    assert [len({data['cohort'][int(i)]['group'] for i in ids}) for ids in (fit, ev)] == [2714, 866]
    torch.manual_seed(SETTINGS['seed'])
    initial = copy.deepcopy(RMEAHead('quantum').double().state_dict())
    assert sum(t.numel() for t in initial.values()) == 81028
    rng = np.random.default_rng(SETTINGS['seed'])
    orders = [rng.permutation(fit) for _ in range(SETTINGS['epochs'])]
    return data, initial, orders


def batch(data, ids, split):
    h, r, mask, y, donor = runner.batch(data, ids, split, False)
    assert donor is None
    return h.to('cuda', torch.float64), r.to('cuda', torch.float64), mask.to('cuda'), y.to('cuda')


def fresh(kind, initial):
    model = TrainableTokenTransportHead(kind).double()
    model.load_state_dict(initial)
    assert runner.state_fingerprint(model.state_dict()) == runner.state_fingerprint(initial)
    assert sum(p.numel() for p in model.parameters()) == 81028
    return model.to('cuda')


def finite_gradients(model):
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def profile(a, design):
    """One warm first batch, then first/longest-fit stress batches per condition."""
    started = time.monotonic()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    # One run-wide allocator high-water mark: include setup, warmup and transitions.
    # This is an allocator acceptance threshold, not a kernel device-memory cap.
    torch.cuda.reset_peak_memory_stats()
    data, initial, orders = setup(a, design)
    lengths = np.diff(data['offsets'])
    longest = np.asarray(sorted(map(int, data['splits']['fit']),
        key=lambda i: (-int(lengths[i]), i))[:32], dtype=np.int64)
    plans = [('warm_first_fit', orders[0][:32]), ('first_fit', orders[0][:32]), ('longest_fit', longest)]
    records = {}
    for kind in KINDS:
        model = fresh(kind, initial).train()
        before = runner.state_fingerprint(model.state_dict())
        rows = []
        for name, ids in plans:
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t = time.monotonic()
            h, r, mask, y = batch(data, ids, 'fit')
            logits, _ = model(h, r, mask)
            loss = F.cross_entropy(logits, y)
            assert torch.isfinite(loss)
            loss.backward()
            finite_gradients(model)
            torch.cuda.synchronize()
            rows.append(dict(batch=name, examples=len(ids), padded_tokens=int(mask.shape[1]),
                indices_sha256=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),
                seconds=time.monotonic()-t, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(), finite_forward_backward=True))
            del h, r, mask, y, logits, loss
        assert runner.state_fingerprint(model.state_dict()) == before == runner.state_fingerprint(initial)
        records[kind] = rows
        del model
    measured = [row for rows in records.values() for row in rows if row['batch'] != 'warm_first_fit']
    assert len(measured) == 8
    # This is a conservative planning estimate, never an admission or runtime guarantee.
    step_bound = sum(max(row['seconds'] for row in records[k] if row['batch'] != 'warm_first_fit') for k in KINDS)
    estimate = math.ceil(392*step_bound*2 + 300)
    return dict(status='TOKEN_TRANSPORT_FIT_PROFILE_COMPLETE',conditions=list(KINDS),batches=records,
        backward_calls=12,optimizer_steps=0,evaluation_model_forwards=0,new_checkpoints=0,
        evaluation_labels_used_for_loss_or_selection=False,label_domain_validation_on_load=True,
        initial_state_sha256=runner.state_fingerprint(initial),
        conservative_training_seconds_estimate=estimate,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        memory_scope='Run-wide Torch allocator high-water since before setup; all12 backwards, warmups and transitions included. Per-batch records are cumulative. Not a hard device-use cap or total CUDA-context memory.',
        real_dataset_reads=True,task_training=False,COS_requests=0,cloud_writes=0,
        formal_handoff=False,independent_confirmation=False,L5='NOT_ACCEPTED',
        elapsed_seconds=time.monotonic()-started)


@torch.no_grad()
def evaluate(model, data, ids, *,routing_kind=None):
    model.eval()
    predictions = []
    ce = 0.
    first = None
    for start in range(0, len(ids), 32):
        h, r, mask, y = batch(data, ids[start:start+32], 'eval')
        if routing_kind is None:
            logits = model(h, r, mask)[0]
        else:
            # Freeze preparation, class queries, utility values and bias together.
            d = model(h, r, mask, details=True)
            alpha = route(d['psi'], d['phi'], mask, kind=routing_kind)['alpha']
            logits = (alpha*d['utilities']).sum(-1)+model.bias
        assert torch.isfinite(logits).all()
        if first is None:
            first = logits.clone()
        predictions.extend(logits.argmax(-1).tolist())
        ce += float(F.cross_entropy(logits, y, reduction='sum'))
    gold = np.asarray(data['targets'][ids], dtype=np.int64)
    row = metrics.from_confusion(metrics.confusion(gold, np.asarray(predictions, dtype=np.int64)))
    row['ce'] = ce/len(ids)
    return row, first


@torch.no_grad()
def range_diagnostics(model, data):
    """Known generalized-Rayleigh bounds on the first32 immutable eval units.

    A violated native coordinate bound suffices for region separation. It is
    not necessary, and is neither semantic attribution nor a utility theorem.
    """
    selected = data['splits']['eval'][:32]
    h, r, mask, _ = batch(data, selected, 'eval')
    d = model(h, r, mask, details=True)
    padded_alpha = {k:route(d['psi'], d['phi'], mask, kind=k)['alpha'] for k in KINDS}
    assert torch.allclose(padded_alpha['cubic_coherent'], d['alpha'], atol=1e-10, rtol=0)
    padded_logits = (padded_alpha['cubic_coherent']*d['utilities']).sum(-1)+model.bias
    assert torch.allclose(padded_logits, d['logits'], atol=1e-9, rtol=0)
    eye = torch.eye(8, dtype=torch.complex128, device='cuda')
    records = []
    for b, i in enumerate(selected):
        psi = d['psi'][b, mask[b]]
        utilities = d['utilities'][b, :, mask[b]]
        m = torch.ones((1, len(psi)), dtype=torch.bool, device='cuda')
        projectors = psi[:, :, None]*psi.conj()[:, None, :]
        B = projectors.sum(0)+len(psi)*EPS*eye
        vals, vecs = torch.linalg.eigh(B)
        assert vals.min() > 0
        inverse = (vecs*vals.rsqrt()[None, :])@vecs.mH
        def bounds(effects):
            assert torch.isfinite(effects).all()
            whitened = inverse@effects@inverse
            assert torch.isfinite(whitened).all()
            result = torch.linalg.eigvalsh((whitened+whitened.mH)/2)
            assert torch.isfinite(result).all()
            return result
        native_effects = projectors+EPS*eye
        native_upper = bounds(native_effects)[:, -1]
        native_readout = bounds(torch.einsum('rt,tab->rab', utilities.to(torch.complex128), native_effects))
        rows = {}
        for kind in KINDS:
            routed = route(psi[None], d['phi'], m, kind=kind)
            alpha = routed['alpha'][0]
            selected_alpha = padded_alpha[kind][b][:, mask[b]]
            padding_residual = float((alpha-selected_alpha).abs().max())
            assert math.isfinite(padding_residual) and padding_residual < 1e-9
            z = (alpha*utilities).sum(-1)
            if kind == 'cubic_coherent':
                assert torch.allclose(z+model.bias, d['logits'][b], atol=1e-8, rtol=0)
            if kind == 'native':
                effects = native_effects
            elif kind == 'cubic_initial_dephased':
                effects = torch.einsum('tj,jab->tab', routed['unitary'][0].abs().square().to(torch.complex128), projectors)+EPS*eye
            else:
                effective = routed['unitary'][0].conj()@psi
                effects = effective[:, :, None]*effective.conj()[:, None, :]+EPS*eye
            phi = d['phi']
            denominator = torch.einsum('rd,de,re->r', phi.conj(), B, phi)
            numerator = torch.einsum('rd,tde,re->rt', phi.conj(), effects, phi)
            assert torch.isfinite(denominator).all() and torch.isfinite(numerator).all()
            assert denominator.real.min() > 0 and denominator.imag.abs().max() < 1e-10
            assert numerator.imag.abs().max() < 1e-10
            direct_alpha = numerator.real/denominator.real[:, None]
            score_residual = float((direct_alpha-alpha).abs().max())
            readout_residual = float(((direct_alpha*utilities).sum(-1)-z).abs().max())
            assert math.isfinite(score_residual) and math.isfinite(readout_residual)
            assert score_residual < 1e-9 and readout_residual < 1e-8
            own = bounds(torch.einsum('rt,tab->rab', utilities.to(torch.complex128), effects))
            assert torch.isfinite(z).all() and torch.isfinite(alpha).all()
            residual = max(0., float((own[:, 0]-z).max()), float((z-own[:, -1]).max()))
            assert residual < 1e-8
            excess = alpha-native_upper[None, :]
            rows[kind] = dict(native_coordinate_exceedances=int((excess > 1e-9).sum()),
                max_native_coordinate_excess=float(excess.max()),
                native_readout_upper_exceedances=int((z-native_readout[:, -1] > 1e-8).sum()),
                native_readout_lower_exceedances=int((native_readout[:, 0]-z > 1e-8).sum()),
                max_native_readout_upper_excess=float((z-native_readout[:, -1]).max()),
                max_native_readout_lower_excess=float((native_readout[:, 0]-z).max()),
                direct_quadratic_score_residual=score_residual,
                direct_quadratic_readout_residual=readout_residual,
                masked_unpadded_score_residual=padding_residual,
                own_readout_range_residual=residual)
        records.append(dict(unit_sha256=data['metadata'][int(i)]['id_hash'],tokens=len(psi),routing=rows))
    return dict(selection='First32 immutable evaluation units, before outcomes',units=records,
        claim_ceiling='Fixed-current-state geometric diagnostics; no semantic causality, total-gain mediation or sample-trace.v2 certification.')


class BoundedWriter:
    """Enforce the selected checkpoint-file bound before each serialized write."""
    def __init__(self, handle, maximum):
        self.handle, self.maximum, self.written = handle, maximum, 0

    def write(self, value):
        if self.written+len(value) > self.maximum:
            raise OSError('Declared checkpoint write bound exceeded')
        size = self.handle.write(value)
        assert size == len(value)
        self.written += size
        return size

    def flush(self):
        self.handle.flush()


def stage_checkpoint(path, state, maximum=1024**2):
    pending = path.with_suffix('.pt.pending')
    assert not path.exists() and not pending.exists()
    fd = os.open(pending, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        torch.save(state, BoundedWriter(handle, maximum))
        handle.flush()
        os.fsync(handle.fileno())
    assert pending.stat().st_size <= maximum
    return pending


def seal_checkpoints(checkpoints, files, maximum):
    marker = dict(schema='qattn.token-transport.screen.checkpoints.v1',
        files={n:v['sha256'] for n,v in files.items()},selection='Fixed fourth-epoch final states; no evaluation selection.')
    assert not (checkpoints/'RUN_COMPLETE').exists()
    assert set(p.name for p in checkpoints.iterdir()) == set(files)
    assert all(sha(checkpoints/n) == v['sha256'] and (checkpoints/n).stat().st_size == v['bytes'] for n,v in files.items())
    marker_bytes = len((json.dumps(marker, indent=2, allow_nan=False)+'\n').encode())
    assert sum(p.stat().st_size for p in checkpoints.iterdir())+marker_bytes <= maximum
    save(checkpoints/'RUN_COMPLETE', marker)


def train(a, design):
    """Dormant until separately reviewed resource/recovery contract is admitted."""
    assert design['task_training_admitted'] is True
    started = time.monotonic()
    data, initial, orders = setup(a, design)
    keep_checkpoints = design.get('checkpoint_policy','private_final') == 'private_final'
    assert design.get('checkpoint_policy','private_final') in ('private_final','none')
    checkpoints = a.output.parent/'scientific-work/checkpoints' if keep_checkpoints else None
    if keep_checkpoints:
        assert not checkpoints.exists()
        checkpoints.mkdir(mode=0o700)
    initial_sha = runner.state_fingerprint(initial)
    order_sha = hashlib.sha256(b''.join(o.astype('<i8').tobytes() for o in orders)).hexdigest()
    results, files, diagnostics = {}, {}, None
    for kind in KINDS:
        model = fresh(kind, initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS['lr'],
            weight_decay=SETTINGS['weight_decay'], betas=tuple(SETTINGS['betas']), eps=SETTINGS['eps'])
        epochs, steps = [], 0
        for epoch, order in enumerate(orders, 1):
            model.train()
            loss_sum = 0.
            for start in range(0, len(order), 32):
                if hasattr(a,'check_launch'):a.check_launch()
                ids = order[start:start+32]
                h, r, mask, y = batch(data, ids, 'fit')
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(h, r, mask)[0], y)
                assert torch.isfinite(loss)
                loss.backward()
                finite_gradients(model)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), SETTINGS['clip_norm'])
                assert torch.isfinite(norm)
                optimizer.step()
                assert all(torch.isfinite(p).all() for p in model.parameters())
                loss_sum += float(loss.detach())*len(ids)
                steps += 1
            epochs.append(dict(epoch=epoch,fit_ce=loss_sum/len(order)))
            print(json.dumps(dict(progress='EPOCH_COMPLETE',condition=kind,epoch=epoch,optimizer_steps=steps)),file=sys.stderr,flush=True)
        assert steps == 392
        measured, first = evaluate(model, data, data['splits']['eval'])
        state = {n:t.detach().cpu() for n,t in model.state_dict().items()}
        final_sha = runner.state_fingerprint(state)
        restored = fresh(kind, initial)
        if keep_checkpoints:
            path = checkpoints/(kind+'-final.pt')
            pending = stage_checkpoint(path, state)
            restored.load_state_dict(torch.load(pending, map_location='cpu', weights_only=True))
        else:restored.load_state_dict(state)
        assert runner.state_fingerprint(restored.state_dict()) == final_sha
        _, replay = evaluate(restored, data, data['splits']['eval'][:32])
        assert torch.equal(replay, first)
        if keep_checkpoints:
            assert pending.stat().st_size <= design['private_checkpoint_bytes_max']
            pending.replace(path)
            files[path.name] = dict(sha256=sha(path),bytes=path.stat().st_size)
        results[kind] = dict(metrics=measured,epochs=epochs,parameters=81028,optimizer_steps=steps,
            initial_state_sha256=initial_sha,final_state_sha256=final_sha,
            checkpoint=files[path.name] if keep_checkpoints else None,
            checkpoint_reload_first_batch_exact=keep_checkpoints,state_replay_first_batch_exact=True)
        if kind == 'cubic_coherent':
            frozen = {k:evaluate(restored, data, data['splits']['eval'], routing_kind=k)[0] for k in KINDS}
            assert frozen[kind] == measured
            diagnostics = dict(frozen_routing_metrics=frozen,ranges=range_diagnostics(restored,data))
        del optimizer, restored, model
    if keep_checkpoints:seal_checkpoints(checkpoints, files, design['private_checkpoint_bytes_max'])
    coherent = results['cubic_coherent']['metrics']
    gate = (all(coherent['macro_f1'] > results[k]['metrics']['macro_f1'] for k in KINDS if k != 'cubic_coherent')
            and coherent['ce'] <= results['native']['metrics']['ce'])
    contrasts = {k:dict(macro_f1=coherent['macro_f1']-results[k]['metrics']['macro_f1'],
        ce=coherent['ce']-results[k]['metrics']['ce']) for k in KINDS if k != 'cubic_coherent'}
    return dict(status='TOKEN_TRANSPORT_BOUNDED_TRAINING_COMPLETE',results=results,
        conditions=list(KINDS),coherent_contrasts=contrasts,discovery_gate=gate,
        decision='EXPLORATORY_GATE_PASS' if gate else 'INCONCLUSIVE_NO_PROMOTION',
        diagnostics=diagnostics,optimizer_steps=1568,common_epoch_order_sha256=order_sha,
        common_initial_state_sha256=initial_sha,fit_examples=3124,evaluation_examples=972,
        private_checkpoints=files,checkpoint_marker_sha256=sha(checkpoints/'RUN_COMPLETE') if keep_checkpoints else None,
        checkpoint_policy='private_final' if keep_checkpoints else 'none',
        task_training=True,full_formal_run=False,independent_confirmation=False,formal_handoff=False,
        COS_requests=0,cloud_writes=0,L5='NOT_ACCEPTED',elapsed_seconds=time.monotonic()-started,
        scope='Single common initialization, repeatedly explored original-training group partition; no independent confirmation or inherited L4.')
