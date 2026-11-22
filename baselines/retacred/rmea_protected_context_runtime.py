"""Protected-context scientific callbacks; shared scheduling/recovery are unchanged."""
import copy
import hashlib
import math
import os
from pathlib import Path
import sys

from . import rmea_protected_context_protocol as protocol
from .rmea_entity_scattering_runtime import chunks, backward, score_batch
from .rmea_protected_context_trace import require


def make_head(kind):
    from .rmea_protected_context import ProtectedContextHead
    if kind not in protocol.MODES:
        raise ValueError('Unknown training condition')
    class LogitsHead(ProtectedContextHead):
        def forward(self, h, roles, mask, details=False):
            result = super().forward(h, roles, mask, details=details)
            return result if details else result[0]
    return LogitsHead(kind)


def launch_spec(output, kind, device, *, resume):
    import uuid
    require(kind in protocol.MODES and str(device).strip(), 'Invalid worker assignment')
    token = uuid.uuid4().hex
    protocol.atomic_json(Path(output)/'.workers'/kind/'assignment.json',
        dict(token=token, parent_pid=os.getpid(), condition=kind, device=str(device),
             source=protocol.identity(protocol.sources())))
    argv = [sys.executable, '-B', str(protocol.ROOT/'experiments/q_rmea_protected_context_train.py'),
            'worker', '--output', str(Path(output).resolve()), '--condition', kind,
            '--parent-pid', str(os.getpid()), '--assignment-token', token]
    if resume:
        argv.append('--resume')
    return dict(argv=argv, cwd=str(protocol.ROOT), physical_gpu=str(device),
                condition=kind, revision=protocol.git('rev-parse', 'HEAD'))


def prepare(recovery, labels, data_dir):
    from .rmea_entity_scattering_runtime import prepare as shared_prepare
    return shared_prepare(recovery, labels, data_dir, protocol_override=protocol)


def capture(output, head, kind, split, batch, metadata):
    """CPU replay from the actual frozen final model, independent of live GPU graph."""
    import torch
    from .rmea_protected_context import ProtectedContextHead
    from .rmea_protected_context_trace import capture as capture_values
    cp = output/'checkpoints'/(kind+'-final.pt')
    frozen = ProtectedContextHead(kind).double().eval()
    frozen.load_state_dict(torch.load(cp, map_location='cpu', weights_only=True), strict=True)
    def cpu_batch(s, indices):
        inputs, target = batch(s, indices)
        return tuple(v.detach().cpu() for v in inputs), target.detach().cpu()
    return capture_values(output, frozen, kind, split, cpu_batch, metadata, reuse_identical=True)


def worker_receipt(output, condition):
    # Result.json contains the full bounded metadata and every private artifact pin.
    return dict(trace_schema='sample-trace.v2', trace_samples=6)


def validate_completed(output, kind, row, receipt):
    p = protocol
    require(row.get('diagnostics') is None and receipt.get('trace_schema') == 'sample-trace.v2'
            and receipt.get('trace_samples') == 6, 'Wrong final-head completion scope')
    expected = {f'{kind}:final:{s}:{i}' for s in ('train', 'valid', 'test') for i in (0, 1)}
    require(len(row['samples']) == 6 and {s['sample_id'] for s in row['samples']} == expected,
            'Condition trace population mismatch')
    cp = output/'checkpoints'/(kind+'-final.pt')
    for sample in row['samples']:
        prov = sample['roles']['provenance']
        require(prov['condition'] == kind and prov['checkpoint_sha256'] == p.digest(cp)
                and prov['checkpoint_file'] == cp.relative_to(output).as_posix(), 'Trace model mismatch')
        for rep in sample['representations']:
            for ref in rep['artifact_refs']:
                path = output/ref['path']
                require(path.resolve().is_relative_to((output/'trace').resolve())
                        and not path.is_symlink() and path.is_file()
                        and path.stat().st_size == ref['bytes'] and p.digest(path) == ref['sha256'],
                        'Private complete-value artifact mismatch')


def initialize_inputs(root, inputs):
    """The old native model is a small required analysis dependency in preparation."""
    p = protocol
    source = inputs/'checkpoints/native-final.pt'
    require(not source.is_symlink() and source.is_file() and p.digest(source) == p.NATIVE_CHECKPOINT,
            'Historical native checkpoint mismatch')
    target = root/'recovery-inputs/native-final.pt'
    if target.exists():
        require(not target.is_symlink() and p.digest(target) == p.NATIVE_CHECKPOINT, 'Existing native differs')
    else:
        os.link(source, target)


def validate_inputs(root):
    path = root/'recovery-inputs/native-final.pt'
    require(not path.is_symlink() and path.is_file() and protocol.digest(path) == protocol.NATIVE_CHECKPOINT,
            'Durable native analysis dependency mismatch')


def validate_training_cache(output, cache):
    """Run before parent launch/resume: same-size corruption cannot reach an update."""
    for name, row in cache.items():
        path = output/'cache'/name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == row['bytes']
                and protocol.digest(path) == row['sha256'], 'Training input bytes differ: '+name)


def _native_fixture(output):
    validate_inputs(output.parent)
    source = output.parent/'recovery-inputs/native-final.pt'
    target = output/'checkpoints/native-final.pt'
    if target.exists():
        require(not target.is_symlink() and protocol.digest(target) == protocol.NATIVE_CHECKPOINT,
                'Existing final native differs')
    else:
        os.link(source, target)
    from .rmea_token_transport_recovery import sync_directory
    sync_directory(target.parent)
    return target


def _batch(output):
    import numpy as np
    import torch
    p = protocol
    data = {s: {name: np.load(output/'cache'/f'{s}-{name}.npy', mmap_mode='r', allow_pickle=False)
                for name in ('h', 'roles', 'offsets', 'targets')} for s in ('train', 'valid', 'test')}
    def batch(split, indices):
        require(len(indices) == 1 and indices[0] in (0, 1), 'Fixed final trace indices only')
        i = indices[0]; lo, hi = map(int, data[split]['offsets'][i:i+2])
        inputs = tuple(torch.from_numpy(np.array(data[split][n][lo:hi], dtype=np.float64))[None]
                       for n in ('h', 'roles'))
        return (*inputs, torch.ones(1, hi-lo, dtype=torch.bool)), torch.tensor([int(data[split]['targets'][i])])
    return batch


def finish(cfg, output, parts, *, data_dir=None, assets=None):
    import torch
    from .rmea_protected_context import ProtectedContextHead
    from .rmea_protected_context_trace import capture as capture_values, close
    from .rmea_token_transport_recovery import sync_directory
    p = protocol; results = {}; samples = []; initials = set(); orders = set()
    for row in parts:
        require(not set(results)&set(row['results']), 'Duplicate condition result')
        results.update(row['results']); samples += row['samples']
        initials.add(row['initial_sha']); orders.add(row['order_sha'])
    require(set(results) == set(p.MODES) and initials == {p.BASE_INITIAL} and len(orders) == 1,
            'Condition/init/order coverage mismatch')
    manifest = p.read(output/'manifest.json')
    require(cfg == manifest['config'] == p.config(cfg['profile']), 'Config mismatch')
    if cfg['profile'] == 'discovery':
        require(next(iter(orders)) == manifest['reference']['order_sha256'], 'Historical order mismatch')
    hashes = {kind: results[kind]['checkpoint_sha256'] for kind in p.MODES}
    native = ProtectedContextHead('native').double().eval()
    native.load_state_dict(torch.load(_native_fixture(output), map_location='cpu', weights_only=True), strict=True)
    hashes['native'] = p.NATIVE_CHECKPOINT
    metadata = p.read(output/'cache/metadata.json')
    for split in cfg['counts']:
        samples += capture_values(output, native, 'native', split, _batch(output), metadata, reuse_identical=True)
    report = close(output, samples, cfg, metadata, cache_manifest=manifest['cache'], checkpoint_hashes=hashes)
    # Freeze these source-mechanism checks before interpreting task-level deltas.
    import numpy as np
    measurements = {}
    for kind in p.MODES:
        rows = []
        for split in cfg['counts']:
            for index in (0, 1):
                with np.load(output/'trace'/f'{kind}-final-{split}-{index}.npz', allow_pickle=False) as z:
                    delta = z['corrected']-z['phi'][None]
                    original = z['phi'][None]
                    energy = lambda q: np.einsum('bri,bij,brj->br', q.conj(), z['used_covariance'], q).real
                    decrease = energy(original)-energy(z['corrected'])
                    expected = energy(delta)+2/8*np.sum(np.abs(delta)**2, -1)
                    residual = (delta.conj()@z['psi'].transpose(0, 2, 1))*z['mention'][:, None]
                    rows.append(dict(protected_amplitude_residual=float(np.max(np.abs(residual))),
                        solve_power_identity_residual=float(np.max(np.abs(decrease-expected))),
                        minimum_covariance_power_decrease=float(np.min(decrease))))
        measurements[kind] = dict(samples=len(rows),
            protected_amplitude_residual_max=max(x['protected_amplitude_residual'] for x in rows),
            solve_power_identity_residual_max=max(x['solve_power_identity_residual'] for x in rows),
            covariance_power_decrease_min=min(x['minimum_covariance_power_decrease'] for x in rows),
            limits='Retained-span amplitude identity only for protected/scrambled; no amplitude constraint for unprotected; no utility implication')
    reference = manifest['reference']['result']
    contrasts = {}; utility = attribution = None
    if cfg['profile'] == 'discovery':
        candidate = results['protected_context']['test']
        for kind, control in [('native_reference', reference['test'])]+[(k, results[k]['test']) for k in p.MODES[1:]]:
            delta = candidate['positive39_macro_f1']-control['positive39_macro_f1']
            denom = abs(control['positive39_macro_f1'])
            contrasts[kind] = dict(absolute_macro_delta=delta, relative_macro_delta=delta/denom if denom else None,
                                   ce_delta=candidate['ce']-control['ce'])
        utility = contrasts['native_reference']['absolute_macro_delta'] > 0
        attribution = all(contrasts[k]['absolute_macro_delta'] > 0 for k in p.MODES[1:])
    summary = dict(schema=p.SCHEMA_PREFIX+'.training.result.v1', results=results, reference=reference,
        contrasts=contrasts, mechanism_measurements=measurements,
        ordinary_utility_gate=utility, matched_control_gate=attribution,
        promotion_gate=bool(utility and attribution) if utility is not None else None,
        seed=13, full_data_training=cfg['profile']=='discovery', common_initial_state_sha256=p.BASE_INITIAL,
        common_epoch_order_sha256=next(iter(orders)), source_sha256=p.identity(p.sources()),
        config_sha256=p.identity(cfg), selection=cfg['selection'], L5='NOT_ACCEPTED',
        limitations='One seed and exposed cohort; no automatic independent confirmation, novelty or inherited L4/L5')
    p.atomic_json(output/'summary.json', summary)
    readiness = dict(schema=p.SCHEMA_PREFIX+'.analysis-readiness.v1', selection=cfg['selection'], checks=report,
        checkpoints={k: dict(path='checkpoints/'+k+'-final.pt', sha256=hashes[k]) for k in cfg['trace_conditions']},
        trace_artifacts={n: dict(sha256=p.digest(output/n), bytes=(output/n).stat().st_size)
                         for n in ('sample-trace.json', 'trace-validation.json')},
        status='Materialized full-value v2 final-head replay validated; no encoder/training-trajectory claim')
    p.atomic_json(output/'analysis-readiness.json', readiness)
    names = ('manifest.json', 'summary.json', 'analysis-readiness.json')
    sync_directory(output)
    p.atomic_json(output/'RUN_COMPLETE', dict(artifacts={n: dict(sha256=p.digest(output/n), bytes=(output/n).stat().st_size) for n in names}))
    sync_directory(output)


def closed_analysis(output):
    from .rmea_protected_context_trace import validate_trace
    p = protocol; manifest = p.read(output/'manifest.json'); ready = p.read(output/'analysis-readiness.json')
    validate_training_cache(output, manifest['cache'])
    seal = p.read(output/'RUN_COMPLETE')
    require(p.digest(output/'analysis-readiness.json') == seal['artifacts']['analysis-readiness.json']['sha256'],
            'Sealed replay inventory mismatch')
    require(set(ready['checkpoints']) == set(manifest['config']['trace_conditions']), 'Model coverage mismatch')
    for kind, row in ready['checkpoints'].items():
        require(row['path'] == 'checkpoints/'+kind+'-final.pt' and p.digest(output/row['path']) == row['sha256'],
                'Reported final model mismatch')
    for name, row in ready['trace_artifacts'].items():
        require(name in ('sample-trace.json', 'trace-validation.json') and p.digest(output/name) == row['sha256']
                and (output/name).stat().st_size == row['bytes'], 'Sealed trace mismatch')
    report = validate_trace(output, p.read(output/'sample-trace.json'), cfg=manifest['config'],
        metadata=p.read(output/'cache/metadata.json'), cache_manifest=manifest['cache'],
        checkpoint_hashes={k: v['sha256'] for k, v in ready['checkpoints'].items()})
    require(report == ready['checks'] == p.read(output/'trace-validation.json'), 'Closed replay validation differs')
    return ready


def replay(output, condition, split):
    require(condition in protocol.TRACE_MODES and split in ('train', 'valid', 'test'), 'Replay scope mismatch')
    ready = closed_analysis(output)
    return dict(condition=condition, split=split, replay='pass', checkpoint_sha256=ready['checkpoints'][condition]['sha256'])


def replay_all(output):
    return [closed_analysis(output)['checks']]


def export(output, target, *, inject_failure=False):
    from .rmea_entity_scattering_runtime import export as shared_export
    return shared_export(output, target, inject_failure=inject_failure,
                         protocol_override=protocol, closed_analysis_override=closed_analysis)


def unit(out):
    """Focused real-fixture training adapter parity; not another derivative study."""
    import subprocess
    import torch
    from .rmea_protected_context import ProtectedContextHead
    from .rmea_token_transport_study import state_identity
    from .rmea_frozen_training import INPUT_RUN
    from experiments.q_rmea_query_alignment import fresh_output
    fresh_output(out); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    original = protocol.read(INPUT_RUN.parent/'canary/manifest.json')
    require(protocol.digest(INPUT_RUN.parent/'canary/manifest.json') == protocol.CANARY_MANIFEST, 'Canary pin mismatch')
    for name in ['metadata.json']+[s+'-'+k+'.npy' for s in ('train','valid','test') for k in ('h','roles','offsets','targets')]:
        path = INPUT_RUN.parent/'canary/cache'/name
        require(path.is_file() and not path.is_symlink() and protocol.digest(path) == original['cache'][name]['sha256'],
                'Pinned canary cache member mismatch')
    # Five fixed real examples exercise variable-length padding and4+1 chunks.
    import numpy as np
    cached = {k: np.load(INPUT_RUN.parent/'canary/cache'/('train-'+k+'.npy'), mmap_mode='r')
              for k in ('h', 'roles', 'offsets', 'targets')}
    lengths = [int(cached['offsets'][i+1]-cached['offsets'][i]) for i in range(5)]
    h = torch.zeros(5, max(lengths), 1024, dtype=torch.float64)
    roles = torch.zeros(5, max(lengths), 4, dtype=torch.float64)
    mask = torch.zeros(5, max(lengths), dtype=torch.bool)
    for i, length in enumerate(lengths):
        lo, hi = map(int, cached['offsets'][i:i+2])
        h[i, :length] = torch.from_numpy(np.array(cached['h'][lo:hi], dtype=np.float64))
        roles[i, :length] = torch.from_numpy(np.array(cached['roles'][lo:hi], dtype=np.float64))
        mask[i, :length] = True
    inputs = (h, roles, mask)
    target = torch.tensor(np.array(cached['targets'][:5], dtype=np.int64))
    errors = {}; rejected_nonfinite = []
    def finite_gradients(model):
        for name, parameter in model.named_parameters():
            require(parameter.grad is not None and torch.isfinite(parameter.grad).all(),
                    'Nonfinite or missing named gradient: '+name)
    for kind in protocol.MODES:
        torch.manual_seed(13); actual = make_head(kind).double()
        expected = ProtectedContextHead(kind).double(); expected.load_state_dict(actual.state_dict())
        require(state_identity(actual.state_dict()) == protocol.BASE_INITIAL, 'Initialization mismatch')
        require(torch.equal(actual(*inputs), expected(*inputs)[0]), 'Training adapter forward differs')
        loss = torch.nn.functional.cross_entropy(actual(*inputs), target); loss.backward()
        other = make_head(kind).double(); other.load_state_dict(actual.state_dict()); value = backward(other, inputs, target)
        finite_gradients(actual); finite_gradients(other)
        error = max(float((a.grad-b.grad).abs().max()) for a, b in zip(actual.parameters(), other.parameters()))
        require(error < 1e-9 and abs(float(loss.detach()-value)) < 1e-10, 'Training adapter microgradient mismatch')
        late = list(other.parameters())[-1]; original_grad = late.grad.clone()
        late.grad.flatten()[-1] = float('nan')
        try:
            finite_gradients(other)
        except ValueError:
            rejected_nonfinite.append(kind)
        else:
            raise ValueError('Late microbatch NaN gradient admitted')
        finally:
            late.grad.copy_(original_grad)
        optimizer = torch.optim.AdamW(other.parameters(), lr=.001, weight_decay=.01, betas=(.9, .999), eps=1e-8)
        finite_gradients(other)
        torch.nn.utils.clip_grad_norm_(other.parameters(), 1.); optimizer.step()
        require(all(torch.isfinite(x).all() for x in other.parameters()), 'Nonfinite microbatch update')
        errors[kind] = error
    commands = {}
    for name in ('experiments/q_rmea_protected_context_train.py', 'experiments/q_rmea_entity_scattering_train.py'):
        v = subprocess.run([sys.executable, '-B', str(protocol.ROOT/name), '--help'], cwd=protocol.ROOT,
                           capture_output=True, text=True, timeout=30)
        require(v.returncode == 0 and 'worker' in v.stdout, 'Parent/child CLI target failed')
        commands[name] = dict(returncode=v.returncode, stdout_sha256=hashlib.sha256(v.stdout.encode()).hexdigest())
    protocol.atomic_json(out/'unit-validation.json', dict(gate=True, source_sha256=protocol.identity(protocol.sources()),
        contract_sha256=protocol.identity(protocol.contract()), gradient_errors=errors, real_fixture_indices=list(range(5)),
        CLI_checks=commands, baseline_initial=protocol.BASE_INITIAL, optimizer_steps_per_condition=1,
        late_nonfinite_microgradients_rejected=rejected_nonfinite,
        scope='Actual training logits adapter and microgradient/finite update; prior source derivative/v2 replay evidence reused'))
    protocol.atomic_json(out/'RUN_COMPLETE', dict(artifacts={'unit-validation.json': protocol.digest(out/'unit-validation.json')}))
